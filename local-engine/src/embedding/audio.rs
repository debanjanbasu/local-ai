//! Native `EmbeddingGemma` 2 **audio** embeddings on Metal.
//!
//! An audio clip is embedded the way the reference model (transformers
//! 5.19.0 / sentence-transformers 6.1.0) embeds a media input with no text:
//!
//! 1. 16 kHz mono samples ([`decode_audio`] decodes WAV/FLAC/Ogg
//!    Vorbis/MP3 with Symphonia, downmixes by averaging the channels and
//!    resamples with Rubato's FFT resampler) → 128-bin log-mel frames
//!    exactly as `Gemma4AudioFeatureExtractor` computes them
//!    ([`FeatureExtractor`]).
//! 2. The `gemma4_audio` Conformer on the GPU: a two-layer stride-2 conv stem
//!    (layer norm + `ReLU`) and a 1024→1024 projection, then 12 layers of
//!    macaron feed-forward (1024→4096→1024, `SiLU`, half-step residual),
//!    chunked local self-attention (each frame sees itself and the 11
//!    previous ones, relative-position logits, logit cap 50), a light
//!    convolution block (GLU, causal depthwise conv k=5, RMS norm, `SiLU`)
//!    and a second feed-forward; every linear is a clipped linear whose input
//!    and output clamp bounds come from the checkpoint. `output_proj`
//!    (1024→1536, bias) and `embed_audio` (RMS norm without scale,
//!    1536→512) give one soft token per 40 ms of audio.
//! 3. The sequence `<bos> <start_of_audio> soft tokens… <end_of_audio> <eos>`
//!    runs through the shared text encoder (the same 24 layers, mean pooling
//!    over every position, 512→768 projection, L2 normalization) as text.
//!
//! The audio artifact (`tools/convert_embeddinggemma2_audio.py`) is an
//! optional add-on to the text installation and is loaded explicitly with
//! [`EmbeddingModel::load_audio`]. Matrices keep the checkpoint's BF16
//! values; activations are F32. Clips longer than 30 s are refused (the
//! reference would silently keep only the first 30 s).

mod features;

use std::path::Path;

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::embedding::Bf16Rows;
use local_metal::embedding_audio::{AudioKernels, ClippedLinear, LocalAttention};
use local_metal::shaders::ShaderLibrary;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

pub use features::{
    FeatureExtractor, LogMel, MAX_SAMPLES, MEL_BINS, SAMPLE_RATE, audio_tokens, frame_counts,
};

use super::artifact::Artifact;
use super::{EmbeddingModel, HIDDEN, MediaPart};
use crate::Error;

/// Audio artifact file name inside the text installation directory.
pub const AUDIO_ARTIFACT: &str = "embeddinggemma2-audio-bf16.bin";
/// Most audio soft tokens per clip (30 s at 25 tokens per second).
pub const MAX_AUDIO_TOKENS: usize = audio_tokens(frame_counts(MAX_SAMPLES).1);

const MAGIC: &[u8; 8] = b"EG2AUD\x00\x01";
const FORMAT: &str = "embeddinggemma2-audio";
const VERSION: u32 = 1;

const WIDTH: usize = 1024;
const LAYERS: usize = 12;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FF: usize = 4096;
const OUT_DIMS: usize = 1536;
const STEM0: usize = 128;
const STEM1: usize = 32;
const CONV_KERNEL: usize = 5;
const RESIDUAL_WEIGHT: f32 = 0.5;
const LOGIT_CAP: f32 = 50.0;
const EPSILON: f32 = 1e-6;
/// Relative positions of the attention context (`context_left - 1` = 12
/// past positions plus the frame itself); the reference's sliding-window
/// mask (`0 <= q - k < context_left - 1`) attends distances 0..=11.
const REL_POSITIONS: usize = 13;
const MAX_DISTANCE: u32 = 11;

/// Tower rows per pass and log-mel frames per pass (a 30 s clip is 750
/// tokens from 2,999 frames).
const TOWER_TOKENS: usize = 1088;
const PASS_FRAMES: usize = 4200;
const PASS_CLIPS: usize = 64;
/// Token-layers per command buffer before the pacing hook runs (whole
/// layers, at least one). A Conformer token-layer costs about seven text
/// token-layers (40 M vs 5.5 M multiply-adds), so this is the text path's
/// 1,024 scaled down: one layer per buffer from 128 tokens up.
const SUBMIT_TOKEN_LAYERS: usize = 128;

/// Decode an audio file to 16 kHz mono F32 samples.
///
/// WAV/PCM, FLAC, Ogg Vorbis or MP3 (the container is probed from the
/// bytes); channels are averaged and other sample rates are resampled with
/// Rubato's FFT resampler.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the bytes are not a supported audio file.
pub fn decode_audio(bytes: Vec<u8>) -> crate::Result<Vec<f32>> {
    let (interleaved, channels, rate) = decode_interleaved(bytes)?;
    to_mono_16k(&interleaved, channels, rate)
}

fn bad_audio(error: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(format!("cannot decode audio: {error}"))
}

fn decode_interleaved(bytes: Vec<u8>) -> crate::Result<(Vec<f32>, usize, u32)> {
    let stream = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(bad_audio)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| bad_audio("no audio track"))?;
    let track_id = track.id;
    let Some(CodecParameters::Audio(params)) = track.codec_params.clone() else {
        return Err(bad_audio("no audio codec parameters"));
    };
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(bad_audio)?;
    let mut samples = Vec::new();
    let mut chunk: Vec<f32> = Vec::new();
    let (mut channels, mut rate) = (0_usize, 0_u32);
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(error) => return Err(bad_audio(error)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buffer: GenericAudioBufferRef<'_> = match decoder.decode(&packet) {
            Ok(buffer) => buffer,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(error) => return Err(bad_audio(error)),
        };
        let spec = buffer.spec();
        let (c, r) = (spec.channels().count(), spec.rate());
        if channels == 0 {
            (channels, rate) = (c, r);
        } else if (channels, rate) != (c, r) {
            return Err(bad_audio(
                "the channel layout or sample rate changes mid-stream",
            ));
        }
        buffer.copy_to_vec_interleaved(&mut chunk);
        samples.extend_from_slice(&chunk);
        if samples.len() / channels.max(1) > MAX_DECODE_SECONDS * rate as usize {
            return Err(Error::InvalidArgument(format!(
                "audio is longer than {MAX_DECODE_SECONDS} s"
            )));
        }
    }
    if channels == 0 || rate == 0 {
        return Err(bad_audio("no decodable audio"));
    }
    Ok((samples, channels, rate))
}

/// Decoding stops past this duration (clips longer than 30 s are refused
/// anyway; this only bounds memory on hostile input).
const MAX_DECODE_SECONDS: usize = 600;

/// Average interleaved channels to mono and resample to 16 kHz (Rubato FFT
/// resampler) if needed.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for zero channels or rate, or a resampler error.
pub fn to_mono_16k(interleaved: &[f32], channels: usize, rate: u32) -> crate::Result<Vec<f32>> {
    if channels == 0 || rate == 0 {
        return Err(bad_audio("zero channels or sample rate"));
    }
    let mono: Vec<f32> = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    if rate == SAMPLE_RATE || mono.is_empty() {
        return Ok(mono);
    }
    resample(&mono, rate)
}

/// Input frames per resampler chunk, in one FFT block: the anti-aliasing
/// cutoff rises with the block size. Against librosa (soxr) on a 44.1 kHz
/// probe, Rubato's default 1,024 frames in four sub-blocks differed by 0.70
/// in log-mel over the top 8 mel bins (7–8 kHz); one 8,192-frame block
/// differs by 0.16 (larger blocks and other windows: no change).
const RESAMPLER_CHUNK: usize = 8192;

fn resample(mono: &[f32], rate: u32) -> crate::Result<Vec<f32>> {
    use rubato::audioadapter::Adapter as _;
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler as _};
    let input: Vec<f64> = mono.iter().map(|v| f64::from(*v)).collect();
    let mut resampler = Fft::<f64>::new_custom(
        rate as usize,
        SAMPLE_RATE as usize,
        RESAMPLER_CHUNK,
        1,
        1,
        rubato::WindowFunction::BlackmanHarris2,
        FixedSync::Input,
    )
    .map_err(bad_audio)?;
    let adapter = InterleavedSlice::new(&input, 1, input.len()).map_err(bad_audio)?;
    let output = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(bad_audio)?;
    Ok((0..output.frames())
        .map(|i| output.read_sample(0, i).unwrap_or(0.0) as f32)
        .collect())
}

#[derive(Clone, Copy)]
struct Linear {
    offset: usize,
    clip: [f32; 4],
}

struct FeedForward {
    pre_norm: usize,
    up: Linear,
    down: Linear,
    post_norm: usize,
}

struct Layer {
    ff1: FeedForward,
    ff2: FeedForward,
    pre_attn_norm: usize,
    post_attn_norm: usize,
    out_norm: usize,
    q: Linear,
    k: Linear,
    v: Linear,
    post: Linear,
    /// Offsets into the tower's `derived` buffer.
    q_scale: usize,
    rel: usize,
    conv_pre_norm: usize,
    conv_start: Linear,
    depthwise: usize,
    conv_norm: usize,
    conv_end: Linear,
}

struct Workspace {
    frames: MetalBuffer,
    stem: MetalBuffer,
    hidden: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attention: MetalBuffer,
    branch: MetalBuffer,
    wide: MetalBuffer,
    projected: MetalBuffer,
    soft: MetalBuffer,
    first: MetalBuffer,
}

/// The loaded audio encoder (kernels, weights, workspace), owned by an
/// [`EmbeddingModel`].
pub(super) struct AudioTower {
    kernels: AudioKernels,
    artifact: Artifact,
    weights: MetalBuffer,
    derived: MetalBuffer,
    stem: [usize; 5],
    layers: Vec<Layer>,
    output_proj: usize,
    output_bias: usize,
    embed: usize,
    features: FeatureExtractor,
    workspace: Workspace,
}

const NO_CLIP: [f32; 4] = [
    f32::NEG_INFINITY,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::INFINITY,
];

fn invalid(message: &str) -> Error {
    Error::InvalidFormat(format!("embedding audio artifact: {message}"))
}

/// The derived per-layer constants: the per-dimension query scale
/// `head_dim^-0.5 / ln 2 * softplus(per_dim_scale)` and the relative keys
/// `relative_k_proj(sinusoid(distance))` for distances 0..=12, laid out as
/// `[distance][head][dim]`.
fn derived_constants(artifact: &Artifact, layer: usize) -> crate::Result<(Vec<f32>, Vec<f32>)> {
    let per_dim =
        artifact.f32_values(&format!("layers.{layer}.self_attn.per_dim_scale"), HEAD_DIM)?;
    let base = (HEAD_DIM as f64).powf(-0.5) / std::f64::consts::LN_2;
    let q_scale = per_dim
        .iter()
        .map(|p| (base * f64::from(*p).exp().ln_1p()) as f32)
        .collect();
    let name = format!("layers.{layer}.self_attn.relative_k_proj");
    let offset = artifact.bf16(&name, WIDTH, WIDTH)?;
    let matrix = &artifact.payload()[offset..offset + WIDTH * WIDTH * 2];
    // Sinusoid of the reference's Gemma4AudioRelPositionalEncoding:
    // [sin(p * inv_t), cos(p * inv_t)], inv_t[i] = exp(-i ln(1e4) / 511).
    let half = WIDTH / 2;
    let increment = (10_000_f64).ln() / (half - 1) as f64;
    let mut rel = vec![0.0_f32; REL_POSITIONS * WIDTH];
    for distance in 0..REL_POSITIONS {
        let position = distance as f32;
        let sinusoid: Vec<f64> = (0..WIDTH)
            .map(|i| {
                let inv = ((i % half) as f32 * -(increment as f32)).exp();
                let angle = position * inv;
                f64::from(if i < half { angle.sin() } else { angle.cos() })
            })
            .collect();
        for (row, out) in rel[distance * WIDTH..(distance + 1) * WIDTH]
            .iter_mut()
            .enumerate()
        {
            let weights = &matrix[row * WIDTH * 2..(row + 1) * WIDTH * 2];
            *out = weights
                .as_chunks::<2>()
                .0
                .iter()
                .zip(&sinusoid)
                .map(|(w, s)| {
                    f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*w)) << 16)) * s
                })
                .sum::<f64>() as f32;
        }
    }
    Ok((q_scale, rel))
}

fn check_config(config: &serde_json::Value) -> crate::Result<()> {
    let int = |key: &str| config.get(key).and_then(serde_json::Value::as_u64);
    let float = |key: &str| config.get(key).and_then(serde_json::Value::as_f64);
    let ok = int("hidden_size") == Some(WIDTH as u64)
        && int("num_layers") == Some(LAYERS as u64)
        && int("num_attention_heads") == Some(HEADS as u64)
        && int("intermediate_size") == Some(FF as u64)
        && int("attention_chunk_size") == Some(12)
        && int("attention_context_left") == Some(13)
        && int("attention_context_right") == Some(0)
        && float("attention_logit_cap") == Some(f64::from(LOGIT_CAP))
        && int("conv_kernel_size") == Some(CONV_KERNEL as u64)
        && float("residual_weight") == Some(f64::from(RESIDUAL_WEIGHT))
        && int("output_proj_dims") == Some(OUT_DIMS as u64)
        && int("text_hidden_size") == Some(HIDDEN as u64)
        && config.get("use_clipped_linears") == Some(&serde_json::Value::Bool(true))
        && config.get("subsampling_conv_channels") == Some(&serde_json::json!([STEM0, STEM1]));
    if ok {
        Ok(())
    } else {
        Err(invalid(
            "architecture differs from EmbeddingGemma 2's gemma4_audio",
        ))
    }
}

impl AudioTower {
    pub(super) fn open(
        path: &Path,
        context: &local_metal::context::MetalContext,
    ) -> crate::Result<Self> {
        let (artifact, config) = Artifact::open_format(path, *MAGIC, FORMAT, VERSION)?;
        check_config(&config)?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = AudioKernels::new(context, &shaders)?;
        let weights = MetalBuffer::from_slice(context.device(), artifact.payload())?;
        let linear = |name: &str, rows: usize, columns: usize| -> crate::Result<Linear> {
            let clip = artifact.f32_values(&format!("{name}.clip"), 4)?;
            if clip.iter().any(|v| v.is_nan()) || clip[0] > clip[1] || clip[2] > clip[3] {
                return Err(invalid("bad clip bounds"));
            }
            Ok(Linear {
                offset: artifact.bf16(name, rows, columns)?,
                clip: [clip[0], clip[1], clip[2], clip[3]],
            })
        };
        let mut layers = Vec::with_capacity(LAYERS);
        let mut derived = Vec::new();
        for i in 0..LAYERS {
            let p = |name: &str| format!("layers.{i}.{name}");
            let vector = |name: &str, n: usize| artifact.f32_offset(&p(name), n);
            let ff = |which: &str| -> crate::Result<FeedForward> {
                Ok(FeedForward {
                    pre_norm: vector(&format!("{which}.pre_layer_norm"), WIDTH)?,
                    up: linear(&p(&format!("{which}.ffw_layer_1")), FF, WIDTH)?,
                    down: linear(&p(&format!("{which}.ffw_layer_2")), WIDTH, FF)?,
                    post_norm: vector(&format!("{which}.post_layer_norm"), WIDTH)?,
                })
            };
            let (q_scale, rel) = derived_constants(&artifact, i)?;
            let q_scale_offset = derived.len() * 4;
            derived.extend(q_scale);
            let rel_offset = derived.len() * 4;
            derived.extend(rel);
            layers.push(Layer {
                ff1: ff("feed_forward1")?,
                ff2: ff("feed_forward2")?,
                pre_attn_norm: vector("norm_pre_attn", WIDTH)?,
                post_attn_norm: vector("norm_post_attn", WIDTH)?,
                out_norm: vector("norm_out", WIDTH)?,
                q: linear(&p("self_attn.q_proj"), WIDTH, WIDTH)?,
                k: linear(&p("self_attn.k_proj"), WIDTH, WIDTH)?,
                v: linear(&p("self_attn.v_proj"), WIDTH, WIDTH)?,
                post: linear(&p("self_attn.post"), WIDTH, WIDTH)?,
                q_scale: q_scale_offset,
                rel: rel_offset,
                conv_pre_norm: vector("lconv1d.pre_layer_norm", WIDTH)?,
                conv_start: linear(&p("lconv1d.linear_start"), 2 * WIDTH, WIDTH)?,
                depthwise: vector("lconv1d.depthwise_conv1d", WIDTH * CONV_KERNEL)?,
                conv_norm: vector("lconv1d.conv_norm", WIDTH)?,
                conv_end: linear(&p("lconv1d.linear_end"), WIDTH, WIDTH)?,
            });
        }
        let stem = [
            artifact.f32_offset("stem.layer0.conv", STEM0 * 9)?,
            artifact.f32_offset("stem.layer0.norm", STEM0)?,
            artifact.f32_offset("stem.layer1.conv", STEM1 * STEM0 * 9)?,
            artifact.f32_offset("stem.layer1.norm", STEM1)?,
            artifact.bf16("stem.input_proj", WIDTH, WIDTH)?,
        ];
        let output_proj = artifact.bf16("output_proj", OUT_DIMS, WIDTH)?;
        let output_bias = artifact.f32_offset("output_proj.bias", OUT_DIMS)?;
        let embed = artifact.bf16("embed_audio", HIDDEN, OUT_DIMS)?;
        let device = context.device();
        let rows = |width: usize| MetalBuffer::empty(device, TOWER_TOKENS * width * 4);
        let workspace = Workspace {
            frames: MetalBuffer::empty(device, PASS_FRAMES * MEL_BINS * 4)?,
            // Stem layer-0 output of every packed clip: ceil(frames / 2) rows
            // of [64][128] each, plus one rounding row per clip.
            stem: MetalBuffer::empty(device, (PASS_FRAMES / 2 + PASS_CLIPS) * 64 * STEM0 * 4)?,
            hidden: rows(WIDTH)?,
            normed: rows(WIDTH)?,
            q: rows(WIDTH)?,
            k: rows(WIDTH)?,
            v: rows(WIDTH)?,
            attention: rows(WIDTH)?,
            branch: rows(WIDTH)?,
            wide: rows(FF)?,
            projected: rows(OUT_DIMS)?,
            soft: rows(HIDDEN)?,
            first: rows(1)?,
        };
        Ok(Self {
            kernels,
            weights,
            derived: MetalBuffer::from_slice(device, &derived)?,
            artifact,
            stem,
            layers,
            output_proj,
            output_bias,
            embed,
            features: FeatureExtractor::new(),
            workspace,
        })
    }

    pub(super) fn payload_sha256(&self) -> &str {
        &self.artifact.payload_sha256
    }

    const fn matrix(&self, offset: usize, rows: usize, columns: usize) -> Bf16Rows<'_> {
        Bf16Rows {
            weights: &self.weights,
            offset,
            rows: rows as u32,
            columns: columns as u32,
        }
    }

    const fn linear(&self, layer: Linear, rows: usize, columns: usize) -> ClippedLinear<'_> {
        ClippedLinear {
            matrix: self.matrix(layer.offset, rows, columns),
            clip: layer.clip,
            bias: None,
            silu: false,
            accumulate: false,
        }
    }

    /// Soft tokens (`[tokens, 512]` row-major, one block per clip) of clips
    /// whose features fit one pass.
    #[allow(clippy::too_many_lines)]
    fn pass(
        &self,
        model: &EmbeddingModel,
        clips: &[LogMel],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let ws = &self.workspace;
        let kernels = &self.kernels;
        let text = &model.kernels;
        let weights = &self.weights;
        let mut batch = CommandBatch::new(&model.context)?;
        // Stem, clip by clip, into packed rows.
        let mut frame_row = 0_usize;
        let mut stem_row = 0_usize;
        let mut token = 0_usize;
        let mut first = Vec::new();
        let mut tokens_per_clip = Vec::with_capacity(clips.len());
        for clip in clips {
            let n = audio_tokens(clip.valid);
            ws.frames.copy_from_bytes(
                bytemuck::cast_slice(&clip.features),
                frame_row * MEL_BINS * 4,
            );
            let t1 = clip.frames.div_ceil(2);
            let stem_offset = stem_row * 64 * STEM0 * 4;
            kernels.conv2d_s2(
                &mut batch,
                (&ws.frames, frame_row * MEL_BINS * 4),
                (weights, self.stem[0]),
                (&ws.stem, stem_offset),
                [clip.frames as u32, MEL_BINS as u32, 1],
                t1 as u32,
                STEM0 as u32,
            )?;
            kernels.layer_norm_relu(
                &mut batch,
                (&ws.stem, stem_offset),
                (weights, self.stem[1]),
                (t1 * 64) as u32,
                STEM0 as u32,
                64,
                clip.valid.div_ceil(2) as u32,
                EPSILON,
            )?;
            let out_offset = token * STEM1 * STEM1 * 4;
            kernels.conv2d_s2(
                &mut batch,
                (&ws.stem, stem_offset),
                (weights, self.stem[2]),
                (&ws.normed, out_offset),
                [t1 as u32, 64, STEM0 as u32],
                n as u32,
                STEM1 as u32,
            )?;
            kernels.layer_norm_relu(
                &mut batch,
                (&ws.normed, out_offset),
                (weights, self.stem[3]),
                (n * STEM1) as u32,
                STEM1 as u32,
                STEM1 as u32,
                n as u32,
                EPSILON,
            )?;
            first.extend(std::iter::repeat_n(token as u32, n));
            tokens_per_clip.push(n);
            frame_row += clip.frames;
            stem_row += t1;
            token += n;
        }
        ws.first.copy_from_bytes(bytemuck::cast_slice(&first), 0);
        let count = token as u32;
        let width = WIDTH as u32;
        kernels.linear(
            &mut batch,
            ClippedLinear {
                clip: NO_CLIP,
                ..self.linear(
                    Linear {
                        offset: self.stem[4],
                        clip: NO_CLIP,
                    },
                    WIDTH,
                    WIDTH,
                )
            },
            &ws.normed,
            &ws.hidden,
            count,
        )?;
        let layers_per_submit = (SUBMIT_TOKEN_LAYERS / token).max(1);
        let feed_forward = |batch: &mut CommandBatch, ff: &FeedForward| -> crate::Result<()> {
            text.rms(
                batch,
                &ws.hidden,
                Some((weights, ff.pre_norm)),
                &ws.normed,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            kernels.linear(
                batch,
                ClippedLinear {
                    silu: true,
                    ..self.linear(ff.up, FF, WIDTH)
                },
                &ws.normed,
                &ws.wide,
                count,
            )?;
            kernels.linear(
                batch,
                self.linear(ff.down, WIDTH, FF),
                &ws.wide,
                &ws.branch,
                count,
            )?;
            kernels.rms_add(
                batch,
                &ws.hidden,
                &ws.branch,
                (weights, ff.post_norm),
                count,
                width,
                RESIDUAL_WEIGHT,
                EPSILON,
            )?;
            Ok(())
        };
        let k_scale = (std::f64::consts::E.ln_1p() / std::f64::consts::LN_2) as f32;
        for (i, layer) in self.layers.iter().enumerate() {
            feed_forward(&mut batch, &layer.ff1)?;
            // Chunked local self-attention.
            text.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.pre_attn_norm)),
                &ws.normed,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            for (projection, out) in [(layer.q, &ws.q), (layer.k, &ws.k), (layer.v, &ws.v)] {
                kernels.linear(
                    &mut batch,
                    self.linear(projection, WIDTH, WIDTH),
                    &ws.normed,
                    out,
                    count,
                )?;
            }
            kernels.local_attention(
                &mut batch,
                LocalAttention {
                    tokens: count,
                    heads: HEADS as u32,
                    max_distance: MAX_DISTANCE,
                    k_scale,
                    logit_cap: LOGIT_CAP,
                    first: &ws.first,
                    q_scale: (&self.derived, layer.q_scale),
                    rel: (&self.derived, layer.rel),
                },
                &ws.q,
                &ws.k,
                &ws.v,
                &ws.attention,
            )?;
            kernels.linear(
                &mut batch,
                self.linear(layer.post, WIDTH, WIDTH),
                &ws.attention,
                &ws.branch,
                count,
            )?;
            kernels.rms_add(
                &mut batch,
                &ws.hidden,
                &ws.branch,
                (weights, layer.post_attn_norm),
                count,
                width,
                1.0,
                EPSILON,
            )?;
            // Light convolution block.
            text.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.conv_pre_norm)),
                &ws.normed,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            kernels.linear(
                &mut batch,
                self.linear(layer.conv_start, 2 * WIDTH, WIDTH),
                &ws.normed,
                &ws.wide,
                count,
            )?;
            kernels.glu_dwconv(
                &mut batch,
                &ws.wide,
                (weights, layer.depthwise),
                &ws.branch,
                &ws.first,
                count,
                width,
                CONV_KERNEL as u32,
            )?;
            kernels.rms_silu(
                &mut batch,
                &ws.branch,
                (weights, layer.conv_norm),
                count,
                width,
                EPSILON,
            )?;
            kernels.linear(
                &mut batch,
                ClippedLinear {
                    accumulate: true,
                    ..self.linear(layer.conv_end, WIDTH, WIDTH)
                },
                &ws.branch,
                &ws.hidden,
                count,
            )?;
            feed_forward(&mut batch, &layer.ff2)?;
            text.rms(
                &mut batch,
                &ws.hidden,
                Some((weights, layer.out_norm)),
                &ws.hidden,
                count,
                width,
                1.0,
                EPSILON,
            )?;
            if (i + 1) % layers_per_submit == 0 && i + 1 < LAYERS {
                batch.commit_and_renew(&model.context)?;
                pace()?;
            }
        }
        // output_proj (bias), then embed_audio: RMSNorm without scale, 1536 -> 512.
        kernels.linear(
            &mut batch,
            ClippedLinear {
                matrix: self.matrix(self.output_proj, OUT_DIMS, WIDTH),
                clip: NO_CLIP,
                bias: Some((weights, self.output_bias)),
                silu: false,
                accumulate: false,
            },
            &ws.hidden,
            &ws.projected,
            count,
        )?;
        text.rms(
            &mut batch,
            &ws.projected,
            None,
            &ws.projected,
            count,
            OUT_DIMS as u32,
            1.0,
            EPSILON,
        )?;
        kernels.linear(
            &mut batch,
            ClippedLinear {
                clip: NO_CLIP,
                ..self.linear(
                    Linear {
                        offset: self.embed,
                        clip: NO_CLIP,
                    },
                    HIDDEN,
                    OUT_DIMS,
                )
            },
            &ws.projected,
            &ws.soft,
            count,
        )?;
        batch.commit_and_wait()?;
        pace()?;
        let soft = &ws.soft.as_slice::<f32>()[..token * HIDDEN];
        let mut out = Vec::with_capacity(clips.len());
        let mut row = 0;
        for n in tokens_per_clip {
            out.push(soft[row * HIDDEN..(row + n) * HIDDEN].to_vec());
            row += n;
        }
        Ok(out)
    }
}

/// Check a 16 kHz mono clip: 1 to 30 s worth of samples (at least one audio
/// token), all finite.
///
/// # Errors
///
/// [`Error::InvalidArgument`] or [`Error::ContextOverflow`] describing the
/// problem.
pub fn validate_clip(samples: &[f32]) -> crate::Result<()> {
    if samples.len() > MAX_SAMPLES {
        return Err(Error::ContextOverflow(format!(
            "audio clip has {} samples ({:.2} s); the limit is {MAX_SAMPLES} (30 s at 16 kHz)",
            samples.len(),
            samples.len() as f64 / f64::from(SAMPLE_RATE)
        )));
    }
    if audio_tokens(frame_counts(samples.len()).1) == 0 {
        return Err(Error::InvalidArgument(
            "audio clip is too short (at least 161 samples at 16 kHz)".into(),
        ));
    }
    if samples.iter().any(|s| !s.is_finite()) {
        return Err(Error::InvalidArgument(
            "audio clip contains non-finite samples".into(),
        ));
    }
    Ok(())
}

impl EmbeddingModel {
    /// Load the audio encoder artifact from the installation directory
    /// (`dir/embeddinggemma2-audio-bf16.bin`, written by
    /// `tools/convert_embeddinggemma2_audio.py`). About 610 MB of GPU memory.
    /// Optional: the first audio input loads it from the model's own install
    /// directory.
    ///
    /// # Errors
    ///
    /// Fails when the artifact is missing, does not verify, or Metal fails.
    pub fn load_audio(&mut self, dir: &Path) -> crate::Result<()> {
        self.audio = Some(AudioTower::open(&dir.join(AUDIO_ARTIFACT), &self.context)?);
        Ok(())
    }

    /// Whether the audio encoder is loaded or installed next to the text
    /// artifact (audio inputs are refused otherwise).
    #[must_use]
    pub fn supports_audio(&self) -> bool {
        self.audio.is_some() || self.dir.join(AUDIO_ARTIFACT).is_file()
    }

    /// Whether the audio encoder is loaded.
    #[must_use]
    pub const fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// SHA-256 of the verified audio artifact payload, if loaded.
    #[must_use]
    pub fn audio_payload_sha256(&self) -> Option<&str> {
        self.audio.as_ref().map(AudioTower::payload_sha256)
    }

    /// The reference log-mel features of a 16 kHz mono clip.
    #[must_use]
    pub fn audio_features(samples: &[f32]) -> LogMel {
        FeatureExtractor::new().extract(samples)
    }

    /// Audio soft tokens (`[tokens, 512]`, the text encoder's input
    /// embeddings for the clip) of 16 kHz mono clips, loading the audio
    /// artifact from the install directory on first use.
    ///
    /// # Errors
    ///
    /// As [`Self::embed_audio`].
    pub fn audio_soft_tokens(
        &mut self,
        clips: &[&[f32]],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        for clip in clips {
            validate_clip(clip)?;
        }
        if self.audio.is_none() {
            let path = self.dir.join(AUDIO_ARTIFACT);
            if !path.is_file() {
                return Err(Error::InvalidArgument(format!(
                    "audio embeddings need {} (see docs/EMBEDDINGS_AUDIO.md)",
                    path.display()
                )));
            }
            self.audio = Some(AudioTower::open(&path, &self.context)?);
        }
        let tower = self
            .audio
            .as_ref()
            .ok_or_else(|| Error::Generation("audio encoder missing".into()))?;
        let mut out = Vec::with_capacity(clips.len());
        let mut pending: Vec<LogMel> = Vec::new();
        let (mut frames, mut tokens) = (0, 0);
        for clip in clips {
            let features = tower.features.extract(clip);
            let n = audio_tokens(features.valid);
            if !pending.is_empty()
                && (frames + features.frames > PASS_FRAMES
                    || tokens + n > TOWER_TOKENS
                    || pending.len() == PASS_CLIPS)
            {
                out.extend(tower.pass(self, &pending, pace)?);
                pending.clear();
                (frames, tokens) = (0, 0);
            }
            frames += features.frames;
            tokens += n;
            pending.push(features);
        }
        if !pending.is_empty() {
            out.extend(tower.pass(self, &pending, pace)?);
        }
        Ok(out)
    }

    /// Embed 16 kHz mono clips (at most 30 s each; see [`decode_audio`] for
    /// files): one full-width L2-normalized vector per clip, in the same
    /// space as text embeddings. Each clip is a one-part
    /// [`Self::embed_multimodal`] input, the reference's media sequence
    /// `<bos> <start_of_audio> soft tokens <end_of_audio> <eos>` with no text
    /// prompt. `pace` runs between command buffers as in
    /// [`Self::embed_ids`].
    ///
    /// # Errors
    ///
    /// The audio artifact is not installed, a clip is invalid
    /// ([`validate_clip`]), a `pace` error, or a Metal failure.
    pub fn embed_audio(
        &mut self,
        clips: &[&[f32]],
        pace: &mut dyn FnMut() -> crate::Result<()>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let inputs: Vec<Vec<MediaPart<'_>>> = clips
            .iter()
            .map(|clip| vec![MediaPart::Audio(clip)])
            .collect();
        self.embed_multimodal(&inputs, pace)
    }
}

#[cfg(test)]
mod tests;
