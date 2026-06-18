use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::f16;
use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kernels::Kernels;
use rustfft::{FftPlanner, num_complex::Complex32};

use crate::Error;
use crate::layer::QuantWeight;
use crate::multimodal::PcmAudio;

const TARGET_SAMPLE_RATE: u32 = 16_000;
const FEATURE_SIZE: usize = 128;
const FRAME_LENGTH: usize = 320;
const HOP_LENGTH: usize = 160;
const FFT_LENGTH: usize = 512;
const MEL_FLOOR: f32 = 1e-5;
const MAX_SAMPLES: usize = 480_000;
const AUDIO_HIDDEN: usize = 1024;
const AUDIO_INTERMEDIATE: usize = AUDIO_HIDDEN * 4;
const AUDIO_LAYERS: usize = 12;
const TEXT_HIDDEN: usize = 1536;
const AUDIO_HEADS: usize = 8;
const AUDIO_HEAD_DIM: usize = AUDIO_HIDDEN / AUDIO_HEADS;
const AUDIO_CHUNK_SIZE: usize = 12;
const AUDIO_LEFT_CONTEXT: usize = 12;
const AUDIO_REL_POSITIONS: usize = AUDIO_LEFT_CONTEXT + 1;
const AUDIO_ATTENTION_SOFTCAP: f32 = 50.0;
const AUDIO_CONV_KERNEL: usize = 5;
const AUDIO_RESIDUAL_WEIGHT: f32 = 0.5;
const F16: usize = std::mem::size_of::<f16>();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavInfo {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    format: u16,
    data_offset: usize,
    data_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub frames: Vec<[f32; FEATURE_SIZE]>,
    pub mask: Vec<bool>,
}

pub struct AudioFrontendOutput {
    pub soft_tokens: usize,
    projected: MetalBuffer,
}

pub fn extract_audio_features(path: &Path) -> crate::Result<AudioFeatures> {
    let samples = load_wav_mono_f32_resampled_16khz(path)?;
    Ok(extract_log_mel_features(&samples))
}

pub fn extract_pcm_audio_features(audio: &PcmAudio) -> crate::Result<AudioFeatures> {
    let samples = pcm_mono_f32_resampled_16khz(audio)?;
    Ok(extract_log_mel_features(&samples))
}

pub fn run_audio_frontend(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    path: &Path,
) -> crate::Result<AudioFrontendOutput> {
    let features = extract_audio_features(path)?;
    if features.frames.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "audio {} produced no Gemma 4 log-mel frames",
            path.display()
        )));
    }
    run_audio_frontend_from_features(ctx, kernels, weights, &features)
}

pub fn run_pcm_audio_frontend(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    audio: &PcmAudio,
) -> crate::Result<AudioFrontendOutput> {
    let features = extract_pcm_audio_features(audio)?;
    if features.frames.is_empty() {
        return Err(Error::InvalidArgument(
            "PCM audio produced no Gemma 4 log-mel frames".into(),
        ));
    }
    run_audio_frontend_from_features(ctx, kernels, weights, &features)
}

fn run_audio_frontend_from_features(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    features: &AudioFeatures,
) -> crate::Result<AudioFrontendOutput> {
    let mel_frames = features.frames.len();
    let device = ctx.device();
    let mut input_data = Vec::with_capacity(mel_frames * FEATURE_SIZE);
    for (frame, &valid) in features.frames.iter().zip(&features.mask) {
        input_data.extend(
            frame
                .iter()
                .map(|&v| if valid { f16::from_f32(v) } else { f16::ZERO }),
        );
    }
    let input = MetalBuffer::from_slice(device, &input_data)
        .map_err(|e| Error::InvalidArgument(format!("audio feature input buffer: {e}")))?;

    let time0 = mel_frames.div_ceil(2);
    let freq0 = FEATURE_SIZE / 2;
    let stage0 = MetalBuffer::empty(device, 128 * time0 * freq0 * std::mem::size_of::<f16>())
        .map_err(|e| Error::InvalidArgument(format!("audio frontend stage0 buffer: {e}")))?;
    audio_conv_stage(
        ctx,
        kernels,
        weights,
        &input,
        &stage0,
        "a.conv1d.0",
        1,
        128,
        mel_frames,
        FEATURE_SIZE,
        time0,
        freq0,
    )?;

    let time1 = time0.div_ceil(2);
    let freq1 = freq0 / 2;
    let stage1 = MetalBuffer::empty(device, 32 * time1 * freq1 * std::mem::size_of::<f16>())
        .map_err(|e| Error::InvalidArgument(format!("audio frontend stage1 buffer: {e}")))?;
    audio_conv_stage(
        ctx,
        kernels,
        weights,
        &stage0,
        &stage1,
        "a.conv1d.1",
        128,
        32,
        time0,
        freq0,
        time1,
        freq1,
    )?;

    let packed_dim = freq1 * 32;
    let packed = MetalBuffer::empty(device, time1 * packed_dim * std::mem::size_of::<f16>())
        .map_err(|e| Error::InvalidArgument(format!("audio frontend packed buffer: {e}")))?;
    kernels
        .audio_pack_frontend(ctx, &stage1, &packed, 32, time1 as u32, freq1 as u32)
        .map_err(Error::Metal)?;

    let input_projection = weights
        .get("a.input_projection.weight")
        .ok_or_else(|| Error::InvalidFormat("missing a.input_projection.weight".into()))?;
    if input_projection.cols() as usize != packed_dim
        || input_projection.rows() as usize != AUDIO_HIDDEN
    {
        return Err(Error::InvalidFormat(format!(
            "a.input_projection.weight has shape [{}, {}], expected [{AUDIO_HIDDEN}, {packed_dim}]",
            input_projection.rows(),
            input_projection.cols()
        )));
    }
    let projected = MetalBuffer::empty(device, time1 * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio frontend projection buffer: {e}")))?;
    let mut batch = CommandBatch::new(ctx).map_err(Error::Metal)?;
    input_projection.matmul_nt_into(&mut batch, kernels, &packed, &projected, time1 as u32)?;
    batch.commit_and_wait().map_err(Error::Metal)?;

    Ok(AudioFrontendOutput {
        soft_tokens: time1,
        projected,
    })
}

pub fn run_audio_first_ffn_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &mut AudioFrontendOutput,
) -> crate::Result<()> {
    for layer in 0..AUDIO_LAYERS {
        run_audio_ffn1_stage(ctx, kernels, weights, frontend, layer)?;
    }
    Ok(())
}

pub fn project_audio_soft_tokens(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &AudioFrontendOutput,
) -> crate::Result<Vec<Vec<f16>>> {
    let tokens = frontend.soft_tokens;
    let device = ctx.device();
    let pre_encoded = MetalBuffer::empty(device, tokens * TEXT_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio pre-encode output buffer: {e}")))?;
    let pre_weight_name = "a.pre_encode.out.weight";
    let pre_weight = weights
        .get(pre_weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {pre_weight_name}")))?;
    if pre_weight.cols() as usize != AUDIO_HIDDEN || pre_weight.rows() as usize != TEXT_HIDDEN {
        return Err(Error::InvalidFormat(format!(
            "{pre_weight_name} has shape [{}, {}], expected [{TEXT_HIDDEN}, {AUDIO_HIDDEN}]",
            pre_weight.rows(),
            pre_weight.cols()
        )));
    }
    let pre_bias = weights
        .get("a.pre_encode.out.bias")
        .ok_or_else(|| Error::InvalidFormat("missing a.pre_encode.out.bias".into()))?
        .as_f16_buffer("a.pre_encode.out.bias")?;
    kernels
        .matmul_f16_bias(
            ctx,
            &frontend.projected,
            pre_weight.as_f16_buffer(pre_weight_name)?,
            Some(pre_bias),
            &pre_encoded,
            tokens as u32,
            AUDIO_HIDDEN as u32,
            TEXT_HIDDEN as u32,
        )
        .map_err(Error::Metal)?;

    let ones = MetalBuffer::from_slice(device, &vec![f16::from_f32(1.0); TEXT_HIDDEN])
        .map_err(|e| Error::InvalidArgument(format!("audio projector norm weight buffer: {e}")))?;
    let normed = MetalBuffer::empty(device, tokens * TEXT_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio projector norm buffer: {e}")))?;
    kernels
        .rms_norm(
            ctx,
            &pre_encoded,
            &ones,
            &normed,
            TEXT_HIDDEN as u32,
            tokens as u32,
            1e-6,
        )
        .map_err(Error::Metal)?;

    let projected = MetalBuffer::empty(device, tokens * TEXT_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio text projection buffer: {e}")))?;
    let projector_name = "mm.a.input_projection.weight";
    let projector = weights
        .get(projector_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {projector_name}")))?;
    if projector.cols() as usize != TEXT_HIDDEN || projector.rows() as usize != TEXT_HIDDEN {
        return Err(Error::InvalidFormat(format!(
            "{projector_name} has shape [{}, {}], expected [{TEXT_HIDDEN}, {TEXT_HIDDEN}]",
            projector.rows(),
            projector.cols()
        )));
    }
    let mut batch = CommandBatch::new(ctx).map_err(Error::Metal)?;
    projector.matmul_nt_into(&mut batch, kernels, &normed, &projected, tokens as u32)?;
    batch.commit_and_wait().map_err(Error::Metal)?;

    let flat = projected.as_slice::<f16>();
    Ok(flat
        .chunks_exact(TEXT_HIDDEN)
        .map(<[f16]>::to_vec)
        .collect())
}

fn run_audio_ffn1_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &mut AudioFrontendOutput,
    layer: usize,
) -> crate::Result<()> {
    let tokens = frontend.soft_tokens;
    if tokens == 0 {
        return Err(Error::InvalidArgument(
            "audio frontend produced zero soft-token candidates".into(),
        ));
    }

    let device = ctx.device();
    let norm = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn norm buffer: {e}")))?;
    let up = MetalBuffer::empty(device, tokens * AUDIO_INTERMEDIATE * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn up buffer: {e}")))?;
    let down = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn down buffer: {e}")))?;
    let post = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn post-norm buffer: {e}")))?;
    let residual_out = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn residual buffer: {e}")))?;

    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &frontend.projected,
        &norm,
        &format!("a.blk.{layer}.ffn_norm.weight"),
        tokens,
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &up,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_INTERMEDIATE,
        &format!("a.blk.{layer}.ffn_up"),
    )?;
    kernels
        .silu_inplace(ctx, &up, (tokens * AUDIO_INTERMEDIATE) as u32)
        .map_err(Error::Metal)?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &up,
        &down,
        tokens,
        AUDIO_INTERMEDIATE,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.ffn_down"),
    )?;
    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &down,
        &post,
        &format!("a.blk.{layer}.ffn_post_norm.weight"),
        tokens,
    )?;
    kernels
        .scale_in_place_gpu(
            ctx,
            &post,
            AUDIO_RESIDUAL_WEIGHT,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    kernels
        .residual_add(
            ctx,
            &frontend.projected,
            &post,
            &residual_out,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    frontend.projected = residual_out;
    run_audio_attention_stage(ctx, kernels, weights, frontend, layer)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_audio_attention_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &mut AudioFrontendOutput,
    layer: usize,
) -> crate::Result<()> {
    let tokens = frontend.soft_tokens;
    let device = ctx.device();
    let norm = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention norm buffer: {e}")))?;
    let q = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention q buffer: {e}")))?;
    let k = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention k buffer: {e}")))?;
    let v = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention v buffer: {e}")))?;
    let rel_pos = MetalBuffer::from_slice(device, &audio_rel_pos_embedding())
        .map_err(|e| Error::InvalidArgument(format!("audio relative position buffer: {e}")))?;
    let rel_k = MetalBuffer::empty(device, AUDIO_REL_POSITIONS * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio relative key buffer: {e}")))?;
    let attn = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention output buffer: {e}")))?;
    let attn_proj = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention projection buffer: {e}")))?;
    let post = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention post-norm buffer: {e}")))?;
    let residual_out = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio attention residual buffer: {e}")))?;

    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &frontend.projected,
        &norm,
        &format!("a.blk.{layer}.attn_pre_norm.weight"),
        tokens,
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &q,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.attn_q"),
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &k,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.attn_k"),
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &v,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.attn_v"),
    )?;

    let rel_weight_name = format!("a.blk.{layer}.attn_k_rel.weight");
    let rel_weight = weights
        .get(&rel_weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {rel_weight_name}")))?;
    if rel_weight.cols() as usize != AUDIO_HIDDEN || rel_weight.rows() as usize != AUDIO_HIDDEN {
        return Err(Error::InvalidFormat(format!(
            "{rel_weight_name} has shape [{}, {}], expected [{AUDIO_HIDDEN}, {AUDIO_HIDDEN}]",
            rel_weight.rows(),
            rel_weight.cols()
        )));
    }
    let mut batch = CommandBatch::new(ctx).map_err(Error::Metal)?;
    rel_weight.matmul_nt_into(
        &mut batch,
        kernels,
        &rel_pos,
        &rel_k,
        AUDIO_REL_POSITIONS as u32,
    )?;
    batch.commit_and_wait().map_err(Error::Metal)?;

    let per_dim_scale_name = format!("a.blk.{layer}.per_dim_scale.weight");
    let per_dim_scale = weights
        .get(&per_dim_scale_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {per_dim_scale_name}")))?
        .as_f16_buffer(&per_dim_scale_name)?;
    kernels
        .audio_chunked_attention(
            ctx,
            &q,
            &k,
            &v,
            &rel_k,
            per_dim_scale,
            &attn,
            tokens as u32,
            AUDIO_HEADS as u32,
            AUDIO_HEAD_DIM as u32,
            AUDIO_CHUNK_SIZE as u32,
            AUDIO_LEFT_CONTEXT as u32,
            AUDIO_ATTENTION_SOFTCAP,
        )
        .map_err(Error::Metal)?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &attn,
        &attn_proj,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.attn_out"),
    )?;
    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &attn_proj,
        &post,
        &format!("a.blk.{layer}.attn_post_norm.weight"),
        tokens,
    )?;
    kernels
        .residual_add(
            ctx,
            &frontend.projected,
            &post,
            &residual_out,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    frontend.projected = residual_out;
    run_audio_light_conv_stage(ctx, kernels, weights, frontend, layer)?;
    Ok(())
}

fn run_audio_light_conv_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &mut AudioFrontendOutput,
    layer: usize,
) -> crate::Result<()> {
    let tokens = frontend.soft_tokens;
    let device = ctx.device();
    let norm = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv norm buffer: {e}")))?;
    let pw1 = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * 2 * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv pw1 buffer: {e}")))?;
    let glu = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv glu buffer: {e}")))?;
    let dw = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv depthwise buffer: {e}")))?;
    let conv_norm = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv conv-norm buffer: {e}")))?;
    let pw2 = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv pw2 buffer: {e}")))?;
    let residual_out = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio light-conv residual buffer: {e}")))?;

    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &frontend.projected,
        &norm,
        &format!("a.blk.{layer}.norm_conv.weight"),
        tokens,
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &pw1,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN * 2,
        &format!("a.blk.{layer}.conv_pw1"),
    )?;
    kernels
        .audio_glu(ctx, &pw1, &glu, tokens as u32, AUDIO_HIDDEN as u32)
        .map_err(Error::Metal)?;

    let conv_dw_name = format!("a.blk.{layer}.conv_dw.weight");
    let conv_dw = weights
        .get(&conv_dw_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {conv_dw_name}")))?;
    if conv_dw.cols() as usize != AUDIO_CONV_KERNEL || conv_dw.rows() as usize != AUDIO_HIDDEN {
        return Err(Error::InvalidFormat(format!(
            "{conv_dw_name} has shape [{}, {}], expected [{AUDIO_HIDDEN}, {AUDIO_CONV_KERNEL}]",
            conv_dw.rows(),
            conv_dw.cols()
        )));
    }
    kernels
        .depthwise_conv1d(
            ctx,
            &glu,
            conv_dw.as_f16_buffer(&conv_dw_name)?,
            &dw,
            tokens as u32,
            AUDIO_HIDDEN as u32,
            AUDIO_CONV_KERNEL as u32,
        )
        .map_err(Error::Metal)?;
    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &dw,
        &conv_norm,
        &format!("a.blk.{layer}.conv_norm.weight"),
        tokens,
    )?;
    kernels
        .silu_inplace(ctx, &conv_norm, (tokens * AUDIO_HIDDEN) as u32)
        .map_err(Error::Metal)?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &conv_norm,
        &pw2,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.conv_pw2"),
    )?;
    kernels
        .residual_add(
            ctx,
            &frontend.projected,
            &pw2,
            &residual_out,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    frontend.projected = residual_out;
    run_audio_ffn2_stage(ctx, kernels, weights, frontend, layer)?;
    Ok(())
}

fn run_audio_ffn2_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    frontend: &mut AudioFrontendOutput,
    layer: usize,
) -> crate::Result<()> {
    let tokens = frontend.soft_tokens;
    let device = ctx.device();
    let norm = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn2 norm buffer: {e}")))?;
    let up = MetalBuffer::empty(device, tokens * AUDIO_INTERMEDIATE * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn2 up buffer: {e}")))?;
    let down = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn2 down buffer: {e}")))?;
    let post = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn2 post-norm buffer: {e}")))?;
    let residual_out = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio ffn2 residual buffer: {e}")))?;
    let norm_out = MetalBuffer::empty(device, tokens * AUDIO_HIDDEN * F16)
        .map_err(|e| Error::InvalidArgument(format!("audio layer output norm buffer: {e}")))?;

    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &frontend.projected,
        &norm,
        &format!("a.blk.{layer}.ffn_norm_1.weight"),
        tokens,
    )?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &norm,
        &up,
        tokens,
        AUDIO_HIDDEN,
        AUDIO_INTERMEDIATE,
        &format!("a.blk.{layer}.ffn_up_1"),
    )?;
    kernels
        .silu_inplace(ctx, &up, (tokens * AUDIO_INTERMEDIATE) as u32)
        .map_err(Error::Metal)?;
    audio_clipped_linear(
        ctx,
        kernels,
        weights,
        &up,
        &down,
        tokens,
        AUDIO_INTERMEDIATE,
        AUDIO_HIDDEN,
        &format!("a.blk.{layer}.ffn_down_1"),
    )?;
    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &down,
        &post,
        &format!("a.blk.{layer}.ffn_post_norm_1.weight"),
        tokens,
    )?;
    kernels
        .scale_in_place_gpu(
            ctx,
            &post,
            AUDIO_RESIDUAL_WEIGHT,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    kernels
        .residual_add(
            ctx,
            &frontend.projected,
            &post,
            &residual_out,
            (tokens * AUDIO_HIDDEN) as u32,
        )
        .map_err(Error::Metal)?;
    audio_rms_norm(
        ctx,
        kernels,
        weights,
        &residual_out,
        &norm_out,
        &format!("a.blk.{layer}.ln2.weight"),
        tokens,
    )?;
    frontend.projected = norm_out;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn audio_conv_stage(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    in_time: usize,
    in_freq: usize,
    out_time: usize,
    out_freq: usize,
) -> crate::Result<()> {
    let weight_name = format!("{prefix}.weight");
    let norm_name = format!("{prefix}.norm.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {weight_name}")))?
        .as_f16_buffer(&weight_name)?;
    let norm = weights
        .get(&norm_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {norm_name}")))?
        .as_f16_buffer(&norm_name)?;
    kernels
        .audio_subsample_conv2d_ln_relu(
            ctx,
            input,
            weight,
            norm,
            output,
            in_channels as u32,
            out_channels as u32,
            in_time as u32,
            in_freq as u32,
            out_time as u32,
            out_freq as u32,
            1e-6,
        )
        .map_err(Error::Metal)
}

fn audio_rms_norm(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    weight_name: &str,
    rows: usize,
) -> crate::Result<()> {
    let weight = weights
        .get(weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {weight_name}")))?
        .as_f16_buffer(weight_name)?;
    kernels
        .rms_norm(
            ctx,
            input,
            weight,
            output,
            AUDIO_HIDDEN as u32,
            rows as u32,
            1e-6,
        )
        .map_err(Error::Metal)
}

#[allow(clippy::too_many_arguments)]
fn audio_clipped_linear(
    ctx: &MetalContext,
    kernels: &Kernels,
    weights: &HashMap<String, QuantWeight>,
    input: &MetalBuffer,
    output: &MetalBuffer,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    prefix: &str,
) -> crate::Result<()> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {weight_name}")))?;
    if weight.cols() as usize != input_dim || weight.rows() as usize != output_dim {
        return Err(Error::InvalidFormat(format!(
            "{weight_name} has shape [{}, {}], expected [{output_dim}, {input_dim}]",
            weight.rows(),
            weight.cols()
        )));
    }
    kernels
        .clipped_linear(
            ctx,
            input,
            weight.as_f16_buffer(&weight_name)?,
            None,
            output,
            rows as u32,
            input_dim as u32,
            output_dim as u32,
            scalar(weights, &format!("{prefix}.input_min"))?,
            scalar(weights, &format!("{prefix}.input_max"))?,
            scalar(weights, &format!("{prefix}.output_min"))?,
            scalar(weights, &format!("{prefix}.output_max"))?,
        )
        .map_err(Error::Metal)
}

fn scalar(weights: &HashMap<String, QuantWeight>, name: &str) -> crate::Result<f32> {
    let buf = weights
        .get(name)
        .ok_or_else(|| Error::InvalidFormat(format!("missing {name}")))?
        .as_f16_buffer(name)?;
    Ok(buf
        .as_slice::<f16>()
        .first()
        .copied()
        .unwrap_or(f16::ZERO)
        .to_f32())
}

fn audio_rel_pos_embedding() -> Vec<f16> {
    let half_dim = AUDIO_HIDDEN / 2;
    let log_increment = 10000.0_f32.ln() / (half_dim.saturating_sub(1).max(1) as f32);
    let mut out = Vec::with_capacity(AUDIO_REL_POSITIONS * AUDIO_HIDDEN);
    for pos in (0..=AUDIO_LEFT_CONTEXT).rev() {
        for i in 0..half_dim {
            let inv_timescale = (-log_increment * i as f32).exp();
            out.push(f16::from_f32((pos as f32 * inv_timescale).sin()));
        }
        for i in 0..half_dim {
            let inv_timescale = (-log_increment * i as f32).exp();
            out.push(f16::from_f32((pos as f32 * inv_timescale).cos()));
        }
    }
    out
}

pub fn wav_sample_count_16khz(path: &Path) -> crate::Result<usize> {
    let bytes = fs::read(path).map_err(|e| {
        Error::InvalidArgument(format!("failed to read audio {}: {e}", path.display()))
    })?;
    let info = parse_wav_info(&bytes)?;
    if info.sample_rate == 0 {
        return Err(Error::InvalidArgument(format!(
            "audio {} declares a zero sample rate",
            path.display()
        )));
    }
    if info.channels == 0 {
        return Err(Error::InvalidFormat(format!(
            "audio {} declares zero channels",
            path.display()
        )));
    }
    let bytes_per_sample = usize::from(info.bits_per_sample / 8);
    if bytes_per_sample == 0 {
        return Err(Error::InvalidFormat(format!(
            "audio {} has invalid bit depth {}",
            path.display(),
            info.bits_per_sample
        )));
    }
    let source_frames = info.data_bytes / bytes_per_sample / usize::from(info.channels);
    Ok(resampled_len(source_frames, info.sample_rate).min(MAX_SAMPLES))
}

pub fn pcm_sample_count_16khz(audio: &PcmAudio) -> crate::Result<usize> {
    if audio.sample_rate == 0 {
        return Err(Error::InvalidArgument(
            "PCM audio declares a zero sample rate".into(),
        ));
    }
    if audio.channels == 0 {
        return Err(Error::InvalidFormat(
            "PCM audio declares zero channels".into(),
        ));
    }
    let source_frames = audio.samples.len() / usize::from(audio.channels);
    Ok(resampled_len(source_frames, audio.sample_rate).min(MAX_SAMPLES))
}

fn load_wav_mono_f32_resampled_16khz(path: &Path) -> crate::Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|e| {
        Error::InvalidArgument(format!("failed to read audio {}: {e}", path.display()))
    })?;
    let info = parse_wav_info(&bytes)?;
    validate_audio_shape(
        info.sample_rate,
        info.channels,
        &format!("audio {}", path.display()),
    )?;
    let data = &bytes[info.data_offset..info.data_offset + info.data_bytes];
    let bytes_per_sample = usize::from(info.bits_per_sample / 8);
    let channels = usize::from(info.channels);
    let frame_count = data.len() / bytes_per_sample / channels;
    let source = decode_interleaved_to_mono(frame_count, info.sample_rate, channels, |idx| {
        let offset = idx * bytes_per_sample;
        match (info.format, info.bits_per_sample) {
            (1, 16) => {
                let raw = i16::from_le_bytes([data[offset], data[offset + 1]]);
                f32::from(raw) / 32768.0
            }
            (3, 32) => f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]),
            _ => unreachable!("parse_wav_info rejects unsupported formats"),
        }
    });
    Ok(resample_to_16khz(&source, info.sample_rate))
}

/// Decode in-memory WAV bytes (PCM16 or float32) into a [`PcmAudio`] payload,
/// preserving the original sample rate and channel layout. Used by the HTTP
/// server to accept `OpenAI` `input_audio` data without touching the filesystem.
///
/// # Errors
///
/// Returns an error when the bytes are not a supported RIFF/WAVE PCM16/float32
/// file.
pub fn decode_wav_bytes(bytes: &[u8]) -> crate::Result<PcmAudio> {
    let info = parse_wav_info(bytes)?;
    let data = &bytes[info.data_offset..info.data_offset + info.data_bytes];
    let bytes_per_sample = usize::from(info.bits_per_sample / 8);
    let channels = usize::from(info.channels);
    if bytes_per_sample == 0 || channels == 0 {
        return Err(Error::InvalidFormat(
            "WAV reports zero channels or sample width".into(),
        ));
    }
    let sample_count = data.len() / bytes_per_sample;
    let mut samples = Vec::with_capacity(sample_count);
    for idx in 0..sample_count {
        let offset = idx * bytes_per_sample;
        let value = match (info.format, info.bits_per_sample) {
            (1, 16) => f32::from(i16::from_le_bytes([data[offset], data[offset + 1]])) / 32768.0,
            (3, 32) => f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]),
            _ => unreachable!("parse_wav_info rejects unsupported formats"),
        };
        samples.push(value);
    }
    Ok(PcmAudio {
        samples,
        sample_rate: info.sample_rate,
        channels: info.channels,
    })
}

fn pcm_mono_f32_resampled_16khz(audio: &PcmAudio) -> crate::Result<Vec<f32>> {
    validate_audio_shape(audio.sample_rate, audio.channels, "PCM audio")?;
    let channels = usize::from(audio.channels);
    let frame_count = audio.samples.len() / channels;
    let source = decode_interleaved_to_mono(frame_count, audio.sample_rate, channels, |idx| {
        audio.samples[idx]
    });
    Ok(resample_to_16khz(&source, audio.sample_rate))
}

fn validate_audio_shape(sample_rate: u32, channels: u16, label: &str) -> crate::Result<()> {
    if sample_rate == 0 {
        return Err(Error::InvalidArgument(format!(
            "{label} declares a zero sample rate"
        )));
    }
    if channels == 0 {
        return Err(Error::InvalidFormat(format!(
            "{label} declares zero channels"
        )));
    }
    Ok(())
}

fn max_source_frames_for_16khz_limit(source_rate: u32) -> usize {
    if source_rate == TARGET_SAMPLE_RATE {
        MAX_SAMPLES
    } else {
        ((MAX_SAMPLES as u64 * u64::from(source_rate)) / u64::from(TARGET_SAMPLE_RATE))
            .saturating_add(2)
            .min(usize::MAX as u64) as usize
    }
}

fn decode_interleaved_to_mono(
    frame_count: usize,
    sample_rate: u32,
    channels: usize,
    mut sample_at: impl FnMut(usize) -> f32,
) -> Vec<f32> {
    let decode_frames = frame_count.min(max_source_frames_for_16khz_limit(sample_rate));
    let mut source = Vec::with_capacity(decode_frames);
    for frame in 0..decode_frames {
        let offset = frame * channels;
        let mut sum = 0.0_f32;
        for channel in 0..channels {
            sum += sample_at(offset + channel);
        }
        source.push(sum / channels as f32);
    }
    source
}

fn resampled_len(source_frames: usize, source_rate: u32) -> usize {
    if source_frames == 0 || source_rate == 0 {
        return 0;
    }
    ((source_frames as u128 * u128::from(TARGET_SAMPLE_RATE)).div_ceil(u128::from(source_rate)))
        .min(usize::MAX as u128) as usize
}

fn resample_to_16khz(samples: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == TARGET_SAMPLE_RATE {
        return samples.iter().copied().take(MAX_SAMPLES).collect();
    }
    if samples.is_empty() || source_rate == 0 {
        return Vec::new();
    }
    let out_len = resampled_len(samples.len(), source_rate).min(MAX_SAMPLES);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * f64::from(source_rate) / f64::from(TARGET_SAMPLE_RATE);
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a.mul_add(1.0 - frac, b * frac));
    }
    out
}

fn extract_log_mel_features(samples: &[f32]) -> AudioFeatures {
    let real_len = samples.len().min(MAX_SAMPLES);
    let padded_len = round_up(real_len, 128);
    let mut waveform = Vec::with_capacity(padded_len + FRAME_LENGTH / 2);
    waveform.extend(std::iter::repeat_n(0.0, FRAME_LENGTH / 2));
    waveform.extend_from_slice(&samples[..real_len]);
    waveform.extend(std::iter::repeat_n(0.0, padded_len - real_len));

    let mut attention = Vec::with_capacity(waveform.len());
    attention.extend(std::iter::repeat_n(false, FRAME_LENGTH / 2));
    attention.extend(std::iter::repeat_n(true, real_len));
    attention.extend(std::iter::repeat_n(false, padded_len - real_len));

    let window = periodic_hann_window();
    let filters = mel_filter_bank();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_LENGTH);
    let frame_size_for_unfold = FRAME_LENGTH + 1;
    let num_frames = waveform
        .len()
        .saturating_sub(frame_size_for_unfold)
        .checked_div(HOP_LENGTH)
        .map_or(0, |n| n + 1);
    let mut frames = Vec::with_capacity(num_frames);
    let mut mask = Vec::with_capacity(num_frames);
    let mut buf = vec![Complex32::new(0.0, 0.0); FFT_LENGTH];
    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        buf.fill(Complex32::new(0.0, 0.0));
        for i in 0..FRAME_LENGTH {
            buf[i].re = waveform[start + i] * window[i];
        }
        fft.process(&mut buf);
        let mut mel = [0.0_f32; FEATURE_SIZE];
        for (bin, coeff) in buf.iter().take(FFT_LENGTH / 2 + 1).enumerate() {
            let mag = coeff.norm();
            for m in 0..FEATURE_SIZE {
                mel[m] = mag.mul_add(filters[bin][m], mel[m]);
            }
        }
        for value in &mut mel {
            *value = (*value + MEL_FLOOR).ln();
        }
        frames.push(mel);
        let frame_end = start + frame_size_for_unfold - 1;
        mask.push(attention.get(frame_end).copied().unwrap_or(false));
    }
    AudioFeatures { frames, mask }
}

const fn round_up(value: usize, multiple: usize) -> usize {
    if value == 0 {
        0
    } else {
        value.div_ceil(multiple) * multiple
    }
}

fn periodic_hann_window() -> [f32; FRAME_LENGTH] {
    let mut out = [0.0_f32; FRAME_LENGTH];
    for (i, value) in out.iter_mut().enumerate() {
        *value = 0.5f32.mul_add(
            -(2.0 * std::f32::consts::PI * i as f32 / FRAME_LENGTH as f32).cos(),
            0.5,
        );
    }
    out
}

fn mel_filter_bank() -> Vec<[f32; FEATURE_SIZE]> {
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(TARGET_SAMPLE_RATE as f32 / 2.0);
    let mut mel_points = [0.0_f32; FEATURE_SIZE + 2];
    for (i, point) in mel_points.iter_mut().enumerate() {
        *point = min_mel + (max_mel - min_mel) * i as f32 / (FEATURE_SIZE + 1) as f32;
    }
    let hz_points = mel_points.map(mel_to_hz);
    let bin_freqs: Vec<f32> = (0..=FFT_LENGTH / 2)
        .map(|bin| bin as f32 * TARGET_SAMPLE_RATE as f32 / FFT_LENGTH as f32)
        .collect();
    let mut filters = vec![[0.0_f32; FEATURE_SIZE]; FFT_LENGTH / 2 + 1];
    for mel in 0..FEATURE_SIZE {
        let left = hz_points[mel];
        let center = hz_points[mel + 1];
        let right = hz_points[mel + 2];
        for (bin, &freq) in bin_freqs.iter().enumerate() {
            filters[bin][mel] = if freq < left || freq > right {
                0.0
            } else if freq <= center {
                (freq - left) / (center - left).max(f32::EPSILON)
            } else {
                (right - freq) / (right - center).max(f32::EPSILON)
            };
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10_f32.powf(mel / 2595.0) - 1.0)
}

const fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

const fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn parse_wav_info(bytes: &[u8]) -> crate::Result<WavInfo> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(Error::InvalidFormat(
            "audio input must be a RIFF/WAVE file".into(),
        ));
    }

    let mut offset: usize = 12;
    let mut format = None;
    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data_offset = None;
    let mut data_bytes = None;
    while offset.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_len = read_u32_le(bytes, offset + 4) as usize;
        offset += 8;
        let chunk_end = offset.checked_add(chunk_len).ok_or_else(|| {
            Error::InvalidFormat("WAV chunk length overflows address space".into())
        })?;
        if chunk_end > bytes.len() {
            return Err(Error::InvalidFormat(
                "WAV chunk extends past end of file".into(),
            ));
        }

        match chunk_id {
            b"fmt " => {
                if chunk_len < 16 {
                    return Err(Error::InvalidFormat("WAV fmt chunk is too short".into()));
                }
                format = Some(read_u16_le(bytes, offset));
                channels = Some(read_u16_le(bytes, offset + 2));
                sample_rate = Some(read_u32_le(bytes, offset + 4));
                bits_per_sample = Some(read_u16_le(bytes, offset + 14));
            }
            b"data" => {
                data_offset = Some(offset);
                data_bytes = Some(chunk_len);
            }
            _ => {}
        }

        offset = chunk_end + (chunk_len & 1);
    }

    let info = WavInfo {
        sample_rate: sample_rate
            .ok_or_else(|| Error::InvalidFormat("WAV missing fmt sample rate".into()))?,
        channels: channels
            .ok_or_else(|| Error::InvalidFormat("WAV missing channel count".into()))?,
        bits_per_sample: bits_per_sample
            .ok_or_else(|| Error::InvalidFormat("WAV missing bit depth".into()))?,
        format: format.ok_or_else(|| Error::InvalidFormat("WAV missing format tag".into()))?,
        data_offset: data_offset
            .ok_or_else(|| Error::InvalidFormat("WAV missing data chunk".into()))?,
        data_bytes: data_bytes
            .ok_or_else(|| Error::InvalidFormat("WAV missing data chunk".into()))?,
    };
    if !matches!((info.format, info.bits_per_sample), (1, 16) | (3, 32)) {
        return Err(Error::InvalidFormat(format!(
            "unsupported WAV format {}; expected PCM16 or float32",
            info.format
        )));
    }
    Ok(info)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn wav_header(
        sample_rate: u32,
        channels: u16,
        bits: u16,
        format: u16,
        data_bytes: u32,
    ) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16_u32.to_le_bytes());
        out.extend_from_slice(&format.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        out.resize(out.len() + data_bytes as usize, 0);
        out
    }

    fn write_temp_wav(bytes: &[u8]) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "local-ai-gemma4-audio-{}-{nanos}.wav",
            std::process::id()
        ));
        fs::write(&path, bytes).expect("write temp wav");
        path
    }

    #[test]
    fn parses_pcm16_wav_info() {
        let wav = wav_header(16_000, 2, 16, 1, 64_000);
        let info = parse_wav_info(&wav).expect("wav info");
        assert_eq!(info.sample_rate, 16_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.data_offset, 44);
        assert_eq!(info.data_bytes, 64_000);
    }

    #[test]
    fn extracts_log_mel_at_gemma4_cadence() {
        let samples = vec![0.0; 16_000];
        let features = extract_log_mel_features(&samples);
        assert_eq!(features.frames.len(), 99);
        assert_eq!(features.frames[0].len(), FEATURE_SIZE);
        assert_eq!(features.mask.iter().filter(|&&valid| valid).count(), 99);
        assert!(features.frames[0].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn wav_sample_count_reports_resampled_16khz_length() {
        let wav = wav_header(48_000, 2, 16, 1, 48_000 * 2 * 2);
        let path = write_temp_wav(&wav);
        let count = wav_sample_count_16khz(&path).expect("sample count");
        fs::remove_file(path).expect("remove temp wav");
        assert_eq!(count, 16_000);
    }

    #[test]
    fn loads_wav_resampled_to_16khz() {
        let wav = wav_header(8_000, 1, 16, 1, 800 * 2);
        let path = write_temp_wav(&wav);
        let samples = load_wav_mono_f32_resampled_16khz(&path).expect("load wav");
        fs::remove_file(path).expect("remove temp wav");
        assert_eq!(samples.len(), 1_600);
        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn pcm_audio_sample_count_reports_resampled_16khz_length() {
        let audio = PcmAudio {
            samples: vec![0.0; 48_000 * 2],
            sample_rate: 48_000,
            channels: 2,
        };
        let count = pcm_sample_count_16khz(&audio).expect("PCM sample count");
        assert_eq!(count, 16_000);
    }
}
