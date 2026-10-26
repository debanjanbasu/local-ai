//! The Whisper large-v3 encoder and decoder on Metal.
//!
//! The encoder runs one 30-second window (3,000 mel frames → two kernel-3
//! convolutions with GELU → 1,500 frames + the sinusoidal table → 32
//! pre-`LayerNorm` blocks → final `LayerNorm`), then projects the output into
//! every decoder layer's cross-attention keys and values once. The decoder
//! keeps a self-attention KV cache of up to 448 positions; the logits are
//! the final hidden state times the token table (tied output embedding).
//!
//! Work is split into command buffers by an estimate of GPU time: while
//! Bonsai is generating ([`Pace::busy`]), about 2 ms per submission, each
//! followed by [`Pace::pause`] (which yields to a decode step and checks
//! for cancellation); otherwise about 50 ms, pipelined.

use half::f16;
use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::shaders::ShaderLibrary;
use local_metal::whisper::{DecoderKeys, Epilogue, F16Rows, Region, WhisperKernels};

use super::artifact::{Artifact, Dtype};
use super::decode::Forward;
use super::mel::WINDOW_FRAMES;
use super::{CTX_AUDIO, CTX_TEXT, FFN, HEADS, LAYERS, N_MELS, Pace, STATE, VOCAB};
use crate::Error;

const EPSILON: f32 = 1e-5;
/// Estimated GPU throughput for splitting work into submissions.
const FLOPS_PER_US: f64 = 2.0e6;
const BYTES_PER_US: f64 = 2.0e5;
/// GPU time per submission while Bonsai is decoding, and otherwise.
const FINE_US: f64 = 2_000.0;
const COARSE_US: f64 = 50_000.0;

struct Linear {
    weight: usize,
    bias: Option<usize>,
    rows: usize,
    cols: usize,
}

struct Norm {
    weight: usize,
    bias: usize,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm: Norm,
}

struct Block {
    attention: Attention,
    cross: Option<Attention>,
    fc1: Linear,
    fc2: Linear,
    norm: Norm,
}

struct Workspace {
    mel: MetalBuffer,
    columns: MetalBuffer,
    wide: MetalBuffer,
    x: MetalBuffer,
    normed: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attention: MetalBuffer,
    output: MetalBuffer,
    dx: MetalBuffer,
    dnormed: MetalBuffer,
    dq: MetalBuffer,
    dattention: MetalBuffer,
    dwide: MetalBuffer,
}

/// Command buffers sized by estimated GPU time, with pacing between them.
struct Submit<'c, 'p, 'q> {
    context: &'c MetalContext,
    batch: CommandBatch,
    pending_us: f64,
    pace: &'p mut Pace<'q>,
}

impl<'c, 'p, 'q> Submit<'c, 'p, 'q> {
    fn new(context: &'c MetalContext, pace: &'p mut Pace<'q>) -> crate::Result<Self> {
        Ok(Self {
            context,
            batch: CommandBatch::new(context)?,
            pending_us: 0.0,
            pace,
        })
    }

    fn fine(&self) -> bool {
        (self.pace.busy)()
    }

    fn spent(&mut self, us: f64) -> crate::Result<()> {
        self.pending_us += us;
        let fine = self.fine();
        if fine && self.pending_us >= FINE_US {
            self.batch.commit_and_renew(self.context)?;
            self.pending_us = 0.0;
            (self.pace.pause)()?;
        } else if !fine && self.pending_us >= COARSE_US {
            self.batch.submit_and_renew(self.context)?;
            self.pending_us = 0.0;
            (self.pace.pause)()?;
        }
        Ok(())
    }

    fn finish(self) -> crate::Result<()> {
        self.batch.commit_and_wait()?;
        Ok(())
    }
}

pub(super) struct WhisperGpu {
    context: MetalContext,
    kernels: WhisperKernels,
    weights: MetalBuffer,
    conv1: Linear,
    conv2: Linear,
    encoder_positions: usize,
    encoder: Vec<Block>,
    encoder_norm: Norm,
    token_table: usize,
    decoder_positions: usize,
    decoder: Vec<Block>,
    decoder_norm: Norm,
    ws: Workspace,
    cross_k: MetalBuffer,
    cross_v: MetalBuffer,
    self_k: MetalBuffer,
    self_v: MetalBuffer,
    logits: MetalBuffer,
    /// Cross-attention queries of the alignment layers (word timestamps).
    captured: MetalBuffer,
    /// Decoder positions in the KV cache.
    length: usize,
    /// Whether a window is encoded.
    encoded: bool,
}

// SAFETY: Metal devices, queues, pipelines and buffers may be transferred
// between threads. The model owns every object exclusively and is `!Sync`
// (its Metal objects are), so no encoder or workspace is used concurrently;
// sharing goes through a mutex.
#[expect(unsafe_code)]
#[expect(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for WhisperGpu {}

fn linear(
    artifact: &Artifact,
    name: &str,
    rows: usize,
    cols: usize,
    bias: bool,
) -> crate::Result<Linear> {
    Ok(Linear {
        weight: artifact.offset(&format!("{name}.weight"), Dtype::F16, rows, cols)?,
        bias: if bias {
            Some(artifact.offset(&format!("{name}.bias"), Dtype::F32, 1, rows)?)
        } else {
            None
        },
        rows,
        cols,
    })
}

fn norm(artifact: &Artifact, name: &str) -> crate::Result<Norm> {
    Ok(Norm {
        weight: artifact.offset(&format!("{name}.weight"), Dtype::F32, 1, STATE)?,
        bias: artifact.offset(&format!("{name}.bias"), Dtype::F32, 1, STATE)?,
    })
}

fn attention(artifact: &Artifact, prefix: &str, kind: &str) -> crate::Result<Attention> {
    let p = |proj: &str| format!("{prefix}.{kind}.{proj}");
    Ok(Attention {
        q: linear(artifact, &p("q_proj"), STATE, STATE, true)?,
        k: linear(artifact, &p("k_proj"), STATE, STATE, false)?,
        v: linear(artifact, &p("v_proj"), STATE, STATE, true)?,
        out: linear(artifact, &p("out_proj"), STATE, STATE, true)?,
        norm: norm(artifact, &format!("{prefix}.{kind}_layer_norm"))?,
    })
}

fn block(artifact: &Artifact, prefix: &str, cross: bool) -> crate::Result<Block> {
    Ok(Block {
        attention: attention(artifact, prefix, "self_attn")?,
        cross: if cross {
            Some(attention(artifact, prefix, "encoder_attn")?)
        } else {
            None
        },
        fc1: linear(artifact, &format!("{prefix}.fc1"), FFN, STATE, true)?,
        fc2: linear(artifact, &format!("{prefix}.fc2"), STATE, FFN, true)?,
        norm: norm(artifact, &format!("{prefix}.final_layer_norm"))?,
    })
}

const fn floats(count: usize) -> usize {
    count * 4
}

/// `region` advanced by `start` rows of `width` floats.
const fn shift(region: Region<'_>, start: usize, width: usize) -> Region<'_> {
    Region::at(region.buffer, region.offset + floats(start * width))
}

impl WhisperGpu {
    pub(super) fn new(artifact: &Artifact) -> crate::Result<Self> {
        let context = MetalContext::new()?;
        let shaders = ShaderLibrary::new(context.device())?;
        let kernels = WhisperKernels::new(&context, &shaders)?;
        let weights = MetalBuffer::from_slice(context.device(), artifact.payload())?;
        let encoder = (0..LAYERS)
            .map(|i| block(artifact, &format!("encoder.layers.{i}"), false))
            .collect::<crate::Result<Vec<_>>>()?;
        let decoder = (0..LAYERS)
            .map(|i| block(artifact, &format!("decoder.layers.{i}"), true))
            .collect::<crate::Result<Vec<_>>>()?;
        let device = context.device();
        let buffer = |bytes: usize| MetalBuffer::empty(device, bytes);
        let frames = WINDOW_FRAMES;
        let ws = Workspace {
            mel: buffer(floats(frames * N_MELS))?,
            columns: buffer(floats(CTX_AUDIO * STATE * 3))?,
            wide: buffer(floats((frames * STATE).max(CTX_AUDIO * FFN)))?,
            x: buffer(floats(CTX_AUDIO * STATE))?,
            normed: buffer(floats(CTX_AUDIO * STATE))?,
            q: buffer(floats(CTX_AUDIO * STATE))?,
            k: buffer(floats(CTX_AUDIO * STATE))?,
            v: buffer(floats(CTX_AUDIO * STATE))?,
            attention: buffer(floats(CTX_AUDIO * STATE))?,
            output: buffer(floats(CTX_AUDIO * STATE))?,
            dx: buffer(floats(CTX_TEXT * STATE))?,
            dnormed: buffer(floats(CTX_TEXT * STATE))?,
            dq: buffer(floats(CTX_TEXT * STATE))?,
            dattention: buffer(floats(CTX_TEXT * STATE))?,
            dwide: buffer(floats(CTX_TEXT * FFN))?,
        };
        Ok(Self {
            conv1: Linear {
                weight: artifact.offset("encoder.conv1.weight", Dtype::F16, STATE, N_MELS * 3)?,
                bias: Some(artifact.offset("encoder.conv1.bias", Dtype::F32, 1, STATE)?),
                rows: STATE,
                cols: N_MELS * 3,
            },
            conv2: Linear {
                weight: artifact.offset("encoder.conv2.weight", Dtype::F16, STATE, STATE * 3)?,
                bias: Some(artifact.offset("encoder.conv2.bias", Dtype::F32, 1, STATE)?),
                rows: STATE,
                cols: STATE * 3,
            },
            encoder_positions: artifact.offset(
                "encoder.embed_positions.weight",
                Dtype::F32,
                CTX_AUDIO,
                STATE,
            )?,
            encoder,
            encoder_norm: norm(artifact, "encoder.layer_norm")?,
            token_table: artifact.offset(
                "decoder.embed_tokens.weight",
                Dtype::F16,
                VOCAB,
                STATE,
            )?,
            decoder_positions: artifact.offset(
                "decoder.embed_positions.weight",
                Dtype::F16,
                CTX_TEXT,
                STATE,
            )?,
            decoder,
            decoder_norm: norm(artifact, "decoder.layer_norm")?,
            cross_k: buffer(floats(LAYERS * CTX_AUDIO * STATE))?,
            cross_v: buffer(floats(LAYERS * CTX_AUDIO * STATE))?,
            self_k: buffer(floats(LAYERS * CTX_TEXT * STATE))?,
            self_v: buffer(floats(LAYERS * CTX_TEXT * STATE))?,
            logits: buffer(floats(2 * VOCAB))?,
            captured: buffer(4)?,
            ws,
            context,
            kernels,
            weights,
            length: 0,
            encoded: false,
        })
    }

    const fn weight_region(&self, offset: usize) -> Region<'_> {
        Region::at(&self.weights, offset)
    }

    /// `output = epilogue(W · input)` for `tokens` rows, split by estimated
    /// cost when Bonsai is decoding.
    #[expect(clippy::too_many_arguments)]
    fn linear(
        &self,
        sub: &mut Submit<'_, '_, '_>,
        layer: &Linear,
        input: Region<'_>,
        output: Region<'_>,
        tokens: usize,
        gelu: bool,
        residual: Option<Region<'_>>,
    ) -> crate::Result<()> {
        let matrix = F16Rows {
            weights: &self.weights,
            offset: layer.weight,
            rows: layer.rows as u32,
            columns: layer.cols as u32,
        };
        let bias = layer.bias.map(|offset| self.weight_region(offset));
        if tokens == 1 {
            self.kernels.gemv(
                &mut sub.batch,
                matrix,
                input,
                output,
                &Epilogue {
                    bias,
                    gelu,
                    residual,
                },
            )?;
            return sub.spent((layer.rows * layer.cols * 2) as f64 / BYTES_PER_US);
        }
        let flops = (2 * layer.rows * layer.cols) as f64;
        let chunk = if sub.fine() {
            (((FINE_US * FLOPS_PER_US / flops) as usize) / 32 * 32).max(32)
        } else {
            tokens
        };
        let mut start = 0;
        while start < tokens {
            let n = chunk.min(tokens - start);
            let shifted = residual.map(|r| shift(r, start, layer.rows));
            self.kernels.gemm(
                &mut sub.batch,
                matrix,
                shift(input, start, layer.cols),
                shift(output, start, layer.rows),
                n as u32,
                &Epilogue {
                    bias,
                    gelu,
                    residual: shifted,
                },
            )?;
            sub.spent(n as f64 * flops / FLOPS_PER_US)?;
            start += n;
        }
        Ok(())
    }

    fn layernorm(
        &self,
        sub: &mut Submit<'_, '_, '_>,
        input: Region<'_>,
        norm: &Norm,
        output: Region<'_>,
        rows: usize,
    ) -> crate::Result<()> {
        self.kernels.layernorm(
            &mut sub.batch,
            input,
            self.weight_region(norm.weight),
            self.weight_region(norm.bias),
            output,
            rows as u32,
            STATE as u32,
            EPSILON,
        )?;
        sub.spent(floats(rows * STATE * 2) as f64 / BYTES_PER_US)
    }

    /// The final encoder states of the last window, `[1500][1280]`.
    #[expect(unsafe_code)]
    pub(super) fn encoder_output(&self) -> Vec<f32> {
        // SAFETY: every encoder pass waits for its GPU work before returning;
        // the borrow ends with this copy.
        unsafe { self.ws.output.as_slice::<f32>()[..CTX_AUDIO * STATE].to_vec() }
    }

    #[expect(clippy::too_many_lines)]
    fn encode_window(&mut self, window: &[f32], pace: &mut Pace<'_>) -> crate::Result<()> {
        if window.len() != WINDOW_FRAMES * N_MELS {
            return Err(Error::InvalidArgument(
                "mel window has the wrong size".into(),
            ));
        }
        self.encoded = false;
        self.ws.mel.copy_from_bytes(bytemuck::cast_slice(window), 0);
        let mut sub = Submit::new(&self.context, pace)?;
        let ws = &self.ws;
        let whole = Region::whole;
        let frames = WINDOW_FRAMES as u32;
        self.kernels.im2col3(
            &mut sub.batch,
            whole(&ws.mel),
            whole(&ws.columns),
            N_MELS as u32,
            frames,
            frames,
            1,
        )?;
        self.linear(
            &mut sub,
            &self.conv1,
            whole(&ws.columns),
            whole(&ws.wide),
            WINDOW_FRAMES,
            true,
            None,
        )?;
        self.kernels.im2col3(
            &mut sub.batch,
            whole(&ws.wide),
            whole(&ws.columns),
            STATE as u32,
            frames,
            CTX_AUDIO as u32,
            2,
        )?;
        self.linear(
            &mut sub,
            &self.conv2,
            whole(&ws.columns),
            whole(&ws.x),
            CTX_AUDIO,
            true,
            Some(self.weight_region(self.encoder_positions)),
        )?;
        let attention_flops = (CTX_AUDIO * CTX_AUDIO * STATE * 4) as f64;
        for layer in &self.encoder {
            let a = &layer.attention;
            self.layernorm(
                &mut sub,
                whole(&ws.x),
                &a.norm,
                whole(&ws.normed),
                CTX_AUDIO,
            )?;
            self.linear(
                &mut sub,
                &a.q,
                whole(&ws.normed),
                whole(&ws.q),
                CTX_AUDIO,
                false,
                None,
            )?;
            self.linear(
                &mut sub,
                &a.k,
                whole(&ws.normed),
                whole(&ws.k),
                CTX_AUDIO,
                false,
                None,
            )?;
            self.linear(
                &mut sub,
                &a.v,
                whole(&ws.normed),
                whole(&ws.v),
                CTX_AUDIO,
                false,
                None,
            )?;
            let chunk = if sub.fine() {
                let per_query = attention_flops / CTX_AUDIO as f64;
                (((FINE_US * FLOPS_PER_US / per_query) as usize) / 128 * 128).max(128)
            } else {
                CTX_AUDIO
            };
            let mut start = 0;
            while start < CTX_AUDIO {
                let n = chunk.min(CTX_AUDIO - start);
                self.kernels.encoder_attention(
                    &mut sub.batch,
                    whole(&ws.q),
                    whole(&ws.k),
                    whole(&ws.v),
                    whole(&ws.attention),
                    CTX_AUDIO as u32,
                    HEADS as u32,
                    (start as u32, n as u32),
                )?;
                sub.spent(attention_flops * n as f64 / CTX_AUDIO as f64 / FLOPS_PER_US)?;
                start += n;
            }
            self.linear(
                &mut sub,
                &a.out,
                whole(&ws.attention),
                whole(&ws.x),
                CTX_AUDIO,
                false,
                Some(whole(&ws.x)),
            )?;
            self.layernorm(
                &mut sub,
                whole(&ws.x),
                &layer.norm,
                whole(&ws.normed),
                CTX_AUDIO,
            )?;
            self.linear(
                &mut sub,
                &layer.fc1,
                whole(&ws.normed),
                whole(&ws.wide),
                CTX_AUDIO,
                true,
                None,
            )?;
            self.linear(
                &mut sub,
                &layer.fc2,
                whole(&ws.wide),
                whole(&ws.x),
                CTX_AUDIO,
                false,
                Some(whole(&ws.x)),
            )?;
        }
        self.layernorm(
            &mut sub,
            whole(&ws.x),
            &self.encoder_norm,
            whole(&ws.output),
            CTX_AUDIO,
        )?;
        let plane = floats(CTX_AUDIO * STATE);
        for (l, layer) in self.decoder.iter().enumerate() {
            let cross = layer.cross.as_ref().ok_or_else(|| {
                Error::InvalidFormat("decoder layer without cross-attention".into())
            })?;
            self.linear(
                &mut sub,
                &cross.k,
                whole(&ws.output),
                Region::at(&self.cross_k, l * plane),
                CTX_AUDIO,
                false,
                None,
            )?;
            self.linear(
                &mut sub,
                &cross.v,
                whole(&ws.output),
                Region::at(&self.cross_v, l * plane),
                CTX_AUDIO,
                false,
                None,
            )?;
        }
        sub.finish()?;
        self.length = 0;
        self.encoded = true;
        Ok(())
    }

    /// Token plus position embeddings of `tokens` at positions `start..`.
    fn embed(&self, artifact_payload: &[u8], tokens: &[u32], start: usize) -> Vec<f32> {
        let row = |offset: usize, index: usize| {
            artifact_payload[offset + index * STATE * 2..][..STATE * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_le_bytes(*b).to_f32())
                .collect::<Vec<f32>>()
        };
        let mut out = Vec::with_capacity(tokens.len() * STATE);
        for (i, token) in tokens.iter().enumerate() {
            let e = row(self.token_table, *token as usize);
            let p = row(self.decoder_positions, start + i);
            out.extend(e.iter().zip(&p).map(|(a, b)| a + b));
        }
        out
    }

    /// Run `tokens` at positions `start..` and return the logits at `rows`
    /// (indices into `tokens`).
    #[expect(unsafe_code, clippy::too_many_lines)]
    fn decode_tokens(
        &mut self,
        payload: &[u8],
        tokens: &[u32],
        start: usize,
        rows: &[usize],
        capture: &[usize],
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        if !self.encoded {
            return Err(Error::Generation("no audio window is encoded".into()));
        }
        let n = tokens.len();
        if n == 0 || start + n > CTX_TEXT || rows.iter().any(|r| *r >= n) {
            return Err(Error::ContextOverflow(format!(
                "{} decoder positions exceed the {CTX_TEXT}-token text context",
                start + n
            )));
        }
        if let Some(bad) = tokens.iter().find(|t| **t as usize >= VOCAB) {
            return Err(Error::InvalidArgument(format!(
                "token {bad} is out of range"
            )));
        }
        let x0 = self.embed(payload, tokens, start);
        self.ws.dx.copy_from_bytes(bytemuck::cast_slice(&x0), 0);
        if self.logits.length() < floats(rows.len() * VOCAB) {
            self.logits = MetalBuffer::empty(self.context.device(), floats(rows.len() * VOCAB))?;
        }
        let capture_plane = floats(n * STATE);
        if !capture.is_empty() && self.captured.length() < capture.len() * capture_plane {
            self.captured =
                MetalBuffer::empty(self.context.device(), capture.len() * capture_plane)?;
        }
        let mut sub = Submit::new(&self.context, pace)?;
        let ws = &self.ws;
        let whole = Region::whole;
        let cache_plane = floats(CTX_TEXT * STATE);
        let cross_plane = floats(CTX_AUDIO * STATE);
        for (l, layer) in self.decoder.iter().enumerate() {
            let a = &layer.attention;
            self.layernorm(&mut sub, whole(&ws.dx), &a.norm, whole(&ws.dnormed), n)?;
            self.linear(
                &mut sub,
                &a.q,
                whole(&ws.dnormed),
                whole(&ws.dq),
                n,
                false,
                None,
            )?;
            let slot = l * cache_plane + floats(start * STATE);
            self.linear(
                &mut sub,
                &a.k,
                whole(&ws.dnormed),
                Region::at(&self.self_k, slot),
                n,
                false,
                None,
            )?;
            self.linear(
                &mut sub,
                &a.v,
                whole(&ws.dnormed),
                Region::at(&self.self_v, slot),
                n,
                false,
                None,
            )?;
            self.kernels.decoder_attention(
                &mut sub.batch,
                whole(&ws.dq),
                Region::at(&self.self_k, l * cache_plane),
                Region::at(&self.self_v, l * cache_plane),
                whole(&ws.dattention),
                n as u32,
                HEADS as u32,
                DecoderKeys::Causal { base: start as u32 },
            )?;
            sub.spent(floats((start + n) * STATE * 2) as f64 / BYTES_PER_US)?;
            self.linear(
                &mut sub,
                &a.out,
                whole(&ws.dattention),
                whole(&ws.dx),
                n,
                false,
                Some(whole(&ws.dx)),
            )?;
            let c = layer.cross.as_ref().ok_or_else(|| {
                Error::InvalidFormat("decoder layer without cross-attention".into())
            })?;
            self.layernorm(&mut sub, whole(&ws.dx), &c.norm, whole(&ws.dnormed), n)?;
            self.linear(
                &mut sub,
                &c.q,
                whole(&ws.dnormed),
                whole(&ws.dq),
                n,
                false,
                None,
            )?;
            if let Some(slot) = capture.iter().position(|layer| *layer == l) {
                // The cross-attention queries of an alignment layer, kept.
                self.linear(
                    &mut sub,
                    &c.q,
                    whole(&ws.dnormed),
                    Region::at(&self.captured, slot * capture_plane),
                    n,
                    false,
                    None,
                )?;
            }
            self.kernels.decoder_attention(
                &mut sub.batch,
                whole(&ws.dq),
                Region::at(&self.cross_k, l * cross_plane),
                Region::at(&self.cross_v, l * cross_plane),
                whole(&ws.dattention),
                n as u32,
                HEADS as u32,
                DecoderKeys::All {
                    keys: CTX_AUDIO as u32,
                },
            )?;
            sub.spent(n as f64 * cross_plane as f64 * 2.0 / BYTES_PER_US)?;
            self.linear(
                &mut sub,
                &c.out,
                whole(&ws.dattention),
                whole(&ws.dx),
                n,
                false,
                Some(whole(&ws.dx)),
            )?;
            self.layernorm(&mut sub, whole(&ws.dx), &layer.norm, whole(&ws.dnormed), n)?;
            self.linear(
                &mut sub,
                &layer.fc1,
                whole(&ws.dnormed),
                whole(&ws.dwide),
                n,
                true,
                None,
            )?;
            self.linear(
                &mut sub,
                &layer.fc2,
                whole(&ws.dwide),
                whole(&ws.dx),
                n,
                false,
                Some(whole(&ws.dx)),
            )?;
        }
        self.layernorm(
            &mut sub,
            whole(&ws.dx),
            &self.decoder_norm,
            whole(&ws.dnormed),
            n,
        )?;
        let table = Linear {
            weight: self.token_table,
            bias: None,
            rows: VOCAB,
            cols: STATE,
        };
        for (j, row) in rows.iter().enumerate() {
            self.linear(
                &mut sub,
                &table,
                Region::at(&ws.dnormed, floats(row * STATE)),
                Region::at(&self.logits, floats(j * VOCAB)),
                1,
                false,
                None,
            )?;
        }
        sub.finish()?;
        self.length = start + n;
        // SAFETY: `sub.finish` waited for the decode, and the borrow ends with
        // the copies below.
        let all = unsafe { self.logits.as_slice::<f32>() };
        Ok((0..rows.len())
            .map(|j| all[j * VOCAB..(j + 1) * VOCAB].to_vec())
            .collect())
    }
}

/// [`WhisperGpu`] with the artifact payload it reads embeddings from.
pub(super) struct Bound<'a> {
    pub gpu: &'a mut WhisperGpu,
    pub payload: &'a [u8],
}

impl Forward for Bound<'_> {
    fn encode(&mut self, window: &[f32], pace: &mut Pace<'_>) -> crate::Result<()> {
        self.gpu.encode_window(window, pace)
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        rows: &[usize],
        pace: &mut Pace<'_>,
    ) -> crate::Result<Vec<Vec<f32>>> {
        self.gpu
            .decode_tokens(self.payload, tokens, 0, rows, &[], pace)
    }

    #[expect(unsafe_code)]
    fn alignment(
        &mut self,
        tokens: &[u32],
        heads: &[[u32; 2]],
        frames: usize,
        first_row: usize,
        rows: usize,
        pace: &mut Pace<'_>,
    ) -> crate::Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut layers: Vec<usize> = heads.iter().map(|h| h[0] as usize).collect();
        layers.sort_unstable();
        layers.dedup();
        if heads
            .iter()
            .any(|h| h[0] as usize >= LAYERS || h[1] as usize >= HEADS)
            || frames > CTX_AUDIO
        {
            return Err(Error::InvalidFormat("alignment head out of range".into()));
        }
        let logit_rows: Vec<usize> = (first_row..first_row + rows).collect();
        let logits = self
            .gpu
            .decode_tokens(self.payload, tokens, 0, &logit_rows, &layers, pace)?;
        let n = tokens.len();
        // SAFETY: `decode_tokens` waited for its GPU work, and nothing is
        // encoded while these views live.
        let q_all = unsafe { self.gpu.captured.as_slice::<f32>() };
        // SAFETY: as above.
        let k_all = unsafe { self.gpu.cross_k.as_slice::<f32>() };
        let mut weights = vec![0.0_f32; heads.len() * n * frames];
        for (h, &[layer, head]) in heads.iter().enumerate() {
            let slot = layers
                .iter()
                .position(|l| *l == layer as usize)
                .unwrap_or(0);
            let q = &q_all[slot * n * STATE..(slot + 1) * n * STATE];
            let k = &k_all[layer as usize * CTX_AUDIO * STATE..][..CTX_AUDIO * STATE];
            let offset = head as usize * 64;
            for t in 0..n {
                let qv = &q[t * STATE + offset..t * STATE + offset + 64];
                for f in 0..frames {
                    let kv = &k[f * STATE + offset..f * STATE + offset + 64];
                    let dot: f32 = qv.iter().zip(kv).map(|(a, b)| a * b).sum();
                    weights[(h * n + t) * frames + f] = dot * 0.125;
                }
            }
        }
        Ok((weights, logits))
    }

    fn step(&mut self, token: u32, pace: &mut Pace<'_>) -> crate::Result<Vec<f32>> {
        let start = self.gpu.length;
        self.gpu
            .decode_tokens(self.payload, &[token], start, &[0], &[], pace)?
            .pop()
            .ok_or_else(|| Error::Generation("no logits".into()))
    }
}
