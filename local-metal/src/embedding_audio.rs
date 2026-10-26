//! Kernels of the `EmbeddingGemma` 2 audio encoder (`shaders/embedding_audio.metal`).
//!
//! The `gemma4_audio` Conformer runs packed blocks of clips: every operation
//! takes `[tokens, width]` F32 rows, and the sequence-aware ones (local
//! attention, the causal depthwise convolution) read each token's sequence
//! start, so several clips share one pass without padding. Matrices are the
//! checkpoint's BF16 values widened exactly; the small convolution and norm
//! weights are F32. Plain RMS norms reuse
//! [`EmbeddingKernels`](crate::embedding::EmbeddingKernels).

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _, MTLSize,
};

use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::embedding::{Bf16Rows, GEMM_COLUMN_MULTIPLE};
use crate::shaders::ShaderLibrary;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>;

/// Head dimension [`AudioKernels::local_attention`] supports.
pub const AUDIO_HEAD_DIM: u32 = 128;
const GEMM_TILE: usize = 32;

/// A region of a buffer: the buffer and a byte offset into it.
pub type Region<'a> = (&'a MetalBuffer, usize);

/// One `Gemma4ClippableLinear` (optionally fused with its activation or
/// residual): `y = clamp(W clamp(x, in) + bias, out)`, then `SiLU` if `silu`,
/// then added to the existing output if `accumulate`.
#[derive(Clone, Copy)]
pub struct ClippedLinear<'a> {
    pub matrix: Bf16Rows<'a>,
    /// `[input_min, input_max, output_min, output_max]`; infinities for none.
    pub clip: [f32; 4],
    /// F32 bias of `matrix.rows` values.
    pub bias: Option<Region<'a>>,
    pub silu: bool,
    pub accumulate: bool,
}

/// One packed local-attention call.
///
/// `q`/`k`/`v`/`out` are `[tokens, heads, 128]`; `first` holds each token's sequence start (`u32`); `q_scale` 128
/// F32 per-dimension query scales; `rel` `[max_distance + 1, heads, 128]` F32
/// relative-position keys indexed by distance.
#[derive(Clone, Copy)]
pub struct LocalAttention<'a> {
    pub tokens: u32,
    pub heads: u32,
    pub max_distance: u32,
    pub k_scale: f32,
    pub logit_cap: f32,
    pub first: &'a MetalBuffer,
    pub q_scale: Region<'a>,
    pub rel: Region<'a>,
}

pub struct AudioKernels {
    conv2d: Pipeline,
    layer_norm_relu: Pipeline,
    rms_add: Pipeline,
    rms_silu: Pipeline,
    glu_dwconv: Pipeline,
    attention: Pipeline,
    gemm: Pipeline,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(message.to_owned())
}

fn need(region: Region<'_>, bytes: Option<usize>) -> crate::Result<()> {
    match bytes.and_then(|bytes| bytes.checked_add(region.1)) {
        Some(end) if end <= region.0.length() => Ok(()),
        _ => Err(arg("audio kernel buffer too short")),
    }
}

fn f32s(count: u64) -> Option<usize> {
    usize::try_from(count.checked_mul(4)?).ok()
}

fn same(a: &MetalBuffer, b: &MetalBuffer) -> bool {
    std::ptr::eq(a.raw(), b.raw())
}

#[expect(unsafe_code)]
fn set<T: Copy>(encoder: &Encoder, value: &T, index: usize) {
    // SAFETY: `value` is a live plain value; Metal copies the bytes.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            index,
        );
    }
}

#[expect(unsafe_code)]
fn bind(encoder: &Encoder, region: Region<'_>, index: usize) {
    // SAFETY: the buffer is retained by the caller until GPU completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(region.0.raw()), region.1, index) };
}

fn grid(encoder: &Encoder, groups: (usize, usize), threads: usize) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups.0,
            height: groups.1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

impl AudioKernels {
    /// # Errors
    ///
    /// Fails when a kernel is missing or cannot run 256-thread threadgroups.
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str| -> crate::Result<Pipeline> {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < 256
            {
                return Err(arg("audio pipeline cannot run 256-thread threadgroups"));
            }
            Ok(pipeline)
        };
        Ok(Self {
            conv2d: pipeline("ea_conv2d_s2")?,
            layer_norm_relu: pipeline("ea_layer_norm_relu")?,
            rms_add: pipeline("ea_rms_add")?,
            rms_silu: pipeline("ea_rms_silu")?,
            glu_dwconv: pipeline("ea_glu_dwconv")?,
            attention: pipeline("ea_local_attention")?,
            gemm: pipeline("ea_bf16_gemm_clip")?,
        })
    }

    /// 3x3 stride-2 padding-1 convolution (no bias) of a channel-last image
    /// `[in_shape[0]][in_shape[1]][in_shape[2]]` with F32 weights
    /// `[out_c][in_c][3][3]`, writing the first `out_rows` time steps of the
    /// `[out_t][out_f][out_c]` result (`out_f = ceil(in_f / 2)`).
    #[expect(clippy::too_many_arguments)]
    pub fn conv2d_s2(
        &self,
        batch: &mut CommandBatch,
        input: Region<'_>,
        weight: Region<'_>,
        output: Region<'_>,
        in_shape: [u32; 3],
        out_rows: u32,
        out_c: u32,
    ) -> crate::Result<()> {
        let [in_t, in_f, in_c] = in_shape;
        let out_f = in_f.div_ceil(2);
        if in_t == 0 || in_f == 0 || in_c == 0 || out_c == 0 || out_rows == 0 {
            return Err(arg("invalid audio convolution shape"));
        }
        if out_rows > in_t.div_ceil(2) {
            return Err(arg("audio convolution writes past its output"));
        }
        let count = u64::from(out_rows) * u64::from(out_f) * u64::from(out_c);
        need(
            input,
            f32s(u64::from(in_t) * u64::from(in_f) * u64::from(in_c)),
        )?;
        need(weight, f32s(u64::from(out_c) * u64::from(in_c) * 9))?;
        need(output, f32s(count))?;
        if same(input.0, output.0) {
            return Err(arg("audio convolution output aliases its input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.conv2d);
        bind(encoder, input, 0);
        bind(encoder, weight, 1);
        bind(encoder, output, 2);
        set(encoder, &[in_t, in_f, in_c, 0_u32], 3);
        set(encoder, &[out_rows, out_f, out_c, 0_u32], 4);
        grid(encoder, ((count as usize).div_ceil(256), 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// In-place `LayerNorm` (weight, no bias) over `dimension` channels and
    /// `ReLU` for `rows` rows; rows whose time step (`row / rows_per_t`) is
    /// at or past `valid_t` are zeroed instead.
    #[expect(clippy::too_many_arguments)]
    pub fn layer_norm_relu(
        &self,
        batch: &mut CommandBatch,
        x: Region<'_>,
        weight: Region<'_>,
        rows: u32,
        dimension: u32,
        rows_per_t: u32,
        valid_t: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || rows_per_t == 0 || epsilon.is_nan() || epsilon <= 0.0 {
            return Err(arg("invalid audio LayerNorm parameters"));
        }
        need(x, f32s(u64::from(rows) * u64::from(dimension)))?;
        need(weight, f32s(u64::from(dimension)))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.layer_norm_relu);
        bind(encoder, x, 0);
        bind(encoder, weight, 1);
        set(encoder, &dimension, 2);
        set(encoder, &rows_per_t, 3);
        set(encoder, &valid_t, 4);
        set(encoder, &epsilon, 5);
        grid(encoder, (rows as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// `hidden += rms(branch) * weight * scale` over `rows` rows.
    #[expect(clippy::too_many_arguments)]
    pub fn rms_add(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weight: Region<'_>,
        rows: u32,
        dimension: u32,
        scale: f32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || !scale.is_finite() || epsilon.is_nan() || epsilon <= 0.0 {
            return Err(arg("invalid audio residual parameters"));
        }
        let bytes = f32s(u64::from(rows) * u64::from(dimension));
        need((hidden, 0), bytes)?;
        need((branch, 0), bytes)?;
        need(weight, f32s(u64::from(dimension)))?;
        if same(hidden, branch) || same(hidden, weight.0) {
            return Err(arg("audio residual output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms_add);
        bind(encoder, (hidden, 0), 0);
        bind(encoder, (branch, 0), 1);
        bind(encoder, weight, 2);
        set(encoder, &dimension, 3);
        set(encoder, &scale, 4);
        set(encoder, &epsilon, 5);
        grid(encoder, (rows as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// `x = silu(rms(x) * weight)` in place over `rows` rows.
    pub fn rms_silu(
        &self,
        batch: &mut CommandBatch,
        x: &MetalBuffer,
        weight: Region<'_>,
        rows: u32,
        dimension: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || epsilon.is_nan() || epsilon <= 0.0 {
            return Err(arg("invalid audio RMSNorm-SiLU parameters"));
        }
        need((x, 0), f32s(u64::from(rows) * u64::from(dimension)))?;
        need(weight, f32s(u64::from(dimension)))?;
        if same(x, weight.0) {
            return Err(arg("audio RMSNorm-SiLU output aliases its weight"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms_silu);
        bind(encoder, (x, 0), 0);
        bind(encoder, weight, 1);
        set(encoder, &dimension, 2);
        set(encoder, &epsilon, 3);
        grid(encoder, (rows as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// GLU over the `[tokens, 2 * channels]` input fused with the causal
    /// depthwise convolution (F32 weights `[channels][kernel_size]`, left
    /// padding `kernel_size - 1`, sequence starts in `first`), writing
    /// `[tokens, channels]`.
    #[expect(clippy::too_many_arguments)]
    pub fn glu_dwconv(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weight: Region<'_>,
        output: &MetalBuffer,
        first: &MetalBuffer,
        tokens: u32,
        channels: u32,
        kernel_size: u32,
    ) -> crate::Result<()> {
        if tokens == 0 || channels == 0 || kernel_size == 0 {
            return Err(arg("invalid audio depthwise convolution shape"));
        }
        let count = u64::from(tokens) * u64::from(channels);
        need((input, 0), f32s(count * 2))?;
        need((output, 0), f32s(count))?;
        need((first, 0), f32s(u64::from(tokens)))?;
        need(weight, f32s(u64::from(channels) * u64::from(kernel_size)))?;
        if [input, first, weight.0].iter().any(|b| same(b, output)) {
            return Err(arg("audio depthwise convolution output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.glu_dwconv);
        bind(encoder, (input, 0), 0);
        bind(encoder, weight, 1);
        bind(encoder, (output, 0), 2);
        bind(encoder, (first, 0), 3);
        set(encoder, &channels, 4);
        set(encoder, &kernel_size, 5);
        set(encoder, &tokens, 6);
        grid(encoder, ((count as usize).div_ceil(256), 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// Chunked local self-attention with relative position logits; see
    /// [`LocalAttention`].
    pub fn local_attention(
        &self,
        batch: &mut CommandBatch,
        shape: LocalAttention<'_>,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        if shape.tokens == 0
            || shape.heads == 0
            || shape.heads > 32
            || !shape.k_scale.is_finite()
            || !shape.logit_cap.is_finite()
            || shape.logit_cap <= 0.0
        {
            return Err(arg("invalid audio attention parameters"));
        }
        let width = u64::from(shape.heads) * u64::from(AUDIO_HEAD_DIM);
        let bytes = f32s(u64::from(shape.tokens) * width);
        for buffer in [q, k, v, out] {
            need((buffer, 0), bytes)?;
        }
        need((shape.first, 0), f32s(u64::from(shape.tokens)))?;
        need(shape.q_scale, f32s(u64::from(AUDIO_HEAD_DIM)))?;
        need(shape.rel, f32s((u64::from(shape.max_distance) + 1) * width))?;
        if [q, k, v, shape.first, shape.q_scale.0, shape.rel.0]
            .iter()
            .any(|b| same(b, out))
        {
            return Err(arg("audio attention output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention);
        bind(encoder, (q, 0), 0);
        bind(encoder, (k, 0), 1);
        bind(encoder, (v, 0), 2);
        bind(encoder, (out, 0), 3);
        bind(encoder, (shape.first, 0), 4);
        bind(encoder, shape.q_scale, 5);
        bind(encoder, shape.rel, 6);
        set(encoder, &shape.heads, 7);
        set(encoder, &shape.max_distance, 8);
        set(encoder, &shape.k_scale, 9);
        set(encoder, &shape.logit_cap, 10);
        grid(
            encoder,
            (shape.tokens as usize, 1),
            32 * shape.heads as usize,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// A clipped linear over `tokens` input rows; see [`ClippedLinear`].
    pub fn linear(
        &self,
        batch: &mut CommandBatch,
        layer: ClippedLinear<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let matrix = layer.matrix;
        if tokens == 0
            || matrix.rows == 0
            || matrix.columns == 0
            || !matrix.columns.is_multiple_of(GEMM_COLUMN_MULTIPLE)
            || !matrix.offset.is_multiple_of(16)
            || layer.clip.iter().any(|v| v.is_nan())
            || layer.clip[0] > layer.clip[1]
            || layer.clip[2] > layer.clip[3]
        {
            return Err(arg("invalid clipped linear"));
        }
        let rows = u64::from(matrix.rows);
        let columns = u64::from(matrix.columns);
        need(
            (matrix.weights, matrix.offset),
            usize::try_from(rows * columns * 2).ok(),
        )?;
        need((input, 0), f32s(u64::from(tokens) * columns))?;
        need((output, 0), f32s(u64::from(tokens) * rows))?;
        if let Some(bias) = layer.bias {
            need(bias, f32s(rows))?;
            if same(bias.0, output) {
                return Err(arg("clipped linear output aliases its bias"));
            }
        }
        if same(input, output) || same(matrix.weights, output) {
            return Err(arg("clipped linear output aliases an input"));
        }
        let flags = u32::from(layer.bias.is_some())
            | (u32::from(layer.silu) << 1)
            | (u32::from(layer.accumulate) << 2);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gemm);
        bind(encoder, (matrix.weights, matrix.offset), 0);
        bind(encoder, (input, 0), 1);
        bind(encoder, (output, 0), 2);
        set(encoder, &matrix.rows, 3);
        set(encoder, &matrix.columns, 4);
        set(encoder, &tokens, 5);
        set(encoder, &layer.clip, 6);
        bind(encoder, layer.bias.unwrap_or((input, 0)), 7);
        set(encoder, &flags, 8);
        grid(
            encoder,
            (
                (matrix.rows as usize).div_ceil(GEMM_TILE),
                (tokens as usize).div_ceil(GEMM_TILE),
            ),
            128,
        );
        batch.record_dispatch();
        Ok(())
    }
}
