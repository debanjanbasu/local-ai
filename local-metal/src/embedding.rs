//! Kernels of the `EmbeddingGemma` 2 text encoder (`shaders/embedding.metal`).
//!
//! The encoder runs whole packed blocks of sequences at once: every operation
//! takes `[tokens, width]` F32 rows, and attention reads each token's sequence
//! bounds, so several inputs share one pass without padding. Matrices keep
//! the checkpoint's BF16 values (widened to F32 exactly in the kernel);
//! activations, accumulation and softmax are F32 (the model card rules out
//! F16 activations).

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
use crate::shaders::ShaderLibrary;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Column counts of [`EmbeddingKernels::bf16_gemm`] must be a multiple of this.
pub const GEMM_COLUMN_MULTIPLE: u32 = 32;
const GEMM_TILE: usize = 32;

/// Head dimensions [`EmbeddingKernels::attention`] supports.
pub const ATTENTION_HEAD_DIMS: [u32; 2] = [256, 512];

/// Row-major BF16 matrix at a byte offset inside a weight buffer, which must
/// stay unmodified until GPU completion.
#[derive(Clone, Copy)]
pub struct Bf16Rows<'a> {
    pub weights: &'a MetalBuffer,
    pub offset: usize,
    pub rows: u32,
    pub columns: u32,
}

/// One packed attention call: `q` `[tokens, q_heads, head_dim]`, `k`/`v`
/// `[tokens, kv_heads, head_dim]`, `bounds[t]` = (first, end) token of t's
/// sequence (two `u32`), `window` 0 for full attention.
#[derive(Clone, Copy)]
pub struct AttentionShape {
    pub tokens: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,
}

pub struct EmbeddingKernels {
    rms: Pipeline,
    rms_add: Pipeline,
    rope: Pipeline,
    gelu_mul: Pipeline,
    attention_256: Pipeline,
    attention_512: Pipeline,
    gemm: Pipeline,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(message.to_owned())
}

fn need(buffer: &MetalBuffer, bytes: Option<usize>) -> crate::Result<()> {
    match bytes {
        Some(bytes) if bytes <= buffer.length() => Ok(()),
        _ => Err(arg("embedding kernel buffer too short")),
    }
}

fn f32s(count: u64) -> Option<usize> {
    usize::try_from(count.checked_mul(4)?).ok()
}

fn same(a: &MetalBuffer, b: &MetalBuffer) -> bool {
    std::ptr::eq(a.raw(), b.raw())
}

#[allow(unsafe_code)]
fn set<T: Copy>(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    // SAFETY: `value` is a live plain value; Metal copies the bytes.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            index,
        );
    }
}

#[allow(unsafe_code)]
fn bind(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    buffer: &MetalBuffer,
    offset: usize,
    index: usize,
) {
    // SAFETY: the buffer is retained by the caller until GPU completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), offset, index) };
}

fn grid(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    groups: (usize, usize),
    threads: usize,
) {
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

impl EmbeddingKernels {
    /// # Errors
    ///
    /// Fails when a kernel is missing or cannot run 128-thread threadgroups.
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str| -> crate::Result<Pipeline> {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < 128
            {
                return Err(arg("embedding pipeline cannot run 128-thread threadgroups"));
            }
            Ok(pipeline)
        };
        Ok(Self {
            rms: pipeline("eg_rms")?,
            rms_add: pipeline("eg_rms_add")?,
            rope: pipeline("eg_rope")?,
            gelu_mul: pipeline("eg_gelu_mul")?,
            attention_256: pipeline("eg_attention_256")?,
            attention_512: pipeline("eg_attention_512")?,
            gemm: pipeline("eg_bf16_gemm")?,
        })
    }

    /// `RMSNorm` of `rows` contiguous rows of `dimension`, optionally scaled by
    /// `weight[i % weight_period]` (one shared weight vector), after
    /// multiplying the input by `scale`. `output` may be `input` (in place).
    #[allow(clippy::too_many_arguments)]
    pub fn rms(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weight: Option<(&MetalBuffer, usize)>,
        output: &MetalBuffer,
        rows: u32,
        dimension: u32,
        scale: f32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || !scale.is_finite() || epsilon.is_nan() || epsilon <= 0.0 {
            return Err(arg("invalid embedding RMSNorm parameters"));
        }
        let bytes = f32s(u64::from(rows) * u64::from(dimension));
        need(input, bytes)?;
        need(output, bytes)?;
        let (weight_buffer, weight_offset) = weight.unwrap_or((input, 0));
        if weight.is_some() {
            need(
                weight_buffer,
                weight_offset.checked_add(dimension as usize * 4),
            )?;
            if same(weight_buffer, output) {
                return Err(arg("RMSNorm output aliases its weight"));
            }
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms);
        bind(encoder, input, 0, 0);
        bind(encoder, weight_buffer, weight_offset, 1);
        bind(encoder, output, 0, 2);
        set(encoder, &dimension, 3);
        set(encoder, &u32::from(weight.is_some()), 4);
        set(encoder, &dimension, 5);
        set(encoder, &scale, 6);
        set(encoder, &epsilon, 7);
        grid(encoder, (rows as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// `hidden = (hidden + rms(branch) * weight) * scalar` over `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_add(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        branch: &MetalBuffer,
        weight: (&MetalBuffer, usize),
        rows: u32,
        dimension: u32,
        scalar: f32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || !scalar.is_finite() || epsilon.is_nan() || epsilon <= 0.0
        {
            return Err(arg("invalid embedding residual parameters"));
        }
        let bytes = f32s(u64::from(rows) * u64::from(dimension));
        need(hidden, bytes)?;
        need(branch, bytes)?;
        need(weight.0, weight.1.checked_add(dimension as usize * 4))?;
        if same(hidden, branch) || same(hidden, weight.0) {
            return Err(arg("residual output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms_add);
        bind(encoder, hidden, 0, 0);
        bind(encoder, branch, 0, 1);
        bind(encoder, weight.0, weight.1, 2);
        set(encoder, &dimension, 3);
        set(encoder, &scalar, 4);
        set(encoder, &epsilon, 5);
        grid(encoder, (rows as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Rotary embedding in place on `[tokens, heads, dimension]`;
    /// `positions` holds one `u32` per token.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        batch: &mut CommandBatch,
        x: &MetalBuffer,
        positions: &MetalBuffer,
        tokens: u32,
        heads: u32,
        dimension: u32,
        theta: f32,
    ) -> crate::Result<()> {
        if tokens == 0
            || heads == 0
            || dimension == 0
            || !dimension.is_multiple_of(2)
            || theta.is_nan()
            || theta <= 0.0
        {
            return Err(arg("invalid rotary parameters"));
        }
        let rows = tokens
            .checked_mul(heads)
            .ok_or_else(|| arg("rotary row overflow"))?;
        need(x, f32s(u64::from(rows) * u64::from(dimension)))?;
        need(positions, f32s(u64::from(tokens)))?;
        let threads = rows as usize * (dimension as usize / 2);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rope);
        bind(encoder, x, 0, 0);
        bind(encoder, positions, 0, 1);
        set(encoder, &heads, 2);
        set(encoder, &dimension, 3);
        set(encoder, &rows, 4);
        set(encoder, &theta, 5);
        grid(encoder, (threads.div_ceil(128), 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// `output[t, c] = gelu_tanh(gate[t, c]) * other[t * other_stride +
    /// other_offset + c]` for `rows` rows of `columns`.
    #[allow(clippy::too_many_arguments)]
    pub fn gelu_mul(
        &self,
        batch: &mut CommandBatch,
        gate: &MetalBuffer,
        other: &MetalBuffer,
        output: &MetalBuffer,
        rows: u32,
        columns: u32,
        other_stride: u32,
        other_offset: u32,
    ) -> crate::Result<()> {
        if rows == 0
            || columns == 0
            || other_offset
                .checked_add(columns)
                .is_none_or(|end| end > other_stride)
        {
            return Err(arg("invalid gated-GELU parameters"));
        }
        let count = u64::from(rows) * u64::from(columns);
        need(gate, f32s(count))?;
        need(output, f32s(count))?;
        need(other, f32s(u64::from(rows) * u64::from(other_stride)))?;
        if same(output, other) {
            return Err(arg("gated-GELU output aliases its multiplier"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gelu_mul);
        bind(encoder, gate, 0, 0);
        bind(encoder, other, 0, 1);
        bind(encoder, output, 0, 2);
        set(encoder, &columns, 3);
        set(encoder, &rows, 4);
        set(encoder, &other_stride, 5);
        set(encoder, &other_offset, 6);
        grid(encoder, ((count as usize).div_ceil(128), 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional packed attention; see [`AttentionShape`].
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        shape: AttentionShape,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        bounds: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        let pipeline = match shape.head_dim {
            256 => &self.attention_256,
            512 => &self.attention_512,
            _ => return Err(arg("attention head dim must be 256 or 512")),
        };
        if shape.tokens == 0
            || shape.q_heads == 0
            || shape.q_heads > 32
            || shape.kv_heads == 0
            || !shape.q_heads.is_multiple_of(shape.kv_heads)
        {
            return Err(arg("invalid attention head counts"));
        }
        let tokens = u64::from(shape.tokens);
        let dim = u64::from(shape.head_dim);
        need(q, f32s(tokens * u64::from(shape.q_heads) * dim))?;
        need(out, f32s(tokens * u64::from(shape.q_heads) * dim))?;
        need(k, f32s(tokens * u64::from(shape.kv_heads) * dim))?;
        need(v, f32s(tokens * u64::from(shape.kv_heads) * dim))?;
        need(bounds, f32s(tokens * 2))?;
        if [q, k, v, bounds].iter().any(|input| same(input, out)) {
            return Err(arg("attention output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        bind(encoder, q, 0, 0);
        bind(encoder, k, 0, 1);
        bind(encoder, v, 0, 2);
        bind(encoder, out, 0, 3);
        bind(encoder, bounds, 0, 4);
        set(encoder, &shape.q_heads, 5);
        set(encoder, &shape.kv_heads, 6);
        set(encoder, &shape.window, 7);
        grid(
            encoder,
            (shape.tokens as usize, 1),
            32 * shape.q_heads as usize,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// `output[t, r] = Σ_c W[r, c] · input[t, c]` for `tokens` contiguous
    /// input rows; the output is `[tokens, rows]`.
    pub fn bf16_gemm(
        &self,
        batch: &mut CommandBatch,
        matrix: Bf16Rows<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0
            || matrix.rows == 0
            || matrix.columns == 0
            || !matrix.columns.is_multiple_of(GEMM_COLUMN_MULTIPLE)
            || !matrix.offset.is_multiple_of(16)
        {
            return Err(arg("invalid BF16 GEMM shape"));
        }
        let rows = u64::from(matrix.rows);
        let columns = u64::from(matrix.columns);
        need(
            matrix.weights,
            usize::try_from(rows * columns * 2)
                .ok()
                .and_then(|bytes| bytes.checked_add(matrix.offset)),
        )?;
        need(input, f32s(u64::from(tokens) * columns))?;
        need(output, f32s(u64::from(tokens) * rows))?;
        if same(input, output) || same(matrix.weights, output) {
            return Err(arg("BF16 GEMM output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gemm);
        bind(encoder, matrix.weights, matrix.offset, 0);
        bind(encoder, input, 0, 1);
        bind(encoder, output, 0, 2);
        set(encoder, &matrix.rows, 3);
        set(encoder, &matrix.columns, 4);
        set(encoder, &tokens, 5);
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

#[cfg(test)]
mod tests;
