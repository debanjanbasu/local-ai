//! Kernels of the Bonsai 2 27B vision tower (`shaders/bonsai_vision.metal`):
//! the Qwen3.5 `ViT` shipped as the checkpoint's mmproj.
//!
//! A BF16 x F32 GEMM with bias and a fused epilogue (GELU or residual add),
//! `LayerNorm`, the 2-D vision `RoPE` and bidirectional head-dim-72
//! attention over one image. All F32.

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

/// Width of a patch row.
pub const VISION_WIDTH: u32 = 1152;
/// Attention heads.
pub const VISION_HEADS: u32 = 16;
/// Head dimension.
pub const VISION_HEAD_DIM: u32 = 72;

/// What [`VisionKernels::gemm`] does with each result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// Store `W x + b`.
    Store = 0,
    /// Store `gelu_tanh(W x + b)`.
    GeluTanh = 1,
    /// Store `gelu(W x + b)` (exact, erf).
    GeluErf = 2,
    /// Add `W x + b` to the output.
    Accumulate = 3,
}

/// A BF16 (or, with `f32`, F32) matrix `[rows][columns]` at a byte offset of
/// a buffer, with its F32 bias `[rows]`.
#[derive(Clone, Copy)]
pub struct Linear<'a> {
    pub f32: bool,
    pub weights: &'a MetalBuffer,
    pub weights_offset: usize,
    pub bias: &'a MetalBuffer,
    pub bias_offset: usize,
    pub rows: u32,
    pub columns: u32,
}

pub struct VisionKernels {
    gemm: Pipeline,
    gemm_f32: Pipeline,
    layernorm: Pipeline,
    rope: Pipeline,
    attention: Pipeline,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(message.to_owned())
}

fn need(buffer: &MetalBuffer, bytes: Option<usize>) -> crate::Result<()> {
    match bytes {
        Some(bytes) if bytes <= buffer.length() => Ok(()),
        _ => Err(arg("vision kernel buffer too short")),
    }
}

fn bytes4(count: u64) -> Option<usize> {
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

impl VisionKernels {
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
                return Err(arg("vision pipeline cannot run 128-thread threadgroups"));
            }
            Ok(pipeline)
        };
        Ok(Self {
            gemm: pipeline("bv_gemm")?,
            gemm_f32: pipeline("bv_gemm_f32")?,
            layernorm: pipeline("bv_layernorm")?,
            rope: pipeline("bv_rope")?,
            attention: pipeline("bv_attention")?,
        })
    }

    /// `epilogue(W x_t + b)` for `tokens` contiguous input rows of
    /// `linear.columns` (a multiple of 8) into `[tokens, linear.rows]`.
    pub fn gemm(
        &self,
        batch: &mut CommandBatch,
        linear: Linear<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        epilogue: Epilogue,
    ) -> crate::Result<()> {
        if tokens == 0
            || linear.rows == 0
            || linear.columns == 0
            || !linear.columns.is_multiple_of(8)
            || !linear.weights_offset.is_multiple_of(16)
            || !linear.bias_offset.is_multiple_of(4)
        {
            return Err(arg("invalid vision GEMM shape"));
        }
        let rows = u64::from(linear.rows);
        let columns = u64::from(linear.columns);
        let element = if linear.f32 { 4 } else { 2 };
        need(
            linear.weights,
            usize::try_from(rows * columns * element)
                .ok()
                .and_then(|bytes| bytes.checked_add(linear.weights_offset)),
        )?;
        need(
            linear.bias,
            bytes4(rows).and_then(|bytes| bytes.checked_add(linear.bias_offset)),
        )?;
        need(input, bytes4(u64::from(tokens) * columns))?;
        need(output, bytes4(u64::from(tokens) * rows))?;
        if same(input, output) || same(linear.weights, output) || same(linear.bias, output) {
            return Err(arg("vision GEMM output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(if linear.f32 {
            &self.gemm_f32
        } else {
            &self.gemm
        });
        bind(encoder, linear.weights, linear.weights_offset, 0);
        bind(encoder, linear.bias, linear.bias_offset, 1);
        bind(encoder, input, 0, 2);
        bind(encoder, output, 0, 3);
        set(encoder, &linear.rows, 4);
        set(encoder, &linear.columns, 5);
        set(encoder, &tokens, 6);
        set(encoder, &(epilogue as u32), 7);
        grid(
            encoder,
            (
                (linear.rows as usize).div_ceil(32),
                (tokens as usize).div_ceil(32),
            ),
            128,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// `LayerNorm` of `tokens` rows of `width`, weight and bias `[width]` F32
    /// at the given byte offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        weight: (&MetalBuffer, usize),
        bias: (&MetalBuffer, usize),
        output: &MetalBuffer,
        tokens: u32,
        width: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if tokens == 0 || width == 0 || !(epsilon > 0.0 && epsilon.is_finite()) {
            return Err(arg("invalid vision LayerNorm parameters"));
        }
        let elements = u64::from(tokens) * u64::from(width);
        need(input, bytes4(elements))?;
        need(output, bytes4(elements))?;
        need(
            weight.0,
            bytes4(u64::from(width)).and_then(|b| b.checked_add(weight.1)),
        )?;
        need(
            bias.0,
            bytes4(u64::from(width)).and_then(|b| b.checked_add(bias.1)),
        )?;
        if same(input, output) {
            return Err(arg("vision LayerNorm output aliases its input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.layernorm);
        bind(encoder, input, 0, 0);
        bind(encoder, weight.0, weight.1, 1);
        bind(encoder, bias.0, bias.1, 2);
        bind(encoder, output, 0, 3);
        set(encoder, &width, 4);
        set(encoder, &epsilon, 5);
        grid(encoder, (tokens as usize, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Rotate the q and k thirds of `tokens` qkv rows in place by the
    /// `[tokens][36]` cosine and sine tables.
    pub fn rope(
        &self,
        batch: &mut CommandBatch,
        qkv: &MetalBuffer,
        cosines: &MetalBuffer,
        sines: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(arg("invalid vision RoPE parameters"));
        }
        let half = u64::from(VISION_HEAD_DIM / 2);
        need(qkv, bytes4(u64::from(tokens) * 3 * u64::from(VISION_WIDTH)))?;
        need(cosines, bytes4(u64::from(tokens) * half))?;
        need(sines, bytes4(u64::from(tokens) * half))?;
        if same(qkv, cosines) || same(qkv, sines) {
            return Err(arg("vision RoPE output aliases a table"));
        }
        let threads = u64::from(tokens) * 2 * u64::from(VISION_HEADS) * half;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rope);
        bind(encoder, qkv, 0, 0);
        bind(encoder, cosines, 0, 1);
        bind(encoder, sines, 0, 2);
        set(encoder, &tokens, 3);
        let groups =
            usize::try_from(threads.div_ceil(128)).map_err(|_| arg("vision RoPE overflow"))?;
        grid(encoder, (groups, 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional attention among the `count` patches of one image, from
    /// qkv rows `[count][3][16][72]` (byte offset `qkv_offset`) into
    /// `[count][16][72]` rows of `out` at `out_offset`, for the queries
    /// `queries` (a start that is a multiple of 64), so a large image can
    /// split its attention over several command buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        qkv: &MetalBuffer,
        qkv_offset: usize,
        out: &MetalBuffer,
        out_offset: usize,
        count: u32,
        queries: std::ops::Range<u32>,
        scale: f32,
    ) -> crate::Result<()> {
        if count == 0
            || !scale.is_finite()
            || queries.is_empty()
            || queries.end > count
            || !queries.start.is_multiple_of(64)
        {
            return Err(arg("invalid vision attention parameters"));
        }
        let width = u64::from(VISION_WIDTH);
        need(
            qkv,
            bytes4(u64::from(count) * 3 * width).and_then(|b| b.checked_add(qkv_offset)),
        )?;
        need(
            out,
            bytes4(u64::from(count) * width).and_then(|b| b.checked_add(out_offset)),
        )?;
        if same(qkv, out) {
            return Err(arg("vision attention output aliases its input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention);
        bind(encoder, qkv, qkv_offset, 0);
        bind(encoder, out, out_offset, 1);
        set(encoder, &count, 2);
        set(encoder, &scale, 3);
        set(encoder, &queries.start, 4);
        grid(
            encoder,
            ((queries.len()).div_ceil(64), VISION_HEADS as usize),
            256,
        );
        batch.record_dispatch();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
