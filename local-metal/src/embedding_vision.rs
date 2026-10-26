//! Kernels of the `EmbeddingGemma` 2 vision encoder
//! (`shaders/embedding_vision.metal`).
//!
//! The vision tower shares the text encoder's GEMM, `RMSNorm`, residual, `RoPE`
//! and gated-GELU kernels ([`crate::embedding::EmbeddingKernels`]); these
//! add the learned 2-D position table, head-dim-64 bidirectional attention
//! over one image at a time, and the 3x3 average pooling. All F32.

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

/// Head dimension of [`VisionKernels::attention`].
pub const VISION_HEAD_DIM: u32 = 64;
/// Dynamic threadgroup memory of `egv_attention_64`, its `ks` tile of
/// `EGV_BK` (32) rows of `EGV_LD` (68) floats: Metal's shader validation
/// counts static threadgroup memory twice against the 32 KiB limit, so the
/// kernel aborted under `MTL_SHADER_VALIDATION=1` with the tile static.
const ATTENTION_KS_BYTES: usize = 32 * 68 * 4;
/// Patches averaged into one pooled token (3 x 3).
pub const POOL_MEMBERS: usize = 9;

pub struct VisionKernels {
    add_positions: Pipeline,
    pool: Pipeline,
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

#[expect(unsafe_code)]
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

#[expect(unsafe_code)]
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
            add_positions: pipeline("egv_add_positions")?,
            pool: pipeline("egv_pool")?,
            attention: pipeline("egv_attention_64")?,
        })
    }

    /// `hidden[t, c] += table[x_t, c] + table[table_rows + y_t, c]` for
    /// `tokens` rows of `width`; `table` is BF16 `[2 * table_rows, width]` at
    /// `table_offset` and `positions` holds `(x, y)` `u32` pairs, each below
    /// `table_rows` (the caller checks).
    #[expect(clippy::too_many_arguments)]
    pub fn add_positions(
        &self,
        batch: &mut CommandBatch,
        hidden: &MetalBuffer,
        table: (&MetalBuffer, usize),
        table_rows: u32,
        positions: &MetalBuffer,
        tokens: u32,
        width: u32,
    ) -> crate::Result<()> {
        let count = tokens
            .checked_mul(width)
            .ok_or_else(|| arg("position add overflow"))?;
        if count == 0 || table_rows == 0 || !table.1.is_multiple_of(2) {
            return Err(arg("invalid position add parameters"));
        }
        need(hidden, bytes4(u64::from(count)))?;
        need(positions, bytes4(u64::from(tokens) * 2))?;
        need(
            table.0,
            usize::try_from(u64::from(table_rows) * 2 * u64::from(width) * 2)
                .ok()
                .and_then(|bytes| bytes.checked_add(table.1)),
        )?;
        if same(hidden, table.0) || same(hidden, positions) {
            return Err(arg("position add output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.add_positions);
        bind(encoder, hidden, 0, 0);
        bind(encoder, table.0, table.1, 1);
        bind(encoder, positions, 0, 2);
        set(encoder, &width, 3);
        set(encoder, &count, 4);
        set(encoder, &table_rows, 5);
        grid(encoder, ((count as usize).div_ceil(128), 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// `output[o, c] = scale * mean_j input[members[9 o + j], c]` for
    /// `outputs` rows of `width`; every member index must be a row of
    /// `input` (the caller checks).
    #[expect(clippy::too_many_arguments)]
    pub fn pool(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        members: &MetalBuffer,
        output: &MetalBuffer,
        outputs: u32,
        width: u32,
        scale: f32,
    ) -> crate::Result<()> {
        let count = outputs
            .checked_mul(width)
            .ok_or_else(|| arg("pool overflow"))?;
        if count == 0 || !scale.is_finite() {
            return Err(arg("invalid pool parameters"));
        }
        need(output, bytes4(u64::from(count)))?;
        need(members, bytes4(u64::from(outputs) * POOL_MEMBERS as u64))?;
        if same(output, input) || same(output, members) {
            return Err(arg("pool output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.pool);
        bind(encoder, input, 0, 0);
        bind(encoder, members, 0, 1);
        bind(encoder, output, 0, 2);
        set(encoder, &width, 3);
        set(encoder, &count, 4);
        set(encoder, &scale, 5);
        grid(encoder, ((count as usize).div_ceil(128), 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional attention among tokens `[first, first + count)` only;
    /// `q`, `k`, `v` and `out` are `[tokens, heads, 64]` F32 rows.
    #[expect(clippy::too_many_arguments, unsafe_code)]
    pub fn attention(
        &self,
        batch: &mut CommandBatch,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        out: &MetalBuffer,
        first: u32,
        count: u32,
        heads: u32,
    ) -> crate::Result<()> {
        if count == 0 || heads == 0 {
            return Err(arg("invalid vision attention parameters"));
        }
        let end = u64::from(first) + u64::from(count);
        let bytes = bytes4(end * u64::from(heads) * u64::from(VISION_HEAD_DIM));
        for buffer in [q, k, v, out] {
            need(buffer, bytes)?;
        }
        if [q, k, v].iter().any(|input| same(input, out)) {
            return Err(arg("vision attention output aliases an input"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.attention);
        bind(encoder, q, 0, 0);
        bind(encoder, k, 0, 1);
        bind(encoder, v, 0, 2);
        bind(encoder, out, 0, 3);
        set(encoder, &first, 4);
        set(encoder, &count, 5);
        set(encoder, &heads, 6);
        // SAFETY: index 0 is the kernel's only threadgroup argument.
        unsafe { encoder.setThreadgroupMemoryLength_atIndex(ATTENTION_KS_BYTES, 0) };
        grid(
            encoder,
            ((count as usize).div_ceil(32), heads as usize),
            128,
        );
        batch.record_dispatch();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
