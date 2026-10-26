//! GPU side of the Neural Engine prefill FFN (`shaders/bonsai_ane.metal`).
//!
//! Pack F32 activation rows into the ANE's int8 input surfaces, convert one
//! layer's ternary weights into per-row int8 surfaces, and join the ANE's
//! partial output back into F32 rows.

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::Error;
use crate::ane::Surface;
use crate::batch::CommandBatch;
use crate::bonsai::{PTQ1_BLOCK_ELEMENTS, Ptq1Matrix};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[repr(C)]
struct ScaleParams {
    tokens: u32,
    width: u32,
}

#[repr(C)]
struct PackParams {
    tokens: u32,
    width: u32,
    split: u32,
    stride: u32,
}

#[repr(C)]
struct StageParams {
    blocks: u32,
    first: u32,
    count: u32,
    stride: u32,
    scale_stride: u32,
}

#[repr(C)]
struct JoinParams {
    tokens: u32,
    width: u32,
    ystride: u32,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(format!("Neural Engine FFN: {message}"))
}

/// Pipelines of the pack, weight-staging and join kernels.
pub struct AneKernels {
    token_scales: Pipeline,
    pack: Pipeline,
    stage: Pipeline,
    join: Pipeline,
}

#[expect(unsafe_code)]
unsafe fn bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &T,
    index: usize,
) {
    // SAFETY: `value` is a plain `repr(C)` struct read during this call.
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            index,
        );
    }
}

#[expect(unsafe_code)]
unsafe fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &MetalBuffer,
    offset: usize,
    index: usize,
) {
    // SAFETY: the caller checked `buffer` covers what the kernel reads.
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), offset, index) };
}

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: (usize, usize),
    threads: (usize, usize),
) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups.0,
            height: groups.1,
            depth: 1,
        },
        MTLSize {
            width: threads.0,
            height: threads.1,
            depth: 1,
        },
    );
}

impl AneKernels {
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let pipeline = |name: &str| -> crate::Result<Pipeline> {
            let function = shaders.get_function(name)?;
            context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))
        };
        Ok(Self {
            token_scales: pipeline("bonsai_ane_token_scales")?,
            pack: pipeline("bonsai_ane_pack")?,
            stage: pipeline("bonsai_ane_stage")?,
            join: pipeline("bonsai_ane_join")?,
        })
    }

    /// Quantize `tokens` F32 rows of `input` (`width` wide) per token: the
    /// fp16 scales into `scales` (one row of at least `rows` elements, zero
    /// past `tokens`) and the int8 codes channel-major into `outputs`, each
    /// holding `width / 2` channels of `rows` tokens. `rows`, the ANE
    /// function's row count, is a multiple of 32 no smaller than `tokens`.
    #[expect(unsafe_code)]
    #[expect(clippy::too_many_arguments)]
    pub fn pack(
        &self,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        width: u32,
        tokens: u32,
        rows: u32,
        scales: &Surface,
        outputs: [&Surface; 2],
    ) -> crate::Result<()> {
        let split = width / 2;
        let stride = outputs[0].layout().stride;
        if tokens == 0
            || tokens > rows
            || !rows.is_multiple_of(32)
            || !split.is_multiple_of(32)
            || input.length() < tokens as usize * width as usize * size_of::<f32>()
            || !scales.layout().fp16
            || scales.layout().width < rows
            || outputs.iter().any(|surface| {
                surface.layout().fp16
                    || surface.layout().rows != split
                    || surface.layout().width < rows
                    || surface.layout().stride != stride
            })
        {
            return Err(arg("pack surfaces do not match the block"));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.token_scales);
        let params = ScaleParams { tokens, width };
        // SAFETY: the buffers were checked above.
        unsafe {
            bind(encoder, input, 0, 0);
            bind(encoder, scales.buffer(), 0, 1);
            bytes(encoder, &params, 2);
        }
        dispatch(encoder, (rows as usize, 1), (256, 1));
        batch.record_dispatch();
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.pack);
        let params = PackParams {
            tokens,
            width,
            split,
            stride: stride as u32,
        };
        // SAFETY: as above.
        unsafe {
            bind(encoder, input, 0, 0);
            bind(encoder, scales.buffer(), 0, 1);
            bind(encoder, outputs[0].buffer(), 0, 2);
            bind(encoder, outputs[1].buffer(), 0, 3);
            bytes(encoder, &params, 4);
        }
        dispatch(
            encoder,
            ((rows / 32) as usize, (width / 32) as usize),
            (32, 8),
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Convert the column blocks `first..first + count` of every row of a
    /// ternary `matrix` into per-row int8 codes in `output` (`count * 128`
    /// columns), and with `scales` write each row's fp16 scale (times 128)
    /// there. Each row's scale is its largest block scale over 127, so the
    /// codes are the trits times `rint(127 s_b / max |s|)`.
    #[expect(unsafe_code)]
    pub fn stage(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        first: u32,
        count: u32,
        output: &Surface,
        scales: Option<&Surface>,
    ) -> crate::Result<()> {
        let blocks = matrix.columns / PTQ1_BLOCK_ELEMENTS as u32;
        let layout = output.layout();
        if count == 0
            || first + count > blocks
            || !matrix.rows.is_multiple_of(8)
            || layout.fp16
            || layout.rows != matrix.rows
            || layout.width != count * PTQ1_BLOCK_ELEMENTS as u32
            || scales
                .is_some_and(|scales| !scales.layout().fp16 || scales.layout().rows != matrix.rows)
        {
            return Err(arg("staged surfaces do not match the matrix"));
        }
        let params = StageParams {
            blocks,
            first,
            count,
            stride: layout.stride as u32,
            scale_stride: scales.map_or(0, |scales| (scales.layout().stride / 2) as u32),
        };
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.stage);
        // SAFETY: the matrix view was checked against its buffer when made;
        // the surfaces above.
        unsafe {
            bind(encoder, matrix.buffer, matrix.offset, 0);
            bind(encoder, output.buffer(), 0, 1);
            // Bound but never written when there are no scales.
            bind(encoder, scales.unwrap_or(output).buffer(), 0, 2);
            bytes(encoder, &params, 3);
        }
        dispatch(encoder, ((matrix.rows / 8) as usize, 1), (256, 1));
        batch.record_dispatch();
        Ok(())
    }

    /// `output[t][c] = partial[c][t] * partial[width][t] * scales[t]` for
    /// `t < tokens`, in F32; a non-finite partial or scale sets the `u32` at
    /// `status`.
    #[expect(unsafe_code)]
    #[expect(clippy::too_many_arguments)]
    pub fn join(
        &self,
        batch: &mut CommandBatch,
        partial: &Surface,
        scales: &Surface,
        output: &MetalBuffer,
        status: &MetalBuffer,
        width: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        let layout = partial.layout();
        if tokens == 0
            || !width.is_multiple_of(32)
            || !layout.fp16
            || layout.rows != width + 1
            || layout.width < tokens.next_multiple_of(32)
            || scales.layout().width < tokens.next_multiple_of(32)
            || output.length() < tokens as usize * width as usize * size_of::<f32>()
            || status.length() < size_of::<u32>()
        {
            return Err(arg("join buffers do not match the block"));
        }
        let params = JoinParams {
            tokens,
            width,
            ystride: (layout.stride / 2) as u32,
        };
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.join);
        // SAFETY: the buffers were checked above.
        unsafe {
            bind(encoder, partial.buffer(), 0, 0);
            bind(encoder, scales.buffer(), 0, 1);
            bind(encoder, output, 0, 2);
            bind(encoder, status, 0, 3);
            bytes(encoder, &params, 4);
        }
        dispatch(
            encoder,
            (tokens.div_ceil(32) as usize, (width / 32) as usize),
            (32, 8),
        );
        batch.record_dispatch();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
