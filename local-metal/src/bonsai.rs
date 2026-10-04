//! Native primitives for Bonsai's `PTQ1_0` weights and signed 1024-wide rotation.
//!
//! This is not a model loader. The caller supplies the checkpoint's actual signs,
//! keeps sensitive state in its original precision, and owns graph scheduling.
//! `PTQ1_0` bytes stay packed during matvec; decoding one row is for embeddings and
//! diagnostics, not an invitation to expand the whole model.
//!
//! Format reference: PrismML-Eng/llama.cpp at
//! `9a9394a895b96003ca842a6041cb28ac49a108f7`. See `THIRD_PARTY_NOTICES.md`.

use core::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

pub const PTQ1_BLOCK_ELEMENTS: usize = 128;
pub const PTQ1_BLOCK_BYTES: usize = 28;
pub const HADAMARD_BLOCK_ELEMENTS: usize = 1024;

fn row_bytes(columns: usize) -> crate::Result<usize> {
    if columns == 0 || !columns.is_multiple_of(PTQ1_BLOCK_ELEMENTS) {
        return Err(Error::InvalidArgument(
            "PTQ1_0 columns must be a positive multiple of 128".into(),
        ));
    }
    (columns / PTQ1_BLOCK_ELEMENTS)
        .checked_mul(PTQ1_BLOCK_BYTES)
        .ok_or_else(|| Error::InvalidArgument("PTQ1_0 row size overflow".into()))
}

/// Decode packed, little-endian `PTQ1_0` blocks into caller-owned row storage.
///
/// The exact length must match: 24 five-trit bytes, two four-trit bytes, then one
/// FP16 scale per 128 values. Multiplication modulo 256 is part of the codec.
pub fn decode_ptq1_row(packed: &[u8], output: &mut [f32]) -> crate::Result<()> {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    if row_bytes(output.len())? != packed.len() {
        return Err(Error::InvalidArgument(
            "PTQ1_0 packed length does not match output row".into(),
        ));
    }
    for (block, values) in packed
        .as_chunks::<PTQ1_BLOCK_BYTES>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<PTQ1_BLOCK_ELEMENTS>().0)
    {
        let scale = f16::from_bits(u16::from_le_bytes([block[26], block[27]])).to_f32();
        for (element, value) in values.iter_mut().enumerate() {
            let (byte, power) = if element < 80 {
                (element % 16, element / 16)
            } else if element < 120 {
                (16 + (element - 80) % 8, (element - 80) / 8)
            } else {
                (24 + (element - 120) % 2, (element - 120) / 2)
            };
            let q = block[byte].wrapping_mul(POW3[power]);
            let trit = f32::from((u16::from(q) * 3) >> 8) - 1.0;
            *value = trit * scale;
        }
    }
    Ok(())
}

/// Checked row-major view into immutable packed weights, including an mmap or pool.
///
/// The buffer and any pool slot must not be modified/reused until GPU completion.
#[derive(Clone, Copy)]
pub struct Ptq1Matrix<'a> {
    buffer: &'a MetalBuffer,
    offset: usize,
    rows: u32,
    columns: u32,
}

impl<'a> Ptq1Matrix<'a> {
    pub fn new(
        buffer: &'a MetalBuffer,
        offset: usize,
        rows: u32,
        columns: u32,
    ) -> crate::Result<Self> {
        let bytes = row_bytes(columns as usize)?
            .checked_mul(rows as usize)
            .ok_or_else(|| Error::InvalidArgument("PTQ1_0 matrix size overflow".into()))?;
        if rows == 0
            || !offset.is_multiple_of(size_of::<f16>())
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > buffer.length())
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 matrix is empty, misaligned, or exceeds its buffer".into(),
            ));
        }
        Ok(Self {
            buffer,
            offset,
            rows,
            columns,
        })
    }

    #[must_use]
    pub const fn rows(self) -> u32 {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> u32 {
        self.columns
    }
}

/// Checkpoint-provided signs, one per column, repeated for each activation row.
///
/// Different 1024-wide column blocks can have different signs. No seed, signs, or
/// missing rotation metadata are guessed. Sign storage is immutable after upload.
pub struct SignedHadamard {
    signs: MetalBuffer,
    columns: u32,
}

impl SignedHadamard {
    pub fn new(context: &MetalContext, signs: &[f32]) -> crate::Result<Self> {
        if signs.is_empty()
            || signs.len() > u32::MAX as usize
            || !signs.len().is_multiple_of(HADAMARD_BLOCK_ELEMENTS)
            || signs
                .iter()
                .any(|&sign| sign.abs().to_bits() != 1.0_f32.to_bits())
        {
            return Err(Error::InvalidArgument(
                "Bonsai signs must be +/-1 in complete 1024-wide blocks".into(),
            ));
        }
        Ok(Self {
            signs: MetalBuffer::from_slice(context.device(), signs)?,
            columns: signs.len() as u32,
        })
    }

    #[must_use]
    pub const fn columns(&self) -> u32 {
        self.columns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardDirection {
    /// Activation rotation: normalized Hadamard after multiplying by signs.
    Forward,
    /// Embedding reconstruction: signs after normalized Hadamard, NOT forward twice.
    Inverse,
}

/// Execution primitive for multi-token Bonsai projections; both keep F32 inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefillKernel {
    /// Portable SIMD-group operations with F32 accumulation.
    #[default]
    SimdF32,
    /// Metal 4 tensor operations; exact half PTQ1 weights, F32 activations and sums.
    TensorF32,
}

impl PrefillKernel {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SimdF32 => "simd_f32",
            Self::TensorF32 => "tensor_f32",
        }
    }
}

/// Allocation-free encoding of F32 rotations and packed-weight matvecs.
///
/// The caller supplies scratch and completes the command batch before host reads.
/// Rotated activations can be shared by projections only when their signs match.
pub struct BonsaiKernels {
    forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    inverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Tensor-tile variants with 64- and 32-token tiles for verify-sized blocks.
    matmul_short: Option<[Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2]>,
    /// Exact-token kernels for 2..=`SMALL_BATCH_KERNEL_TOKENS` activation rows.
    small_batch: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    small_batch_max: u32,
    prefill_kernel: PrefillKernel,
}

/// Largest activation-row count handled by one small-batch dispatch.
///
/// Measured per-token cost is flat through four rows and steps up at five, so
/// larger blocks are split into balanced chunks of three or four.
pub const SMALL_BATCH_KERNEL_TOKENS: u32 = 4;

/// Token blocks up to this size use chunked small-batch dispatches; larger
/// blocks use the prefill tensor tile (32-, 64- or 128-token tiles).
///
/// Whole-model verify blocks on an M4 Pro at a 1,024-token prefix
/// (`benchmark_verify_block_rows`): small batch 643 ms at 24 rows and 841 ms at
/// 32; the 32-token tile is flat at about 650 ms from 9 to 32 rows and the
/// 64-token tile at about 860 ms from 33 to 64 rows. Small batch wins through 24.
pub const DEFAULT_SMALL_BATCH_MAX: u32 = 24;

impl BonsaiKernels {
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        // Older SDKs omit this function; unsupported devices can reject its PSO.
        // In either case retain the portable F32 path, not a precision fallback.
        Self::new_with_prefill_kernel(context, shaders, PrefillKernel::TensorF32)
            .or_else(|_| Self::new_with_prefill_kernel(context, shaders, PrefillKernel::SimdF32))
    }

    pub fn new_with_prefill_kernel(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        prefill_kernel: PrefillKernel,
    ) -> crate::Result<Self> {
        let pipeline = |name: &str, threads: usize| -> crate::Result<_> {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < threads
            {
                return Err(Error::InvalidArgument(
                    "Bonsai pipeline cannot support the required SIMD/threadgroup size".into(),
                ));
            }
            Ok(pipeline)
        };
        Ok(Self {
            forward: pipeline("bonsai_fwht_forward", 128)?,
            inverse: pipeline("bonsai_fwht_inverse", 128)?,
            matvec: pipeline("bonsai_ptq1_matvec_reuse", 32)?,
            matmul: pipeline(
                match prefill_kernel {
                    PrefillKernel::SimdF32 => "bonsai_ptq1_matmul_bytewise_32",
                    PrefillKernel::TensorF32 => "bonsai_ptq1_matmul_tensor",
                },
                128,
            )?,
            matmul_short: match prefill_kernel {
                PrefillKernel::SimdF32 => None,
                PrefillKernel::TensorF32 => Some([
                    pipeline("bonsai_ptq1_matmul_tensor_64", 128)?,
                    pipeline("bonsai_ptq1_matmul_tensor_32", 128)?,
                ]),
            },
            small_batch: (2..=SMALL_BATCH_KERNEL_TOKENS)
                .map(|tokens| pipeline(&format!("bonsai_ptq1_small_batch_{tokens}"), 32))
                .collect::<crate::Result<_>>()?,
            small_batch_max: DEFAULT_SMALL_BATCH_MAX,
            prefill_kernel,
        })
    }

    #[must_use]
    pub const fn prefill_kernel(&self) -> PrefillKernel {
        self.prefill_kernel
    }

    /// Largest token block routed to the small-batch kernels by [`Self::matmul`].
    #[must_use]
    pub const fn small_batch_max(&self) -> u32 {
        self.small_batch_max
    }

    /// Override the small-batch routing threshold; `1` disables it so every
    /// multi-token block uses the prefill tile (for measurements and controls).
    #[must_use]
    pub const fn with_small_batch_max(mut self, tokens: u32) -> Self {
        self.small_batch_max = tokens;
        self
    }

    /// Transform contiguous F32 rows. In-place transforms are supported.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn transform(
        &self,
        batch: &mut CommandBatch,
        rotation: &SignedHadamard,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        direction: HadamardDirection,
    ) -> crate::Result<()> {
        let elements = rotation
            .columns
            .checked_mul(tokens)
            .filter(|&count| count != 0)
            .ok_or_else(|| {
                Error::InvalidArgument("Bonsai activation shape overflow/empty".into())
            })?;
        let bytes = elements as usize * size_of::<f32>();
        if input.length() < bytes || output.length() < bytes {
            return Err(Error::InvalidArgument(
                "Bonsai rotation buffers are too short".into(),
            ));
        }
        let blocks_per_row = rotation.columns / HADAMARD_BLOCK_ELEMENTS as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(match direction {
            HadamardDirection::Forward => &self.forward,
            HadamardDirection::Inverse => &self.inverse,
        });
        unsafe {
            bind(encoder, input, 0, 0);
            bind(encoder, &rotation.signs, 0, 1);
            bind(encoder, output, 0, 2);
            set_u32(encoder, &blocks_per_row, 3);
        }
        dispatch(encoder, elements as usize / HADAMARD_BLOCK_ELEMENTS, 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Multiply packed weights by an already-rotated F32 vector; output is F32.
    ///
    /// This operation does not apply or infer a rotation. Separating it permits
    /// reuse of a matching transform across gate/up or attention projections.
    #[allow(unsafe_code)]
    pub fn matvec(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        if input.length() < matrix.columns as usize * size_of::<f32>()
            || output.length() < matrix.rows as usize * size_of::<f32>()
            || std::ptr::eq(input.raw(), output.raw())
            || std::ptr::eq(matrix.buffer.raw(), output.raw())
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 matvec buffers are too short or output aliases an input".into(),
            ));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.matvec);
        unsafe {
            bind(encoder, matrix.buffer, matrix.offset, 0);
            bind(encoder, input, 0, 1);
            bind(encoder, output, 0, 2);
            set_u32(encoder, &matrix.rows, 3);
            set_u32(encoder, &matrix.columns, 4);
        }
        // One SIMD group shares activation coefficients across four rows.
        dispatch(encoder, (matrix.rows as usize).div_ceil(4), 32);
        batch.record_dispatch();
        Ok(())
    }

    /// Packed-weight projection of contiguous `[tokens, columns]` F32 rows.
    /// Output is `[tokens, rows]`. A single token retains the measured matvec;
    /// blocks up to [`Self::small_batch_max`] stream the weights once per
    /// `SMALL_BATCH_KERNEL_TOKENS` rows; larger blocks use the constructor-
    /// selected prefill primitive. Nothing expands the model; activations,
    /// accumulation and output stay F32.
    #[allow(unsafe_code)]
    pub fn matmul(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 1 {
            return self.matvec(batch, matrix, input, output);
        }
        let extent = |width: u32| {
            (tokens as usize)
                .checked_mul(width as usize)
                .and_then(|count| count.checked_mul(size_of::<f32>()))
                .ok_or_else(|| Error::InvalidArgument("PTQ1_0 matmul extent overflow".into()))
        };
        if tokens == 0
            || input.length() < extent(matrix.columns)?
            || output.length() < extent(matrix.rows)?
            || std::ptr::eq(input.raw(), output.raw())
            || std::ptr::eq(matrix.buffer.raw(), output.raw())
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 matmul buffers are empty, too short, or aliased".into(),
            ));
        }
        if tokens <= self.small_batch_max {
            self.small_batch(batch, matrix, input, output, tokens);
            return Ok(());
        }
        let (pipeline, tile_tokens) = match (&self.matmul_short, tokens) {
            (Some([_, short]), 0..=32) => (short, 32),
            (Some([short, _]), 33..=64) => (short, 64),
            _ => (
                &self.matmul,
                match self.prefill_kernel {
                    PrefillKernel::SimdF32 => 32,
                    PrefillKernel::TensorF32 => 128,
                },
            ),
        };
        let encoder = batch.encoder();
        encoder.setComputePipelineState(pipeline);
        unsafe {
            bind(encoder, matrix.buffer, matrix.offset, 0);
            bind(encoder, input, 0, 1);
            bind(encoder, output, 0, 2);
            set_u32(encoder, &matrix.rows, 3);
            set_u32(encoder, &matrix.columns, 4);
            set_u32(encoder, &tokens, 5);
        }
        let tile_rows = 32;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (matrix.rows as usize).div_ceil(tile_rows),
                height: (tokens as usize).div_ceil(tile_tokens),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Exact-token dispatches of at most `SMALL_BATCH_KERNEL_TOKENS` rows each;
    /// the caller has validated extents and aliasing. Each dispatch reads the
    /// packed matrix once, so cost grows with `ceil(tokens / 4)`, not tiles.
    #[allow(unsafe_code)]
    fn small_batch(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        debug_assert!(tokens >= 2);
        // Balanced chunks (5 -> 3 + 2, 9 -> 3 + 3 + 3) keep every dispatch at
        // two or more rows and never revisit a row.
        let chunks = tokens.div_ceil(SMALL_BATCH_KERNEL_TOKENS);
        let mut start = 0;
        for chunk in 0..chunks {
            let end = (tokens * (chunk + 1)).div_ceil(chunks);
            let count = end - start;
            let encoder = batch.encoder();
            encoder.setComputePipelineState(&self.small_batch[count as usize - 2]);
            unsafe {
                bind(encoder, matrix.buffer, matrix.offset, 0);
                bind(encoder, input, 0, 1);
                bind(encoder, output, 0, 2);
                set_u32(encoder, &matrix.rows, 3);
                set_u32(encoder, &matrix.columns, 4);
                set_u32(encoder, &start, 5);
            }
            dispatch(encoder, (matrix.rows as usize).div_ceil(4), 32);
            batch.record_dispatch();
            start = end;
        }
    }
}

fn dispatch(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, groups: usize, threads: usize) {
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

#[allow(unsafe_code)]
unsafe fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffer: &MetalBuffer,
    offset: usize,
    index: usize,
) {
    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.raw()), offset, index) };
}

#[allow(unsafe_code, clippy::trivially_copy_pass_by_ref)]
unsafe fn set_u32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &u32,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<u32>(),
            index,
        );
    }
}
