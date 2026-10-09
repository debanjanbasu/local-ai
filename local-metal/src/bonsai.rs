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
use crate::bonsai_ops::Bf16Matrix;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

mod int8;

pub use self::int8::{INT8_CHUNK_TOKENS, INT8_COLUMN_MULTIPLE, INT8_VECTOR_TOKENS, Int8Matrix};

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
/// Single-token projections also need a 4-byte-aligned offset (GGUF tensors
/// are 32-aligned); they reject a 2-byte-aligned view.
#[derive(Clone, Copy)]
pub struct Ptq1Matrix<'a> {
    pub(crate) buffer: &'a MetalBuffer,
    pub(crate) offset: usize,
    pub(crate) rows: u32,
    pub(crate) columns: u32,
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

    /// The single-token kernels read each block's code bytes as aligned
    /// four-byte words.
    const fn word_aligned(self) -> bool {
        self.offset.is_multiple_of(4)
    }
}

/// Checkpoint-provided signs, one per column, repeated for each activation row.
///
/// Different 1024-wide column blocks can have different signs. No seed, signs, or
/// missing rotation metadata are guessed. Sign storage is immutable after upload.
pub struct SignedHadamard {
    pub(crate) signs: MetalBuffer,
    pub(crate) columns: u32,
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
    matvec_swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_concat: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matvec_concat_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rms_forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    swiglu_forward: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Tensor-tile variants with 64- and 32-token tiles for verify-sized blocks.
    matmul_short: Option<[Retained<ProtocolObject<dyn MTLComputePipelineState>>; 2]>,
    /// Exact-token kernels for 2..=`SMALL_BATCH_KERNEL_TOKENS` activation rows.
    small_batch: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// Simdgroup-matrix kernel for up to `SMALL_BATCH_WIDE_TOKENS` rows.
    small_batch_wide: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// 64-token by 64-row threadgroup tiles for blocks of `large_batch_min` rows or more.
    large_batch: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    small_batch_max: u32,
    large_batch_min: u32,
    prefill_kernel: PrefillKernel,
    int8: int8::Int8Kernels,
}

/// Largest activation-row count handled by one scalar small-batch dispatch.
///
/// On an M4 Pro (17408x5120 / 5120x17408) the scalar kernel measures 191/194,
/// 235/237 and 281/286 us at 2/3/4 rows and 337/351 us at five, where the wide
/// kernel's 340/345 us is no slower; five or more rows use the wide kernel.
/// With the half-prefix trit decoder: 155/162, 198/200 and 235/244 us against
/// the wide kernel's 280/286; with balanced prefixes and a two-row loop order
/// that keeps both input rows in registers, 141/143, 188/189 and 228/233 us.
pub const SMALL_BATCH_KERNEL_TOKENS: u32 = 4;

/// Largest activation-row count handled by one wide small-batch dispatch.
///
/// The simdgroup-matrix kernel's cost is flat in the row count, 340-348 us on the
/// two FFN shapes for 5 to 8 rows (280-301 us with half-prefix decoding,
/// 233-281 us with eight-byte activation loads),
/// against 452-720 us for the former two scalar dispatches. A 16-row variant
/// sharing the decoded trits across two token tiles measured 721-735 us, twice
/// the 8-row cost: the F32 8x8 multiplies themselves, about 270 us per eight
/// rows, are the limit, not trit decoding.
pub const SMALL_BATCH_WIDE_TOKENS: u32 = 8;

/// Token blocks up to this size (and below [`DEFAULT_LARGE_BATCH_MIN`]) use
/// chunked small-batch dispatches.
///
/// Larger blocks use the prefill tensor tile (32, 64 or 128 tokens) only when
/// the large-batch kernel is disabled.
///
/// Whole-model blocks on an M4 Pro at a 1,024-token prefix, best of five,
/// with eight-byte activation loads in the wide kernel: small batch 520 / 588 /
/// 594 / 634 ms at 56 / 60 / 64 / 65 verify rows against 776 / 1,224 ms on the
/// 64- and 128-token tiles at 64 / 65, and 866 / 1,155 ms at 96 / 128 rows
/// without logits against the 128-token tile's 1,221. Every verify block and
/// prefill chunk (at most 128 rows) therefore stays off the tensor tile: below
/// [`DEFAULT_LARGE_BATCH_MIN`] rows on the small-batch kernels, from there on
/// the large-batch kernel. (Before those loads the range ended at 60 rows:
/// 860.5 ms against the 64-token tile's 876.6.)
pub const DEFAULT_SMALL_BATCH_MAX: u32 = 128;

/// Token blocks of at least this many rows use the large-batch kernel.
///
/// It decodes each packed block once per 64-token by 64-row threadgroup tile
/// and runs the multiplies near the 8x8 matrix units' peak (3.4 T
/// multiply-adds per second against the wide kernel's 3.0 T). A remainder of
/// fewer rows past the last whole tile goes to the small-batch kernels.
///
/// Whole-model blocks on an M4 Pro (`prefill_block_timings`, no logits), ms,
/// small-batch against large-batch routing: 32 rows 285 / 483, 48 rows 427 /
/// 490, 64 rows 568-574 / 497-499, 128 rows 1,141-1,165 / 965-967. An empty
/// tile costs what a full one does, so the crossover is where seven or eight
/// eight-row passes (500 / 570 ms) exceed one tile.
pub const DEFAULT_LARGE_BATCH_MIN: u32 = 56;

/// Token rows per large-batch threadgroup tile.
const LARGE_BATCH_TILE: usize = 64;

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
            matvec_swiglu: pipeline("bonsai_ptq1_matvec_swiglu", 32)?,
            matvec_concat: pipeline("bonsai_ptq1_matvec_concat", 32)?,
            matvec_concat_bf16: pipeline("bonsai_ptq1_matvec_concat_bf16", 32)?,
            rms_forward: pipeline("bonsai_rms_fwht_forward", 128)?,
            swiglu_forward: pipeline("bonsai_swiglu_fwht_forward", 128)?,
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
            small_batch_wide: pipeline("bonsai_ptq1_small_batch_wide", 128)?,
            large_batch: pipeline("bonsai_ptq1_large_batch", 128)?,
            small_batch_max: DEFAULT_SMALL_BATCH_MAX,
            large_batch_min: DEFAULT_LARGE_BATCH_MIN,
            prefill_kernel,
            int8: int8::Int8Kernels::new(pipeline)?,
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
    /// multi-token block below [`Self::large_batch_min`] uses the prefill tile
    /// (for measurements and controls). Large-batch remainders still use the
    /// small-batch kernels; also pass `u32::MAX` to
    /// [`Self::with_large_batch_min`] to put every block on the tile.
    #[must_use]
    pub const fn with_small_batch_max(mut self, tokens: u32) -> Self {
        self.small_batch_max = tokens;
        self
    }

    /// Smallest token block routed to the large-batch kernel by [`Self::matmul`].
    #[must_use]
    pub const fn large_batch_min(&self) -> u32 {
        self.large_batch_min
    }

    /// Override the large-batch routing threshold; `u32::MAX` disables it so
    /// blocks keep the small-batch kernels or the prefill tile (for
    /// measurements and controls).
    #[must_use]
    pub const fn with_large_batch_min(mut self, tokens: u32) -> Self {
        self.large_batch_min = tokens;
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

    /// `silu(gate) * up` over `tokens` contiguous rows as wide as the rotation,
    /// rotated forward into `output`, in one dispatch; the product is never
    /// stored. Values are bitwise those of `BonsaiOps::swiglu` followed by
    /// [`Self::transform`] with [`HadamardDirection::Forward`]. The output may
    /// alias `gate` or `up` (in place), as each threadgroup reads its block
    /// before writing it.
    #[allow(unsafe_code)]
    pub fn swiglu_transform(
        &self,
        batch: &mut CommandBatch,
        rotation: &SignedHadamard,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let elements = rotation
            .columns
            .checked_mul(tokens)
            .filter(|&count| count != 0)
            .ok_or_else(|| {
                Error::InvalidArgument("Bonsai activation shape overflow/empty".into())
            })?;
        let bytes = elements as usize * size_of::<f32>();
        if gate.length() < bytes || up.length() < bytes || output.length() < bytes {
            return Err(Error::InvalidArgument(
                "Bonsai SwiGLU-rotate buffers are too short".into(),
            ));
        }
        let blocks_per_row = rotation.columns / HADAMARD_BLOCK_ELEMENTS as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.swiglu_forward);
        unsafe {
            bind(encoder, gate, 0, 0);
            bind(encoder, up, 0, 1);
            bind(encoder, &rotation.signs, 0, 2);
            bind(encoder, output, 0, 3);
            set_u32(encoder, &blocks_per_row, 4);
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
        if !matrix.word_aligned()
            || input.length() < matrix.columns as usize * size_of::<f32>()
            || output.length() < matrix.rows as usize * size_of::<f32>()
            || std::ptr::eq(input.raw(), output.raw())
            || std::ptr::eq(matrix.buffer.raw(), output.raw())
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 matvec matrix is not 4-byte aligned, buffers are too short, or output \
                 aliases an input"
                    .into(),
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
        // One SIMD group shares activation coefficients across eight rows.
        dispatch(encoder, (matrix.rows as usize).div_ceil(8), 32);
        batch.record_dispatch();
        Ok(())
    }

    /// Single-token `silu(gate * input) * (up * input)` in one dispatch;
    /// neither projection is stored. Both matrices must have the same shape.
    #[allow(unsafe_code)]
    pub fn matvec_swiglu(
        &self,
        batch: &mut CommandBatch,
        gate: Ptq1Matrix<'_>,
        up: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        if gate.rows != up.rows
            || !gate.word_aligned()
            || !up.word_aligned()
            || gate.columns != up.columns
            || input.length() < gate.columns as usize * size_of::<f32>()
            || output.length() < gate.rows as usize * size_of::<f32>()
            || std::ptr::eq(input.raw(), output.raw())
            || std::ptr::eq(gate.buffer.raw(), output.raw())
            || std::ptr::eq(up.buffer.raw(), output.raw())
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 SwiGLU shapes differ, are misaligned, buffers are too short, or output aliases"
                    .into(),
            ));
        }
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.matvec_swiglu);
        unsafe {
            bind(encoder, gate.buffer, gate.offset, 0);
            bind(encoder, input, 0, 1);
            bind(encoder, output, 0, 2);
            set_u32(encoder, &gate.rows, 3);
            set_u32(encoder, &gate.columns, 4);
            bind(encoder, up.buffer, up.offset, 5);
        }
        // Four rows of each matrix per SIMD group.
        dispatch(encoder, (gate.rows as usize).div_ceil(4), 32);
        batch.record_dispatch();
        Ok(())
    }

    /// Single-token projections of one input by one to three matrices with the
    /// same column count, in one dispatch. Outputs must be distinct from the
    /// input and every weight buffer.
    pub fn matvec_concat(
        &self,
        batch: &mut CommandBatch,
        projections: &[(Ptq1Matrix<'_>, &MetalBuffer)],
        input: &MetalBuffer,
    ) -> crate::Result<()> {
        self.concat(batch, projections, input, None)
    }

    /// [`Self::matvec_concat`] plus two same-shape single-token BF16
    /// projections of `bf16_input`, bitwise as `BonsaiOps::bf16_matvec`
    /// computes them, in the same dispatch.
    pub fn matvec_concat_bf16(
        &self,
        batch: &mut CommandBatch,
        projections: &[(Ptq1Matrix<'_>, &MetalBuffer)],
        input: &MetalBuffer,
        bf16: [(Bf16Matrix<'_>, &MetalBuffer); 2],
        bf16_input: &MetalBuffer,
    ) -> crate::Result<()> {
        let [(first, _), (second, _)] = bf16;
        if first.rows == 0
            || first.columns == 0
            || (first.rows, first.columns) != (second.rows, second.columns)
            || bf16_input.length() < first.columns as usize * size_of::<f32>()
            || bf16.iter().any(|(matrix, output)| {
                !matrix.offset.is_multiple_of(2)
                    || (matrix.rows as usize)
                        .checked_mul(matrix.columns as usize * 2)
                        .and_then(|bytes| bytes.checked_add(matrix.offset))
                        .is_none_or(|end| end > matrix.buffer.length())
                    || output.length() < matrix.rows as usize * size_of::<f32>()
                    || [input, bf16_input, matrix.buffer]
                        .iter()
                        .any(|read| std::ptr::eq(read.raw(), output.raw()))
                    || projections.iter().any(|(other, packed)| {
                        std::ptr::eq(other.buffer.raw(), output.raw())
                            || std::ptr::eq(packed.raw(), output.raw())
                    })
            })
            || std::ptr::eq(bf16[0].1.raw(), bf16[1].1.raw())
            || projections.iter().any(|(_, packed)| {
                std::ptr::eq(packed.raw(), bf16_input.raw())
                    || bf16
                        .iter()
                        .any(|(matrix, _)| std::ptr::eq(matrix.buffer.raw(), packed.raw()))
            })
        {
            return Err(Error::InvalidArgument(
                "BF16 projections in a concatenated matvec are invalid or aliased".into(),
            ));
        }
        self.concat(batch, projections, input, Some((bf16, bf16_input)))
    }

    #[allow(unsafe_code, clippy::type_complexity)]
    fn concat(
        &self,
        batch: &mut CommandBatch,
        projections: &[(Ptq1Matrix<'_>, &MetalBuffer)],
        input: &MetalBuffer,
        bf16: Option<([(Bf16Matrix<'_>, &MetalBuffer); 2], &MetalBuffer)>,
    ) -> crate::Result<()> {
        let Some(&(first, first_output)) = projections.first() else {
            return Err(Error::InvalidArgument(
                "PTQ1_0 concatenated matvec needs one to three matrices".into(),
            ));
        };
        if projections.len() > 3
            || input.length() < first.columns as usize * size_of::<f32>()
            || projections.iter().any(|(matrix, output)| {
                matrix.columns != first.columns
                    || !matrix.word_aligned()
                    || output.length() < matrix.rows as usize * size_of::<f32>()
                    || std::ptr::eq(input.raw(), output.raw())
                    || projections
                        .iter()
                        .any(|(other, _)| std::ptr::eq(other.buffer.raw(), output.raw()))
            })
        {
            return Err(Error::InvalidArgument(
                "PTQ1_0 concatenated matvec shapes differ, are misaligned, buffers are short, or aliased"
                    .into(),
            ));
        }
        let mut rows = [0_u32; 3];
        let mut groups = 0;
        let encoder = batch.encoder();
        if let Some((matrices, bf16_input)) = bf16 {
            let shape = [matrices[0].0.rows, matrices[0].0.columns];
            groups += 2 * shape[0] as usize;
            encoder.setComputePipelineState(&self.matvec_concat_bf16);
            unsafe {
                for (index, (matrix, output)) in matrices.iter().enumerate() {
                    bind(encoder, matrix.buffer, matrix.offset, 9 + 2 * index);
                    bind(encoder, output, 0, 10 + 2 * index);
                }
                bind(encoder, bf16_input, 0, 13);
                encoder.setBytes_length_atIndex(
                    NonNull::new_unchecked(shape.as_ptr().cast_mut().cast::<c_void>()),
                    size_of_val(&shape),
                    14,
                );
            }
        } else {
            encoder.setComputePipelineState(&self.matvec_concat);
        }
        unsafe {
            bind(encoder, input, 0, 1);
            set_u32(encoder, &first.columns, 4);
            // Unused segments have zero rows but still bind valid buffers.
            for (segment, (rows, (weights_index, output_index))) in
                rows.iter_mut().zip([(0, 2), (5, 6), (7, 8)]).enumerate()
            {
                let (matrix, output) = match projections.get(segment) {
                    Some(&projection) => {
                        *rows = projection.0.rows;
                        groups += (projection.0.rows as usize).div_ceil(8);
                        projection
                    }
                    None => (first, first_output),
                };
                bind(encoder, matrix.buffer, matrix.offset, weights_index);
                bind(encoder, output, 0, output_index);
            }
            encoder.setBytes_length_atIndex(
                NonNull::new_unchecked(rows.as_ptr().cast_mut().cast::<c_void>()),
                size_of_val(&rows),
                3,
            );
        }
        dispatch(encoder, groups, 32);
        batch.record_dispatch();
        Ok(())
    }

    /// RMS-normalize `tokens` rows of `input` (F32 `weights`, same width as the
    /// rotation) into `normalized`, and write their forward rotation to
    /// `output`, in one dispatch. Values are bitwise those of an `RMSNorm`
    /// followed by [`Self::transform`] with [`HadamardDirection::Forward`].
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn normalize_transform(
        &self,
        batch: &mut CommandBatch,
        rotation: &SignedHadamard,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        normalized: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        let elements = rotation
            .columns
            .checked_mul(tokens)
            .filter(|&count| count != 0)
            .ok_or_else(|| {
                Error::InvalidArgument("Bonsai activation shape overflow/empty".into())
            })?;
        let bytes = elements as usize * size_of::<f32>();
        if input.length() < bytes
            || normalized.length() < bytes
            || output.length() < bytes
            || weights.length() < rotation.columns as usize * size_of::<f32>()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || [normalized, output].iter().any(|out| {
                [input, weights]
                    .iter()
                    .any(|r| std::ptr::eq(out.raw(), r.raw()))
            })
            || std::ptr::eq(normalized.raw(), output.raw())
        {
            return Err(Error::InvalidArgument(
                "Bonsai normalize-rotate buffers are short, aliased, or epsilon invalid".into(),
            ));
        }
        let blocks_per_row = rotation.columns / HADAMARD_BLOCK_ELEMENTS as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.rms_forward);
        unsafe {
            bind(encoder, input, 0, 0);
            bind(encoder, weights, 0, 1);
            bind(encoder, normalized, 0, 2);
            bind(encoder, &rotation.signs, 0, 3);
            bind(encoder, output, 0, 4);
            set_u32(encoder, &blocks_per_row, 5);
            set_f32(encoder, &epsilon, 6);
        }
        dispatch(encoder, elements as usize / HADAMARD_BLOCK_ELEMENTS, 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Packed-weight projection of contiguous `[tokens, columns]` F32 rows.
    /// Output is `[tokens, rows]`. A single token retains the measured matvec;
    /// blocks of at least [`Self::large_batch_min`] rows use 64-token tiles
    /// that decode the weights once per tile; smaller blocks up to
    /// [`Self::small_batch_max`] stream the weights once per
    /// `SMALL_BATCH_WIDE_TOKENS` rows; larger blocks use the constructor-
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
        if tokens >= self.large_batch_min {
            // Whole 64-token tiles go to the large-batch kernel; a remainder
            // below the threshold costs less as small-batch passes than as a
            // mostly empty tile (a remainder of one keeps two rows so no
            // small-batch dispatch has one).
            let tail = match tokens % LARGE_BATCH_TILE as u32 {
                1 => 2,
                remainder if remainder < self.large_batch_min => remainder,
                _ => 0,
            };
            self.large_batch(batch, matrix, input, output, tokens - tail);
            if tail != 0 {
                self.small_batch(batch, matrix, input, output, tokens - tail, tail);
            }
            return Ok(());
        }
        if tokens <= self.small_batch_max {
            self.small_batch(batch, matrix, input, output, 0, tokens);
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

    /// One large-batch dispatch; the caller has validated extents and aliasing.
    #[allow(unsafe_code)]
    fn large_batch(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        tokens: u32,
    ) {
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.large_batch);
        unsafe {
            bind(encoder, matrix.buffer, matrix.offset, 0);
            bind(encoder, input, 0, 1);
            bind(encoder, output, 0, 2);
            set_u32(encoder, &matrix.rows, 3);
            set_u32(encoder, &matrix.columns, 4);
            set_u32(encoder, &tokens, 5);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (matrix.rows as usize).div_ceil(LARGE_BATCH_TILE),
                height: (tokens as usize).div_ceil(LARGE_BATCH_TILE),
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
    }

    /// Exact-token dispatches for activation rows `first..first + tokens`;
    /// the caller has validated extents and aliasing.
    /// Each dispatch reads the packed matrix once. Blocks of up to
    /// `SMALL_BATCH_KERNEL_TOKENS` rows use one scalar dispatch; larger blocks
    /// use wide dispatches of at most `SMALL_BATCH_WIDE_TOKENS` rows, plus one
    /// scalar dispatch for a remainder of one to four rows (a remainder of one
    /// borrows a row from the last wide chunk, so no dispatch has one row).
    #[allow(unsafe_code)]
    fn small_batch(
        &self,
        batch: &mut CommandBatch,
        matrix: Ptq1Matrix<'_>,
        input: &MetalBuffer,
        output: &MetalBuffer,
        first: u32,
        tokens: u32,
    ) {
        debug_assert!(tokens >= 2);
        let tail = match tokens % SMALL_BATCH_WIDE_TOKENS {
            _ if tokens <= SMALL_BATCH_KERNEL_TOKENS => tokens,
            1 => 2,
            remainder @ 2..=SMALL_BATCH_KERNEL_TOKENS => remainder,
            _ => 0,
        };
        let wide = tokens - tail;
        let chunks = wide.div_ceil(SMALL_BATCH_WIDE_TOKENS);
        let mut start = first;
        for chunk in 0..chunks {
            let end = first + (wide * (chunk + 1)).div_ceil(chunks);
            let count = end - start;
            let encoder = batch.encoder();
            encoder.setComputePipelineState(&self.small_batch_wide);
            unsafe {
                bind(encoder, matrix.buffer, matrix.offset, 0);
                bind(encoder, input, 0, 1);
                bind(encoder, output, 0, 2);
                set_u32(encoder, &matrix.rows, 3);
                set_u32(encoder, &matrix.columns, 4);
                set_u32(encoder, &start, 5);
                set_u32(encoder, &count, 6);
            }
            dispatch(encoder, (matrix.rows as usize).div_ceil(8), 128);
            batch.record_dispatch();
            start = end;
        }
        if tail != 0 {
            let encoder = batch.encoder();
            encoder.setComputePipelineState(&self.small_batch[tail as usize - 2]);
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

#[allow(unsafe_code, clippy::trivially_copy_pass_by_ref)]
unsafe fn set_f32(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: &f32,
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(value).cast_mut().cast::<c_void>()),
            size_of::<f32>(),
            index,
        );
    }
}
