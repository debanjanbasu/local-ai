//! Kernels of the Whisper large-v3 speech model (`shaders/whisper.metal`).
//!
//! Matrices keep the checkpoint's F16 values (widened to F32 exactly in the
//! kernel); activations, accumulation, `LayerNorm` and softmax are F32, like
//! the FP32 reference. Activations are row-major `[tokens, width]` F32 rows.
//! Linear layers take an optional bias, an optional exact GELU and an optional
//! residual added after the activation (which may be the output itself, for
//! `x += f(x)`).

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
type Encoder = ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>;

/// Column counts of [`WhisperKernels::gemm`] must be a multiple of this.
pub const GEMM_COLUMN_MULTIPLE: u32 = 32;
/// Column counts of [`WhisperKernels::gemv`] must be a multiple of this.
pub const GEMV_COLUMN_MULTIPLE: u32 = 128;
/// Head dimension of every attention kernel here.
pub const HEAD_DIM: u32 = 64;
/// Most keys [`WhisperKernels::decoder_attention`] attends.
pub const MAX_DECODER_KEYS: u32 = 1504;
const GEMM_TILE: usize = 32;

const FLAG_BIAS: u32 = 1;
const FLAG_GELU: u32 = 2;
const FLAG_RESIDUAL: u32 = 4;

/// Row-major F16 matrix at a byte offset inside a weight buffer, which must
/// stay unmodified until GPU completion.
#[derive(Clone, Copy)]
pub struct F16Rows<'a> {
    pub weights: &'a MetalBuffer,
    pub offset: usize,
    pub rows: u32,
    pub columns: u32,
}

/// An F32 buffer region: buffer and byte offset.
#[derive(Clone, Copy)]
pub struct Region<'a> {
    pub buffer: &'a MetalBuffer,
    pub offset: usize,
}

impl<'a> Region<'a> {
    #[must_use]
    pub const fn at(buffer: &'a MetalBuffer, offset: usize) -> Self {
        Self { buffer, offset }
    }

    #[must_use]
    pub const fn whole(buffer: &'a MetalBuffer) -> Self {
        Self { buffer, offset: 0 }
    }

    fn need(self, bytes: Option<usize>) -> crate::Result<()> {
        match bytes.and_then(|bytes| bytes.checked_add(self.offset)) {
            Some(end) if end <= self.buffer.length() && self.offset.is_multiple_of(4) => Ok(()),
            _ => Err(arg("whisper kernel buffer too short or misaligned")),
        }
    }
}

/// What a linear layer adds after the matrix product.
#[derive(Clone, Copy, Default)]
pub struct Epilogue<'a> {
    /// One F32 per output row.
    pub bias: Option<Region<'a>>,
    pub gelu: bool,
    /// Added after the activation, element for element (may be the output).
    pub residual: Option<Region<'a>>,
}

/// Decoder attention keys: the self-attention cache (causal: query `i` sees
/// keys `0 ..= base + i`) or every encoder frame.
#[derive(Clone, Copy)]
pub enum DecoderKeys {
    Causal { base: u32 },
    All { keys: u32 },
}

pub struct WhisperKernels {
    layernorm: Pipeline,
    im2col: Pipeline,
    gemm: Pipeline,
    gemv: Pipeline,
    encoder_attention: Pipeline,
    decoder_attention: Pipeline,
}

fn arg(message: &str) -> Error {
    Error::InvalidArgument(message.to_owned())
}

fn f32s(count: u64) -> Option<usize> {
    usize::try_from(count.checked_mul(4)?).ok()
}

#[allow(unsafe_code)]
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

#[allow(unsafe_code)]
fn bind(encoder: &Encoder, region: Region<'_>, index: usize) {
    // SAFETY: the buffer is retained by the caller until GPU completion.
    unsafe { encoder.setBuffer_offset_atIndex(Some(region.buffer.raw()), region.offset, index) };
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

fn overlaps(a: Region<'_>, a_len: usize, b: Region<'_>, b_len: usize) -> bool {
    std::ptr::eq(a.buffer.raw(), b.buffer.raw())
        && a.offset < b.offset.saturating_add(b_len)
        && b.offset < a.offset.saturating_add(a_len)
}

impl WhisperKernels {
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
            if pipeline.threadExecutionWidth() != 32 {
                return Err(arg("whisper pipeline needs 32-wide SIMD groups"));
            }
            Ok(pipeline)
        };
        let kernels = Self {
            layernorm: pipeline("wh_layernorm")?,
            im2col: pipeline("wh_im2col3")?,
            gemm: pipeline("wh_gemm")?,
            gemv: pipeline("wh_gemv")?,
            encoder_attention: pipeline("wh_encoder_attention")?,
            decoder_attention: pipeline("wh_decoder_attention")?,
        };
        for (pipeline, threads) in [
            (&kernels.layernorm, 256),
            (&kernels.im2col, 256),
            (&kernels.gemm, 128),
            (&kernels.gemv, 128),
            (&kernels.encoder_attention, 128),
            (&kernels.decoder_attention, 256),
        ] {
            if pipeline.maxTotalThreadsPerThreadgroup() < threads {
                return Err(arg("whisper pipeline cannot run its threadgroup size"));
            }
        }
        Ok(kernels)
    }

    /// `LayerNorm` (biased variance) of `rows` rows of `dimension`, with F32
    /// weight and bias. `output` may be `input`.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        &self,
        batch: &mut CommandBatch,
        input: Region<'_>,
        weight: Region<'_>,
        bias: Region<'_>,
        output: Region<'_>,
        rows: u32,
        dimension: u32,
        epsilon: f32,
    ) -> crate::Result<()> {
        if rows == 0 || dimension == 0 || epsilon.is_nan() || epsilon <= 0.0 {
            return Err(arg("invalid LayerNorm parameters"));
        }
        let bytes = f32s(u64::from(rows) * u64::from(dimension));
        input.need(bytes)?;
        output.need(bytes)?;
        weight.need(f32s(u64::from(dimension)))?;
        bias.need(f32s(u64::from(dimension)))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.layernorm);
        bind(encoder, input, 0);
        bind(encoder, weight, 1);
        bind(encoder, bias, 2);
        bind(encoder, output, 3);
        set(encoder, &dimension, 4);
        set(encoder, &epsilon, 5);
        grid(encoder, (rows as usize, 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    /// Columns of a kernel-3, padding-1 `Conv1d` over `[t_in, channels]`:
    /// `[t_out, channels * 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn im2col3(
        &self,
        batch: &mut CommandBatch,
        input: Region<'_>,
        output: Region<'_>,
        channels: u32,
        t_in: u32,
        t_out: u32,
        stride: u32,
    ) -> crate::Result<()> {
        let count = u64::from(t_out) * u64::from(channels) * 3;
        if channels == 0 || t_in == 0 || t_out == 0 || stride == 0 || count > u64::from(u32::MAX) {
            return Err(arg("invalid im2col parameters"));
        }
        input.need(f32s(u64::from(t_in) * u64::from(channels)))?;
        output.need(f32s(count))?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.im2col);
        bind(encoder, input, 0);
        bind(encoder, output, 1);
        set(encoder, &channels, 2);
        set(encoder, &t_in, 3);
        set(encoder, &t_out, 4);
        set(encoder, &stride, 5);
        grid(encoder, ((count as usize).div_ceil(256), 1), 256);
        batch.record_dispatch();
        Ok(())
    }

    fn epilogue_regions<'a>(
        epilogue: &Epilogue<'a>,
        rows: u32,
        out_len: usize,
        output: Region<'a>,
    ) -> crate::Result<(u32, Region<'a>, Region<'a>)> {
        let mut flags = 0;
        if let Some(bias) = epilogue.bias {
            bias.need(f32s(u64::from(rows)))?;
            if overlaps(bias, rows as usize * 4, output, out_len) {
                return Err(arg("linear output aliases its bias"));
            }
            flags |= FLAG_BIAS;
        }
        if epilogue.gelu {
            flags |= FLAG_GELU;
        }
        if let Some(residual) = epilogue.residual {
            residual.need(Some(out_len))?;
            if overlaps(residual, out_len, output, out_len)
                && (!std::ptr::eq(residual.buffer.raw(), output.buffer.raw())
                    || residual.offset != output.offset)
            {
                return Err(arg("linear residual partially overlaps its output"));
            }
            flags |= FLAG_RESIDUAL;
        }
        Ok((
            flags,
            epilogue.bias.unwrap_or(output),
            epilogue.residual.unwrap_or(output),
        ))
    }

    /// `output[t, r] = epilogue(Σ_c W[r, c] · input[t, c])` for `tokens`
    /// input rows; the output is `[tokens, rows]`.
    pub fn gemm(
        &self,
        batch: &mut CommandBatch,
        matrix: F16Rows<'_>,
        input: Region<'_>,
        output: Region<'_>,
        tokens: u32,
        epilogue: &Epilogue<'_>,
    ) -> crate::Result<()> {
        if tokens == 0
            || matrix.rows == 0
            || matrix.columns == 0
            || !matrix.columns.is_multiple_of(GEMM_COLUMN_MULTIPLE)
            || !matrix.offset.is_multiple_of(16)
            || !input.offset.is_multiple_of(16)
        {
            return Err(arg("invalid F16 GEMM shape"));
        }
        let (rows, columns) = (u64::from(matrix.rows), u64::from(matrix.columns));
        Region::at(matrix.weights, matrix.offset).need(usize::try_from(rows * columns * 2).ok())?;
        let in_len =
            f32s(u64::from(tokens) * columns).ok_or_else(|| arg("GEMM input size overflow"))?;
        let out_len =
            f32s(u64::from(tokens) * rows).ok_or_else(|| arg("GEMM output size overflow"))?;
        input.need(Some(in_len))?;
        output.need(Some(out_len))?;
        if overlaps(input, in_len, output, out_len)
            || std::ptr::eq(matrix.weights.raw(), output.buffer.raw())
        {
            return Err(arg("F16 GEMM output aliases an input"));
        }
        let (flags, bias, residual) =
            Self::epilogue_regions(epilogue, matrix.rows, out_len, output)?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gemm);
        bind(encoder, Region::at(matrix.weights, matrix.offset), 0);
        bind(encoder, input, 1);
        bind(encoder, output, 2);
        bind(encoder, bias, 3);
        bind(encoder, residual, 4);
        set(encoder, &matrix.rows, 5);
        set(encoder, &matrix.columns, 6);
        set(encoder, &tokens, 7);
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

    /// One-vector [`Self::gemm`]: `output[r] = epilogue(Σ_c W[r, c] · input[c])`.
    pub fn gemv(
        &self,
        batch: &mut CommandBatch,
        matrix: F16Rows<'_>,
        input: Region<'_>,
        output: Region<'_>,
        epilogue: &Epilogue<'_>,
    ) -> crate::Result<()> {
        if matrix.rows == 0
            || !matrix.columns.is_multiple_of(GEMV_COLUMN_MULTIPLE)
            || matrix.columns == 0
            || !matrix.offset.is_multiple_of(16)
            || !input.offset.is_multiple_of(16)
        {
            return Err(arg("invalid F16 GEMV shape"));
        }
        let (rows, columns) = (u64::from(matrix.rows), u64::from(matrix.columns));
        Region::at(matrix.weights, matrix.offset).need(usize::try_from(rows * columns * 2).ok())?;
        let in_len = f32s(columns).ok_or_else(|| arg("GEMV size overflow"))?;
        let out_len = f32s(rows).ok_or_else(|| arg("GEMV size overflow"))?;
        input.need(Some(in_len))?;
        output.need(Some(out_len))?;
        if overlaps(input, in_len, output, out_len)
            || std::ptr::eq(matrix.weights.raw(), output.buffer.raw())
        {
            return Err(arg("F16 GEMV output aliases an input"));
        }
        let (flags, bias, residual) =
            Self::epilogue_regions(epilogue, matrix.rows, out_len, output)?;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.gemv);
        bind(encoder, Region::at(matrix.weights, matrix.offset), 0);
        bind(encoder, input, 1);
        bind(encoder, output, 2);
        bind(encoder, bias, 3);
        bind(encoder, residual, 4);
        set(encoder, &matrix.rows, 5);
        set(encoder, &matrix.columns, 6);
        set(encoder, &flags, 7);
        grid(encoder, ((matrix.rows as usize).div_ceil(8), 1), 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Bidirectional encoder attention, head dim 64, over `tokens` rows of
    /// `[tokens, heads * 64]` q/k/v, for queries `queries.0 .. queries.0 +
    /// queries.1` (so one layer can be split across submissions).
    #[allow(clippy::too_many_arguments)]
    pub fn encoder_attention(
        &self,
        batch: &mut CommandBatch,
        q: Region<'_>,
        k: Region<'_>,
        v: Region<'_>,
        out: Region<'_>,
        tokens: u32,
        heads: u32,
        queries: (u32, u32),
    ) -> crate::Result<()> {
        let (start, count) = queries;
        if tokens == 0
            || heads == 0
            || count == 0
            || start.checked_add(count).is_none_or(|end| end > tokens)
        {
            return Err(arg("invalid encoder attention shape"));
        }
        let len = f32s(u64::from(tokens) * u64::from(heads) * u64::from(HEAD_DIM))
            .ok_or_else(|| arg("attention size overflow"))?;
        for region in [q, k, v, out] {
            region.need(Some(len))?;
        }
        if [q, k, v]
            .iter()
            .any(|input| overlaps(*input, len, out, len))
        {
            return Err(arg("attention output aliases an input"));
        }
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.encoder_attention);
        bind(encoder, q, 0);
        bind(encoder, k, 1);
        bind(encoder, v, 2);
        bind(encoder, out, 3);
        set(encoder, &tokens, 4);
        set(encoder, &heads, 5);
        set(encoder, &start, 6);
        set(encoder, &count, 7);
        set(encoder, &scale, 8);
        grid(
            encoder,
            ((count as usize).div_ceil(128), heads as usize),
            128,
        );
        batch.record_dispatch();
        Ok(())
    }

    /// Decoder attention for `queries` rows of `q` (`[queries, heads * 64]`)
    /// over `k`/`v` (`[keys, heads * 64]`), see [`DecoderKeys`].
    #[allow(clippy::too_many_arguments)]
    pub fn decoder_attention(
        &self,
        batch: &mut CommandBatch,
        q: Region<'_>,
        k: Region<'_>,
        v: Region<'_>,
        out: Region<'_>,
        queries: u32,
        heads: u32,
        keys: DecoderKeys,
    ) -> crate::Result<()> {
        let (base, causal, n_keys) = match keys {
            DecoderKeys::Causal { base } => (base, 1_u32, base.saturating_add(queries)),
            DecoderKeys::All { keys } => (0, 0, keys),
        };
        if queries == 0 || heads == 0 || n_keys == 0 || n_keys > MAX_DECODER_KEYS {
            return Err(arg("invalid decoder attention shape"));
        }
        let width = u64::from(heads) * u64::from(HEAD_DIM);
        let q_len = f32s(u64::from(queries) * width).ok_or_else(|| arg("size overflow"))?;
        let kv_len = f32s(u64::from(n_keys) * width).ok_or_else(|| arg("size overflow"))?;
        q.need(Some(q_len))?;
        out.need(Some(q_len))?;
        k.need(Some(kv_len))?;
        v.need(Some(kv_len))?;
        if overlaps(q, q_len, out, q_len)
            || overlaps(k, kv_len, out, q_len)
            || overlaps(v, kv_len, out, q_len)
        {
            return Err(arg("attention output aliases an input"));
        }
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.decoder_attention);
        bind(encoder, q, 0);
        bind(encoder, k, 1);
        bind(encoder, v, 2);
        bind(encoder, out, 3);
        set(encoder, &heads, 4);
        set(encoder, &base, 5);
        set(encoder, &causal, 6);
        set(encoder, &n_keys, 7);
        set(encoder, &scale, 8);
        grid(encoder, (queries as usize, heads as usize), 256);
        batch.record_dispatch();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
