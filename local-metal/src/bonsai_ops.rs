//! Checked graph operations for the fixed Qwen3.5 Bonsai profile.

#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::{
    Error, batch::CommandBatch, buffer::MetalBuffer, context::MetalContext, shaders::ShaderLibrary,
};

mod attention;
mod conv;
mod gdn;
pub use gdn::GdnStateFormat;
mod kv;
mod matmul;
mod norm;
mod rows;

pub const MODEL_WIDTH: u32 = 5120;
pub const Q_HEADS: u32 = 24;
pub const KV_HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 256;
pub const GDN_HEADS: u32 = 48;
pub const GDN_DIM: u32 = 128;
pub const MAX_PREFILL_TOKENS: u32 = 128;
const SPLIT: u32 = 128;
/// Tokens per threadgroup of `bo_attn_split_tensor` (a multiple of its
/// 64-key tile). Short prefixes use the small split so the four KV heads
/// still yield enough threadgroups to fill the GPU; long prefixes use the
/// large one.
const SPLIT_TENSOR_SHORT: u32 = 64;
const SPLIT_TENSOR_LONG: u32 = 256;
const SPLIT_TENSOR_LONG_MIN_PREFIX: u32 = 4096;
/// Below this prefix the SIMD split kernel is used even on tensor builds: a
/// handful of tensor threadgroups walking one tile each has worse latency
/// than 24+ SIMD groups, which showed as a 5 % short-prompt MTP decode loss.
const SPLIT_TENSOR_MIN_PREFIX: u32 = 1024;
/// Query heads served by one SIMD group of `bo_attn_split`; must match the
/// shader's `BONSAI_SPLIT_HEADS` and divide the six heads per KV head.
const SPLIT_HEADS: u32 = 2;
/// Blocks of at most this many rows whose causal prefix ends past
/// `ROW_BLOCK_MIN_PREFIX` attend row by row through the split kernel (one
/// dispatch pair per row) rather than the block kernels, which would leave
/// most of their eight-row tiles idle while walking a long prefix.
const ROW_BLOCK_TOKENS: u32 = 8;
const ROW_BLOCK_MIN_PREFIX: u32 = 1024;
/// `bo_bf16_mm` tile geometry; must match the shader constants.
const BF16_TILE_ROWS: u32 = 32;
const BF16_TILE_TOKENS: u32 = 32;
const BF16_TILE_COLUMNS: u32 = 64;
/// Blocks with fewer tokens (decode, draft and verify rows) keep the per-token
/// matvec; from here the tiled GEMM reads the matrix at most once per 32 rows.
pub const BF16_TILE_MIN_TOKENS: u32 = 8;

/// Full-attention prefill operands; all variants keep F32 softmax statistics,
/// accumulation and output, F16 KV storage, and the same F32 decode path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernel {
    SimdF32,
    TensorF32,
}

impl AttentionKernel {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SimdF32 => "simd_f32",
            Self::TensorF32 => "tensor_f32",
        }
    }

    const fn threads(self) -> usize {
        match self {
            Self::SimdF32 => 256,
            Self::TensorF32 => 128,
        }
    }
}

#[derive(Clone, Copy)]
pub struct RmsNormParams {
    pub dimension: u32,
    pub rows: u32,
    pub stride: u32,
    pub epsilon: f32,
    pub weight_offset: usize,
}
#[derive(Clone, Copy)]
pub struct Bf16Matrix<'a> {
    pub buffer: &'a MetalBuffer,
    pub offset: usize,
    pub rows: u32,
    pub columns: u32,
}

pub struct AttentionWorkspace {
    partials: MetalBuffer,
    max_context: u32,
    bytes: usize,
}
impl AttentionWorkspace {
    pub fn new(context: &MetalContext, max_context: u32) -> crate::Result<Self> {
        if max_context == 0 || max_context > 262_144 {
            return Err(arg("attention context must be within 1..=262144"));
        }
        // Short tensor splits (prefixes 1,024..4,095) need more records than
        // the 128-token SIMD splits a short context implies.
        let tensor_short = if max_context >= SPLIT_TENSOR_MIN_PREFIX {
            max_context
                .min(SPLIT_TENSOR_LONG_MIN_PREFIX - 1)
                .div_ceil(SPLIT_TENSOR_SHORT)
        } else {
            0
        };
        let splits = max_context.div_ceil(SPLIT).max(tensor_short) as usize;
        let bytes = splits
            .checked_mul(Q_HEADS as usize)
            .and_then(|n| n.checked_mul(258 * 4))
            .ok_or_else(|| arg("workspace size overflow"))?;
        Ok(Self {
            partials: MetalBuffer::empty(context.device(), bytes)?,
            max_context,
            bytes,
        })
    }
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.bytes
    }
    #[must_use]
    pub const fn max_context(&self) -> u32 {
        self.max_context
    }
}

/// Storage format of one K or V cache.
///
/// Every format stores 4 KV heads x 256 dims per token; the quantized ones
/// use 32-value blocks with an F16 absmax scale each (see `bonsai_ops.metal`,
/// `bonsai_kv_store`). Rows are quantized once when written and dequantized
/// in registers on every read, so the attention math itself is unchanged;
/// only the stored operands are rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// 16 bits per value; the historical layout, bitwise unchanged.
    F16,
    /// int8 + 1/32 F16 scale: 8.5 bits per value (1.88x smaller than F16).
    Q8,
}

impl KvFormat {
    #[must_use]
    pub const fn token_bytes(self) -> usize {
        match self {
            Self::F16 => 4 * 256 * 2,
            Self::Q8 => 4 * 256 + 4 * 8 * 2,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
        }
    }

    /// Code passed to the kernels (`BONSAI_KV_*` in the shader).
    const fn code(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::Q8 => 1,
        }
    }

    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "f16" => Some(Self::F16),
            "q8" => Some(Self::Q8),
            _ => None,
        }
    }
}

/// Key and value cache formats. Production layouts use the same format for both;
/// the pair remains explicit because prompt-cache descriptors persist it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayout {
    pub key: KvFormat,
    pub value: KvFormat,
}

impl KvLayout {
    pub const F16: Self = Self {
        key: KvFormat::F16,
        value: KvFormat::F16,
    };
    pub const Q8: Self = Self {
        key: KvFormat::Q8,
        value: KvFormat::Q8,
    };

    /// Bytes per cached token across both caches.
    #[must_use]
    pub const fn token_bytes(self) -> usize {
        self.key.token_bytes() + self.value.token_bytes()
    }

    #[must_use]
    pub const fn is_f16(self) -> bool {
        matches!(self.key, KvFormat::F16) && matches!(self.value, KvFormat::F16)
    }

    /// `f16` or `q8`, applied to both caches.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let format = KvFormat::parse(text)?;
        Some(Self {
            key: format,
            value: format,
        })
    }

    #[must_use]
    pub fn name(self) -> String {
        debug_assert_eq!(self.key, self.value);
        self.key.name().to_owned()
    }

    const fn codes(self) -> [u32; 2] {
        [self.key.code(), self.value.code()]
    }
}

/// Quantized layouts with tensor attention kernels; each names its
/// `bo_attn_split_tensor_*` and `bo_attn_tensor_*` instantiations.
const QUANTIZED_TENSOR_LAYOUTS: [KvLayout; 1] = [KvLayout::Q8];

pub struct BonsaiOps {
    p: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    attention_kernel: AttentionKernel,
    /// Quantized layout -> index in `p` of its tensor decode-split pipeline;
    /// its causal prefill pipeline follows at the next index.
    quantized_tensor: Vec<(KvLayout, usize)>,
}
impl BonsaiOps {
    pub fn new(context: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        Self::new_with_attention_kernel(context, shaders, AttentionKernel::TensorF32).or_else(
            |_| Self::new_with_attention_kernel(context, shaders, AttentionKernel::SimdF32),
        )
    }

    /// Select an exact implementation for numerical/performance comparisons.
    /// Unlike automatic selection, an unavailable explicit kernel is an error.
    pub fn new_with_attention_kernel(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        attention_kernel: AttentionKernel,
    ) -> crate::Result<Self> {
        let names = [
            "bo_rms",
            "bo_add",
            "bo_swiglu",
            "bo_bf16_mv",
            "bo_conv",
            "bo_l2_qk",
            "bo_decay",
            "bo_gdn",
            "bo_gdn_post",
            "bo_attn_prep",
            "bo_attn_split",
            "bo_attn_reduce",
            "bo_sigmoid_mul",
            match attention_kernel {
                AttentionKernel::SimdF32 => "bo_attn_block",
                AttentionKernel::TensorF32 => "bo_attn_tensor",
            },
            "bo_gdn_rows_4",
            "bo_kv_prep",
            "bo_bf16_mm",
            // Quantized caches without a tensor kernel of their own take the
            // SIMD block kernel, which dequantizes in registers.
            "bo_attn_block",
            // Decode over an F16 cache on the tensor units (six GQA heads as
            // one Q tile); the SIMD build keeps the split kernel here.
            match attention_kernel {
                AttentionKernel::SimdF32 => "bo_attn_split",
                AttentionKernel::TensorF32 => "bo_attn_split_tensor",
            },
            "bo_conv_l2_decay",
            "bo_attn_unrotate",
            // Reduced-precision recurrent state (see `GdnStateFormat`): the
            // token-at-a-time and four-row kernels for F16, then BF16.
            "bo_gdn_f16",
            "bo_gdn_rows_4_f16",
            "bo_gdn_bf16",
            "bo_gdn_rows_4_bf16",
        ];
        // Quantized layouts with tensor kernels of their own (decode split and
        // causal prefill), each K/V tile dequantized to half in threadgroup
        // memory. Their pipelines follow the named ones in `p`.
        let quantized: &[KvLayout] = if attention_kernel == AttentionKernel::TensorF32 {
            &QUANTIZED_TENSOR_LAYOUTS
        } else {
            &[]
        };
        let quantized_names = quantized
            .iter()
            .flat_map(|layout| {
                let suffix = layout.name();
                [
                    format!("bo_attn_split_tensor_{suffix}"),
                    format!("bo_attn_tensor_{suffix}"),
                ]
            })
            .collect::<Vec<_>>();
        let mut p = Vec::with_capacity(names.len() + quantized_names.len());
        for name in names
            .into_iter()
            .chain(quantized_names.iter().map(String::as_str))
        {
            let f = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| Error::PipelineCreation(e.to_string()))?;
            let threads = if name.contains("tensor") {
                attention_kernel.threads()
            } else {
                256
            };
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < threads
            {
                return Err(arg(
                    "Bonsai pipeline cannot support the required SIMD/threadgroup size",
                ));
            }
            p.push(pipeline);
        }
        let quantized_tensor = quantized
            .iter()
            .enumerate()
            .map(|(index, &layout)| (layout, names.len() + 2 * index))
            .collect();
        Ok(Self {
            p,
            attention_kernel,
            quantized_tensor,
        })
    }

    #[must_use]
    pub const fn attention_kernel(&self) -> AttentionKernel {
        self.attention_kernel
    }

    /// Index of `layout`'s quantized tensor decode-split pipeline, if it has one.
    fn quantized_tensor(&self, layout: KvLayout) -> Option<usize> {
        self.quantized_tensor
            .iter()
            .find(|(candidate, _)| *candidate == layout)
            .map(|&(_, index)| index)
    }

    #[allow(unsafe_code)]
    fn go(
        &self,
        b: &mut CommandBatch,
        pi: usize,
        bufs: &[(&MetalBuffer, usize)],
        u: &[u32],
        f: &[f32],
        groups: usize,
        threads: usize,
    ) {
        let e = b.encoder();
        e.setComputePipelineState(&self.p[pi]);
        unsafe {
            for (i, (v, o)) in bufs.iter().enumerate() {
                e.setBuffer_offset_atIndex(Some(v.raw()), *o, i);
            }
            let mut ix = bufs.len();
            for v in u {
                scalar(e, v, ix);
                ix += 1;
            }
            for v in f {
                scalar(e, v, ix);
                ix += 1;
            }
        }
        e.dispatchThreadgroups_threadsPerThreadgroup(
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
        b.record_dispatch();
    }
}
fn arg(s: &str) -> Error {
    Error::InvalidArgument(s.into())
}
fn matrix_bytes(rows: u32, columns: u32, element_size: usize) -> crate::Result<usize> {
    (rows as usize)
        .checked_mul(columns as usize)
        .and_then(|count| count.checked_mul(element_size))
        .ok_or_else(|| arg("matrix extent overflow"))
}
fn sequence_bytes(tokens: u32, width: u32) -> crate::Result<usize> {
    if !(1..=MAX_PREFILL_TOKENS).contains(&tokens) {
        return Err(arg("Bonsai token block must be within 1..=128"));
    }
    matrix_bytes(tokens, width, 4)
}
fn need(b: &MetalBuffer, n: usize) -> crate::Result<()> {
    if b.length() < n {
        Err(arg("buffer is too short"))
    } else {
        Ok(())
    }
}
fn same(a: &MetalBuffer, b: &MetalBuffer) -> bool {
    std::ptr::eq(a.raw(), b.raw())
}
fn no_alias(out: &MetalBuffer, x: &[&MetalBuffer]) -> crate::Result<()> {
    if x.iter().any(|v| same(out, v)) {
        Err(arg("writable buffer aliases input"))
    } else {
        Ok(())
    }
}
fn trio(a: &MetalBuffer, b: &MetalBuffer, o: &MetalBuffer, n: u32) -> crate::Result<()> {
    if n == 0 {
        return Err(arg("element count is zero"));
    }
    need(a, n as usize * 4)?;
    need(b, n as usize * 4)?;
    need(o, n as usize * 4)?;
    no_alias(o, &[a, b])
}
#[allow(unsafe_code)]
unsafe fn scalar<T>(e: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: &T, i: usize) {
    unsafe {
        e.setBytes_length_atIndex(
            NonNull::new_unchecked(std::ptr::from_ref(v).cast_mut().cast::<c_void>()),
            size_of::<T>(),
            i,
        );
    }
}
