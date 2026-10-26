//! Bounded GPU top-k selection for sampling.

use core::ffi::c_void;
use std::ptr::NonNull;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::Error;
use crate::batch::CommandBatch;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

pub const MAX_TOP_K: usize = 64;
/// Logits each first-stage threadgroup of the top-k kernels reduces.
const TOPK_GROUP: usize = 4096;
/// Candidates the second stage can hold: 256 threads, 16 each.
const TOPK_FINAL_CANDIDATES: usize = 256 * 16;

/// One selected logit. The logit's exact bit pattern is preserved.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct TopKCandidate {
    pub token_id: u32,
    pub logit: f32,
}

impl PartialEq for TopKCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.token_id == other.token_id && self.logit.to_bits() == other.logit.to_bits()
    }
}

/// The two top-k pipelines: each group of 4,096 logits of a row selects its
/// best k, then each row merges its groups' candidates.
struct TopKPipelines {
    partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    finish: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl TopKPipelines {
    fn new(ctx: &MetalContext, shaders: &ShaderLibrary) -> crate::Result<Self> {
        let make_pipeline = |name: &str| {
            let function = shaders.get_function(name)?;
            ctx.device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))
        };
        Ok(Self {
            partial: make_pipeline("topk_rows_partial")?,
            finish: make_pipeline("topk_rows_final")?,
        })
    }

    /// Encode the best `k` of each of `rows` rows of `vocab` logits, `stride`
    /// floats apart from `offset` bytes into `logits`, into `MAX_TOP_K`
    /// slots per row of `output`. Bounds are the caller's.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    fn encode(
        &self,
        batch: &mut CommandBatch,
        logits: (&MetalBuffer, usize),
        vocab: usize,
        stride: usize,
        rows: usize,
        k: usize,
        partials: &MetalBuffer,
        output: &MetalBuffer,
    ) {
        let groups = vocab.div_ceil(TOPK_GROUP);
        let (vocab, k, stride) = (vocab as u32, k as u32, stride as u32);
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.partial);
        unsafe {
            bind(encoder, logits.0, logits.1, 0);
            bind(encoder, partials, 0, 1);
            set_u32(encoder, &vocab, 2);
            set_u32(encoder, &k, 3);
            set_u32(encoder, &stride, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        let groups = groups as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.finish);
        unsafe {
            bind(encoder, partials, 0, 0);
            bind(encoder, output, 0, 1);
            set_u32(encoder, &groups, 2);
            set_u32(encoder, &k, 3);
        }
        dispatch(encoder, rows, 256);
        batch.record_dispatch();
    }
}

/// The partial candidates of `rows` rows of up to `vocab` logits.
const fn partials_bytes(vocab: usize, rows: usize) -> usize {
    rows * vocab.div_ceil(TOPK_GROUP) * MAX_TOP_K * size_of::<TopKCandidate>()
}

/// Whether the second stage can merge every group of a `vocab`-wide row.
const fn mergeable(vocab: usize) -> bool {
    vocab.div_ceil(TOPK_GROUP) * MAX_TOP_K <= TOPK_FINAL_CANDIDATES
}

/// Reusable workspace for top-k selection.
///
/// `encode` only records GPU work. The caller must complete/wait for the containing
/// command batch before reading `candidates`, and must not overlap another encode
/// using this workspace with unfinished work from an earlier encode.
pub struct GpuTopK {
    pipelines: TopKPipelines,
    partials: MetalBuffer,
    output: MetalBuffer,
    max_vocab: usize,
}

impl GpuTopK {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        max_vocab: usize,
    ) -> crate::Result<Self> {
        if max_vocab == 0 || !mergeable(max_vocab) {
            return Err(Error::InvalidArgument(
                "top-k max_vocab must be in 1..=262144".into(),
            ));
        }
        Ok(Self {
            pipelines: TopKPipelines::new(ctx, shaders)?,
            partials: MetalBuffer::empty(ctx.device(), partials_bytes(max_vocab, 1))?,
            output: MetalBuffer::empty(ctx.device(), MAX_TOP_K * size_of::<TopKCandidate>())?,
            max_vocab,
        })
    }

    /// Encode selection in two dispatches without allocating or synchronizing.
    pub fn encode(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        logits_offset: usize,
        vocab: usize,
        k: usize,
    ) -> crate::Result<()> {
        if !(1..=MAX_TOP_K).contains(&k) {
            return Err(Error::InvalidArgument("top-k k must be in 1..=64".into()));
        }
        if vocab == 0 || vocab > self.max_vocab {
            return Err(Error::InvalidArgument(
                "top-k vocab is empty or exceeds allocated limit".into(),
            ));
        }
        if k > vocab {
            return Err(Error::InvalidArgument("top-k k exceeds vocab".into()));
        }
        let required = vocab
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::InvalidArgument("top-k logit size overflow".into()))?;
        let end = logits_offset
            .checked_add(required)
            .ok_or_else(|| Error::InvalidArgument("top-k logit offset overflow".into()))?;
        if !logits_offset.is_multiple_of(size_of::<f32>()) || logits.length() < end {
            return Err(Error::InvalidArgument(
                "top-k logits offset is unaligned or exceeds the buffer".into(),
            ));
        }
        self.pipelines.encode(
            batch,
            (logits, logits_offset),
            vocab,
            vocab,
            1,
            k,
            &self.partials,
            &self.output,
        );
        Ok(())
    }

    /// Read the first `k` results after GPU completion.
    ///
    /// # Panics
    ///
    /// Panics when `k` exceeds [`MAX_TOP_K`].
    #[must_use]
    pub fn candidates(&self, k: usize) -> &[TopKCandidate] {
        assert!(k <= MAX_TOP_K, "candidate count exceeds maximum top-k");
        &self.output.as_slice()[..k]
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.partials.length() + self.output.length()
    }

    #[must_use]
    pub const fn maximum_supported_k(&self) -> usize {
        MAX_TOP_K
    }
}

/// The best `k` (at most [`MAX_TOP_K`]) logits of each of up to `max_rows`
/// consecutive logit rows, selected as [`GpuTopK`] selects one row, so a
/// block can select every row it produces inside its own command buffer.
///
/// Same encoding contract as [`GpuTopK`].
pub struct TopKRows {
    pipelines: TopKPipelines,
    partials: MetalBuffer,
    output: MetalBuffer,
    vocab: usize,
    max_rows: usize,
}

impl TopKRows {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        vocab: usize,
        max_rows: usize,
    ) -> crate::Result<Self> {
        if vocab == 0 || !mergeable(vocab) || max_rows == 0 {
            return Err(Error::InvalidArgument(
                "top-k rows need a vocabulary of 1..=262144 tokens and one row".into(),
            ));
        }
        Ok(Self {
            pipelines: TopKPipelines::new(ctx, shaders)?,
            partials: MetalBuffer::empty(ctx.device(), partials_bytes(vocab, max_rows))?,
            output: MetalBuffer::empty(
                ctx.device(),
                max_rows * MAX_TOP_K * size_of::<TopKCandidate>(),
            )?,
            vocab,
            max_rows,
        })
    }

    /// Encode the best `k` logits of each of the first `rows` rows of `logits`.
    pub fn encode(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        rows: usize,
        k: usize,
    ) -> crate::Result<()> {
        if rows == 0
            || rows > self.max_rows
            || !(1..=MAX_TOP_K.min(self.vocab)).contains(&k)
            || logits.length() < rows * self.vocab * size_of::<f32>()
        {
            return Err(Error::InvalidArgument(
                "top-k rows exceed the workspace or the logits".into(),
            ));
        }
        self.pipelines.encode(
            batch,
            (logits, 0),
            self.vocab,
            self.vocab,
            rows,
            k,
            &self.partials,
            &self.output,
        );
        Ok(())
    }

    /// Row `row`'s best `k` logits, best first, copied out of the last
    /// completed [`Self::encode`] (which must have selected at least `k`).
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the workspace or `k` exceeds [`MAX_TOP_K`].
    #[must_use]
    pub fn row(&self, row: usize, k: usize) -> Vec<TopKCandidate> {
        assert!(
            row < self.max_rows && k <= MAX_TOP_K,
            "top-k row or count exceeds the workspace"
        );
        let start = row * MAX_TOP_K;
        self.output.as_slice::<TopKCandidate>()[start..start + k].to_vec()
    }

    #[must_use]
    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }
}

/// Most alternatives [`LogprobRows`] reports per row: `OpenAI`'s `top_logprobs` limit.
pub const MAX_TOP_LOGPROBS: usize = 20;
/// Logits each first-stage threadgroup of [`LogprobRows`] reduces.
const LOGPROB_GROUP: usize = 4096;
/// Candidates the second stage can hold: 256 threads, 8 each.
const LOGPROB_FINAL_CANDIDATES: usize = 256 * 8;

/// Raw log-probability statistics of logit rows: each row's log-sum-exp and
/// its best `k` logits, selected in the order of [`GpuTopK`].
///
/// Like the other kernel sets here, `encode` only records work, and only reads
/// the logits; results are readable once the containing batch completes, and
/// two batches using this workspace must not overlap.
pub struct LogprobRows {
    partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    finish: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    stats: MetalBuffer,
    partials: MetalBuffer,
    lse: MetalBuffer,
    top: MetalBuffer,
    vocab: usize,
    max_rows: usize,
}

impl LogprobRows {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        vocab: usize,
        max_rows: usize,
    ) -> crate::Result<Self> {
        let groups = vocab.div_ceil(LOGPROB_GROUP);
        if vocab == 0 || vocab >= u32::MAX as usize || max_rows == 0 {
            return Err(Error::InvalidArgument(
                "log-probability rows need a vocabulary and one row".into(),
            ));
        }
        if groups * MAX_TOP_LOGPROBS > LOGPROB_FINAL_CANDIDATES {
            return Err(Error::InvalidArgument(
                "vocabulary too large for log-probability selection".into(),
            ));
        }
        let make_pipeline = |name: &str| {
            let function = shaders.get_function(name)?;
            ctx.device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))
        };
        let candidate = size_of::<TopKCandidate>();
        Ok(Self {
            partial: make_pipeline("logprob_rows_partial")?,
            finish: make_pipeline("logprob_rows_final")?,
            stats: MetalBuffer::empty(ctx.device(), max_rows * groups * 2 * size_of::<f32>())?,
            partials: MetalBuffer::empty(
                ctx.device(),
                max_rows * groups * MAX_TOP_LOGPROBS * candidate,
            )?,
            lse: MetalBuffer::empty(ctx.device(), max_rows * size_of::<f32>())?,
            top: MetalBuffer::empty(ctx.device(), max_rows * MAX_TOP_LOGPROBS * candidate)?,
            vocab,
            max_rows,
        })
    }

    /// Encode the statistics of the first `rows` rows of `logits`, with the
    /// best `k` (at most [`MAX_TOP_LOGPROBS`]) logits of each.
    #[allow(unsafe_code)]
    pub fn encode(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        rows: usize,
        k: usize,
    ) -> crate::Result<()> {
        if rows == 0
            || rows > self.max_rows
            || k > MAX_TOP_LOGPROBS
            || logits.length() < rows * self.vocab * size_of::<f32>()
        {
            return Err(Error::InvalidArgument(
                "log-probability rows exceed the workspace or the logits".into(),
            ));
        }
        let groups = self.vocab.div_ceil(LOGPROB_GROUP);
        let vocab = self.vocab as u32;
        let k = k as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.partial);
        unsafe {
            bind(encoder, logits, 0, 0);
            bind(encoder, &self.stats, 0, 1);
            bind(encoder, &self.partials, 0, 2);
            set_u32(encoder, &vocab, 3);
            set_u32(encoder, &k, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: rows,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        batch.record_dispatch();
        let groups = groups as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.finish);
        unsafe {
            bind(encoder, &self.stats, 0, 0);
            bind(encoder, &self.partials, 0, 1);
            bind(encoder, &self.lse, 0, 2);
            bind(encoder, &self.top, 0, 3);
            set_u32(encoder, &groups, 4);
            set_u32(encoder, &k, 5);
        }
        dispatch(encoder, rows, 256);
        batch.record_dispatch();
        Ok(())
    }

    /// Row `row`'s log-sum-exp, from the last completed [`Self::encode`].
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the workspace.
    #[must_use]
    pub fn log_sum_exp(&self, row: usize) -> f32 {
        self.lse.as_slice::<f32>()[row]
    }

    /// Row `row`'s best `k` logits, best first, from the last completed
    /// [`Self::encode`] (which must have selected at least `k`).
    ///
    /// # Panics
    ///
    /// Panics when `row` is outside the workspace or `k` exceeds [`MAX_TOP_LOGPROBS`].
    #[must_use]
    pub fn top(&self, row: usize, k: usize) -> &[TopKCandidate] {
        assert!(
            k <= MAX_TOP_LOGPROBS,
            "log-probability count exceeds the maximum"
        );
        let start = row * MAX_TOP_LOGPROBS;
        &self.top.as_slice::<TopKCandidate>()[start..start + k]
    }

    #[must_use]
    pub const fn max_rows(&self) -> usize {
        self.max_rows
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
