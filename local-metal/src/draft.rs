//! GPU-resident greedy drafting primitives.
//!
//! A chained draft step reads its input token from a device buffer the previous
//! step wrote, so several draft steps fit one command buffer with no CPU round
//! trip: [`DraftKernels::embed_inverse`] decodes and inverse-rotates the token's
//! `PTQ1_0` embedding row, and [`DraftKernels::top_two`] selects the next token
//! (and the runner-up, for the caller's confidence gate) from the draft logits.

use core::ffi::c_void;
use std::ptr::NonNull;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use crate::Error;
use crate::batch::CommandBatch;
use crate::bonsai::{HADAMARD_BLOCK_ELEMENTS, Ptq1Matrix, SignedHadamard};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

/// Logits each first-stage threadgroup reduces.
const GROUP_ELEMENTS: usize = 1024;
/// Bytes of one partial top-two (two ids, two logits).
const PARTIAL_BYTES: usize = 16;

/// The two best logits of one draft step in `f32::total_cmp` order, ties to
/// the lower token id: the same selection as the GPU top-k and `top_two`.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct DraftTopTwo {
    pub best_id: u32,
    pub best_logit: f32,
    pub second_id: u32,
    pub second_logit: f32,
}

/// Pipelines plus the reusable first-stage workspace for one vocabulary size.
///
/// Like every kernel set here, encoding only records work; the caller must
/// complete the batch before reading results, and must not overlap two batches
/// that use this workspace.
pub struct DraftKernels {
    embed_inverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    finish: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partials: MetalBuffer,
    max_vocab: usize,
}

impl DraftKernels {
    pub fn new(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        max_vocab: usize,
    ) -> crate::Result<Self> {
        if max_vocab < 2 || max_vocab >= u32::MAX as usize {
            return Err(Error::InvalidArgument(
                "draft vocabulary must hold at least two tokens and fit u32".into(),
            ));
        }
        let pipeline = |name: &str| {
            let function = shaders.get_function(name)?;
            let pipeline = context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < 256
            {
                return Err(Error::InvalidArgument(
                    "draft pipeline cannot support the required SIMD/threadgroup size".into(),
                ));
            }
            Ok(pipeline)
        };
        Ok(Self {
            embed_inverse: pipeline("draft_embed_inverse")?,
            partial: pipeline("draft_top2_partial")?,
            finish: pipeline("draft_top2_final")?,
            partials: MetalBuffer::empty(
                context.device(),
                max_vocab.div_ceil(GROUP_ELEMENTS) * PARTIAL_BYTES,
            )?,
            max_vocab,
        })
    }

    /// Decode row `tokens[token_index]` of the packed `table` and inverse-rotate
    /// it into `output`: bit-identical to `decode_ptq1_row` followed by an
    /// inverse [`crate::bonsai::BonsaiKernels::transform`]. A token beyond the
    /// table produces a zero row.
    #[allow(unsafe_code)]
    pub fn embed_inverse(
        &self,
        batch: &mut CommandBatch,
        table: Ptq1Matrix<'_>,
        rotation: &SignedHadamard,
        tokens: &MetalBuffer,
        token_index: u32,
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        if table.columns != rotation.columns
            || !(table.columns as usize).is_multiple_of(HADAMARD_BLOCK_ELEMENTS)
            || output.length() < table.columns as usize * size_of::<f32>()
            || tokens.length() <= token_index as usize * size_of::<u32>()
            || std::ptr::eq(output.raw(), table.buffer.raw())
            || std::ptr::eq(output.raw(), tokens.raw())
        {
            return Err(Error::InvalidArgument(
                "draft embedding shape, token slot or output is invalid".into(),
            ));
        }
        let blocks_per_row = table.columns / HADAMARD_BLOCK_ELEMENTS as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.embed_inverse);
        unsafe {
            bind(encoder, table.buffer, table.offset, 0);
            bind(encoder, tokens, 0, 1);
            bind(encoder, &rotation.signs, 0, 2);
            bind(encoder, output, 0, 3);
            set_u32(encoder, &token_index, 4);
            set_u32(encoder, &table.rows, 5);
            set_u32(encoder, &blocks_per_row, 6);
        }
        dispatch(encoder, blocks_per_row as usize, 128);
        batch.record_dispatch();
        Ok(())
    }

    /// Exact top two of `logits[..vocab]`: the winner goes to
    /// `tokens[token_index]` for the next chained step and the pair to
    /// `results[result_index]`.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn top_two(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        vocab: usize,
        tokens: &MetalBuffer,
        token_index: u32,
        results: &MetalBuffer,
        result_index: u32,
    ) -> crate::Result<()> {
        if !(2..=self.max_vocab).contains(&vocab)
            || logits.length() < vocab * size_of::<f32>()
            || tokens.length() <= token_index as usize * size_of::<u32>()
            || results.length() < (result_index as usize + 1) * size_of::<DraftTopTwo>()
            || std::ptr::eq(tokens.raw(), results.raw())
        {
            return Err(Error::InvalidArgument(
                "draft top-two vocabulary or result slot is invalid".into(),
            ));
        }
        let groups = vocab.div_ceil(GROUP_ELEMENTS);
        let vocab = vocab as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.partial);
        unsafe {
            bind(encoder, logits, 0, 0);
            bind(encoder, &self.partials, 0, 1);
            set_u32(encoder, &vocab, 2);
        }
        dispatch(encoder, groups, 256);
        batch.record_dispatch();

        let groups = groups as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.finish);
        unsafe {
            bind(encoder, &self.partials, 0, 0);
            bind(encoder, tokens, 0, 1);
            bind(encoder, results, 0, 2);
            set_u32(encoder, &groups, 3);
            set_u32(encoder, &token_index, 4);
            set_u32(encoder, &result_index, 5);
        }
        dispatch(encoder, 1, 256);
        batch.record_dispatch();
        Ok(())
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.partials.length()
    }
}

/// One row's greedy selection: the `f32::total_cmp` maximum (ties to the
/// lower id) and whether any logit of the row is non-finite.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct GreedyRowResult {
    pub best_id: u32,
    pub best_logit: f32,
    pub nonfinite: u32,
    padding: u32,
}

/// Greedy selection over up to `max_rows` consecutive logit rows.
///
/// Two dispatches, so a verify block or a decode step returns token ids
/// rather than a host scan (or one GPU top-k submission) per row. Same
/// encoding contract as [`DraftKernels`].
pub struct GreedyRows {
    partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    finish: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partials: MetalBuffer,
    flags: MetalBuffer,
    results: MetalBuffer,
    max_rows: usize,
    vocab: usize,
}

impl GreedyRows {
    pub fn new(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        vocab: usize,
        max_rows: usize,
    ) -> crate::Result<Self> {
        if !(2..u32::MAX as usize).contains(&vocab) || max_rows == 0 {
            return Err(Error::InvalidArgument(
                "greedy rows need a vocabulary of at least two tokens and one row".into(),
            ));
        }
        let pipeline = |name: &str| {
            let function = shaders.get_function(name)?;
            context
                .device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))
        };
        let slots = max_rows * vocab.div_ceil(GROUP_ELEMENTS);
        Ok(Self {
            partial: pipeline("greedy_rows_partial")?,
            finish: pipeline("greedy_rows_final")?,
            partials: MetalBuffer::empty(context.device(), slots * PARTIAL_BYTES)?,
            flags: MetalBuffer::empty(context.device(), slots * size_of::<u32>())?,
            results: MetalBuffer::empty(context.device(), max_rows * size_of::<GreedyRowResult>())?,
            max_rows,
            vocab,
        })
    }

    /// Select each of the first `rows` rows of `logits` into [`Self::results`].
    #[allow(unsafe_code)]
    pub fn encode(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        rows: usize,
    ) -> crate::Result<()> {
        if rows == 0 || rows > self.max_rows || logits.length() < rows * self.vocab * 4 {
            return Err(Error::InvalidArgument(
                "greedy rows exceed the workspace or the logits".into(),
            ));
        }
        let groups = self.vocab.div_ceil(GROUP_ELEMENTS);
        let vocab = self.vocab as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.partial);
        unsafe {
            bind(encoder, logits, 0, 0);
            bind(encoder, &self.partials, 0, 1);
            bind(encoder, &self.flags, 0, 2);
            set_u32(encoder, &vocab, 3);
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
            bind(encoder, &self.partials, 0, 0);
            bind(encoder, &self.flags, 0, 1);
            bind(encoder, &self.results, 0, 2);
            set_u32(encoder, &groups, 3);
        }
        dispatch(encoder, rows, 256);
        batch.record_dispatch();
        Ok(())
    }

    /// The selections of the last completed [`Self::encode`], row by row.
    #[must_use]
    pub fn results(&self, rows: usize) -> &[GreedyRowResult] {
        &self.results.as_slice::<GreedyRowResult>()[..rows.min(self.max_rows)]
    }

    #[must_use]
    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.partials.length() + self.flags.length() + self.results.length()
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
