//! Drafting for several sequences at once: one row per sequence per draft
//! step.
//!
//! Row `y` of a step reads its input token from `tokens[slots[y]]` and writes
//! its selection into its own slots, so each sequence's chain stays in its own
//! run of the token buffer while the rows share one embedding gather and one
//! top-two reduction. Each row computes bitwise what
//! [`super::DraftKernels::embed_inverse`] and [`super::DraftKernels::top_two`]
//! compute for it alone.

use core::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize};

use super::{DraftTopTwo, GROUP_ELEMENTS, PARTIAL_BYTES, bind, dispatch, set_u32};
use crate::Error;
use crate::batch::CommandBatch;
use crate::bonsai::{HADAMARD_BLOCK_ELEMENTS, Ptq1Matrix, SignedHadamard};
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::shaders::ShaderLibrary;

/// Pipelines and the first-stage workspace for up to `max_rows` rows of one
/// vocabulary size. Same encoding contract as [`super::DraftKernels`].
pub struct DraftRows {
    embed_inverse: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    finish: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    partials: MetalBuffer,
    max_rows: usize,
    max_vocab: usize,
}

impl DraftRows {
    pub fn new(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        max_vocab: usize,
        max_rows: usize,
    ) -> crate::Result<Self> {
        if max_vocab < 2 || max_vocab >= u32::MAX as usize || max_rows == 0 {
            return Err(Error::InvalidArgument(
                "draft rows need a vocabulary of two or more tokens and one row".into(),
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
            embed_inverse: pipeline("draft_embed_inverse_rows")?,
            partial: pipeline("draft_top2_rows_partial")?,
            finish: pipeline("draft_top2_rows_final")?,
            partials: MetalBuffer::empty(
                context.device(),
                max_rows * max_vocab.div_ceil(GROUP_ELEMENTS) * PARTIAL_BYTES,
            )?,
            max_rows,
            max_vocab,
        })
    }

    /// Decode row `tokens[slots[y]]` of the packed `table` into row `y` of
    /// `output`, inverse-rotated, for every `y` of `slots`.
    #[allow(unsafe_code)]
    pub fn embed_inverse(
        &self,
        batch: &mut CommandBatch,
        table: Ptq1Matrix<'_>,
        rotation: &SignedHadamard,
        tokens: &MetalBuffer,
        slots: &[u32],
        output: &MetalBuffer,
    ) -> crate::Result<()> {
        let rows = slots.len();
        let token_slots = tokens.length() / size_of::<u32>();
        if rows == 0
            || rows > self.max_rows
            || table.columns != rotation.columns
            || !(table.columns as usize).is_multiple_of(HADAMARD_BLOCK_ELEMENTS)
            || output.length() < rows * table.columns as usize * size_of::<f32>()
            || slots.iter().any(|&slot| slot as usize >= token_slots)
            || std::ptr::eq(output.raw(), table.buffer.raw())
            || std::ptr::eq(output.raw(), tokens.raw())
        {
            return Err(Error::InvalidArgument(
                "draft row embedding shape, token slots or output is invalid".into(),
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
            set_slots(encoder, slots, 4);
            set_u32(encoder, &table.rows, 5);
            set_u32(encoder, &blocks_per_row, 6);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: blocks_per_row as usize,
                height: rows,
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

    /// Exact top two of each of the first `token_slots.len()` rows of
    /// `logits`: row `y`'s winner goes to `tokens[token_slots[y]]` and its
    /// pair to `results[result_slots[y]]`.
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub fn top_two(
        &self,
        batch: &mut CommandBatch,
        logits: &MetalBuffer,
        vocab: usize,
        tokens: &MetalBuffer,
        token_slots: &[u32],
        results: &MetalBuffer,
        result_slots: &[u32],
    ) -> crate::Result<()> {
        let rows = token_slots.len();
        let token_capacity = tokens.length() / size_of::<u32>();
        let result_capacity = results.length() / size_of::<DraftTopTwo>();
        if rows == 0
            || rows > self.max_rows
            || result_slots.len() != rows
            || !(2..=self.max_vocab).contains(&vocab)
            || logits.length() < rows * vocab * size_of::<f32>()
            || token_slots
                .iter()
                .any(|&slot| slot as usize >= token_capacity)
            || result_slots
                .iter()
                .any(|&slot| slot as usize >= result_capacity)
            || std::ptr::eq(tokens.raw(), results.raw())
        {
            return Err(Error::InvalidArgument(
                "draft row top-two vocabulary or slots are invalid".into(),
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
            bind(encoder, tokens, 0, 1);
            bind(encoder, results, 0, 2);
            set_u32(encoder, &groups, 3);
            set_slots(encoder, token_slots, 4);
            set_slots(encoder, result_slots, 5);
        }
        dispatch(encoder, rows, 256);
        batch.record_dispatch();
        Ok(())
    }

    #[must_use]
    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }

    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.partials.length()
    }
}

#[allow(unsafe_code)]
unsafe fn set_slots(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    slots: &[u32],
    index: usize,
) {
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new_unchecked(slots.as_ptr().cast_mut().cast::<c_void>()),
            std::mem::size_of_val(slots),
            index,
        );
    }
}
