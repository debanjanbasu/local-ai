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
const CHUNK_SIZE: usize = 256;

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

/// Reusable workspace for top-k selection.
///
/// `encode` only records GPU work. The caller must complete/wait for the containing
/// command batch before reading `candidates`, and must not overlap another encode
/// using this workspace with unfinished work from an earlier encode.
pub struct GpuTopK {
    chunk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    merge_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    intermediate: [MetalBuffer; 2],
    output: MetalBuffer,
    max_vocab: usize,
}

impl GpuTopK {
    pub fn new(
        ctx: &MetalContext,
        shaders: &ShaderLibrary,
        max_vocab: usize,
    ) -> crate::Result<Self> {
        if max_vocab == 0 || max_vocab > u32::MAX as usize {
            return Err(Error::InvalidArgument(
                "top-k max_vocab must be in 1..=u32::MAX".into(),
            ));
        }
        let make_pipeline = |name: &str| {
            let function = shaders.get_function(name)?;
            ctx.device()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| Error::PipelineCreation(error.to_string()))
        };
        let chunks = max_vocab.div_ceil(CHUNK_SIZE);
        let intermediate_bytes = chunks * MAX_TOP_K * size_of::<TopKCandidate>();
        Ok(Self {
            chunk_pipeline: make_pipeline("sampling_chunk_topk")?,
            merge_pipeline: make_pipeline("sampling_merge_topk")?,
            intermediate: [
                MetalBuffer::empty(ctx.device(), intermediate_bytes)?,
                MetalBuffer::empty(ctx.device(), intermediate_bytes)?,
            ],
            output: MetalBuffer::empty(ctx.device(), MAX_TOP_K * size_of::<TopKCandidate>())?,
            max_vocab,
        })
    }

    /// Encode hierarchical selection without allocating or synchronizing.
    #[allow(unsafe_code)]
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
        let vocab = vocab as u32;
        let k = k as u32;
        let chunks = (vocab as usize).div_ceil(CHUNK_SIZE) as u32;
        let encoder = batch.encoder();
        encoder.setComputePipelineState(&self.chunk_pipeline);
        unsafe {
            bind(encoder, logits, logits_offset, 0);
            bind(
                encoder,
                if chunks == 1 {
                    &self.output
                } else {
                    &self.intermediate[0]
                },
                0,
                1,
            );
            set_u32(encoder, &vocab, 2);
            set_u32(encoder, &k, 3);
        }
        dispatch(encoder, chunks as usize, CHUNK_SIZE);
        batch.record_dispatch();

        let mut lists = chunks;
        let mut source = 0;
        while lists > 1 {
            let merged = lists.div_ceil(2);
            let destination = if merged == 1 {
                &self.output
            } else {
                &self.intermediate[1 - source]
            };
            let encoder = batch.encoder();
            encoder.setComputePipelineState(&self.merge_pipeline);
            unsafe {
                bind(encoder, &self.intermediate[source], 0, 0);
                bind(encoder, destination, 0, 1);
                set_u32(encoder, &lists, 2);
                set_u32(encoder, &k, 3);
            }
            dispatch(encoder, merged as usize, (2 * k as usize).div_ceil(32) * 32);
            batch.record_dispatch();
            lists = merged;
            source = 1 - source;
        }
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
        self.intermediate
            .iter()
            .map(MetalBuffer::length)
            .sum::<usize>()
            + self.output.length()
    }

    #[must_use]
    pub const fn maximum_supported_k(&self) -> usize {
        MAX_TOP_K
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
