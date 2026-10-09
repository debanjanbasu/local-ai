//! Bonsai 2 27B text generation behind one engine.
//!
//! The GGUF opens the native Metal model ([`crate::bonsai_native`]), which
//! runs the PTQ1 checkpoint straight from its mapping. Every request
//! starts from empty state, with no context shifting or silent truncation,
//! and every speculative draft is verified by the target before it is
//! emitted.

use core::sync::atomic::{AtomicBool, Ordering};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use local_metal::bonsai_ops::AttentionKernel;

use crate::bonsai::{BonsaiPackage, VOCAB};
use crate::bonsai_mtp::{DRAFT_CHAIN_MIN_MARGIN, MtpMode, MtpResolution};
use crate::bonsai_native::{
    BatchRow, BonsaiModel, HeadLag, HostPromptSnapshot, KvOptions, PROMPT_CHECKPOINT_BYTES,
    PromptCheckpoint, PromptSnapshot, SequenceState,
};
use crate::bonsai_ngram::{
    LookupPolicy, NgramSettings, SuffixSession, SuffixStore, fill_verify_tile,
};
use crate::bonsai_tokenizer::BonsaiTokenizer;
use crate::prompt_cache::{self, DiskEntry};
use crate::sampler::{Sampler, SamplingParams, SamplingResult};
use crate::{GenerateParams, GenerationStats, MtpStats, NgramStats, PrefillProgress};

mod cache_policy;
mod checkpoints;
mod construction;
mod generation;
mod scheduler;

pub use self::cache_policy::{DEFAULT_PROMPT_CACHE_CHECKPOINTS, MAX_PROMPT_CACHE_CHECKPOINTS};
pub use self::generation::draft_depth;
pub use self::scheduler::Finished;

use self::checkpoints::{CachedCheckpoint, SessionSnapshot};

/// Shared cancellation flag for one in-flight generation.
///
/// A producer and the engine worker hold the *same* token: the producer calls
/// [`Self::cancel`] when its client goes away, and the worker polls
/// [`Self::is_cancelled`] between GPU dispatches — before every prefill chunk
/// and before every speculative or n-gram verify round. Clones share one flag,
/// so any clone observes another clone's `cancel`; a token built with either
/// [`Self::new`] or [`Default::default`] starts *not* cancelled, which is what
/// lets [`BonsaiEngine::generate_session`] keep its non-cancellable
/// signature.
///
/// Metal offers no way to abort work already submitted: a committed command
/// buffer runs to completion and `MTLCommandBufferStatus` is read-only. The
/// realizable semantics are therefore "submit no further work", never "abort
/// work in flight". A poll always lands between dispatches, where the K/V cache
/// and the position cursor hold the exact state of the last committed chunk or
/// round.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh token: not cancelled.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation. Idempotent, and observed by every clone.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Whether [`Self::cancel`] has been called on this token or any clone.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// One verified speculative round: accepted samples plus per-round counters.
pub struct SpeculativeBatch {
    pub samples: Vec<SamplingResult>,
    pub stats: MtpStats,
    pub ngram: NgramStats,
    pub sampling: Duration,
}

/// Default prefill block.
#[cfg(test)]
pub const DEFAULT_PREFILL_CHUNK: usize = local_metal::bonsai_ops::MAX_PREFILL_TOKENS as usize;
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheSource {
    None,
    Gpu,
    Host,
    Disk,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct BonsaiInfo {
    pub precision: String,
    pub policy: serde_json::Value,
    pub device: String,
    pub context: usize,
    /// Whether the caller explicitly requested `context` rather than using admission sizing.
    pub context_explicit: bool,
    /// Human-readable admission decision, included in startup JSON.
    pub context_reason: String,
    pub prefill_chunk_size: usize,
    pub tensor_bytes: u64,
    pub state_bytes: usize,
    pub estimated_working_set: u64,
    pub working_set_limit: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    TokenLimit,
    Cancelled,
}

pub struct BonsaiGeneration {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub stop_reason: StopReason,
    pub stats: GenerationStats,
    pub cache_source: PromptCacheSource,
}

pub struct BonsaiEngine {
    tokenizer: BonsaiTokenizer,
    model: Box<BonsaiModel>,
    info: BonsaiInfo,
    ngram: NgramSettings,
    suffix_store: SuffixStore,
    cached_tokens: Vec<u32>,
    prompt_checkpoints: Vec<CachedCheckpoint>,
    max_prompt_checkpoints: usize,
    session_snapshots: Vec<SessionSnapshot>,
    /// Committed disk snapshots. The writer reports each one only after its
    /// rename, so nothing here can still be in flight.
    disk_snapshots: Vec<DiskEntry>,
    /// Persists disk snapshots off the request path; spawned on first use.
    /// Dropping it finishes pending writes, so it flushes with the engine.
    disk_writer: Option<prompt_cache::Writer>,
    prompt_cache_bytes: usize,
    prompt_cache_disk_bytes: u64,
    prompt_cache_dir: Option<PathBuf>,
    prompt_cache_model_key: String,
    /// Generations between admission and their last token.
    active: Vec<generation::ActiveGeneration>,
    /// The generation whose buffers are resident in the model, if any.
    resident: Option<u64>,
    /// Free parked buffer sets.
    pool: Vec<SequenceState>,
    /// The buffer set `cached_tokens` and `prompt_checkpoints` describe.
    cache_state: u64,
    next_generation: u64,
}

impl BonsaiEngine {
    pub const fn info(&self) -> &BonsaiInfo {
        &self.info
    }

    /// Whether requests speculate with an MTP head.
    #[must_use]
    pub fn mtp_enabled(&self) -> bool {
        self.model.speculation().is_some()
    }

    /// The decode/prefill attention kernel in use.
    #[must_use]
    pub fn attention_kernel(&self) -> &'static str {
        self.model.attention_kernel_name()
    }

    pub fn final_answer(&self, ids: &[u32], thinking: bool) -> crate::Result<Option<String>> {
        self.tokenizer.final_answer(ids, thinking)
    }

    pub fn encode_prompt(&self, text: &str, raw: bool, thinking: bool) -> crate::Result<Vec<u32>> {
        if raw {
            self.tokenizer.encode(text)
        } else {
            self.tokenizer
                .encode(&BonsaiTokenizer::chat_prompt(text, thinking)?)
        }
    }

    #[doc(hidden)]
    pub fn decode_tokens(&self, ids: &[u32]) -> crate::Result<String> {
        self.tokenizer.decode(ids, true)
    }

    #[doc(hidden)]
    pub fn validate_generation(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
    ) -> crate::Result<()> {
        crate::runtime::validate_generation_params(params)?;
        validate_request(prompt, params.max_tokens, self.info.context)
    }
}

fn validate_request(prompt: &[u32], output: usize, context: usize) -> crate::Result<()> {
    if prompt.is_empty() || prompt.iter().any(|&token| token as usize >= VOCAB) {
        return Err(crate::Error::InvalidArgument(
            "empty or invalid Bonsai prompt tokens".into(),
        ));
    }
    if prompt
        .len()
        .checked_add(output)
        .is_none_or(|length| length > context)
    {
        return Err(crate::Error::ContextOverflow(format!(
            "{} prompt + {output} output tokens exceed {context}; no tokens were truncated",
            prompt.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
