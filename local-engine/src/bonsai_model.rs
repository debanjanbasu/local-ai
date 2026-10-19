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
use crate::bonsai_vision::PromptImages;
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

use self::checkpoints::{CachedCheckpoint, PromptPrefill, SessionSnapshot};

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
///
/// Work that runs outside the engine for the same request (a hosted tool
/// between generations) waits on [`Self::cancelled`] instead of polling.
#[derive(Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    notify: Arc<tokio::sync::Notify>,
}

impl CancelToken {
    /// A fresh token: not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent, and observed by every clone.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Whether [`Self::cancel`] has been called on this token or any clone.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Resolve once [`Self::cancel`] has been called on any clone.
    pub async fn cancelled(&self) {
        let mut notified = std::pin::pin!(self.notify.notified());
        loop {
            // Registered before the flag is read, so a `cancel` between the
            // two still wakes this wait.
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.as_mut().await;
            notified.set(self.notify.notified());
        }
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
    /// The consumer refused a piece. The sequence stops at a token boundary
    /// with its state intact, as at a token limit, so unlike
    /// [`Self::Cancelled`] the prompt cache is kept. The engine API reports
    /// it only for a matched [`crate::ChatRequest::stop`] sequence; any other
    /// refusal is reported as [`Self::Cancelled`].
    Stopped,
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
    /// Closes `disk_writer` with a deadline from outside the engine (see
    /// [`Self::prompt_cache_closer`]); exists before the writer does.
    disk_closer: prompt_cache::Closer,
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
    /// Verify every available draft in batched steps, whatever the cost
    /// model says, so tests exercise batched speculation.
    #[cfg(test)]
    force_batched_drafts: bool,
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

    /// The decoder's input signs (width 5120), the basis image embeddings
    /// are injected in.
    #[doc(hidden)]
    pub fn input_signs(&self) -> crate::Result<&[f32]> {
        self.model
            .package()
            .hadamard()
            .signs(crate::bonsai::WIDTH as u32)
    }

    /// The tokenizer every prompt is encoded with; clone it to share it.
    pub const fn tokenizer(&self) -> &BonsaiTokenizer {
        &self.tokenizer
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
        self.validate_with_images(prompt, params, None)
    }

    /// [`Self::validate_generation`] of a prompt whose image tokens `images`
    /// describes.
    pub(crate) fn validate_with_images(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        images: Option<&PromptImages>,
    ) -> crate::Result<()> {
        crate::runtime::validate_generation_params(params)?;
        validate_request(prompt, params.max_tokens, self.info.context, images)
    }
}

impl BonsaiEngine {
    /// Evaluation hook: the next-token logits after `prompt` (whose image
    /// tokens `images` describes) and after each token of `continuation`
    /// fed in turn, from empty state, bypassing every prompt cache.
    #[doc(hidden)]
    pub fn teacher_forced_logits(
        &mut self,
        prompt: &[u32],
        images: Option<Arc<PromptImages>>,
        continuation: &[u32],
    ) -> crate::Result<Vec<Vec<f32>>> {
        if !self.active.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "teacher forcing cannot run beside admitted generations".into(),
            ));
        }
        validate_request(
            prompt,
            continuation.len(),
            self.info.context,
            images.as_deref(),
        )?;
        self.resident = None;
        self.clear_gpu_cache();
        self.model.reset();
        self.model.set_images(images);
        self.model.set_cancel(CancelToken::new());
        self.model.set_device_greedy(false);
        self.model.prefill(prompt, &mut |_| {})?;
        let mut logits = vec![self.model.last_logits()];
        for &token in continuation {
            self.model.decode(token)?;
            logits.push(self.model.last_logits());
        }
        Ok(logits)
    }
}

fn validate_request(
    prompt: &[u32],
    output: usize,
    context: usize,
    images: Option<&PromptImages>,
) -> crate::Result<()> {
    if prompt.is_empty() || crate::bonsai_vision::check_tokens(prompt, 0, images).is_err() {
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
