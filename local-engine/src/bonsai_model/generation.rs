use std::collections::VecDeque;

use super::checkpoints::Loaded;
use super::{
    BonsaiEngine, BonsaiGeneration, BonsaiModel, CancelToken, GenerateParams, GenerationStats,
    HeadLag, Instant, LookupPolicy, MtpStats, PrefillProgress, PromptCacheSource, PromptCheckpoint,
    PromptPrefill, Sampler, SamplingParams, SamplingResult, SequenceState, StopReason,
    SuffixSession, VOCAB, prompt_cache,
};
use std::sync::Arc;

use crate::bonsai_tokenizer::{BonsaiTokenizer, StreamDecodeState};
use crate::bonsai_vision::PromptImages;
use crate::sampler::TokenLogprob;
use crate::structured::Grammar;

/// Acceptance assumed for a sequence's first lookup drafts: a lookup fires
/// only after a long exact match.
const INITIAL_NGRAM_ACCEPTANCE: f64 = 0.8;
/// Acceptance assumed for a sequence's first head drafts.
pub(super) const INITIAL_MTP_ACCEPTANCE: f64 = 0.6;

/// One request between its prefill and its last token.
///
/// Everything a generation needs between rounds lives here rather than on a
/// call stack, so the engine can advance several of them in turn or together
/// ([`BonsaiEngine::step`]). A request alone runs exactly the rounds it ran
/// before batching existed.
pub(super) struct ActiveGeneration {
    pub(super) id: u64,
    pub(super) prompt: Vec<u32>,
    pub(super) params: GenerateParams,
    session_id: Option<String>,
    pub(super) cancel: CancelToken,
    started: Instant,
    pub(super) stats: GenerationStats,
    token_ids: Vec<u32>,
    stop_reason: StopReason,
    /// No further round will run: the generation only awaits [`BonsaiEngine::finish_generation`].
    pub(super) done: bool,
    pub(super) sampler: Sampler,
    decoder: StreamDecodeState,
    pub(super) suffixes: SuffixSession,
    pub(super) lookup_policy: LookupPolicy,
    pub(super) pending: VecDeque<SamplingResult>,
    /// Log-probabilities of emitted tokens whose text the decoder still
    /// holds (an incomplete character), sent with the piece that holds it.
    logprobs: Vec<TokenLogprob>,
    cache_source: PromptCacheSource,
    prompt_snapshot: Option<PromptCheckpoint>,
    persisted_reusable_boundary: bool,
    /// The last emitted token, which the next round decodes.
    pub(super) seed: Option<u32>,
    /// This sequence's buffers while another sequence's are resident.
    pub(super) state: Option<SequenceState>,
    /// Rows the MTP head missed while this sequence decoded in a batch.
    pub(super) lag: HeadLag,
    /// The prompt's remaining prefill, until its first sample is taken.
    pub(super) prefill: Option<PromptPrefill>,
    /// Skip this step's prefill: a disk write or load that restores more of
    /// the prompt is still in flight (see [`BonsaiEngine::refresh_prefix`]).
    defer_prefill: bool,
    /// The disk snapshot this generation restores, still being read and
    /// verified; its prefill waits for it.
    disk_load: Option<prompt_cache::Load>,
    /// Prefill chunks reported to the caller.
    pub(super) prefill_chunks: usize,
    /// Running estimates of the chance a lookup or head draft is accepted,
    /// for the batched-speculation cost model.
    pub(super) ngram_acceptance: f64,
    pub(super) mtp_acceptance: f64,
}

impl ActiveGeneration {
    /// Stop before the next round, as a cancelled request does.
    pub(super) fn mark_cancelled(&mut self) {
        if self.done {
            return;
        }
        self.stop_reason = StopReason::Cancelled;
        self.stats.sampled_tokens += self.pending.len();
        self.pending.clear();
        self.seed = None;
        self.done = true;
    }

    /// Whether the generation is past its prompt and still running.
    pub(super) const fn decoding(&self) -> bool {
        !self.done && self.prefill.is_none()
    }

    /// Whether the disk snapshot this generation restores is still loading.
    pub(super) fn awaiting_disk(&self) -> bool {
        self.disk_load
            .as_ref()
            .is_some_and(|load| !load.is_finished())
    }

    /// Prompt tokens still to prefill, if the prompt is not prefilled yet.
    pub(super) fn prefill_remaining(&self) -> Option<usize> {
        self.prefill
            .as_ref()
            .map(|plan| plan.remaining(self.prompt.len()))
    }

    /// Prompt tokens this generation prefills in all, past its restored prefix.
    pub(super) fn prefill_total(&self) -> usize {
        self.prefill
            .as_ref()
            .map_or(0, |plan| self.prompt.len() - plan.reused)
    }

    /// The most tokens this generation's sequence can come to hold.
    pub(super) const fn reserved_tokens(&self) -> usize {
        self.prompt.len() + self.params.max_tokens
    }
}

impl BonsaiEngine {
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        emit: impl FnMut(&str) -> bool,
    ) -> crate::Result<BonsaiGeneration> {
        self.generate_session(prompt, params, None, emit)
    }

    /// Generate with `session_id`'s snapshot affinity, under a token that is
    /// never tripped. Kept non-cancellable for callers that have no producer to
    /// cancel from; identical to [`Self::generate_session_cancellable`] with a
    /// fresh token.
    pub fn generate_session(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        emit: impl FnMut(&str) -> bool,
    ) -> crate::Result<BonsaiGeneration> {
        self.generate_session_cancellable(prompt, params, session_id, emit, &CancelToken::new())
    }

    /// Generate with `session_id`'s snapshot affinity, stopping early when
    /// `cancel` is tripped.
    ///
    /// `cancel` reaches inside the model instead of only the token loop: the
    /// engine polls it between GPU dispatches, before each prefill chunk and
    /// before each speculative or n-gram verify round. Metal cannot abort a
    /// committed command buffer, so a trip submits no further work and leaves
    /// the K/V cache and position cursor on the last committed boundary.
    ///
    /// A trip is the success value [`StopReason::Cancelled`], never an error.
    /// The GPU prompt-cache tier keeps the checkpoints of prefixes of the
    /// prompt up to where the request stopped, whose K/V rows are intact, and
    /// drops every other, so a cancelled request cannot poison the next one
    /// and a retry still restores what it shares.
    pub fn generate_session_cancellable(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        emit: impl FnMut(&str) -> bool,
        cancel: &CancelToken,
    ) -> crate::Result<BonsaiGeneration> {
        let mut silent = |_| {};
        self.generate_session_progress(prompt, params, session_id, emit, cancel, &mut silent)
    }

    /// Generate, reporting every prefill chunk boundary to `progress`.
    ///
    /// Prefill is the one phase that emits no [`crate::Event`], so this is the
    /// only place a caller learns the engine is alive between the request and
    /// the first token. The report is a hint, not a record: the boundary is
    /// taken at the same place as the cancel poll, immediately before the chunk
    /// is submitted, so it counts a chunk that is about to run rather than one
    /// that has finished.
    ///
    /// Reporting costs one call per chunk on the prefill critical path of a
    /// single-flight engine, so the reporter must not block; it is called before
    /// any GPU work for that chunk, never after.
    ///
    /// The request runs alone: it must not be called while [`Self::admit`]ted
    /// generations are still active.
    pub fn generate_session_progress(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        emit: impl FnMut(&str) -> bool,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<BonsaiGeneration> {
        let mut emit = emit;
        self.generate_constrained(
            prompt,
            params,
            session_id,
            |piece, _| emit(piece),
            cancel,
            progress,
            None,
            None,
        )
    }

    /// [`Self::generate_session_progress`] under a compiled response format.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn generate_constrained(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        mut emit: impl FnMut(&str, Vec<TokenLogprob>) -> bool,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        grammar: Option<Grammar>,
        images: Option<Arc<PromptImages>>,
    ) -> crate::Result<BonsaiGeneration> {
        if self.active_generations() != 0 {
            return Err(crate::Error::InvalidArgument(
                "a single generation cannot run beside admitted ones".into(),
            ));
        }
        let id =
            self.admit_constrained(prompt, params, session_id, cancel.clone(), grammar, images)?;
        loop {
            for (finished, result) in self.step(
                &mut |_, piece, logprobs| emit(piece, logprobs),
                &mut |_, report| {
                    progress(report);
                },
            ) {
                if finished == id {
                    return result;
                }
            }
        }
    }

    /// Validate a request whose sequence is resident in the model, restore
    /// its reusable prompt prefix and plan the rest of its prefill, which
    /// [`Self::prefill_round`] runs.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn begin_generation(
        &mut self,
        id: u64,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        cancel: CancelToken,
        grammar: Option<Grammar>,
        images: Option<Arc<PromptImages>>,
    ) -> crate::Result<ActiveGeneration> {
        // Installed before anything can prefill, so the model always polls the
        // token belonging to the request in flight.
        self.model.set_cancel(cancel.clone());
        self.validate_with_images(prompt, params, images.as_deref())?;
        // The resident sequence's prompt images, before any prefix restore
        // or prefill reads them; they move with its buffers when parked.
        self.model.set_images(images);
        let started = Instant::now();
        let stats = GenerationStats {
            prompt_tokens: prompt.len(),
            ..GenerationStats::default()
        };
        let mut sampler = Sampler::new(
            VOCAB,
            SamplingParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                min_p: params.min_p,
                presence_penalty: params.presence_penalty,
                repetition_penalty: params.repetition_penalty,
                eos_tokens: self.tokenizer.eos_ids().to_vec(),
                seed: params.seed,
            },
        );
        if let Some(grammar) = grammar {
            sampler.constrain(grammar);
        }
        if let Some(top) = params.logprobs {
            sampler.report_logprobs(top);
        }
        let mut generation = ActiveGeneration {
            id,
            prompt: prompt.to_vec(),
            params: params.clone(),
            session_id: session_id.map(str::to_owned),
            cancel,
            started,
            stats,
            token_ids: Vec::new(),
            stop_reason: StopReason::TokenLimit,
            done: false,
            sampler,
            decoder: StreamDecodeState::default(),
            suffixes: SuffixSession::default(),
            lookup_policy: LookupPolicy::new(self.ngram.max_drafts),
            pending: VecDeque::new(),
            logprobs: Vec::new(),
            cache_source: PromptCacheSource::None,
            prompt_snapshot: None,
            persisted_reusable_boundary: false,
            seed: None,
            state: None,
            lag: HeadLag::default(),
            prefill: None,
            defer_prefill: false,
            disk_load: None,
            prefill_chunks: 0,
            ngram_acceptance: INITIAL_NGRAM_ACCEPTANCE,
            mtp_acceptance: INITIAL_MTP_ACCEPTANCE,
        };
        if params.max_tokens == 0 {
            generation.done = true;
            return Ok(generation);
        }
        self.model.take_gpu_time();
        generation.sampler.observe(prompt);
        let (plan, cache_source, load) = self.begin_prompt(prompt, session_id, None)?;
        generation.stats.reused_prompt_tokens = plan.reused;
        generation.cache_source = cache_source;
        generation.prefill = Some(plan);
        generation.disk_load = load;
        generation.stats.gpu += self.model.take_gpu_time();
        Ok(generation)
    }

    /// Restore a longer prompt prefix into the resident `generation`, which
    /// has prefilled nothing yet, when the prompt cache gained one after its
    /// admission: requests admitted together that share a long uncached
    /// prefix (sub-agents fanned out at once) prefill one at a time, and each
    /// later one then starts from the snapshot the first left at that prefix
    /// instead of prefilling it again. While that snapshot's disk write is
    /// still in flight, the generation skips its prefill each step another
    /// request decodes, and waits for the write only when none does.
    ///
    /// A disk snapshot its admission or this found is read and verified on a
    /// helper thread (see [`BonsaiEngine::restore_reusable_prefix`]); the
    /// generation skips its prefill the same way until that load ends, and is
    /// then restored from it, the only part left on the engine thread being
    /// the copy into the model's buffers.
    pub(super) fn refresh_prefix(
        &mut self,
        generation: &mut ActiveGeneration,
    ) -> crate::Result<()> {
        loop {
            let Some(plan) = &generation.prefill else {
                return Ok(());
            };
            let reused = plan.reused;
            if plan.done() != reused {
                return Ok(());
            }
            let others_decode = self.active.iter().any(ActiveGeneration::decoding);
            if let Some(load) = generation.disk_load.take() {
                if others_decode && !load.is_finished() {
                    // Waiting here would stall every decoding request for the
                    // rest of the read: sit this step out.
                    generation.disk_load = Some(load);
                    generation.defer_prefill = true;
                    return Ok(());
                }
                let path = load.path().to_owned();
                let snapshot = load.wait();
                self.restart_prompt(generation, Some((path, snapshot)))?;
                continue;
            }
            // A restore costs in proportion to the prefix it copies, prefill in
            // proportion to the tokens it saves (about 8 ms against 0.1 ms each).
            let worth = |best: usize| best > reused && (best - reused) * 64 >= best;
            let mut best = self.reusable_prefix(&generation.prompt);
            let held = best.max(reused);
            if self
                .pending_disk_prefix(&generation.prompt, held)
                .is_some_and(worth)
            {
                if others_decode {
                    // Waiting here would stall every decoding request for the
                    // rest of the write: sit this step out and ask again next one.
                    generation.defer_prefill = true;
                    return Ok(());
                }
                self.await_disk_prefix(&generation.prompt, held);
                best = self.reusable_prefix(&generation.prompt);
            }
            if !worth(best) {
                return Ok(());
            }
            self.restart_prompt(generation, None)?;
            if generation.disk_load.is_none() {
                return Ok(());
            }
        }
    }

    /// Restore `generation`'s longest reusable prefix again, with a disk
    /// snapshot that finished `loaded`, and plan its prefill from there.
    fn restart_prompt(
        &mut self,
        generation: &mut ActiveGeneration,
        loaded: Option<Loaded>,
    ) -> crate::Result<()> {
        self.model.take_gpu_time();
        self.claim_gpu_cache();
        let (plan, cache_source, load) =
            self.begin_prompt(&generation.prompt, generation.session_id.as_deref(), loaded)?;
        generation.stats.reused_prompt_tokens = plan.reused;
        generation.cache_source = cache_source;
        generation.prefill = Some(plan);
        generation.disk_load = load;
        generation.stats.gpu += self.model.take_gpu_time();
        Ok(())
    }

    /// Prefill up to `budget` more prompt tokens of the resident `generation`
    /// and, once its prompt is complete, take its first sample.
    pub(super) fn prefill_round(
        &mut self,
        generation: &mut ActiveGeneration,
        budget: usize,
        progress: &mut dyn FnMut(u64, PrefillProgress),
    ) -> crate::Result<()> {
        if std::mem::take(&mut generation.defer_prefill) {
            return Ok(());
        }
        let Some(mut plan) = generation.prefill.take() else {
            return Ok(());
        };
        self.model.take_gpu_time();
        self.model.set_cancel(generation.cancel.clone());
        self.model
            .set_device_greedy(generation.sampler.selects_argmax());
        self.model
            .set_device_top_k(generation.sampler.device_top_k());
        self.model.set_logprobs(generation.sampler.logprobs());
        let id = generation.id;
        let chunks = &mut generation.prefill_chunks;
        let mut report = |report: PrefillProgress| {
            *chunks += 1;
            progress(
                id,
                PrefillProgress {
                    tokens: report.tokens,
                    chunks: *chunks,
                },
            );
        };
        let complete = self.advance_prompt(
            &generation.prompt,
            generation.session_id.as_deref(),
            &mut plan,
            budget,
            &mut report,
        )?;
        generation.stats.gpu += self.model.take_gpu_time();
        if self.model.take_cancel_observed() {
            // Prefill stopped between chunks, so `scratch.logits` holds no row
            // of this prompt: there is nothing to sample. The shared tail
            // drops the checkpoints past where it stopped.
            generation.stop_reason = StopReason::Cancelled;
            generation.done = true;
            return Ok(());
        }
        if !complete {
            generation.prefill = Some(plan);
            return Ok(());
        }
        generation.stats.prefill = generation.started.elapsed();
        generation.prompt_snapshot = plan.snapshot.take();
        generation.persisted_reusable_boundary = plan.persisted_reusable_boundary;
        generation.suffixes = self
            .suffix_store
            .session(&generation.prompt, self.ngram.min_match);
        let sample = sample_current(
            &mut self.model,
            &mut generation.sampler,
            &mut generation.stats,
        )?;
        generation.pending.push_back(sample);
        Ok(())
    }

    /// Emit every pending sample, stopping the generation at EOS, at its token
    /// limit, or when `emit` refuses a piece. Each piece goes with the
    /// log-probabilities of the tokens that produced it, when the request
    /// asked for them. Leaves `seed` set when another round is due.
    pub(super) fn drain_generation(
        tokenizer: &BonsaiTokenizer,
        generation: &mut ActiveGeneration,
        emit: &mut dyn FnMut(&str, Vec<TokenLogprob>) -> bool,
    ) -> crate::Result<()> {
        generation.seed = None;
        if let Some(failure) = generation.sampler.grammar_failure() {
            // Never an end-of-sequence that claims a complete answer.
            return Err(crate::Error::Generation(format!(
                "response format constraint failed: {failure}"
            )));
        }
        while let Some(sample) = generation.pending.pop_front() {
            // Samples and records pair one to one, in selection order.
            let logprob = generation.sampler.take_logprob();
            let stats = &mut generation.stats;
            stats.sampled_tokens += 1;
            stats
                .first_token
                .get_or_insert_with(|| generation.started.elapsed());
            if sample.is_eos {
                generation.stop_reason = StopReason::Eos;
                generation.done = true;
                break;
            }
            generation.token_ids.push(sample.token_id);
            generation.suffixes.append(sample.token_id);
            generation.sampler.observe(&[sample.token_id]);
            if let Some(mut logprob) = logprob {
                logprob.bytes = tokenizer.token_bytes(logprob.token);
                for top in &mut logprob.top {
                    top.bytes = tokenizer.token_bytes(top.token);
                }
                generation.logprobs.push(logprob);
            }
            if let Some(piece) = tokenizer.stream_step(&mut generation.decoder, sample.token_id)?
                && !emit(&piece, std::mem::take(&mut generation.logprobs))
            {
                generation.stop_reason = StopReason::Stopped;
                generation.stats.sampled_tokens += generation.pending.len();
                generation.done = true;
                break;
            }
            if generation.stats.sampled_tokens >= generation.params.max_tokens {
                generation.done = true;
                break;
            }
            if generation.pending.is_empty() {
                generation.seed = Some(sample.token_id);
            }
        }
        if generation.done {
            generation.pending.clear();
            generation.seed = None;
        } else if generation.seed.is_none() {
            // Every round leaves a sample or stops the generation; a running
            // generation with neither could never advance.
            return Err(crate::Error::Generation(
                "generation has no sample to continue from".into(),
            ));
        }
        Ok(())
    }

    /// One single-sequence round for the resident `generation`: n-gram
    /// verification, an MTP speculative round, or a plain decode step,
    /// exactly as an unbatched request runs it.
    pub(super) fn solo_round(&mut self, generation: &mut ActiveGeneration) -> crate::Result<()> {
        let Some(seed) = generation.seed.take() else {
            return Ok(());
        };
        self.model.set_cancel(generation.cancel.clone());
        self.model
            .set_device_greedy(generation.sampler.selects_argmax());
        self.model
            .set_device_top_k(generation.sampler.device_top_k());
        self.model.set_logprobs(generation.sampler.logprobs());
        let head = self.model.speculation().is_some() && generation.lag.usable();
        if head && !generation.lag.is_empty() {
            self.model.catch_up_head(&mut generation.lag)?;
        }
        let drafts = Self::lookup_draft(self.ngram, self.info.context, generation);
        let stats = &mut generation.stats;
        let remaining = generation.params.max_tokens - stats.sampled_tokens;
        let pending = &mut generation.pending;
        if !drafts.is_empty() {
            let batch = self
                .model
                .ngram_step(seed, &drafts, &mut generation.sampler)?;
            stats.sampling += batch.sampling;
            stats.ngram.rounds += batch.ngram.rounds;
            stats.ngram.proposed_tokens += batch.ngram.proposed_tokens;
            stats.ngram.accepted_tokens += batch.ngram.accepted_tokens;
            generation.lookup_policy.observe(
                batch.ngram.accepted_tokens,
                batch.ngram.proposed_tokens,
                self.ngram.max_drafts,
            );
            pending.extend(batch.samples);
        } else if head {
            let batch = self
                .model
                .speculative_step(seed, &mut generation.sampler, remaining)?;
            stats.sampling += batch.sampling;
            accumulate_mtp(&mut stats.mtp, &batch.stats);
            pending.extend(batch.samples);
        } else {
            self.model.decode(seed)?;
        }
        if self.model.take_cancel_observed() {
            // The round below was not submitted, so `pending` stayed empty:
            // do not refill it from logits that were never recomputed.
            generation.stop_reason = StopReason::Cancelled;
            stats.sampled_tokens += pending.len();
            pending.clear();
            generation.done = true;
        } else if pending.is_empty() {
            pending.push_back(sample_current(
                &mut self.model,
                &mut generation.sampler,
                stats,
            )?);
        }
        stats.gpu += self.model.take_gpu_time();
        Ok(())
    }

    /// Close the resident `generation`: catch its state up to everything it
    /// emitted and record it in the prompt cache, or after a cancellation
    /// drop what its prefill or decode overwrote.
    pub(super) fn finish_generation(
        &mut self,
        mut generation: ActiveGeneration,
    ) -> crate::Result<BonsaiGeneration> {
        let mut stats = std::mem::take(&mut generation.stats);
        stats.generated_tokens = generation.token_ids.len();
        stats.response_format_complete = generation
            .sampler
            .response_format_complete(generation.stop_reason == StopReason::Eos);
        stats.tool_constraints_complete = generation
            .sampler
            .tool_constraints_complete(generation.stop_reason == StopReason::Eos);
        let prompt = &generation.prompt;
        let token_ids = std::mem::take(&mut generation.token_ids);
        let session_id = generation.session_id.as_deref();
        if generation.params.max_tokens == 0 {
            return Ok(BonsaiGeneration {
                text: String::new(),
                token_ids,
                stop_reason: generation.stop_reason,
                stats,
                cache_source: PromptCacheSource::None,
            });
        }
        if generation.stop_reason == StopReason::Cancelled {
            self.keep_reached_prompt(prompt);
        } else {
            if self.model.speculation().is_some()
                && generation.lag.usable()
                && !generation.lag.is_empty()
            {
                self.model.catch_up_head(&mut generation.lag)?;
            }
            self.claim_gpu_cache();
            let represented = prompt.len() + token_ids.len();
            if self.model.position() + 1 == represented
                && let Some(&last) = token_ids.last()
            {
                self.model.decode(last)?;
            }
            if self.model.position() == represented {
                self.cached_tokens.clear();
                self.cached_tokens.extend_from_slice(prompt);
                self.cached_tokens.extend_from_slice(&token_ids);
                let represented_tokens = self.cached_tokens.clone();
                let tail = &prompt[..prompt.len() - 1];
                // A request that persisted a boundary leaves its tail to the
                // GPU checkpoint, which its session's next turn restores,
                // unless that copy is already gone or about to be: another
                // buffer set claimed the tier meanwhile, or another request
                // is running, whose next milestone or end will.
                let gpu_tail = self.holds_checkpoint(tail);
                let persist =
                    !generation.persisted_reusable_boundary || !gpu_tail || !self.active.is_empty();
                let keep = self.host_would_keep(tail.len())
                    || (persist && self.disk_would_keep(tail, false));
                // The prompt boundary's snapshot, read back now that decode
                // no longer needs the bandwidth, and only when a tier keeps it:
                // at long prompts the copy is gigabytes on the engine thread.
                let prompt_snapshot = match generation.prompt_snapshot.take() {
                    Some(checkpoint) if keep => self.model.prompt_snapshot_at(&checkpoint)?,
                    _ => None,
                };
                self.save_prompt_checkpoint(&represented_tokens, false)?;
                if let Some(snapshot) = prompt_snapshot {
                    self.save_session_snapshot(tail, session_id, Some(snapshot), persist, false)?;
                }
                if !persist && !self.host_would_keep(tail.len()) {
                    self.owe_to_disk(tail, session_id);
                }
                // Chat templates re-render assistant turns, so the output end is
                // rarely a later prefix: keep it in (volatile) host memory only.
                self.save_session_snapshot(&represented_tokens, session_id, None, false, false)?;
            } else {
                self.keep_reached_prompt(prompt);
            }
        }
        stats.elapsed = generation.started.elapsed();
        stats.gpu += self.model.take_gpu_time();
        self.suffix_store
            .remember(generation.suffixes.history().to_vec());
        Ok(BonsaiGeneration {
            text: self.tokenizer.decode(&token_ids, false)?,
            token_ids,
            stop_reason: generation.stop_reason,
            stats,
            cache_source: generation.cache_source,
        })
    }
}

impl BonsaiEngine {
    /// Capture the output-normalized hidden of `positions` over `tokens`
    /// (see [`BonsaiModel::capture_hidden`]) in blocks of at most `rows`, on
    /// the loaded model: no second model and no new sequence allocation.
    ///
    /// Only between generations: with any [`Self::admit`]ted generation still
    /// active this refuses, so a caller must drain them first. `cancel` is
    /// polled before every block; a trip returns `Ok(None)`.
    ///
    /// Prompt-cache ownership: capture overwrites the buffer set it runs on.
    /// When the resident set is not the one the GPU prompt-cache tier
    /// describes, it runs there and the tier is untouched. When it is and a
    /// free pooled set exists, capture runs on the pooled set, which is then
    /// swapped back out, so the tier still describes intact buffers. Only
    /// when the resident set is the cached one and no pooled set exists is
    /// the GPU tier cleared. Host and disk snapshots are copies and are never
    /// affected. Every later request restores or resets its own state.
    pub(crate) fn capture_hidden(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        rows: usize,
        cancel: &CancelToken,
    ) -> crate::Result<Option<Vec<f32>>> {
        if !self.active.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "hidden capture cannot run beside admitted generations".into(),
            ));
        }
        // No active generation owns the resident set.
        self.resident = None;
        let describes_resident = self.cache_state == self.model.state_id()
            && !(self.cached_tokens.is_empty() && self.prompt_checkpoints.is_empty());
        let mut borrowed = None;
        if describes_resident {
            if let Some(mut slot) = self.pool.pop() {
                if let Err(error) = self.model.swap_sequence(&mut slot) {
                    // An interrupted swap leaves no set trustworthy.
                    self.clear_gpu_cache();
                    return Err(error);
                }
                borrowed = Some(slot);
            } else {
                self.clear_gpu_cache();
            }
        }
        self.model.set_cancel(cancel.clone());
        self.model.take_gpu_time();
        let result = self.model.capture_hidden(tokens, positions, rows);
        self.model.take_gpu_time();
        // Never leave a decision's token installed for later work.
        self.model.set_cancel(CancelToken::new());
        if let Some(mut slot) = borrowed {
            // `slot` holds the cached set; bring it back and pool the
            // overwritten one.
            if let Err(error) = self.model.swap_sequence(&mut slot) {
                self.clear_gpu_cache();
                return Err(error);
            }
            self.pool.push(slot);
        }
        result
    }
}

/// Sample from the logits the backend produced for its newest committed token.
fn sample_current(
    model: &mut BonsaiModel,
    sampler: &mut Sampler,
    stats: &mut GenerationStats,
) -> crate::Result<SamplingResult> {
    let started = Instant::now();
    let sample = model.sample(sampler)?;
    stats.sampling += started.elapsed();
    Ok(sample)
}

fn accumulate_mtp(total: &mut MtpStats, round: &MtpStats) {
    total.rounds += round.rounds;
    total.proposed_tokens += round.proposed_tokens;
    total.accepted_tokens += round.accepted_tokens;
    total.verified_tokens += round.verified_tokens;
    total.drafting += round.drafting;
    total.verification += round.verification;
    total.commit += round.commit;
}

/// How many drafts may follow the seed: never past the request's token budget
/// and never past the last context slot.
pub const fn draft_depth(depth: usize, remaining: usize, context: usize, position: usize) -> usize {
    let by_request = remaining.saturating_sub(1);
    let by_context = context.saturating_sub(position + 1);
    let mut clamped = depth;
    if by_request < clamped {
        clamped = by_request;
    }
    if by_context < clamped {
        clamped = by_context;
    }
    clamped
}
