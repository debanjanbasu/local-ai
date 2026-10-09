use std::collections::VecDeque;

use super::{
    BonsaiEngine, BonsaiGeneration, BonsaiModel, CancelToken, GenerateParams, GenerationStats,
    HeadLag, Instant, LookupPolicy, MtpStats, PrefillProgress, PromptCacheSource, PromptCheckpoint,
    Sampler, SamplingParams, SamplingResult, SequenceState, StopReason, SuffixSession, VOCAB,
    fill_verify_tile,
};
use crate::bonsai_tokenizer::{BonsaiTokenizer, StreamDecodeState};

/// One request between its prefill and its last token.
///
/// Everything a generation needs between rounds lives here rather than on a
/// call stack, so the engine can advance several of them in turn or together
/// ([`BonsaiEngine::step`]). A request alone runs exactly the rounds it ran
/// before batching existed.
pub(super) struct ActiveGeneration {
    pub(super) id: u64,
    prompt: Vec<u32>,
    params: GenerateParams,
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
    suffixes: SuffixSession,
    lookup_policy: LookupPolicy,
    pub(super) pending: VecDeque<SamplingResult>,
    cache_source: PromptCacheSource,
    prompt_snapshot: Option<PromptCheckpoint>,
    persisted_reusable_boundary: bool,
    /// The last emitted token, which the next round decodes.
    pub(super) seed: Option<u32>,
    /// This sequence's buffers while another sequence's are resident.
    pub(super) state: Option<SequenceState>,
    /// Rows the MTP head missed while this sequence decoded in a batch.
    pub(super) lag: HeadLag,
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
    /// A trip is the success value [`StopReason::Cancelled`], never an error,
    /// and it clears the reusable prompt-cache tiers below, so a cancelled
    /// request cannot poison the next one.
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
        mut emit: impl FnMut(&str) -> bool,
        cancel: &CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<BonsaiGeneration> {
        if self.active_generations() != 0 {
            return Err(crate::Error::InvalidArgument(
                "a single generation cannot run beside admitted ones".into(),
            ));
        }
        let id = self.admit(
            prompt,
            params,
            session_id,
            cancel.clone(),
            progress,
            &mut emit,
        )?;
        loop {
            for (finished, result) in self.step(&mut |_, piece| emit(piece)) {
                if finished == id {
                    return result;
                }
            }
        }
    }

    /// Validate, prefill and take the first sample for a request whose
    /// sequence is resident in the model.
    pub(super) fn begin_generation(
        &mut self,
        id: u64,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        cancel: CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<ActiveGeneration> {
        // Installed before anything can prefill, so the model always polls the
        // token belonging to the request in flight.
        self.model.set_cancel(cancel.clone());
        self.validate_generation(prompt, params)?;
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
            sampler: sampler.clone(),
            decoder: StreamDecodeState::default(),
            suffixes: self.suffix_store.session(&[], self.ngram.min_match),
            lookup_policy: LookupPolicy::new(self.ngram.max_drafts),
            pending: VecDeque::new(),
            cache_source: PromptCacheSource::None,
            prompt_snapshot: None,
            persisted_reusable_boundary: false,
            seed: None,
            state: None,
            lag: HeadLag::default(),
        };
        if params.max_tokens == 0 {
            generation.done = true;
            return Ok(generation);
        }
        self.model.take_gpu_time();
        sampler.observe(prompt);
        self.model.set_device_greedy(sampler.selects_argmax());
        let (reused, cache_source, prompt_snapshot, persisted_reusable_boundary) =
            self.prepare_prompt(prompt, session_id, progress)?;
        generation.stats.reused_prompt_tokens = reused;
        generation.stats.prefill = started.elapsed();
        generation.cache_source = cache_source;
        generation.prompt_snapshot = prompt_snapshot;
        generation.persisted_reusable_boundary = persisted_reusable_boundary;
        generation.suffixes = self.suffix_store.session(prompt, self.ngram.min_match);
        if self.model.take_cancel_observed() {
            // Prefill stopped between chunks, so `scratch.logits` still holds an
            // older row: there is nothing new to sample. Fall through to the
            // shared tail, which clears the prompt cache for a cancelled stop.
            generation.stop_reason = StopReason::Cancelled;
            generation.done = true;
        } else {
            let sample = sample_current(&mut self.model, &mut sampler, &mut generation.stats)?;
            generation.pending.push_back(sample);
        }
        generation.sampler = sampler;
        generation.stats.gpu += self.model.take_gpu_time();
        Ok(generation)
    }

    /// Emit every pending sample, stopping the generation at EOS, at its token
    /// limit, or when `emit` refuses a piece. Leaves `seed` set when another
    /// round is due.
    pub(super) fn drain_generation(
        tokenizer: &BonsaiTokenizer,
        generation: &mut ActiveGeneration,
        emit: &mut dyn FnMut(&str) -> bool,
    ) -> crate::Result<()> {
        generation.seed = None;
        while let Some(sample) = generation.pending.pop_front() {
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
            if let Some(piece) = tokenizer.stream_step(&mut generation.decoder, sample.token_id)?
                && !emit(&piece)
            {
                generation.stop_reason = StopReason::Cancelled;
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
        let head = self.model.speculation().is_some() && generation.lag.usable();
        if head && !generation.lag.is_empty() {
            self.model.catch_up_head(&mut generation.lag)?;
        }
        let stats = &mut generation.stats;
        let remaining = generation.params.max_tokens - stats.sampled_tokens;
        let lookup_started = Instant::now();
        let draft = self
            .ngram
            .enabled
            .then(|| generation.suffixes.find(self.ngram.max_drafts))
            .flatten();
        stats.ngram.lookup += lookup_started.elapsed();
        let position = generation.prompt.len() + stats.sampled_tokens - 1;
        let drafts = draft.map_or_else(Vec::new, |mut draft| {
            let adaptive = generation
                .lookup_policy
                .depth(draft.match_len, self.ngram.max_drafts);
            let depth = draft_depth(adaptive, remaining, self.info.context, position);
            let filled = fill_verify_tile(depth, draft.tokens.len());
            let limit = draft_depth(filled, remaining, self.info.context, position);
            draft.tokens.truncate(limit);
            draft.tokens
        });
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
    /// emitted and record it in the prompt cache, or clear the cache after a
    /// cancellation.
    pub(super) fn finish_generation(
        &mut self,
        mut generation: ActiveGeneration,
    ) -> crate::Result<BonsaiGeneration> {
        let mut stats = std::mem::take(&mut generation.stats);
        stats.generated_tokens = generation.token_ids.len();
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
            self.clear_gpu_cache();
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
                // The prompt boundary's snapshot, read back now that decode
                // no longer needs the bandwidth.
                let prompt_snapshot = match generation.prompt_snapshot.take() {
                    Some(checkpoint) => self.model.prompt_snapshot_at(&checkpoint)?,
                    None => None,
                };
                self.save_prompt_checkpoint(&represented_tokens, false)?;
                if let Some(snapshot) = prompt_snapshot {
                    self.save_session_snapshot(
                        &prompt[..prompt.len() - 1],
                        session_id,
                        Some(snapshot),
                        !generation.persisted_reusable_boundary,
                        false,
                    )?;
                }
                // Chat templates re-render assistant turns, so the output end is
                // rarely a later prefix: keep it in (volatile) host memory only.
                self.save_session_snapshot(&represented_tokens, session_id, None, false, false)?;
            } else {
                self.clear_gpu_cache();
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
