use super::{
    BonsaiEngine, BonsaiGeneration, BonsaiModel, CancelToken, GenerateParams, GenerationStats,
    Instant, LookupPolicy, MtpStats, PromptCacheSource, Sampler, SamplingParams, SamplingResult,
    StopReason, VOCAB, fill_verify_tile,
};
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
    #[allow(clippy::too_many_lines)]
    pub fn generate_session_cancellable(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        mut emit: impl FnMut(&str) -> bool,
        cancel: &CancelToken,
    ) -> crate::Result<BonsaiGeneration> {
        // Installed before anything can prefill, so the model always polls the
        // token belonging to the request in flight.
        self.model.set_cancel(cancel.clone());
        self.validate_generation(prompt, params)?;
        let started = Instant::now();
        let mut stats = GenerationStats {
            prompt_tokens: prompt.len(),
            ..GenerationStats::default()
        };
        let mut token_ids = Vec::new();
        let mut stop_reason = StopReason::TokenLimit;
        if params.max_tokens == 0 {
            return Ok(BonsaiGeneration {
                text: String::new(),
                token_ids,
                stop_reason,
                stats,
                cache_source: PromptCacheSource::None,
            });
        }
        self.model.take_gpu_time();
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
        sampler.observe(prompt);
        let (reused, cache_source, prompt_snapshot, persisted_reusable_boundary) =
            self.prepare_prompt(prompt, session_id)?;
        stats.reused_prompt_tokens = reused;
        stats.prefill = started.elapsed();
        let mut decoder = self.tokenizer.stream_decoder();
        let mut suffixes = self.suffix_store.session(prompt, self.ngram.min_match);
        let mut lookup_policy = LookupPolicy::new(self.ngram.max_drafts);
        let mut pending = std::collections::VecDeque::new();
        if self.model.take_cancel_observed() {
            // Prefill stopped between chunks, so `scratch.logits` still holds an
            // older row: there is nothing new to sample. Fall through to the
            // shared tail, which clears the prompt cache for a cancelled stop.
            stop_reason = StopReason::Cancelled;
        } else {
            pending.push_back(sample_current(&mut self.model, &mut sampler, &mut stats)?);
        }
        while let Some(sample) = pending.pop_front() {
            stats.sampled_tokens += 1;
            stats.first_token.get_or_insert_with(|| started.elapsed());
            if sample.is_eos {
                stop_reason = StopReason::Eos;
                break;
            }
            token_ids.push(sample.token_id);
            suffixes.append(sample.token_id);
            sampler.observe(&[sample.token_id]);
            if let Some(piece) = decoder(sample.token_id)?
                && !emit(&piece)
            {
                stop_reason = StopReason::Cancelled;
                stats.sampled_tokens += pending.len();
                break;
            }
            if stats.sampled_tokens >= params.max_tokens {
                break;
            }
            if !pending.is_empty() {
                continue;
            }
            let remaining = params.max_tokens - stats.sampled_tokens;
            let lookup_started = Instant::now();
            let draft = self
                .ngram
                .enabled
                .then(|| suffixes.find(self.ngram.max_drafts))
                .flatten();
            stats.ngram.lookup += lookup_started.elapsed();
            let drafts = draft.map_or_else(Vec::new, |mut draft| {
                let adaptive = lookup_policy.depth(draft.match_len, self.ngram.max_drafts);
                let depth = draft_depth(
                    adaptive,
                    remaining,
                    self.info.context,
                    prompt.len() + stats.sampled_tokens - 1,
                );
                let filled = fill_verify_tile(depth, draft.tokens.len());
                let limit = draft_depth(
                    filled,
                    remaining,
                    self.info.context,
                    prompt.len() + stats.sampled_tokens - 1,
                );
                draft.tokens.truncate(limit);
                draft.tokens
            });
            if !drafts.is_empty() {
                let batch = self
                    .model
                    .ngram_step(sample.token_id, &drafts, &mut sampler)?;
                stats.sampling += batch.sampling;
                stats.ngram.rounds += batch.ngram.rounds;
                stats.ngram.proposed_tokens += batch.ngram.proposed_tokens;
                stats.ngram.accepted_tokens += batch.ngram.accepted_tokens;
                lookup_policy.observe(
                    batch.ngram.accepted_tokens,
                    batch.ngram.proposed_tokens,
                    self.ngram.max_drafts,
                );
                pending.extend(batch.samples);
            } else if self.model.speculation().is_some() {
                let batch =
                    self.model
                        .speculative_step(sample.token_id, &mut sampler, remaining)?;
                stats.sampling += batch.sampling;
                accumulate_mtp(&mut stats.mtp, &batch.stats);
                pending.extend(batch.samples);
            } else {
                self.model.decode(sample.token_id)?;
            }
            if self.model.take_cancel_observed() {
                // The round below was not submitted, so `pending` stayed empty:
                // do not refill it from logits that were never recomputed.
                stop_reason = StopReason::Cancelled;
                stats.sampled_tokens += pending.len();
                pending.clear();
                break;
            }
            if pending.is_empty() {
                pending.push_back(sample_current(&mut self.model, &mut sampler, &mut stats)?);
            }
        }
        stats.generated_tokens = token_ids.len();
        drop(decoder);
        if stop_reason == StopReason::Cancelled {
            self.cached_tokens.clear();
            self.prompt_checkpoints.clear();
        } else {
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
                self.save_prompt_checkpoint(&represented_tokens, false)?;
                if let Some(snapshot) = prompt_snapshot {
                    self.save_session_snapshot(
                        &prompt[..prompt.len() - 1],
                        session_id,
                        Some(snapshot),
                        !persisted_reusable_boundary,
                        false,
                    )?;
                }
                // Chat templates re-render assistant turns, so the output end is
                // rarely a later prefix: keep it in (volatile) host memory only.
                self.save_session_snapshot(&represented_tokens, session_id, None, false, false)?;
            } else {
                self.cached_tokens.clear();
                self.prompt_checkpoints.clear();
            }
        }
        stats.elapsed = started.elapsed();
        stats.gpu = self.model.take_gpu_time();
        self.suffix_store.remember(suffixes.history().to_vec());
        Ok(BonsaiGeneration {
            text: self.tokenizer.decode(&token_ids, false)?,
            token_ids,
            stop_reason,
            stats,
            cache_source,
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
