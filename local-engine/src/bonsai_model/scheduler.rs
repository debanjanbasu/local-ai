//! Continuous batching: several generations advanced together.
//!
//! [`BonsaiEngine::admit`] restores a request's reusable prompt prefix;
//! [`BonsaiEngine::step`] then advances every active generation by one round.
//! A generation alone runs the single-sequence round it always ran (n-gram
//! verification, an MTP speculative round, or one decode step), so a lone
//! request behaves exactly as before. Two or more advance together in a
//! single batched pass ([`BonsaiModel::decode_batch_prefilling`]): every
//! projection reads its weights once for all of them, and a sequence whose
//! drafts the cost model in [`policy`] finds worth their rows verifies them
//! in the same pass, its head drafts made for all of them together
//! ([`BonsaiModel::draft_heads`]).
//!
//! A request's prompt prefills in bounded chunks, inside a batched step's
//! pass or after a step's decode round, so a long prompt delays the running
//! requests' next token by one chunk rather than by its whole prefill.
//! Requests join and leave between steps.
//!
//! Each generation owns a full set of sequence buffers. One set is resident
//! in the model; the others are parked, and a swap exchanges handles only.
//! The GPU prompt-cache tier describes one particular set, so it is only
//! consulted while that set is resident (see [`BonsaiEngine::claim_gpu_cache`]).

use super::generation::ActiveGeneration;
use super::{
    BatchRow, BonsaiEngine, BonsaiGeneration, CancelToken, GenerateParams, NgramSettings,
    PrefillProgress, draft_depth, fill_verify_tile,
};
use crate::bonsai_native::{HeadDraft, MAX_BATCH_SEQUENCES, PrefillRows, SequenceState};
use crate::sampler::TokenLogprob;
use crate::structured::Grammar;

mod policy;

/// Prompt tokens of a long prompt one step prefills in a pass of its own
/// while other generations decode (a lone stream taking solo rounds, or a
/// chunk reaching a prompt-cache milestone; a batched step prefills inside
/// its decode pass instead, see [`PREFILL_FOLD_MIN`]): their next token waits
/// for this chunk. A prefill block costs about
/// 9.2 ms a row from 8 to 128 rows on the M4 Pro, so the chunk trades the
/// others' inter-token gap against the prompt's own time to first token, not
/// prefill efficiency: beside four streams and a 9.2K-token prompt, chunks of
/// 16, 32 and 48 kept their tokens coming every 0.24, 0.39 and 0.54 s (p50)
/// and delayed the prompt's first token to 136, 112 and 104 s.
const PREFILL_CHUNK_DECODING: usize = 32;

/// Fewest prompt tokens a batched step prefills inside its decode pass; the
/// chunk then grows until the pass's rows fill their eight-row group, which
/// costs about what the group's first row does. A pass costs about 9.4 ms a
/// row from 24 rows on (M4 Pro, `prefill_fold_timings`), so beside four
/// streams a 28-token chunk takes 300 ms where a 36-token one takes 377 ms
/// for the same prompt tokens per second: the smaller one keeps the others'
/// tokens coming sooner. Against the step and a separate 32-token prefill
/// pass (about 400 ms beside four streams), both are faster.
const PREFILL_FOLD_MIN: usize = 24;

/// A prompt with at most this many tokens to prefill prefills whole in one
/// step while other generations decode: one prefill block, so a typical chat
/// prompt prefills in a single step. Longer prompts go in chunks to the end,
/// so their last block does not stall the others either.
const PREFILL_WHOLE_DECODING: usize = 128;

/// Free sequence buffer sets kept for the next request instead of being
/// released: one avoids reallocating ~80 MB of state per request under a
/// steady stream of overlapping requests.
const POOLED_SEQUENCES: usize = 1;

/// Move an acceptance estimate a quarter of the way toward a new observation.
fn blend(estimate: f64, observed: f64) -> f64 {
    0.75f64.mul_add(estimate, 0.25 * observed)
}

/// Move a head's acceptance estimate an eighth of the way back toward the
/// prior after a step it did not draft in. A chain is one to three drafts,
/// so one rejection moves the estimate a long way, and an estimate below
/// what drafting needs would otherwise never be observed again: two prose
/// streams whose heads were accepted 75% of the time drafted in 21 of their
/// rounds and gained nothing, against 10% with drafts every step.
fn recover(estimate: f64) -> f64 {
    estimate + (super::generation::INITIAL_MTP_ACCEPTANCE - estimate) / 8.0
}

/// A finished generation and its outcome.
pub type Finished = (u64, crate::Result<BonsaiGeneration>);

impl BonsaiEngine {
    /// Generations admitted and not yet returned by [`Self::step`].
    pub const fn active_generations(&self) -> usize {
        self.active.len()
    }

    /// Whether a request of `prompt_tokens + max_tokens` fits beside the
    /// active generations now.
    ///
    /// Admission at load sized the context so one sequence can grow its K/V
    /// to the whole of it. Concurrent sequences share that budget: their
    /// worst-case lengths, plus each extra sequence's fixed recurrent state
    /// expressed in tokens of K/V, must fit in the admitted context. A request
    /// that does not fit waits for others to finish rather than failing.
    pub fn can_admit(&self, prompt_tokens: usize, max_tokens: usize) -> bool {
        if self.active.is_empty() {
            return true;
        }
        if self.active.len() >= MAX_BATCH_SEQUENCES {
            return false;
        }
        let state_tokens = self.sequence_state_tokens();
        // Saturating: `max_tokens` comes from the request and may be huge.
        let reserved = self
            .active
            .iter()
            .map(ActiveGeneration::reserved_tokens)
            .fold(0, usize::saturating_add)
            .saturating_add(prompt_tokens)
            .saturating_add(max_tokens)
            .saturating_add(self.active.len().saturating_mul(state_tokens));
        reserved <= self.info.context
    }

    /// One parked sequence's fixed state, in tokens of K/V growth.
    fn sequence_state_tokens(&self) -> usize {
        let per_token = self.model.kv_layout().token_bytes() * crate::bonsai::LAYERS
            / crate::bonsai::FULL_INTERVAL;
        crate::bonsai_native::PROMPT_CHECKPOINT_BYTES.div_ceil(per_token.max(1))
    }

    /// Admit a request: restore its reusable prompt prefix and queue the rest
    /// of its prefill. [`Self::step`] prefills it in bounded chunks between
    /// the other generations' decode steps, takes its first sample, and then
    /// advances it every step until it is returned there, finished.
    pub fn admit(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        cancel: CancelToken,
    ) -> crate::Result<u64> {
        self.admit_constrained(prompt, params, session_id, cancel, None, None)
    }

    /// [`Self::admit`] under a compiled response format, which every
    /// target selection of the request then obeys.
    pub(crate) fn admit_constrained(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        cancel: CancelToken,
        grammar: Option<Grammar>,
        images: Option<std::sync::Arc<crate::bonsai_vision::PromptImages>>,
    ) -> crate::Result<u64> {
        if self.active.len() >= MAX_BATCH_SEQUENCES {
            return Err(crate::Error::QueueFull);
        }
        self.model.settle_decode()?;
        self.next_generation += 1;
        let id = self.next_generation;
        self.make_room_for(id)?;
        self.claim_gpu_cache();
        match self.begin_generation(id, prompt, params, session_id, cancel, grammar, images) {
            Ok(generation) => {
                self.active.push(generation);
                Ok(id)
            }
            Err(error) => {
                // The resident set may hold a partial restore: free it.
                self.resident = None;
                self.clear_gpu_cache();
                Err(error)
            }
        }
    }

    /// Advance every active generation by one round and return those that
    /// finished, each with its outcome.
    ///
    /// Generations past their prompt decode one round (batched, or alone),
    /// and the prompt with the fewest tokens left prefills: inside a batched
    /// round's pass when it is long (see [`Self::fold_candidate`]), else after
    /// the round, whole if it had at most [`PREFILL_WHOLE_DECODING`] tokens
    /// to prefill, else one [`PREFILL_CHUNK_DECODING`]-token chunk, while
    /// others decode, and one prefill block while none do.
    /// `emit`
    /// receives each generation's text with its id, `progress` each prefill
    /// chunk boundary.
    pub fn step(
        &mut self,
        emit: &mut dyn FnMut(u64, &str, Vec<TokenLogprob>) -> bool,
        progress: &mut dyn FnMut(u64, PrefillProgress),
    ) -> Vec<Finished> {
        let mut finished = Vec::new();
        for generation in &mut self.active {
            if !generation.done && generation.cancel.is_cancelled() {
                generation.mark_cancelled();
            }
        }
        let running = self
            .active
            .iter()
            .filter(|generation| generation.decoding())
            .count();
        // Two or more decode as one batch: two streams taking speculative
        // rounds in turn measured 32.7 tok/s aggregate against 36.8 batched
        // on the M4 Pro, so batching starts at two.
        let mut failed = Vec::new();
        // Only a lone decoding generation's next round continues a pass its
        // last round left in flight; everything else waits for it.
        let lone = self.active.len() == 1 && running == 1;
        let settled = if lone {
            Ok(())
        } else {
            self.model.settle_decode()
        };
        let outcome = settled.and_then(|()| match running {
            0 => Ok(false),
            1 => self.step_alone(&mut failed).map(|()| false),
            _ => self.step_batched(progress, &mut failed),
        });
        for (id, error) in failed {
            // Only these sequences' state is in doubt.
            if let Some(index) = self
                .active
                .iter()
                .position(|generation| generation.id == id)
            {
                let generation = self.active.remove(index);
                self.discard(generation);
            }
            finished.push((id, Err(error)));
        }
        if let Err(error) = &outcome {
            // A failure no sequence owns (an interrupted swap) leaves every
            // generation's state undefined.
            let message = error.to_string();
            for generation in std::mem::take(&mut self.active) {
                if let Some(state) = generation.state {
                    self.release(state);
                }
                finished.push((
                    generation.id,
                    Err(crate::Error::Generation(message.clone())),
                ));
            }
            self.resident = None;
            self.clear_gpu_cache();
            return finished;
        }
        let folded = outcome.unwrap_or(false);
        if !folded && let Some((id, error)) = self.step_prefill(running, progress) {
            finished.push((id, Err(error)));
        }
        let mut index = 0;
        while index < self.active.len() {
            if !self.active[index].done && self.active[index].prefill.is_none() {
                let id = self.active[index].id;
                let drained = Self::drain_generation(
                    &self.tokenizer,
                    &mut self.active[index],
                    &mut |piece, logprobs| emit(id, piece, logprobs),
                );
                if let Err(error) = drained {
                    let generation = self.active.remove(index);
                    self.discard(generation);
                    finished.push((id, Err(error)));
                    continue;
                }
            }
            if self.active[index].done {
                let generation = self.active.remove(index);
                let id = generation.id;
                finished.push((id, self.close(generation)));
                continue;
            }
            index += 1;
        }
        finished
    }

    /// This step's prefill, beside `running` decoding generations: the
    /// prompt with the fewest tokens left, whole if it fits in one step's
    /// budget, else one chunk. One prompt per step, so each starts decoding
    /// (and emits its first token) as soon as it is prefilled. A failure is
    /// that generation's alone: it is dropped and returned with its error.
    fn step_prefill(
        &mut self,
        running: usize,
        progress: &mut dyn FnMut(u64, PrefillProgress),
    ) -> Option<(u64, crate::Error)> {
        let (index, remaining) = self.next_prefill()?;
        if running > 0 && self.active[index].awaiting_disk() {
            // Its load is still running: it would only sit the step out.
            return None;
        }
        let short = self.active[index].prefill_total() <= PREFILL_WHOLE_DECODING;
        let budget = if running == 0 {
            // One prefill block, each its own GPU round trip either way: a
            // request arriving behind a long prompt is admitted, and can start
            // its own shorter prefill, after at most one block (about four
            // seconds for 512 tokens).
            self.info.prefill_chunk_size
        } else if short {
            remaining
        } else {
            PREFILL_CHUNK_DECODING
        };
        let id = self.active[index].id;
        let result = self.make_resident(index).and_then(|()| {
            let mut generation = self.active.swap_remove(index);
            let result = self
                .refresh_prefix(&mut generation)
                .and_then(|()| self.prefill_round(&mut generation, budget, progress));
            self.active.push(generation);
            result
        });
        let error = result.err()?;
        if let Some(index) = self
            .active
            .iter()
            .position(|generation| generation.id == id)
        {
            let generation = self.active.remove(index);
            self.discard(generation);
        }
        Some((id, error))
    }

    /// The prompt this step prefills and its tokens left: the one with the
    /// fewest, preferring prompts whose disk snapshot is not still loading.
    /// One that is gets the step only when no other prompt prefills, and then
    /// sits it out while others decode, or waits for its load when none do
    /// (see [`Self::refresh_prefix`]).
    fn next_prefill(&self) -> Option<(usize, usize)> {
        self.active
            .iter()
            .enumerate()
            .filter(|(_, generation)| !generation.done)
            .filter_map(|(index, generation)| {
                Some((
                    index,
                    generation.prefill_remaining()?,
                    generation.awaiting_disk(),
                ))
            })
            .min_by_key(|&(_, remaining, loading)| (loading, remaining))
            .map(|(index, remaining, _)| (index, remaining))
    }

    /// Cancel and return every active generation, as a worker shutting down
    /// or recovering from an error would.
    pub fn abort_all(&mut self) -> Vec<Finished> {
        for generation in &mut self.active {
            generation.mark_cancelled();
        }
        self.step(&mut |_, _, _| false, &mut |_, _| {})
    }

    /// The single-sequence round of the one running generation. Its round
    /// touches only its own state, so a failure there is its alone and goes
    /// to `failed`.
    fn step_alone(&mut self, failed: &mut Vec<(u64, crate::Error)>) -> crate::Result<()> {
        let Some(index) = self.active.iter().position(ActiveGeneration::decoding) else {
            return Ok(());
        };
        if self.active[index].cancel.is_cancelled() {
            self.active[index].mark_cancelled();
            return Ok(());
        }
        self.make_resident(index)?;
        let mut generation = self.active.swap_remove(index);
        let result = self.solo_round(&mut generation);
        let id = generation.id;
        self.active.push(generation);
        if let Err(error) = result {
            failed.push((id, error));
        }
        Ok(())
    }

    /// One batched pass over every running generation: its seed and, where
    /// the cost model in [`policy`] finds them worth their rows, drafts from
    /// n-gram lookup or its MTP head, verified in the same pass; and, when a
    /// long prompt is prefilling, its next chunk (see [`Self::fold_candidate`]).
    /// Returns whether it prefilled a chunk, which is then the step's
    /// prefill.
    ///
    /// A failure that belongs to some sequences goes to `failed` with their
    /// ids, and only they stop: each sequence's caches grow before any GPU
    /// work, so running out fails that sequence alone; a sequence whose drafts
    /// or selection fail is dropped from the pass or commits only its seed
    /// row; a failed pass or commit fails the sequences whose state it wrote.
    #[allow(clippy::too_many_lines)]
    fn step_batched(
        &mut self,
        progress: &mut dyn FnMut(u64, PrefillProgress),
        failed: &mut Vec<(u64, crate::Error)>,
    ) -> crate::Result<bool> {
        let greedy = self
            .active
            .iter()
            .any(|generation| generation.decoding() && generation.sampler.selects_argmax());
        let speculating = self.model.speculation().is_some();
        let mut participants = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, generation)| generation.decoding() && generation.seed.is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if participants.is_empty() {
            return Ok(false);
        }
        let mut fold = self.fold_candidate(participants.len());
        if let Some((index, count)) = fold
            && let Err(error) = self.reserve_for(index, count)
        {
            failed.push((self.active[index].id, error));
            fold = None;
        }
        let fold_rows = fold.map_or(0, |(_, rows)| rows);
        // What each sequence could draft, and which drafts the step verifies.
        let mut lookups = vec![None; participants.len()];
        let mut candidates = vec![None; participants.len()];
        let mut dropped = vec![false; participants.len()];
        for (slot, &index) in participants.iter().enumerate() {
            let generation = &mut self.active[index];
            let lookup = Self::lookup_draft(self.ngram, self.info.context, generation);
            if lookup.is_empty() {
                let remaining = generation.params.max_tokens - generation.stats.sampled_tokens;
                let position = generation.prompt.len() + generation.stats.sampled_tokens - 1;
                if let Some(head) = self.model.speculation()
                    && generation.lag.usable()
                {
                    let depth = draft_depth(head.depth, remaining, self.info.context, position);
                    candidates[slot] = (depth > 0).then_some(policy::Candidate {
                        depth,
                        acceptance: generation.mtp_acceptance,
                        draft_ms: policy::COMMIT_MS,
                        head: true,
                    });
                }
            } else {
                candidates[slot] = Some(policy::Candidate {
                    depth: lookup.len(),
                    acceptance: generation.ngram_acceptance,
                    draft_ms: policy::COMMIT_MS,
                    head: false,
                });
                lookups[slot] = Some(lookup);
            }
            // Its seed and every draft it may verify.
            let rows = 1 + candidates[slot].map_or(0, |candidate| candidate.depth);
            if let Err(error) = self.reserve_for(index, rows) {
                failed.push((self.active[index].id, error));
                dropped[slot] = true;
                candidates[slot] = None;
                lookups[slot] = None;
            }
        }
        let chosen = self.choose_drafts(participants.len() + fold_rows, &candidates);
        let mut drafts = vec![Vec::new(); participants.len()];
        let mut from_head = vec![false; participants.len()];
        let mut heads = Vec::new();
        for (slot, &depth) in chosen.iter().enumerate() {
            if depth == 0 {
                if candidates[slot].is_some_and(|candidate| candidate.head) {
                    let generation = &mut self.active[participants[slot]];
                    generation.mtp_acceptance = recover(generation.mtp_acceptance);
                }
                continue;
            }
            if let Some(mut tokens) = lookups[slot].take() {
                tokens.truncate(depth);
                drafts[slot] = tokens;
                continue;
            }
            heads.push((slot, depth));
        }
        if !heads.is_empty() {
            self.draft_from_heads(&participants, &heads, &mut drafts, &mut dropped, failed)?;
            for &(slot, _) in &heads {
                from_head[slot] = !drafts[slot].is_empty();
            }
        }
        // Sequences that failed so far leave the pass.
        let mut slot = 0;
        participants.retain(|_| {
            slot += 1;
            !dropped[slot - 1]
        });
        let mut slot = 0;
        drafts.retain(|_| {
            slot += 1;
            !dropped[slot - 1]
        });
        let mut slot = 0;
        from_head.retain(|_| {
            slot += 1;
            !dropped[slot - 1]
        });
        if participants.is_empty() {
            return Ok(false);
        }
        self.model.set_device_greedy(greedy);
        // The most candidates any participant draws from; each takes its own.
        self.model.set_device_top_k(
            participants
                .iter()
                .filter_map(|&index| self.active[index].sampler.device_top_k())
                .max(),
        );
        // The most alternatives any participant reports; each takes its own.
        self.model.set_logprobs(
            participants
                .iter()
                .filter_map(|&index| self.active[index].sampler.logprobs())
                .max(),
        );
        let resident = self.resident;
        // Everyone whose state the pass writes, collected before the rows
        // borrow `self.active`.
        #[allow(clippy::needless_collect)]
        let written = participants
            .iter()
            .chain(fold.iter().map(|(index, _)| index))
            .map(|&index| self.active[index].id)
            .collect::<Vec<_>>();
        let mut rows = Vec::with_capacity(participants.len());
        let mut samplers = Vec::with_capacity(participants.len());
        let mut drafts = drafts.into_iter();
        let mut prefill = None;
        for (index, generation) in self.active.iter_mut().enumerate() {
            if let Some((fold, count)) = fold
                && fold == index
            {
                let done = generation
                    .prefill
                    .as_ref()
                    .map_or(0, super::checkpoints::PromptPrefill::done);
                let state = if resident == Some(generation.id) {
                    None
                } else {
                    Some(generation.state.as_mut().ok_or_else(|| {
                        crate::Error::InvalidArgument("parked generation has no state".into())
                    })?)
                };
                prefill = Some(PrefillRows {
                    tokens: &generation.prompt[done..done + count],
                    state,
                });
                continue;
            }
            if !participants.contains(&index) {
                continue;
            }
            let token = generation.seed.take().ok_or_else(|| {
                crate::Error::InvalidArgument("batched generation has no seed".into())
            })?;
            let state = if resident == Some(generation.id) {
                None
            } else {
                Some(generation.state.as_mut().ok_or_else(|| {
                    crate::Error::InvalidArgument("parked generation has no state".into())
                })?)
            };
            rows.push(BatchRow {
                token,
                drafts: drafts.next().unwrap_or_default(),
                state,
                lag: speculating.then_some(&mut generation.lag),
            });
            samplers.push(&mut generation.sampler);
        }
        self.model.take_gpu_time();
        if let Err(error) = self.model.decode_batch_prefilling(&mut rows, prefill) {
            let message = error.to_string();
            failed.extend(
                written
                    .into_iter()
                    .map(|id| (id, crate::Error::Generation(message.clone()))),
            );
            return Ok(false);
        }
        let started = std::time::Instant::now();
        let mut committed = vec![1; rows.len()];
        let mut outcomes = Vec::with_capacity(rows.len());
        for (slot, (row, sampler)) in rows.iter().zip(samplers.iter_mut()).enumerate() {
            let verification = if row.drafts.is_empty() {
                self.model.batch_row_start(slot).and_then(|start| {
                    Ok(crate::sampler::Verification {
                        samples: vec![self.model.sample_batch_row(start, sampler)?],
                        accepted: 0,
                    })
                })
            } else {
                self.model.verify_batch_row(slot, &row.drafts, sampler)
            };
            // A failed selection keeps only the seed row, so the commit stays
            // valid for everyone; the sequence then stops.
            committed[slot] = verification
                .as_ref()
                .map_or(1, |verification| verification.accepted + 1);
            outcomes.push((row.drafts.len(), verification));
        }
        let commit = self.model.commit_batch(&mut rows, &committed);
        let verifying = rows
            .iter()
            .map(|row| !row.drafts.is_empty())
            .collect::<Vec<_>>();
        drop(rows);
        drop(samplers);
        let sampling = started.elapsed() / participants.len() as u32;
        let gpu = self.model.take_gpu_time() / participants.len() as u32;
        let commit_failure = commit.err().map(|error| error.to_string());
        for (slot, ((&index, (proposed, verification)), head)) in
            participants.iter().zip(outcomes).zip(from_head).enumerate()
        {
            let generation = &mut self.active[index];
            if let Some(message) = commit_failure.as_ref().filter(|_| verifying[slot]) {
                failed.push((generation.id, crate::Error::Generation(message.clone())));
                continue;
            }
            let verification = match verification {
                Ok(verification) => verification,
                Err(error) => {
                    failed.push((generation.id, error));
                    continue;
                }
            };
            let accepted = verification.accepted;
            generation.stats.sampling += sampling;
            generation.stats.gpu += gpu;
            generation.stats.batched_tokens += verification.samples.len();
            if proposed > 0 {
                let observed = accepted as f64 / proposed as f64;
                if head {
                    let mtp = &mut generation.stats.mtp;
                    mtp.rounds += 1;
                    mtp.proposed_tokens += proposed;
                    mtp.accepted_tokens += accepted;
                    mtp.verified_tokens += proposed + 1;
                    generation.mtp_acceptance = blend(generation.mtp_acceptance, observed);
                } else {
                    let ngram = &mut generation.stats.ngram;
                    ngram.rounds += 1;
                    ngram.proposed_tokens += proposed;
                    ngram.accepted_tokens += accepted;
                    generation
                        .lookup_policy
                        .observe(accepted, proposed, self.ngram.max_drafts);
                    generation.ngram_acceptance = blend(generation.ngram_acceptance, observed);
                }
            }
            generation.pending.extend(verification.samples);
        }
        let Some((index, count)) = fold else {
            return Ok(false);
        };
        let generation = &mut self.active[index];
        let id = generation.id;
        if let Some(plan) = generation.prefill.as_mut() {
            let tokens = plan.done();
            plan.advance(count);
            generation.prefill_chunks += 1;
            progress(
                id,
                PrefillProgress {
                    tokens,
                    chunks: generation.prefill_chunks,
                },
            );
        }
        Ok(true)
    }

    /// Grow the caches of the active generation at `index`, resident or
    /// parked, for `rows` more rows before any GPU work of the step reads
    /// them, so running out fails it alone.
    fn reserve_for(&mut self, index: usize, rows: usize) -> crate::Result<()> {
        let generation = &mut self.active[index];
        #[cfg(test)]
        if self.fail_reservation == Some(generation.id) {
            return Err(crate::Error::Generation(
                "injected K/V reservation failure".into(),
            ));
        }
        let state = if self.resident == Some(generation.id) {
            None
        } else {
            Some(generation.state.as_mut().ok_or_else(|| {
                crate::Error::InvalidArgument("parked generation has no state".into())
            })?)
        };
        self.model.reserve_sequence(state, rows)
    }

    /// `generation`'s n-gram lookup drafts for its next round, solo or
    /// batched: the matched continuation, as deep as its lookup policy allows,
    /// filled to its verify tile, and never past its token budget or the
    /// context. Empty when nothing matches.
    pub(super) fn lookup_draft(
        ngram: NgramSettings,
        context: usize,
        generation: &mut ActiveGeneration,
    ) -> Vec<u32> {
        let remaining = generation.params.max_tokens - generation.stats.sampled_tokens;
        let position = generation.prompt.len() + generation.stats.sampled_tokens - 1;
        let started = std::time::Instant::now();
        let draft = ngram
            .enabled
            .then(|| generation.suffixes.find(ngram.max_drafts))
            .flatten();
        generation.stats.ngram.lookup += started.elapsed();
        draft.map_or_else(Vec::new, |mut draft| {
            let adaptive = generation
                .lookup_policy
                .depth(draft.match_len, ngram.max_drafts);
            let depth = draft_depth(adaptive, remaining, context, position);
            let filled = fill_verify_tile(depth, draft.tokens.len());
            draft
                .tokens
                .truncate(draft_depth(filled, remaining, context, position));
            draft.tokens
        })
    }

    /// The prompt whose next chunk this batched step prefills inside its
    /// pass, and the chunk: the prompt [`Self::step_prefill`] would take,
    /// when it is a long one and its next rows reach no prompt-cache
    /// milestone. The chunk fills the pass to a multiple of eight rows, from
    /// [`PREFILL_FOLD_MIN`] to seven more.
    fn fold_candidate(&self, seeds: usize) -> Option<(usize, usize)> {
        let (index, _) = self.next_prefill()?;
        let generation = &self.active[index];
        if generation.prefill_total() <= PREFILL_WHOLE_DECODING || generation.cancel.is_cancelled()
        {
            return None;
        }
        let plan = generation.prefill.as_ref()?;
        // A prompt's first chunk goes through `step_prefill`, which first
        // looks for a longer cached prefix than the one admission restored.
        if plan.done() == plan.reused {
            return None;
        }
        let run = plan.plain_run(generation.prompt.len());
        let chunk = (seeds + PREFILL_FOLD_MIN).next_multiple_of(8) - seeds;
        let limit = self
            .model
            .max_batch_rows()
            .min(self.model.max_prefill_fold())
            .saturating_sub(seeds);
        let rows = chunk.min(run).min(limit);
        (rows > 0).then_some((index, rows))
    }

    /// Head drafts of `depth` for each `(slot, depth)` of `heads`, slots
    /// indexing `participants`: every sequence's chain drafted together, in
    /// one head pass per depth. A sampler with penalties drafts on the host,
    /// alone, as its solo round would. Sequences whose drafting fails are
    /// marked `dropped` and go to `failed`.
    fn draft_from_heads(
        &mut self,
        participants: &[usize],
        heads: &[(usize, usize)],
        drafts: &mut [Vec<u32>],
        dropped: &mut [bool],
        failed: &mut Vec<(u64, crate::Error)>,
    ) -> crate::Result<()> {
        let (alone, together): (Vec<_>, Vec<_>) = heads
            .iter()
            .partition(|&&(slot, _)| self.active[participants[slot]].sampler.applies_penalties());
        for (slot, depth) in alone {
            // Host drafting reads the resident sequence's head: bring the
            // sequence in (a handle swap) and draft as its solo round does.
            let index = participants[slot];
            self.make_resident(index)?;
            let generation = &mut self.active[index];
            let remaining = generation.params.max_tokens - generation.stats.sampled_tokens;
            let seed = generation.seed.unwrap_or_default();
            self.model.set_cancel(generation.cancel.clone());
            let started = std::time::Instant::now();
            match self.model.draft_resident(
                &mut generation.lag,
                seed,
                &generation.sampler,
                remaining.min(depth + 1),
            ) {
                Ok(mut tokens) => {
                    tokens.truncate(depth);
                    drafts[slot] = tokens;
                }
                Err(error) => {
                    failed.push((generation.id, error));
                    dropped[slot] = true;
                }
            }
            generation.stats.mtp.drafting += started.elapsed();
        }
        if together.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let resident = self.resident;
        let mut slots = Vec::with_capacity(together.len());
        let mut entries = Vec::with_capacity(together.len());
        for (index, generation) in self.active.iter_mut().enumerate() {
            let Some(slot) = participants.iter().position(|&entry| entry == index) else {
                continue;
            };
            let Some(&(_, depth)) = together.iter().find(|&&(entry, _)| entry == slot) else {
                continue;
            };
            let state = if resident == Some(generation.id) {
                None
            } else {
                Some(generation.state.as_mut().ok_or_else(|| {
                    crate::Error::InvalidArgument("parked generation has no state".into())
                })?)
            };
            slots.push(index);
            entries.push(HeadDraft {
                seed: generation.seed.ok_or_else(|| {
                    crate::Error::InvalidArgument("batched generation has no seed".into())
                })?,
                depth,
                state,
                lag: &mut generation.lag,
                sampler: &generation.sampler,
            });
        }
        let chains = self.model.draft_heads(&mut entries);
        drop(entries);
        let chains = match chains {
            Ok(chains) => chains,
            Err(error) => {
                // Every drafting head is in an unknown state.
                let message = error.to_string();
                for &(slot, _) in &together {
                    failed.push((
                        self.active[participants[slot]].id,
                        crate::Error::Generation(message.clone()),
                    ));
                    dropped[slot] = true;
                }
                return Ok(());
            }
        };
        let share = started.elapsed() / slots.len() as u32;
        for (index, chain) in slots.into_iter().zip(chains) {
            self.active[index].stats.mtp.drafting += share;
            if let Some(slot) = participants.iter().position(|&entry| entry == index) {
                drafts[slot] = chain;
            }
        }
        Ok(())
    }

    /// Drafts each batched sequence verifies, by the cost model.
    /// `rows` counts the step's rows before drafts: one per sequence and any
    /// prompt chunk prefilled in the same pass.
    #[cfg(not(test))]
    fn choose_drafts(&self, rows: usize, candidates: &[Option<policy::Candidate>]) -> Vec<usize> {
        policy::choose(
            candidates.len(),
            rows,
            candidates,
            self.model.max_batch_rows(),
        )
    }

    #[cfg(test)]
    fn choose_drafts(&self, rows: usize, candidates: &[Option<policy::Candidate>]) -> Vec<usize> {
        let max_rows = self.model.max_batch_rows();
        if !self.force_batched_drafts {
            return policy::choose(candidates.len(), rows, candidates, max_rows);
        }
        let mut rows = rows;
        candidates
            .iter()
            .map(|candidate| {
                let depth = candidate
                    .map_or(0, |candidate| candidate.depth)
                    .min(max_rows - rows);
                rows += depth;
                depth
            })
            .collect()
    }

    /// Make the active generation at `index` resident, parking whichever
    /// sequence was.
    fn make_resident(&mut self, index: usize) -> crate::Result<()> {
        let id = self.active[index].id;
        if self.resident == Some(id) {
            return Ok(());
        }
        let mut slot = self.active[index].state.take().ok_or_else(|| {
            crate::Error::InvalidArgument("generation has no parked state".into())
        })?;
        self.model.swap_sequence(&mut slot)?;
        self.park_outgoing(slot);
        self.resident = Some(id);
        Ok(())
    }

    /// Give the set that was just swapped out to the generation that owns it,
    /// or to the pool when no active generation does.
    fn park_outgoing(&mut self, slot: SequenceState) {
        if let Some(owner) = self.resident
            && let Some(generation) = self
                .active
                .iter_mut()
                .find(|generation| generation.id == owner)
        {
            generation.state = Some(slot);
            return;
        }
        self.release(slot);
    }

    /// Make a buffer set resident for new generation `id`: the resident set
    /// if no active generation owns it, else a pooled or fresh one, preferring
    /// the set the GPU prompt cache describes.
    fn make_room_for(&mut self, id: u64) -> crate::Result<()> {
        if let Some(owner) = self.resident
            && self.active.iter().any(|generation| generation.id == owner)
        {
            let cached = self
                .pool
                .iter()
                .position(|state| state.id() == self.cache_state);
            let mut slot = match cached.or_else(|| self.pool.len().checked_sub(1)) {
                Some(index) => self.pool.swap_remove(index),
                None => self.model.new_sequence()?,
            };
            self.model.swap_sequence(&mut slot)?;
            self.park_outgoing(slot);
        } else if self.cache_state != self.model.state_id()
            && let Some(index) = self
                .pool
                .iter()
                .position(|state| state.id() == self.cache_state)
        {
            // The free resident set is not the cached one but a pooled set is:
            // bring the cached one back.
            let mut slot = self.pool.swap_remove(index);
            self.model.swap_sequence(&mut slot)?;
            self.release(slot);
        }
        self.resident = Some(id);
        Ok(())
    }

    /// Keep a free set for reuse, or drop it once the pool is full. The set
    /// the GPU prompt cache describes is kept in preference.
    fn release(&mut self, state: SequenceState) {
        self.pool.push(state);
        while self.pool.len() > POOLED_SEQUENCES {
            let victim = self
                .pool
                .iter()
                .position(|state| state.id() != self.cache_state)
                .unwrap_or(0);
            self.pool.swap_remove(victim);
        }
    }

    /// Finish a generation: make it resident and run its tail.
    fn close(&mut self, generation: ActiveGeneration) -> crate::Result<BonsaiGeneration> {
        if let Err(error) = self.model.settle_decode() {
            self.discard(generation);
            return Err(error);
        }
        let id = generation.id;
        self.active.push(generation);
        let index = self.active.len() - 1;
        let resident = self.make_resident(index);
        let generation = self.active.remove(index);
        resident?;
        let result = self.finish_generation(generation);
        // The finished set stays resident but belongs to no generation.
        if self.resident == Some(id) {
            self.resident = None;
        }
        result
    }

    /// Drop a generation that failed, releasing its buffers.
    fn discard(&mut self, generation: ActiveGeneration) {
        // Its state is in doubt either way; only the wait matters.
        let _ = self.model.settle_decode();
        if self.resident == Some(generation.id) {
            self.resident = None;
            self.clear_gpu_cache();
        } else if let Some(state) = generation.state {
            self.release(state);
        }
    }

    /// Make the GPU prompt-cache tier describe the resident buffer set,
    /// dropping it if it described another.
    pub(super) fn claim_gpu_cache(&mut self) {
        if self.cache_state != self.model.state_id() {
            self.cached_tokens.clear();
            self.prompt_checkpoints.clear();
            self.cache_state = self.model.state_id();
        }
    }

    pub(super) fn clear_gpu_cache(&mut self) {
        self.cached_tokens.clear();
        self.prompt_checkpoints.clear();
    }
}

#[cfg(test)]
mod tests;
