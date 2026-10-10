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
    BatchRow, BonsaiEngine, BonsaiGeneration, CancelToken, GenerateParams, PrefillProgress,
    draft_depth,
};
use crate::bonsai_native::{HeadDraft, MAX_BATCH_SEQUENCES, PrefillRows, SequenceState};
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

/// Prompt tokens one step prefills while nothing decodes: four full blocks.
/// Bounded only so a request arriving behind a long prompt is admitted, and
/// can start its own shorter prefill, within about five seconds.
const PREFILL_CHUNK_ALONE: usize = 512;

/// Free sequence buffer sets kept for the next request instead of being
/// released: one avoids reallocating ~80 MB of state per request under a
/// steady stream of overlapping requests.
const POOLED_SEQUENCES: usize = 1;

/// Fewest running generations that decode as one batch while an MTP head is
/// loaded; a lone generation takes whole speculative rounds. Two streams taking
/// speculative rounds in turn measured 32.7 tok/s aggregate against 36.8
/// batched on the M4 Pro, so batching starts at two.
const BATCH_THRESHOLD: usize = 2;

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
        let reserved = self
            .active
            .iter()
            .map(ActiveGeneration::reserved_tokens)
            .sum::<usize>()
            + prompt_tokens
            + max_tokens
            + self.active.len() * state_tokens;
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
        self.admit_constrained(prompt, params, session_id, cancel, None)
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
    ) -> crate::Result<u64> {
        if self.active.len() >= MAX_BATCH_SEQUENCES {
            return Err(crate::Error::QueueFull);
        }
        self.next_generation += 1;
        let id = self.next_generation;
        self.make_room_for(id)?;
        self.claim_gpu_cache();
        match self.begin_generation(id, prompt, params, session_id, cancel, grammar) {
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
    /// others decode, and up to [`PREFILL_CHUNK_ALONE`] tokens while none do.
    /// `emit`
    /// receives each generation's text with its id, `progress` each prefill
    /// chunk boundary.
    pub fn step(
        &mut self,
        emit: &mut dyn FnMut(u64, &str) -> bool,
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
        let outcome = match running {
            0 => Ok(false),
            1 => self.step_alone().map(|()| false),
            _ if self.model.speculation().is_some() && running < BATCH_THRESHOLD => {
                self.step_in_turn().map(|()| false)
            }
            _ => self.step_batched(progress),
        };
        if let Err(error) = &outcome {
            // A failed pass leaves every participant's state undefined.
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
                    &mut |piece| emit(id, piece),
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
        let (index, remaining) = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, generation)| !generation.done)
            .filter_map(|(index, generation)| Some((index, generation.prefill_remaining()?)))
            .min_by_key(|&(_, remaining)| remaining)?;
        let short = self.active[index].prefill_total() <= PREFILL_WHOLE_DECODING;
        let budget = if running == 0 {
            PREFILL_CHUNK_ALONE
        } else if short {
            remaining
        } else {
            PREFILL_CHUNK_DECODING
        };
        let id = self.active[index].id;
        let result = self.make_resident(index).and_then(|()| {
            let mut generation = self.active.swap_remove(index);
            let result = self.prefill_round(&mut generation, budget, progress);
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

    /// Cancel and return every active generation, as a worker shutting down
    /// or recovering from an error would.
    pub fn abort_all(&mut self) -> Vec<Finished> {
        for generation in &mut self.active {
            generation.mark_cancelled();
        }
        self.step(&mut |_, _| false, &mut |_, _| {})
    }

    fn step_alone(&mut self) -> crate::Result<()> {
        self.step_in_turn()
    }

    /// One solo round for each running generation in turn: below the batch
    /// threshold, a speculative round each beats one batched plain token each.
    fn step_in_turn(&mut self) -> crate::Result<()> {
        let ids = self
            .active
            .iter()
            .filter(|generation| generation.decoding())
            .map(|generation| generation.id)
            .collect::<Vec<_>>();
        for id in ids {
            let Some(index) = self
                .active
                .iter()
                .position(|generation| generation.id == id)
            else {
                continue;
            };
            if self.active[index].cancel.is_cancelled() {
                self.active[index].mark_cancelled();
                continue;
            }
            self.make_resident(index)?;
            let mut generation = self.active.swap_remove(index);
            let result = self.solo_round(&mut generation);
            self.active.push(generation);
            result?;
        }
        Ok(())
    }

    /// One batched pass over every running generation: its seed and, where
    /// the cost model in [`policy`] finds them worth their rows, drafts from
    /// n-gram lookup or its MTP head, verified in the same pass; and, when a
    /// long prompt is prefilling, its next chunk (see [`Self::fold_candidate`]).
    /// Returns whether it prefilled a chunk, which is then the step's
    /// prefill.
    #[allow(clippy::too_many_lines)]
    fn step_batched(
        &mut self,
        progress: &mut dyn FnMut(u64, PrefillProgress),
    ) -> crate::Result<bool> {
        let greedy = self
            .active
            .iter()
            .any(|generation| generation.decoding() && generation.sampler.selects_argmax());
        let speculating = self.model.speculation().is_some();
        let participants = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, generation)| generation.decoding() && generation.seed.is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if participants.is_empty() {
            return Ok(false);
        }
        let fold = self.fold_candidate(participants.len());
        let fold_rows = fold.map_or(0, |(_, rows)| rows);
        // What each sequence could draft, and which drafts the step verifies.
        let mut lookups = vec![None; participants.len()];
        let mut candidates = vec![None; participants.len()];
        for (slot, &index) in participants.iter().enumerate() {
            let generation = &mut self.active[index];
            let remaining = generation.params.max_tokens - generation.stats.sampled_tokens;
            let position = generation.prompt.len() + generation.stats.sampled_tokens - 1;
            let lookup_started = std::time::Instant::now();
            let draft = self
                .ngram
                .enabled
                .then(|| generation.suffixes.find(self.ngram.max_drafts))
                .flatten();
            generation.stats.ngram.lookup += lookup_started.elapsed();
            if let Some(mut draft) = draft {
                let adaptive = generation
                    .lookup_policy
                    .depth(draft.match_len, self.ngram.max_drafts);
                let depth = draft_depth(adaptive, remaining, self.info.context, position);
                draft.tokens.truncate(depth);
                if !draft.tokens.is_empty() {
                    candidates[slot] = Some(policy::Candidate {
                        depth: draft.tokens.len(),
                        acceptance: generation.ngram_acceptance,
                        draft_ms: policy::COMMIT_MS,
                        head: false,
                    });
                    lookups[slot] = Some(draft.tokens);
                    continue;
                }
            }
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
            self.draft_from_heads(&participants, &heads, &mut drafts)?;
            for &(slot, _) in &heads {
                from_head[slot] = !drafts[slot].is_empty();
            }
        }
        self.model.set_device_greedy(greedy);
        let resident = self.resident;
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
        self.model.decode_batch_prefilling(&mut rows, prefill)?;
        let started = std::time::Instant::now();
        let mut committed = vec![1; rows.len()];
        let mut outcomes = Vec::with_capacity(rows.len());
        for (slot, (row, sampler)) in rows.iter().zip(samplers.iter_mut()).enumerate() {
            let verification = if row.drafts.is_empty() {
                let start = self.model.batch_row_start(slot)?;
                crate::sampler::Verification {
                    samples: vec![self.model.sample_batch_row(start, sampler)?],
                    accepted: 0,
                }
            } else {
                self.model.verify_batch_row(slot, &row.drafts, sampler)?
            };
            committed[slot] = verification.accepted + 1;
            outcomes.push((row.drafts.len(), verification));
        }
        self.model.commit_batch(&mut rows, &committed)?;
        drop(rows);
        drop(samplers);
        let sampling = started.elapsed() / participants.len() as u32;
        let gpu = self.model.take_gpu_time() / participants.len() as u32;
        for ((&index, (proposed, verification)), head) in
            participants.iter().zip(outcomes).zip(from_head)
        {
            let generation = &mut self.active[index];
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

    /// The prompt whose next chunk this batched step prefills inside its
    /// pass, and the chunk: the prompt [`Self::step_prefill`] would take,
    /// when it is a long one and its next rows reach no prompt-cache
    /// milestone. The chunk fills the pass to a multiple of eight rows, from
    /// [`PREFILL_FOLD_MIN`] to seven more.
    fn fold_candidate(&self, seeds: usize) -> Option<(usize, usize)> {
        let (index, _) = self
            .active
            .iter()
            .enumerate()
            .filter(|(_, generation)| !generation.done)
            .filter_map(|(index, generation)| Some((index, generation.prefill_remaining()?)))
            .min_by_key(|&(_, remaining)| remaining)?;
        let generation = &self.active[index];
        if generation.prefill_total() <= PREFILL_WHOLE_DECODING || generation.cancel.is_cancelled()
        {
            return None;
        }
        let run = generation
            .prefill
            .as_ref()?
            .plain_run(generation.prompt.len());
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
    /// alone, as its solo round would.
    fn draft_from_heads(
        &mut self,
        participants: &[usize],
        heads: &[(usize, usize)],
        drafts: &mut [Vec<u32>],
    ) -> crate::Result<()> {
        let (alone, together): (Vec<_>, Vec<_>) = heads.iter().partition(|&&(slot, _)| {
            self.active[participants[slot]]
                .sampler
                .greedy_draft()
                .applies_penalties()
        });
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
            let mut tokens = self.model.draft_resident(
                &mut generation.lag,
                seed,
                &generation.sampler,
                remaining.min(depth + 1),
            )?;
            tokens.truncate(depth);
            generation.stats.mtp.drafting += started.elapsed();
            drafts[slot] = tokens;
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
        let chains = self.model.draft_heads(&mut entries)?;
        drop(entries);
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
#[allow(clippy::expect_used)]
mod tests;
