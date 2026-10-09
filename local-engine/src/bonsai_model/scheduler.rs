//! Continuous batching: several generations advanced together.
//!
//! [`BonsaiEngine::admit`] prefills a request as a pass of its own and takes
//! its first sample; [`BonsaiEngine::step`] then advances every active
//! generation by one round. A generation alone runs the single-sequence round
//! it always ran (n-gram verification, an MTP speculative round, or one decode
//! step), so a lone request behaves exactly as before. Two or more decode one
//! token each in a single batched pass ([`BonsaiModel::decode_batch`]): every
//! projection reads its weights once for all of them. Requests join and leave
//! between steps.
//!
//! Each generation owns a full set of sequence buffers. One set is resident
//! in the model; the others are parked, and a swap exchanges handles only.
//! The GPU prompt-cache tier describes one particular set, so it is only
//! consulted while that set is resident (see [`BonsaiEngine::claim_gpu_cache`]).

use super::generation::ActiveGeneration;
use super::{
    BatchRow, BonsaiEngine, BonsaiGeneration, CancelToken, GenerateParams, PrefillProgress,
};
use crate::bonsai_native::{MAX_BATCH_SEQUENCES, SequenceState};

/// Free sequence buffer sets kept for the next request instead of being
/// released: one avoids reallocating ~80 MB of state per request under a
/// steady stream of overlapping requests.
const POOLED_SEQUENCES: usize = 1;

/// Fewest running generations that decode as one batch while an MTP head is
/// loaded; a lone generation takes whole speculative rounds. Two streams taking
/// speculative rounds in turn measured 32.7 tok/s aggregate against 36.8
/// batched on the M4 Pro, so batching starts at two.
const BATCH_THRESHOLD: usize = 2;

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

    /// Prefill `prompt` and take its first sample, emitting through `emit`.
    /// The generation then advances with every [`Self::step`] until it is
    /// returned there, finished.
    ///
    /// Runs while other generations are active: their decoding pauses for
    /// this prefill.
    pub fn admit(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        session_id: Option<&str>,
        cancel: CancelToken,
        progress: &mut dyn FnMut(PrefillProgress),
        emit: &mut dyn FnMut(&str) -> bool,
    ) -> crate::Result<u64> {
        if self.active.len() >= MAX_BATCH_SEQUENCES {
            return Err(crate::Error::QueueFull);
        }
        self.next_generation += 1;
        let id = self.next_generation;
        self.make_room_for(id)?;
        self.claim_gpu_cache();
        let result = self.begin_generation(id, prompt, params, session_id, cancel, progress);
        let mut generation = match result {
            Ok(generation) => generation,
            Err(error) => {
                // The resident set holds a partial prefill: free it.
                self.resident = None;
                self.clear_gpu_cache();
                return Err(error);
            }
        };
        if let Err(error) = Self::drain_generation(&self.tokenizer, &mut generation, emit) {
            self.resident = None;
            self.clear_gpu_cache();
            return Err(error);
        }
        self.active.push(generation);
        Ok(id)
    }

    /// Advance every active generation by one round and return those that
    /// finished, each with its outcome. `emit` receives each generation's
    /// text with its id.
    pub fn step(&mut self, emit: &mut dyn FnMut(u64, &str) -> bool) -> Vec<Finished> {
        let mut finished = Vec::new();
        for generation in &mut self.active {
            if !generation.done && generation.cancel.is_cancelled() {
                generation.mark_cancelled();
            }
        }
        let running = self
            .active
            .iter()
            .filter(|generation| !generation.done)
            .count();
        let outcome = match running {
            0 => Ok(()),
            1 => self.step_alone(),
            _ if self.model.speculation().is_some() && running < BATCH_THRESHOLD => {
                self.step_in_turn()
            }
            _ => self.step_batched(),
        };
        if let Err(error) = outcome {
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
        let mut index = 0;
        while index < self.active.len() {
            if !self.active[index].done {
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

    /// Cancel and return every active generation, as a worker shutting down
    /// or recovering from an error would.
    pub fn abort_all(&mut self) -> Vec<Finished> {
        for generation in &mut self.active {
            generation.mark_cancelled();
        }
        self.step(&mut |_, _| false)
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
            .filter(|generation| !generation.done)
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

    fn step_batched(&mut self) -> crate::Result<()> {
        let greedy = self
            .active
            .iter()
            .any(|generation| !generation.done && generation.sampler.selects_argmax());
        self.model.set_device_greedy(greedy);
        let speculating = self.model.speculation().is_some();
        let resident = self.resident;
        let mut participants = Vec::new();
        let mut rows = Vec::new();
        for (index, generation) in self.active.iter_mut().enumerate() {
            if generation.done {
                continue;
            }
            let Some(token) = generation.seed.take() else {
                continue;
            };
            let state = if resident == Some(generation.id) {
                None
            } else {
                Some(generation.state.as_mut().ok_or_else(|| {
                    crate::Error::InvalidArgument("parked generation has no state".into())
                })?)
            };
            participants.push(index);
            rows.push(BatchRow {
                token,
                state,
                lag: speculating.then_some(&mut generation.lag),
            });
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.model.take_gpu_time();
        self.model.decode_batch(&mut rows)?;
        drop(rows);
        let gpu = self.model.take_gpu_time() / participants.len() as u32;
        for (row, index) in participants.into_iter().enumerate() {
            let generation = &mut self.active[index];
            let started = std::time::Instant::now();
            let sample = self.model.sample_batch_row(row, &mut generation.sampler)?;
            generation.stats.sampling += started.elapsed();
            generation.stats.gpu += gpu;
            generation.stats.batched_tokens += 1;
            generation.pending.push_back(sample);
        }
        Ok(())
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
