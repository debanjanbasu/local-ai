use crate::bonsai_model::{BonsaiEngine, PromptCacheSource};
use crate::prompt_cache;

/// Shortest reusable boundary a prefill checkpoints: a shorter prefix costs
/// less to prefill again (about a second) than its snapshot does to copy.
pub(in crate::bonsai_model) const SHARED_PREFIX_MIN_TOKENS: usize = 128;

/// The reusable-prefix arithmetic for one rendered prompt.
pub(super) struct ReuseBounds {
    /// Longest common prefix with the previously cached prompt.
    pub(super) lcp: usize,
    /// Ceiling on a GPU checkpoint position.
    pub(super) reusable: usize,
    /// Ceiling on a host or disk snapshot position.
    pub(super) snapshot_reusable: usize,
    /// Latest position a reusable boundary may occupy.
    pub(super) penultimate: usize,
    // Recurrent state is valid only at an exact position. Materialize one
    // semantic checkpoint per prefill: prefer a newly observed divergence,
    // otherwise the end of the system turn. The minimum
    // (`SHARED_PREFIX_MIN_TOKENS`) avoids copies for short prefixes.
    pub(super) minimum: usize,
}

impl ReuseBounds {
    pub(super) fn new(lcp: usize, prompt_len: usize, minimum: usize) -> Self {
        // The next turn's re-rendered history usually diverges at the final
        // prompt token (the assistant-turn opener), so the reusable boundary
        // for checkpoints and session snapshots is the penultimate position.
        let penultimate = prompt_len - 1;
        let snapshot_reusable = prompt_len.saturating_sub(1);
        Self {
            lcp,
            reusable: lcp.min(snapshot_reusable),
            snapshot_reusable,
            penultimate,
            minimum,
        }
    }
}

impl BonsaiEngine {
    pub(super) fn reuse_bounds(&self, prompt: &[u32]) -> ReuseBounds {
        let lcp = self
            .cached_tokens
            .iter()
            .zip(prompt)
            .take_while(|(cached, new)| cached == new)
            .count();
        ReuseBounds::new(lcp, prompt.len(), SHARED_PREFIX_MIN_TOKENS)
    }

    /// The longest prefix of `prompt` [`Self::restore_reusable_prefix`]
    /// would restore into the resident set now, without restoring it.
    pub(in crate::bonsai_model) fn reusable_prefix(&mut self, prompt: &[u32]) -> usize {
        self.collect_disk_writes();
        let bounds = self.reuse_bounds(prompt);
        self.candidates(prompt, &bounds, None)
            .iter()
            .map(|&(_, _, position)| position)
            .max()
            .unwrap_or(0)
    }

    /// Restore the longest reusable prefix any tier holds: a live GPU
    /// checkpoint, a host session snapshot or a disk snapshot, the cheaper
    /// restore first on a tie. A tier holding a shorter prefix never shadows
    /// a longer one in a slower tier: after a compaction the GPU tier can hold
    /// an old boundary of the conversation while disk holds snapshots turns
    /// newer. A restore that fails falls back to the next candidate, and with
    /// none the sequence starts empty.
    pub(super) fn restore_reusable_prefix(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        session_id: Option<&str>,
    ) -> crate::Result<(usize, PromptCacheSource)> {
        let mut rows_kept = true;
        for (source, index, _) in self.candidates(prompt, bounds, session_id) {
            let restored = match source {
                // A failed snapshot restore may have overwritten the live K/V
                // rows a checkpoint relies on.
                PromptCacheSource::Gpu if !rows_kept => continue,
                PromptCacheSource::Gpu => self.restore_gpu_checkpoint(index)?,
                PromptCacheSource::Host => self.restore_host_snapshot(index)?,
                _ => self.restore_disk_snapshot(index),
            };
            if let Some(position) = restored {
                return Ok((position, source));
            }
            rows_kept &= source == PromptCacheSource::Gpu;
        }
        self.model.reset();
        Ok((0, PromptCacheSource::None))
    }

    /// Each tier's best entry for `prompt` as (tier, index, position),
    /// longest first and in tier order on a tie. Waits for a disk write in
    /// flight that would beat all of them, then counts it.
    fn candidates(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        session_id: Option<&str>,
    ) -> Vec<(PromptCacheSource, usize, usize)> {
        // Checkpoints describe the buffer set `cache_state` names: their K/V
        // rows are that set's live caches.
        let gpu = (self.cache_state == self.model.state_id())
            .then(|| {
                self.prompt_checkpoints
                    .iter()
                    .enumerate()
                    .filter(|(_, checkpoint)| {
                        checkpoint.state.position <= bounds.reusable
                            && checkpoint.tokens.as_slice() == &prompt[..checkpoint.state.position]
                    })
                    .max_by_key(|&(index, checkpoint)| (checkpoint.state.position, index))
                    .map(|(index, checkpoint)| (index, checkpoint.state.position))
            })
            .flatten();
        let host = self
            .session_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                snapshot.state.position() <= bounds.snapshot_reusable
                    && snapshot.tokens.as_slice() == &prompt[..snapshot.state.position()]
            })
            .max_by_key(|(_, snapshot)| {
                (
                    snapshot.state.position(),
                    snapshot.session_id.as_deref() == session_id,
                )
            })
            .map(|(index, snapshot)| (index, snapshot.state.position()));
        let held = gpu
            .into_iter()
            .chain(host)
            .map(|(_, position)| position)
            .max();
        self.await_pending_disk_write(prompt, bounds, held.unwrap_or(0));
        let disk = self
            .disk_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                snapshot.tokens.len() <= bounds.snapshot_reusable
                    && prompt.starts_with(&snapshot.tokens)
            })
            .max_by_key(|(_, snapshot)| {
                (
                    snapshot.tokens.len(),
                    snapshot.session_id.as_deref() == session_id,
                )
            })
            .map(|(index, snapshot)| (index, snapshot.tokens.len()));
        let mut candidates = [
            (PromptCacheSource::Gpu, gpu),
            (PromptCacheSource::Host, host),
            (PromptCacheSource::Disk, disk),
        ]
        .into_iter()
        .filter_map(|(source, entry)| entry.map(|(index, position)| (source, index, position)))
        .collect::<Vec<_>>();
        // Stable: equal lengths keep tier order.
        candidates.sort_by_key(|&(_, _, position)| std::cmp::Reverse(position));
        candidates
    }

    /// Restore GPU checkpoint `index`. One the model refuses (purged) is
    /// evicted and restores nothing.
    fn restore_gpu_checkpoint(&mut self, index: usize) -> crate::Result<Option<usize>> {
        if !self
            .model
            .restore_prompt_checkpoint(&self.prompt_checkpoints[index].state)?
        {
            self.prompt_checkpoints.remove(index);
            return Ok(None);
        }
        self.prompt_checkpoints.rotate_left(index + 1);
        let position = self.prompt_checkpoints[self.prompt_checkpoints.len() - 1]
            .state
            .position;
        Ok(Some(position))
    }

    /// Restore host session snapshot `index`. One the model refuses is
    /// evicted and restores nothing.
    fn restore_host_snapshot(&mut self, index: usize) -> crate::Result<Option<usize>> {
        if !self
            .model
            .restore_host_prompt_snapshot(&self.session_snapshots[index].state)?
        {
            self.session_snapshots.remove(index);
            return Ok(None);
        }
        let snapshot = self.session_snapshots.remove(index);
        let position = snapshot.state.position();
        self.session_snapshots.push(snapshot);
        self.prompt_checkpoints.clear();
        Ok(Some(position))
    }

    /// Restore disk snapshot `index`. One that fails to load or restore is a
    /// cache miss, never a failed request: it is dropped from the index.
    fn restore_disk_snapshot(&mut self, index: usize) -> Option<usize> {
        let snapshot = prompt_cache::load(
            &self.disk_snapshots[index].path,
            &self.prompt_cache_model_key,
        );
        if let Ok(snapshot) = snapshot
            && self.model.restore_prompt_snapshot(&snapshot).is_ok()
        {
            self.prompt_checkpoints.clear();
            return Some(snapshot.position);
        }
        self.disk_snapshots.remove(index);
        self.model.reset();
        None
    }

    /// Wait for a disk write still in flight when it would beat every
    /// committed snapshot for this prompt and the `held` tokens a faster tier
    /// holds, then index it.
    ///
    /// Writes run in the background, so a request that arrives right behind
    /// the one that queued a snapshot could otherwise find it absent and
    /// prefill from further back. Waiting costs at most the rest of that one
    /// write; it is never paid when a committed snapshot or a faster tier
    /// already holds as long a prefix, nor when no write is pending.
    fn await_pending_disk_write(&mut self, prompt: &[u32], bounds: &ReuseBounds, held: usize) {
        let Some(writer) = &self.disk_writer else {
            return;
        };
        let usable =
            |tokens: &[u32]| tokens.len() <= bounds.snapshot_reusable && prompt.starts_with(tokens);
        let committed = self
            .disk_snapshots
            .iter()
            .filter(|snapshot| usable(&snapshot.tokens))
            .map(|snapshot| snapshot.tokens.len())
            .max()
            .unwrap_or(0)
            .max(held);
        writer.wait_for(|tokens| tokens.len() > committed && usable(tokens));
        self.collect_disk_writes();
    }
}
