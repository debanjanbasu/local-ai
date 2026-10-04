use crate::bonsai_model::{BonsaiEngine, PromptCacheSource};
use crate::prompt_cache;

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
    // otherwise the end of the system turn. The minimum follows the
    // resource-selected prefill chunk, avoiding copies for short prefixes.
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
        ReuseBounds::new(lcp, prompt.len(), self.info.prefill_chunk_size)
    }

    /// The three-tier cascade: a live GPU checkpoint, then a host session
    /// snapshot, then a disk snapshot, which always resolves.
    pub(super) fn restore_reusable_prefix(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        session_id: Option<&str>,
    ) -> crate::Result<(usize, PromptCacheSource)> {
        if let Some(reused) = self.restore_from_gpu_tier(prompt, bounds)? {
            return Ok(reused);
        }
        if let Some(reused) = self.restore_from_host_tier(prompt, bounds, session_id)? {
            return Ok(reused);
        }
        Ok(self.restore_from_disk_tier(prompt, bounds, session_id))
    }

    /// Tier 1: live GPU checkpoints, newest first. One the model refuses is
    /// evicted and the tier reports no reuse.
    fn restore_from_gpu_tier(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
    ) -> crate::Result<Option<(usize, PromptCacheSource)>> {
        let selected = self
            .prompt_checkpoints
            .iter()
            .enumerate()
            .rev()
            .find(|(_, checkpoint)| {
                checkpoint.state.position <= bounds.reusable
                    && checkpoint.tokens.as_slice() == &prompt[..checkpoint.state.position]
            })
            .map(|(index, _)| index);
        let selected = if let Some(index) = selected {
            if self
                .model
                .restore_prompt_checkpoint(&self.prompt_checkpoints[index].state)?
            {
                Some(index)
            } else {
                self.prompt_checkpoints.remove(index);
                None
            }
        } else {
            None
        };
        let Some(index) = selected else {
            return Ok(None);
        };
        self.prompt_checkpoints.rotate_left(index + 1);
        let checkpoint = self
            .prompt_checkpoints
            .pop()
            .ok_or_else(|| crate::Error::Generation("prompt checkpoint disappeared".into()))?;
        let position = checkpoint.state.position;
        self.prompt_checkpoints.push(checkpoint);
        Ok(Some((position, PromptCacheSource::Gpu)))
    }

    /// Tier 2: host session snapshots, best key first. One the model refuses
    /// is evicted and the tier reports no reuse.
    fn restore_from_host_tier(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        session_id: Option<&str>,
    ) -> crate::Result<Option<(usize, PromptCacheSource)>> {
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
            .map(|(index, _)| index);
        let host = if let Some(index) = host {
            if self
                .model
                .restore_host_prompt_snapshot(&self.session_snapshots[index].state)?
            {
                Some(index)
            } else {
                self.session_snapshots.remove(index);
                None
            }
        } else {
            None
        };
        let Some(index) = host else {
            return Ok(None);
        };
        let position = self.session_snapshots[index].state.position();
        let snapshot = self.session_snapshots.remove(index);
        self.session_snapshots.push(snapshot);
        self.prompt_checkpoints.clear();
        Ok(Some((position, PromptCacheSource::Host)))
    }

    /// Tier 3: disk snapshots, best key first. This tier always resolves, so
    /// it hands back a prefix rather than reporting no reuse.
    fn restore_from_disk_tier(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        session_id: Option<&str>,
    ) -> (usize, PromptCacheSource) {
        let disk = self
            .disk_snapshots
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| {
                snapshot.tokens.len() <= bounds.snapshot_reusable
                    && snapshot.tokens.as_slice() == &prompt[..snapshot.tokens.len()]
            })
            .max_by_key(|(_, snapshot)| {
                (
                    snapshot.tokens.len(),
                    snapshot.session_id.as_deref() == session_id,
                )
            })
            .map(|(index, _)| index);
        if let Some(index) = disk {
            let snapshot = prompt_cache::load(
                &self.disk_snapshots[index].path,
                &self.prompt_cache_model_key,
            );
            // A snapshot that fails to load or restore is a cache miss,
            // never a failed request.
            if let Ok(snapshot) = snapshot
                && self.model.restore_prompt_snapshot(&snapshot).is_ok()
            {
                self.prompt_checkpoints.clear();
                (snapshot.position, PromptCacheSource::Disk)
            } else {
                self.disk_snapshots.remove(index);
                self.model.reset();
                (0, PromptCacheSource::None)
            }
        } else {
            self.model.reset();
            (0, PromptCacheSource::None)
        }
    }
}
