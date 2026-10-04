use super::{
    BonsaiEngine, DiskEntry, HostPromptSnapshot, PromptCacheSource, PromptCheckpoint,
    PromptSnapshot, prompt_cache,
};
const MAX_HOST_PROMPT_SNAPSHOTS: usize = 16;
pub(super) struct SessionSnapshot {
    tokens: Vec<u32>,
    session_id: Option<String>,
    pub(super) state: HostPromptSnapshot,
    reusable_boundary: bool,
}

pub(super) struct CachedCheckpoint {
    tokens: Vec<u32>,
    pub(super) state: PromptCheckpoint,
    reusable_boundary: bool,
}

mod boundary;
mod reuse;

use self::reuse::ReuseBounds;

impl BonsaiEngine {
    pub(super) fn save_prompt_checkpoint(
        &mut self,
        tokens: &[u32],
        reusable_boundary: bool,
    ) -> crate::Result<()> {
        if self.max_prompt_checkpoints == 0 || self.model.position() != tokens.len() {
            return Ok(());
        }
        self.prompt_checkpoints
            .retain(|checkpoint| checkpoint.tokens != tokens);
        self.prompt_checkpoints.push(CachedCheckpoint {
            tokens: tokens.to_vec(),
            state: self.model.prompt_checkpoint()?,
            reusable_boundary,
        });
        if self.prompt_checkpoints.len() > self.max_prompt_checkpoints {
            let victim = self
                .prompt_checkpoints
                .iter()
                .position(|entry| !entry.reusable_boundary)
                .unwrap_or(0);
            self.prompt_checkpoints.remove(victim);
        }
        Ok(())
    }

    pub(super) fn prepare_prompt(
        &mut self,
        prompt: &[u32],
        session_id: Option<&str>,
    ) -> crate::Result<(usize, PromptCacheSource, Option<PromptSnapshot>, bool)> {
        let bounds: ReuseBounds = self.reuse_bounds(prompt);
        let (reused, source) = self.restore_reusable_prefix(prompt, &bounds, session_id)?;
        let (snapshot, persisted_reusable_boundary) =
            self.materialize_boundary(prompt, &bounds, reused, session_id)?;
        self.model.prefill(&prompt[bounds.penultimate..])?;
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(prompt);
        self.prompt_checkpoints.retain(|checkpoint| {
            checkpoint.tokens.len() <= prompt.len()
                && checkpoint.tokens.as_slice() == &prompt[..checkpoint.tokens.len()]
        });
        self.save_prompt_checkpoint(prompt, false)?;
        Ok((reused, source, snapshot, persisted_reusable_boundary))
    }

    /// `persist` also writes the snapshot to the disk tier; only boundaries a
    /// later request is likely to share are worth the write.
    pub(super) fn save_session_snapshot(
        &mut self,
        tokens: &[u32],
        session_id: Option<&str>,
        state: Option<PromptSnapshot>,
        persist: bool,
        reusable_boundary: bool,
    ) -> crate::Result<()> {
        let save_to_disk =
            persist && self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some();
        if self.prompt_cache_bytes == 0 && !save_to_disk {
            return Ok(());
        }
        let state = match state {
            Some(state) => state,
            None => self.model.prompt_snapshot()?,
        };
        let bytes = state.recurrent.len()
            + state.mtp_prev_hidden.len()
            + state.target_kv.len()
            + state.mtp_kv.len()
            + tokens.len() * 4;
        if bytes > self.prompt_cache_bytes && !save_to_disk {
            return Ok(());
        }
        self.session_snapshots
            .retain(|entry| entry.tokens != tokens);
        if save_to_disk && let Some(dir) = &self.prompt_cache_dir {
            let path = prompt_cache::store(
                dir,
                &self.prompt_cache_model_key,
                tokens,
                session_id,
                &state,
                reusable_boundary,
            )?;
            self.disk_snapshots.retain(|entry| entry.tokens != tokens);
            self.disk_snapshots.push(DiskEntry {
                path,
                tokens: tokens.to_vec(),
                session_id: session_id.map(str::to_owned),
                reusable_boundary,
            });
            prompt_cache::trim(
                dir,
                &self.prompt_cache_model_key,
                self.prompt_cache_disk_bytes,
            );
        }
        if bytes > self.prompt_cache_bytes {
            return Ok(());
        }
        let state = self.model.host_prompt_snapshot(&state)?;
        self.session_snapshots.push(SessionSnapshot {
            tokens: tokens.to_vec(),
            session_id: session_id.map(str::to_owned),
            state,
            reusable_boundary,
        });
        while self.session_snapshots.len() > MAX_HOST_PROMPT_SNAPSHOTS {
            let victim = self
                .session_snapshots
                .iter()
                .position(|entry| {
                    !entry.reusable_boundary && entry.session_id.as_deref() != session_id
                })
                .or_else(|| {
                    self.session_snapshots
                        .iter()
                        .position(|entry| !entry.reusable_boundary)
                })
                .unwrap_or(0);
            self.session_snapshots.remove(victim);
        }
        Ok(())
    }
}
