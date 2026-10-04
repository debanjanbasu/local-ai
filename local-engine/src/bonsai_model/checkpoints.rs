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

    #[allow(clippy::too_many_lines)]
    pub(super) fn prepare_prompt(
        &mut self,
        prompt: &[u32],
        session_id: Option<&str>,
    ) -> crate::Result<(usize, PromptCacheSource, Option<PromptSnapshot>, bool)> {
        let lcp = self
            .cached_tokens
            .iter()
            .zip(prompt)
            .take_while(|(cached, new)| cached == new)
            .count();
        let reusable = lcp.min(prompt.len().saturating_sub(1));
        let snapshot_reusable = prompt.len().saturating_sub(1);
        let selected = self
            .prompt_checkpoints
            .iter()
            .enumerate()
            .rev()
            .find(|(_, checkpoint)| {
                checkpoint.state.position <= reusable
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
        let (reused, source) = if let Some(index) = selected {
            self.prompt_checkpoints.rotate_left(index + 1);
            let checkpoint = self
                .prompt_checkpoints
                .pop()
                .ok_or_else(|| crate::Error::Generation("prompt checkpoint disappeared".into()))?;
            let position = checkpoint.state.position;
            self.prompt_checkpoints.push(checkpoint);
            (position, PromptCacheSource::Gpu)
        } else {
            let host = self
                .session_snapshots
                .iter()
                .enumerate()
                .filter(|(_, snapshot)| {
                    snapshot.state.position() <= snapshot_reusable
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
            if let Some(index) = host {
                let position = self.session_snapshots[index].state.position();
                let snapshot = self.session_snapshots.remove(index);
                self.session_snapshots.push(snapshot);
                self.prompt_checkpoints.clear();
                (position, PromptCacheSource::Host)
            } else {
                let disk = self
                    .disk_snapshots
                    .iter()
                    .enumerate()
                    .filter(|(_, snapshot)| {
                        snapshot.tokens.len() <= snapshot_reusable
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
        };
        // The next turn's re-rendered history usually diverges at the final
        // prompt token (the assistant-turn opener), so the reusable boundary
        // for checkpoints and session snapshots is the penultimate position.
        let penultimate = prompt.len() - 1;
        // Recurrent state is valid only at an exact position. Materialize one
        // semantic checkpoint per prefill: prefer a newly observed divergence,
        // otherwise the end of the system turn. The minimum follows the
        // resource-selected prefill chunk, avoiding copies for short prefixes.
        let minimum = self.info.prefill_chunk_size;
        let divergence = (lcp >= minimum && lcp > reused && lcp <= penultimate).then_some(lcp);
        let system_boundary = prompt
            .iter()
            .position(|token| self.tokenizer.eos_ids().contains(token))
            .map(|index| index + 1)
            .filter(|&position| {
                position >= minimum && position > reused && position <= penultimate
            });
        let reusable_boundary = divergence.or(system_boundary);
        let mut persisted_reusable_boundary = false;
        if let Some(boundary) = reusable_boundary {
            self.model.prefill(&prompt[reused..boundary])?;
            self.save_prompt_checkpoint(&prompt[..boundary], true)?;
            if self.prompt_cache_bytes > 0
                || (self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some())
            {
                let snapshot = self.model.prompt_snapshot()?;
                self.save_session_snapshot(
                    &prompt[..boundary],
                    session_id,
                    Some(snapshot),
                    true,
                    true,
                )?;
                persisted_reusable_boundary = true;
            }
        }
        let prefilled = reusable_boundary.unwrap_or(reused);
        if prefilled < penultimate {
            self.model.prefill(&prompt[prefilled..penultimate])?;
            if self.max_prompt_checkpoints > 0 {
                self.cached_tokens.clear();
                self.cached_tokens.extend_from_slice(&prompt[..penultimate]);
                self.save_prompt_checkpoint(&prompt[..penultimate], false)?;
            }
        }
        let snapshot = (penultimate > 0
            && (self.prompt_cache_bytes > 0
                || (self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some())))
        .then(|| self.model.prompt_snapshot())
        .transpose()?;
        self.model.prefill(&prompt[penultimate..])?;
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
