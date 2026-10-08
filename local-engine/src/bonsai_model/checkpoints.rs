use super::{
    BonsaiEngine, HostPromptSnapshot, PrefillProgress, PromptCacheSource, PromptCheckpoint,
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
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<(usize, PromptCacheSource, Option<PromptCheckpoint>, bool)> {
        self.collect_disk_writes();
        let bounds: ReuseBounds = self.reuse_bounds(prompt);
        let (reused, source) = self.restore_reusable_prefix(prompt, &bounds, session_id)?;
        let (snapshot, persisted_reusable_boundary) =
            self.materialize_boundary(prompt, &bounds, reused, session_id, progress)?;
        self.model
            .prefill(&prompt[bounds.penultimate..], progress)?;
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
    ///
    /// The disk write is handed to the background [`prompt_cache::Writer`]
    /// and does not hold up the request: on a shared-prefix boundary it used to
    /// stall prefill, and on a request tail the response, for the SHA-256,
    /// write and `fsync` of the whole snapshot (~140 ms for 204 MB on an M4
    /// Pro).
    /// The snapshot becomes visible to the disk tier only once the writer
    /// reports it committed. A failed write is therefore no longer this
    /// request's error: the snapshot is simply not cached, as with any other
    /// cache miss, and the failure is reported on stderr.
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
        let save_to_host = bytes <= self.prompt_cache_bytes;
        if !save_to_host && !save_to_disk {
            return Ok(());
        }
        self.session_snapshots
            .retain(|entry| entry.tokens != tokens);
        if save_to_host {
            // Copied out before the disk tier takes ownership of `state`.
            let host = self.model.host_prompt_snapshot(&state)?;
            self.push_host_snapshot(tokens, session_id, host, reusable_boundary);
        }
        if save_to_disk && let Some(root) = self.prompt_cache_dir.clone() {
            if self.disk_writer.is_none() {
                self.disk_writer = Some(prompt_cache::Writer::spawn()?);
            }
            if let Some(writer) = &self.disk_writer {
                writer.submit(prompt_cache::StoreJob {
                    root,
                    model_key: self.prompt_cache_model_key.clone(),
                    tokens: tokens.to_vec(),
                    session_id: session_id.map(str::to_owned),
                    snapshot: state,
                    reusable_boundary,
                    budget: self.prompt_cache_disk_bytes,
                });
            }
        }
        Ok(())
    }

    fn push_host_snapshot(
        &mut self,
        tokens: &[u32],
        session_id: Option<&str>,
        state: HostPromptSnapshot,
        reusable_boundary: bool,
    ) {
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
    }

    /// Index the disk snapshots the writer has committed since the last call,
    /// and forget the ones its trim deleted.
    ///
    /// A snapshot written for a directory or model key this engine has since
    /// moved away from is dropped rather than indexed.
    pub(super) fn collect_disk_writes(&mut self) {
        let Some(writer) = &self.disk_writer else {
            return;
        };
        let current = self
            .prompt_cache_dir
            .as_ref()
            .map(|root| root.join(&self.prompt_cache_model_key));
        for completion in writer.take_completed() {
            match completion.stored {
                Ok(entry) => {
                    if current
                        .as_ref()
                        .is_some_and(|dir| entry.path.starts_with(dir))
                    {
                        self.disk_snapshots
                            .retain(|indexed| indexed.tokens != entry.tokens);
                        self.disk_snapshots.push(entry);
                    }
                }
                Err(error) => {
                    eprintln!("prompt cache: snapshot not written to disk: {error}");
                }
            }
            self.disk_snapshots
                .retain(|indexed| !completion.evicted.contains(&indexed.path));
        }
    }

    /// Wait for every pending disk write and index it.
    ///
    /// Requests never need this — the disk tier waits on its own when the
    /// snapshot it would load is still being written — but a caller that needs
    /// the files on disk now, such as a test, does.
    pub fn flush_prompt_cache(&mut self) {
        if let Some(writer) = &self.disk_writer {
            writer.flush();
        }
        self.collect_disk_writes();
    }
}
