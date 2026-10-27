use super::{
    BonsaiEngine, HostPromptSnapshot, PROMPT_CHECKPOINT_BYTES, PromptCacheSource, PromptCheckpoint,
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
    /// A request tail no other tier holds: it is read back and written to
    /// disk only when something is about to overwrite its K/V rows (see
    /// [`BonsaiEngine::persist_owed_checkpoints`]).
    owed_to_disk: bool,
    session_id: Option<String>,
}

mod boundary;
mod reuse;

pub(super) use self::boundary::PromptPrefill;
use self::reuse::ReuseBounds;
pub(super) use self::reuse::{Loaded, SHARED_PREFIX_MIN_TOKENS};

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
            owed_to_disk: false,
            session_id: None,
        });
        if self.prompt_checkpoints.len() > self.max_prompt_checkpoints {
            // The disk tier's policy: a boundary a few tokens past a kept one
            // goes before the boundary every sharing request restores. The
            // two newest checkpoints (the request's boundary or penultimate
            // token, and its end) are what its next turn restores, so they
            // are never the victim: an agent session's turns each leave a
            // boundary that barely extends the last, and once boundaries
            // filled the tier, every new penultimate checkpoint went at once.
            let protected = self.max_prompt_checkpoints.min(2);
            let entries = self.prompt_checkpoints[..self.prompt_checkpoints.len() - protected]
                .iter()
                .map(|entry| (entry.tokens.as_slice(), entry.reusable_boundary))
                .collect::<Vec<_>>();
            let victim = prompt_cache::next_victim(&entries).unwrap_or(0);
            self.prompt_checkpoints.remove(victim);
        }
        Ok(())
    }

    /// Restore the longest reusable prefix of `prompt` into the resident
    /// sequence and plan the prefill of the rest, with a disk snapshot that
    /// finished `loaded` (see [`Self::restore_reusable_prefix`]). Also
    /// returns the load of a longer disk snapshot, when one was started: the
    /// plan is then a stand-in, not to be prefilled before the load ends.
    pub(super) fn begin_prompt(
        &mut self,
        prompt: &[u32],
        session_id: Option<&str>,
        loaded: Option<Loaded>,
    ) -> crate::Result<(PromptPrefill, PromptCacheSource, Option<prompt_cache::Load>)> {
        self.collect_disk_writes();
        self.persist_owed_checkpoints(prompt)?;
        let bounds: ReuseBounds = self.reuse_bounds(prompt);
        let (reused, source, load) =
            self.restore_reusable_prefix(prompt, &bounds, session_id, loaded)?;
        Ok((self.plan_prefill(prompt, &bounds, reused), source, load))
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
                self.disk_writer = Some(prompt_cache::Writer::spawn_with(
                    &self.disk_closer,
                    self.prompt_cache_daily_writes,
                )?);
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

    /// The bytes a snapshot of the first `tokens` prompt tokens holds, known
    /// before it is read back: the recurrent state and head hidden row, and
    /// each token's target and head K/V rows and id (about 38 KiB a token
    /// with the head).
    fn snapshot_bytes(&self, tokens: usize) -> u64 {
        let head = if self.model.speculation().is_some() {
            2 * local_metal::bonsai_ops::KvFormat::F16.token_bytes()
        } else {
            0
        };
        let per_token = self.model.kv_layout().token_bytes() * crate::bonsai::LAYERS
            / crate::bonsai::FULL_INTERVAL
            + head
            + size_of::<u32>();
        (PROMPT_CHECKPOINT_BYTES + tokens * per_token) as u64
    }

    /// Whether the host tier would keep a snapshot of `tokens` prompt tokens.
    pub(super) fn host_would_keep(&self, tokens: usize) -> bool {
        self.prompt_cache_bytes > 0 && self.snapshot_bytes(tokens) <= self.prompt_cache_bytes as u64
    }

    /// Whether the disk writer would keep a snapshot of `tokens` beside the
    /// committed ones (see [`prompt_cache::would_admit`]): it refuses a
    /// boundary that only barely extends a kept one when the two cannot both
    /// fit, a request tail behind boundaries that fill the budget, and
    /// anything while its daily write budget is overdrawn.
    pub(super) fn disk_would_keep(&self, tokens: &[u32], reusable_boundary: bool) -> bool {
        self.prompt_cache_disk_bytes > 0
            && self.prompt_cache_dir.is_some()
            && self
                .disk_writer
                .as_ref()
                .is_none_or(prompt_cache::Writer::within_budget)
            && prompt_cache::would_admit(
                &self.disk_snapshots,
                tokens,
                self.snapshot_bytes(tokens.len()),
                reusable_boundary,
                self.prompt_cache_disk_bytes,
            )
    }

    /// Whether the GPU tier holds a checkpoint of exactly `tokens`.
    pub(super) fn holds_checkpoint(&self, tokens: &[u32]) -> bool {
        self.prompt_checkpoints
            .iter()
            .any(|checkpoint| checkpoint.tokens == tokens)
    }

    /// Owe the request tail checkpointed at `tokens` to the disk tier: no
    /// other tier holds it, and its GPU checkpoint is the snapshot the
    /// session's next turn restores.
    pub(super) fn owe_to_disk(&mut self, tokens: &[u32], session_id: Option<&str>) {
        if let Some(checkpoint) = self
            .prompt_checkpoints
            .iter_mut()
            .find(|checkpoint| !checkpoint.reusable_boundary && checkpoint.tokens == tokens)
        {
            checkpoint.owed_to_disk = true;
            checkpoint.session_id = session_id.map(str::to_owned);
        }
    }

    /// Before `prompt` runs on the resident set, write to disk each owed
    /// checkpoint it is about to overwrite: one `prompt` does not extend,
    /// whose K/V rows its prefill or decode rewrites. One `prompt` extends is
    /// superseded by this request's own tail and owes nothing more.
    ///
    /// Reading a request tail back costs about 38 KiB a token on the engine
    /// thread, which an agent session's next turn, restoring the checkpoint
    /// from the GPU, never needs. So a tail that also persisted a boundary is
    /// read back only here, when a request of another session takes the set,
    /// rather than at its end: dropped there, as it used to be, it was gone
    /// for good once another session prefilled over it, and the session's next
    /// turn fell back to a boundary turns older.
    fn persist_owed_checkpoints(&mut self, prompt: &[u32]) -> crate::Result<()> {
        if self.cache_state != self.model.state_id() {
            return Ok(());
        }
        let mut index = 0;
        while index < self.prompt_checkpoints.len() {
            let checkpoint = &mut self.prompt_checkpoints[index];
            if !checkpoint.owed_to_disk || prompt.starts_with(&checkpoint.tokens) {
                checkpoint.owed_to_disk = false;
                index += 1;
                continue;
            }
            let checkpoint = self.prompt_checkpoints.remove(index);
            if !self.disk_would_keep(&checkpoint.tokens, false) {
                continue;
            }
            // Purged by the OS, or no longer behind the set's position: a
            // cache miss like any other.
            if let Ok(Some(snapshot)) = self.model.prompt_snapshot_at(&checkpoint.state) {
                self.save_session_snapshot(
                    &checkpoint.tokens,
                    checkpoint.session_id.as_deref(),
                    Some(snapshot),
                    true,
                    false,
                )?;
            }
        }
        Ok(())
    }

    /// Keep what the resident set still holds after a request on it ended
    /// without its position matching the tokens it emitted (cancelled, or
    /// with rows committed past its last emitted token): its K/V rows hold
    /// the prompt up to the position it reached, so the checkpoints of
    /// prefixes of that, and only they, still restore.
    /// The tier is left alone when it describes another set, which the
    /// request never touched.
    pub(super) fn keep_reached_prompt(&mut self, prompt: &[u32]) {
        if self.cache_state != self.model.state_id() {
            return;
        }
        let reached = &prompt[..self.model.position().min(prompt.len())];
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(reached);
        self.prompt_checkpoints
            .retain(|checkpoint| reached.starts_with(&checkpoint.tokens));
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
                Ok(None) => {
                    eprintln!(
                        "prompt cache: snapshot not written to disk: the budget would delete it \
                         again (a request tail behind shared prefixes, or a boundary that only \
                         barely extends a kept one)"
                    );
                }
                Ok(Some(entry)) => {
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
