use super::reuse::ReuseBounds;
use crate::bonsai_model::{BonsaiEngine, PrefillProgress, PromptCheckpoint};

impl BonsaiEngine {
    /// Choose the reusable boundary, materialize it, and build the snapshot
    /// the caller hands on to the next request.
    pub(super) fn materialize_boundary(
        &mut self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        reused: usize,
        session_id: Option<&str>,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<(Option<PromptCheckpoint>, bool)> {
        let lcp = bounds.lcp;
        let penultimate = bounds.penultimate;
        let minimum = bounds.minimum;
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
            self.model.prefill(&prompt[reused..boundary], progress)?;
            self.save_prompt_checkpoint(&prompt[..boundary], true)?;
            if self.prompt_cache_bytes > 0
                || (self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some())
            {
                // Only the GPU readback is paid here, mid-prefill; the disk
                // write is queued to the background writer.
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
            self.model
                .prefill(&prompt[prefilled..penultimate], progress)?;
            if self.max_prompt_checkpoints > 0 {
                self.cached_tokens.clear();
                self.cached_tokens.extend_from_slice(&prompt[..penultimate]);
                self.save_prompt_checkpoint(&prompt[..penultimate], false)?;
            }
        }
        // Pin the boundary's recurrent state on the GPU and read the snapshot
        // back when the request ends: its K/V rows stay put in the live caches
        // meanwhile. Reading it all back now held the whole prefix in host
        // memory through decode (777 MB after a 16K-token prompt).
        let snapshot = (penultimate > 0
            && (self.prompt_cache_bytes > 0
                || (self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some())))
        .then(|| self.model.pinned_prompt_checkpoint())
        .transpose()?;
        Ok((snapshot, persisted_reusable_boundary))
    }
}
