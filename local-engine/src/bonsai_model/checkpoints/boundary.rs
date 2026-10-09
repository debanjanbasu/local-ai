use super::reuse::ReuseBounds;
use crate::bonsai_model::{BonsaiEngine, PrefillProgress, PromptCheckpoint};

/// The rest of one prompt's prefill after its reusable prefix was restored.
///
/// A prompt prefills in bounded segments (see [`BonsaiEngine::advance_prompt`])
/// so the engine can interleave it with other requests' decode steps. The
/// prompt-cache milestones a single pass used to hit on the way — the
/// reusable boundary, the penultimate token, the end — are kept here and
/// handled when the sequence reaches each one, whichever segment that is.
pub(in crate::bonsai_model) struct PromptPrefill {
    /// Prompt tokens restored from the prompt cache.
    pub(in crate::bonsai_model) reused: usize,
    /// Prompt tokens the sequence holds.
    done: usize,
    /// The reusable boundary still to checkpoint at.
    boundary: Option<usize>,
    penultimate: usize,
    /// The penultimate position was prefilled to rather than restored, so it
    /// owes a GPU checkpoint.
    checkpoint_penultimate: bool,
    penultimate_reached: bool,
    /// The pinned prompt-boundary snapshot read back when the request ends.
    pub(in crate::bonsai_model) snapshot: Option<PromptCheckpoint>,
    pub(in crate::bonsai_model) persisted_reusable_boundary: bool,
}

impl PromptPrefill {
    /// Prompt tokens still to prefill.
    pub(in crate::bonsai_model) const fn remaining(&self, prompt_len: usize) -> usize {
        prompt_len.saturating_sub(self.done)
    }
}

impl BonsaiEngine {
    /// Choose the reusable boundary for a prompt whose first `reused` tokens
    /// were restored.
    pub(super) fn plan_prefill(
        &self,
        prompt: &[u32],
        bounds: &ReuseBounds,
        reused: usize,
    ) -> PromptPrefill {
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
        let boundary = divergence.or(system_boundary);
        PromptPrefill {
            reused,
            done: reused,
            boundary,
            penultimate,
            checkpoint_penultimate: boundary.unwrap_or(reused) < penultimate,
            penultimate_reached: false,
            snapshot: None,
            persisted_reusable_boundary: false,
        }
    }

    /// Prefill up to `budget` more prompt tokens into the resident sequence,
    /// handling every milestone reached. Returns whether the prompt is
    /// complete, its last token's logits in scratch.
    ///
    /// A cancelled request stops between blocks; the caller learns it from
    /// the model's `take_cancel_observed`.
    pub(in crate::bonsai_model) fn advance_prompt(
        &mut self,
        prompt: &[u32],
        session_id: Option<&str>,
        plan: &mut PromptPrefill,
        mut budget: usize,
        progress: &mut dyn FnMut(PrefillProgress),
    ) -> crate::Result<bool> {
        loop {
            if plan.boundary == Some(plan.done) {
                self.reach_boundary(&prompt[..plan.done], session_id, plan)?;
            }
            if plan.done == plan.penultimate && !plan.penultimate_reached {
                self.reach_penultimate(&prompt[..plan.done], plan)?;
            }
            if plan.done == prompt.len() {
                self.finish_prompt(prompt)?;
                return Ok(true);
            }
            if budget == 0 {
                return Ok(false);
            }
            let target = match plan.boundary {
                Some(boundary) if boundary > plan.done => boundary,
                _ if plan.done < plan.penultimate => plan.penultimate,
                _ => prompt.len(),
            };
            let end = target.min(plan.done + budget);
            let before = self.model.position();
            self.model
                .prefill_segment(&prompt[plan.done..end], end == prompt.len(), progress)?;
            let advanced = self.model.position() - before;
            plan.done += advanced;
            budget = budget.saturating_sub(advanced);
            if plan.done < end {
                // Cancelled between blocks.
                return Ok(false);
            }
        }
    }

    fn reach_boundary(
        &mut self,
        tokens: &[u32],
        session_id: Option<&str>,
        plan: &mut PromptPrefill,
    ) -> crate::Result<()> {
        plan.boundary = None;
        self.claim_gpu_cache();
        self.save_prompt_checkpoint(tokens, true)?;
        if self.caches_snapshots() {
            // Only the GPU readback is paid here, mid-prefill; the disk
            // write is queued to the background writer.
            let snapshot = self.model.prompt_snapshot()?;
            self.save_session_snapshot(tokens, session_id, Some(snapshot), true, true)?;
            plan.persisted_reusable_boundary = true;
        }
        Ok(())
    }

    fn reach_penultimate(&mut self, tokens: &[u32], plan: &mut PromptPrefill) -> crate::Result<()> {
        plan.penultimate_reached = true;
        if plan.checkpoint_penultimate && self.max_prompt_checkpoints > 0 {
            self.claim_gpu_cache();
            self.cached_tokens.clear();
            self.cached_tokens.extend_from_slice(tokens);
            self.save_prompt_checkpoint(tokens, false)?;
        }
        // Pin the boundary's recurrent state on the GPU and read the snapshot
        // back when the request ends: its K/V rows stay put in the live caches
        // meanwhile. Reading it all back now held the whole prefix in host
        // memory through decode (777 MB after a 16K-token prompt).
        if !tokens.is_empty() && self.caches_snapshots() {
            plan.snapshot = Some(self.model.pinned_prompt_checkpoint()?);
        }
        Ok(())
    }

    fn finish_prompt(&mut self, prompt: &[u32]) -> crate::Result<()> {
        self.claim_gpu_cache();
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(prompt);
        self.prompt_checkpoints.retain(|checkpoint| {
            checkpoint.tokens.len() <= prompt.len()
                && checkpoint.tokens.as_slice() == &prompt[..checkpoint.tokens.len()]
        });
        self.save_prompt_checkpoint(prompt, false)
    }

    /// Whether host or disk session snapshots are kept at all.
    const fn caches_snapshots(&self) -> bool {
        self.prompt_cache_bytes > 0
            || (self.prompt_cache_disk_bytes > 0 && self.prompt_cache_dir.is_some())
    }
}
