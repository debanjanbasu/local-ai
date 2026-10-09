use super::{
    AttentionLayer, AttentionWorkspace, BonsaiModel, BufferCopyRequest, CONV_STATE_BYTES,
    CommandBatch, PromptCheckpoint, RECURRENT_LAYERS, Speculation, Verifier, WIDTH, grow_cache,
};

/// K/V caches double until they hold this many tokens, then grow by this many
/// at a time. Metal makes a whole buffer resident once the GPU uses it, so
/// doubling left up to half of a long request's K/V resident and unused: 478 MB
/// at 16.4K tokens (32,768 allocated, 20,480 now). A step's prefix copy costs
/// about 5 ms at 16K tokens and 40 ms at 128K, once per 4,096 tokens.
pub(super) const KV_GROWTH_STEP: usize = 4096;

impl BonsaiModel {
    pub(super) fn copy_prompt_state(
        &self,
        checkpoint: &PromptCheckpoint,
        restore: bool,
    ) -> crate::Result<()> {
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(layer) => Some(layer),
                AttentionLayer::Full(_) => None,
            });
        let mut copies = Vec::with_capacity(RECURRENT_LAYERS * 2 + 1);
        for (layer, (state, history)) in recurrent.zip(&checkpoint.recurrent) {
            let (state_source, state_destination) = if restore {
                (state, &layer.state)
            } else {
                (&layer.state, state)
            };
            let (history_source, history_destination) = if restore {
                (history, &layer.history)
            } else {
                (&layer.history, history)
            };
            copies.push(BufferCopyRequest {
                source: state_source,
                source_offset: 0,
                destination: state_destination,
                destination_offset: 0,
                size: self.state_format.state_bytes(),
            });
            copies.push(BufferCopyRequest {
                source: history_source,
                source_offset: 0,
                destination: history_destination,
                destination_offset: 0,
                size: CONV_STATE_BYTES,
            });
        }
        if let (Some(speculation), Some(hidden)) = (&self.speculation, &checkpoint.mtp_prev_hidden)
        {
            let (source, destination) = if restore {
                (hidden, speculation.mtp.prev_hidden())
            } else {
                (speculation.mtp.prev_hidden(), hidden)
            };
            copies.push(BufferCopyRequest {
                source,
                source_offset: 0,
                destination,
                destination_offset: 0,
                size: WIDTH * size_of::<f32>(),
            });
        }
        let mut batch = CommandBatch::new(&self.context)?;
        batch.blit_buffer_copies(copies)?;
        self.finish(batch)
    }

    /// Make every K/V cache hold at least `end` tokens, growing toward the
    /// context by [`KV_GROWTH_STEP`] and moving only the written prefix. `head` names the
    /// speculation state when the caller has taken it out of `self`.
    pub(super) fn reserve_kv(
        &mut self,
        end: usize,
        head: Option<&mut Speculation>,
    ) -> crate::Result<()> {
        if end <= self.kv_allocated {
            return Ok(());
        }
        if end > self.info.context {
            return Err(crate::Error::ContextOverflow(
                "Bonsai request exceeds the configured context".into(),
            ));
        }
        let step = self.kv_allocated.min(KV_GROWTH_STEP);
        let target = (self.kv_allocated + step).max(end).min(self.info.context);
        let used = self.position;
        for layer in &mut self.layers {
            let AttentionLayer::Full(full) = &mut layer.attention else {
                continue;
            };
            // Layer by layer, so the transient peak is one layer's new cache
            // rather than a second copy of every cache.
            full.key_cache = grow_cache(
                &self.context,
                &full.key_cache,
                used * self.kv_layout.key.token_bytes(),
                target * self.kv_layout.key.token_bytes(),
            )?;
            full.value_cache = grow_cache(
                &self.context,
                &full.value_cache,
                used * self.kv_layout.value.token_bytes(),
                target * self.kv_layout.value.token_bytes(),
            )?;
        }
        if let Some(speculation) = head.or(self.speculation.as_mut()) {
            speculation.mtp.grow_caches(&self.context, used, target)?;
        }
        self.scratch.attention = AttentionWorkspace::new(&self.context, target as u32)?;
        self.kv_allocated = target;
        Ok(())
    }

    /// Settle the recurrent layers after a verify block of `rows` rows that
    /// committed its first `committed`: with every row committed, adopt the
    /// final state and history the block wrote to the verifier (a buffer
    /// swap, no GPU work); with fewer, replay the committed rows' compact
    /// inputs from the start state the layers kept, in a batch of its own.
    pub(super) fn commit_verified(
        &mut self,
        verifier: &mut Verifier,
        rows: usize,
        committed: usize,
    ) -> crate::Result<()> {
        if committed == rows {
            return self.adopt_verified(verifier);
        }
        let mut batch = CommandBatch::new(&self.context)?;
        self.encode_commit_verified(&mut batch, verifier, rows, committed)?;
        self.finish(batch)
    }

    /// Swap the final state and history a fully committed verify block wrote
    /// to `verifier` into the recurrent layers.
    fn adopt_verified(&mut self, verifier: &mut Verifier) -> crate::Result<()> {
        if verifier.state_format != self.state_format {
            return Err(crate::Error::InvalidArgument(
                "verifier state format differs from the model's".into(),
            ));
        }
        let recurrent = self
            .layers
            .iter_mut()
            .filter_map(|layer| match &mut layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            });
        for (layer, rollback) in recurrent.zip(&mut verifier.rollback) {
            std::mem::swap(&mut layer.state, &mut rollback.state);
            std::mem::swap(&mut layer.history, &mut rollback.history);
        }
        Ok(())
    }

    /// [`Self::commit_verified`] encoded into `batch`, so later work (the
    /// head's commit) shares its submission. Nothing is encoded when every
    /// row is committed.
    pub(super) fn encode_commit_verified(
        &mut self,
        batch: &mut CommandBatch,
        verifier: &mut Verifier,
        rows: usize,
        committed: usize,
    ) -> crate::Result<()> {
        if committed == 0 || committed > rows || rows < 2 || rows > verifier.rows {
            return Err(crate::Error::InvalidArgument(
                "verify commit needs 1..=rows committed rows of a verify block".into(),
            ));
        }
        if committed == rows {
            return self.adopt_verified(verifier);
        }
        if verifier.state_format != self.state_format {
            return Err(crate::Error::InvalidArgument(
                "verifier state format differs from the model's".into(),
            ));
        }
        let scratch = &self.scratch;
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            });
        for (layer, rollback) in recurrent.zip(&verifier.rollback) {
            self.ops.conv_sequence(
                batch,
                &rollback.inputs,
                &layer.convolution,
                &layer.history,
                &scratch.convolved,
                committed as u32,
            )?;
            self.ops.l2_normalize_qk_rows(
                batch,
                &scratch.convolved,
                self.epsilon,
                committed as u32,
            )?;
            self.ops.gdn_sequence_into(
                batch,
                self.state_format,
                &scratch.convolved,
                &rollback.decay,
                &rollback.beta,
                &layer.state,
                &layer.state,
                &scratch.recurrent_output,
                committed as u32,
            )?;
        }
        Ok(())
    }

    /// Store every recurrent state as `format` from here on: reallocates the
    /// layer and verifier states (zeroed) and resets the model.
    #[cfg(test)]
    pub(super) fn set_state_format(
        &mut self,
        format: local_metal::bonsai_ops::GdnStateFormat,
    ) -> crate::Result<()> {
        for layer in &mut self.layers {
            if let AttentionLayer::Recurrent(recurrent) = &mut layer.attention {
                recurrent.state =
                    super::MetalBuffer::empty(self.context.device(), format.state_bytes())?;
            }
        }
        let verifiers = self
            .speculation
            .iter_mut()
            .map(|speculation| &mut speculation.verifier)
            .chain(self.ngram_verifier.as_mut());
        for verifier in verifiers {
            *verifier = Verifier::new(
                &self.context,
                format,
                verifier.rows - 1,
                verifier.max_rows - 1,
            )?;
        }
        self.state_format = format;
        self.reset();
        Ok(())
    }
}
