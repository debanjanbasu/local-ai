use super::{
    AttentionLayer, AttentionWorkspace, BonsaiModel, BufferCopyRequest, CONV_STATE_BYTES,
    CommandBatch, GDN_STATE_BYTES, PromptCheckpoint, RECURRENT_LAYERS, Speculation, Verifier,
    WIDTH, grow_cache,
};

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
                size: GDN_STATE_BYTES,
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

    /// Make every K/V cache hold at least `end` tokens, doubling toward the
    /// context and moving only the written prefix. `head` names the
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
        let target = (self.kv_allocated * 2).max(end).min(self.info.context);
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

    /// Restore the start snapshot and replay `committed` compact recurrence
    /// factor rows for every GDN layer.
    pub(super) fn restore_checkpoint(
        &self,
        verifier: &Verifier,
        committed: usize,
    ) -> crate::Result<()> {
        let mut batch = CommandBatch::new(&self.context)?;
        let recurrent = self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            });
        let copies = recurrent
            .zip(&verifier.rollback)
            .flat_map(|(layer, rollback)| {
                [
                    BufferCopyRequest {
                        source: &rollback.state,
                        source_offset: 0,
                        destination: &layer.state,
                        destination_offset: 0,
                        size: GDN_STATE_BYTES,
                    },
                    BufferCopyRequest {
                        source: &rollback.history,
                        source_offset: 0,
                        destination: &layer.history,
                        destination_offset: 0,
                        size: CONV_STATE_BYTES,
                    },
                ]
            })
            .collect::<Vec<_>>();
        batch.blit_buffer_copies(copies)?;
        if committed == 0 {
            self.finish(batch)?;
            return Ok(());
        }
        for (layer, rollback) in self
            .layers
            .iter()
            .filter_map(|layer| match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => Some(recurrent),
                AttentionLayer::Full(_) => None,
            })
            .zip(&verifier.rollback)
        {
            self.ops.conv_sequence(
                &mut batch,
                &rollback.inputs,
                &layer.convolution,
                &layer.history,
                &self.scratch.convolved,
                committed as u32,
            )?;
            self.ops.l2_normalize_qk_rows(
                &mut batch,
                &self.scratch.convolved,
                self.epsilon,
                committed as u32,
            )?;
            self.ops.gdn_sequence(
                &mut batch,
                &self.scratch.convolved,
                &rollback.decay,
                &rollback.beta,
                &layer.state,
                &self.scratch.recurrent_output,
                committed as u32,
            )?;
        }
        self.finish(batch)?;
        Ok(())
    }
}
