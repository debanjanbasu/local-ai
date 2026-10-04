use super::{
    AttentionLayer, BlockOutput, BonsaiModel, BufferCopyRequest, CommandBatch, HadamardDirection,
    Speculation, VOCAB, WIDTH, decode_embeddings,
};

impl BonsaiModel {
    pub(super) fn forward(&mut self, token: u32, logits: bool) -> crate::Result<()> {
        let output = if logits {
            BlockOutput::LastLogits
        } else {
            BlockOutput::None
        };
        self.forward_block(&[token], output)
    }

    pub(super) fn forward_block(
        &mut self,
        tokens: &[u32],
        output: BlockOutput<'_>,
    ) -> crate::Result<()> {
        self.encode_block(tokens, output, None)
    }

    /// One target block over `tokens`, optionally followed in the same command
    /// batch by MTP head ingestion of those committed rows (`ingest`), which
    /// needs `output` to leave every row output-normalized.
    #[allow(clippy::too_many_lines)]
    pub(super) fn encode_block(
        &mut self,
        tokens: &[u32],
        output: BlockOutput<'_>,
        ingest: Option<&mut Speculation>,
    ) -> crate::Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.block_rows
            || tokens.iter().any(|&token| token as usize >= VOCAB)
            || self
                .position
                .checked_add(tokens.len())
                .is_none_or(|end| end > self.info.context)
        {
            return Err(crate::Error::ContextOverflow(
                "invalid Bonsai token block or position".into(),
            ));
        }
        let verify = match output {
            BlockOutput::Verify(verifier) if tokens.len() > 1 => Some(verifier),
            BlockOutput::Verify(_) => {
                return Err(crate::Error::InvalidArgument(
                    "Bonsai verification needs at least two rows".into(),
                ));
            }
            BlockOutput::None | BlockOutput::Hidden | BlockOutput::LastLogits => None,
        };
        if ingest.is_some() && matches!(output, BlockOutput::None) {
            return Err(crate::Error::InvalidArgument(
                "MTP ingestion needs output-normalized rows".into(),
            ));
        }
        let mut ingest = ingest;
        self.reserve_kv(self.position + tokens.len(), ingest.as_deref_mut())?;
        let count = tokens.len() as u32;
        decode_embeddings(
            &self.package,
            tokens,
            self.scratch.embedding.as_mut_slice::<f32>(),
        )?;
        let ingest = match ingest {
            Some(speculation) => {
                decode_embeddings(
                    &self.package,
                    tokens,
                    speculation.mtp.embedding_rows(tokens.len())?,
                )?;
                Some(&*speculation)
            }
            None => None,
        };
        let mut batch = CommandBatch::new(&self.context)?;
        let scratch = &self.scratch;
        self.kernels.transform(
            &mut batch,
            &self.input_rotation,
            &scratch.embedding,
            &scratch.hidden,
            count,
            HadamardDirection::Inverse,
        )?;
        let mut recurrent_index = 0;
        for layer in &self.layers {
            self.normalize_input(&mut batch, &scratch.hidden, &layer.attention_norm, count)?;
            match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => {
                    let rollback = verify.map(|verifier| &verifier.rollback[recurrent_index]);
                    recurrent_index += 1;
                    self.recurrent(&mut batch, recurrent, count, rollback)?;
                }
                AttentionLayer::Full(full) => self.full_attention(&mut batch, full, count)?,
            }
            self.ops.residual_add(
                &mut batch,
                &scratch.branch,
                &scratch.hidden,
                &scratch.hidden,
                WIDTH as u32 * count,
            )?;
            self.feed_forward(&mut batch, layer, count)?;
        }
        if !matches!(output, BlockOutput::None) {
            // Every row is output-normalized: `scratch.normalized` then holds
            // the hidden rows the MTP head consumes for these committed tokens.
            self.normalize_input(&mut batch, &scratch.hidden, &self.output_norm, count)?;
        }
        match verify {
            None if matches!(output, BlockOutput::LastLogits) => {
                // Only the last row is projected to the vocabulary. Reuse the
                // consumed embedding scratch instead of a per-row vocabulary.
                let rotated = if count == 1 {
                    &scratch.rotated_hidden
                } else {
                    batch.blit_buffer_copies([BufferCopyRequest {
                        source: &scratch.rotated_hidden,
                        source_offset: (tokens.len() - 1) * WIDTH * size_of::<f32>(),
                        destination: &scratch.embedding,
                        destination_offset: 0,
                        size: WIDTH * size_of::<f32>(),
                    }])?;
                    &scratch.embedding
                };
                self.project(&mut batch, &self.output, rotated, &scratch.logits, 1)?;
            }
            Some(verifier) => {
                self.project(
                    &mut batch,
                    &self.output,
                    &scratch.rotated_hidden,
                    &verifier.verify_logits,
                    count,
                )?;
            }
            None => {}
        }
        if let Some(speculation) = ingest {
            self.encode_head_rows(&mut batch, speculation, tokens.len(), self.position)?;
        }
        self.finish(batch)?;
        self.position += tokens.len();
        let logits = match (output, verify) {
            (BlockOutput::LastLogits, None) => scratch.logits.as_slice::<f32>(),
            (_, Some(verifier)) => {
                &verifier.verify_logits.as_slice::<f32>()[..tokens.len() * VOCAB]
            }
            _ => &[],
        };
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        Ok(())
    }

    /// Feed committed target rows to the MTP head in their own command batch:
    /// embeddings of `tokens` and the output-normalized hidden rows the last
    /// forward block left in `scratch.normalized`, ending at the current
    /// position. Production prefill fuses this into the target batch instead
    /// (see [`Self::encode_block`]); tests use it as the two-step reference.
    #[cfg(test)]
    pub(super) fn ingest_committed(&mut self, tokens: &[u32]) -> crate::Result<()> {
        let Some(mut speculation) = self.speculation.take() else {
            return Ok(());
        };
        let result = self.encode_committed(&mut speculation, tokens);
        self.speculation = Some(speculation);
        result
    }

    pub(super) fn encode_committed(
        &self,
        speculation: &mut Speculation,
        tokens: &[u32],
    ) -> crate::Result<()> {
        if tokens.is_empty() || tokens.len() > self.position {
            return Err(crate::Error::InvalidArgument(
                "MTP ingestion needs committed rows".into(),
            ));
        }
        let start = self.position - tokens.len();
        decode_embeddings(
            &self.package,
            tokens,
            speculation.mtp.embedding_rows(tokens.len())?,
        )?;
        let mut batch = CommandBatch::new(&self.context)?;
        self.encode_head_rows(&mut batch, speculation, tokens.len(), start)?;
        self.finish(batch)?;
        Ok(())
    }

    /// Encode K/V-only head ingestion of `rows` committed tokens starting at
    /// `start` into `batch`: the head's embedding rows must already be staged
    /// and `scratch.normalized` must hold those rows' output-normalized hidden.
    fn encode_head_rows(
        &self,
        batch: &mut CommandBatch,
        speculation: &Speculation,
        rows: usize,
        start: usize,
    ) -> crate::Result<()> {
        let mtp = &speculation.mtp;
        mtp.stage_hidden(batch, &self.scratch.normalized, rows)?;
        mtp.encode(
            batch,
            &self.mtp_shared(),
            mtp.hidden_in(),
            start,
            rows,
            false,
        )?;
        mtp.commit_hidden(batch, &self.scratch.normalized, rows - 1)
    }
}
