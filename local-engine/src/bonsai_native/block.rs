use super::{
    AttentionLayer, BlockOutput, BonsaiModel, BufferCopyRequest, CommandBatch, Described,
    HadamardDirection, Rope, Selection, Speculation, VOCAB, WIDTH, decode_embeddings,
};

/// Write the `(t, h, w, 0)` rotary positions of `count` tokens from
/// `position` to rows `row..` of `buffer`.
pub(super) fn write_rope_rows(
    buffer: &super::MetalBuffer,
    images: &crate::bonsai_vision::PromptImages,
    position: usize,
    count: usize,
    row: usize,
) {
    let rows = (0..count)
        .map(|index| images.rope(position + index))
        .collect::<Vec<_>>();
    buffer.copy_from_bytes(bytemuck::cast_slice(&rows), row * size_of::<[u32; 4]>());
}

/// Submit the command buffer encoded so far before encoding these layers.
/// The input rotation goes alone so the GPU's start-up latency (about 0.1 ms)
/// overlaps encoding the first layer; layers 1-3 keep the GPU busy while the
/// host encodes the rest, which took 0.25-0.7 ms for one row and left the GPU
/// idle 0.05-0.35 ms per token when the first buffer held only layer 0.
pub(super) const SUBMIT_BEFORE_LAYERS: [usize; 3] = [0, 1, 4];

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
    #[expect(clippy::too_many_lines)]
    #[expect(unsafe_code)]
    pub(super) fn encode_block(
        &mut self,
        tokens: &[u32],
        output: BlockOutput<'_>,
        ingest: Option<&mut Speculation>,
    ) -> crate::Result<()> {
        if tokens.is_empty()
            || tokens.len() > self.block_rows
            || crate::bonsai_vision::check_tokens(tokens, self.position, self.images.as_deref())
                .is_err()
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
        // Whatever the block writes, the previous selection no longer
        // describes the logits the caller will read.
        self.selection = Selection::None;
        self.distributions.described = Described::None;
        self.candidates.described = Described::None;
        let mut ingest = ingest;
        self.reserve_kv(self.position + tokens.len(), ingest.as_deref_mut())?;
        let count = tokens.len() as u32;
        let images = self.images.as_deref().map(|images| (images, self.position));
        decode_embeddings(
            &self.package,
            tokens,
            // SAFETY: every block waits for its GPU work before returning, and
            // this block is encoded only after the embeddings are written.
            unsafe { self.scratch.embedding.as_mut_slice::<f32>() },
            images,
        )?;
        let rope_rows = self.write_rope(self.position, tokens.len(), 0);
        let ingest = match ingest {
            Some(speculation) => {
                decode_embeddings(
                    &self.package,
                    tokens,
                    speculation.mtp.embedding_rows(tokens.len())?,
                    images,
                )?;
                Some(&*speculation)
            }
            None => None,
        };
        // Verification lets the independent projections of each layer overlap
        // (`CommandBatch::independent`); every other dispatch stays ordered.
        let mut batch = if verify.is_some() {
            CommandBatch::new_concurrent(&self.context)?
        } else {
            CommandBatch::new(&self.context)?
        };
        let scratch = &self.scratch;
        let rope = match rope_rows {
            Ok(offset) => Rope::Text(offset),
            Err(()) => Rope::Rows(&scratch.rope, 0),
        };
        self.kernels.transform(
            &mut batch,
            &self.input_rotation,
            &scratch.embedding,
            &scratch.hidden,
            count,
            HadamardDirection::Inverse,
        )?;
        if count > super::layers::SHORT_BLOCK {
            // Only short blocks fold the tape in.
            self.encode_settle(&mut batch)?;
        }
        let mut recurrent_index = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            if SUBMIT_BEFORE_LAYERS.contains(&index) {
                // Start the GPU on the first layers while the host encodes the
                // rest: encoding a whole block took about 1.2 ms of host time
                // (0.5 ms for one row) during which the GPU sat idle.
                batch.submit_and_renew(&self.context)?;
            }
            // The residual stream alternates between `hidden` (layer input)
            // and `hidden_alt` (after attention), so every residual add rides
            // in the next normalization's dispatch.
            if index == 0 {
                self.normalize_input(&mut batch, &scratch.hidden, &layer.attention_norm, count)?;
            } else {
                self.residual_normalize(
                    &mut batch,
                    &scratch.hidden_alt,
                    &scratch.hidden,
                    &layer.attention_norm,
                    count,
                )?;
            }
            // `hidden` now holds the residual after the previous layer.
            self.encode_taps(&mut batch, index, tokens.len(), verify.is_some())?;
            match &layer.attention {
                AttentionLayer::Recurrent(recurrent) => {
                    let rollback = verify.map(|verifier| &verifier.rollback[recurrent_index]);
                    self.recurrent(&mut batch, recurrent, recurrent_index, count, rollback)?;
                    recurrent_index += 1;
                }
                AttentionLayer::Full(full) => {
                    self.full_attention(&mut batch, full, count, rope)?;
                }
            }
            self.residual_normalize(
                &mut batch,
                &scratch.hidden,
                &scratch.hidden_alt,
                &layer.post_attention_norm,
                count,
            )?;
            self.feed_forward_branch(&mut batch, layer, count)?;
        }
        // Every recurrent layer folded the tape's pending rows in.
        self.tape.clear();
        if matches!(output, BlockOutput::None) {
            self.ops.residual_add(
                &mut batch,
                &scratch.branch,
                &scratch.hidden_alt,
                &scratch.hidden,
                WIDTH as u32 * count,
            )?;
        } else {
            // Every row is output-normalized: `scratch.normalized` then holds
            // the hidden rows the MTP head consumes for these committed tokens.
            self.residual_normalize(
                &mut batch,
                &scratch.hidden_alt,
                &scratch.hidden,
                &self.output_norm,
                count,
            )?;
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
        let produced = match (output, verify) {
            (BlockOutput::LastLogits, None) => Some((&scratch.logits, 1, Selection::Last)),
            (_, Some(verifier)) => Some((
                &verifier.verify_logits,
                tokens.len(),
                Selection::Rows(tokens.len()),
            )),
            _ => None,
        };
        // The argmax rides in the block's command buffer, and its kernel
        // flags non-finite rows, so the host neither submits a selection per
        // row nor scans the logits; the selection itself is used only for
        // greedy samplers, and the best k only for samplers drawing from them.
        let flagged = produced.filter(|&(_, rows, _)| rows <= self.greedy.max_rows());
        if let Some((logits, rows, _)) = flagged {
            self.greedy.encode(&mut batch, logits, rows)?;
        }
        let described = |selection| match selection {
            Selection::Last => Described::Last,
            Selection::Rows(rows) => Described::Rows(rows),
            Selection::None => Described::None,
        };
        if let (Some(k), Some((logits, rows, selection))) = (self.top_k, produced) {
            self.candidates.reserve(&self.context, rows)?;
            self.candidates.encode(&mut batch, logits, rows, k)?;
            self.candidates.described = described(selection);
            self.candidates.k = k;
        }
        if let (Some(top), Some((logits, rows, selection))) = (self.logprobs, produced) {
            let described = described(selection);
            self.distributions.reserve(&self.context, rows)?;
            self.distributions.encode(&mut batch, logits, rows, top)?;
            self.distributions.described = described;
            self.distributions.top = top;
        }
        if let Some(speculation) = ingest {
            self.encode_head_rows(&mut batch, speculation, tokens.len(), self.position)?;
        }
        self.finish(batch)?;
        self.position += tokens.len();
        let nonfinite = match (flagged, produced) {
            (Some((_, rows, selection)), _) => {
                if self.device_greedy {
                    self.selection = selection;
                }
                self.greedy
                    .results(rows)
                    .iter()
                    .any(|result| result.nonfinite != 0)
            }
            // SAFETY: `finish` above waited for the block, and nothing writes
            // the logits while they are scanned.
            (None, Some((logits, rows, _))) => unsafe {
                logits.as_slice::<f32>()[..rows * VOCAB]
                    .iter()
                    .any(|value| !value.is_finite())
            },
            (None, None) => false,
        };
        if nonfinite {
            self.selection = Selection::None;
            return Err(crate::Error::Generation("non-finite Bonsai logits".into()));
        }
        Ok(())
    }

    /// The rotary positions of `count` rows from `position` of the resident
    /// sequence: `Ok(offset)` when they are all text at `position + offset`
    /// (plain `RoPE`), else `Err` after writing explicit rows to
    /// `scratch.rope` from row `row`.
    pub(super) fn write_rope(&self, position: usize, count: usize, row: usize) -> Result<i32, ()> {
        let Some(images) = self.images.as_deref() else {
            return Ok(0);
        };
        if let Some(offset) = images.text_offset(position, count) {
            return Ok(offset);
        }
        write_rope_rows(&self.scratch.rope, images, position, count, row);
        Err(())
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

    #[cfg(test)]
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
            None,
        )?;
        let mut batch = CommandBatch::new(&self.context)?;
        self.encode_head_rows(&mut batch, speculation, tokens.len(), start)?;
        self.finish(batch)?;
        Ok(())
    }

    /// Encode K/V-only head ingestion of `rows` committed tokens starting at
    /// `start` into `batch`: the head's embedding rows must already be staged
    /// and `scratch.normalized` must hold those rows' output-normalized hidden.
    pub(super) fn encode_head_rows(
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
