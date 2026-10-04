use super::{
    BlockOutput, BonsaiModel, CommandBatch, DRAFT_CHAIN_MIN_MARGIN, Instant, MtpStats, NgramStats,
    Sampler, Speculation, SpeculativeBatch, VOCAB, cancelled_speculative_batch, decode_embeddings,
    draft_depth, verify_greedy_drafts,
};

impl BonsaiModel {
    /// One verified speculative round. A cancelled request submits neither the
    /// draft pass nor the verify block and hands back
    /// `cancelled_speculative_batch`; the position and both caches keep the
    /// exact state of the last committed round.
    #[allow(clippy::too_many_lines)]
    pub(super) fn speculative_round(
        &mut self,
        speculation: &mut Speculation,
        seed: u32,
        sampler: &mut Sampler,
        remaining: usize,
    ) -> crate::Result<SpeculativeBatch> {
        if self.stop_for_cancel() {
            return Ok(cancelled_speculative_batch());
        }
        let depth = draft_depth(
            speculation.mtp.depth,
            remaining,
            self.info.context,
            self.position,
        );
        if depth == 0 {
            // No room to speculate: only the request's final token remains.
            // The head is not fed; every request restarts from `reset`.
            self.forward(seed, true)?;
            let sampling_started = Instant::now();
            let sample = sampler.sample_buffer(
                &mut self.scratch.logits,
                0,
                &self.context,
                &self.sampling,
            )?;
            return Ok(SpeculativeBatch {
                samples: vec![sample],
                stats: MtpStats::default(),
                ngram: NgramStats::default(),
                sampling: sampling_started.elapsed(),
            });
        }
        let started = Instant::now();
        let start = self.position;
        // The head drafts through `start + depth - 1` and the verify block
        // commits through `start + depth`; grow both caches once, up front.
        self.reserve_kv(start + depth + 1, Some(speculation))?;
        let mut drafts = Vec::with_capacity(depth);
        let mut draft_sampler = sampler.greedy_draft();
        let mut token = seed;
        for step in 0..depth {
            decode_embeddings(&self.package, &[token], speculation.mtp.embedding_rows(1)?)?;
            let mtp = &speculation.mtp;
            // Row 0 reads the target's committed hidden; deeper drafts chain
            // the head's own predicted hidden.
            let hidden = if step == 0 {
                mtp.prev_hidden()
            } else {
                mtp.predicted()
            };
            let mut batch = CommandBatch::new(&self.context)?;
            mtp.encode(
                &mut batch,
                &self.mtp_shared(),
                hidden,
                start + step,
                1,
                true,
            )?;
            self.finish(batch)?;
            let (next, margin) = draft_sampler.sample_buffer_with_margin(
                &mut speculation.mtp.logits,
                0,
                &self.context,
                &self.sampling,
            )?;
            drafts.push(next.token_id);
            // A deeper draft chains the head's own hidden for this token; when
            // the head is not sure of the token, the chained proposal is
            // rarely accepted and the extra verify row is wasted.
            if next.is_eos || margin < DRAFT_CHAIN_MIN_MARGIN {
                break;
            }
            token = next.token_id;
            draft_sampler.observe(&[token]);
        }
        let mut stats = MtpStats {
            rounds: 1,
            proposed_tokens: drafts.len(),
            drafting: started.elapsed(),
            ..MtpStats::default()
        };

        let inputs = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect::<Vec<_>>();
        let verify_started = Instant::now();
        self.forward_block(&inputs, BlockOutput::Verify(&speculation.verifier))?;
        stats.verified_tokens = inputs.len();
        stats.verification = verify_started.elapsed();

        let sampling_started = Instant::now();
        let verified = verify_greedy_drafts(sampler, &drafts, |row, sampler| {
            sampler.sample_buffer(
                &mut speculation.verifier.verify_logits,
                row * VOCAB * size_of::<f32>(),
                &self.context,
                &self.sampling,
            )
        })?;
        let sampling = sampling_started.elapsed();
        stats.accepted_tokens = verified.accepted;
        let committed = verified.accepted + 1;

        let replay_started = Instant::now();
        if committed < inputs.len() {
            self.restore_checkpoint(&speculation.verifier, committed)?;
            self.position = start + committed;
        }
        if committed == 1 {
            // The seed's draft row already used the committed hidden, so its
            // KV row is exact; only the hidden handoff advances.
            let mut batch = CommandBatch::new(&self.context)?;
            speculation
                .mtp
                .commit_hidden(&mut batch, &self.scratch.normalized, 0)?;
            self.finish(batch)?;
        } else {
            // Deeper draft rows used predicted hiddens: rebuild the head's KV
            // for the committed tokens from the target's actual normalized rows.
            self.encode_committed(speculation, &inputs[..committed])?;
        }
        stats.commit = replay_started.elapsed();
        Ok(SpeculativeBatch {
            samples: verified.samples,
            stats,
            ngram: NgramStats::default(),
            sampling,
        })
    }
}
