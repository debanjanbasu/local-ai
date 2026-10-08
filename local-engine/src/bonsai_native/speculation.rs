use local_metal::draft::{DraftKernels, DraftTopTwo};

use super::{
    BlockOutput, BonsaiMetalTensor, BonsaiModel, BonsaiPackage, CommandBatch,
    DRAFT_CHAIN_MIN_MARGIN, Instant, MetalBuffer, MetalContext, MtpStats, NgramStats, Sampler,
    ShaderLibrary, Speculation, SpeculativeBatch, VOCAB, cancelled_speculative_batch,
    decode_embeddings, draft_depth,
};

/// GPU-resident greedy drafting: each draft step gathers its token's
/// embedding on the GPU and selects the next token there, so a step is one
/// submission with no host-side embedding decode or logit selection.
pub(super) struct Drafter {
    kernels: DraftKernels,
    /// `tokens[0]` is the round's seed; step `k` writes its draft to `tokens[k + 1]`.
    tokens: MetalBuffer,
    /// Step `k`'s top two at index `k`.
    results: MetalBuffer,
    /// The target's `PTQ1_0` `token_embd`, bound for the GPU gather.
    embeddings: BonsaiMetalTensor,
}

impl Drafter {
    pub(super) fn new(
        context: &MetalContext,
        shaders: &ShaderLibrary,
        package: &BonsaiPackage,
        depth: usize,
    ) -> crate::Result<Self> {
        let depth = depth.max(1);
        Ok(Self {
            kernels: DraftKernels::new(context, shaders, VOCAB)?,
            tokens: MetalBuffer::empty(context.device(), (depth + 1) * size_of::<u32>())?,
            results: MetalBuffer::empty(context.device(), depth * size_of::<DraftTopTwo>())?,
            embeddings: package.metal_tensor(context.device(), "token_embd.weight")?,
        })
    }
}

/// One draft step's proposal and the head's logit lead for it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Draft {
    token: u32,
    margin: f32,
}

impl BonsaiModel {
    /// One verified speculative round. A cancelled request submits neither the
    /// draft pass nor the verify block and hands back
    /// `cancelled_speculative_batch`; the position and both caches keep the
    /// exact state of the last committed round.
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
            let sample = self.sample(sampler)?;
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
        let mut draft_sampler = sampler.greedy_draft();
        let drafts = if draft_sampler.applies_penalties() {
            self.draft_on_host(
                speculation,
                &mut draft_sampler,
                seed,
                depth,
                DRAFT_CHAIN_MIN_MARGIN,
            )?
        } else {
            self.draft_on_device(
                speculation,
                &draft_sampler,
                seed,
                depth,
                DRAFT_CHAIN_MIN_MARGIN,
            )?
        };
        let drafts = drafts.iter().map(|draft| draft.token).collect::<Vec<_>>();
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
        let verified =
            self.verify_rows(sampler, &drafts, &mut speculation.verifier.verify_logits)?;
        let sampling = sampling_started.elapsed();
        stats.accepted_tokens = verified.accepted;
        let committed = verified.accepted + 1;

        let replay_started = Instant::now();
        // The target's rollback and the head's commit share one submission.
        let mut batch = CommandBatch::new(&self.context)?;
        self.encode_commit_verified(
            &mut batch,
            &mut speculation.verifier,
            inputs.len(),
            committed,
        )?;
        self.position = start + committed;
        if committed == 1 {
            // The seed's draft row already used the committed hidden, so its
            // KV row is exact; only the hidden handoff advances.
            speculation
                .mtp
                .commit_hidden(&mut batch, &self.scratch.normalized, 0)?;
        } else {
            // Deeper draft rows used predicted hiddens: rebuild the head's KV
            // for the committed tokens from the target's actual normalized rows.
            let committed_tokens = &inputs[..committed];
            decode_embeddings(
                &self.package,
                committed_tokens,
                speculation.mtp.embedding_rows(committed)?,
            )?;
            self.encode_head_rows(&mut batch, speculation, committed, start)?;
        }
        self.finish(batch)?;
        stats.commit = replay_started.elapsed();
        Ok(SpeculativeBatch {
            samples: verified.samples,
            stats,
            ngram: NgramStats::default(),
            sampling,
        })
    }

    /// Draft up to `depth` greedy tokens after `seed` at the current position:
    /// row 0 reads the target's committed hidden, deeper rows chain the head's
    /// own predicted hidden. A draft that is EOS, or whose margin is below
    /// `min_margin`, is still proposed but ends the chain: when the head is not
    /// sure of a token, a deeper proposal is rarely accepted and its verify row
    /// is wasted.
    ///
    /// Each step is one submission: the embedding gather, the head layer and
    /// the exact top two over its logits all run on the GPU. The host reads
    /// only the top two, which equals [`Self::draft_on_host`]'s selection
    /// whenever the sampler applies no penalties.
    fn draft_on_device(
        &self,
        speculation: &mut Speculation,
        sampler: &Sampler,
        seed: u32,
        depth: usize,
        min_margin: f32,
    ) -> crate::Result<Vec<Draft>> {
        let start = self.position;
        let drafter = &mut speculation.drafter;
        let slots = drafter.results.length() / size_of::<DraftTopTwo>();
        if depth > slots {
            return Err(crate::Error::InvalidArgument(
                "draft depth exceeds the drafter's slots".into(),
            ));
        }
        drafter.tokens.as_mut_slice::<u32>()[0] = seed;
        let drafter = &speculation.drafter;
        let mtp = &speculation.mtp;
        let embeddings = drafter.embeddings.ptq1_matrix()?;
        let mut drafts = Vec::with_capacity(depth);
        for step in 0..depth {
            let hidden = if step == 0 {
                mtp.prev_hidden()
            } else {
                mtp.predicted()
            };
            let mut batch = CommandBatch::new(&self.context)?;
            mtp.encode_draft(
                &mut batch,
                &self.mtp_shared(),
                &drafter.kernels,
                embeddings,
                &drafter.tokens,
                step as u32,
                hidden,
                start + step,
            )?;
            drafter.kernels.top_two(
                &mut batch,
                &mtp.logits,
                VOCAB,
                &drafter.tokens,
                step as u32 + 1,
                &drafter.results,
                step as u32,
            )?;
            self.finish(batch)?;
            let top = drafter.results.as_slice::<DraftTopTwo>()[step];
            let (next, margin) =
                sampler.greedy_from_top_two(top.best_id, top.best_logit, top.second_logit);
            drafts.push(Draft {
                token: next.token_id,
                margin,
            });
            if next.is_eos || margin < min_margin {
                break;
            }
        }
        Ok(drafts)
    }

    /// The host-selected draft chain [`Self::draft_on_device`] reproduces:
    /// host embedding decode, then the sampler's own penalized greedy
    /// selection after every step. Required whenever penalties apply.
    fn draft_on_host(
        &self,
        speculation: &mut Speculation,
        sampler: &mut Sampler,
        seed: u32,
        depth: usize,
        min_margin: f32,
    ) -> crate::Result<Vec<Draft>> {
        let start = self.position;
        let mut drafts = Vec::with_capacity(depth);
        let mut token = seed;
        for step in 0..depth {
            decode_embeddings(&self.package, &[token], speculation.mtp.embedding_rows(1)?)?;
            let mtp = &speculation.mtp;
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
            let (next, margin) = sampler.sample_buffer_with_margin(
                &mut speculation.mtp.logits,
                0,
                &self.context,
                &self.sampling,
            )?;
            drafts.push(Draft {
                token: next.token_id,
                margin,
            });
            if next.is_eos || margin < min_margin {
                break;
            }
            token = next.token_id;
            sampler.observe(&[token]);
        }
        Ok(drafts)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::super::{KvOptions, NgramSettings};
    use super::*;
    use crate::bonsai::DEFAULT_BONSAI_GGUF;
    use crate::bonsai_mtp::{DEFAULT_BONSAI_MTP_ARTIFACT, MtpSettings};
    use crate::sampler::SamplingParams;

    fn draft_sampler(presence_penalty: f32, eos_tokens: Vec<u32>) -> Sampler {
        Sampler::new(
            VOCAB,
            SamplingParams {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                presence_penalty,
                repetition_penalty: 1.0,
                eos_tokens,
                seed: 0,
            },
        )
        .greedy_draft()
    }

    /// The GPU draft chain must propose exactly what the host chain proposes:
    /// same tokens and bit-identical margins at every depth, through the
    /// margin gate and EOS, from several committed states and seeds.
    #[test]
    #[ignore = "requires the Bonsai GGUF, the MTP head and a Metal device"]
    fn device_drafts_match_host_drafts_bit_for_bit() {
        let depth = 4;
        let settings =
            MtpSettings::new(DEFAULT_BONSAI_MTP_ARTIFACT.into(), depth).expect("head settings");
        let package = BonsaiPackage::open(DEFAULT_BONSAI_GGUF).expect("open Bonsai GGUF");
        let mut model = BonsaiModel::load(
            package,
            64,
            16,
            None,
            Some(&settings),
            NgramSettings::default(),
            KvOptions::default(),
        )
        .expect("load");
        let prefix = [
            248_045_u32,
            846,
            198,
            814,
            20139,
            1204,
            264,
            5010,
            2336,
            13081,
        ];
        model.reset();
        model
            .forward_block(&prefix, BlockOutput::Hidden)
            .expect("prefix block");
        model.ingest_committed(&prefix).expect("ingest prefix");
        let mut speculation = model.speculation.take().expect("speculation");
        model
            .reserve_kv(model.position + depth + 1, Some(&mut speculation))
            .expect("reserve");
        let mut gated = 0;
        let mut compared = 0;
        for seed in [45_776_u32, 11, 264, 198, 13, 3010] {
            let mut eos_case = None;
            for case in 0..3 {
                let (min_margin, eos) = match case {
                    0 => (f32::NEG_INFINITY, Vec::new()),
                    1 => (DRAFT_CHAIN_MIN_MARGIN, Vec::new()),
                    // The ungated chain's second draft as EOS: both must stop there.
                    _ => (f32::NEG_INFINITY, eos_case.into_iter().collect()),
                };
                let mut host_sampler = draft_sampler(0.0, eos.clone());
                let host = model
                    .draft_on_host(&mut speculation, &mut host_sampler, seed, depth, min_margin)
                    .expect("host drafts");
                if case == 0 {
                    eos_case = host.get(1).map(|draft| draft.token);
                }
                // Poison what the device path must not read: the host-staged
                // rows and the previous draft's logits.
                speculation
                    .mtp
                    .embedding_rows(1)
                    .expect("rows")
                    .fill(f32::NAN);
                speculation.mtp.logits.as_mut_slice::<f32>().fill(f32::NAN);
                let device = model
                    .draft_on_device(
                        &mut speculation,
                        &draft_sampler(0.0, eos),
                        seed,
                        depth,
                        min_margin,
                    )
                    .expect("device drafts");
                assert_eq!(host.len(), device.len(), "seed {seed} case {case}");
                for (host, device) in host.iter().zip(&device) {
                    assert_eq!(host.token, device.token, "seed {seed} case {case}");
                    assert_eq!(
                        host.margin.to_bits(),
                        device.margin.to_bits(),
                        "seed {seed} case {case}"
                    );
                }
                if host.len() < depth {
                    gated += 1;
                }
                compared += host.len();
            }
        }
        assert!(
            gated > 0,
            "no chain stopped early, so the gate went untested"
        );
        assert!(compared > 18, "too few draft steps compared: {compared}");
        // Penalties stay on the host path, which the device path cannot express.
        assert!(draft_sampler(0.5, Vec::new()).applies_penalties());
        assert!(!draft_sampler(0.0, Vec::new()).applies_penalties());
        model.speculation = Some(speculation);
    }
}
