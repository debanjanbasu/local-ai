//! Continuous-batching decode driver: run one decode step over `n_lanes`
//! **independent** sequences in a single batched forward.
//!
//! This is the host orchestration on top of the verified GPU primitives
//! (`rope_batch_decode`, `write_kv_cache_decode`, `flash_attention_decode_
//! batched`) and the existing M=N matmul/FFN/norm path. The weight-heavy
//! projections/FFN run once at M=`n_lanes` (the ~2.7× amortization), while the
//! three per-lane ops index each lane through a shared GPU `positions` buffer.
//!
//! [`BatchedDecodeState`] owns the per-lane KV pools (one
//! [`QuantizedUnifiedKvPool`] per KV slot, mirroring the single-sequence cache
//! slotting) plus the GPU positions buffer, so the immutable model weights stay
//! shared and only the lane state is per-batch.

use local_metal::batch::CommandBatch;
use local_metal::buffer::MetalBuffer;

use crate::Error;
use crate::kv_cache::{QuantizedUnifiedKvPool, kv_bits_from_env};

use super::Pipeline;

const F16: usize = std::mem::size_of::<half::f16>();

/// Mutable per-batch lane state for continuous-batching decode. Holds one
/// [`QuantizedUnifiedKvPool`] per KV slot and the shared GPU positions buffer.
pub struct BatchedDecodeState {
    pools: Vec<QuantizedUnifiedKvPool>,
    /// `[n_lanes]` u32 absolute positions, uploaded each step.
    pos_buf: MetalBuffer,
    /// Per-lane logits-tail scratch so all lanes' final norm + logits projection
    /// encode into ONE command buffer (distinct buffers = no hazard, no per-lane
    /// GPU sync). `hidden_scratch[l]` is `[hidden]` f16, `logits_scratch[l]` is
    /// `[vocab]` f32.
    hidden_scratch: Vec<MetalBuffer>,
    logits_scratch: Vec<MetalBuffer>,
    n_lanes: usize,
    lane_capacity: usize,
}

impl BatchedDecodeState {
    /// Number of decode lanes this state backs.
    #[must_use]
    pub const fn n_lanes(&self) -> usize {
        self.n_lanes
    }

    /// Per-lane KV capacity (max context per sequence).
    #[must_use]
    pub const fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }

    /// Zero a single lane's KV across every layer (recycle a finished lane).
    pub fn reset_lane(&self, lane: usize) {
        for pool in &self.pools {
            pool.reset_lane(lane);
        }
    }
}

impl Pipeline {
    /// Allocate continuous-batching lane state: `n_lanes` sequences, each with
    /// up to `lane_capacity` tokens of KV. Pools mirror the single-sequence KV
    /// slotting (one per non-shared layer; shared layers reuse a slot).
    ///
    /// # Errors
    ///
    /// Returns an error if the pool or positions buffers cannot be allocated, or
    /// if this pipeline does not use the FP16 KV backing (batched decode is
    /// FP16-only — the quantized KV path round-trips through the CPU).
    pub fn new_batched_decode_state(
        &self,
        n_lanes: usize,
        lane_capacity: usize,
    ) -> crate::Result<BatchedDecodeState> {
        if n_lanes == 0 {
            return Err(Error::InvalidArgument("n_lanes must be >= 1".into()));
        }
        if n_lanes > self.batch_scratch.max_batch {
            return Err(Error::InvalidArgument(format!(
                "n_lanes {n_lanes} exceeds batch scratch capacity {}",
                self.batch_scratch.max_batch
            )));
        }
        let device = self.ctx.device();
        let n_kv = self.config.num_key_value_heads;
        // Distinct KV slots, with each slot's head_dim taken from a layer that
        // owns it — identical structure to `build_kv_caches`.
        let n_slots = self.kv_index_map.iter().copied().max().map_or(0, |m| m + 1);
        let mut head_dim_of = vec![0usize; n_slots];
        for (layer_idx, &slot) in self.kv_index_map.iter().enumerate() {
            head_dim_of[slot] = self.layers[layer_idx].params.head_dim;
        }
        let bits = kv_bits_from_env();
        let mut pools = Vec::with_capacity(n_slots);
        for (slot, &head_dim) in head_dim_of.iter().enumerate() {
            // Mirror the single-sequence staging cache's ring so a wrapped
            // sliding-window prefill copies slot-for-slot into the lane pool.
            let staging = &self.kv_caches[slot];
            let lane_ring = if staging.is_ringed() {
                let ring = staging.physical_positions();
                if lane_capacity < ring {
                    return Err(Error::InvalidArgument(format!(
                        "lane_capacity {lane_capacity} below sliding-window ring {ring}"
                    )));
                }
                ring
            } else {
                0
            };
            pools.push(QuantizedUnifiedKvPool::new(
                device,
                n_lanes,
                lane_capacity,
                n_kv,
                head_dim,
                bits,
                lane_ring,
            )?);
        }
        let pos_buf = MetalBuffer::empty(device, n_lanes * std::mem::size_of::<u32>())
            .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        let h = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let mut hidden_scratch = Vec::with_capacity(n_lanes);
        let mut logits_scratch = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes {
            hidden_scratch.push(
                MetalBuffer::empty(device, h * F16)
                    .map_err(|e| Error::InvalidArgument(e.to_string()))?,
            );
            logits_scratch.push(
                MetalBuffer::empty(device, vocab * std::mem::size_of::<f32>())
                    .map_err(|e| Error::InvalidArgument(e.to_string()))?,
            );
        }
        Ok(BatchedDecodeState {
            pools,
            pos_buf,
            hidden_scratch,
            logits_scratch,
            n_lanes,
            lane_capacity,
        })
    }

    /// Prefill `lane` from a prompt, seeding its KV in the pool for positions
    /// `[0..tokens.len())` and returning the last-prompt-token logits (the
    /// caller samples the lane's first generated token from these). The lane's
    /// next decode position is `tokens.len()`.
    ///
    /// Prefill runs on the single-sequence staging caches (the proven chunked
    /// prefill path), then the resulting KV rows are GPU-copied into the lane's
    /// pool region — so each lane can start at its own context length, which is
    /// the whole point of continuous batching.
    ///
    /// # Errors
    ///
    /// Returns an error on empty prompt, capacity overflow, a non-FP16 KV
    /// backing, or any GPU dispatch failure.
    pub fn prefill_lane(
        &mut self,
        state: &BatchedDecodeState,
        lane: usize,
        tokens: &[u32],
    ) -> crate::Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::InvalidArgument("prefill prompt is empty".into()));
        }
        if lane >= state.n_lanes {
            return Err(Error::InvalidArgument(format!(
                "lane {lane} out of range (n_lanes {})",
                state.n_lanes
            )));
        }
        let len = tokens.len();
        if len > state.lane_capacity {
            return Err(Error::InvalidArgument(format!(
                "prompt length {len} exceeds lane_capacity {}",
                state.lane_capacity
            )));
        }
        // Stage the prefill on the single-sequence caches, capture last-token
        // logits, then copy the freshly encoded KV codes into the lane region.
        self.reset();
        let final_in_a = self.prefill_prompt(tokens)?;
        self.write_logits(final_in_a);
        let vocab = self.config.vocab_size;
        let logits = self.logits_buf.as_slice::<f32>()[..vocab].to_vec();

        let caches = &self.kv_caches;
        for (slot, pool) in state.pools.iter().enumerate() {
            pool.copy_prefill_from(&caches[slot], lane, len)?;
        }
        Ok(logits)
    }

    /// Prefill a lane from an already-prepared multimodal prompt (text + media
    /// soft-token embeddings). Mirrors [`Self::prefill_lane`] but stages the
    /// soft-token-aware single-sequence prefill
    /// ([`Pipeline::prefill_prepared_multimodal`]) instead of the plain
    /// token-only prefill, then copies the freshly written KV rows into `lane`.
    ///
    /// The KV copy length is `prepared.tokens.len()`: media soft tokens replace
    /// the embedding at existing placeholder positions and do not add KV rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the prompt is empty, the PLE/token lengths disagree,
    /// the lane is out of range, the prepared length exceeds the effective lane
    /// context, the KV backing is not FP16, or any GPU dispatch fails.
    pub fn prefill_lane_prepared(
        &mut self,
        state: &BatchedDecodeState,
        lane: usize,
        prepared: &crate::multimodal::PreparedMultimodalPrompt,
    ) -> crate::Result<Vec<f32>> {
        let len = prepared.tokens.len();
        if len == 0 {
            return Err(Error::InvalidArgument(
                "multimodal prefill prompt is empty".into(),
            ));
        }
        if prepared.ple_tokens.len() != len {
            return Err(Error::InvalidArgument(format!(
                "PLE token count {} does not match token count {len}",
                prepared.ple_tokens.len()
            )));
        }
        if lane >= state.n_lanes() {
            return Err(Error::InvalidArgument(format!(
                "lane {lane} out of range (n_lanes {})",
                state.n_lanes()
            )));
        }
        let limit = state.lane_capacity().min(self.max_effective_context());
        if len > limit {
            return Err(Error::InvalidArgument(format!(
                "multimodal prompt length {len} exceeds effective lane context {limit}"
            )));
        }
        // Stage the multimodal prefill on the single-sequence caches, capture
        // last-token logits, then copy the freshly encoded KV codes into the lane.
        self.reset();
        let final_in_a = self.prefill_prepared_multimodal(prepared)?;
        self.write_logits(final_in_a);
        let vocab = self.config.vocab_size;
        let logits = self.logits_buf.as_slice::<f32>()[..vocab].to_vec();

        let caches = &self.kv_caches;
        for (slot, pool) in state.pools.iter().enumerate() {
            pool.copy_prefill_from(&caches[slot], lane, len)?;
        }
        Ok(logits)
    }

    /// Run one batched decode step: each lane consumes its own `(token,
    /// ple_token, position)` and produces a full vocab logits vector. Updates
    /// each lane's KV at its position. Returns per-lane logits.
    ///
    /// `positions[l]` is lane `l`'s **current** absolute position (where the new
    /// token's K/V is written, and the query attends `[0..=position]`). Callers
    /// advance positions between steps.
    ///
    /// # Errors
    ///
    /// Returns an error on argument-shape mismatch, capacity overflow, or any
    /// GPU dispatch failure.
    #[allow(clippy::too_many_lines)]
    pub fn decode_batch_step(
        &mut self,
        state: &mut BatchedDecodeState,
        tokens: &[u32],
        ple_tokens: &[u32],
        positions: &[u32],
    ) -> crate::Result<Vec<Vec<f32>>> {
        let n = tokens.len();
        if n == 0 || n > state.n_lanes {
            return Err(Error::InvalidArgument(format!(
                "active lanes {n} must be in 1..={}",
                state.n_lanes
            )));
        }
        if ple_tokens.len() != n || positions.len() != n {
            return Err(Error::InvalidArgument(
                "tokens, ple_tokens and positions must have equal length".into(),
            ));
        }
        for (l, &pos) in positions.iter().enumerate() {
            if pos as usize >= state.lane_capacity {
                return Err(Error::InvalidArgument(format!(
                    "lane {l} position {pos} exceeds capacity {}",
                    state.lane_capacity
                )));
            }
        }

        // Upload per-lane positions for rope/scatter/attention.
        state
            .pos_buf
            .copy_from_bytes(bytemuck::cast_slice(positions), 0);

        let h = self.config.hidden_size as u32;
        let m = n as u32;
        let eps = self.config.rms_norm_eps as f32;

        // --- Embeddings + scale + PLE (batched, M=n), mirroring the prefill
        //     batch path. ---
        self.gather_embeddings_with_overrides(tokens, &self.batch_scratch.hidden_a, &[], 0)?;
        let mut batch = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        let embed_scale = (self.config.hidden_size as f32).sqrt();
        self.kernels.scale_in_place_gpu_into(
            &mut batch,
            &self.batch_scratch.hidden_a,
            embed_scale,
            m * h,
        )?;

        let ple_ready = if let (Some(table), Some(model_proj), Some(proj_norm)) = (
            self.ple_table.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let bs = &self.batch_scratch;
            let pld = self.config.hidden_size_per_layer_input;
            let nl = self.config.num_hidden_layers;
            let row = pld * nl;
            let mut r = Vec::new();
            for (i, &tok) in ple_tokens.iter().enumerate() {
                if table.dequant_row_into(tok as usize, &mut r).is_ok() {
                    bs.ple_tok
                        .copy_from_bytes(bytemuck::cast_slice(&r), i * row * F16);
                }
            }
            let total = (n * row) as u32;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_tok,
                (pld as f32).sqrt(),
                total,
            )?;
            model_proj.matmul_nt_into(&mut batch, &self.kernels, &bs.hidden_a, &bs.ple_proj, m)?;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_proj,
                1.0 / embed_scale,
                total,
            )?;
            self.kernels.rms_norm_into(
                &mut batch,
                &bs.ple_proj,
                proj_norm,
                &bs.ple_proj,
                pld as u32,
                (n * nl) as u32,
                eps,
            )?;
            self.kernels.residual_add_into(
                &mut batch,
                &bs.ple_proj,
                &bs.ple_tok,
                &bs.ple_all,
                total,
            )?;
            self.kernels.scale_in_place_gpu_into(
                &mut batch,
                &bs.ple_all,
                std::f32::consts::FRAC_1_SQRT_2,
                total,
            )?;
            true
        } else {
            false
        };
        batch.commit_and_wait().map_err(Error::Metal)?;

        // --- Transformer layers (batched, M=n) with per-lane attention. ---
        let ple_arg =
            ple_ready.then_some((&self.batch_scratch.ple_all, self.config.num_hidden_layers));
        let chunk = std::env::var("LOCAL_AI_BATCH_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(8);
        let mut cur_in_a = true;
        let mut batch = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let bs = &self.batch_scratch;
            let (inp, out) = if cur_in_a {
                (&bs.hidden_a, &bs.hidden_b)
            } else {
                (&bs.hidden_b, &bs.hidden_a)
            };
            let slot = self.kv_index_map[i];
            layer.forward_decode_batch_turbo(
                &mut batch,
                &self.kernels,
                n,
                &state.pos_buf,
                inp,
                out,
                bs,
                &self.scratch.ones,
                &state.pools[slot],
                ple_arg,
            )?;
            if (i + 1) % chunk == 0 && i + 1 < self.layers.len() {
                batch.commit_and_renew(&self.ctx).map_err(Error::Metal)?;
            }
            cur_in_a = !cur_in_a;
        }
        batch.commit_and_wait().map_err(Error::Metal)?;

        // --- Final norm (batched) + per-lane logits projection. ---
        let final_in_a = cur_in_a;
        let final_hidden = if final_in_a {
            &self.batch_scratch.hidden_a
        } else {
            &self.batch_scratch.hidden_b
        };
        let vocab = self.config.vocab_size as u32;
        // All lanes' final-norm + logits projection encode into ONE command
        // buffer using per-lane scratch (distinct buffers ⇒ no hazard, so the
        // GPU runs them concurrently and we pay a single sync instead of N).
        let mut tail = CommandBatch::new(&self.ctx).map_err(Error::Metal)?;
        for lane in 0..n {
            let hid = &state.hidden_scratch[lane];
            let lg = &state.logits_scratch[lane];
            self.kernels.gather_strided_f16_into(
                &mut tail,
                final_hidden,
                hid,
                h,
                (lane as u32) * h,
                h,
                1,
            )?;
            self.kernels
                .rms_norm_into(&mut tail, hid, &self.output_norm, hid, h, 1, eps)?;
            self.token_embd
                .matvec_f32out_into(&mut tail, &self.kernels, hid, lg)?;
            if self.logit_softcap > 0.0 {
                self.kernels
                    .logit_softcap_gpu_into(&mut tail, lg, self.logit_softcap, vocab)?;
            }
        }
        tail.commit_and_wait().map_err(Error::Metal)?;

        let mut out_logits = Vec::with_capacity(n);
        for lane in 0..n {
            out_logits
                .push(state.logits_scratch[lane].as_slice::<f32>()[..vocab as usize].to_vec());
        }
        Ok(out_logits)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::explicit_counter_loop
    )]

    use super::Pipeline;
    use std::path::{Path, PathBuf};

    /// Bounded logit L∞ envelope (relative to the top logit) between the
    /// single-sequence (M=1 matvec) and batched (M=N matmul) decode paths.
    /// 4-bit `TurboQuant` KV makes ~1e-3 cross-kernel FP16 drift discontinuous
    /// (a sub-bin input change can flip a K/V code), so this is a compatibility
    /// cap, not a kernel-equality bound — see `batched_tq_decode_matches_
    /// independent_single_seq` for the strict (1e-4) same-codes proof.
    const TQ_CROSS_KERNEL_LOGIT_REL_TOL: f32 = 0.15;

    fn argmax(v: &[f32]) -> usize {
        let mut bi = 0;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }

    /// Top-2 logit gap of the reference distribution at a step. A greedy argmax
    /// flip between two near-numerically-identical paths is only acceptable when
    /// this gap is within the cross-path noise envelope (a genuine tie).
    fn top2_gap(v: &[f32]) -> f32 {
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for &x in v {
            if x > top1 {
                top2 = top1;
                top1 = x;
            } else if x > top2 {
                top2 = x;
            }
        }
        top1 - top2
    }

    fn find_model_dir() -> Option<PathBuf> {
        for cand in [
            "../models/gemma-4-e2b-it",
            "models/gemma-4-e2b-it",
            "../models/gemma-4-e2b-q4",
            "models/gemma-4-e2b-q4",
        ] {
            let p = Path::new(cand);
            if p.join("model.lma").exists() || p.join("config.json").exists() {
                return Some(p.to_path_buf());
            }
        }
        None
    }

    /// End-to-end continuous-batching correctness on the real model: decoding N
    /// distinct first tokens as N lanes at position 0 must produce, per lane, the
    /// same argmax (and near-identical logits) as the proven single-sequence
    /// decode of that token at position 0. Run with:
    /// `cargo test -p local-engine --release batched_decode_matches_single -- --ignored --nocapture`
    #[test]
    #[ignore = "requires the local model bundle; run manually"]
    fn batched_decode_matches_single_sequence_at_pos0() {
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");
        let vocab = pipe.config.vocab_size;

        // A handful of distinct, valid first tokens.
        let tokens: Vec<u32> = [2u32, 100, 1000, 5000]
            .into_iter()
            .filter(|&t| (t as usize) < vocab)
            .collect();
        let n = tokens.len();

        // Reference: single-sequence decode of each token at position 0.
        let mut ref_argmax = Vec::with_capacity(n);
        let mut ref_logits = Vec::with_capacity(n);
        for &tok in &tokens {
            pipe.reset();
            let final_in_a = pipe.forward_token(tok, 0).expect("single forward");
            pipe.write_logits(final_in_a);
            let logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
            ref_argmax.push(argmax(&logits));
            ref_logits.push(logits);
        }

        // Batched: all tokens as N lanes at position 0.
        let mut state = pipe
            .new_batched_decode_state(n, 4096)
            .expect("batched state");
        let positions = vec![0u32; n];
        let got = pipe
            .decode_batch_step(&mut state, &tokens, &tokens, &positions)
            .expect("batched step");

        assert_eq!(got.len(), n);

        // Correctness signal #1: every lane's greedy token (argmax) must match
        // the proven single-sequence decode exactly. This is the meaningful
        // invariant for a decode engine.
        for lane in 0..n {
            let ga = argmax(&got[lane]);
            assert_eq!(
                ga, ref_argmax[lane],
                "lane {lane} (token {}) argmax: batched {ga} != single {}",
                tokens[lane], ref_argmax[lane]
            );
        }

        // Correctness signal #2 (quantization-aware envelope): the single path
        // uses M=1 matvec/single-token kernels; the batched path uses M=N
        // matmul/batched kernels. Those differ at ~1e-3 FP16 precision, and 4-bit
        // TurboQuant KV encoding is *discontinuous* — a sub-bin input difference
        // can flip a K/V code by one level, a jump that propagates through all 35
        // layers. So the cross-path logit L∞ is a bounded *compatibility* check,
        // not the bitwise kernel-equality check (that is
        // `batched_tq_decode_matches_independent_single_seq`, which holds to 1e-4
        // given identical codes). A regression past this envelope is a real bug.
        for lane in 0..n {
            let r = &ref_logits[lane];
            let g = &got[lane];
            let mut max_abs = 0.0f32;
            for i in 0..vocab {
                max_abs = max_abs.max((r[i] - g[i]).abs());
            }
            let scale = r[ref_argmax[lane]].abs().max(1.0);
            assert!(
                max_abs <= TQ_CROSS_KERNEL_LOGIT_REL_TOL * scale,
                "lane {lane}: max logit diff {max_abs} exceeds quantized cross-path \
                 envelope (scale {scale}, rel {:.4})",
                max_abs / scale
            );
        }

        // Guardrail A — lane isolation: N lanes decoding the SAME token at the
        // same position must produce bit-identical logits. A cross-lane
        // addressing / scratch-aliasing bug would break this while still
        // preserving argmax, so it is the decisive check that the relaxed
        // envelope above is hiding quantization noise, not a lane bug.
        let dup: Vec<u32> = vec![tokens[0]; n];
        let dup_got = pipe
            .decode_batch_step(&mut state, &dup, &dup, &positions)
            .expect("dup batched step");
        for lane in 1..n {
            assert_eq!(
                dup_got[lane], dup_got[0],
                "lane {lane} differs from lane 0 for identical duplicate-token input \
                 (cross-lane addressing bug)"
            );
        }

        // Guardrail B — diff follows the token, not the lane index: reversing the
        // lane order must move each lane's argmax with its token.
        let mut rev = tokens.clone();
        rev.reverse();
        let rev_got = pipe
            .decode_batch_step(&mut state, &rev, &rev, &positions)
            .expect("rev batched step");
        for lane in 0..n {
            assert_eq!(
                argmax(&rev_got[lane]),
                ref_argmax[n - 1 - lane],
                "reversed lane {lane} argmax does not follow its token \
                 (diff is lane-indexed, not token-indexed)"
            );
        }

        eprintln!("batched decode matches single-sequence for {n} lanes at pos 0");
    }

    /// Multi-step greedy correctness: this is the real continuous-batching loop.
    /// For each lane we greedily decode `STEPS` tokens batched (writing KV into
    /// the unified pool at advancing positions and attending the growing
    /// history), and assert the generated token sequence is identical to the
    /// proven single-sequence greedy decode. This pins the cross-step KV
    /// accumulation, not just a single forward.
    #[test]
    #[ignore = "requires the local model bundle; run manually"]
    fn batched_greedy_matches_single_sequence_multistep() {
        const STEPS: usize = 6;
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");
        let vocab = pipe.config.vocab_size;

        let starts: Vec<u32> = [2u32, 100, 1000, 5000]
            .into_iter()
            .filter(|&t| (t as usize) < vocab)
            .collect();
        let n = starts.len();

        // Reference: single-sequence greedy continuation per lane, keeping each
        // step's reference logits so a divergence can be checked for near-tie.
        let mut ref_seqs: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut ref_step_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n);
        for &start in &starts {
            pipe.reset();
            let mut seq = Vec::with_capacity(STEPS);
            let mut steps = Vec::with_capacity(STEPS);
            let mut tok = start;
            for pos in 0..STEPS {
                let fa = pipe.forward_token(tok, pos).expect("single forward");
                pipe.write_logits(fa);
                let logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
                tok = argmax(&logits) as u32;
                seq.push(tok);
                steps.push(logits);
            }
            ref_seqs.push(seq);
            ref_step_logits.push(steps);
        }

        // Batched: all lanes greedily decode together, advancing positions.
        let mut state = pipe.new_batched_decode_state(n, 4096).expect("state");
        let mut cur: Vec<u32> = starts.clone();
        let mut got_seqs: Vec<Vec<u32>> = vec![Vec::with_capacity(STEPS); n];
        for pos in 0..STEPS {
            let positions = vec![pos as u32; n];
            let logits = pipe
                .decode_batch_step(&mut state, &cur, &cur, &positions)
                .expect("batched step");
            for lane in 0..n {
                let t = argmax(&logits[lane]) as u32;
                got_seqs[lane].push(t);
                cur[lane] = t;
            }
        }

        // Compare per step. Tokens must match until a step where the reference's
        // top-2 logits are a genuine near-tie (gap within the cross-path noise
        // envelope) — there a one-bin quantization flip can legitimately pick the
        // runner-up. Once the paths pick different tokens their histories diverge,
        // so the suffix is no longer comparable and we stop. A high-margin flip
        // (gap above the envelope) is a real bug and fails.
        for lane in 0..n {
            for pos in 0..STEPS {
                if got_seqs[lane][pos] == ref_seqs[lane][pos] {
                    continue;
                }
                let ref_logits = &ref_step_logits[lane][pos];
                let scale = ref_logits[ref_seqs[lane][pos] as usize].abs().max(1.0);
                let gap = top2_gap(ref_logits);
                assert!(
                    gap <= 2.0 * TQ_CROSS_KERNEL_LOGIT_REL_TOL * scale,
                    "lane {lane} (start {}) step {pos}: high-margin greedy divergence \
                     batched={} single={} (top-2 gap {gap}, scale {scale}) — not a tie",
                    starts[lane],
                    got_seqs[lane][pos],
                    ref_seqs[lane][pos]
                );
                // Histories now differ; suffix is not comparable.
                break;
            }
        }
        eprintln!("batched greedy matches single-sequence over {STEPS} steps, {n} lanes");
    }

    /// Prefill + decode with lanes at **different** context lengths — the real
    /// continuous-batching scenario. Two lanes get prompts of different lengths,
    /// so they decode at different positions in each batched step. Each lane's
    /// greedy continuation must match single-sequence greedy generation of the
    /// same prompt. This validates `prefill_lane` (pool seeding) + multi-step
    /// decode with per-lane positions together.
    #[test]
    #[ignore = "requires the local model bundle; run manually"]
    fn prefill_then_batched_greedy_matches_single_sequence() {
        const STEPS: usize = 5;
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");
        let vocab = pipe.config.vocab_size;

        let prompts: Vec<Vec<u32>> = vec![vec![2u32, 100, 1000], vec![2u32, 100, 1000, 5000, 7]]
            .into_iter()
            .map(|p| p.into_iter().filter(|&t| (t as usize) < vocab).collect())
            .collect();
        let n = prompts.len();

        // Reference: single-sequence greedy generation per prompt.
        let mut ref_seqs: Vec<Vec<u32>> = Vec::with_capacity(n);
        for prompt in &prompts {
            pipe.reset();
            let fa = pipe.prefill_prompt(prompt).expect("prefill");
            pipe.write_logits(fa);
            let mut logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
            let mut seq = Vec::with_capacity(STEPS);
            let mut pos = prompt.len();
            for _ in 0..STEPS {
                let tok = argmax(&logits) as u32;
                seq.push(tok);
                let fa = pipe.forward_token(tok, pos).expect("decode");
                pipe.write_logits(fa);
                logits = pipe.logits_buf.as_slice::<f32>()[..vocab].to_vec();
                pos += 1;
            }
            ref_seqs.push(seq);
        }

        // Batched: prefill both lanes, then decode together with per-lane
        // positions advancing from each prompt's length.
        let mut state = pipe.new_batched_decode_state(n, 4096).expect("state");
        let mut cur = vec![0u32; n];
        let mut lane_pos = vec![0usize; n];
        for (lane, prompt) in prompts.iter().enumerate() {
            let logits = pipe
                .prefill_lane(&state, lane, prompt)
                .expect("prefill_lane");
            cur[lane] = argmax(&logits) as u32;
            lane_pos[lane] = prompt.len();
        }
        let mut got_seqs: Vec<Vec<u32>> = vec![Vec::with_capacity(STEPS); n];
        for lane in 0..n {
            got_seqs[lane].push(cur[lane]);
        }
        for _ in 1..STEPS {
            let positions: Vec<u32> = lane_pos.iter().map(|&p| p as u32).collect();
            let logits = pipe
                .decode_batch_step(&mut state, &cur, &cur, &positions)
                .expect("batched step");
            for lane in 0..n {
                let t = argmax(&logits[lane]) as u32;
                got_seqs[lane].push(t);
                cur[lane] = t;
                lane_pos[lane] += 1;
            }
        }

        for lane in 0..n {
            assert_eq!(
                got_seqs[lane],
                ref_seqs[lane],
                "lane {lane} (prompt len {}) diverged:\n batched={:?}\n single ={:?}",
                prompts[lane].len(),
                got_seqs[lane],
                ref_seqs[lane]
            );
        }
        eprintln!(
            "prefill+batched greedy matches single-sequence: {n} lanes, prompt lens {:?}",
            prompts.iter().map(Vec::len).collect::<Vec<_>>()
        );
    }

    /// Throughput payoff: aggregate tokens/sec as lane count grows. Confirms the
    /// roofline prediction (~2.7× aggregate around 8 lanes) on the real model.
    /// Run with:
    /// `cargo test -p local-engine --release batched_decode_throughput -- --ignored --nocapture`
    #[test]
    #[ignore = "perf timing; run manually"]
    fn batched_decode_throughput() {
        const STEPS: usize = 32;
        let Some(model_dir) = find_model_dir() else {
            eprintln!("model bundle not found; skipping");
            return;
        };
        let mut pipe = Pipeline::new_qat(&model_dir, 4096).expect("load model");
        let vocab = pipe.config.vocab_size as u32;

        let base = 1.0f64; // single-lane reference, filled in first.
        let mut single_tps = base;
        for &lanes in &[1usize, 2, 4, 8, 16, 32] {
            if lanes > pipe.batch_scratch.max_batch {
                continue;
            }
            let mut state = pipe.new_batched_decode_state(lanes, 4096).expect("state");
            let tokens: Vec<u32> = (0..lanes).map(|l| ((l * 7 + 2) as u32) % vocab).collect();
            let positions = vec![0u32; lanes];
            // Warmup.
            for _ in 0..4 {
                let _ = pipe
                    .decode_batch_step(&mut state, &tokens, &tokens, &positions)
                    .expect("warm");
            }
            let start = std::time::Instant::now();
            for s in 0..STEPS {
                let positions = vec![s as u32; lanes];
                let _ = pipe
                    .decode_batch_step(&mut state, &tokens, &tokens, &positions)
                    .expect("step");
            }
            let elapsed = start.elapsed().as_secs_f64();
            let per_step_ms = elapsed / STEPS as f64 * 1e3;
            let agg_tps = (lanes * STEPS) as f64 / elapsed;
            if lanes == 1 {
                single_tps = agg_tps;
            }
            eprintln!(
                "[batched decode] lanes={lanes:>2}: {per_step_ms:6.2} ms/step, \
                 {agg_tps:7.1} tok/s aggregate ({:.2}x vs 1 lane)",
                agg_tps / single_tps
            );
        }
    }
}
