use super::{BonsaiOps, CommandBatch, GDN_HEADS, MetalBuffer, arg, need, no_alias, sequence_bytes};

impl BonsaiOps {
    pub fn conv_step(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        self.conv_sequence(b, input, weights, history, out, 1)
    }

    pub fn conv_sequence(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.conv_sequence_into(b, input, weights, history, history, out, tokens)
    }

    /// [`Self::conv_sequence`] writing the final history to `final_history`,
    /// which may be `history` itself or a separate buffer.
    pub fn conv_sequence_into(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        final_history: &MetalBuffer,
        out: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let n = 10240usize;
        let bytes = sequence_bytes(tokens, n as u32)?;
        need(input, bytes)?;
        need(weights, n * 4 * 4)?;
        need(history, n * 3 * 4)?;
        need(final_history, n * 3 * 4)?;
        need(out, bytes)?;
        no_alias(out, &[input, weights, history, final_history])?;
        no_alias(history, &[input, weights])?;
        no_alias(final_history, &[input, weights])?;
        self.go(
            b,
            4,
            &[
                (input, 0),
                (weights, 0),
                (history, 0),
                (out, 0),
                (final_history, 0),
            ],
            &[tokens],
            &[],
            n.div_ceil(256),
            256,
        );
        Ok(())
    }
    pub fn l2_normalize_qk(
        &self,
        b: &mut CommandBatch,
        qkv: &MetalBuffer,
        epsilon: f32,
    ) -> crate::Result<()> {
        self.l2_normalize_qk_rows(b, qkv, epsilon, 1)
    }

    pub fn l2_normalize_qk_rows(
        &self,
        b: &mut CommandBatch,
        qkv: &MetalBuffer,
        epsilon: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid L2 epsilon"));
        }
        // Only Q/K in the last row are accessed; V may be absent for callers
        // normalizing a single Q/K vector instead of the complete QKV tensor.
        need(qkv, sequence_bytes(tokens, 10240)? - 6144 * 4)?;
        self.go(
            b,
            5,
            &[(qkv, 0)],
            &[],
            &[epsilon],
            32 * tokens as usize,
            128,
        );
        Ok(())
    }

    /// [`Self::conv_sequence`], [`Self::l2_normalize_qk_rows`] and
    /// [`Self::decay_beta_rows`] in one dispatch, with bitwise identical
    /// results. One threadgroup normalizes a head for every token in turn, so
    /// this is for short blocks (decode and verification), not prefill.
    pub fn conv_l2_decay(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
        epsilon: f32,
        decay_inputs: [&MetalBuffer; 4],
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.conv_l2_decay_into(
            b,
            input,
            weights,
            [history, history],
            out,
            epsilon,
            decay_inputs,
            decay,
            beta,
            tokens,
        )
    }

    /// [`Self::conv_l2_decay`] reading `histories[0]` and writing the final
    /// history to `histories[1]`, which may be the same buffer.
    pub fn conv_l2_decay_into(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        histories: [&MetalBuffer; 2],
        out: &MetalBuffer,
        epsilon: f32,
        decay_inputs: [&MetalBuffer; 4],
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let [history, final_history] = histories;
        let n = 10240usize;
        let [a, alpha, dt, beta_raw] = decay_inputs;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid L2 epsilon"));
        }
        let bytes = sequence_bytes(tokens, n as u32)?;
        need(input, bytes)?;
        need(weights, n * 4 * 4)?;
        need(history, n * 3 * 4)?;
        need(final_history, n * 3 * 4)?;
        need(out, bytes)?;
        for v in [a, dt] {
            need(v, GDN_HEADS as usize * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, sequence_bytes(tokens, GDN_HEADS)?)?;
        }
        no_alias(
            out,
            &[input, weights, history, a, alpha, dt, beta_raw, decay, beta],
        )?;
        for history in [history, final_history] {
            no_alias(
                history,
                &[input, weights, a, alpha, dt, beta_raw, decay, beta],
            )?;
        }
        no_alias(out, &[final_history])?;
        no_alias(decay, &[input, a, alpha, dt, beta_raw, beta])?;
        no_alias(beta, &[input, a, alpha, dt, beta_raw])?;
        self.go(
            b,
            19,
            &[
                (input, 0),
                (weights, 0),
                (history, 0),
                (out, 0),
                (a, 0),
                (alpha, 0),
                (dt, 0),
                (beta_raw, 0),
                (decay, 0),
                (beta, 0),
                (final_history, 0),
            ],
            &[tokens],
            &[epsilon],
            n / 128 + 1,
            128,
        );
        Ok(())
    }
}
