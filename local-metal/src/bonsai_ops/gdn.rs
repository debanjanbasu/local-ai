use super::{BonsaiOps, CommandBatch, GDN_HEADS, MetalBuffer, arg, need, no_alias, sequence_bytes};

impl BonsaiOps {
    pub fn decay_beta(
        &self,
        b: &mut CommandBatch,
        a: &MetalBuffer,
        alpha: &MetalBuffer,
        dt: &MetalBuffer,
        beta_raw: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
    ) -> crate::Result<()> {
        self.decay_beta_rows(b, a, alpha, dt, beta_raw, decay, beta, 1)
    }

    pub fn decay_beta_rows(
        &self,
        b: &mut CommandBatch,
        a: &MetalBuffer,
        alpha: &MetalBuffer,
        dt: &MetalBuffer,
        beta_raw: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        let bytes = sequence_bytes(tokens, GDN_HEADS)?;
        for v in [a, dt] {
            need(v, GDN_HEADS as usize * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, bytes)?;
        }
        no_alias(decay, &[a, alpha, dt, beta_raw, beta])?;
        no_alias(beta, &[a, alpha, dt, beta_raw, decay])?;
        self.go(
            b,
            6,
            &[
                (a, 0),
                (alpha, 0),
                (dt, 0),
                (beta_raw, 0),
                (decay, 0),
                (beta, 0),
            ],
            &[tokens],
            &[],
            (GDN_HEADS as usize * tokens as usize).div_ceil(64),
            64,
        );
        Ok(())
    }
    pub fn gdn_step(
        &self,
        b: &mut CommandBatch,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        self.gdn_sequence(b, qkv, decay, beta, state, out, 1)
    }

    /// Causal recurrence with state resident in registers across a token block.
    pub fn gdn_sequence(
        &self,
        b: &mut CommandBatch,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        need(qkv, sequence_bytes(tokens, 10240)?)?;
        need(decay, sequence_bytes(tokens, 48)?)?;
        need(beta, sequence_bytes(tokens, 48)?)?;
        need(state, 48 * 128 * 128 * 4)?;
        need(out, sequence_bytes(tokens, 6144)?)?;
        no_alias(out, &[qkv, decay, beta, state])?;
        no_alias(state, &[qkv, decay, beta])?;
        // Four value rows share Q/K loads during prefill. Single-token decode
        // remains on the original kernel: the wider variant's gain was noisy.
        let (pipeline, rows) = if tokens > 1 { (14, 4) } else { (7, 1) };
        self.go(
            b,
            pipeline,
            &[(qkv, 0), (decay, 0), (beta, 0), (state, 0), (out, 0)],
            &[tokens],
            &[],
            48 * 128 / rows,
            32,
        );
        Ok(())
    }
    pub fn gdn_postprocess(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        z: &MetalBuffer,
        norm: &MetalBuffer,
        out: &MetalBuffer,
        epsilon: f32,
    ) -> crate::Result<()> {
        self.gdn_postprocess_rows(b, input, z, norm, out, epsilon, 1)
    }

    pub fn gdn_postprocess_rows(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        z: &MetalBuffer,
        norm: &MetalBuffer,
        out: &MetalBuffer,
        epsilon: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(arg("invalid post RMS epsilon"));
        }
        let bytes = sequence_bytes(tokens, 6144)?;
        for v in [input, z, out] {
            need(v, bytes)?;
        }
        need(norm, 128 * 4)?;
        no_alias(out, &[input, z, norm])?;
        self.go(
            b,
            8,
            &[(input, 0), (z, 0), (norm, 0), (out, 0)],
            &[],
            &[epsilon],
            48 * tokens as usize,
            128,
        );
        Ok(())
    }
}
