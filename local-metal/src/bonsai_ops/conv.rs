use super::{BonsaiOps, CommandBatch, MetalBuffer, arg, need, no_alias, sequence_bytes};

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
        let n = 10240usize;
        let bytes = sequence_bytes(tokens, n as u32)?;
        need(input, bytes)?;
        need(weights, n * 4 * 4)?;
        need(history, n * 3 * 4)?;
        need(out, bytes)?;
        no_alias(out, &[input, weights, history])?;
        no_alias(history, &[input, weights])?;
        self.go(
            b,
            4,
            &[(input, 0), (weights, 0), (history, 0), (out, 0)],
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
}
