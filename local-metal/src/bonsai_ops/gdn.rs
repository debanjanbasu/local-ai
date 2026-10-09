use super::{
    BonsaiOps, CommandBatch, GDN_DIM, GDN_HEADS, MetalBuffer, arg, need, no_alias, sequence_bytes,
};
use crate::bonsai::SignedHadamard;

/// Storage of a 48-head x 128 x 128 recurrent state. The recurrence always
/// computes in F32 registers; the format only rounds what a block leaves in
/// memory (once per dispatch, round to nearest even).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GdnStateFormat {
    #[default]
    F32,
    F16,
    Bf16,
}

impl GdnStateFormat {
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// Bytes of one layer's state.
    #[must_use]
    pub const fn state_bytes(self) -> usize {
        GDN_HEADS as usize * 128 * 128 * self.element_bytes()
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    /// Indexes of the single-row and four-row pipelines in `BonsaiOps::p`.
    const fn pipelines(self) -> (usize, usize) {
        match self {
            Self::F32 => (7, 14),
            Self::F16 => (21, 22),
            Self::Bf16 => (23, 24),
        }
    }
}

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
        self.gdn_sequence_into(
            b,
            GdnStateFormat::F32,
            qkv,
            decay,
            beta,
            state,
            state,
            out,
            tokens,
        )
    }

    /// [`Self::gdn_sequence`] reading `state` and writing the block's final
    /// state to `final_state`, which may be `state` itself (in place) or a
    /// separate buffer that leaves `state` intact, both in `format`.
    pub fn gdn_sequence_into(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        final_state: &MetalBuffer,
        out: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        need(qkv, sequence_bytes(tokens, 10240)?)?;
        need(decay, sequence_bytes(tokens, 48)?)?;
        need(beta, sequence_bytes(tokens, 48)?)?;
        need(state, format.state_bytes())?;
        need(final_state, format.state_bytes())?;
        need(out, sequence_bytes(tokens, 6144)?)?;
        no_alias(out, &[qkv, decay, beta, state, final_state])?;
        no_alias(state, &[qkv, decay, beta])?;
        no_alias(final_state, &[qkv, decay, beta])?;
        // Four value rows share Q/K loads during prefill. Single-token decode
        // keeps one row per SIMD group (the wider variant's gain was noisy),
        // four SIMD groups per threadgroup.
        let (single, wide) = format.pipelines();
        let (pipeline, threads) = if tokens > 1 {
            (wide, 32)
        } else {
            (single, 128)
        };
        self.go(
            b,
            pipeline,
            &[
                (qkv, 0),
                (decay, 0),
                (beta, 0),
                (state, 0),
                (out, 0),
                (final_state, 0),
            ],
            &[tokens],
            &[],
            48 * 128 / 4,
            threads,
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

    /// [`Self::gdn_postprocess_rows`] followed by the forward `rotation`
    /// (6144 columns) of its output, in one dispatch: `out` receives bitwise
    /// what the separate post-processing then rotation would leave in the
    /// rotated buffer. The unrotated rows are not stored.
    pub fn gdn_postprocess_transform(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        z: &MetalBuffer,
        norm: &MetalBuffer,
        rotation: &SignedHadamard,
        out: &MetalBuffer,
        epsilon: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        if epsilon <= 0.0 || !epsilon.is_finite() {
            return Err(arg("invalid post RMS epsilon"));
        }
        if rotation.columns != GDN_HEADS * GDN_DIM {
            return Err(arg("GDN output rotation must be 6144 wide"));
        }
        let bytes = sequence_bytes(tokens, GDN_HEADS * GDN_DIM)?;
        for v in [input, z, out] {
            need(v, bytes)?;
        }
        need(norm, GDN_DIM as usize * 4)?;
        no_alias(out, &[input, z, norm, &rotation.signs])?;
        self.go(
            b,
            25,
            &[
                (input, 0),
                (z, 0),
                (norm, 0),
                (&rotation.signs, 0),
                (out, 0),
            ],
            &[],
            &[epsilon],
            6 * tokens as usize,
            128,
        );
        Ok(())
    }
}
