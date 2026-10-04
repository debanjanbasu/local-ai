use super::{BonsaiOps, CommandBatch, KvLayout, MetalBuffer, arg, need, no_alias, sequence_bytes};

impl BonsaiOps {
    pub fn prepare_attention(
        &self,
        b: &mut CommandBatch,
        qg: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_norm: &MetalBuffer,
        k_norm: &MetalBuffer,
        q_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
    ) -> crate::Result<()> {
        self.prepare_attention_rows(
            b, qg, k, v, q_norm, k_norm, q_out, gate_out, k_cache, v_cache, position, capacity,
            epsilon, rope_base, 1,
        )
    }

    /// F16-cache form of [`Self::prepare_attention_rows_kv`].
    pub fn prepare_attention_rows(
        &self,
        b: &mut CommandBatch,
        qg: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_norm: &MetalBuffer,
        k_norm: &MetalBuffer,
        q_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        self.prepare_attention_rows_kv(
            KvLayout::F16,
            b,
            qg,
            k,
            v,
            q_norm,
            k_norm,
            q_out,
            gate_out,
            k_cache,
            v_cache,
            position,
            capacity,
            epsilon,
            rope_base,
            tokens,
        )
    }

    /// Normalize/rotate `tokens` query rows and append their K/V rows to
    /// caches stored in `layout`, whose `capacity` is in tokens.
    pub fn prepare_attention_rows_kv(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        qg: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        q_norm: &MetalBuffer,
        k_norm: &MetalBuffer,
        q_out: &MetalBuffer,
        gate_out: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        if capacity == 0
            || capacity > 262_144
            || position
                .checked_add(tokens)
                .is_none_or(|end| end > capacity)
            || epsilon <= 0.0
            || !epsilon.is_finite()
            || rope_base <= 0.0
            || !rope_base.is_finite()
        {
            return Err(arg("invalid attention prep parameters"));
        }
        let codes = layout.codes();
        need(qg, sequence_bytes(tokens, 12288)?)?;
        need(k, sequence_bytes(tokens, 1024)?)?;
        need(v, sequence_bytes(tokens, 1024)?)?;
        need(q_norm, 256 * 4)?;
        need(k_norm, 256 * 4)?;
        need(q_out, sequence_bytes(tokens, 6144)?)?;
        need(gate_out, sequence_bytes(tokens, 6144)?)?;
        need(k_cache, capacity as usize * layout.key.token_bytes())?;
        need(v_cache, capacity as usize * layout.value.token_bytes())?;
        no_alias(
            q_out,
            &[qg, k, v, q_norm, k_norm, gate_out, k_cache, v_cache],
        )?;
        no_alias(
            gate_out,
            &[qg, k, v, q_norm, k_norm, q_out, k_cache, v_cache],
        )?;
        no_alias(k_cache, &[qg, k, v, q_norm, k_norm, v_cache])?;
        no_alias(v_cache, &[qg, k, v, q_norm, k_norm])?;
        self.go(
            b,
            9,
            &[
                (qg, 0),
                (k, 0),
                (v, 0),
                (q_norm, 0),
                (k_norm, 0),
                (q_out, 0),
                (gate_out, 0),
                (k_cache, 0),
                (v_cache, 0),
            ],
            &[position, codes[0], codes[1]],
            &[epsilon, rope_base],
            24 * tokens as usize,
            256,
        );
        Ok(())
    }

    /// F16-cache form of [`Self::prepare_kv_rows_kv`].
    pub fn prepare_kv_rows(
        &self,
        b: &mut CommandBatch,
        k: &MetalBuffer,
        v: &MetalBuffer,
        k_norm: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        self.prepare_kv_rows_kv(
            KvLayout::F16,
            b,
            k,
            v,
            k_norm,
            k_cache,
            v_cache,
            position,
            capacity,
            epsilon,
            rope_base,
            tokens,
        )
    }

    /// Write only the K/V cache rows for `tokens` positions.
    ///
    /// Same key norm and `RoPE` as [`Self::prepare_attention_rows_kv`], without
    /// the query or gate work, for rows whose attention output is never consumed.
    pub fn prepare_kv_rows_kv(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        k: &MetalBuffer,
        v: &MetalBuffer,
        k_norm: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
        tokens: u32,
    ) -> crate::Result<()> {
        if capacity == 0
            || capacity > 262_144
            || position
                .checked_add(tokens)
                .is_none_or(|end| end > capacity)
            || epsilon <= 0.0
            || !epsilon.is_finite()
            || rope_base <= 0.0
            || !rope_base.is_finite()
        {
            return Err(arg("invalid KV prep parameters"));
        }
        let codes = layout.codes();
        need(k, sequence_bytes(tokens, 1024)?)?;
        need(v, sequence_bytes(tokens, 1024)?)?;
        need(k_norm, 256 * 4)?;
        need(k_cache, capacity as usize * layout.key.token_bytes())?;
        need(v_cache, capacity as usize * layout.value.token_bytes())?;
        no_alias(k_cache, &[k, v, k_norm, v_cache])?;
        no_alias(v_cache, &[k, v, k_norm])?;
        self.go(
            b,
            15,
            &[(k, 0), (v, 0), (k_norm, 0), (k_cache, 0), (v_cache, 0)],
            &[position, codes[0], codes[1]],
            &[epsilon, rope_base],
            4 * tokens as usize,
            256,
        );
        Ok(())
    }
}
