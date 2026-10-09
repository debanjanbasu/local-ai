//! Single-row forms of the sequence-specific mixer kernels, reading and
//! writing one row of a multi-row activation block.
//!
//! Batched decode runs every projection over the rows of several independent
//! sequences at once, but the convolution, the recurrence and the K/V append
//! each read and write one sequence's own state. These bind the same
//! pipelines as their whole-block counterparts with a byte offset to `row`,
//! so one sequence's row is processed against that sequence's state without
//! copying it out of the block. Results are bitwise those of the one-row
//! block kernels on a buffer holding only that row.

use super::{
    Bf16Matrix, BonsaiOps, CommandBatch, GDN_HEADS, GdnStateFormat, KvLayout, MetalBuffer, arg,
    need, no_alias,
};

const QKV: usize = 10_240;
const MIXED: usize = 6144;

/// Byte offset and required length for row `row` of `width` F32 columns.
const fn row_extent(row: u32, width: usize) -> (usize, usize) {
    let offset = row as usize * width * size_of::<f32>();
    (offset, offset + width * size_of::<f32>())
}

impl BonsaiOps {
    /// [`Self::bf16_matmul`], whose rows are bitwise the per-row matvec's at
    /// every row count, so each row's result is independent of how many
    /// sequences share the step. (It once avoided a token-tiled GEMM that
    /// `bf16_matmul` switched to at eight rows: one batched decode step took
    /// 125 ms at eight sequences against 95 ms at seven.)
    pub fn bf16_matmul_rows(
        &self,
        b: &mut CommandBatch,
        m: Bf16Matrix<'_>,
        x: &MetalBuffer,
        y: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        self.bf16_matmul(b, m, x, y, tokens)
    }

    /// [`Self::conv_l2_decay_into`] over row `row` of the block, with one
    /// sequence's convolution history updated in place.
    pub fn conv_l2_decay_row(
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
        row: u32,
    ) -> crate::Result<()> {
        let [a, alpha, dt, beta_raw] = decay_inputs;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(arg("invalid L2 epsilon"));
        }
        let (wide, wide_end) = row_extent(row, QKV);
        let (scalar, scalar_end) = row_extent(row, GDN_HEADS as usize);
        need(input, wide_end)?;
        need(out, wide_end)?;
        need(weights, QKV * 4 * 4)?;
        need(history, QKV * 3 * 4)?;
        for v in [a, dt] {
            need(v, GDN_HEADS as usize * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, scalar_end)?;
        }
        no_alias(
            out,
            &[input, weights, history, a, alpha, dt, beta_raw, decay, beta],
        )?;
        no_alias(
            history,
            &[input, weights, a, alpha, dt, beta_raw, decay, beta],
        )?;
        no_alias(decay, &[input, a, alpha, dt, beta_raw, beta])?;
        no_alias(beta, &[input, a, alpha, dt, beta_raw])?;
        self.go(
            b,
            19,
            &[
                (input, wide),
                (weights, 0),
                (history, 0),
                (out, wide),
                (a, 0),
                (alpha, scalar),
                (dt, 0),
                (beta_raw, scalar),
                (decay, scalar),
                (beta, scalar),
                (history, 0),
            ],
            &[1],
            &[epsilon],
            QKV / 128 + 1,
            128,
        );
        Ok(())
    }

    /// One-token [`Self::gdn_sequence_into`] over row `row`, updating one
    /// sequence's state in place.
    pub fn gdn_row(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        qkv: &MetalBuffer,
        decay: &MetalBuffer,
        beta: &MetalBuffer,
        state: &MetalBuffer,
        out: &MetalBuffer,
        row: u32,
    ) -> crate::Result<()> {
        let (wide, wide_end) = row_extent(row, QKV);
        let (scalar, scalar_end) = row_extent(row, GDN_HEADS as usize);
        let (mixed, mixed_end) = row_extent(row, MIXED);
        need(qkv, wide_end)?;
        need(decay, scalar_end)?;
        need(beta, scalar_end)?;
        need(state, format.state_bytes())?;
        need(out, mixed_end)?;
        no_alias(out, &[qkv, decay, beta, state])?;
        no_alias(state, &[qkv, decay, beta])?;
        let pipeline = match format {
            GdnStateFormat::F32 => 7,
            GdnStateFormat::F16 => 21,
            GdnStateFormat::Bf16 => 23,
        };
        self.go(
            b,
            pipeline,
            &[
                (qkv, wide),
                (decay, scalar),
                (beta, scalar),
                (state, 0),
                (out, mixed),
                (state, 0),
            ],
            &[1],
            &[],
            48 * 128,
            32,
        );
        Ok(())
    }

    /// One-token [`Self::prepare_attention_rows_kv`] over row `row`: the
    /// query and gate land in that row of `q_out`/`gate_out`, the K/V row at
    /// `position` of one sequence's caches.
    pub fn prepare_attention_row_kv(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        projections: [&MetalBuffer; 3],
        norms: [&MetalBuffer; 2],
        outputs: [&MetalBuffer; 2],
        caches: [&MetalBuffer; 2],
        position: u32,
        capacity: u32,
        epsilon: f32,
        rope_base: f32,
        row: u32,
    ) -> crate::Result<()> {
        let [qg, k, v] = projections;
        let [q_norm, k_norm] = norms;
        let [q_out, gate_out] = outputs;
        let [k_cache, v_cache] = caches;
        if capacity == 0
            || capacity > 262_144
            || position >= capacity
            || epsilon <= 0.0
            || !epsilon.is_finite()
            || rope_base <= 0.0
            || !rope_base.is_finite()
        {
            return Err(arg("invalid attention prep parameters"));
        }
        let (query_gate, query_gate_end) = row_extent(row, 2 * MIXED);
        let (kv, kv_end) = row_extent(row, 1024);
        let (mixed, mixed_end) = row_extent(row, MIXED);
        need(qg, query_gate_end)?;
        need(k, kv_end)?;
        need(v, kv_end)?;
        need(q_norm, 256 * 4)?;
        need(k_norm, 256 * 4)?;
        need(q_out, mixed_end)?;
        need(gate_out, mixed_end)?;
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
        let codes = layout.codes();
        self.go(
            b,
            9,
            &[
                (qg, query_gate),
                (k, kv),
                (v, kv),
                (q_norm, 0),
                (k_norm, 0),
                (q_out, mixed),
                (gate_out, mixed),
                (k_cache, 0),
                (v_cache, 0),
            ],
            &[position, codes[0], codes[1]],
            &[epsilon, rope_base],
            24,
            256,
        );
        Ok(())
    }
}

/// Byte offset of row `row` of `width` F32 columns, and the length a buffer
/// needs for `tokens` rows from there.
fn rows_extent(row: u32, tokens: u32, width: usize) -> crate::Result<(usize, usize)> {
    let offset = row as usize * width * size_of::<f32>();
    let bytes = super::sequence_bytes(tokens, width as u32)?;
    Ok((offset, offset + bytes))
}

/// Multi-row forms of the sequence-specific mixer kernels, over `tokens`
/// consecutive rows starting at row `row` of a block that stacks several
/// sequences' rows: one sequence's speculative verify rows (or its replay)
/// inside a batched pass. They bind the whole-block pipelines with byte
/// offsets, so a sequence's rows compute bitwise what the block kernels
/// compute on a block holding only those rows.
impl BonsaiOps {
    /// [`Self::conv_l2_decay_into`] over rows `row..row + tokens`.
    pub fn conv_l2_decay_rows_at(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        histories: [&MetalBuffer; 2],
        out: &MetalBuffer,
        epsilon: f32,
        decay_inputs: [&MetalBuffer; 4],
        outputs: [&MetalBuffer; 2],
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        let [history, final_history] = histories;
        let [a, alpha, dt, beta_raw] = decay_inputs;
        let [decay, beta] = outputs;
        if !epsilon.is_finite() || epsilon <= 0.0 || tokens == 0 {
            return Err(arg("invalid L2 epsilon or row count"));
        }
        let (wide, wide_end) = rows_extent(row, tokens, QKV)?;
        let (scalar, scalar_end) = rows_extent(row, tokens, GDN_HEADS as usize)?;
        need(input, wide_end)?;
        need(out, wide_end)?;
        need(weights, QKV * 4 * 4)?;
        need(history, QKV * 3 * 4)?;
        need(final_history, QKV * 3 * 4)?;
        for v in [a, dt] {
            need(v, GDN_HEADS as usize * 4)?;
        }
        for v in [alpha, beta_raw, decay, beta] {
            need(v, scalar_end)?;
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
                (input, wide),
                (weights, 0),
                (history, 0),
                (out, wide),
                (a, 0),
                (alpha, scalar),
                (dt, 0),
                (beta_raw, scalar),
                (decay, scalar),
                (beta, scalar),
                (final_history, 0),
            ],
            &[tokens],
            &[epsilon],
            QKV / 128 + 1,
            128,
        );
        Ok(())
    }

    /// [`Self::conv_sequence`] over rows `row..row + tokens`, updating one
    /// sequence's history in place: the replay of a verify block's kept rows.
    pub fn conv_rows_at(
        &self,
        b: &mut CommandBatch,
        input: &MetalBuffer,
        weights: &MetalBuffer,
        history: &MetalBuffer,
        out: &MetalBuffer,
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        if tokens == 0 {
            return Err(arg("invalid convolution row count"));
        }
        let (wide, wide_end) = rows_extent(row, tokens, QKV)?;
        need(input, wide_end)?;
        need(weights, QKV * 4 * 4)?;
        need(history, QKV * 3 * 4)?;
        need(out, wide_end)?;
        no_alias(out, &[input, weights, history])?;
        no_alias(history, &[input, weights])?;
        self.go(
            b,
            4,
            &[
                (input, wide),
                (weights, 0),
                (history, 0),
                (out, wide),
                (history, 0),
            ],
            &[tokens],
            &[],
            QKV.div_ceil(256),
            256,
        );
        Ok(())
    }

    /// [`Self::l2_normalize_qk_rows`] over rows `row..row + tokens`.
    pub fn l2_normalize_qk_rows_at(
        &self,
        b: &mut CommandBatch,
        qkv: &MetalBuffer,
        epsilon: f32,
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        if !epsilon.is_finite() || epsilon <= 0.0 || tokens == 0 {
            return Err(arg("invalid L2 epsilon or row count"));
        }
        let (wide, wide_end) = rows_extent(row, tokens, QKV)?;
        need(qkv, wide_end - MIXED * 4)?;
        self.go(
            b,
            5,
            &[(qkv, wide)],
            &[],
            &[epsilon],
            32 * tokens as usize,
            128,
        );
        Ok(())
    }

    /// [`Self::gdn_sequence_into`] over rows `row..row + tokens`.
    pub fn gdn_rows_at(
        &self,
        b: &mut CommandBatch,
        format: GdnStateFormat,
        inputs: [&MetalBuffer; 3],
        states: [&MetalBuffer; 2],
        out: &MetalBuffer,
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        let [qkv, decay, beta] = inputs;
        let [state, final_state] = states;
        if tokens == 0 {
            return Err(arg("invalid GDN row count"));
        }
        let (wide, wide_end) = rows_extent(row, tokens, QKV)?;
        let (scalar, scalar_end) = rows_extent(row, tokens, GDN_HEADS as usize)?;
        let (mixed, mixed_end) = rows_extent(row, tokens, MIXED)?;
        need(qkv, wide_end)?;
        need(decay, scalar_end)?;
        need(beta, scalar_end)?;
        need(state, format.state_bytes())?;
        need(final_state, format.state_bytes())?;
        need(out, mixed_end)?;
        no_alias(out, &[qkv, decay, beta, state, final_state])?;
        no_alias(state, &[qkv, decay, beta])?;
        no_alias(final_state, &[qkv, decay, beta])?;
        // `GdnStateFormat::pipelines`: the single-row and four-row kernels.
        let (single, four) = match format {
            GdnStateFormat::F32 => (7, 14),
            GdnStateFormat::F16 => (21, 22),
            GdnStateFormat::Bf16 => (23, 24),
        };
        let (pipeline, rows) = if tokens > 1 { (four, 4) } else { (single, 1) };
        self.go(
            b,
            pipeline,
            &[
                (qkv, wide),
                (decay, scalar),
                (beta, scalar),
                (state, 0),
                (out, mixed),
                (final_state, 0),
            ],
            &[tokens],
            &[],
            48 * 128 / rows,
            32,
        );
        Ok(())
    }

    /// [`Self::prepare_attention_rows_kv`] over rows `row..row + tokens`,
    /// appending their K/V rows at `position` of one sequence's caches.
    pub fn prepare_attention_rows_kv_at(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        projections: [&MetalBuffer; 3],
        norms: [&MetalBuffer; 2],
        outputs: [&MetalBuffer; 2],
        caches: [&MetalBuffer; 2],
        extent: [u32; 2],
        constants: [f32; 2],
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        let [qg, k, v] = projections;
        let [q_norm, k_norm] = norms;
        let [q_out, gate_out] = outputs;
        let [k_cache, v_cache] = caches;
        let [position, capacity] = extent;
        let [epsilon, rope_base] = constants;
        if tokens == 0
            || capacity == 0
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
        let (query_gate, query_gate_end) = rows_extent(row, tokens, 2 * MIXED)?;
        let (kv, kv_end) = rows_extent(row, tokens, 1024)?;
        let (mixed, mixed_end) = rows_extent(row, tokens, MIXED)?;
        need(qg, query_gate_end)?;
        need(k, kv_end)?;
        need(v, kv_end)?;
        need(q_norm, 256 * 4)?;
        need(k_norm, 256 * 4)?;
        need(q_out, mixed_end)?;
        need(gate_out, mixed_end)?;
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
        let codes = layout.codes();
        self.go(
            b,
            9,
            &[
                (qg, query_gate),
                (k, kv),
                (v, kv),
                (q_norm, 0),
                (k_norm, 0),
                (q_out, mixed),
                (gate_out, mixed),
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

    /// [`Self::prepare_kv_rows_kv`] over rows `row..row + tokens`: only the
    /// K/V rows, appended at `position` of one sequence's caches, for rows
    /// whose attention output nobody reads (an MTP head catching up).
    pub fn prepare_kv_rows_kv_at(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        projections: [&MetalBuffer; 2],
        k_norm: &MetalBuffer,
        caches: [&MetalBuffer; 2],
        extent: [u32; 2],
        constants: [f32; 2],
        row: u32,
        tokens: u32,
    ) -> crate::Result<()> {
        let [k, v] = projections;
        let [k_cache, v_cache] = caches;
        let [position, capacity] = extent;
        let [epsilon, rope_base] = constants;
        if tokens == 0
            || capacity == 0
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
        let (kv, kv_end) = rows_extent(row, tokens, 1024)?;
        need(k, kv_end)?;
        need(v, kv_end)?;
        need(k_norm, 256 * 4)?;
        need(k_cache, capacity as usize * layout.key.token_bytes())?;
        need(v_cache, capacity as usize * layout.value.token_bytes())?;
        no_alias(k_cache, &[k, v, k_norm, v_cache])?;
        no_alias(v_cache, &[k, v, k_norm])?;
        let codes = layout.codes();
        self.go(
            b,
            15,
            &[(k, kv), (v, kv), (k_norm, 0), (k_cache, 0), (v_cache, 0)],
            &[position, codes[0], codes[1]],
            &[epsilon, rope_base],
            4 * tokens as usize,
            256,
        );
        Ok(())
    }

    /// [`Self::attention_block_kv`] for query rows `row..row + tokens` of a
    /// stacked block: row `row + r` attends through cache position
    /// `position + r`, inclusive, choosing the kernel exactly as the block
    /// form does for a block of `tokens` rows.
    pub fn attention_block_kv_at(
        &self,
        layout: KvLayout,
        b: &mut CommandBatch,
        q: &MetalBuffer,
        caches: [&MetalBuffer; 2],
        gate: &MetalBuffer,
        out: &MetalBuffer,
        position: u32,
        tokens: u32,
        workspace: &super::AttentionWorkspace,
        row: u32,
    ) -> crate::Result<()> {
        let [k_cache, v_cache] = caches;
        let end = position
            .checked_add(tokens)
            .ok_or_else(|| arg("attention block position overflow"))?;
        if !(1..=super::MAX_PREFILL_TOKENS).contains(&tokens)
            || end > workspace.max_context
            || end == 0
        {
            return Err(arg("invalid attention block extent"));
        }
        if tokens == 1 || (tokens <= super::ROW_BLOCK_TOKENS && end > super::ROW_BLOCK_MIN_PREFIX) {
            for index in 0..tokens {
                self.attention_row_kv(
                    layout,
                    b,
                    q,
                    k_cache,
                    v_cache,
                    Some(gate),
                    out,
                    position + index + 1,
                    workspace,
                    row + index,
                )?;
            }
            return Ok(());
        }
        let (mixed, mixed_end) = rows_extent(row, tokens, MIXED)?;
        need(q, mixed_end)?;
        need(k_cache, end as usize * layout.key.token_bytes())?;
        need(v_cache, end as usize * layout.value.token_bytes())?;
        need(out, mixed_end)?;
        need(gate, mixed_end)?;
        no_alias(out, &[gate, q, k_cache, v_cache, &workspace.partials])?;
        let bufs = [
            (q, mixed),
            (k_cache, 0),
            (v_cache, 0),
            (gate, mixed),
            (out, mixed),
        ];
        let groups = 24 * tokens.div_ceil(8) as usize;
        let rotated = !layout.is_f16();
        let gated = u32::from(!rotated);
        let codes = layout.codes();
        if layout.is_f16() {
            self.go(
                b,
                13,
                &bufs,
                &[position, tokens, gated, codes[0], codes[1]],
                &[],
                groups,
                self.attention_kernel.threads(),
            );
        } else if let Some(split) = self.quantized_tensor(layout) {
            self.go(
                b,
                split + 1,
                &bufs,
                &[position, tokens, gated],
                &[],
                groups,
                self.attention_kernel.threads(),
            );
        } else {
            self.go(
                b,
                17,
                &bufs,
                &[position, tokens, gated, codes[0], codes[1]],
                &[],
                groups,
                256,
            );
        }
        if rotated {
            // Rotate back from the quantized cache's basis, then gate, as
            // `attention_block_kv` does.
            self.go(
                b,
                20,
                &[(out, mixed), (gate, mixed)],
                &[1],
                &[],
                24 * tokens as usize,
                256,
            );
        }
        Ok(())
    }
}
