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
    matrix_bytes, need, no_alias, sequence_bytes,
};

const QKV: usize = 10_240;
const MIXED: usize = 6144;

/// Byte offset and required length for row `row` of `width` F32 columns.
const fn row_extent(row: u32, width: usize) -> (usize, usize) {
    let offset = row as usize * width * size_of::<f32>();
    (offset, offset + width * size_of::<f32>())
}

impl BonsaiOps {
    /// [`Self::bf16_matmul`] on the per-row kernel at every row count.
    ///
    /// The token-tiled GEMM that `bf16_matmul` switches to at eight rows is
    /// built for prefill blocks: a 48-row matrix gives it two threadgroups,
    /// and one batched decode step took 125 ms at eight sequences against
    /// 95 ms at seven. Keeping the per-row kernel also keeps each row's
    /// result independent of how many sequences share the step.
    pub fn bf16_matmul_rows(
        &self,
        b: &mut CommandBatch,
        m: Bf16Matrix<'_>,
        x: &MetalBuffer,
        y: &MetalBuffer,
        tokens: u32,
    ) -> crate::Result<()> {
        if m.rows == 0 || m.columns == 0 || !m.offset.is_multiple_of(2) {
            return Err(arg("invalid BF16 matrix"));
        }
        let end = m
            .offset
            .checked_add(matrix_bytes(m.rows, m.columns, 2)?)
            .ok_or_else(|| arg("BF16 offset overflow"))?;
        need(m.buffer, end)?;
        need(x, sequence_bytes(tokens, m.columns)?)?;
        need(y, sequence_bytes(tokens, m.rows)?)?;
        no_alias(y, &[m.buffer, x])?;
        self.go(
            b,
            3,
            &[(m.buffer, m.offset), (x, 0), (y, 0)],
            &[m.rows, m.columns],
            &[],
            m.rows as usize * tokens as usize,
            32,
        );
        Ok(())
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
