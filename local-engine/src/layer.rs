use local_metal::batch::{BufferCopyRequest, CommandBatch};
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kernels::Kernels;

use crate::gguf::GGUFType;
use crate::kv_cache::{QuantizedKvCache, QuantizedUnifiedKvPool};

/// A weight matrix held in device memory as FP16 or in its original GGUF
/// quantized block format (`Q4_0` / `TQ2_0`).
///
/// The matvec/matmul dispatches to the kernel matching the stored format.
/// Keeping the dominant formats quantized at rest cuts resident weight memory
/// ~3.5–8× and the per-token weight bandwidth by the same factor.
///
/// `rows` is the output dimension, `cols` the input (reduction) dimension —
/// the engine's standard `[out, in]` row-major layout.
pub struct QuantWeight {
    buf: MetalBuffer,
    ty: GGUFType,
    rows: u32,
    cols: u32,
}

impl QuantWeight {
    /// Wrap an FP16 weight buffer.
    #[must_use]
    pub const fn f16(buf: MetalBuffer, rows: u32, cols: u32) -> Self {
        Self {
            buf,
            ty: GGUFType::F16,
            rows,
            cols,
        }
    }

    /// Wrap a quantized payload uploaded verbatim. Only `Q4_0` and `TQ2_0`
    /// have device kernels.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported formats or non-block-aligned `cols`.
    pub fn quantized(buf: MetalBuffer, ty: GGUFType, rows: u32, cols: u32) -> crate::Result<Self> {
        let block = match ty {
            GGUFType::Q4_0 => 32,
            GGUFType::TQ2_0 => 256,
            _ => {
                return Err(crate::Error::InvalidFormat(format!(
                    "QuantWeight: no device kernel for {ty:?}"
                )));
            }
        };
        if !(cols as usize).is_multiple_of(block) {
            return Err(crate::Error::InvalidFormat(format!(
                "QuantWeight: cols {cols} not a multiple of {ty:?} block {block}"
            )));
        }
        Ok(Self {
            buf,
            ty,
            rows,
            cols,
        })
    }

    #[must_use]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> u32 {
        self.cols
    }

    #[must_use]
    pub const fn ty(&self) -> GGUFType {
        self.ty
    }

    /// Bytes resident in device memory.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.buf.length()
    }

    /// Borrow the underlying FP16 Metal buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the weight is still in a quantized-at-rest format.
    pub(crate) fn as_f16_buffer(&self, name: &str) -> crate::Result<&MetalBuffer> {
        if self.ty == GGUFType::F16 {
            Ok(&self.buf)
        } else {
            Err(crate::Error::InvalidFormat(format!(
                "{name}: expected FP16 tensor, got {:?}",
                self.ty
            )))
        }
    }

    /// Consume into the raw FP16 buffer; errors if the weight is quantized.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored format is not FP16.
    pub fn into_f16(self, name: &str) -> crate::Result<MetalBuffer> {
        if self.ty == GGUFType::F16 {
            Ok(self.buf)
        } else {
            Err(crate::Error::InvalidFormat(format!(
                "{name}: expected FP16 tensor, got {:?}",
                self.ty
            )))
        }
    }

    /// `out[rows] (f16) = self × x[cols]`, dispatched by format.
    ///
    /// # Errors
    ///
    /// Returns an error on kernel encoding failure.
    pub fn matvec(
        &self,
        ctx: &MetalContext,
        kernels: &Kernels,
        x: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        match self.ty {
            GGUFType::Q4_0 => kernels.matvec_q4_0(ctx, &self.buf, x, out, self.rows, self.cols),
            GGUFType::TQ2_0 => kernels.matvec_tq2_0(ctx, &self.buf, x, out, self.rows, self.cols),
            _ => kernels.matvec(ctx, &self.buf, x, out, self.rows, self.cols),
        }
        .map_err(crate::Error::Metal)
    }

    /// `out[rows] (f16) = self × x[cols]`, encoded into an existing command batch.
    ///
    /// # Errors
    ///
    /// Returns an error on kernel encoding failure.
    pub fn matvec_into(
        &self,
        batch: &mut CommandBatch,
        kernels: &Kernels,
        x: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        match self.ty {
            GGUFType::Q4_0 => {
                kernels.matvec_q4_0_into(batch, &self.buf, x, out, self.rows, self.cols)
            }
            GGUFType::TQ2_0 => {
                kernels.matvec_tq2_0_into(batch, &self.buf, x, out, self.rows, self.cols)
            }
            _ => kernels.matvec_offset_into(batch, &self.buf, 0, x, out, self.rows, self.cols),
        }
        .map_err(crate::Error::Metal)
    }

    /// `out[rows] (f32) = self × x[cols]` — logits projection.
    ///
    /// # Errors
    ///
    /// Returns an error on kernel encoding failure.
    pub fn matvec_f32out(
        &self,
        ctx: &MetalContext,
        kernels: &Kernels,
        x: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        match self.ty {
            GGUFType::Q4_0 => {
                kernels.matvec_q4_0_f32out(ctx, &self.buf, x, out, self.rows, self.cols)
            }
            GGUFType::TQ2_0 => {
                kernels.matvec_tq2_0_f32out(ctx, &self.buf, x, out, self.rows, self.cols)
            }
            _ => kernels.matvec_f32out_offset(ctx, &self.buf, 0, x, out, self.rows, self.cols),
        }
        .map_err(crate::Error::Metal)
    }

    /// `out[rows] (f32) = self × x[cols]`, encoded into an existing command batch.
    ///
    /// # Errors
    ///
    /// Returns an error on kernel encoding failure.
    pub fn matvec_f32out_into(
        &self,
        batch: &mut CommandBatch,
        kernels: &Kernels,
        x: &MetalBuffer,
        out: &MetalBuffer,
    ) -> crate::Result<()> {
        match self.ty {
            GGUFType::Q4_0 => {
                kernels.matvec_q4_0_f32out_into(batch, &self.buf, x, out, self.rows, self.cols)
            }
            GGUFType::TQ2_0 => {
                kernels.matvec_tq2_0_f32out_into(batch, &self.buf, x, out, self.rows, self.cols)
            }
            _ => {
                kernels.matvec_f32out_offset_into(batch, &self.buf, 0, x, out, self.rows, self.cols)
            }
        }
        .map_err(crate::Error::Metal)
    }

    /// Batched `[m, cols] × selfᵀ → [m, rows]` into a command batch.
    ///
    /// # Errors
    ///
    /// Returns an error on kernel encoding failure.
    pub fn matmul_nt_into(
        &self,
        batch: &mut CommandBatch,
        kernels: &Kernels,
        input: &MetalBuffer,
        output: &MetalBuffer,
        m: u32,
    ) -> crate::Result<()> {
        // When the MMA (simdgroup_matrix) GEMM is available it beats both the
        // scalar tiled kernel and the multivec GEMV-style kernel at every batch
        // size (see `matmul_prefill_timing`), so `matmul_nt_*_into` routes all
        // M through it. Only when MMA is disabled (`LOCAL_AI_MMA=0`) do small
        // batches fall back to the multi-vector matvec — the quantized weight
        // streams once and is reused across all m input vectors, avoiding the
        // scalar tiled matmul's 32-row-tile waste.
        let small = !Kernels::mma_enabled() && m <= Kernels::MULTIVEC_MAX_M;
        match self.ty {
            GGUFType::Q4_0 if small => kernels
                .multivec_nt_q4_0_into(batch, input, &self.buf, output, m, self.cols, self.rows),
            GGUFType::TQ2_0 if small => kernels
                .multivec_nt_tq2_0_into(batch, input, &self.buf, output, m, self.cols, self.rows),
            GGUFType::Q4_0 => kernels
                .matmul_nt_q4_0_into(batch, input, &self.buf, output, m, self.cols, self.rows),
            GGUFType::TQ2_0 => kernels
                .matmul_nt_tq2_0_into(batch, input, &self.buf, output, m, self.cols, self.rows),
            _ => kernels.matmul_nt_into(batch, input, &self.buf, output, m, self.cols, self.rows),
        }
        .map_err(crate::Error::Metal)
    }

    /// Dequantize whole rows to FP16 on the CPU (token-embedding gather).
    /// Reads the shared-memory device buffer directly — no separate CPU copy.
    /// Out-of-range rows clamp to the last row.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload cannot be decoded.
    pub fn dequant_row(&self, row: usize) -> crate::Result<Vec<half::f16>> {
        let mut out = Vec::new();
        self.dequant_row_into(row, &mut out)?;
        Ok(out)
    }

    /// [`Self::dequant_row`] writing into a caller-owned `out` (cleared first) so
    /// a single scratch buffer can be reused across per-token embedding gathers
    /// instead of allocating a fresh `Vec` for every token. Produces identical
    /// contents to [`Self::dequant_row`].
    ///
    /// # Errors
    ///
    /// Returns an error if the payload cannot be decoded.
    pub fn dequant_row_into(&self, row: usize, out: &mut Vec<half::f16>) -> crate::Result<()> {
        let cols = self.cols as usize;
        let row = row.min((self.rows as usize).saturating_sub(1));
        let row_bytes = self.ty.tensor_bytes(cols);
        let raw = &self.buf.as_slice::<u8>()[row * row_bytes..(row + 1) * row_bytes];
        if self.ty == GGUFType::F16 {
            out.clear();
            out.extend_from_slice(bytemuck::cast_slice(raw));
            Ok(())
        } else {
            crate::qat_recover::dequant_into(raw, self.ty, cols, out)
        }
    }
}

/// Per-layer attention and feed-forward dimensions plus rotary parameters.
///
/// Gemma 4 E2B varies these by layer: sliding-window layers use `head_dim` 256
/// with rotary base `10_000`, full-attention layers use 512 with base
/// `1_000_000`, and later layers use a double-wide feed-forward. KV-shared
/// layers compute only Q and reuse a source layer's K/V cache.
#[derive(Clone)]
pub struct LayerParams {
    pub layer_idx: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub per_layer_dim: usize,
    pub rope_theta: f32,
    /// HF `rope_type: "proportional"`: fraction of the head that rotates
    /// (1.0 = standard full `NEOX` `RoPE`; Gemma 4 full-attention layers use 0.25).
    pub partial_rotary_factor: f32,
    /// Sliding-attention window in positions (`0` = full attention).
    pub sliding_window: usize,
    pub rms_norm_eps: f32,
    pub is_kv_shared: bool,
}

pub struct LayerWeights {
    pub attn_norm: MetalBuffer,
    pub attn_q: QuantWeight,
    pub attn_k: Option<QuantWeight>,
    pub attn_v: Option<QuantWeight>,
    pub q_norm: Option<MetalBuffer>,
    pub k_norm: Option<MetalBuffer>,
    pub attn_output: QuantWeight,
    pub attn_post_norm: MetalBuffer,
    pub ffn_norm: MetalBuffer,
    pub ffn_gate: QuantWeight,
    pub ffn_up: QuantWeight,
    pub ffn_down: QuantWeight,
    pub ffn_post_norm: MetalBuffer,
    pub per_layer_inp_gate: Option<QuantWeight>,
    pub per_layer_proj: Option<QuantWeight>,
    pub per_layer_post_norm: Option<MetalBuffer>,
    pub layer_output_scale: Option<MetalBuffer>,
}

pub struct TransformerLayer {
    pub params: LayerParams,
    weights: LayerWeights,
}

impl TransformerLayer {
    #[must_use]
    pub const fn new(params: LayerParams, weights: LayerWeights) -> Self {
        Self { params, weights }
    }

    /// Single-token decode step for one Gemma 4 transformer layer.
    ///
    /// - `input` / `output`: hidden state (`hidden_size` × f16) in and out.
    /// - `kv_cache`: this layer's own cache (non-shared) or the source layer's
    ///   cache (KV-shared layers; read-only — already written by the source
    ///   layer earlier in the same step).
    /// - `per_layer_input`: the full `[per_layer_dim * n_layer]` PLE buffer; this
    ///   layer's slice is gathered out internally on the GPU.
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel is unavailable, encoding fails, or a command
    /// buffer commit fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn forward_decode(
        &self,
        kernels: &Kernels,
        batch: &mut CommandBatch,
        input: &MetalBuffer,
        output: &MetalBuffer,
        position: usize,
        scratch: &mut ScratchBuffers,
        kv_cache: &mut QuantizedKvCache,
        per_layer_input: Option<&MetalBuffer>,
        n_layer_total: usize,
    ) -> crate::Result<()> {
        let p = &self.params;
        let w = &self.weights;
        let h = p.hidden_size as u32;
        let hd = p.head_dim as u32;
        let n_head = p.num_attention_heads as u32;
        let n_kv = p.num_key_value_heads as u32;
        let eps = p.rms_norm_eps;
        let pos = position as u32;
        let seq_len = pos + 1;

        // 1. input_layernorm → Q projection → q_norm → RoPE
        kernels.rms_norm_into(batch, input, &w.attn_norm, &scratch.normed, h, 1, eps)?;
        w.attn_q
            .matvec_into(batch, kernels, &scratch.normed, &scratch.q)?;
        if let Some(qn) = &w.q_norm {
            kernels.qk_norm_gpu_into(batch, &scratch.q, qn, hd, n_head, eps)?;
        }
        kernels.rope_into(
            batch,
            &scratch.q,
            &scratch.q,
            hd,
            pos,
            p.rope_theta,
            p.partial_rotary_factor,
            n_head,
        )?;

        // 2. self-attention (write own K/V, or reuse the source layer's cache)
        if !p.is_kv_shared
            && let (Some(wk), Some(wv)) = (&w.attn_k, &w.attn_v)
        {
            wk.matvec_into(batch, kernels, &scratch.normed, &scratch.k)?;
            wv.matvec_into(batch, kernels, &scratch.normed, &scratch.v)?;
            if let Some(kn) = &w.k_norm {
                kernels.qk_norm_gpu_into(batch, &scratch.k, kn, hd, n_kv, eps)?;
            }
            // V is RMS-normalized with no weight (ones) per Gemma 4.
            kernels.rms_norm_into(batch, &scratch.v, &scratch.ones, &scratch.v, hd, n_kv, eps)?;
            kernels.rope_into(
                batch,
                &scratch.k,
                &scratch.k,
                hd,
                pos,
                p.rope_theta,
                p.partial_rotary_factor,
                n_kv,
            )?;
            // Quantize K/V into the TurboQuant cache (codes + per-head norm)
            // on the GPU — no CPU readback + encode round-trip, so it stays in
            // the same command batch.
            kv_cache.write_kv_gpu_into(batch, kernels, &scratch.k, &scratch.v, position, 1)?;
        }
        // Attention over the TurboQuant cache via the fused path: it reads the
        // packed K/V codes directly (rotate query → attend over codes →
        // inverse-rotate output), avoiding any FP16 expansion — the decode read
        // drops from 16-bit to `bits`-bit per coordinate, the bandwidth-bound
        // cost of re-reading the whole KV cache each token.
        kv_cache.fused_attention_into(
            batch,
            kernels,
            &scratch.q,
            &scratch.rq,
            &scratch.vacc,
            &scratch.attn_out,
            seq_len,
            pos,
            n_head,
            p.sliding_window as u32,
        )?;

        // 3. output projection → post_attention_norm → residual #1
        w.attn_output
            .matvec_into(batch, kernels, &scratch.attn_out, &scratch.ffn_out)?;
        // Fused post_attention_norm + residual #1 (mirrors forward_batch):
        // attn_resid = rms_norm(ffn_out) * attn_post_norm + input
        kernels.rms_norm_add_into(
            batch,
            &scratch.ffn_out,
            &w.attn_post_norm,
            input,
            &scratch.attn_resid,
            h,
            1,
            eps,
        )?;

        // 4. GeGLU FFN → post_feedforward_norm → residual #2
        let inter = p.intermediate_size as u32;
        kernels.rms_norm_into(
            batch,
            &scratch.attn_resid,
            &w.ffn_norm,
            &scratch.normed,
            h,
            1,
            eps,
        )?;
        w.ffn_gate
            .matvec_into(batch, kernels, &scratch.normed, &scratch.gate)?;
        w.ffn_up
            .matvec_into(batch, kernels, &scratch.normed, &scratch.up)?;
        kernels.gelu_mul_into(batch, &scratch.gate, &scratch.up, &scratch.gate, inter)?;
        w.ffn_down
            .matvec_into(batch, kernels, &scratch.gate, &scratch.ffn_out)?;
        // Fused post_feedforward_norm + residual #2 (mirrors forward_batch):
        // output = rms_norm(ffn_out) * ffn_post_norm + attn_resid
        kernels.rms_norm_add_into(
            batch,
            &scratch.ffn_out,
            &w.ffn_post_norm,
            &scratch.attn_resid,
            output,
            h,
            1,
            eps,
        )?;

        // 5. per-layer input (PLE) injection → residual #3
        if let (Some(gate_w), Some(proj_w), Some(post_w), Some(ple_all)) = (
            &w.per_layer_inp_gate,
            &w.per_layer_proj,
            &w.per_layer_post_norm,
            per_layer_input,
        ) {
            let pld = p.per_layer_dim as u32;
            // Copy this layer's PLE slice in the command buffer. A CPU copy into shared
            // scratch here would race with GPU reads from earlier encoded
            // layers when decode is batched into one command buffer; Metal
            // hazard tracking orders GPU↔GPU use, not CPU writes to shared
            // scratch after encoding.
            let byte_off = p.layer_idx * p.per_layer_dim * std::mem::size_of::<half::f16>();
            let bytes = p.per_layer_dim * std::mem::size_of::<half::f16>();
            let _ = n_layer_total;
            batch.blit_buffer_copies([BufferCopyRequest {
                source: ple_all,
                source_offset: byte_off,
                destination: &scratch.ple,
                destination_offset: 0,
                size: bytes,
            }])?;
            gate_w.matvec_into(batch, kernels, output, &scratch.ple_h)?;
            kernels.gelu_mul_into(batch, &scratch.ple_h, &scratch.ple, &scratch.ple_h, pld)?;
            proj_w.matvec_into(batch, kernels, &scratch.ple_h, &scratch.ffn_out)?;
            kernels.rms_norm_into(batch, &scratch.ffn_out, post_w, &scratch.normed, h, 1, eps)?;
            kernels.residual_add_into(batch, output, &scratch.normed, output, h)?;
        }

        // 6. per-layer output scale (layer_scalar)
        if let Some(scale) = &w.layer_output_scale {
            let s: &[half::f16] = scale.as_slice();
            if let Some(&v) = s.first() {
                kernels.scale_in_place_gpu_into(batch, output, half::f16::to_f32(v), h)?;
            }
        }
        Ok(())
    }

    /// Batched decode step: process `n` consecutive tokens (rows) at positions
    /// `start_pos..start_pos+n` through this layer in (at most) two GPU command
    /// buffers — the same math as [`Self::forward_decode`] with every matvec
    /// replaced by a tiled batch matmul, so the weights stream once for all
    /// `n` tokens and per-kernel CPU↔GPU sync collapses to one or two waits.
    ///
    /// The only CPU boundary is the `TurboQuant` K/V encode between the
    /// projection phase and the attention/FFN phase (KV-shared layers skip it
    /// and run in a single command buffer).
    ///
    /// `input` / `output` are `[n × hidden]`; `ple_all` is the batched
    /// per-layer-input tensor `[n × n_layer × per_layer_dim]`.
    ///
    /// # Errors
    ///
    /// Returns an error if a kernel is unavailable or encoding fails.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn forward_batch(
        &self,
        kernels: &Kernels,
        batch: &mut CommandBatch,
        n_rows: usize,
        start_pos: usize,
        input: &MetalBuffer,
        output: &MetalBuffer,
        bs: &BatchScratch,
        ones: &MetalBuffer,
        kv_cache: &mut QuantizedKvCache,
        ple_all: Option<(&MetalBuffer, usize)>,
    ) -> crate::Result<()> {
        let p = &self.params;
        let w = &self.weights;
        let rows = n_rows as u32;
        let h = p.hidden_size as u32;
        let hd = p.head_dim as u32;
        let n_head = p.num_attention_heads as u32;
        let n_kv = p.num_key_value_heads as u32;
        let eps = p.rms_norm_eps;
        let kv_len = (start_pos + n_rows) as u32;

        // Phase 1: input norm → Q (+K/V) projections, norms, RoPE. The command
        // buffer is owned by the caller, which runs many layers in one buffer
        // (chunk-committing periodically) so there is no per-layer GPU drain.
        kernels.rms_norm_into(batch, input, &w.attn_norm, &bs.normed, h, rows, eps)?;
        w.attn_q
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.q, rows)?;
        if let Some(qn) = &w.q_norm {
            kernels.qk_norm_gpu_into(batch, &bs.q, qn, hd, rows * n_head, eps)?;
        }
        kernels.rope_batch_into(
            batch,
            &bs.q,
            start_pos as u32,
            p.rope_theta,
            hd,
            n_head,
            rows,
            p.partial_rotary_factor,
        )?;

        let own_kv = if !p.is_kv_shared
            && let (Some(wk), Some(wv)) = (&w.attn_k, &w.attn_v)
        {
            wk.matmul_nt_into(batch, kernels, &bs.normed, &bs.k, rows)?;
            wv.matmul_nt_into(batch, kernels, &bs.normed, &bs.v, rows)?;
            if let Some(kn) = &w.k_norm {
                kernels.qk_norm_gpu_into(batch, &bs.k, kn, hd, rows * n_kv, eps)?;
            }
            // V is RMS-normalized with no weight (ones) per Gemma 4.
            kernels.rms_norm_into(batch, &bs.v, ones, &bs.v, hd, rows * n_kv, eps)?;
            kernels.rope_batch_into(
                batch,
                &bs.k,
                start_pos as u32,
                p.rope_theta,
                hd,
                n_kv,
                rows,
                p.partial_rotary_factor,
            )?;
            true
        } else {
            false
        };

        // TurboQuant-encode the n new K/V rows into the cache. The GPU encoder
        // keeps the whole batch on the GPU (the freshly written codes are read
        // back by the fused attention kernel further down the same command
        // buffer, which Metal orders correctly), avoiding a per-layer CPU
        // readback + `commit_and_wait` round-trip — the dominant cost of
        // batched decode.
        if own_kv {
            kv_cache.write_kv_gpu_into(batch, kernels, &bs.k, &bs.v, start_pos, n_rows)?;
        }

        // Phase 2: attention over the (just-updated) cache, then FFN + PLE.
        // Fused TurboQuant path: each query row attends the packed K/V codes
        // directly (rotate query → per-row causal attend over codes →
        // inverse-rotate), so the cache is never re-expanded to FP16 — the
        // bandwidth-bound cost that dominated prefill. The codes were written by
        // `write_kv_gpu_into` earlier in this same command batch, so they are
        // visible in-stream (GPU→GPU hazard-tracked); no per-layer drain.
        kv_cache.fused_attention_batch_into(
            batch,
            kernels,
            &bs.q,
            &bs.rq,
            &bs.vacc,
            &bs.attn_out,
            kv_len,
            start_pos as u32,
            rows,
            n_head,
            p.sliding_window as u32,
        )?;
        w.attn_output
            .matmul_nt_into(batch, kernels, &bs.attn_out, &bs.ffn_out, rows)?;
        // Fused post_attention_norm + residual #1 (mirrors the decode path):
        // attn_resid = rms_norm(ffn_out) * attn_post_norm + input
        kernels.rms_norm_add_into(
            batch,
            &bs.ffn_out,
            &w.attn_post_norm,
            input,
            &bs.attn_resid,
            h,
            rows,
            eps,
        )?;

        let inter = p.intermediate_size as u32;
        kernels.rms_norm_into(batch, &bs.attn_resid, &w.ffn_norm, &bs.normed, h, rows, eps)?;
        w.ffn_gate
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.gate, rows)?;
        w.ffn_up
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.up, rows)?;
        kernels.gelu_mul_into(batch, &bs.gate, &bs.up, &bs.gate, rows * inter)?;
        w.ffn_down
            .matmul_nt_into(batch, kernels, &bs.gate, &bs.ffn_out, rows)?;
        // Fused post_feedforward_norm + residual #2:
        // output = rms_norm(ffn_out) * ffn_post_norm + attn_resid
        kernels.rms_norm_add_into(
            batch,
            &bs.ffn_out,
            &w.ffn_post_norm,
            &bs.attn_resid,
            output,
            h,
            rows,
            eps,
        )?;

        // Per-layer input (PLE) injection: copy this layer's [n × pld] slice
        // out of the batched PLE tensor (CPU, before commit) and inject.
        if let (Some(gate_w), Some(proj_w), Some(post_w), Some((ple_all, n_layer_total))) = (
            &w.per_layer_inp_gate,
            &w.per_layer_proj,
            &w.per_layer_post_norm,
            ple_all,
        ) {
            let pld = p.per_layer_dim;
            let pldu = pld as u32;
            // Extract this layer's `[n_rows × pld]` slice from the batched PLE
            // tensor (tightly packed `[n_rows × n_layer_total × pld]`) on the
            // GPU. A CPU `copy_from_bytes` here would race the not-yet-executed
            // GPU reads of the shared `ple_slice` scratch whenever more than one
            // layer shares a command buffer (`LOCAL_AI_BATCH_CHUNK > 1`), since
            // hazard tracking only orders GPU↔GPU, not CPU-write↔GPU-read on
            // shared memory. The strided gather keeps the work in-stream and
            // hazard-tracked.
            kernels.gather_strided_f16_into(
                batch,
                ple_all,
                &bs.ple_slice,
                (n_layer_total * pld) as u32,
                (p.layer_idx * pld) as u32,
                pldu,
                rows,
            )?;
            gate_w.matmul_nt_into(batch, kernels, output, &bs.ple_g, rows)?;
            kernels.gelu_mul_into(batch, &bs.ple_g, &bs.ple_slice, &bs.ple_g, rows * pldu)?;
            proj_w.matmul_nt_into(batch, kernels, &bs.ple_g, &bs.ffn_out, rows)?;
            // Fused post-norm + residual #3: output += rms_norm(ffn_out) * post_w
            kernels.rms_norm_add_into(batch, &bs.ffn_out, post_w, output, output, h, rows, eps)?;
        }

        if let Some(scale) = &w.layer_output_scale
            && let Some(&v) = scale.as_slice::<half::f16>().first()
        {
            kernels.scale_in_place_gpu_into(batch, output, v.to_f32(), rows * h)?;
        }
        Ok(())
    }

    /// Continuous-batching decode over the `TurboQuant` KV backing: one layer
    /// over `n_lanes` **independent** sequences. All weight-heavy
    /// matmuls/FFN/norms run once at M=`n_lanes` (the ~2.7× amortization), while
    /// the three per-lane ops use the batched-decode kernels: `RoPE` reads each
    /// lane's position from `positions`, the per-lane K/V is encoded straight
    /// into `bits`-bit codes ([`QuantizedUnifiedKvPool::write_kv_into`]), and
    /// attention reads those codes directly via the fused
    /// rotate → attend → inverse-rotate path
    /// ([`QuantizedUnifiedKvPool::attention_into`]) up to each lane's own
    /// position. `positions` is a GPU `[n_lanes]` u32 buffer of current absolute
    /// positions shared by rope, scatter and attention so they stay consistent.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn forward_decode_batch_turbo(
        &self,
        batch: &mut CommandBatch,
        kernels: &Kernels,
        n_lanes: usize,
        positions: &MetalBuffer,
        input: &MetalBuffer,
        output: &MetalBuffer,
        bs: &BatchScratch,
        ones: &MetalBuffer,
        pool: &QuantizedUnifiedKvPool,
        ple_all: Option<(&MetalBuffer, usize)>,
    ) -> crate::Result<()> {
        let p = &self.params;
        let w = &self.weights;
        let rows = n_lanes as u32;
        let h = p.hidden_size as u32;
        let hd = p.head_dim as u32;
        let n_head = p.num_attention_heads as u32;
        let n_kv = p.num_key_value_heads as u32;
        let eps = p.rms_norm_eps;
        let window = p.sliding_window as u32;

        kernels.rms_norm_into(batch, input, &w.attn_norm, &bs.normed, h, rows, eps)?;
        w.attn_q
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.q, rows)?;
        if let Some(qn) = &w.q_norm {
            kernels.qk_norm_gpu_into(batch, &bs.q, qn, hd, rows * n_head, eps)?;
        }
        kernels.rope_batch_decode_into(
            batch,
            &bs.q,
            positions,
            p.rope_theta,
            hd,
            n_head,
            rows,
            p.partial_rotary_factor,
        )?;

        if !p.is_kv_shared
            && let (Some(wk), Some(wv)) = (&w.attn_k, &w.attn_v)
        {
            wk.matmul_nt_into(batch, kernels, &bs.normed, &bs.k, rows)?;
            wv.matmul_nt_into(batch, kernels, &bs.normed, &bs.v, rows)?;
            if let Some(kn) = &w.k_norm {
                kernels.qk_norm_gpu_into(batch, &bs.k, kn, hd, rows * n_kv, eps)?;
            }
            kernels.rms_norm_into(batch, &bs.v, ones, &bs.v, hd, rows * n_kv, eps)?;
            kernels.rope_batch_decode_into(
                batch,
                &bs.k,
                positions,
                p.rope_theta,
                hd,
                n_kv,
                rows,
                p.partial_rotary_factor,
            )?;
            pool.write_kv_into(batch, kernels, &bs.k, &bs.v, positions, n_lanes)?;
        }

        pool.attention_into(
            batch,
            kernels,
            &bs.q,
            &bs.rq,
            &bs.vacc,
            &bs.attn_out,
            positions,
            n_head,
            window,
            n_lanes,
        )?;

        w.attn_output
            .matmul_nt_into(batch, kernels, &bs.attn_out, &bs.ffn_out, rows)?;
        kernels.rms_norm_add_into(
            batch,
            &bs.ffn_out,
            &w.attn_post_norm,
            input,
            &bs.attn_resid,
            h,
            rows,
            eps,
        )?;

        let inter = p.intermediate_size as u32;
        kernels.rms_norm_into(batch, &bs.attn_resid, &w.ffn_norm, &bs.normed, h, rows, eps)?;
        w.ffn_gate
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.gate, rows)?;
        w.ffn_up
            .matmul_nt_into(batch, kernels, &bs.normed, &bs.up, rows)?;
        kernels.gelu_mul_into(batch, &bs.gate, &bs.up, &bs.gate, rows * inter)?;
        w.ffn_down
            .matmul_nt_into(batch, kernels, &bs.gate, &bs.ffn_out, rows)?;
        kernels.rms_norm_add_into(
            batch,
            &bs.ffn_out,
            &w.ffn_post_norm,
            &bs.attn_resid,
            output,
            h,
            rows,
            eps,
        )?;

        if let (Some(gate_w), Some(proj_w), Some(post_w), Some((ple_all, n_layer_total))) = (
            &w.per_layer_inp_gate,
            &w.per_layer_proj,
            &w.per_layer_post_norm,
            ple_all,
        ) {
            let pld = p.per_layer_dim;
            let pldu = pld as u32;
            kernels.gather_strided_f16_into(
                batch,
                ple_all,
                &bs.ple_slice,
                (n_layer_total * pld) as u32,
                (p.layer_idx * pld) as u32,
                pldu,
                rows,
            )?;
            gate_w.matmul_nt_into(batch, kernels, output, &bs.ple_g, rows)?;
            kernels.gelu_mul_into(batch, &bs.ple_g, &bs.ple_slice, &bs.ple_g, rows * pldu)?;
            proj_w.matmul_nt_into(batch, kernels, &bs.ple_g, &bs.ffn_out, rows)?;
            kernels.rms_norm_add_into(batch, &bs.ffn_out, post_w, output, output, h, rows, eps)?;
        }

        if let Some(scale) = &w.layer_output_scale
            && let Some(&v) = scale.as_slice::<half::f16>().first()
        {
            kernels.scale_in_place_gpu_into(batch, output, v.to_f32(), rows * h)?;
        }
        Ok(())
    }
}

/// Pre-allocated GPU scratch for batched (multi-token) forward passes, sized
/// for `max_batch` rows at the widest layer dimensions.
pub struct BatchScratch {
    pub max_batch: usize,
    pub hidden_a: MetalBuffer,
    pub hidden_b: MetalBuffer,
    pub normed: MetalBuffer,
    pub q: MetalBuffer,
    pub k: MetalBuffer,
    pub v: MetalBuffer,
    pub attn_out: MetalBuffer,
    /// Per-(lane, q-head) rotated query (`rq`) and rotated-space value
    /// accumulation (`vacc`) for the fused `TurboQuant` batched attention, f32 at
    /// `[max_batch × n_q_heads × head_dim]`.
    pub rq: MetalBuffer,
    pub vacc: MetalBuffer,
    pub attn_resid: MetalBuffer,
    pub gate: MetalBuffer,
    pub up: MetalBuffer,
    pub ffn_out: MetalBuffer,
    /// Batched PLE tensors: `[max_batch × n_layer × per_layer_dim]`.
    pub ple_tok: MetalBuffer,
    pub ple_proj: MetalBuffer,
    pub ple_all: MetalBuffer,
    /// Per-layer staging: `[max_batch × per_layer_dim]`.
    pub ple_g: MetalBuffer,
    pub ple_slice: MetalBuffer,
}

impl BatchScratch {
    /// Dimensions must be the maxima across all layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        max_batch: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        per_layer_dim: usize,
        num_layers: usize,
    ) -> crate::Result<Self> {
        const F16: usize = std::mem::size_of::<half::f16>();
        const F32: usize = std::mem::size_of::<f32>();
        let alloc = |elems: usize| {
            MetalBuffer::empty(device, (elems * F16).max(2))
                .map_err(|e| crate::Error::InvalidArgument(e.to_string()))
        };
        let alloc_f32 = |elems: usize| {
            MetalBuffer::empty(device, (elems * F32).max(4))
                .map_err(|e| crate::Error::InvalidArgument(e.to_string()))
        };
        let mb = max_batch;
        let ple_total = mb * num_layers * per_layer_dim;
        Ok(Self {
            max_batch,
            hidden_a: alloc(mb * hidden_size)?,
            hidden_b: alloc(mb * hidden_size)?,
            normed: alloc(mb * hidden_size)?,
            q: alloc(mb * num_attention_heads * head_dim)?,
            k: alloc(mb * num_key_value_heads * head_dim)?,
            v: alloc(mb * num_key_value_heads * head_dim)?,
            attn_out: alloc(mb * num_attention_heads * head_dim)?,
            rq: alloc_f32(mb * num_attention_heads * head_dim)?,
            vacc: alloc_f32(mb * num_attention_heads * head_dim)?,
            attn_resid: alloc(mb * hidden_size)?,
            gate: alloc(mb * intermediate_size)?,
            up: alloc(mb * intermediate_size)?,
            ffn_out: alloc(mb * hidden_size)?,
            ple_tok: alloc(ple_total)?,
            ple_proj: alloc(ple_total)?,
            ple_all: alloc(ple_total)?,
            ple_g: alloc(mb * per_layer_dim.max(1))?,
            ple_slice: alloc(mb * per_layer_dim.max(1))?,
        })
    }
}

/// Pre-allocated GPU scratch buffers sized for the widest layer.
pub struct ScratchBuffers {
    pub normed: MetalBuffer,
    pub q: MetalBuffer,
    pub k: MetalBuffer,
    pub v: MetalBuffer,
    /// f32 scratch for the fused `TurboQuant` attention path: rotated query
    /// (`rq`) and rotated-space value accumulation (`vacc`), one per Q head.
    pub rq: MetalBuffer,
    pub vacc: MetalBuffer,
    pub attn_out: MetalBuffer,
    pub attn_resid: MetalBuffer,
    pub gate: MetalBuffer,
    pub up: MetalBuffer,
    pub ffn_out: MetalBuffer,
    pub ple: MetalBuffer,
    pub ple_h: MetalBuffer,
    pub ones: MetalBuffer,
    /// Per-split partials for the flash-decoding attention path
    /// (`num_attention_heads * MAX_FLASH_SPLITS` f32 for max/sum, ×`head_dim`
    /// for acc). Lets long-context decode split the KV range across more
    /// threadgroups to recover GPU occupancy.
    pub fa_pmax: MetalBuffer,
    pub fa_psum: MetalBuffer,
    pub fa_pacc: MetalBuffer,
}

/// Maximum KV splits the decode attention path may use; must match the
/// `MAX_SPLITS` cap in `Kernels::flash_split_count`.
pub const MAX_FLASH_SPLITS: usize = 16;

impl ScratchBuffers {
    /// `head_dim` / `intermediate_size` must be the maximum across all layers.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        hidden_size: usize,
        intermediate_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        per_layer_dim: usize,
    ) -> crate::Result<Self> {
        const F16: usize = std::mem::size_of::<half::f16>();
        let alloc = |elems: usize| {
            MetalBuffer::empty(device, elems * F16)
                .map_err(|e| crate::Error::InvalidArgument(e.to_string()))
        };
        let q_elems = num_attention_heads * head_dim;
        let kv_elems = num_key_value_heads * head_dim;
        let ones = MetalBuffer::from_slice(device, &vec![half::f16::ONE; head_dim])
            .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?;
        Ok(Self {
            normed: alloc(hidden_size)?,
            q: alloc(q_elems)?,
            k: alloc(kv_elems)?,
            v: alloc(kv_elems)?,
            rq: MetalBuffer::empty(device, q_elems * std::mem::size_of::<f32>())
                .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?,
            vacc: MetalBuffer::empty(device, q_elems * std::mem::size_of::<f32>())
                .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?,
            attn_out: alloc(q_elems)?,
            attn_resid: alloc(hidden_size)?,
            gate: alloc(intermediate_size)?,
            up: alloc(intermediate_size)?,
            ffn_out: alloc(hidden_size)?,
            ple: alloc(per_layer_dim.max(1))?,
            ple_h: alloc(hidden_size)?,
            ones,
            fa_pmax: MetalBuffer::empty(
                device,
                num_attention_heads * MAX_FLASH_SPLITS * std::mem::size_of::<f32>(),
            )
            .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?,
            fa_psum: MetalBuffer::empty(
                device,
                num_attention_heads * MAX_FLASH_SPLITS * std::mem::size_of::<f32>(),
            )
            .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?,
            fa_pacc: MetalBuffer::empty(
                device,
                num_attention_heads * MAX_FLASH_SPLITS * head_dim * std::mem::size_of::<f32>(),
            )
            .map_err(|e| crate::Error::InvalidArgument(e.to_string()))?,
        })
    }
}
