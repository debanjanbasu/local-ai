//! `TurboQuant` per-(token, head) quantized KV cache for the decode path.
//!
//! K/V vectors are stored with the data-oblivious vector quantizer of
//! arXiv:2504.19874 from [`crate::turboquant`]: a randomized Hadamard
//! rotation, MSE-optimal Lloyd–Max codes at `bits` ∈ 2/3/4 per coordinate,
//! and one FP16 norm per `(token, head)` — zero per-block scale/zero-point
//! overhead. At the default 4 bits this cuts KV memory ~4× versus FP16 while
//! staying near-lossless (and measures the same decode speed as the int8
//! affine cache it replaced; 2/3 bits trade accuracy for further compression).
//!
//! On read, the inverse-FWHT GPU kernel (`dequantize_kv_turboquant`) expands
//! the active `[0..seq_len]` range back to FP16 into shared scratch, which the
//! existing flash-attention kernel consumes unchanged.
//!
//! The richer paged / multi-request the paged multi-request machinery is left
//! untouched; this type is used directly by [`crate::pipeline::Pipeline`].

use half::f16;
use local_metal::buffer::MetalBuffer;
use local_metal::context::MetalContext;
use local_metal::kernels::Kernels;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use crate::Result;
use crate::turboquant::TurboQuant;

/// Fixed rotation seed shared by every cache so encode (CPU) and tests agree.
/// (The module is private, so this stays crate-internal.)
pub const TQ_SEED: u64 = 0x7459_C0DE;

/// Default bits per coordinate: 2-bit `TurboQuant`. All bit-widths share the same
/// fused GPU encode + attention path (post-QJL removal), so 2-bit runs at the same
/// speed as 4-bit while giving the smallest KV footprint — which is what lets every
/// device, down to the iPhone SE 3, hold the model's full context window. Override
/// with `LOCAL_AI_KV_QUANT` (`tq2`/`tq3`/`tq4`).
pub const DEFAULT_KV_BITS: u8 = 2;

/// Largest multi-row write a single prefill chunk can append in one call. A
/// sliding-window ring must hold the window plus the rows still in flight from
/// such a chunk, so its capacity is `sliding_window + MAX_PREFILL_CHUNK - 1` at
/// minimum. Kept in sync with the prefill chunk size in `pipeline`.
pub const MAX_PREFILL_CHUNK: usize = 256;

/// Ring capacity for a sliding-window layer: the window plus prefill slack,
/// rounded up to a power of two for cheap modulo, capped at the logical max.
#[must_use]
pub fn swa_ring_capacity(window: usize, max_positions: usize) -> usize {
    let min_safe = window + MAX_PREFILL_CHUNK - 1;
    (window * 2)
        .max(min_safe)
        .next_power_of_two()
        .min(max_positions)
}

/// Read the cache bit-width from `LOCAL_AI_KV_QUANT` (`tq2` | `tq3` | `tq4`);
/// anything else (including unset) is the 2-bit default (smallest footprint, same
/// fused-path speed as 4-bit).
#[must_use]
pub fn kv_bits_from_env() -> u8 {
    match std::env::var("LOCAL_AI_KV_QUANT").ok().as_deref() {
        Some("tq3") => 3,
        Some("tq4") => 4,
        _ => DEFAULT_KV_BITS,
    }
}

/// Per-layer `TurboQuant` KV cache with per-(token, head) norms.
///
/// `keys_q` / `values_q` hold the packed codes
/// (`max_positions × n_kv_heads × code_bytes`); `key_norms` / `value_norms`
/// hold one f16 vector norm per `(token, head)`.
pub struct QuantizedKvCache {
    keys_q: MetalBuffer,
    values_q: MetalBuffer,
    key_norms: MetalBuffer,
    value_norms: MetalBuffer,
    codec: TurboQuant,
    /// GPU copies of the codec's reconstruction levels and rotation signs.
    levels: MetalBuffer,
    signs: MetalBuffer,
    n_kv_heads: usize,
    head_dim: usize,
    /// Logical context limit (the model's max positions for this layer).
    max_positions: usize,
    /// Physically allocated token rows. Equals `max_positions` for full
    /// attention; for sliding-window layers it is the ring capacity
    /// (`window + decode slack`), so logical position `p` is stored at physical
    /// slot `p % physical_positions`.
    physical_positions: usize,
}

/// CPU-owned prefix snapshot for prompt/prefix reuse.
///
/// For full-attention caches only rows `[0..positions)` are copied, so a cached
/// prompt scales with the useful prefix length. For ringed sliding-window caches
/// the entire physical ring is copied (it is small — `physical_positions` rows)
/// and `positions` records the logical length at snapshot time, so a restore can
/// check the live window is still covered before reusing it.
pub struct QuantizedKvCacheSnapshot {
    positions: usize,
    physical_positions: usize,
    is_ring: bool,
    keys_q: Vec<u8>,
    values_q: Vec<u8>,
    key_norms: Vec<u8>,
    value_norms: Vec<u8>,
}

/// `TurboQuant`-encode one head's FP16 slice into `dst_codes`, returning the norm.
///
/// `x_scratch` and `rot_scratch` are caller-owned buffers reused across heads so
/// the hot path allocates nothing per head; codes are written straight into
/// `dst_codes` via [`TurboQuant::encode_into`].
fn quantize_head(
    codec: &TurboQuant,
    src: &[f16],
    dst_codes: &mut [u8],
    x_scratch: &mut Vec<f32>,
    rot_scratch: &mut Vec<f32>,
) -> f16 {
    x_scratch.clear();
    x_scratch.extend(src.iter().map(|v| v.to_f32()));
    codec.encode_into(x_scratch, rot_scratch, dst_codes)
}

impl QuantizedKvCache {
    /// Allocate a 4-bit (near-lossless) KV cache for `max_positions` tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails or `head_dim` is not
    /// a power of two ≤ 512.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n_kv_heads: usize,
        head_dim: usize,
        max_positions: usize,
    ) -> Result<Self> {
        Self::new_with_bits(device, n_kv_heads, head_dim, max_positions, DEFAULT_KV_BITS)
    }

    /// Allocate a KV cache for `max_positions` tokens at `bits` per coordinate.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails, `bits` is outside
    /// 2–4, or `head_dim` is not a power of two ≤ 512 (the inverse-FWHT decode
    /// kernel's threadgroup limit).
    pub fn new_with_bits(
        device: &ProtocolObject<dyn MTLDevice>,
        n_kv_heads: usize,
        head_dim: usize,
        max_positions: usize,
        bits: u8,
    ) -> Result<Self> {
        // Full-attention layout: physical == logical, no sliding window.
        Self::new_with_bits_and_ring(
            device,
            n_kv_heads,
            head_dim,
            max_positions,
            max_positions,
            0,
            bits,
        )
    }

    /// Allocate a KV cache that physically holds only `physical_positions` token
    /// rows while logically addressing up to `max_positions`. When
    /// `physical_positions < max_positions` the cache is a **sliding-window
    /// ring**: logical position `p` is stored at physical slot
    /// `p % physical_positions`, and only the last `sliding_window` tokens are
    /// ever read by attention. Pass `physical_positions == max_positions` and
    /// `sliding_window == 0` for the full-attention (absolute) layout.
    ///
    /// # Errors
    ///
    /// Returns an error if Metal buffer allocation fails, `bits` is outside
    /// 2–4, `head_dim` is not a power of two ≤ 512, or the ring capacity is too
    /// small to hold the sliding window plus the multi-row prefill slack.
    pub fn new_with_bits_and_ring(
        device: &ProtocolObject<dyn MTLDevice>,
        n_kv_heads: usize,
        head_dim: usize,
        max_positions: usize,
        physical_positions: usize,
        sliding_window: usize,
        bits: u8,
    ) -> Result<Self> {
        if !(2..=4).contains(&bits) {
            return Err(crate::Error::InvalidArgument(format!(
                "TurboQuant bits must be 2..=4, got {bits}"
            )));
        }
        if !head_dim.is_power_of_two() || head_dim > 512 {
            return Err(crate::Error::InvalidArgument(format!(
                "TurboQuant head_dim must be a power of two <= 512, got {head_dim}"
            )));
        }
        let physical_positions = physical_positions.min(max_positions).max(1);
        let is_ring = physical_positions < max_positions;
        if is_ring && physical_positions < sliding_window + MAX_PREFILL_CHUNK - 1 {
            return Err(crate::Error::InvalidArgument(format!(
                "ring capacity {physical_positions} too small for window {sliding_window} \
                 + prefill slack {}",
                MAX_PREFILL_CHUNK - 1
            )));
        }
        let codec = TurboQuant::new(head_dim, bits, TQ_SEED);
        let slots = physical_positions * n_kv_heads;
        let payload_bytes = slots * codec.code_bytes();
        let levels = MetalBuffer::from_slice(device, codec.levels())?;
        let signs = MetalBuffer::from_slice(device, codec.signs())?;
        Ok(Self {
            keys_q: MetalBuffer::empty(device, payload_bytes.max(1))?,
            values_q: MetalBuffer::empty(device, payload_bytes.max(1))?,
            key_norms: MetalBuffer::empty(device, slots.max(1) * std::mem::size_of::<f16>())?,
            value_norms: MetalBuffer::empty(device, slots.max(1) * std::mem::size_of::<f16>())?,
            codec,
            levels,
            signs,
            n_kv_heads,
            head_dim,
            max_positions,
            physical_positions,
        })
    }

    /// Number of KV heads in this cache.
    #[must_use]
    pub const fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Per-head dimension of this cache.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Physically allocated token rows (equals `max_positions` for full
    /// attention, the ring capacity for sliding-window layers).
    #[must_use]
    pub const fn physical_positions(&self) -> usize {
        self.physical_positions
    }

    /// Whether this cache is a sliding-window ring (physical < logical).
    #[must_use]
    pub const fn is_ringed(&self) -> bool {
        self.physical_positions < self.max_positions
    }

    /// Ring capacity to hand the GPU kernels: the physical row count for a
    /// ringed cache, or `0` (identity / absolute addressing) for full attention.
    #[must_use]
    pub const fn ring_capacity(&self) -> u32 {
        if self.is_ringed() {
            self.physical_positions as u32
        } else {
            0
        }
    }

    /// Map a logical token position to its physical row in the ring.
    #[must_use]
    const fn physical_slot(&self, position: usize) -> usize {
        if self.is_ringed() {
            position % self.physical_positions
        } else {
            position
        }
    }

    /// Packed code bytes per (token, head) vector.
    #[must_use]
    pub const fn code_bytes(&self) -> usize {
        self.codec.code_bytes()
    }

    /// Packed key codes (`[max_positions, n_kv_heads, code_bytes]` `uchar`).
    pub(crate) const fn keys_q(&self) -> &MetalBuffer {
        &self.keys_q
    }

    /// Packed value codes (same layout as [`Self::keys_q`]).
    pub(crate) const fn values_q(&self) -> &MetalBuffer {
        &self.values_q
    }

    /// Per-(token, head) key norms (`[max_positions, n_kv_heads]` f16).
    pub(crate) const fn key_norms(&self) -> &MetalBuffer {
        &self.key_norms
    }

    /// Per-(token, head) value norms (same layout as [`Self::key_norms`]).
    pub(crate) const fn value_norms(&self) -> &MetalBuffer {
        &self.value_norms
    }

    const fn code_row_bytes(&self) -> usize {
        self.n_kv_heads * self.codec.code_bytes()
    }

    const fn norm_row_bytes(&self) -> usize {
        self.n_kv_heads * std::mem::size_of::<f16>()
    }

    /// Longest logical prefix that can be snapshot/restored as a contiguous
    /// run of physical rows. For full attention this is the whole context; for
    /// a sliding-window ring it is the physical capacity, beyond which writes
    /// wrap and a flat prefix copy would alias older rows.
    #[must_use]
    pub const fn max_snapshot_positions(&self) -> usize {
        if self.is_ringed() {
            self.physical_positions
        } else {
            self.max_positions
        }
    }

    /// Copy cache rows `[0..positions)` into CPU memory for prompt-prefix reuse.
    ///
    /// `positions` must not exceed [`Self::max_snapshot_positions`]; the caller
    /// (`Pipeline::remember_prompt`) guarantees this so the copied rows are a
    /// contiguous, un-wrapped run that any later [`Self::restore_prefix`] can
    /// reload verbatim.
    #[must_use]
    pub(crate) fn snapshot_prefix(&self, positions: usize) -> QuantizedKvCacheSnapshot {
        let positions = positions.min(self.max_snapshot_positions());
        let code_bytes = positions * self.code_row_bytes();
        let norm_bytes = positions * self.norm_row_bytes();
        QuantizedKvCacheSnapshot {
            positions,
            physical_positions: self.physical_positions,
            is_ring: self.is_ringed(),
            keys_q: self.keys_q.as_slice::<u8>()[..code_bytes].to_vec(),
            values_q: self.values_q.as_slice::<u8>()[..code_bytes].to_vec(),
            key_norms: self.key_norms.as_slice::<u8>()[..norm_bytes].to_vec(),
            value_norms: self.value_norms.as_slice::<u8>()[..norm_bytes].to_vec(),
        }
    }

    /// Restore up to `positions` rows from a CPU prefix snapshot.
    pub(crate) fn restore_prefix(&self, snapshot: &QuantizedKvCacheSnapshot, positions: usize) {
        debug_assert_eq!(snapshot.physical_positions, self.physical_positions);
        debug_assert_eq!(snapshot.is_ring, self.is_ringed());
        let positions = positions
            .min(snapshot.positions)
            .min(self.max_snapshot_positions());
        let code_bytes = positions * self.code_row_bytes();
        let norm_bytes = positions * self.norm_row_bytes();
        self.keys_q
            .copy_from_bytes(&snapshot.keys_q[..code_bytes], 0);
        self.values_q
            .copy_from_bytes(&snapshot.values_q[..code_bytes], 0);
        self.key_norms
            .copy_from_bytes(&snapshot.key_norms[..norm_bytes], 0);
        self.value_norms
            .copy_from_bytes(&snapshot.value_norms[..norm_bytes], 0);
    }

    /// Zero the code buffers and norms; a zero norm annihilates whatever the
    /// codes say, so any stale slot dequantizes to exact zero.
    #[allow(unsafe_code)]
    pub(crate) fn reset(&self) {
        for buf in [
            &self.keys_q,
            &self.values_q,
            &self.key_norms,
            &self.value_norms,
        ] {
            let ptr = buf.contents_mut_ptr();
            // SAFETY: pointer/len belong to this shared, CPU-writable buffer.
            unsafe {
                std::ptr::write_bytes(ptr, 0, buf.length());
            }
        }
    }

    /// GPU-resident KV write: `TurboQuant`-encode `n_rows` freshly projected
    /// FP16 K/V rows (`k_src`/`v_src`, `[n_rows, n_kv_heads, head_dim]` starting
    /// at element 0) straight into the cache at `position`, with no CPU
    /// readback/encode round-trip.
    ///
    /// # Errors
    ///
    /// Returns an error if the encode kernel is unavailable or the range
    /// exceeds the cache capacity.
    pub fn write_kv_gpu_into(
        &self,
        batch: &mut local_metal::batch::CommandBatch,
        kernels: &Kernels,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
        position: usize,
        n_rows: usize,
    ) -> Result<()> {
        if position + n_rows > self.max_positions {
            return Err(crate::Error::InvalidArgument(format!(
                "KV write [{position}..{}) exceeds capacity {}",
                position + n_rows,
                self.max_positions
            )));
        }
        let hd = self.head_dim as u32;
        let n_kv = self.n_kv_heads as u32;
        let bits = u32::from(self.codec.bits());
        let pos = u32::try_from(position)
            .map_err(|_| crate::Error::InvalidArgument("KV position overflow".into()))?;
        let rows = u32::try_from(n_rows)
            .map_err(|_| crate::Error::InvalidArgument("KV row count overflow".into()))?;
        let ring = self.ring_capacity();
        kernels.encode_kv_turboquant_into(
            batch,
            k_src,
            &self.levels,
            &self.signs,
            &self.keys_q,
            &self.key_norms,
            hd,
            n_kv,
            rows,
            bits,
            pos,
            ring,
        )?;
        kernels.encode_kv_turboquant_into(
            batch,
            v_src,
            &self.levels,
            &self.signs,
            &self.values_q,
            &self.value_norms,
            hd,
            n_kv,
            rows,
            bits,
            pos,
            ring,
        )?;
        Ok(())
    }

    /// Quantize and store the current token's K and V vectors at `position`.
    ///
    /// `key` and `value` are `n_kv_heads * head_dim` FP16 values each.
    pub fn write_kv(&mut self, position: usize, key: &[f16], value: &[f16]) {
        debug_assert!(position < self.max_positions);
        let hd = self.head_dim;
        let n_kv = self.n_kv_heads;
        let cb = self.codec.code_bytes();
        let slot = self.physical_slot(position);
        let norm_off = slot * n_kv;
        let code_off = slot * n_kv * cb;

        let keys = self.keys_q.as_mut_slice::<u8>();
        let key_norms = self.key_norms.as_mut_slice::<f16>();
        let values = self.values_q.as_mut_slice::<u8>();
        let value_norms = self.value_norms.as_mut_slice::<f16>();
        // Scratch buffers reused across every head (and K/V) for this token.
        let mut x_scratch: Vec<f32> = Vec::with_capacity(hd);
        let mut rot_scratch: Vec<f32> = Vec::with_capacity(hd);
        for h in 0..n_kv {
            let src_k = &key[h * hd..(h + 1) * hd];
            let dst_k = &mut keys[code_off + h * cb..code_off + (h + 1) * cb];
            key_norms[norm_off + h] =
                quantize_head(&self.codec, src_k, dst_k, &mut x_scratch, &mut rot_scratch);

            let src_v = &value[h * hd..(h + 1) * hd];
            let dst_v = &mut values[code_off + h * cb..code_off + (h + 1) * cb];
            value_norms[norm_off + h] =
                quantize_head(&self.codec, src_v, dst_v, &mut x_scratch, &mut rot_scratch);
        }
    }

    /// Expand positions `[0..seq_len)` of the quantized cache back to FP16
    /// into `k_out` / `v_out` (each `seq_len * n_kv_heads * head_dim` f16)
    /// via the inverse-FWHT GPU kernel.
    ///
    /// Positions `[token_start, seq_len)` are decoded into their absolute output
    /// slots; earlier slots are left untouched. Sliding-window layers pass the
    /// window start so only the active window is expanded (the attention kernel
    /// masks everything below it anyway); full-attention layers pass `0`.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel pipeline is unavailable or encoding fails.
    pub fn dequantize_into(
        &self,
        ctx: &MetalContext,
        kernels: &Kernels,
        k_out: &MetalBuffer,
        v_out: &MetalBuffer,
        seq_len: u32,
        token_start: u32,
    ) -> Result<()> {
        let hd = self.head_dim as u32;
        let n_kv = self.n_kv_heads as u32;
        let bits = u32::from(self.codec.bits());
        let ring = self.ring_capacity();
        kernels.dequantize_kv_turboquant(
            ctx,
            &self.keys_q,
            &self.key_norms,
            &self.levels,
            &self.signs,
            k_out,
            hd,
            n_kv,
            seq_len,
            bits,
            token_start,
            ring,
        )?;
        kernels.dequantize_kv_turboquant(
            ctx,
            &self.values_q,
            &self.value_norms,
            &self.levels,
            &self.signs,
            v_out,
            hd,
            n_kv,
            seq_len,
            bits,
            token_start,
            ring,
        )?;
        Ok(())
    }

    /// Fused `TurboQuant` flash-attention decode for a single query token: the
    /// attention logits and value accumulation are computed straight from the
    /// packed K/V codes (no FP16 expansion of the cache), cutting the
    /// bandwidth-bound decode read from 16-bit to `bits`-bit per coordinate.
    ///
    /// `q` is the `RoPE`'d query (`n_q_heads * head_dim` f16); `rq` and `vacc` are
    /// caller-owned f32 scratch buffers (`n_q_heads * head_dim` each) for the
    /// rotated query and rotated-space value accumulation; `attn_out` receives
    /// the f16 attention output. `window == 0` disables the sliding window.
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_attention_into(
        &self,
        batch: &mut local_metal::batch::CommandBatch,
        kernels: &Kernels,
        q: &MetalBuffer,
        rq: &MetalBuffer,
        vacc: &MetalBuffer,
        attn_out: &MetalBuffer,
        seq_len: u32,
        current_pos: u32,
        n_q_heads: u32,
        window: u32,
    ) -> Result<()> {
        let hd = self.head_dim as u32;
        let n_kv = self.n_kv_heads as u32;
        let bits = u32::from(self.codec.bits());
        // Rotate the query once: Rq = (1/√d)·H(sign⊙q).
        kernels.hadamard_rotate_hf_into(batch, q, &self.signs, rq, hd, n_q_heads, true, false)?;
        // Attention over the packed codes; output is the value accumulation in
        // rotated/codebook space.
        kernels.flash_attention_tq_into(
            batch,
            rq,
            &self.keys_q,
            &self.key_norms,
            &self.values_q,
            &self.value_norms,
            &self.levels,
            vacc,
            hd,
            seq_len,
            current_pos,
            n_q_heads,
            n_kv,
            window,
            bits,
            self.ring_capacity(),
        )?;
        // Inverse-rotate the accumulation back to model space → f16 attn_out.
        kernels.hadamard_rotate_fh_into(
            batch,
            vacc,
            &self.signs,
            attn_out,
            hd,
            n_q_heads,
            false,
            true,
        )?;
        Ok(())
    }

    /// Fused `TurboQuant` flash-attention for a multi-row prefill / MTP-verify
    /// chunk: the `n_rows` query rows (row `r` at absolute position
    /// `start_pos + r`) attend the just-written batch straight from the packed
    /// K/V codes — no FP16 expansion of the cache — with a per-row causal cutoff
    /// and sliding window. This is the prefill analogue of
    /// [`Self::fused_attention_into`] and replaces the
    /// `dequantize_into_batch` + `flash_attention_prefill` pair.
    ///
    /// `q` is the `RoPE`'d query laid out `[n_rows, n_q_heads, head_dim]` (f16);
    /// `rq` and `vacc` are caller-owned f32 scratch buffers of the same element
    /// count for the rotated queries and rotated-space value accumulation;
    /// `attn_out` receives the f16 attention output in the same layout.
    /// `window == 0` disables the sliding window. Must be called after the K/V
    /// rows have been written with [`Self::write_kv_gpu_into`] into the same
    /// command batch so the codes are visible in-stream.
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_attention_batch_into(
        &self,
        batch: &mut local_metal::batch::CommandBatch,
        kernels: &Kernels,
        q: &MetalBuffer,
        rq: &MetalBuffer,
        vacc: &MetalBuffer,
        attn_out: &MetalBuffer,
        kv_len: u32,
        start_pos: u32,
        n_rows: u32,
        n_q_heads: u32,
        window: u32,
    ) -> Result<()> {
        let hd = self.head_dim as u32;
        let n_kv = self.n_kv_heads as u32;
        let bits = u32::from(self.codec.bits());
        let vecs = n_rows * n_q_heads;
        // Rotate every query row: Rq = (1/√d)·H(sign⊙q).
        kernels.hadamard_rotate_hf_into(batch, q, &self.signs, rq, hd, vecs, true, false)?;
        // Per-row causal attention over the packed codes; output is the value
        // accumulation in rotated/codebook space. The query-tiled kernel
        // dequantizes each K/V code tile into threadgroup memory once and reuses
        // it across a block of query rows (~8× less K/V code bandwidth — the
        // quadratic prefill cost). `LOCAL_AI_TQ_PREFILL_NOTILE` selects the
        // simpler non-tiled kernel (the validated correctness reference).
        let tiled = std::env::var("LOCAL_AI_TQ_PREFILL_NOTILE").as_deref() != Ok("1");
        if tiled {
            kernels.flash_attention_tq_prefill_tiled_into(
                batch,
                rq,
                &self.keys_q,
                &self.key_norms,
                &self.values_q,
                &self.value_norms,
                &self.levels,
                vacc,
                hd,
                kv_len,
                start_pos,
                n_rows,
                n_q_heads,
                n_kv,
                window,
                bits,
                self.ring_capacity(),
            )?;
        } else {
            kernels.flash_attention_tq_prefill_into(
                batch,
                rq,
                &self.keys_q,
                &self.key_norms,
                &self.values_q,
                &self.value_norms,
                &self.levels,
                vacc,
                hd,
                kv_len,
                start_pos,
                n_rows,
                n_q_heads,
                n_kv,
                window,
                bits,
                self.ring_capacity(),
            )?;
        }
        // Inverse-rotate each row's accumulation back to model space → f16.
        kernels.hadamard_rotate_fh_into(
            batch,
            vacc,
            &self.signs,
            attn_out,
            hd,
            vecs,
            false,
            true,
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use local_metal::buffer::get_default_device;
    use local_metal::shaders::ShaderLibrary;

    fn seeded_f16(n: usize, seed: u64) -> Vec<f16> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 31;
                f16::from_f32(((z >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
            })
            .collect()
    }

    fn gpu_setup() -> (MetalContext, Kernels) {
        let ctx = MetalContext::new().expect("Metal context");
        let shaders = ShaderLibrary::new(ctx.device()).expect("shader library");
        let kernels = Kernels::new(&ctx, &shaders).expect("kernels");
        (ctx, kernels)
    }

    #[test]
    fn reset_zeroes_codes_and_norms() {
        let device = get_default_device().expect("Metal device");
        let cache = QuantizedKvCache::new(&device, 1, 256, 4).expect("alloc");
        cache.reset();
        assert!(cache.keys_q().as_slice::<u8>().iter().all(|&b| b == 0));
        assert!(
            cache
                .key_norms()
                .as_slice::<f16>()
                .iter()
                .all(|&s| s.to_f32() == 0.0)
        );
    }

    #[test]
    fn snapshot_restore_roundtrips_prefix_rows_only() {
        let device = get_default_device().expect("Metal device");
        let (head_dim, n_kv, max_pos) = (256usize, 2usize, 4usize);
        let mut cache =
            QuantizedKvCache::new_with_bits(&device, n_kv, head_dim, max_pos, 3).expect("alloc");
        cache.reset();

        let k0 = seeded_f16(n_kv * head_dim, 101);
        let v0 = seeded_f16(n_kv * head_dim, 202);
        let k1 = seeded_f16(n_kv * head_dim, 303);
        let v1 = seeded_f16(n_kv * head_dim, 404);
        cache.write_kv(0, &k0, &v0);
        cache.write_kv(1, &k1, &v1);
        let snapshot = cache.snapshot_prefix(2);

        cache.reset();
        cache.restore_prefix(&snapshot, 1);
        let code_row_bytes = cache.code_row_bytes();
        let norm_row_bytes = cache.norm_row_bytes();
        assert_eq!(
            &cache.keys_q().as_slice::<u8>()[..code_row_bytes],
            &snapshot.keys_q[..code_row_bytes]
        );
        assert_eq!(
            &cache.key_norms().as_slice::<u8>()[..norm_row_bytes],
            &snapshot.key_norms[..norm_row_bytes]
        );
        assert!(
            cache.key_norms().as_slice::<f16>()[n_kv..]
                .iter()
                .all(|&s| s.to_f32() == 0.0),
            "restore_prefix(1) must not populate later rows"
        );

        cache.restore_prefix(&snapshot, 2);
        assert_eq!(
            &cache.keys_q().as_slice::<u8>()[..2 * code_row_bytes],
            &snapshot.keys_q[..]
        );
    }

    /// GPU inverse-FWHT decode must match the CPU `TurboQuant::decode` path,
    /// and unwritten (reset) rows must come back as exact zeros.
    fn assert_gpu_matches_cpu_decode(bits: u8) {
        let (ctx, kernels) = gpu_setup();
        let device = ctx.device();
        let (head_dim, n_kv, max_pos) = (256usize, 2usize, 4usize);
        let mut cache =
            QuantizedKvCache::new_with_bits(device, n_kv, head_dim, max_pos, bits).expect("alloc");
        cache.reset();

        let k0 = seeded_f16(n_kv * head_dim, 11);
        let v0 = seeded_f16(n_kv * head_dim, 22);
        let k1 = seeded_f16(n_kv * head_dim, 33);
        let v1 = seeded_f16(n_kv * head_dim, 44);
        cache.write_kv(0, &k0, &v0);
        cache.write_kv(1, &k1, &v1);

        // seq_len 3 includes one never-written row, which must decode to zeros.
        let seq_len = 3usize;
        let out_elems = seq_len * n_kv * head_dim;
        let k_out = MetalBuffer::empty(device, out_elems * 2).expect("k_out");
        let v_out = MetalBuffer::empty(device, out_elems * 2).expect("v_out");
        cache
            .dequantize_into(&ctx, &kernels, &k_out, &v_out, seq_len as u32, 0)
            .expect("dequantize");

        let codec = TurboQuant::new(head_dim, bits, TQ_SEED);
        let cb = codec.code_bytes();
        let key_codes = cache.keys_q().as_slice::<u8>();
        let norms = cache.key_norms().as_slice::<f16>();
        let gpu: &[f16] = k_out.as_slice();

        for slot in 0..2 * n_kv {
            let q = crate::turboquant::QuantizedVector {
                codes: key_codes[slot * cb..(slot + 1) * cb].to_vec(),
                norm: norms[slot],
            };
            let cpu = codec.decode(&q);
            let g = &gpu[slot * head_dim..(slot + 1) * head_dim];
            let num: f32 = cpu
                .iter()
                .zip(g)
                .map(|(c, g)| (c - g.to_f32()) * (c - g.to_f32()))
                .sum();
            let den: f32 = cpu.iter().map(|c| c * c).sum();
            let rel = (num / den.max(1e-12)).sqrt();
            assert!(
                rel < 0.01,
                "bits={bits} slot={slot}: GPU vs CPU decode rel err {rel}"
            );
        }
        // Unwritten row (position 2) decodes to exact zeros (norm == 0).
        let zero_row = &gpu[2 * n_kv * head_dim..3 * n_kv * head_dim];
        assert!(
            zero_row.iter().all(|&v| v.to_f32() == 0.0),
            "stale row must be zero"
        );
    }

    #[test]
    fn turboquant_gpu_matches_cpu_decode_4bit() {
        assert_gpu_matches_cpu_decode(4);
    }

    #[test]
    fn turboquant_gpu_matches_cpu_decode_3bit_byte_straddling() {
        assert_gpu_matches_cpu_decode(3);
    }

    #[test]
    fn turboquant_gpu_matches_cpu_decode_2bit() {
        assert_gpu_matches_cpu_decode(2);
    }

    /// End to end at 4 bits the GPU-reconstructed vectors stay close to the
    /// original FP16 inputs. The MSE-optimal 4-bit Lloyd–Max quantizer on
    /// Gaussian coordinates has ≈ √0.0095 ≈ 9.7% relative L2 error, so ~10%
    /// is the *theoretical* value; a wrong kernel (sign/order/packing) would
    /// blow far past 100%.
    #[test]
    fn turboquant_gpu_roundtrip_close_to_original_4bit() {
        let (ctx, kernels) = gpu_setup();
        let device = ctx.device();
        let (head_dim, n_kv) = (512usize, 1usize);
        let mut cache =
            QuantizedKvCache::new_with_bits(device, n_kv, head_dim, 2, 4).expect("alloc");
        cache.reset();

        let key = seeded_f16(n_kv * head_dim, 7);
        let value = seeded_f16(n_kv * head_dim, 8);
        cache.write_kv(0, &key, &value);

        let out_elems = n_kv * head_dim;
        let k_out = MetalBuffer::empty(device, out_elems * 2).expect("k_out");
        let v_out = MetalBuffer::empty(device, out_elems * 2).expect("v_out");
        cache
            .dequantize_into(&ctx, &kernels, &k_out, &v_out, 1, 0)
            .expect("dequantize");

        for (orig, out) in [(&key, &k_out), (&value, &v_out)] {
            let gpu: &[f16] = out.as_slice();
            let num: f32 = orig
                .iter()
                .zip(gpu)
                .map(|(o, g)| (o.to_f32() - g.to_f32()).powi(2))
                .sum();
            let den: f32 = orig.iter().map(|o| o.to_f32().powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel < 0.12, "4-bit roundtrip rel err too high: {rel}");
        }
    }

    /// Windowed dequant (`token_start > 0`) must produce, in the active
    /// window `[token_start, seq_len)`, exactly the same FP16 values as a full
    /// `token_start == 0` decode. Slots below `token_start` are left untouched —
    /// here they stay at their reset zeros — which is what the sliding-window
    /// attention mask relies on.
    #[test]
    fn windowed_dequant_matches_full_in_window() {
        let (ctx, kernels) = gpu_setup();
        let device = ctx.device();
        let (head_dim, n_kv, max_pos) = (256usize, 2usize, 8usize);
        let mut cache = QuantizedKvCache::new(device, n_kv, head_dim, max_pos).expect("alloc");
        cache.reset();

        let seq_len = 6usize;
        for pos in 0..seq_len {
            let k = seeded_f16(n_kv * head_dim, 100 + pos as u64);
            let v = seeded_f16(n_kv * head_dim, 200 + pos as u64);
            cache.write_kv(pos, &k, &v);
        }

        let out_elems = seq_len * n_kv * head_dim;
        let full_k = MetalBuffer::empty(device, out_elems * 2).expect("full_k");
        let full_v = MetalBuffer::empty(device, out_elems * 2).expect("full_v");
        let win_k = MetalBuffer::empty(device, out_elems * 2).expect("win_k");
        let win_v = MetalBuffer::empty(device, out_elems * 2).expect("win_v");

        cache
            .dequantize_into(&ctx, &kernels, &full_k, &full_v, seq_len as u32, 0)
            .expect("full dequant");
        let token_start = 4u32;
        cache
            .dequantize_into(&ctx, &kernels, &win_k, &win_v, seq_len as u32, token_start)
            .expect("windowed dequant");

        let per_tok = n_kv * head_dim;
        let start_elem = token_start as usize * per_tok;
        for (full, win) in [(&full_k, &win_k), (&full_v, &win_v)] {
            let full: &[f16] = full.as_slice();
            let win: &[f16] = win.as_slice();
            // Active window: bit-identical to the full decode.
            for i in start_elem..out_elems {
                assert_eq!(
                    full[i].to_bits(),
                    win[i].to_bits(),
                    "windowed slot {i} must match full decode"
                );
            }
            // Below the window: never written, stays at reset zero.
            assert!(
                win[..start_elem].iter().all(|&v| v.to_f32() == 0.0),
                "slots below the window must be left untouched"
            );
        }
    }

    #[test]
    fn rejects_bad_config() {
        let device = get_default_device().expect("Metal device");
        // bits out of range
        assert!(QuantizedKvCache::new_with_bits(&device, 1, 256, 4, 5).is_err());
        assert!(QuantizedKvCache::new_with_bits(&device, 1, 256, 4, 1).is_err());
        // non-power-of-two head_dim
        assert!(QuantizedKvCache::new_with_bits(&device, 1, 192, 4, 4).is_err());
    }

    #[test]
    fn swa_ring_capacity_covers_window_plus_prefill_slack() {
        // Always at least window + the largest prefill chunk in flight, and
        // never larger than the logical context.
        let cap = swa_ring_capacity(512, 131_072);
        assert!(cap >= 512 + MAX_PREFILL_CHUNK - 1);
        assert!(cap.is_power_of_two());
        assert_eq!(cap, 1024);
        // Capped at the logical max for short contexts.
        assert_eq!(swa_ring_capacity(512, 600), 600);
    }

    #[test]
    fn ring_constructor_rejects_too_small_capacity() {
        let device = get_default_device().expect("Metal device");
        // cap below window + prefill slack must be refused.
        assert!(
            QuantizedKvCache::new_with_bits_and_ring(&device, 1, 256, 4096, 32, 512, 4).is_err()
        );
    }

    /// After enough single-token writes to wrap the ring several times, the
    /// in-window dequant of a ringed cache must be bit-identical to a full
    /// (absolute) cache holding the same token stream — proving the physical
    /// `position % capacity` slot mapping is consistent on both write and read.
    #[test]
    fn ring_dequant_in_window_matches_full_after_wrap() {
        let (ctx, kernels) = gpu_setup();
        let device = ctx.device();
        let (head_dim, n_kv, window) = (256usize, 2usize, 64usize);
        let cap = swa_ring_capacity(window, 4096); // 128
        let total = cap * 3 + 17; // wrap the ring three-plus times
        let max_pos = total + 1;

        let mut full =
            QuantizedKvCache::new_with_bits(device, n_kv, head_dim, max_pos, 4).expect("full");
        let mut ring = QuantizedKvCache::new_with_bits_and_ring(
            device, n_kv, head_dim, max_pos, cap, window, 4,
        )
        .expect("ring");
        full.reset();
        ring.reset();
        assert!(ring.is_ringed());
        assert_eq!(ring.physical_positions(), cap);

        for pos in 0..total {
            let k = seeded_f16(n_kv * head_dim, 1_000 + pos as u64);
            let v = seeded_f16(n_kv * head_dim, 2_000 + pos as u64);
            full.write_kv(pos, &k, &v);
            ring.write_kv(pos, &k, &v);
        }

        let seq_len = total as u32;
        let token_start = seq_len - window as u32;
        let out_elems = total * n_kv * head_dim;
        let full_k = MetalBuffer::empty(device, out_elems * 2).expect("fk");
        let full_v = MetalBuffer::empty(device, out_elems * 2).expect("fv");
        let ring_k = MetalBuffer::empty(device, out_elems * 2).expect("rk");
        let ring_v = MetalBuffer::empty(device, out_elems * 2).expect("rv");
        full.dequantize_into(&ctx, &kernels, &full_k, &full_v, seq_len, token_start)
            .expect("full dequant");
        ring.dequantize_into(&ctx, &kernels, &ring_k, &ring_v, seq_len, token_start)
            .expect("ring dequant");

        let per_tok = n_kv * head_dim;
        let start = token_start as usize * per_tok;
        for (f, r) in [(&full_k, &ring_k), (&full_v, &ring_v)] {
            let f: &[f16] = f.as_slice();
            let r: &[f16] = r.as_slice();
            for i in start..out_elems {
                assert_eq!(
                    f[i].to_bits(),
                    r[i].to_bits(),
                    "ringed dequant slot {i} must match full decode in window"
                );
            }
        }
    }

    /// The fused decode attention (the hot decode read path) over a wrapped
    /// ring must produce bit-identical output to the full cache for the same
    /// query and sliding window.
    #[test]
    fn ring_fused_attention_matches_full_after_wrap() {
        let (ctx, kernels) = gpu_setup();
        let device = ctx.device();
        let (head_dim, n_kv, n_q_heads, window) = (128usize, 1usize, 2usize, 64usize);
        let cap = swa_ring_capacity(window, 4096); // 128
        let total = cap * 2 + 9;
        let max_pos = total + 1;

        let mut full =
            QuantizedKvCache::new_with_bits(device, n_kv, head_dim, max_pos, 4).expect("full");
        let mut ring = QuantizedKvCache::new_with_bits_and_ring(
            device, n_kv, head_dim, max_pos, cap, window, 4,
        )
        .expect("ring");
        full.reset();
        ring.reset();

        for pos in 0..total {
            let k = seeded_f16(n_kv * head_dim, 5_000 + pos as u64);
            let v = seeded_f16(n_kv * head_dim, 6_000 + pos as u64);
            full.write_kv(pos, &k, &v);
            ring.write_kv(pos, &k, &v);
        }

        let q = seeded_f16(n_q_heads * head_dim, 9_999);
        let make_out = || -> (MetalBuffer, MetalBuffer, MetalBuffer, MetalBuffer) {
            let q_buf = MetalBuffer::from_slice(device, &q).expect("q");
            let rq = MetalBuffer::empty(device, n_q_heads * head_dim * 4).expect("rq");
            let vacc = MetalBuffer::empty(device, n_q_heads * head_dim * 4).expect("vacc");
            let out = MetalBuffer::empty(device, n_q_heads * head_dim * 2).expect("out");
            (q_buf, rq, vacc, out)
        };
        let current_pos = (total - 1) as u32;

        let run = |cache: &QuantizedKvCache| -> Vec<u16> {
            let (q_buf, rq, vacc, out) = make_out();
            let mut batch = local_metal::batch::CommandBatch::new(&ctx).expect("batch");
            cache
                .fused_attention_into(
                    &mut batch,
                    &kernels,
                    &q_buf,
                    &rq,
                    &vacc,
                    &out,
                    total as u32,
                    current_pos,
                    n_q_heads as u32,
                    window as u32,
                )
                .expect("fused attn");
            batch.commit_and_wait().expect("commit");
            out.as_slice::<f16>().iter().map(|v| v.to_bits()).collect()
        };

        assert_eq!(
            run(&full),
            run(&ring),
            "ringed fused attention must match full attention in the window"
        );
    }

    /// Prompt-cache snapshot/restore on a ringed cache is valid for any prefix
    /// up to the ring capacity (the pipeline never snapshots past it): the
    /// restored rows must be bit-identical to the originals.
    #[test]
    fn ring_snapshot_restore_unwrapped_prefix() {
        let device = get_default_device().expect("Metal device");
        let (head_dim, n_kv, window) = (256usize, 1usize, 64usize);
        let cap = swa_ring_capacity(window, 4096); // 128
        let mut cache =
            QuantizedKvCache::new_with_bits_and_ring(&device, n_kv, head_dim, 4096, cap, window, 4)
                .expect("ring");
        cache.reset();
        assert_eq!(cache.max_snapshot_positions(), cap);

        let prefix = cap - 5;
        for pos in 0..prefix {
            let k = seeded_f16(n_kv * head_dim, 70 + pos as u64);
            let v = seeded_f16(n_kv * head_dim, 80 + pos as u64);
            cache.write_kv(pos, &k, &v);
        }
        let snapshot = cache.snapshot_prefix(prefix);
        let saved = cache.keys_q().as_slice::<u8>()[..prefix * cache.code_row_bytes()].to_vec();

        cache.reset();
        cache.restore_prefix(&snapshot, prefix);
        assert_eq!(
            &cache.keys_q().as_slice::<u8>()[..prefix * cache.code_row_bytes()],
            &saved[..],
            "ring snapshot/restore must round-trip an unwrapped prefix"
        );
    }
}
