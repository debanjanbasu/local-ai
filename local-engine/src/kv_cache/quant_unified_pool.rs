//! Unified by-lane `TurboQuant` KV pool for continuous batching (decoding N
//! independent sequences in one batched forward).
//!
//! This is the quantized analogue of the FP16 unified pool: where a
//! single-sequence [`super::QuantizedKvCache`] stores one sequence's packed
//! codes as `[max_positions, n_kv_heads, code_bytes]`, this pool stores
//! `n_lanes` sequences as `[n_lanes, lane_capacity, n_kv_heads, code_bytes]` —
//! lane `l` owns the contiguous token region `[l*lane_capacity,
//! (l+1)*lane_capacity)`. A single GPU `positions` buffer carries each lane's
//! current absolute position so the batched encode
//! ([`Kernels::encode_kv_turboquant_batched_into`]) and fused attention
//! ([`Kernels::flash_attention_tq_batched_into`]) kernels address every lane in
//! one dispatch, while the weight-heavy projections/FFN run once at M=`n_lanes`.
//!
//! Each lane's freshly projected K/V is encoded straight into `bits`-bit codes
//! on write and the attention reads those codes directly (no FP16 expansion),
//! so the resident KV footprint is ~4× smaller than the FP16 pool — the whole
//! point of running serve on the `TurboQuant` backing: maximum concurrent context
//! on memory-constrained devices.

use local_metal::buffer::MetalBuffer;
use local_metal::kernels::Kernels;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::quant_cache::{QuantizedKvCache, TQ_SEED};
use crate::Result;
use crate::turboquant::TurboQuant;

const F16: usize = std::mem::size_of::<half::f16>();

/// One layer's unified `TurboQuant` KV pool across `n_lanes` decode lanes.
pub struct QuantizedUnifiedKvPool {
    keys_q: MetalBuffer,
    values_q: MetalBuffer,
    key_norms: MetalBuffer,
    value_norms: MetalBuffer,
    levels: MetalBuffer,
    signs: MetalBuffer,
    bits: u8,
    code_bytes: usize,
    n_lanes: usize,
    lane_capacity: usize,
    /// Sliding-window ring capacity within each lane (`0` = absolute addressing
    /// over the whole `lane_capacity`). When non-zero, logical position `p` of a
    /// lane maps to physical row `p % lane_ring_capacity`, so a sliding-window
    /// layer only ever touches `lane_ring_capacity` rows per lane.
    lane_ring_capacity: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl QuantizedUnifiedKvPool {
    /// Allocate a pool for `n_lanes` sequences, each up to `lane_capacity`
    /// tokens, at `bits` per coordinate.
    ///
    /// The codec parameters mirror [`QuantizedKvCache`] exactly (same seed,
    /// bits, `head_dim`) so codes copied from a single-sequence prefill staging
    /// cache via [`Self::copy_prefill_from`] stay consistent with codes written
    /// by the batched decode encode kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the device buffers cannot be allocated.
    /// `lane_ring_capacity` rings each lane's KV at the given physical row count
    /// (sliding-window layers); pass `0` for absolute full-attention addressing.
    /// When non-zero it must be `<= lane_capacity`; each lane physically uses
    /// only its first `lane_ring_capacity` rows.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n_lanes: usize,
        lane_capacity: usize,
        n_kv_heads: usize,
        head_dim: usize,
        bits: u8,
        lane_ring_capacity: usize,
    ) -> Result<Self> {
        let lane_ring_capacity = lane_ring_capacity.min(lane_capacity);
        // A ringed lane only needs its first `lane_ring_capacity` rows resident;
        // an absolute lane needs the whole `lane_capacity`.
        let lane_rows = if lane_ring_capacity == 0 {
            lane_capacity
        } else {
            lane_ring_capacity
        };
        let codec = TurboQuant::new(head_dim, bits, TQ_SEED);
        let code_bytes = codec.code_bytes();
        let slots = (n_lanes * lane_rows * n_kv_heads).max(1);
        Ok(Self {
            keys_q: MetalBuffer::empty(device, slots * code_bytes)?,
            values_q: MetalBuffer::empty(device, slots * code_bytes)?,
            key_norms: MetalBuffer::empty(device, slots * F16)?,
            value_norms: MetalBuffer::empty(device, slots * F16)?,
            levels: MetalBuffer::from_slice(device, codec.levels())?,
            signs: MetalBuffer::from_slice(device, codec.signs())?,
            bits,
            code_bytes,
            n_lanes,
            lane_capacity,
            lane_ring_capacity,
            n_kv_heads,
            head_dim,
        })
    }

    /// Per-lane token capacity.
    #[must_use]
    pub const fn lane_capacity(&self) -> usize {
        self.lane_capacity
    }

    /// Physical rows allocated per lane: the ring capacity for a sliding-window
    /// pool, else the full `lane_capacity`. This is the lane base stride handed
    /// to the batched kernels.
    #[must_use]
    const fn lane_stride(&self) -> usize {
        if self.lane_ring_capacity == 0 {
            self.lane_capacity
        } else {
            self.lane_ring_capacity
        }
    }

    /// Number of lanes this pool backs.
    #[must_use]
    pub const fn n_lanes(&self) -> usize {
        self.n_lanes
    }

    /// Encode + scatter one freshly projected K/V row per lane (`k_src`/`v_src`
    /// are `[n_lanes, n_kv_heads * head_dim]` f16) into the pool at each lane's
    /// `positions[lane]`. `active_lanes` may be `<= n_lanes` to write only the
    /// leading lanes.
    ///
    /// # Errors
    ///
    /// Returns an error if the encode kernel is unavailable.
    pub fn write_kv_into(
        &self,
        batch: &mut local_metal::batch::CommandBatch,
        kernels: &Kernels,
        k_src: &MetalBuffer,
        v_src: &MetalBuffer,
        positions: &MetalBuffer,
        active_lanes: usize,
    ) -> Result<()> {
        let hd = u32::try_from(self.head_dim)
            .map_err(|_| crate::Error::InvalidArgument("head_dim overflow".into()))?;
        let n_kv = u32::try_from(self.n_kv_heads)
            .map_err(|_| crate::Error::InvalidArgument("n_kv_heads overflow".into()))?;
        let cap = u32::try_from(self.lane_stride())
            .map_err(|_| crate::Error::InvalidArgument("lane stride overflow".into()))?;
        let ring = u32::try_from(self.lane_ring_capacity)
            .map_err(|_| crate::Error::InvalidArgument("lane ring overflow".into()))?;
        let lanes = u32::try_from(active_lanes.min(self.n_lanes))
            .map_err(|_| crate::Error::InvalidArgument("lane count overflow".into()))?;
        let bits = u32::from(self.bits);
        for (src, codes, norms) in [
            (k_src, &self.keys_q, &self.key_norms),
            (v_src, &self.values_q, &self.value_norms),
        ] {
            kernels
                .encode_kv_turboquant_batched_into(
                    batch,
                    src,
                    &self.levels,
                    &self.signs,
                    codes,
                    norms,
                    positions,
                    hd,
                    n_kv,
                    cap,
                    bits,
                    lanes,
                    ring,
                )
                .map_err(crate::Error::Metal)?;
        }
        Ok(())
    }

    /// Fused per-lane attention: rotate each lane's `RoPE`'d query into codebook
    /// space (`q` → `rq`), attend over the packed codes
    /// ([`Kernels::flash_attention_tq_batched_into`], output `vacc` in rotated
    /// space), then inverse-rotate back to model space (`vacc` → `attn_out`).
    /// `q`/`attn_out` are `[n_lanes, n_q_heads, head_dim]` f16 and `rq`/`vacc`
    /// the f32 scratch in the same layout.
    ///
    /// # Errors
    ///
    /// Returns an error if any kernel pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_into(
        &self,
        batch: &mut local_metal::batch::CommandBatch,
        kernels: &Kernels,
        q: &MetalBuffer,
        rq: &MetalBuffer,
        vacc: &MetalBuffer,
        attn_out: &MetalBuffer,
        positions: &MetalBuffer,
        n_q_heads: u32,
        window: u32,
        active_lanes: usize,
    ) -> Result<()> {
        let hd = u32::try_from(self.head_dim)
            .map_err(|_| crate::Error::InvalidArgument("head_dim overflow".into()))?;
        let n_kv = u32::try_from(self.n_kv_heads)
            .map_err(|_| crate::Error::InvalidArgument("n_kv_heads overflow".into()))?;
        let cap = u32::try_from(self.lane_stride())
            .map_err(|_| crate::Error::InvalidArgument("lane stride overflow".into()))?;
        let ring = u32::try_from(self.lane_ring_capacity)
            .map_err(|_| crate::Error::InvalidArgument("lane ring overflow".into()))?;
        let lanes = u32::try_from(active_lanes.min(self.n_lanes))
            .map_err(|_| crate::Error::InvalidArgument("lane count overflow".into()))?;
        // One rotation per (lane, q-head): the rotate kernel is head-major and
        // the buffers are laid out [lane, head, head_dim], so the lanes' heads
        // are simply concatenated.
        let total_heads = lanes * n_q_heads;
        kernels
            .hadamard_rotate_hf_into(batch, q, &self.signs, rq, hd, total_heads, true, false)
            .map_err(crate::Error::Metal)?;
        kernels
            .flash_attention_tq_batched_into(
                batch,
                rq,
                &self.keys_q,
                &self.key_norms,
                &self.values_q,
                &self.value_norms,
                &self.levels,
                vacc,
                positions,
                hd,
                cap,
                n_q_heads,
                n_kv,
                window,
                u32::from(self.bits),
                lanes,
                ring,
            )
            .map_err(crate::Error::Metal)?;
        kernels
            .hadamard_rotate_fh_into(
                batch,
                vacc,
                &self.signs,
                attn_out,
                hd,
                total_heads,
                false,
                true,
            )
            .map_err(crate::Error::Metal)?;
        Ok(())
    }

    /// Seed `lane` from a single-sequence prefill staging cache: copy the codes
    /// and norms for positions `[0..positions)` into the lane's pool region. The
    /// staging cache must share this pool's codec parameters (it does — both use
    /// [`TQ_SEED`] and the same `bits`/`head_dim`), so the copied codes are
    /// directly consumable by the batched attention kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if `positions` exceeds `lane_capacity`.
    pub fn copy_prefill_from(
        &self,
        cache: &QuantizedKvCache,
        lane: usize,
        positions: usize,
    ) -> Result<()> {
        if positions > self.lane_capacity {
            return Err(crate::Error::InvalidArgument(format!(
                "prefill {positions} positions exceeds lane_capacity {}",
                self.lane_capacity
            )));
        }
        if lane >= self.n_lanes || positions == 0 {
            return Ok(());
        }
        // A ringed lane and its ringed staging cache share the same modulus, so
        // physical slot `s` holds the same logical token in both; once the
        // prefill has wrapped we copy the whole physical ring rather than the
        // (now-aliased) logical prefix.
        debug_assert!(
            self.lane_ring_capacity == 0 || self.lane_ring_capacity == cache.physical_positions(),
            "pool ring {} must match staging cache ring {}",
            self.lane_ring_capacity,
            cache.physical_positions()
        );
        let copy_positions = if self.lane_ring_capacity == 0 {
            positions
        } else {
            positions.min(self.lane_ring_capacity)
        };
        let rows = copy_positions * self.n_kv_heads;
        let code_len = rows * self.code_bytes;
        let norm_len = rows * F16;
        let lane_row0 = lane * self.lane_stride() * self.n_kv_heads;
        let code_off = lane_row0 * self.code_bytes;
        let norm_off = lane_row0 * F16;
        self.keys_q
            .copy_from_bytes(&cache.keys_q().as_slice::<u8>()[..code_len], code_off);
        self.values_q
            .copy_from_bytes(&cache.values_q().as_slice::<u8>()[..code_len], code_off);
        self.key_norms
            .copy_from_bytes(&cache.key_norms().as_slice::<u8>()[..norm_len], norm_off);
        self.value_norms
            .copy_from_bytes(&cache.value_norms().as_slice::<u8>()[..norm_len], norm_off);
        Ok(())
    }

    /// Zero the whole pool (recycling all lanes). A zero norm annihilates
    /// whatever the codes say, so any stale slot reads back as exact zero.
    pub fn reset(&self) {
        for buf in [
            &self.keys_q,
            &self.values_q,
            &self.key_norms,
            &self.value_norms,
        ] {
            let zeros = vec![0u8; buf.length()];
            buf.copy_from_bytes(&zeros, 0);
        }
    }

    /// Zero a single lane's region (when a finished request's lane is recycled).
    pub fn reset_lane(&self, lane: usize) {
        if lane >= self.n_lanes {
            return;
        }
        let rows = self.lane_stride() * self.n_kv_heads;
        let lane_row0 = lane * rows;
        for (buf, stride) in [
            (&self.keys_q, self.code_bytes),
            (&self.values_q, self.code_bytes),
            (&self.key_norms, F16),
            (&self.value_norms, F16),
        ] {
            let zeros = vec![0u8; rows * stride];
            buf.copy_from_bytes(&zeros, lane_row0 * stride);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::kv_cache::swa_ring_capacity;
    use half::f16;
    use local_metal::batch::CommandBatch;
    use local_metal::context::MetalContext;
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

    /// A ringed lane pool decoded after wrapping a lane several times must match
    /// an absolute (non-ring) pool holding the same per-lane token stream — for
    /// every lane, proving the batched `lane_base + position % ring` addressing
    /// is consistent across encode and attention.
    #[test]
    fn batched_ring_pool_matches_full_after_lane_wrap() {
        let ctx = MetalContext::new().expect("ctx");
        let shaders = ShaderLibrary::new(ctx.device()).expect("shaders");
        let kernels = Kernels::new(&ctx, &shaders).expect("kernels");
        let device = ctx.device();

        let (n_lanes, n_kv, head_dim, n_q_heads, window) = (2usize, 1usize, 128usize, 2usize, 64);
        let cap = swa_ring_capacity(window, 4096); // 128
        let lane_capacity = 4096usize;
        let total = cap * 2 + 11; // wrap each lane twice-plus

        let full =
            QuantizedUnifiedKvPool::new(device, n_lanes, lane_capacity, n_kv, head_dim, 4, 0)
                .expect("full pool");
        let ring =
            QuantizedUnifiedKvPool::new(device, n_lanes, lane_capacity, n_kv, head_dim, 4, cap)
                .expect("ring pool");
        full.reset();
        ring.reset();

        let pos_buf = MetalBuffer::empty(device, n_lanes * 4).expect("pos");
        for pos in 0..total {
            // Distinct K/V per (lane, position) so a wrong slot would diverge.
            let mut k = Vec::with_capacity(n_lanes * n_kv * head_dim);
            let mut v = Vec::with_capacity(n_lanes * n_kv * head_dim);
            for lane in 0..n_lanes {
                k.extend(seeded_f16(n_kv * head_dim, 1 + (pos * 7 + lane) as u64));
                v.extend(seeded_f16(n_kv * head_dim, 9_001 + (pos * 7 + lane) as u64));
            }
            let k_src = MetalBuffer::from_slice(device, &k).expect("k");
            let v_src = MetalBuffer::from_slice(device, &v).expect("v");
            let positions: Vec<u32> = vec![pos as u32; n_lanes];
            pos_buf.copy_from_bytes(bytemuck::cast_slice(&positions), 0);
            for pool in [&full, &ring] {
                let mut batch = CommandBatch::new(&ctx).expect("batch");
                pool.write_kv_into(&mut batch, &kernels, &k_src, &v_src, &pos_buf, n_lanes)
                    .expect("write");
                batch.commit_and_wait().expect("commit");
            }
        }

        // Decode attention at the last position for every lane.
        let positions: Vec<u32> = vec![(total - 1) as u32; n_lanes];
        pos_buf.copy_from_bytes(bytemuck::cast_slice(&positions), 0);
        let q = seeded_f16(n_lanes * n_q_heads * head_dim, 4_242);

        let run = |pool: &QuantizedUnifiedKvPool| -> Vec<u16> {
            let q_buf = MetalBuffer::from_slice(device, &q).expect("q");
            let rq = MetalBuffer::empty(device, n_lanes * n_q_heads * head_dim * 4).expect("rq");
            let vacc =
                MetalBuffer::empty(device, n_lanes * n_q_heads * head_dim * 4).expect("vacc");
            let out = MetalBuffer::empty(device, n_lanes * n_q_heads * head_dim * 2).expect("out");
            let mut batch = CommandBatch::new(&ctx).expect("batch");
            pool.attention_into(
                &mut batch,
                &kernels,
                &q_buf,
                &rq,
                &vacc,
                &out,
                &pos_buf,
                n_q_heads as u32,
                window as u32,
                n_lanes,
            )
            .expect("attn");
            batch.commit_and_wait().expect("commit");
            out.as_slice::<f16>().iter().map(|v| v.to_bits()).collect()
        };

        assert_eq!(
            run(&full),
            run(&ring),
            "batched ring pool attention must match full pool after lane wrap"
        );
    }
}
