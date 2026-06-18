#include <metal_stdlib>
using namespace metal;

// Fused TurboQuant flash-attention decode: attention logits and the value
// accumulation are computed directly from the packed 3-4 bit codes, so the
// keys/values are never expanded to FP16. Decode attention is memory-bandwidth
// bound (it re-reads the whole KV cache every token), so reading 4-bit codes
// instead of 16-bit values is ~4x less traffic — the source of the speedup the
// TurboQuant paper reports for attention-logit computation.
//
// Math (per head, query q, key/value vector reconstructed as
//   x = norm · sign ⊙ (1/√d)·H(levels[codes])  where H is the Walsh-Hadamard
//   butterfly transform):
//
//   ⟨q, k⟩ = norm_k · Σ_i (Rq)_i · levels[Kcode_i],   Rq = (1/√d)·H(sign⊙q)
//   attn_out = R⁻¹( Σ_t p_t · norm_vt · levels[Vcode_t] ),  R⁻¹(y)=sign⊙(1/√d)·H(y)
//
// so the query is rotated once up front, the value accumulation happens in the
// rotated space, and a single inverse rotation is applied at the end. Both
// rotations are handled by `hadamard_rotate`; this kernel does the bandwidth-
// bound middle stage.

#define FLASH_TQ_MAX_DPL 16
#define FLASH_TQ_NSG 8
#define FLASH_TQ_MAX_HEAD_DIM 512
// TurboQuant uses 2..=4 bits, so the reconstruction table has at most 1<<4=16
// entries. It is re-read for every dimension of every cached position, so it is
// staged into threadgroup memory once per dispatch; otherwise each lookup is an
// uncoalesced, data-dependent load from device memory and dominates the
// (otherwise bandwidth-bound) decode loop, getting worse the longer the context.
#define FLASH_TQ_MAX_LEVELS 16

inline uint tq_unpack_code(
    device const uchar *codes,
    uint slot,
    uint d,
    uint bits,
    uint code_bytes
) {
    uint bit_off = d * bits;
    uint byte_idx = slot * code_bytes + (bit_off >> 3);
    uint shift = bit_off & 7u;
    uint v = uint(codes[byte_idx]);
    if (shift + bits > 8u) {
        v |= uint(codes[byte_idx + 1u]) << 8u;
    }
    return (v >> shift) & ((1u << bits) - 1u);
}

/// In-place randomized Hadamard rotation per head, one threadgroup per head,
/// `head_dim` threads. `pre_sign` multiplies by the rotation signs before the
/// butterflies (forward rotation R), `post_sign` after (inverse rotation R⁻¹);
/// both scale by 1/√d. Used to rotate the query (pre_sign=1, post_sign=0) and
/// to inverse-rotate the accumulated value (pre_sign=0, post_sign=1). The
/// templated I/O types let the query enter as f16 and the attention output
/// leave as f16 without an extra conversion pass, while the rotated query /
/// value accumulation stay f32 for the fused-attention kernel.
template <typename TIn, typename TOut>
inline void hadamard_rotate_impl(
    device const TIn  *input,
    device const float *signs,
    device TOut       *output,
    threadgroup float *sh,
    uint head_dim,
    uint num_heads,
    uint pre_sign,
    uint post_sign,
    uint head,
    uint lane
) {
    const uint d = head_dim;
    if (head >= num_heads) return;

    float x = float(input[head * d + lane]);
    sh[lane] = (pre_sign != 0u) ? (x * signs[lane]) : x;
    for (uint len = 1u; len < d; len <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float self_v = sh[lane];
        float other_v = sh[lane ^ len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[lane] = (lane & len) ? (other_v - self_v) : (self_v + other_v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = sh[lane] * rsqrt(float(d));
    if (post_sign != 0u) {
        y *= signs[lane];
    }
    output[head * d + lane] = TOut(y);
}

kernel void hadamard_rotate(
    device const float *input   [[buffer(0)]],
    device const float *signs   [[buffer(1)]],
    device float       *output  [[buffer(2)]],
    constant uint      &head_dim[[buffer(3)]],
    constant uint      &num_heads[[buffer(4)]],
    constant uint      &pre_sign[[buffer(5)]],
    constant uint      &post_sign[[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[FLASH_TQ_MAX_HEAD_DIM];
    hadamard_rotate_impl<float, float>(
        input, signs, output, sh, head_dim, num_heads, pre_sign, post_sign, head, lane);
}

/// f16 input → f32 output (rotate the RoPE'd query into the fused kernel).
kernel void hadamard_rotate_hf(
    device const half  *input   [[buffer(0)]],
    device const float *signs   [[buffer(1)]],
    device float       *output  [[buffer(2)]],
    constant uint      &head_dim[[buffer(3)]],
    constant uint      &num_heads[[buffer(4)]],
    constant uint      &pre_sign[[buffer(5)]],
    constant uint      &post_sign[[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[FLASH_TQ_MAX_HEAD_DIM];
    hadamard_rotate_impl<half, float>(
        input, signs, output, sh, head_dim, num_heads, pre_sign, post_sign, head, lane);
}

/// f32 input → f16 output (inverse-rotate the value accumulation to attn_out).
kernel void hadamard_rotate_fh(
    device const float *input   [[buffer(0)]],
    device const float *signs   [[buffer(1)]],
    device half        *output  [[buffer(2)]],
    constant uint      &head_dim[[buffer(3)]],
    constant uint      &num_heads[[buffer(4)]],
    constant uint      &pre_sign[[buffer(5)]],
    constant uint      &post_sign[[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[FLASH_TQ_MAX_HEAD_DIM];
    hadamard_rotate_impl<float, half>(
        input, signs, output, sh, head_dim, num_heads, pre_sign, post_sign, head, lane);
}

/// Fused TurboQuant flash-attention decode. Same barrier-free split-K structure
/// as `flash_attention` (one threadgroup per Q head, `FLASH_TQ_NSG` simdgroups
/// each owning a strided KV stripe), but K/V are read as packed codes and the
/// codebook `levels` are applied inline. `rq` is the pre-rotated query; the
/// output is the value accumulation in rotated space (apply `hadamard_rotate`
/// with post_sign to recover the attention output).
///
/// rq:      [num_q_heads * head_dim]    f32  (rotated query)
/// k_codes: [kv_len * num_kv_heads * code_bytes] uchar
/// k_norms: [kv_len * num_kv_heads]     f16
/// v_codes, v_norms: as K
/// levels:  [1 << bits]                 float (pre-scaled by 1/√d on the host)
/// out:     [num_q_heads * head_dim]    f32  (rotated-space value accumulation)
kernel void flash_attention_tq(
    device const float *rq        [[buffer(0)]],
    device const uchar *k_codes   [[buffer(1)]],
    device const half  *k_norms   [[buffer(2)]],
    device const uchar *v_codes   [[buffer(3)]],
    device const half  *v_norms   [[buffer(4)]],
    device const float *levels    [[buffer(5)]],
    device float       *out       [[buffer(6)]],
    constant uint      &head_dim  [[buffer(7)]],
    constant uint      &kv_len    [[buffer(8)]],
    constant uint      &current_pos[[buffer(9)]],
    constant uint      &num_q_heads[[buffer(10)]],
    constant uint      &num_kv_heads[[buffer(11)]],
    constant uint      &window    [[buffer(12)]],
    constant uint      &bits      [[buffer(13)]],
    constant uint      &code_bytes[[buffer(14)]],
    constant uint      &ring_capacity[[buffer(15)]],
    uint q_head  [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    if (q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint seq_len = min(kv_len, current_pos + 1u);
    uint win_start = (window > 0u && current_pos + 1u > window) ? (current_pos + 1u - window) : 0u;
    uint dpl = (head_dim + 31u) / 32u;
    uint q_base = q_head * head_dim;

    // Stage the reconstruction table into threadgroup memory (see note above).
    threadgroup float tg_levels[FLASH_TQ_MAX_LEVELS];
    uint nlev = 1u << bits;
    for (uint i = simd_id * 32u + lane; i < nlev; i += FLASH_TQ_NSG * 32u) {
        tg_levels[i] = levels[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This lane's slice of the rotated query.
    float rq_reg[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        rq_reg[i] = (d < head_dim) ? rq[q_base + d] : 0.0f;
    }

    float acc[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = win_start + simd_id; t < seq_len; t += FLASH_TQ_NSG) {
        // Logical position `t` maps to a physical ring slot when ring_capacity
        // != 0 (sliding-window layers allocate only `ring_capacity` rows); the
        // window guarantees the live span never overwrites a slot still needed.
        uint physical_t = (ring_capacity != 0u) ? (t % ring_capacity) : t;
        uint slot = physical_t * num_kv_heads + kv_head;

        // Logit = norm_k · Σ_i (Rq)_i · levels[Kcode_i].
        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d < head_dim) {
                uint code = tq_unpack_code(k_codes, slot, d, bits, code_bytes);
                dot += rq_reg[i] * tg_levels[code];
            }
        }
        dot = simd_sum(dot);
        dot *= float(k_norms[slot]);

        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;

        // Accumulate value in rotated space: acc_i += w · norm_v · levels[Vcode_i].
        float vnorm = float(v_norms[slot]);
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            float vv = 0.0f;
            if (d < head_dim) {
                uint code = tq_unpack_code(v_codes, slot, d, bits, code_bytes);
                vv = tg_levels[code] * vnorm;
            }
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    threadgroup float tg_m[FLASH_TQ_NSG];
    threadgroup float tg_l[FLASH_TQ_NSG];
    threadgroup float tg_acc[FLASH_TQ_NSG * FLASH_TQ_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        if (d < head_dim) { tg_acc[simd_id * FLASH_TQ_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_TQ_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float gsum = 0.0f;
        float factor[FLASH_TQ_NSG];
        for (uint s = 0; s < FLASH_TQ_NSG; s++) {
            factor[s] = exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_TQ_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_TQ_MAX_HEAD_DIM + d];
            }
            out[q_base + d] = o * inv_sum;
        }
    }
}

/// Fused TurboQuant flash-attention PREFILL / multi-row verify. Identical
/// online-softmax + codebook math as `flash_attention_tq`, but the grid is
/// `(num_q_heads * n_rows)`: threadgroup `tg_id` handles query row
/// `row = tg_id / num_q_heads` (at absolute position `start_pos + row`) for head
/// `q_head = tg_id % num_q_heads`. Each row applies its own causal cutoff
/// (`seq_len = start_pos + row + 1`) and sliding window, reading the packed K/V
/// codes of the just-written batch straight from the global ring cache (same
/// `slot = physical_t * num_kv_heads + kv_head` addressing as the decode kernel),
/// so the cache is never expanded to FP16. The ring is sized
/// `window + MAX_PREFILL_CHUNK - 1`, so within one ≤64-row chunk a later row
/// never overwrites a slot an earlier row still needs.
///
/// rq:  [n_rows * num_q_heads * head_dim] f32 (rotated queries, row-major)
/// out: [n_rows * num_q_heads * head_dim] f32 (rotated-space value accumulation)
kernel void flash_attention_tq_prefill(
    device const float *rq        [[buffer(0)]],
    device const uchar *k_codes   [[buffer(1)]],
    device const half  *k_norms   [[buffer(2)]],
    device const uchar *v_codes   [[buffer(3)]],
    device const half  *v_norms   [[buffer(4)]],
    device const float *levels    [[buffer(5)]],
    device float       *out       [[buffer(6)]],
    constant uint      &head_dim  [[buffer(7)]],
    constant uint      &kv_len    [[buffer(8)]],
    constant uint      &start_pos [[buffer(9)]],
    constant uint      &n_rows    [[buffer(10)]],
    constant uint      &num_q_heads[[buffer(11)]],
    constant uint      &num_kv_heads[[buffer(12)]],
    constant uint      &window    [[buffer(13)]],
    constant uint      &bits      [[buffer(14)]],
    constant uint      &code_bytes[[buffer(15)]],
    constant uint      &ring_capacity[[buffer(16)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint q_head = tg_id % num_q_heads;
    uint row    = tg_id / num_q_heads;
    if (row >= n_rows || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint current_pos = start_pos + row;
    uint seq_len = min(kv_len, current_pos + 1u);
    uint win_start = (window > 0u && current_pos + 1u > window) ? (current_pos + 1u - window) : 0u;
    uint dpl = (head_dim + 31u) / 32u;
    uint q_base = (row * num_q_heads + q_head) * head_dim;

    threadgroup float tg_levels[FLASH_TQ_MAX_LEVELS];
    uint nlev = 1u << bits;
    for (uint i = simd_id * 32u + lane; i < nlev; i += FLASH_TQ_NSG * 32u) {
        tg_levels[i] = levels[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rq_reg[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        rq_reg[i] = (d < head_dim) ? rq[q_base + d] : 0.0f;
    }

    float acc[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = win_start + simd_id; t < seq_len; t += FLASH_TQ_NSG) {
        uint physical_t = (ring_capacity != 0u) ? (t % ring_capacity) : t;
        uint slot = physical_t * num_kv_heads + kv_head;

        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d < head_dim) {
                uint code = tq_unpack_code(k_codes, slot, d, bits, code_bytes);
                dot += rq_reg[i] * tg_levels[code];
            }
        }
        dot = simd_sum(dot);
        dot *= float(k_norms[slot]);

        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;

        float vnorm = float(v_norms[slot]);
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            float vv = 0.0f;
            if (d < head_dim) {
                uint code = tq_unpack_code(v_codes, slot, d, bits, code_bytes);
                vv = tg_levels[code] * vnorm;
            }
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    threadgroup float tg_m[FLASH_TQ_NSG];
    threadgroup float tg_l[FLASH_TQ_NSG];
    threadgroup float tg_acc[FLASH_TQ_NSG * FLASH_TQ_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        if (d < head_dim) { tg_acc[simd_id * FLASH_TQ_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_TQ_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float gsum = 0.0f;
        float factor[FLASH_TQ_NSG];
        for (uint s = 0; s < FLASH_TQ_NSG; s++) {
            factor[s] = exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_TQ_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_TQ_MAX_HEAD_DIM + d];
            }
            out[q_base + d] = o * inv_sum;
        }
    }
}

/// Query-TILED fused TurboQuant flash-attention prefill. Same per-row causal
/// math and rotated-space output as `flash_attention_tq_prefill`, but each K/V
/// code tile is dequantized into threadgroup memory ONCE and reused across all
/// `FLASH_TQ_TILED_BR` query rows in the block (one simdgroup per row). The
/// non-tiled kernel re-reads every K/V code from device memory for every query
/// row; for long prompts that bandwidth dominates, so tiling cuts the
/// quadratic attention cost by ~BR×. Codes are dequantized as
/// `tile = norm · levels[code]`, folding the per-head norm into the tile so the
/// inner loop is a plain `dot += rq · K_tile` / `acc += w · V_tile`.
///
/// Tile holds `FLASH_TQ_TILE_HALFS` halfs each for K and V (16 KiB total):
/// BK keys = FLASH_TQ_TILE_HALFS / head_dim. Dispatch:
/// threadgroups = (ceil(n_rows/BR) * num_q_heads), threads = BR*32.
#define FLASH_TQ_TILED_BR 8
#define FLASH_TQ_TILE_HALFS 4096

kernel void flash_attention_tq_prefill_tiled(
    device const float *rq        [[buffer(0)]],
    device const uchar *k_codes   [[buffer(1)]],
    device const half  *k_norms   [[buffer(2)]],
    device const uchar *v_codes   [[buffer(3)]],
    device const half  *v_norms   [[buffer(4)]],
    device const float *levels    [[buffer(5)]],
    device float       *out       [[buffer(6)]],
    constant uint      &head_dim  [[buffer(7)]],
    constant uint      &kv_len    [[buffer(8)]],
    constant uint      &start_pos [[buffer(9)]],
    constant uint      &n_rows    [[buffer(10)]],
    constant uint      &num_q_heads[[buffer(11)]],
    constant uint      &num_kv_heads[[buffer(12)]],
    constant uint      &window    [[buffer(13)]],
    constant uint      &bits      [[buffer(14)]],
    constant uint      &code_bytes[[buffer(15)]],
    constant uint      &ring_capacity[[buffer(16)]],
    uint gid     [[threadgroup_position_in_grid]],
    uint tcount  [[threads_per_threadgroup]],
    uint tid     [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup half K_tile[FLASH_TQ_TILE_HALFS];
    threadgroup half V_tile[FLASH_TQ_TILE_HALFS];
    threadgroup float tg_levels[FLASH_TQ_MAX_LEVELS];

    uint total_q_blocks = (n_rows + FLASH_TQ_TILED_BR - 1u) / FLASH_TQ_TILED_BR;
    uint q_block = gid % total_q_blocks;
    uint head    = gid / total_q_blocks;
    if (head >= num_q_heads) return;

    uint kv_head = head / (num_q_heads / num_kv_heads);
    uint q_start = q_block * FLASH_TQ_TILED_BR;
    uint dpl = (head_dim + 31u) / 32u;
    uint bk = FLASH_TQ_TILE_HALFS / head_dim; // keys per tile

    uint nlev = 1u << bits;
    for (uint i = tid; i < nlev; i += tcount) {
        tg_levels[i] = levels[i];
    }

    // This simdgroup owns one query row of the block.
    uint qi = simd_id;
    uint q_row = q_start + qi;
    bool active = (q_row < n_rows);
    uint q_global = start_pos + q_row;          // absolute position
    uint q_base = (q_row * num_q_heads + head) * head_dim;

    float q_vals[FLASH_TQ_MAX_DPL];
    float acc[FLASH_TQ_MAX_DPL];
    for (uint j = 0; j < dpl; j++) { q_vals[j] = 0.0f; acc[j] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    uint eff_kv = active ? min(kv_len, q_global + 1u) : 0u;
    uint row_win_start = (window > 0u && q_global + 1u > window) ? (q_global + 1u - window) : 0u;
    if (active) {
        for (uint j = 0; j < dpl; j++) {
            uint d = simd_lane + j * 32u;
            q_vals[j] = (d < head_dim) ? rq[q_base + d] : 0.0f;
        }
    }

    // Uniform block-level key range (so every thread hits the same barriers):
    // the highest absolute position in this block, and the earliest key the
    // first row of the block can attend under the sliding window.
    uint active_rows = min(uint(FLASH_TQ_TILED_BR), n_rows - q_start);
    uint block_hi = min(kv_len, start_pos + q_start + active_rows);
    uint first_pos = start_pos + q_start;
    uint block_lo = (window > 0u && first_pos + 1u > window) ? (first_pos + 1u - window) : 0u;
    uint tile0 = (block_lo / bk) * bk; // align down so cooperative load is simple

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile_start = tile0; tile_start < block_hi; tile_start += bk) {
        uint tile_n = min(bk, kv_len - tile_start);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Cooperatively dequantize this K/V code tile into threadgroup memory,
        // folding the per-slot norm into the value: tile = norm · levels[code].
        for (uint idx = tid; idx < tile_n * head_dim; idx += tcount) {
            uint kk = idx / head_dim;
            uint dd = idx - kk * head_dim;
            uint t = tile_start + kk;
            uint physical_t = (ring_capacity != 0u) ? (t % ring_capacity) : t;
            uint slot = physical_t * num_kv_heads + kv_head;
            uint kc = tq_unpack_code(k_codes, slot, dd, bits, code_bytes);
            uint vc = tq_unpack_code(v_codes, slot, dd, bits, code_bytes);
            K_tile[idx] = half(float(k_norms[slot]) * tg_levels[kc]);
            V_tile[idx] = half(float(v_norms[slot]) * tg_levels[vc]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint k_lo = max(tile_start, row_win_start);
            uint k_hi = min(tile_start + tile_n, eff_kv);
            for (uint k_pos = k_lo; k_pos < k_hi; k_pos++) {
                uint t = k_pos - tile_start;
                float dot = 0.0f;
                for (uint j = 0; j < dpl; j++) {
                    uint d = simd_lane + j * 32u;
                    if (d < head_dim) {
                        dot += q_vals[j] * float(K_tile[t * head_dim + d]);
                    }
                }
                dot = simd_sum(dot);

                float new_max = max(running_max, dot);
                float rescale = exp(running_max - new_max);
                float w = exp(dot - new_max);
                running_sum = running_sum * rescale + w;
                for (uint j = 0; j < dpl; j++) {
                    uint d = simd_lane + j * 32u;
                    float vv = (d < head_dim) ? float(V_tile[t * head_dim + d]) : 0.0f;
                    acc[j] = acc[j] * rescale + w * vv;
                }
                running_max = new_max;
            }
        }
    }

    if (active) {
        float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        for (uint j = 0; j < dpl; j++) {
            uint d = simd_lane + j * 32u;
            if (d < head_dim) {
                out[q_base + d] = acc[j] * inv_sum;
            }
        }
    }
}

/// Batched fused TurboQuant flash-attention decode (one new query token per
/// lane). Identical online-softmax + codebook math as `flash_attention_tq`, but
/// the grid is `(num_q_heads * n_lanes)` and K/V are read from a UNIFIED
/// by-lane pool: lane `l` owns the contiguous token region
/// `[l*lane_capacity, (l+1)*lane_capacity)` and sits at its own absolute
/// position `positions[l]` (so independent agents decode at different context
/// lengths in one dispatch). `rq` holds the per-lane rotated queries laid out
/// `[n_lanes, num_q_heads, head_dim]`; `out` receives the per-lane rotated-space
/// value accumulation in the same layout (apply `hadamard_rotate` with
/// post_sign to recover the attention output). This is the turbo analogue of
/// `flash_attention_decode_batched`.
kernel void flash_attention_tq_batched(
    device const float *rq        [[buffer(0)]],
    device const uchar *k_codes   [[buffer(1)]],
    device const half  *k_norms   [[buffer(2)]],
    device const uchar *v_codes   [[buffer(3)]],
    device const half  *v_norms   [[buffer(4)]],
    device const float *levels    [[buffer(5)]],
    device float       *out       [[buffer(6)]],
    constant uint      *positions [[buffer(7)]],
    constant uint      &head_dim  [[buffer(8)]],
    constant uint      &lane_capacity[[buffer(9)]],
    constant uint      &num_q_heads[[buffer(10)]],
    constant uint      &num_kv_heads[[buffer(11)]],
    constant uint      &window    [[buffer(12)]],
    constant uint      &bits      [[buffer(13)]],
    constant uint      &code_bytes[[buffer(14)]],
    constant uint      &n_lanes   [[buffer(15)]],
    constant uint      &ring_capacity[[buffer(16)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint lane    [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    uint q_head   = tg_id % num_q_heads;
    uint lane_idx = tg_id / num_q_heads;
    if (lane_idx >= n_lanes || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint current_pos = positions[lane_idx];
    uint seq_len = current_pos + 1u;
    uint win_start = (window > 0u && current_pos + 1u > window) ? (current_pos + 1u - window) : 0u;
    uint kv_lane_base = lane_idx * lane_capacity;
    uint dpl = (head_dim + 31u) / 32u;
    uint q_base = (lane_idx * num_q_heads + q_head) * head_dim;

    // Stage the reconstruction table into threadgroup memory (see note above).
    threadgroup float tg_levels[FLASH_TQ_MAX_LEVELS];
    uint nlev = 1u << bits;
    for (uint i = simd_id * 32u + lane; i < nlev; i += FLASH_TQ_NSG * 32u) {
        tg_levels[i] = levels[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rq_reg[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        rq_reg[i] = (d < head_dim) ? rq[q_base + d] : 0.0f;
    }

    float acc[FLASH_TQ_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = win_start + simd_id; t < seq_len; t += FLASH_TQ_NSG) {
        // `lane_capacity` is the physical per-lane stride; when ring_capacity
        // != 0 the logical position `t` wraps within the lane's ring region.
        uint physical_t = (ring_capacity != 0u) ? (t % ring_capacity) : t;
        uint slot = (kv_lane_base + physical_t) * num_kv_heads + kv_head;

        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d < head_dim) {
                uint code = tq_unpack_code(k_codes, slot, d, bits, code_bytes);
                dot += rq_reg[i] * tg_levels[code];
            }
        }
        dot = simd_sum(dot);
        dot *= float(k_norms[slot]);

        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;

        float vnorm = float(v_norms[slot]);
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            float vv = 0.0f;
            if (d < head_dim) {
                uint code = tq_unpack_code(v_codes, slot, d, bits, code_bytes);
                vv = tg_levels[code] * vnorm;
            }
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    threadgroup float tg_m[FLASH_TQ_NSG];
    threadgroup float tg_l[FLASH_TQ_NSG];
    threadgroup float tg_acc[FLASH_TQ_NSG * FLASH_TQ_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32u;
        if (d < head_dim) { tg_acc[simd_id * FLASH_TQ_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_TQ_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float gsum = 0.0f;
        float factor[FLASH_TQ_NSG];
        for (uint s = 0; s < FLASH_TQ_NSG; s++) {
            factor[s] = exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32u;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_TQ_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_TQ_MAX_HEAD_DIM + d];
            }
            out[q_base + d] = o * inv_sum;
        }
    }
}
