#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FLASH_BR 32
/// Max head_dim / 32 we support in registers (Gemma 4: 256 or 512 ⇒ ≤16).
#define FLASH_MAXD 16

/// Flash Attention for batch prefill.
///
/// Q:      [seq_len, num_heads, head_dim] FP16
/// K, V:   [kv_len, num_kv_heads, head_dim] FP16
/// output: [seq_len, num_heads, head_dim] FP16
///
/// One threadgroup per (query_block, head) pair; one **simdgroup per query
/// row** within the block, so up to `simdgroups_per_tg` query rows run
/// concurrently and the head_dim reduction is a single `simd_sum` — no
/// threadgroup barriers and no shared scratch (the previous version processed
/// one query and one key at a time behind two barriers per key, which made
/// long-context prefill barrier-bound).
///
/// Dispatch: threadgroups = (ceil(seq_len/BR) * num_heads, 1),
///           threads_per_tg = min(head_dim, 256).
kernel void flash_attention_prefill(
    device const half *Q      [[buffer(0)]],
    device const half *K      [[buffer(1)]],
    device const half *V      [[buffer(2)]],
    device half *output       [[buffer(3)]],
    constant uint &seq_len    [[buffer(4)]],
    constant uint &kv_len     [[buffer(5)]],
    constant uint &num_heads  [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim   [[buffer(8)]],
    constant float &scale     [[buffer(9)]],
    constant uint &window     [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tcount [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint total_q_blocks = (seq_len + FLASH_BR - 1) / FLASH_BR;
    uint q_block = gid % total_q_blocks;
    uint head = gid / total_q_blocks;
    if (head >= num_heads) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_start = q_block * FLASH_BR;
    uint num_sg = tcount / 32;          // simdgroups per threadgroup
    uint dpl = (head_dim + 31) / 32;    // head_dim elements per lane (≤ FLASH_MAXD)

    // Each simdgroup owns query rows q_start + {simd_id, simd_id+num_sg, …}.
    for (uint qi = simd_id; qi < FLASH_BR; qi += num_sg) {
        uint q_pos = q_start + qi;
        if (q_pos >= seq_len) break;

        // This lane's slice of the Q row (dims simd_lane, simd_lane+32, …).
        float q_vals[FLASH_MAXD];
        for (uint j = 0; j < dpl; j++) {
            uint d = simd_lane + j * 32;
            q_vals[j] = (d < head_dim)
                ? float(Q[(q_pos * num_heads + head) * head_dim + d]) * scale
                : 0.0f;
        }

        float running_max = -INFINITY;
        float running_sum = 0.0f;
        float acc[FLASH_MAXD];
        for (uint j = 0; j < dpl; j++) acc[j] = 0.0f;

        // Causal mask with history offset: query q_pos sits at global position
        // (kv_len - seq_len + q_pos) and attends everything up to itself.
        uint effective_kv_len = min(kv_len, kv_len - seq_len + q_pos + 1);
        // Sliding-window mask: attend keys k with q_global - window < k <= q_global.
        uint q_global = kv_len - seq_len + q_pos;
        uint win_start = (window > 0 && q_global + 1 > window) ? (q_global + 1 - window) : 0;

        for (uint k_pos = win_start; k_pos < effective_kv_len; k_pos++) {
            // Q·K for this key, reduced across the simdgroup's lanes.
            float dot = 0.0f;
            for (uint j = 0; j < dpl; j++) {
                uint d = simd_lane + j * 32;
                if (d < head_dim) {
                    dot += q_vals[j]
                        * float(K[(k_pos * num_kv_heads + kv_head) * head_dim + d]);
                }
            }
            dot = simd_sum(dot); // every lane now holds the full score

            // Online softmax + V accumulation (all lanes compute the scalars).
            float new_max = max(running_max, dot);
            float rescale = exp(running_max - new_max);
            float w = exp(dot - new_max);
            running_sum = running_sum * rescale + w;
            for (uint j = 0; j < dpl; j++) {
                uint d = simd_lane + j * 32;
                float vv = (d < head_dim)
                    ? float(V[(k_pos * num_kv_heads + kv_head) * head_dim + d])
                    : 0.0f;
                acc[j] = acc[j] * rescale + w * vv;
            }
            running_max = new_max;
        }

        float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        for (uint j = 0; j < dpl; j++) {
            uint d = simd_lane + j * 32;
            if (d < head_dim) {
                output[(q_pos * num_heads + head) * head_dim + d] = half(acc[j] * inv_sum);
            }
        }
    }
}

/// Tiled flash-attention prefill that streams each K/V tile through threadgroup
/// memory ONCE and reuses it across all `FLASH_TILED_BR` query rows in the
/// block. The non-tiled kernel above re-reads all of K/V from device memory for
/// every query row; for long prompts that is the dominant cost. Here BR rows map
/// 1:1 to simdgroups (so each row keeps its online-softmax accumulators in
/// registers — no threadgroup accumulator state) while the K/V tile is shared.
///
/// Tile is `FLASH_TILE_HALFS` halfs each for K and V (16 KB total): BK keys =
/// FLASH_TILE_HALFS / head_dim. Works for head_dim 32 / 256 / 512.
///
/// Dispatch: threadgroups = (ceil(seq_len/FLASH_TILED_BR) * num_heads, 1),
///           threads_per_tg = FLASH_TILED_BR * 32 (one simdgroup per query row).
#define FLASH_TILED_BR 8
#define FLASH_TILE_HALFS 4096

kernel void flash_attention_prefill_tiled(
    device const half *Q      [[buffer(0)]],
    device const half *K      [[buffer(1)]],
    device const half *V      [[buffer(2)]],
    device half *output       [[buffer(3)]],
    constant uint &seq_len    [[buffer(4)]],
    constant uint &kv_len     [[buffer(5)]],
    constant uint &num_heads  [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim   [[buffer(8)]],
    constant float &scale     [[buffer(9)]],
    constant uint &window     [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tcount [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup half K_tile[FLASH_TILE_HALFS];
    threadgroup half V_tile[FLASH_TILE_HALFS];

    uint total_q_blocks = (seq_len + FLASH_TILED_BR - 1) / FLASH_TILED_BR;
    uint q_block = gid % total_q_blocks;
    uint head = gid / total_q_blocks;

    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_start = q_block * FLASH_TILED_BR;
    uint dpl = (head_dim + 31) / 32;     // head_dim elements per lane (≤ FLASH_MAXD)
    uint bk = FLASH_TILE_HALFS / head_dim; // keys per tile

    // This simdgroup owns exactly one query row.
    uint qi = simd_id;
    uint q_pos = q_start + qi;
    bool active = (q_pos < seq_len);

    float q_vals[FLASH_MAXD];
    float acc[FLASH_MAXD];
    for (uint j = 0; j < dpl; j++) { q_vals[j] = 0.0f; acc[j] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    uint q_global = kv_len - seq_len + q_pos;
    uint effective_kv_len = active ? min(kv_len, q_global + 1) : 0u;
    uint win_start = (window > 0 && q_global + 1 > window) ? (q_global + 1 - window) : 0u;

    if (active) {
        for (uint j = 0; j < dpl; j++) {
            uint d = simd_lane + j * 32;
            q_vals[j] = (d < head_dim)
                ? float(Q[(q_pos * num_heads + head) * head_dim + d]) * scale
                : 0.0f;
        }
    }

    // Stream K/V in tiles; the loop bound (kv_len) is uniform across the
    // threadgroup so every thread reaches the same barriers.
    for (uint tile_start = 0; tile_start < kv_len; tile_start += bk) {
        uint tile_n = min(bk, kv_len - tile_start);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < tile_n * head_dim; idx += tcount) {
            uint kk = idx / head_dim;
            uint dd = idx - kk * head_dim;
            uint g = ((tile_start + kk) * num_kv_heads + kv_head) * head_dim + dd;
            K_tile[idx] = K[g];
            V_tile[idx] = V[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint k_lo = max(tile_start, win_start);
            uint k_hi = min(tile_start + tile_n, effective_kv_len);
            for (uint k_pos = k_lo; k_pos < k_hi; k_pos++) {
                uint t = k_pos - tile_start;
                float dot = 0.0f;
                for (uint j = 0; j < dpl; j++) {
                    uint d = simd_lane + j * 32;
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
                    uint d = simd_lane + j * 32;
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
            uint d = simd_lane + j * 32;
            if (d < head_dim) {
                output[(q_pos * num_heads + head) * head_dim + d] = half(acc[j] * inv_sum);
            }
        }
    }
}

/// MQA multi-head tiled flash-attention prefill: one threadgroup per query
/// block processes **all** query heads, loading each K/V tile through
/// threadgroup memory ONCE and reusing it across all heads AND all
/// `FLASH_TILED_BR` rows. Because this model is MQA (num_kv_heads == 1) every
/// head shares the same K/V, so the per-head kernel above reloads identical K/V
/// `num_heads` times from device memory; this kernel removes that redundancy
/// (an extra `num_heads`× cut in K/V device traffic on top of the row reuse).
///
/// Each simdgroup still owns one query row; its lanes split head_dim and keep
/// per-head online-softmax accumulators in registers (`[MAXHEADS][MAXD]`), so
/// no threadgroup accumulator state is needed. Requires num_kv_heads == 1 and
/// num_heads <= FLASH_MH_MAXHEADS.
///
/// Dispatch: threadgroups = (ceil(seq_len/FLASH_TILED_BR), 1),
///           threads_per_tg = FLASH_TILED_BR * 32.
#define FLASH_MH_MAXHEADS 8

kernel void flash_attention_prefill_mqa(
    device const half *Q      [[buffer(0)]],
    device const half *K      [[buffer(1)]],
    device const half *V      [[buffer(2)]],
    device half *output       [[buffer(3)]],
    constant uint &seq_len    [[buffer(4)]],
    constant uint &kv_len     [[buffer(5)]],
    constant uint &num_heads  [[buffer(6)]],
    constant uint &num_kv_heads [[buffer(7)]],
    constant uint &head_dim   [[buffer(8)]],
    constant float &scale     [[buffer(9)]],
    constant uint &window     [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tcount [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    threadgroup half K_tile[FLASH_TILE_HALFS];
    threadgroup half V_tile[FLASH_TILE_HALFS];

    uint q_block = gid;                     // one threadgroup per query block
    uint q_start = q_block * FLASH_TILED_BR;
    uint dpl = (head_dim + 31) / 32;
    uint bk = FLASH_TILE_HALFS / head_dim;
    uint nh = min(num_heads, (uint)FLASH_MH_MAXHEADS);

    uint qi = simd_id;
    uint q_pos = q_start + qi;
    bool active = (q_pos < seq_len);

    float q_vals[FLASH_MH_MAXHEADS][FLASH_MAXD];
    float acc[FLASH_MH_MAXHEADS][FLASH_MAXD];
    float running_max[FLASH_MH_MAXHEADS];
    float running_sum[FLASH_MH_MAXHEADS];
    for (uint h = 0; h < nh; h++) {
        running_max[h] = -INFINITY;
        running_sum[h] = 0.0f;
        for (uint j = 0; j < dpl; j++) { q_vals[h][j] = 0.0f; acc[h][j] = 0.0f; }
    }

    uint q_global = kv_len - seq_len + q_pos;
    uint effective_kv_len = active ? min(kv_len, q_global + 1) : 0u;
    uint win_start = (window > 0 && q_global + 1 > window) ? (q_global + 1 - window) : 0u;

    if (active) {
        for (uint h = 0; h < nh; h++) {
            for (uint j = 0; j < dpl; j++) {
                uint d = simd_lane + j * 32;
                q_vals[h][j] = (d < head_dim)
                    ? float(Q[(q_pos * num_heads + h) * head_dim + d]) * scale
                    : 0.0f;
            }
        }
    }

    for (uint tile_start = 0; tile_start < kv_len; tile_start += bk) {
        uint tile_n = min(bk, kv_len - tile_start);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < tile_n * head_dim; idx += tcount) {
            uint kk = idx / head_dim;
            uint dd = idx - kk * head_dim;
            uint g = ((tile_start + kk) * num_kv_heads + 0u) * head_dim + dd; // MQA: kv_head 0
            K_tile[idx] = K[g];
            V_tile[idx] = V[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            uint k_lo = max(tile_start, win_start);
            uint k_hi = min(tile_start + tile_n, effective_kv_len);
            for (uint k_pos = k_lo; k_pos < k_hi; k_pos++) {
                uint t = k_pos - tile_start;
                for (uint h = 0; h < nh; h++) {
                    float dot = 0.0f;
                    for (uint j = 0; j < dpl; j++) {
                        uint d = simd_lane + j * 32;
                        if (d < head_dim) {
                            dot += q_vals[h][j] * float(K_tile[t * head_dim + d]);
                        }
                    }
                    dot = simd_sum(dot);

                    float new_max = max(running_max[h], dot);
                    float rescale = exp(running_max[h] - new_max);
                    float w = exp(dot - new_max);
                    running_sum[h] = running_sum[h] * rescale + w;
                    for (uint j = 0; j < dpl; j++) {
                        uint d = simd_lane + j * 32;
                        float vv = (d < head_dim) ? float(V_tile[t * head_dim + d]) : 0.0f;
                        acc[h][j] = acc[h][j] * rescale + w * vv;
                    }
                    running_max[h] = new_max;
                }
            }
        }
    }

    if (active) {
        for (uint h = 0; h < nh; h++) {
            float inv_sum = (running_sum[h] > 0.0f) ? (1.0f / running_sum[h]) : 0.0f;
            for (uint j = 0; j < dpl; j++) {
                uint d = simd_lane + j * 32;
                if (d < head_dim) {
                    output[(q_pos * num_heads + h) * head_dim + d] = half(acc[h][j] * inv_sum);
                }
            }
        }
    }
}
