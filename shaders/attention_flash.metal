#include <metal_stdlib>
using namespace metal;

inline ulong paged_token_slot(
    device const ulong *page_table,
    ulong logical_pos,
    ulong page_size_tokens
) {
    ulong page_idx = logical_pos / page_size_tokens;
    ulong in_page = logical_pos % page_size_tokens;
    ulong physical_page = page_table[page_idx];
    return physical_page * page_size_tokens + in_page;
}

/// Full causal attention with GQA support.
/// Grid: (head_dim, num_q_heads, 1)
/// Each thread computes one output dimension for one Q head.
kernel void attention_full(
    device const half *Q          [[buffer(0)]],
    device const half *K          [[buffer(1)]],
    device const half *V          [[buffer(2)]],
    device half       *output     [[buffer(3)]],
    constant uint     &head_dim   [[buffer(4)]],
    constant uint     &kv_len     [[buffer(5)]],
    constant uint     &current_pos[[buffer(6)]],
    constant uint     &num_q_heads[[buffer(7)]],
    constant uint     &num_kv_heads[[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint q_head = tid.y;

    if (d >= head_dim || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);

    threadgroup float scores[8192];

    if (d == 0) {
        // Gemma 4 uses scaling=1.0 (QK-norm handles magnitude stabilization)
        float max_score = -INFINITY;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) {
            float score = 0.0;
            uint q_offset = q_head * head_dim;
            uint k_offset = t * num_kv_heads * head_dim + kv_head * head_dim;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += float(Q[q_offset + dd]) * float(K[k_offset + dd]);
            }
            scores[t] = score;
            max_score = max(max_score, scores[t]);
        }
        float sum_exp = 0.0;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) {
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) { scores[t] *= inv_sum; }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = 0.0;
    for (uint t = 0; t < kv_len && t <= current_pos; t++) {
        uint v_offset = t * num_kv_heads * head_dim + kv_head * head_dim + d;
        out_val += scores[t] * float(V[v_offset]);
    }

    output[q_head * head_dim + d] = half(out_val);
}

kernel void attention_full_paged(
    device const half *Q            [[buffer(0)]],
    device const half *K            [[buffer(1)]],
    device const half *V            [[buffer(2)]],
    device const ulong *page_table  [[buffer(3)]],
    device half       *output       [[buffer(4)]],
    constant uint     &head_dim     [[buffer(5)]],
    constant uint     &kv_len       [[buffer(6)]],
    constant uint     &current_pos  [[buffer(7)]],
    constant ulong    &page_size_tokens [[buffer(8)]],
    constant uint     &num_q_heads  [[buffer(9)]],
    constant uint     &num_kv_heads [[buffer(10)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint q_head = tid.y;

    if (d >= head_dim || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);

    threadgroup float scores[8192];

    if (d == 0) {
        float max_score = -INFINITY;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) {
            float score = 0.0;
            uint q_offset = q_head * head_dim;
            ulong physical_pos = paged_token_slot(page_table, (ulong)t, page_size_tokens);
            ulong k_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
                + (ulong)kv_head * (ulong)head_dim;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += float(Q[q_offset + dd]) * float(K[k_offset + dd]);
            }
            scores[t] = score;
            max_score = max(max_score, scores[t]);
        }
        float sum_exp = 0.0;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) {
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (uint t = 0; t < kv_len && t <= current_pos; t++) { scores[t] *= inv_sum; }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = 0.0;
    for (uint t = 0; t < kv_len && t <= current_pos; t++) {
        ulong physical_pos = paged_token_slot(page_table, (ulong)t, page_size_tokens);
        ulong v_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
            + (ulong)kv_head * (ulong)head_dim + (ulong)d;
        out_val += scores[t] * float(V[v_offset]);
    }

    output[q_head * head_dim + d] = half(out_val);
}

#define FLASH_TILE_SIZE 64

// Max head_dim / 32 lanes. Gemma 4 full-attention layers use head_dim 512
// (16 dims/lane); sliding-window layers use 256 (8 dims/lane).
#define FLASH_MAX_DPL 16
// Simdgroups per threadgroup (= per Q head). Splits the KV sequence across
// simdgroups for token-level parallelism, then combines once.
#define FLASH_NSG 8
#define FLASH_MAX_HEAD_DIM 512

/// Flash Attention v2 for single-query decode — one threadgroup per Q head,
/// `FLASH_NSG` simdgroups (256 threads) each owning a strided stripe of the KV
/// sequence (`t = win_start + simd_id, +NSG, +2·NSG, …`). Within a simdgroup
/// each lane owns a strided slice of the head dimension, so the Q·K reduction
/// is a single `simd_sum` with **no barriers inside the token loop**. The eight
/// partial (max, sum, acc) results are combined with online softmax in a single
/// threadgroup barrier at the end.
///
/// Dispatch: grid = (num_q_heads, 1, 1) threadgroups of `FLASH_NSG * 32` threads.
///
/// Q:      [num_q_heads * head_dim] FP16
/// K, V:   [kv_len * num_kv_heads * head_dim] FP16
/// output: [num_q_heads * head_dim] FP16
///
/// window: sliding-window size. The query at current_pos attends keys k with
/// current_pos - window < k <= current_pos. window == 0 means unlimited.
kernel void flash_attention(
    device const half *Q          [[buffer(0)]],
    device const half *K          [[buffer(1)]],
    device const half *V          [[buffer(2)]],
    device half       *output     [[buffer(3)]],
    constant uint     &head_dim   [[buffer(4)]],
    constant uint     &kv_len     [[buffer(5)]],
    constant uint     &current_pos[[buffer(6)]],
    constant uint     &num_q_heads[[buffer(7)]],
    constant uint     &num_kv_heads[[buffer(8)]],
    constant uint     &window     [[buffer(9)]],
    uint q_head    [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    if (q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint seq_len = min(kv_len, current_pos + 1);
    uint win_start = (window > 0 && current_pos + 1 > window) ? (current_pos + 1 - window) : 0;

    uint dpl = (head_dim + 31) / 32;  // dims handled by this lane (≤ FLASH_MAX_DPL)
    uint q_base = q_head * head_dim;

    // Cache this lane's slice of Q in registers.
    float q_reg[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        q_reg[i] = (d < head_dim) ? float(Q[q_base + d]) : 0.0f;
    }

    // This simdgroup's barrier-free online softmax over its KV stripe.
    float acc[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = win_start + simd_id; t < seq_len; t += FLASH_NSG) {
        uint kv_off = t * num_kv_heads * head_dim + kv_head * head_dim;

        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d < head_dim) { dot += q_reg[i] * float(K[kv_off + d]); }
        }
        dot = simd_sum(dot);  // every lane in this simdgroup holds the full score

        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            float vv = (d < head_dim) ? float(V[kv_off + d]) : 0.0f;
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    // Publish each simdgroup's partials, then combine with online softmax.
    threadgroup float tg_m[FLASH_NSG];
    threadgroup float tg_l[FLASH_NSG];
    threadgroup float tg_acc[FLASH_NSG * FLASH_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        if (d < head_dim) { tg_acc[simd_id * FLASH_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine: simdgroup 0 reduces the FLASH_NSG partials into the output.
    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float gsum = 0.0f;
        float factor[FLASH_NSG];
        for (uint s = 0; s < FLASH_NSG; s++) {
            factor[s] = exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_MAX_HEAD_DIM + d];
            }
            output[q_base + d] = half(o * inv_sum);
        }
    }
}

/// Batched-decode flash attention over N independent lanes (continuous
/// batching). Identical online-softmax math to `flash_attention`, but the grid
/// is `(num_q_heads * n_lanes)` threadgroups and each lane reads:
///   - its query row at  Q[(lane*num_q_heads + q_head)*head_dim ..]
///   - its KV from a UNIFIED pool where lane `l` owns the contiguous region
///     `[l*lane_capacity, (l+1)*lane_capacity)` tokens, and
///   - its own absolute position from `positions[lane]` (so every lane can sit
///     at a different context length, which is the whole point of batching
///     independent agents/sub-agents).
/// Output mirrors Q's layout. This is the per-lane attention step that lets the
/// weight-heavy projections/FFN run once at M=N (the ~2.7x amortization win)
/// while attention stays correct per sequence.
kernel void flash_attention_decode_batched(
    device const half *Q            [[buffer(0)]],
    device const half *K            [[buffer(1)]],
    device const half *V            [[buffer(2)]],
    device half       *output       [[buffer(3)]],
    constant uint     *positions    [[buffer(4)]],
    constant uint     &head_dim     [[buffer(5)]],
    constant uint     &lane_capacity[[buffer(6)]],
    constant uint     &num_q_heads  [[buffer(7)]],
    constant uint     &num_kv_heads [[buffer(8)]],
    constant uint     &window       [[buffer(9)]],
    constant uint     &n_lanes      [[buffer(10)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    uint q_head   = tg_id % num_q_heads;
    uint lane_idx = tg_id / num_q_heads;
    if (lane_idx >= n_lanes || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint current_pos = positions[lane_idx];
    uint seq_len = current_pos + 1;
    uint win_start = (window > 0 && current_pos + 1 > window) ? (current_pos + 1 - window) : 0;

    // Base of this lane's KV region inside the unified pool.
    uint kv_lane_base = lane_idx * lane_capacity * num_kv_heads * head_dim;

    uint dpl = (head_dim + 31) / 32;
    uint q_base = (lane_idx * num_q_heads + q_head) * head_dim;

    float q_reg[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        q_reg[i] = (d < head_dim) ? float(Q[q_base + d]) : 0.0f;
    }

    float acc[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = win_start + simd_id; t < seq_len; t += FLASH_NSG) {
        uint kv_off = kv_lane_base + t * num_kv_heads * head_dim + kv_head * head_dim;

        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d < head_dim) { dot += q_reg[i] * float(K[kv_off + d]); }
        }
        dot = simd_sum(dot);

        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            float vv = (d < head_dim) ? float(V[kv_off + d]) : 0.0f;
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    threadgroup float tg_m[FLASH_NSG];
    threadgroup float tg_l[FLASH_NSG];
    threadgroup float tg_acc[FLASH_NSG * FLASH_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        if (d < head_dim) { tg_acc[simd_id * FLASH_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float gsum = 0.0f;
        float factor[FLASH_NSG];
        for (uint s = 0; s < FLASH_NSG; s++) {
            factor[s] = exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        float inv_sum = (gsum > 0.0f) ? (1.0f / gsum) : 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_MAX_HEAD_DIM + d];
            }
            output[q_base + d] = half(o * inv_sum);
        }
    }
}

/// Scatter one new K/V row per lane into the unified paged-by-lane KV pool.
/// `k_src`/`v_src` are `[n_lanes, row]` (row = num_kv_heads*head_dim); lane `l`
/// is written at token `positions[l]` inside its region
/// `[l*lane_capacity, (l+1)*lane_capacity)`. One dispatch writes all lanes.
kernel void write_kv_cache_decode(
    device const half *k_src       [[buffer(0)]],
    device const half *v_src       [[buffer(1)]],
    device half       *k_cache     [[buffer(2)]],
    device half       *v_cache     [[buffer(3)]],
    constant uint     *positions   [[buffer(4)]],
    constant uint     &row         [[buffer(5)]],
    constant uint     &lane_capacity[[buffer(6)]],
    constant uint     &n_lanes     [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = n_lanes * row;
    if (tid >= total) return;
    uint lane = tid / row;
    uint e = tid - lane * row;
    uint dst = (lane * lane_capacity + positions[lane]) * row + e;
    uint src = lane * row + e;
    k_cache[dst] = k_src[src];
    v_cache[dst] = v_src[src];
}

/// Window-aware split for flash-decoding. Same barrier-free online-softmax math
/// as `flash_attention`, but the grid is `(num_q_heads * num_splits)`: the KV
/// range `[win_start, seq_len)` is divided into `num_splits` contiguous chunks
/// and threadgroup `(q_head, split_id)` reduces chunk `split_id` only. Each TG
/// combines its `FLASH_NSG` simdgroups into a single UNNORMALIZED partial
/// `(max, sum, acc)` written in the layout `flash_decoding_reduce` consumes
/// (`partial_*[q_head*num_splits + split_id]`). Splitting the sequence across
/// threadgroups raises occupancy: single-query decode otherwise launches only
/// `num_q_heads` (=8) threadgroups, using a fraction of the GPU.
kernel void flash_attention_windowed_split(
    device const half *Q          [[buffer(0)]],
    device const half *K          [[buffer(1)]],
    device const half *V          [[buffer(2)]],
    device float      *partial_max[[buffer(3)]],
    device float      *partial_sum[[buffer(4)]],
    device float      *partial_acc[[buffer(5)]],
    constant uint     &head_dim   [[buffer(6)]],
    constant uint     &kv_len     [[buffer(7)]],
    constant uint     &current_pos[[buffer(8)]],
    constant uint     &num_q_heads[[buffer(9)]],
    constant uint     &num_kv_heads[[buffer(10)]],
    constant uint     &window     [[buffer(11)]],
    constant uint     &num_splits [[buffer(12)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
) {
    uint q_head   = tg_id / num_splits;
    uint split_id = tg_id % num_splits;
    if (q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint seq_len = min(kv_len, current_pos + 1);
    uint win_start = (window > 0 && current_pos + 1 > window) ? (current_pos + 1 - window) : 0;

    // Contiguous chunk of [win_start, seq_len) owned by this split.
    uint range = (seq_len > win_start) ? (seq_len - win_start) : 0u;
    uint chunk = (range + num_splits - 1) / num_splits;
    uint chunk_start = win_start + split_id * chunk;
    uint chunk_end = min(chunk_start + chunk, seq_len);

    uint dpl = (head_dim + 31) / 32;
    uint q_base = q_head * head_dim;
    uint pm_idx = q_head * num_splits + split_id;
    uint acc_base = pm_idx * head_dim;

    float q_reg[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        q_reg[i] = (d < head_dim) ? float(Q[q_base + d]) : 0.0f;
    }

    float acc[FLASH_MAX_DPL];
    for (uint i = 0; i < dpl; i++) { acc[i] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint t = chunk_start + simd_id; t < chunk_end; t += FLASH_NSG) {
        uint kv_off = t * num_kv_heads * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d < head_dim) { dot += q_reg[i] * float(K[kv_off + d]); }
        }
        dot = simd_sum(dot);
        float new_max = max(running_max, dot);
        float rescale = exp(running_max - new_max);
        float w = exp(dot - new_max);
        running_sum = running_sum * rescale + w;
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            float vv = (d < head_dim) ? float(V[kv_off + d]) : 0.0f;
            acc[i] = acc[i] * rescale + w * vv;
        }
        running_max = new_max;
    }

    threadgroup float tg_m[FLASH_NSG];
    threadgroup float tg_l[FLASH_NSG];
    threadgroup float tg_acc[FLASH_NSG * FLASH_MAX_HEAD_DIM];
    if (lane == 0) {
        tg_m[simd_id] = running_max;
        tg_l[simd_id] = running_sum;
    }
    for (uint i = 0; i < dpl; i++) {
        uint d = lane + i * 32;
        if (d < head_dim) { tg_acc[simd_id * FLASH_MAX_HEAD_DIM + d] = acc[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0 combines the FLASH_NSG partials into one UNNORMALIZED partial.
    if (simd_id == 0) {
        float gmax = -INFINITY;
        for (uint s = 0; s < FLASH_NSG; s++) { gmax = max(gmax, tg_m[s]); }
        float factor[FLASH_NSG];
        float gsum = 0.0f;
        for (uint s = 0; s < FLASH_NSG; s++) {
            factor[s] = (tg_m[s] == -INFINITY) ? 0.0f : exp(tg_m[s] - gmax);
            gsum += factor[s] * tg_l[s];
        }
        if (lane == 0) {
            partial_max[pm_idx] = (gmax == -INFINITY) ? -INFINITY : gmax;
            partial_sum[pm_idx] = gsum;
        }
        for (uint i = 0; i < dpl; i++) {
            uint d = lane + i * 32;
            if (d >= head_dim) { continue; }
            float o = 0.0f;
            for (uint s = 0; s < FLASH_NSG; s++) {
                o += factor[s] * tg_acc[s * FLASH_MAX_HEAD_DIM + d];
            }
            partial_acc[acc_base + d] = o;
        }
    }
}

kernel void flash_attention_paged(
    device const half *Q             [[buffer(0)]],
    device const half *K             [[buffer(1)]],
    device const half *V             [[buffer(2)]],
    device const ulong *page_table   [[buffer(3)]],
    device half       *output        [[buffer(4)]],
    constant uint     &head_dim      [[buffer(5)]],
    constant uint     &kv_len        [[buffer(6)]],
    constant uint     &current_pos   [[buffer(7)]],
    constant ulong    &page_size_tokens [[buffer(8)]],
    constant uint     &num_q_heads   [[buffer(9)]],
    constant uint     &num_kv_heads  [[buffer(10)]],
    uint q_head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint seq_len = min(kv_len, current_pos + 1);

    threadgroup float q_shared[512];
    for (uint d = tid; d < head_dim; d += tcount) {
        q_shared[d] = float(Q[q_head * head_dim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    threadgroup float tile_scores[FLASH_TILE_SIZE];
    threadgroup float simd_dots[32];
    threadgroup float shared_tile_max;

    uint num_sg = (tcount + 31) / 32;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += FLASH_TILE_SIZE) {
        uint tile_end = min(tile_start + FLASH_TILE_SIZE, seq_len);
        uint tile_len = tile_end - tile_start;

        for (uint t = 0; t < tile_len; t++) {
            ulong physical_pos = paged_token_slot(
                page_table,
                (ulong)(tile_start + t),
                page_size_tokens
            );
            ulong k_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
                + (ulong)kv_head * (ulong)head_dim;

            float dot = 0.0f;
            for (uint d = tid; d < head_dim; d += tcount) {
                dot += q_shared[d] * float(K[k_offset + d]);
            }

            dot = simd_sum(dot);

            if (simd_lane == 0) simd_dots[simd_id] = dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_id == 0) {
                dot = (simd_lane < num_sg) ? simd_dots[simd_lane] : 0.0f;
                dot = simd_sum(dot);
            }

            if (tid == 0) tile_scores[t] = dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            float tm = -INFINITY;
            for (uint t = 0; t < tile_len; t++) {
                tm = max(tm, tile_scores[t]);
            }
            shared_tile_max = tm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tile_max = shared_tile_max;

        float new_max = max(running_max, tile_max);
        float rescale = exp(running_max - new_max);
        for (uint di = 0; di < 8; di++) {
            acc[di] *= rescale;
        }
        running_sum *= rescale;

        for (uint t = 0; t < tile_len; t++) {
            float w = exp(tile_scores[t] - new_max);
            running_sum += w;

            ulong physical_pos = paged_token_slot(
                page_table,
                (ulong)(tile_start + t),
                page_size_tokens
            );
            ulong v_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
                + (ulong)kv_head * (ulong)head_dim;
            uint di = 0;
            for (uint d = tid; d < head_dim; d += tcount) {
                acc[di] += w * float(V[v_offset + d]);
                di++;
            }
        }

        running_max = new_max;
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    uint di = 0;
    for (uint d = tid; d < head_dim; d += tcount) {
        output[q_head * head_dim + d] = half(acc[di] * inv_sum);
        di++;
    }
}
