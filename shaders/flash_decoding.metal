#include <metal_stdlib>
using namespace metal;

#define FD_SPLIT_SIZE 256
#define FD_TILE_SIZE  64

/// Flash-decoding split kernel.
/// Each threadgroup processes one (q_head, kv_split) pair.
/// Dispatch: (num_q_heads * num_splits, 1, 1)
kernel void flash_decoding_split(
    device const half  *Q            [[buffer(0)]],
    device const half  *K            [[buffer(1)]],
    device const half  *V            [[buffer(2)]],
    device float       *partial_max  [[buffer(3)]],
    device float       *partial_sum  [[buffer(4)]],
    device float       *partial_acc  [[buffer(5)]],
    constant uint      &head_dim     [[buffer(6)]],
    constant uint      &kv_len       [[buffer(7)]],
    constant uint      &current_pos  [[buffer(8)]],
    constant uint      &num_q_heads  [[buffer(9)]],
    constant uint      &num_kv_heads [[buffer(10)]],
    constant uint      &num_splits   [[buffer(11)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tcount   [[threads_per_threadgroup]],
    uint simd_id  [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint q_head  = tg_id / num_splits;
    uint split_id = tg_id % num_splits;

    if (q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);
    uint seq_len = min(kv_len, current_pos + 1);

    uint split_start = split_id * FD_SPLIT_SIZE;
    uint split_end   = min(split_start + FD_SPLIT_SIZE, seq_len);

    // Output indices for this split
    uint pm_idx = q_head * num_splits + split_id;
    uint acc_base = (q_head * num_splits + split_id) * head_dim;

    // Handle empty splits
    if (split_start >= split_end) {
        if (tid == 0) {
            partial_max[pm_idx] = -INFINITY;
            partial_sum[pm_idx] = 0.0f;
        }
        for (uint d = tid; d < head_dim; d += tcount) {
            partial_acc[acc_base + d] = 0.0f;
        }
        return;
    }

    // Load Q vector into threadgroup memory
    threadgroup float q_shared[512];
    for (uint d = tid; d < head_dim; d += tcount) {
        q_shared[d] = float(Q[q_head * head_dim + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-thread accumulators
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    threadgroup float tile_scores[FD_TILE_SIZE];
    threadgroup float simd_dots[32];
    threadgroup float shared_tile_max;

    uint num_sg = (tcount + 31) / 32;

    for (uint tile_start = split_start; tile_start < split_end; tile_start += FD_TILE_SIZE) {
        uint tile_end = min(tile_start + FD_TILE_SIZE, split_end);
        uint tile_len = tile_end - tile_start;

        // Step 1: Compute Q·K scores for this tile
        for (uint t = 0; t < tile_len; t++) {
            uint k_offset = (tile_start + t) * num_kv_heads * head_dim + kv_head * head_dim;

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

        // Step 2: Online softmax update
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

            uint v_offset = (tile_start + t) * num_kv_heads * head_dim + kv_head * head_dim;
            uint di = 0;
            for (uint d = tid; d < head_dim; d += tcount) {
                acc[di] += w * float(V[v_offset + d]);
                di++;
            }
        }

        running_max = new_max;
    }

    // Write partial results (unnormalized)
    if (tid == 0) {
        partial_max[pm_idx] = running_max;
        partial_sum[pm_idx] = running_sum;
    }
    uint di = 0;
    for (uint d = tid; d < head_dim; d += tcount) {
        partial_acc[acc_base + d] = acc[di];
        di++;
    }
}

/// Flash-decoding reduce kernel.
/// Merges partial results via log-sum-exp correction.
/// Dispatch: (num_q_heads, 1, 1)
kernel void flash_decoding_reduce(
    device const float *partial_max  [[buffer(0)]],
    device const float *partial_sum  [[buffer(1)]],
    device const float *partial_acc  [[buffer(2)]],
    device half        *output       [[buffer(3)]],
    constant uint      &head_dim     [[buffer(4)]],
    constant uint      &num_splits   [[buffer(5)]],
    constant uint      &num_q_heads  [[buffer(6)]],
    uint q_head [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]],
    uint tcount  [[threads_per_threadgroup]]
) {
    if (q_head >= num_q_heads) return;

    // Find global max across splits
    float global_max = -INFINITY;
    for (uint s = 0; s < num_splits; s++) {
        global_max = max(global_max, partial_max[q_head * num_splits + s]);
    }

    // Compute corrected global sum
    float global_sum = 0.0f;
    for (uint s = 0; s < num_splits; s++) {
        float m = partial_max[q_head * num_splits + s];
        float local_sum = partial_sum[q_head * num_splits + s];
        global_sum += local_sum * exp(m - global_max);
    }

    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Merge and normalize accumulators
    for (uint d = tid; d < head_dim; d += tcount) {
        float val = 0.0f;
        for (uint s = 0; s < num_splits; s++) {
            float m = partial_max[q_head * num_splits + s];
            float correction = exp(m - global_max);
            uint acc_idx = (q_head * num_splits + s) * head_dim + d;
            val += partial_acc[acc_idx] * correction;
        }
        output[q_head * head_dim + d] = half(val * inv_sum);
    }
}
