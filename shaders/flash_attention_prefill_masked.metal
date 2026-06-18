#include <metal_stdlib>
using namespace metal;

#define FLASH_BR 32
#define FLASH_BC 32

/// Flash Attention v2 for batch prefill with an explicit additive attention mask.
///
/// Q:      [seq_len, num_heads, head_dim] FP16
/// K, V:   [kv_len, num_kv_heads, head_dim] FP16
/// mask:   [seq_len, seq_len] FP16  (additive: 0.0 = attend, -inf = block)
/// output: [seq_len, num_heads, head_dim] FP16
///
/// The mask replaces the implicit causal masking used by the unmasked variant.
/// For vision models, the mask encodes bidirectional attention over image tokens
/// combined with causal masking for text tokens.
///
/// Dispatch: threadgroups = (ceil(seq_len/BR) * num_heads, 1), threads_per_tg = (min(head_dim, 256))
kernel void flash_attention_prefill_masked(
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
    device const half *mask   [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Decode flat grid index into (q_block, head)
    uint total_q_blocks = (seq_len + FLASH_BR - 1) / FLASH_BR;
    uint q_block = gid % total_q_blocks;
    uint head = gid / total_q_blocks;

    if (head >= num_heads) return;

    uint kv_head = head / (num_heads / num_kv_heads);
    uint q_start = q_block * FLASH_BR;
    uint q_end = min(q_start + FLASH_BR, seq_len);
    uint num_q = q_end - q_start;

    if (num_q == 0) return;

    // Shared memory for dot products and reductions
    threadgroup float shared_scores[FLASH_BC];
    threadgroup float shared_max;
    threadgroup float simd_dots[32];

    uint num_sg = (tcount + 31) / 32;

    // Process each query row sequentially within this threadgroup
    for (uint qi = 0; qi < num_q; qi++) {
        uint q_pos = q_start + qi;

        // Load Q row distributed across threads
        float q_vals[8];
        uint num_dims = 0;
        for (uint d = tid; d < head_dim; d += tcount) {
            q_vals[num_dims] = float(Q[(q_pos * num_heads + head) * head_dim + d]) * scale;
            num_dims++;
        }

        float running_max = -INFINITY;
        float running_sum = 0.0f;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        // Use the full kv_len — masking is handled by the additive mask
        for (uint kv_start = 0; kv_start < kv_len; kv_start += FLASH_BC) {
            uint kv_end_tile = min(kv_start + FLASH_BC, kv_len);
            uint tile_len = kv_end_tile - kv_start;

            // Compute Q·K^T scores
            for (uint t = 0; t < tile_len; t++) {
                uint k_pos = kv_start + t;
                float dot = 0.0f;
                uint di = 0;
                for (uint d = tid; d < head_dim; d += tcount) {
                    dot += q_vals[di] * float(K[(k_pos * num_kv_heads + kv_head) * head_dim + d]);
                    di++;
                }
                dot = simd_sum(dot);
                if (simd_lane == 0) simd_dots[simd_id] = dot;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_id == 0) {
                    dot = (simd_lane < num_sg) ? simd_dots[simd_lane] : 0.0f;
                    dot = simd_sum(dot);
                }
                // Add mask value (0.0 or -inf) — mask is [seq_len, seq_len]
                // k_pos indexes into the kv cache; for prefill the first seq_len
                // positions correspond to the current sequence.
                if (tid == 0) {
                    float mask_val = (k_pos < seq_len) ? float(mask[q_pos * seq_len + k_pos]) : -INFINITY;
                    shared_scores[t] = dot + mask_val;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Online softmax
            if (tid == 0) {
                float tm = -INFINITY;
                for (uint t = 0; t < tile_len; t++) {
                    tm = max(tm, shared_scores[t]);
                }
                shared_max = tm;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float tile_max = shared_max;
            float new_max = max(running_max, tile_max);

            float rescale = exp(running_max - new_max);
            for (uint di = 0; di < num_dims; di++) {
                acc[di] *= rescale;
            }
            running_sum *= rescale;

            for (uint t = 0; t < tile_len; t++) {
                float w = exp(shared_scores[t] - new_max);
                running_sum += w;
                uint v_pos = kv_start + t;
                uint di2 = 0;
                for (uint d = tid; d < head_dim; d += tcount) {
                    acc[di2] += w * float(V[(v_pos * num_kv_heads + kv_head) * head_dim + d]);
                    di2++;
                }
            }
            running_max = new_max;
        }

        // Normalize and write
        float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
        uint di = 0;
        for (uint d = tid; d < head_dim; d += tcount) {
            output[(q_pos * num_heads + head) * head_dim + d] = half(acc[di] * inv_sum);
            di++;
        }
    }
}
