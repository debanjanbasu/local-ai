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

/// Sliding window causal attention with GQA support.
/// Grid: (head_dim, num_q_heads, 1)
/// Each thread computes one output dimension for one Q head.
kernel void attention_sliding(
    device const half *Q          [[buffer(0)]],  // [num_q_heads * head_dim]
    device const half *K          [[buffer(1)]],  // [kv_len * num_kv_heads * head_dim]
    device const half *V          [[buffer(2)]],  // [kv_len * num_kv_heads * head_dim]
    device half       *output     [[buffer(3)]],  // [num_q_heads * head_dim]
    constant uint     &head_dim   [[buffer(4)]],
    constant uint     &kv_len     [[buffer(5)]],
    constant uint     &current_pos[[buffer(6)]],
    constant uint     &window     [[buffer(7)]],
    constant uint     &num_q_heads[[buffer(8)]],
    constant uint     &num_kv_heads[[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint q_head = tid.y;

    if (d >= head_dim || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);

    uint win_start = (current_pos + 1 > window) ? (current_pos + 1 - window) : 0;
    uint win_end = min(kv_len, current_pos + 1);
    uint actual_len = win_end - win_start;

    threadgroup float scores[2048];

    if (d == 0) {
        // Gemma 4 uses scaling=1.0 (QK-norm handles magnitude stabilization)
        float max_score = -INFINITY;
        for (uint t = 0; t < actual_len; t++) {
            uint pos = win_start + t;
            if (pos > current_pos) { scores[t] = -INFINITY; continue; }
            float score = 0.0;
            uint q_offset = q_head * head_dim;
            uint k_offset = pos * num_kv_heads * head_dim + kv_head * head_dim;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += float(Q[q_offset + dd]) * float(K[k_offset + dd]);
            }
            scores[t] = score;
            max_score = max(max_score, scores[t]);
        }
        float sum_exp = 0.0;
        for (uint t = 0; t < actual_len; t++) {
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (uint t = 0; t < actual_len; t++) { scores[t] *= inv_sum; }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = 0.0;
    for (uint t = 0; t < actual_len; t++) {
        uint pos = win_start + t;
        uint v_offset = pos * num_kv_heads * head_dim + kv_head * head_dim + d;
        out_val += scores[t] * float(V[v_offset]);
    }

    output[q_head * head_dim + d] = half(out_val);
}

kernel void attention_sliding_paged(
    device const half *Q             [[buffer(0)]],
    device const half *K             [[buffer(1)]],
    device const half *V             [[buffer(2)]],
    device const ulong *page_table   [[buffer(3)]],
    device half       *output        [[buffer(4)]],
    constant uint     &head_dim      [[buffer(5)]],
    constant uint     &kv_len        [[buffer(6)]],
    constant uint     &current_pos   [[buffer(7)]],
    constant uint     &window        [[buffer(8)]],
    constant ulong    &page_size_tokens [[buffer(9)]],
    constant uint     &num_q_heads   [[buffer(10)]],
    constant uint     &num_kv_heads  [[buffer(11)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint d = tid.x;
    uint q_head = tid.y;

    if (d >= head_dim || q_head >= num_q_heads) return;

    uint kv_head = q_head / (num_q_heads / num_kv_heads);

    uint win_start = (current_pos + 1 > window) ? (current_pos + 1 - window) : 0;
    uint win_end = min(kv_len, current_pos + 1);
    uint actual_len = win_end - win_start;

    threadgroup float scores[2048];

    if (d == 0) {
        float max_score = -INFINITY;
        for (uint t = 0; t < actual_len; t++) {
            uint pos = win_start + t;
            if (pos > current_pos) { scores[t] = -INFINITY; continue; }
            float score = 0.0;
            uint q_offset = q_head * head_dim;
            ulong physical_pos = paged_token_slot(page_table, (ulong)pos, page_size_tokens);
            ulong k_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
                + (ulong)kv_head * (ulong)head_dim;
            for (uint dd = 0; dd < head_dim; dd++) {
                score += float(Q[q_offset + dd]) * float(K[k_offset + dd]);
            }
            scores[t] = score;
            max_score = max(max_score, scores[t]);
        }
        float sum_exp = 0.0;
        for (uint t = 0; t < actual_len; t++) {
            scores[t] = exp(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : 0.0;
        for (uint t = 0; t < actual_len; t++) { scores[t] *= inv_sum; }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_val = 0.0;
    for (uint t = 0; t < actual_len; t++) {
        uint pos = win_start + t;
        ulong physical_pos = paged_token_slot(page_table, (ulong)pos, page_size_tokens);
        ulong v_offset = physical_pos * (ulong)num_kv_heads * (ulong)head_dim
            + (ulong)kv_head * (ulong)head_dim + (ulong)d;
        out_val += scores[t] * float(V[v_offset]);
    }

    output[q_head * head_dim + d] = half(out_val);
}
