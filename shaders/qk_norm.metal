#include <metal_stdlib>
using namespace metal;

/// Per-head RMSNorm with learned weights (in-place).
/// One threadgroup per head. Threads within a threadgroup cooperate on the reduction.
kernel void qk_norm(
    device half *qk_data           [[buffer(0)]],
    device const half *weights     [[buffer(1)]],
    constant uint &head_dim        [[buffer(2)]],
    constant float &eps            [[buffer(3)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]]
) {
    uint base = head_idx * head_dim;

    // Step 1: compute sum of squares (reduction)
    float sum_sq = 0.0f;
    for (uint j = tid; j < head_dim; j += tg_size) {
        float val = float(qk_data[base + j]);
        sum_sq += val * val;
    }

    // SIMD reduction within simdgroup
    sum_sq = simd_sum(sum_sq);

    // Cross-simdgroup reduction via threadgroup memory
    threadgroup float simd_sums[32];
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;
    uint num_simdgroups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < num_simdgroups; s++) {
            total += simd_sums[s];
        }
        simd_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sum_sq = simd_sums[0];

    // Step 2: compute RMS and normalize with weights
    float rms = rsqrt(total_sum_sq / float(head_dim) + eps);
    for (uint j = tid; j < head_dim; j += tg_size) {
        float val = float(qk_data[base + j]);
        float w = float(weights[j]);
        qk_data[base + j] = half(val * rms * w);
    }
}

/// Largest `head_dim` the fused `qk_norm_rope` kernel supports (Gemma 4 = 256).
/// Must be >= the model's attention head_dim; the threadgroup staging array is
/// sized to this and the Rust dispatch asserts `head_dim <= QK_NORM_MAX_HEAD_DIM`.
constant uint QK_NORM_MAX_HEAD_DIM = 256;

/// Fused per-head RMSNorm + NEOX RoPE (one threadgroup per head).
///
/// Collapses the back-to-back `qk_norm` and `rope` decode dispatches into one.
/// The normalized head is staged in threadgroup memory (no global round-trip
/// between norm and rotate), then the RoPE pairing `(i, i + head_dim/2)` reads
/// from there. Math is identical to running `qk_norm` followed by the `rope`
/// kernel: `proportional` partial-rotary semantics — only the first
/// `partial_rotary_factor * head_dim / 2` pairs rotate, the rest pass through.
kernel void qk_norm_rope(
    device const half *input               [[buffer(0)]],
    device half       *output              [[buffer(1)]],
    device const half *weights             [[buffer(2)]],
    constant uint     &head_dim            [[buffer(3)]],
    constant float    &eps                 [[buffer(4)]],
    constant uint     &position            [[buffer(5)]],
    constant float    &theta               [[buffer(6)]],
    constant float    &partial_rotary_factor [[buffer(7)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid      [[thread_index_in_threadgroup]],
    uint tg_size  [[threads_per_threadgroup]]
) {
    uint base = head_idx * head_dim;

    // Staged as `half` (not `float`) so the normalized value is rounded to FP16
    // before RoPE — bit-identical to running the standalone `qk_norm` (which
    // writes FP16) followed by `rope` (which reads that FP16). Keeping it in
    // FP32 here would be *more* accurate but would shift greedy argmax vs the
    // shipping two-kernel path; matching the rounding keeps output unchanged.
    threadgroup half normed[QK_NORM_MAX_HEAD_DIM];
    threadgroup float simd_sums[32];

    // Step 1: sum of squares over the head (reduction).
    float sum_sq = 0.0f;
    for (uint j = tid; j < head_dim; j += tg_size) {
        float val = float(input[base + j]);
        sum_sq += val * val;
    }
    sum_sq = simd_sum(sum_sq);
    uint simd_id = tid / 32;
    uint simd_lane = tid % 32;
    uint num_simdgroups = (tg_size + 31) / 32;
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < num_simdgroups; s++) {
            total += simd_sums[s];
        }
        simd_sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rsqrt(simd_sums[0] / float(head_dim) + eps);

    // Step 2: normalize with learned weights into threadgroup memory (FP16).
    for (uint j = tid; j < head_dim; j += tg_size) {
        normed[j] = half(float(input[base + j]) * rms * float(weights[j]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: NEOX RoPE over the staged values; thread `i` rotates pair
    // (i, i + half_dim) for i < half_dim.
    uint half_dim = head_dim / 2;
    uint rope_angles = uint(float(head_dim) * partial_rotary_factor) / 2;
    for (uint i = tid; i < half_dim; i += tg_size) {
        float x0 = float(normed[i]);
        float x1 = float(normed[i + half_dim]);
        uint idx0 = base + i;
        uint idx1 = base + i + half_dim;
        if (i >= rope_angles) {
            output[idx0] = half(x0);
            output[idx1] = half(x1);
        } else {
            float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_val = cos(angle);
            float sin_val = sin(angle);
            output[idx0] = half(x0 * cos_val - x1 * sin_val);
            output[idx1] = half(x0 * sin_val + x1 * cos_val);
        }
    }
}
