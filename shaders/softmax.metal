#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const half *input  [[buffer(0)]],
    device half       *output [[buffer(1)]],
    constant uint     &dim    [[buffer(2)]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint tcount               [[threads_per_threadgroup]],
    uint gid                  [[threadgroup_position_in_grid]],
    uint simd_id              [[simdgroup_index_in_threadgroup]],
    uint simd_lane            [[thread_index_in_simdgroup]]
) {
    threadgroup float simd_vals[32];
    threadgroup float row_max_tg;
    threadgroup float inv_sum_tg;
    uint row_offset = gid * dim;
    uint num_sg = (tcount + 31) / 32;

    // --- Max reduction ---
    float local_max = -INFINITY;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        local_max = max(local_max, val);
    }
    float mx = simd_max(local_max);
    if (simd_lane == 0) simd_vals[simd_id] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        mx = (simd_lane < num_sg) ? simd_vals[simd_lane] : -INFINITY;
        mx = simd_max(mx);
        row_max_tg = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_max = row_max_tg;

    // --- Exp + sum reduction ---
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tcount) {
        float val = exp(float(input[row_offset + i]) - row_max);
        output[row_offset + i] = half(val);
        local_sum += val;
    }
    float sm = simd_sum(local_sum);
    if (simd_lane == 0) simd_vals[simd_id] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sm = (simd_lane < num_sg) ? simd_vals[simd_lane] : 0.0f;
        sm = simd_sum(sm);
        inv_sum_tg = 1.0f / sm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_sum = inv_sum_tg;
    for (uint i = tid; i < dim; i += tcount) {
        output[row_offset + i] = half(float(output[row_offset + i]) * inv_sum);
    }
}
