#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const half *input    [[buffer(0)]],
    device const half *weight   [[buffer(1)]],
    device half       *output   [[buffer(2)]],
    constant uint     &dim      [[buffer(3)]],
    constant float    &eps      [[buffer(4)]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tcount                 [[threads_per_threadgroup]],
    uint gid                    [[threadgroup_position_in_grid]],
    uint simd_id                [[simdgroup_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]]
) {
    threadgroup float simd_sums[32];
    threadgroup float final_rms;
    uint row_offset = gid * dim;
    uint num_sg = (tcount + 31) / 32;

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        local_sum += val * val;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) simd_sums[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < num_sg) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        final_rms = rsqrt(sum / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = final_rms;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        float w = float(weight[i]);
        output[row_offset + i] = half(val * rms * w);
    }
}

/// Fused RMSNorm + residual: output[i] = rms_norm(input, weight)[i] + residual[i].
/// Collapses the post-norm + residual_add pair into one dispatch, removing a
/// round-trip through a scratch buffer in the decode chain.
kernel void rms_norm_add(
    device const half *input    [[buffer(0)]],
    device const half *weight   [[buffer(1)]],
    device half       *output   [[buffer(2)]],
    constant uint     &dim      [[buffer(3)]],
    constant float    &eps      [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tcount                 [[threads_per_threadgroup]],
    uint gid                    [[threadgroup_position_in_grid]],
    uint simd_id                [[simdgroup_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]]
) {
    threadgroup float simd_sums[32];
    threadgroup float final_rms;
    uint row_offset = gid * dim;
    uint num_sg = (tcount + 31) / 32;

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        local_sum += val * val;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) simd_sums[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < num_sg) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        final_rms = rsqrt(sum / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = final_rms;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        float w = float(weight[i]);
        output[row_offset + i] = half(val * rms * w + float(residual[row_offset + i]));
    }
}

kernel void rms_norm_centered(
    device const half *input    [[buffer(0)]],
    device const half *weight   [[buffer(1)]],
    device half       *output   [[buffer(2)]],
    constant uint     &dim      [[buffer(3)]],
    constant float    &eps      [[buffer(4)]],
    uint tid                    [[thread_position_in_threadgroup]],
    uint tcount                 [[threads_per_threadgroup]],
    uint gid                    [[threadgroup_position_in_grid]],
    uint simd_id                [[simdgroup_index_in_threadgroup]],
    uint simd_lane              [[thread_index_in_simdgroup]]
) {
    threadgroup float simd_sums[32];
    threadgroup float final_rms;
    uint row_offset = gid * dim;
    uint num_sg = (tcount + 31) / 32;

    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        local_sum += val * val;
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) simd_sums[simd_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        sum = (simd_lane < num_sg) ? simd_sums[simd_lane] : 0.0f;
        sum = simd_sum(sum);
        final_rms = rsqrt(sum / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = final_rms;
    for (uint i = tid; i < dim; i += tcount) {
        float val = float(input[row_offset + i]);
        float w = float(weight[i]);
        output[row_offset + i] = half(val * rms * (1.0f + w));
    }
}
