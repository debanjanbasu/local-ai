#include <metal_stdlib>
using namespace metal;

/// Clipped linear: y = clamp(clamp(x, in_min, in_max) @ W^T + bias, out_min, out_max)
/// Input: [M, K] FP16, Weight: [N, K] FP16, Bias: [N] FP16 (optional)
/// Output: [M, N] FP16
///
/// Grid: threadgroups=[N, M], threads_per_tg=[1,1]
/// Each thread computes one output element.
kernel void clipped_linear(
    device const half *input     [[buffer(0)]],
    device const half *weight    [[buffer(1)]],
    device const half *bias      [[buffer(2)]],
    device half       *output    [[buffer(3)]],
    constant uint     &M         [[buffer(4)]],
    constant uint     &K         [[buffer(5)]],
    constant uint     &N         [[buffer(6)]],
    constant float    &input_min [[buffer(7)]],
    constant float    &input_max [[buffer(8)]],
    constant float    &output_min [[buffer(9)]],
    constant float    &output_max [[buffer(10)]],
    constant uint     &has_bias  [[buffer(11)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint col = gid.x; // output column [0, N)
    uint row = gid.y; // output row [0, M)

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float x = clamp(float(input[row * K + k]), input_min, input_max);
        float w = float(weight[col * K + k]);
        acc += x * w;
    }
    if (has_bias != 0) {
        acc += float(bias[col]);
    }
    acc = clamp(acc, output_min, output_max);
    output[row * N + col] = half(acc);
}

/// Standard matmul with optional bias (no clamping): y = x @ W^T + bias
/// Input: [M, K] FP16, Weight: [N, K] FP16, Bias: [N] FP16 (optional)
/// Output: [M, N] FP16
kernel void matmul_f16_bias(
    device const half *input     [[buffer(0)]],
    device const half *weight    [[buffer(1)]],
    device const half *bias      [[buffer(2)]],
    device half       *output    [[buffer(3)]],
    constant uint     &M         [[buffer(4)]],
    constant uint     &K         [[buffer(5)]],
    constant uint     &N         [[buffer(6)]],
    constant uint     &has_bias  [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint col = gid.x; // [0, N)
    uint row = gid.y; // [0, M)

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += float(input[row * K + k]) * float(weight[col * K + k]);
    }
    if (has_bias != 0) {
        acc += float(bias[col]);
    }
    output[row * N + col] = half(acc);
}
