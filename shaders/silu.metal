#include <metal_stdlib>
using namespace metal;

/// In-place SiLU activation: x = x / (1 + exp(-x))
kernel void silu_inplace(
    device half *data          [[buffer(0)]],
    constant uint &count       [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(data[tid]);
        float s = x / (1.0f + exp(-x));
        data[tid] = half(s);
    }
}

/// Out-of-place SiLU activation: output = input / (1 + exp(-input))
kernel void silu(
    device const half *input   [[buffer(0)]],
    device half *output        [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(input[tid]);
        float s = x / (1.0f + exp(-x));
        output[tid] = half(s);
    }
}
