#include <metal_stdlib>
using namespace metal;

/// Elementwise FP16 addition: output[i] = a[i] + b[i]
kernel void residual_add(
    device const half *a       [[buffer(0)]],
    device const half *b       [[buffer(1)]],
    device half *output        [[buffer(2)]],
    constant uint &count       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = a[tid] + b[tid];
    }
}
