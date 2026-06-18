#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul_f16(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *out      [[buffer(2)]],
    constant uint &count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        out[tid] = a[tid] * b[tid];
    }
}

/// Float32 elementwise multiply.
///
/// Used for MLP gate*up where both inputs are float32 (from matvec_f16w_f32out
/// and gelu_tanh_f32). Avoids FP16 overflow — Gemma 4 intermediate activations
/// can produce products exceeding FP16 max (65504).
kernel void elementwise_mul_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out      [[buffer(2)]],
    constant uint &count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        out[tid] = a[tid] * b[tid];
    }
}
