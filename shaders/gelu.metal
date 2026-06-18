#include <metal_stdlib>
using namespace metal;

kernel void gelu_tanh(
    device const half *input  [[buffer(0)]],
    device half       *output [[buffer(1)]],
    constant uint     &count  [[buffer(2)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = float(input[gid]);
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    // Clamp inner to prevent tanh overflow: exp(2*inner) overflows float32
    // when |inner| > ~44. tanh(10) ≈ 1.0 in float32, so clamping is lossless.
    float t = tanh(clamp(inner, -10.0f, 10.0f));
    float result = 0.5f * x * (1.0f + t);
    output[gid] = half(result);
}

/// Fused GeGLU elementwise: output[i] = gelu(a[i]) * b[i].
/// Collapses the separate gelu + elementwise_mul dispatches into one.
kernel void gelu_mul_f16(
    device const half *a      [[buffer(0)]],
    device const half *b      [[buffer(1)]],
    device half       *output [[buffer(2)]],
    constant uint     &count  [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = float(a[gid]);
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float t = tanh(clamp(inner, -10.0f, 10.0f));
    float g = 0.5f * x * (1.0f + t);
    output[gid] = half(g * float(b[gid]));
}

/// Float32 in/out GELU (tanh approximation).
///
/// Used for MLP intermediate path where activations stay in float32
/// to avoid FP16 overflow in downstream elementwise multiply.
kernel void gelu_tanh_f32(
    device const float *input  [[buffer(0)]],
    device float       *output [[buffer(1)]],
    constant uint      &count  [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    float x = input[gid];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float t = tanh(clamp(inner, -10.0f, 10.0f));
    output[gid] = 0.5f * x * (1.0f + t);
}
