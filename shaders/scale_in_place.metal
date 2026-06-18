#include <metal_stdlib>
using namespace metal;

/// In-place FP16 scaling: data[i] *= scalar
kernel void scale_in_place(
    device half *data          [[buffer(0)]],
    constant float &scalar     [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        data[tid] = half(float(data[tid]) * scalar);
    }
}

/// Strided f16 gather: dst[i*pld + p] = src[i*row_stride + base_off + p].
/// Used to extract one layer's PLE slice from the packed
/// `[n_rows × n_layer × pld]` batch tensor on the GPU, replacing a per-layer
/// CPU copy into shared scratch (which forced a command-buffer commit per layer
/// in the multi-row verify path).
kernel void gather_strided_f16(
    device const half *src     [[buffer(0)]],
    device half       *dst     [[buffer(1)]],
    constant uint &row_stride  [[buffer(2)]],
    constant uint &base_off    [[buffer(3)]],
    constant uint &pld         [[buffer(4)]],
    constant uint &n_rows      [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = n_rows * pld;
    if (gid >= total) return;
    uint i = gid / pld;
    uint p = gid - i * pld;
    dst[gid] = src[i * row_stride + base_off + p];
}

/// In-place Gemma final-logit soft-capping: data[i] = cap * tanh(data[i] / cap).
/// Mirrors the CPU `apply_logit_softcap` so device-side application is exact.
kernel void logit_softcap_f32(
    device float *data         [[buffer(0)]],
    constant float &cap        [[buffer(1)]],
    constant uint &count       [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        data[tid] = cap * tanh(data[tid] / cap);
    }
}
