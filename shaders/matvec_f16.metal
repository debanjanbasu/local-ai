#include <metal_stdlib>
using namespace metal;

/// FP16 matrix-vector multiply: output = matrix × vector
/// matrix: [rows, cols] row-major (FP16)
/// vector: [cols] (FP16)
/// output: [rows] (FP16)
///
/// One simdgroup (32 lanes) computes one output row; a threadgroup packs
/// `tcount / 32` rows. The dot product is reduced with a single `simd_sum`,
/// so there is no threadgroup memory and no barrier. This keeps small PLE
/// matrices (e.g. cols=256 down-proj) from over-subscribing 256 threads per
/// row and paying a cross-simdgroup barrier on every output element.
kernel void matvec_f16(
    device const half *matrix [[buffer(0)]],
    device const half *vector [[buffer(1)]],
    device half       *output [[buffer(2)]],
    constant uint     &rows   [[buffer(3)]],
    constant uint     &cols   [[buffer(4)]],
    uint tg_id                [[threadgroup_position_in_grid]],
    uint tcount               [[threads_per_threadgroup]],
    uint simd_id              [[simdgroup_index_in_threadgroup]],
    uint simd_lane            [[thread_index_in_simdgroup]]
) {
    uint rows_per_tg = tcount / 32u;
    uint row = tg_id * rows_per_tg + simd_id;
    if (row >= rows) return;

    uint row_offset = row * cols;
    float local_sum = 0.0f;
    uint cols4 = cols / 4;
    device const half4 *mat4 = (device const half4 *)(matrix + row_offset);
    device const half4 *vec4 = (device const half4 *)vector;
    for (uint j = simd_lane; j < cols4; j += 32u) {
        half4 m = mat4[j];
        half4 v = vec4[j];
        local_sum += float(m.x) * float(v.x) + float(m.y) * float(v.y)
                   + float(m.z) * float(v.z) + float(m.w) * float(v.w);
    }
    uint remainder_start = cols4 * 4;
    for (uint j = remainder_start + simd_lane; j < cols; j += 32u) {
        local_sum += float(matrix[row_offset + j]) * float(vector[j]);
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        output[row] = half(sum);
    }
}

/// FP16 weights × FP16 vector → float32 output.
///
/// Used for MLP gate/up projections where the intermediate result feeds into
/// GELU and elementwise multiply that can overflow FP16 (Gemma 4 was designed
/// for BF16). Keeping the output in float32 avoids precision loss and overflow.
kernel void matvec_f16w_f32out(
    device const half  *matrix [[buffer(0)]],
    device const half  *vector [[buffer(1)]],
    device float       *output [[buffer(2)]],
    constant uint      &rows   [[buffer(3)]],
    constant uint      &cols   [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint tcount                [[threads_per_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint rows_per_tg = tcount / 32u;
    uint row = tg_id * rows_per_tg + simd_id;
    if (row >= rows) return;

    uint row_offset = row * cols;
    float local_sum = 0.0f;
    uint cols4 = cols / 4;
    device const half4 *mat4 = (device const half4 *)(matrix + row_offset);
    device const half4 *vec4 = (device const half4 *)vector;
    for (uint j = simd_lane; j < cols4; j += 32u) {
        half4 m = mat4[j];
        half4 v = vec4[j];
        local_sum += float(m.x) * float(v.x) + float(m.y) * float(v.y)
                   + float(m.z) * float(v.z) + float(m.w) * float(v.w);
    }
    uint remainder_start = cols4 * 4;
    for (uint j = remainder_start + simd_lane; j < cols; j += 32u) {
        local_sum += float(matrix[row_offset + j]) * float(vector[j]);
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        output[row] = sum;
    }
}

/// FP16 weights × float32 vector → FP16 output.
///
/// Used for the MLP down projection: the input vector is the float32
/// gate*up product, but the output (which feeds into post-FF norm and
/// residual) can safely be stored as FP16.
kernel void matvec_f16w_f32in(
    device const half  *matrix [[buffer(0)]],
    device const float *vector [[buffer(1)]],
    device half        *output [[buffer(2)]],
    constant uint      &rows   [[buffer(3)]],
    constant uint      &cols   [[buffer(4)]],
    uint tg_id                 [[threadgroup_position_in_grid]],
    uint tcount                [[threads_per_threadgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]]
) {
    uint rows_per_tg = tcount / 32u;
    uint row = tg_id * rows_per_tg + simd_id;
    if (row >= rows) return;

    uint row_offset = row * cols;
    float local_sum = 0.0f;
    uint cols4 = cols / 4;
    device const half4 *mat4 = (device const half4 *)(matrix + row_offset);
    device const float4 *vec4 = (device const float4 *)vector;
    for (uint j = simd_lane; j < cols4; j += 32u) {
        half4 m = mat4[j];
        float4 v = vec4[j];
        local_sum += float(m.x) * v.x + float(m.y) * v.y
                   + float(m.z) * v.z + float(m.w) * v.w;
    }
    uint remainder_start = cols4 * 4;
    for (uint j = remainder_start + simd_lane; j < cols; j += 32u) {
        local_sum += float(matrix[row_offset + j]) * vector[j];
    }

    float sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        output[row] = half(sum);
    }
}
