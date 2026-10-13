// Whisper large-v3 kernels (see local-metal/src/whisper.rs).
//
// Matrices keep the checkpoint's F16 values and widen to F32 exactly in the
// kernel; activations, accumulation, LayerNorm and softmax are F32, like the
// FP32 reference (openai-whisper on the CPU). Activations are row-major
// [tokens, width] F32.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- helpers

// erfc(x) with fractional error below 1.2e-7 everywhere (Numerical Recipes'
// Chebyshev fit), enough for F32 GELU.
static inline float wh_erfc(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float r = t * precise::exp(-z * z - 1.26551223f +
        t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
        t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f +
        t * 0.17087277f)))))))));
    return x >= 0.0f ? r : 2.0f - r;
}

// Exact (erf) GELU, PyTorch's nn.GELU default: 0.5 x (1 + erf(x / sqrt 2)).
static inline float wh_gelu(float x) {
    return 0.5f * x * wh_erfc(-x * 0.70710678118654752f);
}

constant constexpr uint WH_BIAS = 1;
constant constexpr uint WH_GELU = 2;
constant constexpr uint WH_RESIDUAL = 4;

// y = act(acc + bias) + residual, per the flags.
static inline float wh_epilogue(float acc, uint flags, device const float *bias, uint row,
                                device const float *residual, ulong index) {
    float v = acc;
    if ((flags & WH_BIAS) != 0) v += bias[row];
    if ((flags & WH_GELU) != 0) v = wh_gelu(v);
    if ((flags & WH_RESIDUAL) != 0) v += residual[index];
    return v;
}

// Sum and max over a 256-thread threadgroup (8 SIMD groups), in every thread.
static inline float wh_sum_256(float value, threadgroup float *partial, uint simd, uint lane) {
    value = simd_sum(value);
    if (lane == 0) partial[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

static inline float wh_max_256(float value, threadgroup float *partial, uint simd, uint lane) {
    value = simd_max(value);
    if (lane == 0) partial[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = partial[0];
    for (uint i = 1; i < 8; ++i) total = max(total, partial[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ---------------------------------------------------------------- LayerNorm

// y = (x - mean) / sqrt(var + eps) * w + b (biased variance), one 256-thread
// threadgroup per row. In place is allowed.
kernel void wh_layernorm(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &dimension [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    device const float *x = input + ulong(row) * dimension;
    device float *y = output + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 256) sum += x[i];
    const float mean = wh_sum_256(sum, partial, simd, lane) / float(dimension);
    float sq = 0.0f;
    for (uint i = tid; i < dimension; i += 256) {
        const float d = x[i] - mean;
        sq += d * d;
    }
    const float var = wh_sum_256(sq, partial, simd, lane) / float(dimension);
    const float inv = 1.0f / precise::sqrt(var + epsilon);
    for (uint i = tid; i < dimension; i += 256) {
        y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

// ---------------------------------------------------------------- im2col

// Kernel-3, padding-1 Conv1d input columns: out[t, c * 3 + k] =
// in[t * stride + k - 1, c] (zero outside), matching a [out, in, 3] weight
// flattened to [out, in * 3].
kernel void wh_im2col3(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &channels [[buffer(2)]], constant uint &t_in [[buffer(3)]],
    constant uint &t_out [[buffer(4)]], constant uint &stride [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint width = channels * 3;
    if (gid >= t_out * width) return;
    const uint t = gid / width, rem = gid % width;
    const uint c = rem / 3, k = rem % 3;
    const int src = int(t * stride + k) - 1;
    output[gid] = (src >= 0 && uint(src) < t_in) ? input[ulong(src) * channels + c] : 0.0f;
}

// ---------------------------------------------------------------- F16 GEMM

// Y[t, r] = epilogue(sum_k W[r, k] * X[t, k]) for F16 W [rows, columns]
// (row-major, the checkpoint's values) and F32 X [tokens, columns];
// columns % 32 == 0. 32-token x 32-row tiles, 128 threads, 2 x 2 8x8 F32
// simdgroup matrices per SIMD group. Grid: (ceil(rows / 32), ceil(tokens / 32)).
constant constexpr uint WH_TILE = 32;
constant constexpr uint WH_PAD = 4;

kernel void wh_gemm(
    device const half *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const float *bias [[buffer(3)]],
    device const float *residual [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]], constant uint &tokens [[buffer(7)]],
    constant uint &flags [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[WH_TILE][WH_TILE + WH_PAD];
    threadgroup float ws[WH_TILE][WH_TILE + WH_PAD];
    threadgroup float cs[WH_TILE][WH_TILE + WH_PAD];
    const uint row0 = group.x * WH_TILE, token0 = group.y * WH_TILE;
    const uint line = tid / 4, k8 = (tid % 4) * 8;
    const uint token = token0 + line, row = row0 + line;
    const bool token_live = token < tokens;
    device const float *x_line = input + ulong(min(token, tokens - 1)) * columns + k8;
    device const half *w_line = weights + ulong(min(row, rows - 1)) * columns + k8;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < columns; k0 += WH_TILE) {
        const float4 a0 = token_live ? *reinterpret_cast<device const float4 *>(x_line + k0) : 0.0f;
        const float4 a1 =
            token_live ? *reinterpret_cast<device const float4 *>(x_line + k0 + 4) : 0.0f;
        const half4 b0 = *reinterpret_cast<device const half4 *>(w_line + k0);
        const half4 b1 = *reinterpret_cast<device const half4 *>(w_line + k0 + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[line][k8 + 0] = a0.x; xs[line][k8 + 1] = a0.y;
        xs[line][k8 + 2] = a0.z; xs[line][k8 + 3] = a0.w;
        xs[line][k8 + 4] = a1.x; xs[line][k8 + 5] = a1.y;
        xs[line][k8 + 6] = a1.z; xs[line][k8 + 7] = a1.w;
        ws[k8 + 0][line] = float(b0.x); ws[k8 + 1][line] = float(b0.y);
        ws[k8 + 2][line] = float(b0.z); ws[k8 + 3][line] = float(b0.w);
        ws[k8 + 4][line] = float(b1.x); ws[k8 + 5][line] = float(b1.y);
        ws[k8 + 6][line] = float(b1.z); ws[k8 + 7][line] = float(b1.w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < WH_TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], &xs[sm + 8 * i][kk], WH_TILE + WH_PAD);
                simdgroup_load(b[i], &ws[kk][sn + 8 * i], WH_TILE + WH_PAD);
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], WH_TILE + WH_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r8 = (tid % 4) * 8;
    if (!token_live) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8; ++r) {
        const uint out_row = row0 + r8 + r;
        if (out_row < rows) {
            const ulong index = ulong(token) * rows + out_row;
            output[index] = wh_epilogue(cs[line][r8 + r], flags, bias, out_row, residual, index);
        }
    }
}

// ---------------------------------------------------------------- F16 GEMV

// y[r] = epilogue(sum_k W[r, k] * x[k]) for one F32 vector; columns % 128 == 0.
// 128 threads (4 SIMD groups) per threadgroup, 2 rows per SIMD group.
// Grid: ceil(rows / 8).
kernel void wh_gemv(
    device const half *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const float *bias [[buffer(3)]],
    device const float *residual [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]], constant uint &flags [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = group * 8 + simd * 2;
    if (row0 >= rows) return;
    const uint row1 = min(row0 + 1, rows - 1);
    device const half4 *w0 = reinterpret_cast<device const half4 *>(weights + ulong(row0) * columns);
    device const half4 *w1 = reinterpret_cast<device const half4 *>(weights + ulong(row1) * columns);
    device const float4 *x4 = reinterpret_cast<device const float4 *>(input);
    float acc0 = 0.0f, acc1 = 0.0f;
    const uint n4 = columns / 4;
    for (uint c = lane; c < n4; c += 32) {
        const float4 xv = x4[c];
        acc0 += dot(float4(w0[c]), xv);
        acc1 += dot(float4(w1[c]), xv);
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        output[row0] = wh_epilogue(acc0, flags, bias, row0, residual, row0);
        if (row0 + 1 < rows) {
            output[row0 + 1] = wh_epilogue(acc1, flags, bias, row0 + 1, residual, row0 + 1);
        }
    }
}

// ---------------------------------------------------------------- attention

// Encoder self-attention (bidirectional), head dim 64, F32. q/k/v/out are
// [tokens, heads * 64]. One thread per query: 128 queries of one head per
// threadgroup share 32-key K/V tiles in threadgroup memory; softmax is online
// per tile. Queries q_start .. q_start + q_count; grid (ceil(q_count / 128), heads).
kernel void wh_encoder_attention(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &q_start [[buffer(6)]], constant uint &q_count [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 ks[32][16];
    threadgroup float4 vs[32][16];
    const uint head = group.y;
    const uint local = group.x * 128 + tid;
    const bool live = local < q_count;
    const uint query = q_start + min(local, q_count - 1);
    const ulong stride = ulong(heads) * 64;
    device const float4 *qp = reinterpret_cast<device const float4 *>(q + query * stride + head * 64);
    float4 qv[16], acc[16];
    #pragma clang loop unroll(full)
    for (uint e = 0; e < 16; ++e) {
        qv[e] = qp[e] * scale;
        acc[e] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    for (uint j0 = 0; j0 < tokens; j0 += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 512; i += 128) {
            const uint key = j0 + i / 16, e = i % 16;
            const bool ok = key < tokens;
            const ulong base = ulong(min(key, tokens - 1)) * stride + head * 64;
            const float4 kk = reinterpret_cast<device const float4 *>(k + base)[e];
            const float4 vv = reinterpret_cast<device const float4 *>(v + base)[e];
            ks[i / 16][e] = ok ? kk : 0.0f;
            vs[i / 16][e] = ok ? vv : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint n = min(32u, tokens - j0);
        float s[32];
        float tile_max = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            float d = 0.0f;
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 16; ++e) d += dot(qv[e], ks[j][e]);
            s[j] = j < n ? d : -INFINITY;
            tile_max = max(tile_max, s[j]);
        }
        const float m_new = max(m, tile_max);
        const float correction = precise::exp(m - m_new);
        l *= correction;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 16; ++e) acc[e] *= correction;
        for (uint j = 0; j < n; ++j) {
            const float p = precise::exp(s[j] - m_new);
            l += p;
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 16; ++e) acc[e] += p * vs[j][e];
        }
        m = m_new;
    }
    if (!live) return;
    const float inv = 1.0f / l;
    device float4 *op = reinterpret_cast<device float4 *>(out + query * stride + head * 64);
    #pragma clang loop unroll(full)
    for (uint e = 0; e < 16; ++e) op[e] = acc[e] * inv;
}

// Decoder attention for a few queries, head dim 64, F32: self-attention over
// the KV cache (causal: query i sees keys 0 ..= key_base + i) or
// cross-attention over the encoder output (n_keys keys). q/out
// [queries, heads * 64], k/v [keys, heads * 64]. One 256-thread threadgroup
// per (query, head); grid (queries, heads). At most 1,504 keys.
constant constexpr uint WH_MAX_KEYS = 1504;

kernel void wh_decoder_attention(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &heads [[buffer(4)]], constant uint &key_base [[buffer(5)]],
    constant uint &causal [[buffer(6)]], constant uint &n_keys [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 qs[16];
    threadgroup float scores[WH_MAX_KEYS];
    threadgroup float partial[8];
    threadgroup float parts[4][64];
    const uint i = group.x, head = group.y;
    const ulong stride = ulong(heads) * 64;
    const uint keys = min(causal != 0 ? key_base + i + 1 : n_keys, WH_MAX_KEYS);
    if (tid < 16) {
        qs[tid] = reinterpret_cast<device const float4 *>(q + i * stride + head * 64)[tid] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = -INFINITY;
    for (uint j = tid; j < keys; j += 256) {
        device const float4 *kp = reinterpret_cast<device const float4 *>(k + j * stride + head * 64);
        float d = 0.0f;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 16; ++e) d += dot(qs[e], kp[e]);
        scores[j] = d;
        local_max = max(local_max, d);
    }
    const float mx = wh_max_256(local_max, partial, simd, lane);
    float local_sum = 0.0f;
    for (uint j = tid; j < keys; j += 256) {
        const float p = precise::exp(scores[j] - mx);
        scores[j] = p;
        local_sum += p;
    }
    const float total = wh_sum_256(local_sum, partial, simd, lane);
    const uint d = tid % 64, g = tid / 64;
    float acc = 0.0f;
    for (uint j = g; j < keys; j += 4) acc += scores[j] * v[j * stride + head * 64 + d];
    parts[g][d] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64) {
        out[i * stride + head * 64 + tid] =
            (parts[0][tid] + parts[1][tid] + parts[2][tid] + parts[3][tid]) / total;
    }
}
