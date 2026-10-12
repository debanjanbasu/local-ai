// EmbeddingGemma 2 text encoder kernels (see local-metal/src/embedding.rs).
//
// Everything is F32: activations, accumulation and softmax (the model card
// forbids F16 activations: they overflow). Matrices keep the checkpoint's
// BF16 values and widen to F32 exactly.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- reductions

// Sum of `value` over a 128-thread threadgroup, returned in every thread.
static inline float eg_sum_128(float value, threadgroup float *partial, uint tid) {
    value = simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// RMSNorm as the reference computes it: x * pow(mean(x^2) + eps, -0.5) * w.
// One 128-thread threadgroup per row; `scale` multiplies the input first (the
// PLE projection's 1/sqrt(512)). In place is allowed: each thread rewrites
// only the elements it read.
kernel void eg_rms(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &dimension [[buffer(3)]],
    constant uint &has_weight [[buffer(4)]], constant uint &weight_period [[buffer(5)]],
    constant float &scale [[buffer(6)]], constant float &epsilon [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *x = input + ulong(row) * dimension;
    device float *y = output + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float v = x[i] * scale;
        sum += v * v;
    }
    const float total = eg_sum_128(sum, partial, tid);
    const float inv = precise::pow(total / float(dimension) + epsilon, -0.5f);
    for (uint i = tid; i < dimension; i += 128) {
        const float v = x[i] * scale * inv;
        y[i] = has_weight != 0 ? v * weight[(i % weight_period)] : v;
    }
}

// hidden = (hidden + rms(branch) * weight) * scalar, one threadgroup per row.
kernel void eg_rms_add(
    device float *hidden [[buffer(0)]], device const float *branch [[buffer(1)]],
    device const float *weight [[buffer(2)]], constant uint &dimension [[buffer(3)]],
    constant float &scalar [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *b = branch + ulong(row) * dimension;
    device float *h = hidden + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) sum += b[i] * b[i];
    const float total = eg_sum_128(sum, partial, tid);
    const float inv = precise::pow(total / float(dimension) + epsilon, -0.5f);
    for (uint i = tid; i < dimension; i += 128) {
        h[i] = (h[i] + b[i] * inv * weight[i]) * scalar;
    }
}

// Rotary embedding (rotate_half convention) in place on [tokens, heads, dim]
// rows; `positions[t]` is the token's position inside its own sequence. One
// thread per (row, pair). Frequencies follow the reference: inv_freq =
// 1 / theta^(2i / dim) and angle = position * inv_freq, both in F32.
kernel void eg_rope(
    device float *x [[buffer(0)]], device const uint *positions [[buffer(1)]],
    constant uint &heads [[buffer(2)]], constant uint &dimension [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant float &theta [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint half_dim = dimension / 2;
    const uint row = gid / half_dim, i = gid % half_dim;
    if (row >= rows) return;
    const uint token = row / heads;
    const float inv_freq = 1.0f / precise::pow(theta, float(2 * i) / float(dimension));
    const float angle = float(positions[token]) * inv_freq;
    const float c = precise::cos(angle), s = precise::sin(angle);
    device float *v = x + ulong(row) * dimension;
    const float a = v[i], b = v[i + half_dim];
    v[i] = a * c - b * s;
    v[i + half_dim] = b * c + a * s;
}

static inline float eg_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
}

// out[t, c] = gelu_tanh(gate[t, c]) * other[t * other_stride + other_offset + c]
// for `columns` columns of `rows` rows.
kernel void eg_gelu_mul(
    device const float *gate [[buffer(0)]], device const float *other [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &other_stride [[buffer(5)]],
    constant uint &other_offset [[buffer(6)]], uint gid [[thread_position_in_grid]]
) {
    const uint row = gid / columns, c = gid % columns;
    if (row >= rows) return;
    output[gid] = eg_gelu_tanh(gate[gid]) * other[ulong(row) * other_stride + other_offset + c];
}

// ---------------------------------------------------------------- attention

// Bidirectional attention over packed sequences, F32 throughout, scale 1.0.
// q [tokens, q_heads, dim], k/v [tokens, kv_heads, dim], out like q.
// `bounds[t]` = (first, end) token indices of t's sequence; keys j of that
// sequence with |t - j| <= window are attended (window 0: all of them).
// One threadgroup per token, one SIMD group per query head; each lane owns
// dim / 32 consecutive dimensions (dim is 256 or 512). Online softmax.
template<uint E>
static inline void eg_attention_impl(
    device const float *q, device const float *k, device const float *v, device float *out,
    device const uint2 *bounds, uint q_heads, uint kv_heads, uint window, uint token,
    uint head, uint lane
) {
    const uint dim = E * 32;
    const uint group = q_heads / kv_heads;
    const uint kvh = head / group;
    const uint2 b = bounds[token];
    uint first = b.x, end = b.y;
    if (window != 0) {
        first = max(first, token >= window ? token - window : 0u);
        end = min(end, token + window + 1);
    }
    float qv[E], acc[E];
    device const float *qp = q + (ulong(token) * q_heads + head) * dim + lane * E;
    #pragma clang loop unroll(full)
    for (uint e = 0; e < E; ++e) {
        qv[e] = qp[e];
        acc[e] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    const ulong kv_stride = ulong(kv_heads) * dim;
    device const float *kp = k + ulong(kvh) * dim + lane * E;
    device const float *vp = v + ulong(kvh) * dim + lane * E;
    for (uint j = first; j < end; ++j) {
        device const float4 *k4 = reinterpret_cast<device const float4 *>(kp + j * kv_stride);
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < E / 4; ++e) {
            const float4 kk = k4[e];
            s += qv[4 * e] * kk.x + qv[4 * e + 1] * kk.y + qv[4 * e + 2] * kk.z +
                 qv[4 * e + 3] * kk.w;
        }
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float correction = precise::exp(m - m_new);
        const float p = precise::exp(s - m_new);
        l = l * correction + p;
        device const float4 *v4 = reinterpret_cast<device const float4 *>(vp + j * kv_stride);
        #pragma clang loop unroll(full)
        for (uint e = 0; e < E / 4; ++e) {
            const float4 vv = v4[e];
            acc[4 * e] = acc[4 * e] * correction + p * vv.x;
            acc[4 * e + 1] = acc[4 * e + 1] * correction + p * vv.y;
            acc[4 * e + 2] = acc[4 * e + 2] * correction + p * vv.z;
            acc[4 * e + 3] = acc[4 * e + 3] * correction + p * vv.w;
        }
        m = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    device float *op = out + (ulong(token) * q_heads + head) * dim + lane * E;
    #pragma clang loop unroll(full)
    for (uint e = 0; e < E; ++e) op[e] = acc[e] * inv;
}

#define EG_ATTENTION(NAME, E) \
kernel void NAME( \
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]], \
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]], \
    device const uint2 *bounds [[buffer(4)]], constant uint &q_heads [[buffer(5)]], \
    constant uint &kv_heads [[buffer(6)]], constant uint &window [[buffer(7)]], \
    uint token [[threadgroup_position_in_grid]], uint head [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    eg_attention_impl<E>(q, k, v, out, bounds, q_heads, kv_heads, window, token, head, lane); \
}

EG_ATTENTION(eg_attention_256, 8)
EG_ATTENTION(eg_attention_512, 16)

// ---------------------------------------------------------------- BF16 GEMM

// Y[t, r] = sum_k W[r, k] * X[t, k] for BF16 W [rows, columns] (row-major,
// the checkpoint's own values) and F32 X [tokens, columns]; columns % 32 == 0.
//
// A 128-thread threadgroup owns a 32-token x 32-row tile of Y; each SIMD group
// a 16 x 16 quarter as 2 x 2 F32 8x8 simdgroup matrices. Per 32-column step
// the threadgroup stages X and W (BF16 widened to F32, exact, transposed to
// [k][row]) in threadgroup memory. Grid: (ceil(rows / 32), ceil(tokens / 32)).
constant constexpr uint EG_TILE = 32;
constant constexpr uint EG_PAD = 4;

static inline float eg_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void eg_bf16_gemm(
    device const ushort *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[EG_TILE][EG_TILE + EG_PAD];
    threadgroup float ws[EG_TILE][EG_TILE + EG_PAD];
    threadgroup float cs[EG_TILE][EG_TILE + EG_PAD];
    const uint row0 = group.x * EG_TILE, token0 = group.y * EG_TILE;
    // Staging assignment: thread -> (tile line, 8 consecutive k).
    const uint line = tid / 4, k8 = (tid % 4) * 8;
    const uint token = token0 + line, row = row0 + line;
    const bool token_live = token < tokens;
    device const float *x_line = input + ulong(min(token, tokens - 1)) * columns + k8;
    device const ushort *w_line = weights + ulong(min(row, rows - 1)) * columns + k8;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < columns; k0 += EG_TILE) {
        const float4 a0 = token_live ? *reinterpret_cast<device const float4 *>(x_line + k0) : 0.0f;
        const float4 a1 =
            token_live ? *reinterpret_cast<device const float4 *>(x_line + k0 + 4) : 0.0f;
        const ushort4 b0 = *reinterpret_cast<device const ushort4 *>(w_line + k0);
        const ushort4 b1 = *reinterpret_cast<device const ushort4 *>(w_line + k0 + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[line][k8 + 0] = a0.x; xs[line][k8 + 1] = a0.y;
        xs[line][k8 + 2] = a0.z; xs[line][k8 + 3] = a0.w;
        xs[line][k8 + 4] = a1.x; xs[line][k8 + 5] = a1.y;
        xs[line][k8 + 6] = a1.z; xs[line][k8 + 7] = a1.w;
        ws[k8 + 0][line] = eg_bf16(b0.x); ws[k8 + 1][line] = eg_bf16(b0.y);
        ws[k8 + 2][line] = eg_bf16(b0.z); ws[k8 + 3][line] = eg_bf16(b0.w);
        ws[k8 + 4][line] = eg_bf16(b1.x); ws[k8 + 5][line] = eg_bf16(b1.y);
        ws[k8 + 6][line] = eg_bf16(b1.z); ws[k8 + 7][line] = eg_bf16(b1.w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < EG_TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], &xs[sm + 8 * i][kk], EG_TILE + EG_PAD);
                simdgroup_load(b[i], &ws[kk][sn + 8 * i], EG_TILE + EG_PAD);
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
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], EG_TILE + EG_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread -> (token line, 8 consecutive rows).
    const uint r8 = (tid % 4) * 8;
    if (!token_live) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8; ++r) {
        const uint out_row = row0 + r8 + r;
        if (out_row < rows) output[ulong(token) * rows + out_row] = cs[line][r8 + r];
    }
}
