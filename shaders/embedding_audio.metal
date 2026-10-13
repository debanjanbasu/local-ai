// EmbeddingGemma 2 audio encoder kernels (`gemma4_audio` Conformer; see
// local-metal/src/embedding_audio.rs). F32 activations, accumulation and
// softmax throughout; matrices are the checkpoint's BF16 values widened
// exactly, small kernels (convolutions, norms) are F32.
#include <metal_stdlib>
using namespace metal;

static inline float ea_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline float ea_silu(float x) {
    return x / (1.0f + precise::exp(-x));
}

static inline float ea_sum_128(float value, threadgroup float *partial, uint tid) {
    value = simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// ---------------------------------------------------------------- subsampling stem

// 3x3 stride-2 convolution with padding 1 over a channel-last image
// [in_t][in_f][in_c] -> [out_t][out_f][out_c] (no bias), as torch Conv2d over
// (time, frequency) with weights [out_c][in_c][3][3]. Output rows t >=
// `out_rows` are skipped; `out` is offset by the caller (packed rows).
kernel void ea_conv2d_s2(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint4 &in_shape [[buffer(3)]],
    constant uint4 &out_shape [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    // in_shape = (t, f, c, -), out_shape = (rows to write, f, c, -)
    const uint out_c = out_shape.z, out_f = out_shape.y;
    const uint c = gid % out_c, f = (gid / out_c) % out_f, t = gid / (out_c * out_f);
    if (t >= out_shape.x) return;
    const uint in_t = in_shape.x, in_f = in_shape.y, in_c = in_shape.z;
    float acc = 0.0f;
    for (uint dt = 0; dt < 3; ++dt) {
        const int st = int(2 * t + dt) - 1;
        if (st < 0 || st >= int(in_t)) continue;
        for (uint df = 0; df < 3; ++df) {
            const int sf = int(2 * f + df) - 1;
            if (sf < 0 || sf >= int(in_f)) continue;
            device const float *x = input + (ulong(st) * in_f + uint(sf)) * in_c;
            device const float *w = weight + ulong(c) * in_c * 9 + dt * 3 + df;
            for (uint ci = 0; ci < in_c; ++ci) acc += x[ci] * w[ci * 9];
        }
    }
    output[gid] = acc;
}

// LayerNorm (no bias) over `dimension` channels then ReLU, in place, one
// 128-thread threadgroup per row; rows of time steps >= `valid_t` (with
// `rows_per_t` rows per step) are zeroed instead (the stem's frame mask).
kernel void ea_layer_norm_relu(
    device float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
    constant uint &dimension [[buffer(2)]], constant uint &rows_per_t [[buffer(3)]],
    constant uint &valid_t [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device float *v = x + ulong(row) * dimension;
    if (row / rows_per_t >= valid_t) {
        for (uint i = tid; i < dimension; i += 128) v[i] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) sum += v[i];
    const float mean = ea_sum_128(sum, partial, tid) / float(dimension);
    float var = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float d = v[i] - mean;
        var += d * d;
    }
    const float inv = precise::rsqrt(ea_sum_128(var, partial, tid) / float(dimension) + epsilon);
    for (uint i = tid; i < dimension; i += 128) v[i] = max((v[i] - mean) * inv * weight[i], 0.0f);
}

// ---------------------------------------------------------------- norms

// hidden += rms(branch) * weight * scale (Gemma4RMSNorm: x * pow(mean(x^2) +
// eps, -0.5) * weight), one 128-thread threadgroup per row.
kernel void ea_rms_add(
    device float *hidden [[buffer(0)]], device const float *branch [[buffer(1)]],
    device const float *weight [[buffer(2)]], constant uint &dimension [[buffer(3)]],
    constant float &scale [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *b = branch + ulong(row) * dimension;
    device float *h = hidden + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) sum += b[i] * b[i];
    const float inv = precise::pow(ea_sum_128(sum, partial, tid) / float(dimension) + epsilon, -0.5f);
    for (uint i = tid; i < dimension; i += 128) h[i] += b[i] * inv * weight[i] * scale;
}

// x = silu(rms(x) * weight) in place (the light-conv block's conv_norm + act).
kernel void ea_rms_silu(
    device float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
    constant uint &dimension [[buffer(2)]], constant float &epsilon [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device float *v = x + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) sum += v[i] * v[i];
    const float inv = precise::pow(ea_sum_128(sum, partial, tid) / float(dimension) + epsilon, -0.5f);
    for (uint i = tid; i < dimension; i += 128) v[i] = ea_silu(v[i] * inv * weight[i]);
}

// ---------------------------------------------------------------- light conv

// out[t, c] = sum_j w[c, j] * glu(t - (K-1) + j, c), glu(s, c) = a[s, c] *
// sigmoid(a[s, C + c]) over the [tokens, 2C] linear_start output, zero for
// steps before the token's sequence start (`first[t]`): GLU fused with the
// causal depthwise convolution (kernel K, left padding K-1).
kernel void ea_glu_dwconv(
    device const float *a [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], device const uint *first [[buffer(3)]],
    constant uint &channels [[buffer(4)]], constant uint &kernel_size [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], uint gid [[thread_position_in_grid]]
) {
    const uint t = gid / channels, c = gid % channels;
    if (t >= tokens) return;
    const uint start = first[t];
    float acc = 0.0f;
    for (uint j = 0; j < kernel_size; ++j) {
        const int s = int(t) - int(kernel_size - 1) + int(j);
        if (s < int(start)) continue;
        device const float *row = a + ulong(s) * 2 * channels;
        const float g = row[c] / (1.0f + precise::exp(-row[channels + c]));
        acc += weight[c * kernel_size + j] * g;
    }
    output[gid] = acc;
}

// ---------------------------------------------------------------- attention

// Chunked local self-attention with relative position logits, F32. q/k/v
// [tokens, heads, 128] (the clipped projections' outputs), out like q.
// Query t attends keys s of its own sequence (s >= first[t]) with
// 0 <= t - s <= max_distance:
//   logit = (q * q_scale) . (k * k_scale) + (q * q_scale) . rel[t - s]
//   logit = cap * tanh(logit / cap); softmax; out = sum p v.
// q_scale [128] = head_dim^-0.5 / ln 2 * softplus(per_dim_scale); rel
// [max_distance + 1, heads, 128] = relative_k_proj(sinusoid(distance)).
// One threadgroup per token, one SIMD group per head, 4 dims per lane.
kernel void ea_local_attention(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    device const uint *first [[buffer(4)]], device const float *q_scale [[buffer(5)]],
    device const float *rel [[buffer(6)]], constant uint &heads [[buffer(7)]],
    constant uint &max_distance [[buffer(8)]], constant float &k_scale [[buffer(9)]],
    constant float &cap [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]], uint head [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    const ulong stride = ulong(heads) * D;
    const uint d0 = lane * 4;
    const float4 qs = *reinterpret_cast<device const float4 *>(q_scale + d0);
    const float4 qv = *reinterpret_cast<device const float4 *>(q + token * stride + head * D + d0) * qs;
    const uint start = max(first[token], token >= max_distance ? token - max_distance : 0u);
    float m = -INFINITY, l = 0.0f;
    float4 acc = 0.0f;
    for (uint s = start; s <= token; ++s) {
        const float4 kk = *reinterpret_cast<device const float4 *>(k + s * stride + head * D + d0);
        const float4 rr = *reinterpret_cast<device const float4 *>(
            rel + (ulong(token - s) * heads + head) * D + d0);
        float logit = simd_sum(dot(qv, kk * k_scale)) + simd_sum(dot(qv, rr));
        logit = precise::tanh(logit / cap) * cap;
        const float m_new = max(m, logit);
        const float correction = precise::exp(m - m_new);
        const float p = precise::exp(logit - m_new);
        l = l * correction + p;
        const float4 vv = *reinterpret_cast<device const float4 *>(v + s * stride + head * D + d0);
        acc = acc * correction + p * vv;
        m = m_new;
    }
    *reinterpret_cast<device float4 *>(out + token * stride + head * D + d0) = acc / l;
}

// ---------------------------------------------------------------- clipped BF16 GEMM

// Y[t, r] = epilogue(sum_k W[r, k] * clamp(X[t, k], clip.x, clip.y)) for BF16
// W [rows, columns] and F32 X [tokens, columns]; columns % 32 == 0. The
// epilogue adds bias[r] (if has_bias), clamps to [clip.z, clip.w], applies
// SiLU (if silu) and adds the existing Y (if accumulate): a
// Gemma4ClippableLinear (+ activation / residual) in one pass. The tiling
// is eg_bf16_gemm's (shaders/embedding.metal): 32x32 tiles, 128 threads.
constant constexpr uint EA_TILE = 32;
constant constexpr uint EA_PAD = 4;

kernel void ea_bf16_gemm_clip(
    device const ushort *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    constant float4 &clip [[buffer(6)]], device const float *bias [[buffer(7)]],
    constant uint &flags [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[EA_TILE][EA_TILE + EA_PAD];
    threadgroup float ws[EA_TILE][EA_TILE + EA_PAD];
    threadgroup float cs[EA_TILE][EA_TILE + EA_PAD];
    const uint row0 = group.x * EA_TILE, token0 = group.y * EA_TILE;
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
    for (uint k0 = 0; k0 < columns; k0 += EA_TILE) {
        float4 a0 = token_live ? *reinterpret_cast<device const float4 *>(x_line + k0) : 0.0f;
        float4 a1 = token_live ? *reinterpret_cast<device const float4 *>(x_line + k0 + 4) : 0.0f;
        a0 = clamp(a0, clip.x, clip.y);
        a1 = clamp(a1, clip.x, clip.y);
        const ushort4 b0 = *reinterpret_cast<device const ushort4 *>(w_line + k0);
        const ushort4 b1 = *reinterpret_cast<device const ushort4 *>(w_line + k0 + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[line][k8 + 0] = a0.x; xs[line][k8 + 1] = a0.y;
        xs[line][k8 + 2] = a0.z; xs[line][k8 + 3] = a0.w;
        xs[line][k8 + 4] = a1.x; xs[line][k8 + 5] = a1.y;
        xs[line][k8 + 6] = a1.z; xs[line][k8 + 7] = a1.w;
        ws[k8 + 0][line] = ea_bf16(b0.x); ws[k8 + 1][line] = ea_bf16(b0.y);
        ws[k8 + 2][line] = ea_bf16(b0.z); ws[k8 + 3][line] = ea_bf16(b0.w);
        ws[k8 + 4][line] = ea_bf16(b1.x); ws[k8 + 5][line] = ea_bf16(b1.y);
        ws[k8 + 6][line] = ea_bf16(b1.z); ws[k8 + 7][line] = ea_bf16(b1.w);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < EA_TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], &xs[sm + 8 * i][kk], EA_TILE + EA_PAD);
                simdgroup_load(b[i], &ws[kk][sn + 8 * i], EA_TILE + EA_PAD);
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
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], EA_TILE + EA_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r8 = (tid % 4) * 8;
    if (!token_live) return;
    const bool has_bias = (flags & 1u) != 0, silu = (flags & 2u) != 0, accumulate = (flags & 4u) != 0;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8; ++r) {
        const uint out_row = row0 + r8 + r;
        if (out_row >= rows) continue;
        float y = cs[line][r8 + r];
        if (has_bias) y += bias[out_row];
        y = clamp(y, clip.z, clip.w);
        if (silu) y = ea_silu(y);
        device float *dst = output + ulong(token) * rows + out_row;
        *dst = accumulate ? *dst + y : y;
    }
}
