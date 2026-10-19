// Bonsai 2 27B vision tower kernels: the Qwen3.5 ViT of the checkpoint's
// mmproj (see local-metal/src/bonsai_vision.rs).
//
// Everything is F32 (activations, accumulation, softmax); BF16 weights widen
// to F32 exactly. Patches are rows of 1152; the fused qkv projection leaves
// each row as [3][16 heads][72].
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BV_TILE = 32;
constant constexpr uint BV_PAD = 4;
constant constexpr uint BV_WIDTH = 1152;
constant constexpr uint BV_HEADS = 16;
constant constexpr uint BV_D = 72;
constant constexpr uint BV_HALF = 36;

static inline float bv_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// torch.nn.functional.gelu(approximate="tanh").
static inline float bv_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2 / pi)
    return 0.5f * x * (1.0f + precise::tanh(k * (x + 0.044715f * x * x * x)));
}

// erfc(x) with fractional error below 1.2e-7 everywhere (Numerical Recipes'
// Chebyshev fit, as whisper.metal), for torch's exact nn.GELU.
static inline float bv_erfc(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float r = t * precise::exp(-z * z - 1.26551223f +
        t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
        t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f +
        t * 0.17087277f)))))))));
    return x >= 0.0f ? r : 2.0f - r;
}

static inline float bv_gelu_erf(float x) {
    return 0.5f * x * bv_erfc(-x * 0.70710678118654752f);
}

// ---------------------------------------------------------------- GEMM

// value[t, r] = sum_c W[r, c] * input[t, c] + bias[r] for `tokens` input rows
// of `columns` (a multiple of 8) and BF16 W [rows][columns]; then by `mode`:
//   0: output[t, r] = value
//   1: output[t, r] = gelu_tanh(value)
//   2: output[t, r] = gelu_erf(value)
//   3: output[t, r] += value        (residual add; output is not an input)
// A 128-thread threadgroup computes a 32-token x 32-row tile on 8x8
// simdgroup matrices; X and W stage through threadgroup memory, columns past
// the end read as zero. Grid: (ceil(rows / 32), ceil(tokens / 32)).
// BF16 weights widen exactly; F32 weights (the patch embedding) load as is.
static inline float4 bv_weights4(device const ushort *w) {
    const ushort4 b = *reinterpret_cast<device const ushort4 *>(w);
    return float4(bv_bf16(b.x), bv_bf16(b.y), bv_bf16(b.z), bv_bf16(b.w));
}

static inline float4 bv_weights4(device const float *w) {
    return *reinterpret_cast<device const float4 *>(w);
}

template <typename W>
static inline void bv_gemm_tile(
    device const W *weights, device const float *bias, device const float *input,
    device float *output, uint rows, uint columns, uint tokens, uint mode,
    uint2 group, uint tid, uint simd,
    threadgroup float (*xs)[BV_TILE + BV_PAD], threadgroup float (*ws)[BV_TILE + BV_PAD],
    threadgroup float (*cs)[BV_TILE + BV_PAD]
) {
    const uint row0 = group.x * BV_TILE, token0 = group.y * BV_TILE;
    // Staging assignment: thread -> (tile line, 8 consecutive k).
    const uint line = tid / 4, k8 = (tid % 4) * 8;
    const uint token = token0 + line, row = row0 + line;
    const bool token_live = token < tokens;
    device const float *x_line = input + ulong(min(token, tokens - 1)) * columns + k8;
    device const W *w_line = weights + ulong(min(row, rows - 1)) * columns + k8;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint k0 = 0; k0 < columns; k0 += BV_TILE) {
        const bool k_live = k0 + k8 < columns;
        const bool x_live = token_live && k_live;
        const float4 a0 = x_live ? *reinterpret_cast<device const float4 *>(x_line + k0) : 0.0f;
        const float4 a1 = x_live ? *reinterpret_cast<device const float4 *>(x_line + k0 + 4) : 0.0f;
        const float4 b0 = k_live ? bv_weights4(w_line + k0) : float4(0.0f);
        const float4 b1 = k_live ? bv_weights4(w_line + k0 + 4) : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xs[line][k8 + 0] = a0.x; xs[line][k8 + 1] = a0.y;
        xs[line][k8 + 2] = a0.z; xs[line][k8 + 3] = a0.w;
        xs[line][k8 + 4] = a1.x; xs[line][k8 + 5] = a1.y;
        xs[line][k8 + 6] = a1.z; xs[line][k8 + 7] = a1.w;
        ws[k8 + 0][line] = b0.x; ws[k8 + 1][line] = b0.y;
        ws[k8 + 2][line] = b0.z; ws[k8 + 3][line] = b0.w;
        ws[k8 + 4][line] = b1.x; ws[k8 + 5][line] = b1.y;
        ws[k8 + 6][line] = b1.z; ws[k8 + 7][line] = b1.w;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BV_TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) {
                simdgroup_load(a[i], &xs[sm + 8 * i][kk], BV_TILE + BV_PAD);
                simdgroup_load(b[i], &ws[kk][sn + 8 * i], BV_TILE + BV_PAD);
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
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], BV_TILE + BV_PAD);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread -> (token line, 8 consecutive rows).
    const uint r8 = (tid % 4) * 8;
    if (!token_live) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 8; ++r) {
        const uint out_row = row0 + r8 + r;
        if (out_row >= rows) continue;
        const float value = cs[line][r8 + r] + bias[out_row];
        device float *o = output + ulong(token) * rows + out_row;
        switch (mode) {
            case 1: *o = bv_gelu_tanh(value); break;
            case 2: *o = bv_gelu_erf(value); break;
            case 3: *o = *o + value; break;
            default: *o = value; break;
        }
    }
}

kernel void bv_gemm(
    device const ushort *weights [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &mode [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float ws[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float cs[BV_TILE][BV_TILE + BV_PAD];
    bv_gemm_tile(weights, bias, input, output, rows, columns, tokens, mode, group, tid, simd,
        xs, ws, cs);
}

// bv_gemm with F32 weights.
kernel void bv_gemm_f32(
    device const float *weights [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &mode [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float xs[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float ws[BV_TILE][BV_TILE + BV_PAD];
    threadgroup float cs[BV_TILE][BV_TILE + BV_PAD];
    bv_gemm_tile(weights, bias, input, output, rows, columns, tokens, mode, group, tid, simd,
        xs, ws, cs);
}

// ---------------------------------------------------------------- LayerNorm

static inline float bv_sum_128(float value, threadgroup float *partial, uint tid) {
    value = simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// torch.nn.LayerNorm over `dimension` columns: the mean, then the biased
// variance of the centred values, (x - mean) / sqrt(var + eps) * w + b. One
// 128-thread threadgroup per row; output may not alias input.
kernel void bv_layernorm(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &dimension [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    device const float *x = input + ulong(row) * dimension;
    device float *y = output + ulong(row) * dimension;
    float sum = 0.0f;
    for (uint c = tid; c < dimension; c += 128) sum += x[c];
    const float mean = bv_sum_128(sum, partial, tid) / float(dimension);
    float squares = 0.0f;
    for (uint c = tid; c < dimension; c += 128) {
        const float d = x[c] - mean;
        squares += d * d;
    }
    const float variance = bv_sum_128(squares, partial, tid) / float(dimension);
    const float inverse = 1.0f / precise::sqrt(variance + epsilon);
    for (uint c = tid; c < dimension; c += 128) y[c] = (x[c] - mean) * inverse * weight[c] + bias[c];
}

// ---------------------------------------------------------------- RoPE

// The 2-D vision RoPE, in place on the q and k thirds of qkv rows
// [tokens][3][16][72]: dims j and j + 36 of every head rotate by the angle
// whose cosine and sine are row t, column j of the [tokens][36] tables
// (q * cos + rotate_half(q) * sin). One thread per (token, q|k, head, j).
kernel void bv_rope(
    device float *qkv [[buffer(0)]], device const float *cosines [[buffer(1)]],
    device const float *sines [[buffer(2)]], constant uint &tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint j = gid % BV_HALF;
    uint rest = gid / BV_HALF;
    const uint head = rest % BV_HEADS;
    rest /= BV_HEADS;
    const uint part = rest % 2, t = rest / 2;
    if (t >= tokens) return;
    device float *v = qkv + ulong(t) * (3 * BV_WIDTH) + part * BV_WIDTH + head * BV_D;
    const float c = cosines[ulong(t) * BV_HALF + j], s = sines[ulong(t) * BV_HALF + j];
    const float x1 = v[j], x2 = v[j + BV_HALF];
    v[j] = x1 * c - x2 * s;
    v[j + BV_HALF] = x2 * c + x1 * s;
}

// ---------------------------------------------------------------- attention

// Bidirectional attention over one image's `count` patches: q, k and v are
// the thirds of qkv rows [count][3][16][72], out is [count][16][72], softmax
// of scale * q.k in F32. A 256-thread threadgroup owns 64 queries of one
// head, each SIMD group 8 of them; keys stream through threadgroup memory 32
// at a time, scores and P.V use 8x8 simdgroup matrices and the online softmax
// runs on the score matrices' own lane elements (in base 2, with
// scale * log2(e) folded into Q). A dispatch covers the queries from
// `first_query` (a multiple of 64), so a large image can split its attention
// over several command buffers.
// Grid: (ceil(queries / 64), 16).
constant constexpr uint BV_BQ = 64;
constant constexpr uint BV_BK = 32;
constant constexpr uint BV_LD = BV_D + 4;  // padded row stride (floats)
constant constexpr uint BV_KLD = BV_BK + 4;  // transposed key tile stride
// Rows of BV_LD that also hold a [72][BV_KLD] transposed key tile.
constant constexpr uint BV_KS_ROWS = (BV_D * BV_KLD + BV_LD - 1) / BV_LD;

kernel void bv_attention(
    device const float *qkv [[buffer(0)]], device float *out [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant float &scale [[buffer(3)]],
    constant uint &first_query [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float ks[BV_KS_ROWS][BV_LD];  // Q staging, then K^T [72][BV_KLD], then O
    threadgroup float vs[BV_BK][BV_LD];
    threadgroup float (*kt_tile)[BV_KLD] = (threadgroup float (*)[BV_KLD]) &ks[0][0];

    const uint head = group.y;
    const uint q0 = first_query + group.x * BV_BQ;
    const ulong stride = 3 * BV_WIDTH;
    device const float *qb = qkv + head * BV_D;
    device const float *kb = qkv + BV_WIDTH + head * BV_D;
    device const float *vb = qkv + 2 * BV_WIDTH + head * BV_D;

    // Stage scaled Q (zero rows past the end) through ks and vs, then into
    // registers: 64 rows of 18 float4s.
    {
        const float qscale = scale * M_LOG2E_F;
        for (uint i = tid; i < BV_BQ * 18; i += 256) {
            const uint row = i / 18, c = (i % 18) * 4, qi = q0 + row;
            const float4 q = qi < count
                ? *(device const float4 *)(qb + ulong(qi) * stride + c) * qscale : float4(0.0f);
            threadgroup float *t = row < BV_BK ? &ks[row][c] : &vs[row - BV_BK][c];
            *(threadgroup float4 *)t = q;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float (*qs)[BV_LD] = sg < 4 ? &ks[sg * 8] : &vs[(sg - 4) * 8];
    simdgroup_float8x8 qm[9], om[9];
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 9; ++d) {
        simdgroup_load(qm[d], &qs[0][d * 8], BV_LD);
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Lane ownership of a simdgroup_float8x8: row fm, columns fn and fn + 1;
    // the 4 lanes of a row differ in lane bits 0 and 3.
    const uint quad = lane / 4;
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    float m = -1.0e30f, l = 0.0f;  // finite: shaders build with fast math

    for (uint k0 = 0; k0 < count; k0 += BV_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BV_BK * 18; i += 256) {
            const uint row = i / 18, c = (i % 18) * 4, kj = k0 + row;
            const bool live = kj < count;
            const ulong at = ulong(kj) * stride + c;
            const float4 k = live ? *(device const float4 *)(kb + at) : float4(0.0f);
            kt_tile[c][row] = k.x;
            kt_tile[c + 1][row] = k.y;
            kt_tile[c + 2][row] = k.z;
            kt_tile[c + 3][row] = k.w;
            *(threadgroup float4 *)&vs[row][c] = live ? *(device const float4 *)(vb + at) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T for this SIMD group's 8 queries x 32 keys.
        simdgroup_float8x8 s[4];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            s[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 9; ++d) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &kt_tile[d * 8][j * 8], BV_KLD);
                simdgroup_multiply_accumulate(s[j], qm[d], kt, s[j]);
            }
        }

        // Online softmax over this lane's row.
        const bool tail = k0 + BV_BK > count;
        float block_max = -1.0e30f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                if (tail && k0 + j * 8 + fn + e >= count) s[j].thread_elements()[e] = -1.0e30f;
                block_max = max(block_max, s[j].thread_elements()[e]);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float m_new = max(m, block_max);
        const float correction = exp2(m - m_new);
        float row_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 2; ++e) {
                const float p = exp2(s[j].thread_elements()[e] - m_new);
                s[j].thread_elements()[e] = p;
                row_sum += p;
            }
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 8);
        l = l * correction + row_sum;
        m = m_new;

        // O = diag(correction) O + P V.
        #pragma clang loop unroll(full)
        for (uint d = 0; d < 9; ++d) {
            om[d].thread_elements()[0] *= correction;
            om[d].thread_elements()[1] *= correction;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 9; ++d) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, &vs[j * 8][d * 8], BV_LD);
                simdgroup_multiply_accumulate(om[d], s[j], vt, om[d]);
            }
        }
    }

    // Normalize by the row sums and write the live query rows.
    const float inv = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 9; ++d) {
        om[d].thread_elements()[0] *= inv;
        om[d].thread_elements()[1] *= inv;
        simdgroup_store(om[d], &qs[0][d * 8], BV_LD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < BV_BQ * 18; i += 256) {
        const uint row = i / 18, c = (i % 18) * 4, qi = q0 + row;
        if (qi < count) {
            threadgroup float *t = row < BV_BK ? &ks[row][c] : &vs[row - BV_BK][c];
            *(device float4 *)(out + ulong(qi) * BV_WIDTH + head * BV_D + c) = *(threadgroup float4 *)t;
        }
    }
}
