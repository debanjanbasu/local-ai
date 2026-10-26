// EmbeddingGemma 2 vision encoder kernels (see local-metal/src/embedding_vision.rs).
//
// The vision tower reuses the text encoder's BF16 GEMM, RMSNorm, residual,
// RoPE and gated-GELU kernels (embedding.metal); these are the pieces that
// differ: the learned 2-D position table, bidirectional attention with head
// dim 64 over whole images (simdgroup-matrix flash attention), and the 3x3
// spatial average pooling. Everything is F32 (activations, accumulation,
// softmax); BF16 table values widen to F32 exactly.
#include <metal_stdlib>
using namespace metal;

static inline float egv_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// hidden[t, c] += table[x_t, c] + table[10240 + y_t, c] (BF16 table
// [2 * 10240, width], x rows then y rows; the sum of the two embeddings is
// formed first, as the reference does). One thread per element.
kernel void egv_add_positions(
    device float *hidden [[buffer(0)]], device const ushort *table [[buffer(1)]],
    device const uint2 *positions [[buffer(2)]], constant uint &width [[buffer(3)]],
    constant uint &count [[buffer(4)]], constant uint &table_rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const uint t = gid / width, c = gid % width;
    const uint2 p = positions[t];
    const float e = egv_bf16(table[ulong(p.x) * width + c]) +
                    egv_bf16(table[ulong(table_rows + p.y) * width + c]);
    hidden[gid] = hidden[gid] + e;
}

// out[o, c] = (sum_j input[members[o * 9 + j], c] * (1/9)) * scale: the
// reference's 3x3 average pooling by patch position (a one-hot matmul with
// weight 1/9, in F32) followed by the sqrt(hidden) scaling. One thread per
// output element; members ascend in patch order.
kernel void egv_pool(
    device const float *input [[buffer(0)]], device const uint *members [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &width [[buffer(3)]],
    constant uint &count [[buffer(4)]], constant float &scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const uint o = gid / width, c = gid % width;
    const float w = 1.0f / 9.0f;
    float sum = 0.0f;
    for (uint j = 0; j < 9; ++j) sum += input[ulong(members[o * 9 + j]) * width + c] * w;
    output[gid] = sum * scale;
}

// ---------------------------------------------------------------- attention

// Bidirectional attention over one image (or frame): tokens [first, first +
// count) attend to each other only. q/k/v/out are [tokens, heads, 64] F32,
// scale 1.0 (q and k are RMS-normalized). A 128-thread threadgroup owns 32
// queries of one head; each SIMD group 8 of them. Keys stream through
// threadgroup memory 32 at a time; scores and the P·V product use 8x8
// simdgroup matrices, softmax is online in F32.
// Grid: (ceil(count / 32), heads).
constant constexpr uint EGV_BQ = 32;
constant constexpr uint EGV_BK = 32;
constant constexpr uint EGV_D = 64;
constant constexpr uint EGV_LD = EGV_D + 4;   // padded row stride (floats)
constant constexpr uint EGV_PLD = EGV_BK + 4;

kernel void egv_attention_64(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &first [[buffer(4)]], constant uint &count [[buffer(5)]],
    constant uint &heads [[buffer(6)]],
    threadgroup float *ks_memory [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    // ks is dynamic threadgroup memory (ATTENTION_KS_BYTES in
    // embedding_vision.rs): Metal's shader validation counts static
    // threadgroup memory twice against the 32 KiB limit, and every tile static
    // (22.6 KiB) aborted under MTL_SHADER_VALIDATION=1.
    threadgroup float (*ks)[EGV_LD] = (threadgroup float (*)[EGV_LD]) ks_memory;
    threadgroup float vs[EGV_BK][EGV_LD];
    threadgroup float ps[4][8][EGV_PLD];
    threadgroup float dg[4][8][8];
    threadgroup float lrow[4][8];

    const uint head = group.y;
    const uint q0 = group.x * EGV_BQ;
    const ulong row_stride = ulong(heads) * EGV_D;
    device const float *qb = q + ulong(first) * row_stride + head * EGV_D;
    device const float *kb = k + ulong(first) * row_stride + head * EGV_D;
    device const float *vb = v + ulong(first) * row_stride + head * EGV_D;

    // Thread -> (tile row, 16 consecutive dims) for staging 32 x 64 tiles.
    const uint line = tid / 4, d16 = (tid % 4) * 16;

    // Stage Q (zero rows past the end) through ks, then into registers.
    {
        const uint qi = q0 + line;
        for (uint e = 0; e < 16; e += 4) {
            const float4 x = qi < count
                ? *reinterpret_cast<device const float4 *>(qb + ulong(qi) * row_stride + d16 + e)
                : float4(0.0f);
            ks[line][d16 + e] = x.x; ks[line][d16 + e + 1] = x.y;
            ks[line][d16 + e + 2] = x.z; ks[line][d16 + e + 3] = x.w;
        }
    }
    for (uint i = lane; i < 64; i += 32) dg[sg][i / 8][i % 8] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 qm[8], om[8];
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(qm[d], &ks[sg * 8][d * 8], EGV_LD);
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    // Softmax state for row r = lane / 4 (4 lanes per row, 8 columns each).
    const uint r = lane / 4, c8 = (lane % 4) * 8;
    float m = -INFINITY, l = 0.0f;

    for (uint k0 = 0; k0 < count; k0 += EGV_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const uint kj = k0 + line;
            for (uint e = 0; e < 16; e += 4) {
                float4 a = float4(0.0f), b = float4(0.0f);
                if (kj < count) {
                    a = *reinterpret_cast<device const float4 *>(kb + ulong(kj) * row_stride + d16 + e);
                    b = *reinterpret_cast<device const float4 *>(vb + ulong(kj) * row_stride + d16 + e);
                }
                ks[line][d16 + e] = a.x; ks[line][d16 + e + 1] = a.y;
                ks[line][d16 + e + 2] = a.z; ks[line][d16 + e + 3] = a.w;
                vs[line][d16 + e] = b.x; vs[line][d16 + e + 1] = b.y;
                vs[line][d16 + e + 2] = b.z; vs[line][d16 + e + 3] = b.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T for this SIMD group's 8 queries x 32 keys.
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &ks[j * 8][d * 8], EGV_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qm[d], kt, s);
            }
            simdgroup_store(s, &ps[sg][0][j * 8], EGV_PLD);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax on row r, columns c8 .. c8 + 7.
        float sv[8];
        float block_max = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 8; ++e) {
            const bool live = k0 + c8 + e < count;
            sv[e] = live ? ps[sg][r][c8 + e] : -INFINITY;
            block_max = max(block_max, sv[e]);
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 2));
        const float m_new = max(m, block_max);
        const float correction = precise::exp(m - m_new);
        float row_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint e = 0; e < 8; ++e) {
            const float p = sv[e] == -INFINITY ? 0.0f : precise::exp(sv[e] - m_new);
            ps[sg][r][c8 + e] = p;
            row_sum += p;
        }
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 2);
        l = l * correction + row_sum;
        m = m_new;
        if (lane % 4 == 0) dg[sg][r][r] = correction;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(correction) O + P V.
        simdgroup_float8x8 diag;
        simdgroup_load(diag, &dg[sg][0][0], 8);
        #pragma clang loop unroll(full)
        for (uint d = 0; d < 8; ++d) {
            simdgroup_float8x8 scaled;
            simdgroup_multiply(scaled, diag, om[d]);
            om[d] = scaled;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) {
            simdgroup_float8x8 p;
            simdgroup_load(p, &ps[sg][0][j * 8], EGV_PLD);
            #pragma clang loop unroll(full)
            for (uint d = 0; d < 8; ++d) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, &vs[j * 8][d * 8], EGV_LD);
                simdgroup_multiply_accumulate(om[d], p, vt, om[d]);
            }
        }
    }

    // Normalize by the row sums and write the live query rows.
    if (lane % 4 == 0) lrow[sg][r] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint d = 0; d < 8; ++d) simdgroup_store(om[d], &ks[sg * 8][d * 8], EGV_LD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint qi = q0 + line;
    if (qi < count) {
        const float inv = 1.0f / lrow[line / 8][line % 8];
        device float *o = out + (ulong(first) + qi) * row_stride + head * EGV_D + d16;
        for (uint e = 0; e < 16; ++e) o[e] = ks[line][d16 + e] * inv;
    }
}
