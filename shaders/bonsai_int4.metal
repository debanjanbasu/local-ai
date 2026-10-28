// Group-wise 4-bit projections for the DFlash2 drafter: row-major codes, two
// per byte (column 2i in the low nibble of byte i), and one half2 (scale,
// bias) per group of 64 consecutive columns of a row, so a weight is
// `scale * code + bias` with code in [0, 15]. They multiply the same
// F32 activations the int8 kernels read; weights stay packed in memory and
// accumulation is F32.
#include <metal_stdlib>

using namespace metal;

// sums[t * R + r] = Σ_c w[r, c] · input[t, c] for T activation rows (stride
// `columns`) and R weight rows, returned in every lane.
//
// Each lane owns 32 consecutive columns per step, one 16-byte load per row,
// which lie inside one group, so the group's
// scale multiplies Σ code · x and its bias Σ x, both shared across the 32
// columns. Rows past the end of a matrix are clamped to a real row and the
// caller never writes their sums.
template<uint R, uint T>
static inline void bonsai_int4_matmul_rows(
    device const uchar *weights, device const half2 *groups, device const float *input,
    device float *output, uint rows, uint columns, uint group, uint lane
) {
    const uint first_row = group * R;
    const uint row_groups = columns / 64;
    device const uchar *w_rows[R];
    device const half2 *g_rows[R];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r) {
        const uint row = min(first_row + r, rows - 1);
        w_rows[r] = weights + ulong(row) * (columns / 2);
        g_rows[r] = groups + ulong(row) * row_groups;
    }
    float sums[R * T];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) sums[i] = 0.0f;
    for (uint c = lane * 32; c < columns; c += 1024) {
        // Activations of the even and odd columns of each eight-column word,
        // the order the nibble masks produce, and their sum for the bias.
        float4 even[T][4], odd[T][4];
        float total[T];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            device const float4 *p =
                reinterpret_cast<device const float4 *>(input + ulong(t) * columns + c);
            float4 s = 0.0f;
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                const float4 a = p[2 * q], b = p[2 * q + 1];
                even[t][q] = float4(a.xz, b.xz);
                odd[t][q] = float4(a.yw, b.yw);
                s += a + b;
            }
            total[t] = s.x + s.y + s.z + s.w;
        }
        const uint index = c / 64;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const uint4 bits = *reinterpret_cast<device const uint4 *>(w_rows[r] + c / 2);
            const float2 sb = float2(g_rows[r][index]);
            float4 lo[4], hi[4];
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                lo[q] = float4(as_type<uchar4>(bits[q] & 0x0F0F0F0Fu));
                hi[q] = float4(as_type<uchar4>((bits[q] >> 4) & 0x0F0F0F0Fu));
            }
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) {
                float d = 0.0f;
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) d += dot(lo[q], even[t][q]) + dot(hi[q], odd[t][q]);
                sums[t * R + r] = fma(sb.x, d, fma(sb.y, total[t], sums[t * R + r]));
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const float sum = simd_sum(sums[t * R + r]);
            if (lane == r && first_row + r < rows) output[ulong(t) * rows + first_row + r] = sum;
        }
    }
}

#define BONSAI_INT4_MATMUL(NAME, R, T) \
kernel void NAME( \
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]], \
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]] \
) { \
    bonsai_int4_matmul_rows<R, T>(weights, groups, input, output, rows, columns, group, lane); \
}

// One activation row, four weight rows per SIMD group, one SIMD group per
// threadgroup; dispatch ceil(rows / 4) threadgroups of 32 threads. On an M4
// Pro: 17408x5120 237 us (int8 359), 5120x17408 284 (355), 5120x25600 432
// (524), 4096x5120 64 (99), 1024x5120 18 (26), 256x5120 8.4 (8.5).
BONSAI_INT4_MATMUL(bonsai_int4_matvec, 4, 1)

// 3..8 activation rows with F32 8x8 simdgroup matrices, laid out as
// bonsai_int8_matmul_wide: C[token, row] += X[token, k] * W[k, row], eight SIMD
// groups splitting the 128-column chunks and adding their partials through
// threadgroup memory, step j's k-slot s being column 16s + j. A threadgroup
// owns TILES 8-row W tiles that share each X fragment. A lane's sixteen
// columns of a weight row are one 8-byte load inside one group; each weight
// becomes scale * code + bias in registers. Token rows past `tokens` repeat
// row fm mod tokens and are not stored. Dispatch ceil(rows / (8 * TILES))
// threadgroups of 256 threads.
//
// On an M4 Pro at eight rows, two tiles: 17408x5120 / 5120x17408 264 / 269 us
// (int8 352 / 368), 5120x25600 395 (537), 4096x5120 67 (86); one tile for 1024
// rows 22 (25) and 256 rows 8.6 (8.7). The multiplies, not the 4-bit bytes,
// are the limit: with the activation loads and the dequantizing FMA taken out
// one tile still takes 244 us at 17408x5120. Four tiles spill (306 us); half
// operands with conversions are slower (275 against 244 for the same
// stripped kernel); multiplying the codes themselves in per-group chains,
// scaled and biased once per group, with 4-byte loads per row and group,
// measured 267 / 267 / 393 / 70 / 26.5 us, no faster.
template<uint TILES>
static inline void bonsai_int4_wide(
    device const uchar *weights, device const half2 *groups, device const float *input,
    device float *output, uint rows, uint columns, uint tokens, uint group, uint simd,
    uint lane, threadgroup float *shared
) {
    constexpr uint split = 8;
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = group * 8 * TILES;
    const uint row_groups = columns / 64;
    device const float *x_row = input + ulong(fm % tokens) * columns + 16 * fn;
    device const uchar *w_rows[2 * TILES];
    device const half2 *g_rows[2 * TILES];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2 * TILES; ++r) {
        const uint row = min(first_row + 8 * (r / 2) + fn + r % 2, rows - 1);
        w_rows[r] = weights + ulong(row) * (columns / 2) + 8 * fm;
        g_rows[r] = groups + ulong(row) * row_groups;
    }
    simdgroup_float8x8 sums[TILES];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TILES; ++i) sums[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint start = simd * 128; start < columns; start += split * 128) {
        float w[2 * TILES][16];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2 * TILES; ++r) {
            const uint2 bits = *reinterpret_cast<device const uint2 *>(w_rows[r] + start / 2);
            const float2 sb = float2(g_rows[r][(start + 16 * fm) / 64]);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 2; ++q) {
                const float4 lo = float4(as_type<uchar4>(bits[q] & 0x0F0F0F0Fu));
                const float4 hi = float4(as_type<uchar4>((bits[q] >> 4) & 0x0F0F0F0Fu));
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 4; ++e) {
                    w[r][8 * q + 2 * e] = fma(lo[e], sb.x, sb.y);
                    w[r][8 * q + 2 * e + 1] = fma(hi[e], sb.x, sb.y);
                }
            }
        }
        float xa[16], xb[16];
        #pragma clang loop unroll(full)
        for (uint q = 0; q < 4; ++q) {
            const float4 a = *reinterpret_cast<device const float4 *>(x_row + start + 4 * q);
            const float4 b = *reinterpret_cast<device const float4 *>(x_row + start + 16 + 4 * q);
            #pragma clang loop unroll(full)
            for (uint e = 0; e < 4; ++e) {
                xa[4 * q + e] = a[e];
                xb[4 * q + e] = b[e];
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 16; ++j) {
            simdgroup_float8x8 x;
            x.thread_elements()[0] = xa[j];
            x.thread_elements()[1] = xb[j];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TILES; ++i) {
                simdgroup_float8x8 b;
                b.thread_elements()[0] = w[2 * i][j];
                b.thread_elements()[1] = w[2 * i + 1][j];
                simdgroup_multiply_accumulate(sums[i], x, b, sums[i]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < TILES; ++i) {
        shared[(i * split + simd) * 64 + lane * 2] = sums[i].thread_elements()[0];
        shared[(i * split + simd) * 64 + lane * 2 + 1] = sums[i].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd >= TILES || fm >= tokens) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < split; ++s) total += shared[(simd * split + s) * 64 + lane * 2 + r];
        const uint row = first_row + 8 * simd + fn + r;
        if (row < rows) output[ulong(fm) * rows + row] = total;
    }
}

#define BONSAI_INT4_WIDE(NAME, TILES) \
kernel void NAME( \
    device const uchar *weights [[buffer(0)]], device const half2 *groups [[buffer(1)]], \
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]], \
    constant uint &tokens [[buffer(6)]], \
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    threadgroup float shared[TILES * 8 * 64]; \
    bonsai_int4_wide<TILES>(weights, groups, input, output, rows, columns, tokens, group, simd, \
                            lane, shared); \
}

BONSAI_INT4_WIDE(bonsai_int4_matmul_wide, 1)
BONSAI_INT4_WIDE(bonsai_int4_matmul_wide_2, 2)
