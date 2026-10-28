// Per-row int8 projections for the mixed-precision MTP head: `int8[r, c]` in
// [-127, 127] times one F32 scale per row, multiplying the same forward-rotated
// F32 activations the PTQ1_0 kernels read. Weights stay int8 in memory; each
// value converts to F32 in registers and accumulation is F32.
#include <metal_stdlib>
#include "bonsai_mixer.h"

using namespace metal;

// Base pointers and scales of the rows one SIMD group reduces. Rows past the
// end of a matrix are clamped to a real row rather than branched around, and
// the caller never writes their sums, as in the PTQ1_0 matvec.
template<uint R>
struct BonsaiInt8Rows {
    device const char *rows[R];
    float scales[R];
};

template<uint R>
static inline BonsaiInt8Rows<R> bonsai_int8_rows(
    device const char *weights, device const float *scales,
    uint first_row, uint rows, uint columns
) {
    BonsaiInt8Rows<R> result;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r) {
        const uint row = min(first_row + r, rows - 1);
        result.rows[r] = weights + ulong(row) * columns;
        result.scales[r] = scales[row];
    }
    return result;
}

// sums[t * R + r] = scale[r] * Σ_c w[r, c] · input[t, c] for T activation
// rows (stride `columns`) and R weight rows, returned in every lane.
//
// Each lane owns 16 consecutive columns per step, so a row's 32 lanes read one
// contiguous 512-byte span as 16-byte loads, and the activations each lane
// loads are reused across all R rows.
template<uint R, uint T>
static inline void bonsai_int8_dot(
    thread const BonsaiInt8Rows<R> &w, device const float *input, uint columns, uint lane,
    thread float *sums
) {
    #pragma clang loop unroll(full)
    for (uint i = 0; i < R * T; ++i) sums[i] = 0.0f;
    for (uint c = lane * 16; c < columns; c += 512) {
        float4 x[T][4];
        #pragma clang loop unroll(full)
        for (uint t = 0; t < T; ++t) {
            device const float4 *p =
                reinterpret_cast<device const float4 *>(input + ulong(t) * columns + c);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) x[t][q] = p[q];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const uint4 bits = *reinterpret_cast<device const uint4 *>(w.rows[r] + c);
            const float4 v[4] = {
                float4(as_type<char4>(bits.x)), float4(as_type<char4>(bits.y)),
                float4(as_type<char4>(bits.z)), float4(as_type<char4>(bits.w))};
            #pragma clang loop unroll(full)
            for (uint t = 0; t < T; ++t) {
                float s = sums[t * R + r];
                #pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) s += dot(v[q], x[t][q]);
                sums[t * R + r] = s;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            sums[t * R + r] = simd_sum(sums[t * R + r]) * w.scales[r];
        }
    }
}

// T activation rows by R weight rows per SIMD group, one SIMD group per
// threadgroup; dispatch ceil(rows / R) threadgroups of 32 threads. Output is
// `[T, rows]`.
template<uint R, uint T>
static inline void bonsai_int8_matmul_rows(
    device const char *weights, device const float *scales, device const float *input,
    device float *output, uint rows, uint columns, uint group, uint lane
) {
    const uint first_row = group * R;
    const BonsaiInt8Rows<R> w = bonsai_int8_rows<R>(weights, scales, first_row, rows, columns);
    float sums[R * T];
    bonsai_int8_dot<R, T>(w, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint t = 0; t < T; ++t) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            if (lane == r && first_row + r < rows) {
                output[ulong(t) * rows + first_row + r] = sums[t * R + r];
            }
        }
    }
}

#define BONSAI_INT8_MATMUL(NAME, R, T) \
kernel void NAME( \
    device const char *weights [[buffer(0)]], device const float *scales [[buffer(1)]], \
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]], \
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]] \
) { \
    bonsai_int8_matmul_rows<R, T>(weights, scales, input, output, rows, columns, group, lane); \
}

// On an M4 Pro both run at 236-248 GB/s on the head's 5120x5120 to
// 17408x5120 shapes (360 us for 17408x5120, against 134 us for the same shape
// in PTQ1_0). From three activation rows the F32 activation loads each SIMD
// group repeats outgrow the weight bytes (three rows: 78-86 GB/s, eight:
// 31-41 GB/s), so the wide kernel below takes over.
BONSAI_INT8_MATMUL(bonsai_int8_matvec, 4, 1)
BONSAI_INT8_MATMUL(bonsai_int8_matmul_2, 4, 2)

// 3..8 activation rows with F32 8x8 simdgroup matrices,
// C[token, row] += X[token, k] * W[k, row]. A threadgroup owns one 8-row W
// tile; its eight SIMD groups split the 128-column chunks and add their
// partials through threadgroup memory. Inside a chunk K is permuted so loads
// stay contiguous: step j's k-slot s is column 16s + j, so each lane reads
// sixteen consecutive int8 of two weight rows (16-byte loads) and 32
// consecutive activations of one token per chunk. int8 values are exact in
// F32 and the row scale is applied once to the sum. Token rows past `tokens`
// repeat row fm mod tokens and are not stored: row fm of X only reaches row
// fm of C. Dispatch ceil(rows / 8) threadgroups of 256 threads.
//
// On an M4 Pro, 3 to 8 rows cost what one row costs in the vector kernel,
// the weight bytes at about 250 GB/s: at 8 rows, 17408x5120 / 5120x17408
// 352 / 368 us, 5120x25600 537, 4096x5120 86, 1024x5120 25, 256x5120 8.7
// (one row: 356 / 354, 521, 88, 23, 8.1). The previous layout, four 8-row
// tiles sharing each X tile and four SIMD groups on 64-column chunks with
// 8-byte loads, took 414 / 441, 639, 113, 36 and 28.5 at 201-216 GB/s with
// the same multiplies and conversions, so its loads were the limit (16-byte
// loads and eight SIMD groups with four tiles kept gave about 400 / 390 at
// 17408x5120 / 5120x17408, one tile the rest). Feeding the
// multiplies exact half weights built without integer-to-float conversion
// (0x6400 | (q ^ 0x80) is 1152 + q as a half, minus 1152) measured within
// 1% of this conversion, as did two tiles or sixteen SIMD groups. At three
// rows the 5,120-row shapes cost 2-23% more than at four (cause unknown;
// still at or below the previous layout).
// Larger blocks take even chunks of at most eight rows (64 rows of
// 17408x5120: 2,827 us, from 3,324).
kernel void bonsai_int8_matmul_wide(
    device const char *weights [[buffer(0)]], device const float *scales [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint split = 8;
    threadgroup float shared[split * 64];
    // Lane ownership of simdgroup_float8x8 thread_elements(): row fm,
    // columns fn and fn + 1.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint first_row = group * 8;
    // Clamp to real rows and tokens: the values are discarded, nothing is read
    // out of range.
    device const float *x_row = input + ulong(fm % tokens) * columns + 16 * fn;
    device const char *w_rows[2];
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        w_rows[r] = weights + ulong(min(first_row + fn + r, rows - 1)) * columns + 16 * fm;
    }
    simdgroup_float8x8 sums = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint start = simd * 128; start < columns; start += split * 128) {
        float w[2][16];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const uint4 bits = *reinterpret_cast<device const uint4 *>(w_rows[r] + start);
            #pragma clang loop unroll(full)
            for (uint q = 0; q < 4; ++q) {
                const float4 v = float4(as_type<char4>(bits[q]));
                #pragma clang loop unroll(full)
                for (uint e = 0; e < 4; ++e) w[r][4 * q + e] = v[e];
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
            simdgroup_float8x8 b;
            b.thread_elements()[0] = w[0][j];
            b.thread_elements()[1] = w[1][j];
            simdgroup_multiply_accumulate(sums, x, b, sums);
        }
    }
    shared[simd * 64 + lane * 2] = sums.thread_elements()[0];
    shared[simd * 64 + lane * 2 + 1] = sums.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0 || fm >= tokens) return;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        float total = 0.0f;
        #pragma clang loop unroll(full)
        for (uint s = 0; s < split; ++s) total += shared[s * 64 + lane * 2 + r];
        const uint row = first_row + fn + r;
        if (row < rows) output[ulong(fm) * rows + row] = total * scales[row];
    }
}

// Up to three single-row projections of one input in one dispatch, as
// bonsai_ptq1_matvec_concat does: each SIMD group owns four rows of a single
// matrix and the segment only selects base pointers. Unused segments have
// zero rows.
kernel void bonsai_int8_matvec_concat(
    device const char *weights0 [[buffer(0)]], device const float *scales0 [[buffer(1)]],
    device float *output0 [[buffer(2)]],
    device const char *weights1 [[buffer(3)]], device const float *scales1 [[buffer(4)]],
    device float *output1 [[buffer(5)]],
    device const char *weights2 [[buffer(6)]], device const float *scales2 [[buffer(7)]],
    device float *output2 [[buffer(8)]],
    device const float *input [[buffer(9)]],
    constant uint *segment_rows [[buffer(10)]], constant uint &columns [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint groups0 = (segment_rows[0] + 3) / 4, groups1 = (segment_rows[1] + 3) / 4;
    device const char *weights = weights0;
    device const float *scales = scales0;
    device float *output = output0;
    uint rows = segment_rows[0];
    if (group >= groups0 + groups1) {
        group -= groups0 + groups1;
        weights = weights2;
        scales = scales2;
        output = output2;
        rows = segment_rows[2];
    } else if (group >= groups0) {
        group -= groups0;
        weights = weights1;
        scales = scales1;
        output = output1;
        rows = segment_rows[1];
    }
    bonsai_int8_matmul_rows<4, 1>(weights, scales, input, output, rows, columns, group, lane);
}

// Gate and up projections of one input with silu(gate) * up folded in, two
// rows of each per SIMD group sharing the activation loads, so neither
// projection is stored. Both matrices have the same shape.
kernel void bonsai_int8_matvec_swiglu(
    device const char *gate [[buffer(0)]], device const float *gate_scales [[buffer(1)]],
    device const char *up [[buffer(2)]], device const float *up_scales [[buffer(3)]],
    device const float *input [[buffer(4)]], device float *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]], constant uint &columns [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 2;
    const BonsaiInt8Rows<2> g = bonsai_int8_rows<2>(gate, gate_scales, first_row, rows, columns);
    const BonsaiInt8Rows<2> u = bonsai_int8_rows<2>(up, up_scales, first_row, rows, columns);
    const BonsaiInt8Rows<4> both = {
        {g.rows[0], g.rows[1], u.rows[0], u.rows[1]},
        {g.scales[0], g.scales[1], u.scales[0], u.scales[1]}};
    float sums[4];
    bonsai_int8_dot<4, 1>(both, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        if (lane == r && first_row + r < rows) {
            output[first_row + r] = bonsai_silu(sums[r]) * sums[2 + r];
        }
    }
}
