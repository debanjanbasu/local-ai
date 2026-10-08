// Small-batch PTQ1_0 projection for speculative verification blocks and short
// prefill chunks. The 128-token tensor tile costs a full tile for any block of
// 2..128 rows; here one SIMD group covers `rows_per_group` output rows and
// streams each packed block exactly once, like bonsai_ptq1_matvec_reuse. The
// block's actual -1/0/+1 trits are derived once and reused for every activation
// row. Dot them directly against inputs: telescoping cumulative base-3 factors
// against transformed inputs amplifies cancellation even on modest F32 values.
// Accumulation and outputs are F32; nothing is unpacked outside registers.
// Two kernels share this: a scalar one for 2..4 rows and a simdgroup-matrix
// (`wide`) one for up to eight rows, whose cost does not depend on the row count.
//
// `Packed` is BonsaiNativePacked or BonsaiTensorPacked. `Input`/`Output` are
// indexable rank-one handles (device pointers or tensor handles) holding
// `[tokens_total, columns]` activations and `[tokens_total, rows]` results;
// input elements may be float or half and are widened before use.
#ifndef BONSAI_SMALL_BATCH_H
#define BONSAI_SMALL_BATCH_H

#include "bonsai_projection.h"

// Bytes 2*lane, 2*lane+1 and 16+lane contribute five trits each, then the
// qh byte contributes one. The trits are exact in half (half-prefix decode,
// bonsai_projection.h) before any floating-point activation arithmetic.
template<typename Packed>
static inline void bonsai_ptq1_factors(
    Packed packed, metal::ulong block, metal::uint lane, thread half (&factors)[16]
) {
    #pragma clang loop unroll(full)
    for (metal::uint k = 0; k < 3; ++k) {
        const metal::uint byte = k < 2 ? 2 * lane + k : 16 + lane;
        bonsai_ptq1_trits(bonsai_ptq1_code(packed.code(block, byte)), &factors[5 * k]);
    }
    const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[lane >> 1];
    const float high = float(packed.code(block, 24 + (lane & 1))) * (1.0f / 256.0f) * power;
    factors[15] = half(metal::floor(3.0f * high) - 3.0f * metal::floor(high) - 1.0f);
}

// Gather one activation row's sixteen elements in the decoded trit order.
// Widen half inputs, but never narrow F32 operands or subtract an input sum.
template<typename Input>
static inline void bonsai_ptq1_values(
    Input input,
    metal::ulong base,
    metal::uint block,
    metal::uint part,
    thread float (&values)[16]
) {
    #pragma clang loop unroll(full)
    for (metal::uint k = 0; k < 3; ++k) {
        const metal::uint offset = k < 2 ? 2 * part + k : 80 + part;
        const metal::uint stride = k < 2 ? 16 : 8;
        #pragma clang loop unroll(full)
        for (metal::uint n = 0; n < 5; ++n) {
            values[5 * k + n] = float(input[base + block * 128 + offset + n * stride]);
        }
    }
    values[15] = float(input[base + block * 128 + 120 + part]);
}

// One threadgroup is one SIMD group of 32 lanes: lane / 8 selects the packed
// block in flight, lane & 7 its sixteen-element part. Dispatch
// ceil(rows / rows_per_group) groups; `first_token` selects the activation
// rows [first_token, first_token + tokens) of the flattened handles.
template<
    metal::uint tokens, metal::uint rows_per_group,
    typename Packed, typename Input, typename Output>
static inline void bonsai_ptq1_small_batch_impl(
    Packed packed,
    Input input,
    Output output,
    metal::uint rows,
    metal::uint columns,
    metal::uint first_token,
    metal::uint group,
    metal::uint lane
) {
    const metal::uint first_row = group * rows_per_group;
    const metal::uint blocks = columns / 128;
    const metal::uint part = lane & 7;
    const metal::ulong input_base = metal::ulong(first_token) * columns;
    const metal::ulong output_base = metal::ulong(first_token) * rows;
    float sums[rows_per_group][tokens];
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) {
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tokens; ++t) sums[row][t] = 0.0f;
    }
    // Rows' factors are derived once per block and held while tokens stream
    // through one input row at a time, so register use is
    // 16 * rows_per_group + 16 + rows_per_group * tokens rather than growing
    // by sixteen per token (token-major arrays measured 0.44 ms at
    // four tokens but 2.44 ms at eight on the 17408x5120 projection).
    for (metal::uint block = lane / 8; block < blocks; block += 4) {
        // The trits are exact in half; holding them as half halves their
        // registers, and every use widens them back into an F32 FMA, so the
        // outputs are bit-identical to F32 factors. Measured on M4 Pro
        // (17408x5120 / 5120x17408), 2/3/4 tokens went from 200/256/324 and
        // 197/259/344 us to 191/235/281 and 194/237/286 us, and with the
        // half-prefix decoder (same trits) to 152/192/231 and 155/194/238 us.
        half factors[rows_per_group][16];
        float scales[rows_per_group];
        #pragma clang loop unroll(full)
        for (metal::uint row = 0; row < rows_per_group; ++row) {
            // Clamp to a real row: the value is discarded below, nothing is
            // read out of range.
            const metal::uint source = metal::min(first_row + row, rows - 1);
            const metal::ulong packed_block = metal::ulong(source) * blocks + block;
            bonsai_ptq1_factors(packed, packed_block, part, factors[row]);
            scales[row] = packed.scale(packed_block);
        }
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tokens; ++t) {
            float values[16];
            bonsai_ptq1_values(
                input, input_base + metal::ulong(t) * columns, block, part, values);
            #pragma clang loop unroll(full)
            for (metal::uint row = 0; row < rows_per_group; ++row) {
                float total = 0.0f;
                #pragma clang loop unroll(full)
                for (metal::uint i = 0; i < 16; ++i) total += float(factors[row][i]) * values[i];
                sums[row][t] += total * scales[row];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) {
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tokens; ++t) {
            const float total = metal::simd_sum(sums[row][t]);
            if (lane == 0 && first_row + row < rows) {
                output[output_base + metal::ulong(t) * rows + first_row + row] = total;
            }
        }
    }
}

// Wide variant for 5..8 activation rows, one dispatch instead of two that
// each re-stream the matrix. A threadgroup covers eight output rows with four
// SIMD groups splitting the packed blocks; each SIMD group multiplies 8x8
// simdgroup matrices C[token, row] += X[token, k] * W[k, row]. The W tile holds
// exact trits (each lane decodes two, rows `fn` and `fn + 1` at element `fm`),
// so every product is exact before the F32 MMA accumulation; a block's partial
// sums are scaled once, as in the matvec, then the four SIMD groups' partials
// are added through threadgroup memory. Token rows past `tokens` contribute
// zeros and are not stored. Dispatch ceil(rows / 8) groups of 128 threads.
// The summation order differs from the scalar kernel, so outputs are not bit
// identical: on random FFN-shaped weights the largest gap measured 1.5e-7 to
// 2.2e-7 of the largest output magnitude. Half-prefix trit decoding took the
// FFN shapes from 341-350 to 280-301 us with bit-identical outputs; the F32
// 8x8 multiplies remain the limit.
//
// Inside a block K is permuted: tile j holds elements 8j..8j+7, so the X tile
// is eight contiguous activations per token and the W tile's element 8j + fm
// comes from byte fm + 8 (j & 1) at power j >> 1 (j < 10), byte 16 + fm at
// power j - 10 (j < 15), or qh byte 24 + (fm & 1) at power fm >> 1 (j = 15).
template<typename Packed, typename Input, typename Output>
static inline void bonsai_ptq1_small_batch_wide_impl(
    Packed packed,
    Input input,
    Output output,
    metal::uint rows,
    metal::uint columns,
    metal::uint first_token,
    metal::uint tokens,
    metal::uint group,
    metal::uint simd,
    metal::uint lane,
    threadgroup float *shared
) {
    constexpr metal::uint split = 4;
    // Lane ownership of simdgroup_float8x8 thread_elements(): row fm,
    // columns fn and fn + 1.
    const metal::uint quad = lane / 4;
    const metal::uint fm = (quad & 4) + ((lane / 2) % 4);
    const metal::uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const metal::uint first_row = group * 8;
    const metal::uint blocks = columns / 128;
    // Clamp to real rows and tokens: the values are discarded, nothing is read
    // out of range.
    const metal::ulong row_base[2] = {
        metal::ulong(metal::min(first_row + fn, rows - 1)) * blocks,
        metal::ulong(metal::min(first_row + fn + 1, rows - 1)) * blocks};
    const bool live = fm < tokens;
    const metal::ulong input_row =
        metal::ulong(first_token + metal::min(fm, tokens - 1)) * columns + fn;
    const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[fm >> 1];
    float sums[2] = {0.0f, 0.0f};
    for (metal::uint block = simd; block < blocks; block += split) {
        half trits[2][16];
        float scales[2];
        #pragma clang loop unroll(full)
        for (metal::uint r = 0; r < 2; ++r) {
            const metal::ulong packed_block = row_base[r] + block;
            scales[r] = packed.scale(packed_block);
            #pragma clang loop unroll(full)
            for (metal::uint k = 0; k < 3; ++k) {
                const metal::uint byte = k == 0 ? fm : (k == 1 ? fm + 8 : 16 + fm);
                half f[5];
                bonsai_ptq1_trits(bonsai_ptq1_code(packed.code(packed_block, byte)), f);
                #pragma clang loop unroll(full)
                for (metal::uint n = 0; n < 5; ++n) trits[r][k < 2 ? 2 * n + k : 10 + n] = f[n];
            }
            const float high =
                float(packed.code(packed_block, 24 + (fm & 1))) * (1.0f / 256.0f) * power;
            trits[r][15] = half(metal::floor(3.0f * high) - 3.0f * metal::floor(high) - 1.0f);
        }
        const metal::ulong base = input_row + metal::ulong(block) * 128;
        metal::simdgroup_float8x8 partial;
        #pragma clang loop unroll(full)
        for (metal::uint tile = 0; tile < 16; ++tile) {
            metal::simdgroup_float8x8 x;
            metal::simdgroup_float8x8 w;
            const float x0 = float(input[base + tile * 8]);
            const float x1 = float(input[base + tile * 8 + 1]);
            x.thread_elements()[0] = live ? x0 : 0.0f;
            x.thread_elements()[1] = live ? x1 : 0.0f;
            w.thread_elements()[0] = float(trits[0][tile]);
            w.thread_elements()[1] = float(trits[1][tile]);
            if (tile == 0) {
                metal::simdgroup_multiply(partial, x, w);
            } else {
                metal::simdgroup_multiply_accumulate(partial, x, w, partial);
            }
        }
        sums[0] = metal::fma(partial.thread_elements()[0], scales[0], sums[0]);
        sums[1] = metal::fma(partial.thread_elements()[1], scales[1], sums[1]);
    }
    shared[simd * 64 + lane * 2] = sums[0];
    shared[simd * 64 + lane * 2 + 1] = sums[1];
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simd != 0 || !live) return;
    #pragma clang loop unroll(full)
    for (metal::uint s = 1; s < split; ++s) {
        sums[0] += shared[s * 64 + lane * 2];
        sums[1] += shared[s * 64 + lane * 2 + 1];
    }
    const metal::ulong out = metal::ulong(first_token + fm) * rows + first_row + fn;
    if (first_row + fn < rows) output[out] = sums[0];
    if (first_row + fn + 1 < rows) output[out + 1] = sums[1];
}

#endif
