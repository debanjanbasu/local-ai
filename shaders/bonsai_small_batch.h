// Small-batch PTQ1_0 projection for speculative verification blocks and short
// prefill chunks. The 128-token tensor tile costs a full tile for any block of
// 2..128 rows; here one SIMD group covers `rows_per_group` output rows and
// streams each packed block exactly once, like bonsai_ptq1_matvec_reuse. The
// block's actual -1/0/+1 trits are derived once and reused for every activation
// row. Dot them directly against inputs: telescoping cumulative base-3 factors
// against transformed inputs amplifies cancellation even on modest F32 values.
// Accumulation and outputs are F32; nothing is unpacked outside registers.
//
// `Packed` is BonsaiNativePacked or BonsaiTensorPacked. `Input`/`Output` are
// indexable rank-one handles (device pointers or tensor handles) holding
// `[tokens_total, columns]` activations and `[tokens_total, rows]` results;
// input elements may be float or half and are widened before use.
#ifndef BONSAI_SMALL_BATCH_H
#define BONSAI_SMALL_BATCH_H

#include "bonsai_projection.h"

// Factor order matches bonsai_ptq1_dot_reuse: bytes 2*lane, 2*lane+1, 16+lane
// contribute five trits each, then the qh byte contributes one. Decode the
// small exact integers before any floating-point activation arithmetic.
template<typename Packed>
static inline void bonsai_ptq1_factors(
    Packed packed, metal::ulong block, metal::uint lane, thread float (&factors)[16]
) {
    #pragma clang loop unroll(full)
    for (metal::uint k = 0; k < 3; ++k) {
        const metal::uint byte = k < 2 ? 2 * lane + k : 16 + lane;
        const float code = float(packed.code(block, byte)) * (1.0f / 256.0f);
        factors[5 * k] = metal::floor(3.0f * code);
        factors[5 * k + 1] = metal::floor(9.0f * code);
        factors[5 * k + 2] = metal::floor(27.0f * code);
        factors[5 * k + 3] = metal::floor(81.0f * code);
        factors[5 * k + 4] = metal::floor(243.0f * code);
        #pragma clang loop unroll(full)
        for (metal::uint n = 4; n > 0; --n) {
            factors[5 * k + n] -= 3.0f * factors[5 * k + n - 1] + 1.0f;
        }
        factors[5 * k] -= 1.0f;
    }
    const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[lane >> 1];
    const float high = float(packed.code(block, 24 + (lane & 1))) * (1.0f / 256.0f) * power;
    factors[15] = metal::floor(3.0f * high) - 3.0f * metal::floor(high) - 1.0f;
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
        float factors[rows_per_group][16];
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
                for (metal::uint i = 0; i < 16; ++i) total += factors[row][i] * values[i];
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

#endif
