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
// qh byte contributes one. The trits are exact in half (balanced-prefix
// decode, bonsai_projection.h) before any floating-point activation
// arithmetic. The qh trit keeps the F32 floor: half prefixes from per-lane
// constants measured 1-3% slower at four rows.
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

// Two adjacent activations as F32, one eight-byte load from a device
// pointer (`index` must be even); other handles widen two elements.
static inline metal::float2 bonsai_ptq1_pair(device const float *input, metal::ulong index) {
    return *reinterpret_cast<device const metal::float2 *>(input + index);
}
template<typename Input>
static inline metal::float2 bonsai_ptq1_pair(Input input, metal::ulong index) {
    return metal::float2(float(input[index]), float(input[index + 1]));
}

// One threadgroup is one SIMD group of 32 lanes: lane / 8 selects the packed
// block in flight, lane & 7 its sixteen-element part. Dispatch
// ceil(rows / rows_per_group) groups; `first_token` selects the activation
// rows [first_token, first_token + tokens) of the flattened handles. Each
// output is the SIMD-group sum of every lane's block-ordered sum of
// scale * (the sixteen trit * input products added in factor order), so the
// two kernels below differ in loop order only and give the same bits.
//
// From three tokens, rows' factors are derived once per block and held while
// tokens stream through one input row at a time, so register use is
// 16 * rows_per_group + 16 + rows_per_group * tokens rather than growing by
// sixteen per token (token-major arrays measured 0.44 ms at four tokens but
// 2.44 ms at eight on the 17408x5120 projection).
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
            // Decoded inline: through bonsai_ptq1_factors the same arithmetic
            // measured 193 rather than 188 us at three tokens.
            #pragma clang loop unroll(full)
            for (metal::uint k = 0; k < 3; ++k) {
                const metal::uint byte = k < 2 ? 2 * part + k : 16 + part;
                bonsai_ptq1_trits(
                    bonsai_ptq1_code(packed.code(packed_block, byte)), &factors[row][5 * k]);
            }
            const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[part >> 1];
            const float high =
                float(packed.code(packed_block, 24 + (part & 1))) * (1.0f / 256.0f) * power;
            factors[row][15] =
                half(metal::floor(3.0f * high) - 3.0f * metal::floor(high) - 1.0f);
            scales[row] = packed.scale(packed_block);
        }
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tokens; ++t) {
            float values[16];
            bonsai_ptq1_values(input, input_base + metal::ulong(t) * columns, block, part, values);
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

// Two tokens: both input rows stay in registers and each row is decoded and
// used at once (16 * tokens + 16 live values per lane instead of
// 16 * rows_per_group + 16). Same arithmetic in the same order as above, so
// bit-identical. Each loop order falls off a register cliff in the other's
// range: on M4 Pro (17408x5120) this one measured 140 / 236 / 360 us at
// 2 / 3 / 4 tokens against 145 / 188 / 228 us for the kernel above.
template<
    metal::uint tokens, metal::uint rows_per_group,
    typename Packed, typename Input, typename Output>
static inline void bonsai_ptq1_small_batch_pair_impl(
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
    for (metal::uint block = lane / 8; block < blocks; block += 4) {
        float values[tokens][16];
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tokens; ++t) {
            bonsai_ptq1_values(
                input, input_base + metal::ulong(t) * columns, block, part, values[t]);
        }
        #pragma clang loop unroll(full)
        for (metal::uint row = 0; row < rows_per_group; ++row) {
            const metal::uint source = metal::min(first_row + row, rows - 1);
            const metal::ulong packed_block = metal::ulong(source) * blocks + block;
            half factors[16];
            bonsai_ptq1_factors(packed, packed_block, part, factors);
            const float scale = packed.scale(packed_block);
            #pragma clang loop unroll(full)
            for (metal::uint t = 0; t < tokens; ++t) {
                float total = 0.0f;
                #pragma clang loop unroll(full)
                for (metal::uint i = 0; i < 16; ++i) total += float(factors[i]) * values[t][i];
                sums[row][t] += total * scale;
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
// are added through threadgroup memory. Token rows past `tokens` repeat the
// last row and are not stored: row fm of X only reaches row fm of C. Dispatch
// ceil(rows / 8) groups of 128 threads.
// The summation order differs from the scalar kernel, so outputs are not bit
// identical: on random FFN-shaped weights the largest gap measured 1.5e-7 to
// 2.2e-7 of the largest output magnitude. Half-prefix trit decoding took the
// FFN shapes from 341-350 to 280-301 us with bit-identical outputs, and one
// eight-byte activation load per tile (instead of two loads and two selects)
// to 233-237 / 236-281 us on 17408x5120 / 5120x17408, also bit-identical;
// the F32 8x8 multiplies remain the limit.
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
            // fn is even and rows are whole blocks, so the pair is aligned.
            const metal::float2 pair = bonsai_ptq1_pair(input, base + tile * 8);
            x.thread_elements()[0] = pair.x;
            x.thread_elements()[1] = pair.y;
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
    if (simd != 0 || fm >= tokens) return;
    #pragma clang loop unroll(full)
    for (metal::uint s = 1; s < split; ++s) {
        sums[0] += shared[s * 64 + lane * 2];
        sums[1] += shared[s * 64 + lane * 2 + 1];
    }
    const metal::ulong out = metal::ulong(first_token + fm) * rows + first_row + fn;
    if (first_row + fn < rows) output[out] = sums[0];
    if (first_row + fn + 1 < rows) output[out + 1] = sums[1];
}

// Large-batch variant for prefill chunks: a threadgroup owns 64 output rows
// and 64 tokens and decodes each packed block once into threadgroup memory
// for all of them, as trit * scale in half (exact: the trit is -1/0/+1 and
// the scale is a half), instead of once per eight tokens. Four SIMD groups
// each multiply a 32-token by 32-row quarter with sixteen 8x8 accumulators,
// so every W tile load serves four multiplies and every activation tile load
// four more. Activations stay F32 and go straight from device memory into
// the multiplies (rows past `tokens` repeat the last row and are not
// stored); the F32 x half multiplies keep the F32 operand (checked by
// `matmul_keeps_f32_operand_bits_and_accumulates_beyond_f16_range`) and
// accumulate in F32, so only the summation order differs from the wide
// kernel. Dispatch (ceil(rows / 64), ceil(tokens / 64)) groups of 128
// threads.
//
// M4 Pro, 17408x5120 / 5120x17408: 1,687 / 1,734 us at 64 tokens and
// 3,317 / 3,333 at 128, against eight-token wide passes' 1,912 / 2,292 and
// 3,834 / 4,586 (3.4 T against 3.0 T multiply-adds per second; the 8x8
// multiplies peak near 3.9 T on this part whatever their operand types).
// With decoding skipped after the first block 64 tokens took 1,634 us, so
// the multiplies, not the decode, set the cost. Measured and not kept: a
// padded threadgroup stride (72 halves), 1-2% slower; 32 x 64 quarters per
// SIMD group (two or four groups for 64-256 tokens), 1.3-2x slower from
// register pressure; a 32-token tile (64 threads), 950 / 1,027 us at 32
// tokens against the wide passes' 958 / 1,146.
constant constexpr metal::uint bonsai_large_tile = 64;

template<typename Packed, typename Output>
static inline void bonsai_ptq1_large_batch_impl(
    Packed packed,
    device const float *input,
    Output output,
    metal::uint rows,
    metal::uint columns,
    metal::uint tokens,
    metal::uint2 group,
    metal::uint tid,
    metal::uint simd,
    metal::uint lane,
    threadgroup half *decoded
) {
    // decoded[element * stride + row] holds one block of the tile's rows.
    constexpr metal::uint stride = bonsai_large_tile;
    const metal::uint blocks = columns / 128;
    const metal::uint first_row = group.x * bonsai_large_tile;
    const metal::uint first_token = group.y * bonsai_large_tile;
    // Decoder role: thread tid fills row tid / 2's elements from qs bytes
    // 12h..12h+11 and qh byte 24 + h, h = tid & 1. Rows past the end are
    // clamped to a real row; their products are never stored.
    const metal::uint decode_row = tid / 2, decode_half = tid & 1;
    const metal::ulong decode_base =
        metal::ulong(metal::min(first_row + decode_row, rows - 1)) * blocks;
    // Multiplier role: SIMD group simd covers tokens 32 (simd / 2).. and
    // rows 32 (simd & 1)..; lane owns row fm and columns fn, fn + 1 of each
    // 8x8 tile, as in the wide kernel.
    const metal::uint quad = lane / 4;
    const metal::uint fm = (quad & 4) + ((lane / 2) % 4);
    const metal::uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const metal::uint token_base = first_token + (simd / 2) * 32;
    const metal::uint row_base = (simd & 1) * 32;
    device const float *x[4];
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 4; ++i) {
        x[i] = input +
            metal::ulong(metal::min(token_base + 8 * i + fm, tokens - 1)) * columns + fn;
    }
    metal::simdgroup_float8x8 sums[4][4];
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 4; ++i) {
        #pragma clang loop unroll(full)
        for (metal::uint j = 0; j < 4; ++j) {
            sums[i][j] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    for (metal::uint block = 0; block < blocks; ++block) {
        {
            const metal::ulong packed_block = decode_base + block;
            const half scale = half(packed.scale(packed_block));
            threadgroup half *column = decoded + decode_row;
            #pragma clang loop unroll(full)
            for (metal::uint b = 0; b < 12; ++b) {
                const metal::uint byte = 12 * decode_half + b;
                half trits[5];
                bonsai_ptq1_trits(bonsai_ptq1_code(packed.code(packed_block, byte)), trits);
                #pragma clang loop unroll(full)
                for (metal::uint n = 0; n < 5; ++n) {
                    // Element layout as in bonsai_ptq1_trit.
                    const metal::uint element = byte < 16 ? byte + 16 * n : 64 + byte + 8 * n;
                    column[element * stride] = trits[n] * scale;
                }
            }
            half trits[5];
            bonsai_ptq1_trits(
                bonsai_ptq1_code(packed.code(packed_block, 24 + decode_half)), trits);
            #pragma clang loop unroll(full)
            for (metal::uint n = 0; n < 4; ++n) {
                column[(120 + decode_half + 2 * n) * stride] = trits[n] * scale;
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        const metal::uint offset = block * 128;
        #pragma clang loop unroll(full)
        for (metal::uint k = 0; k < 128; k += 8) {
            metal::simdgroup_half8x8 w[4];
            #pragma clang loop unroll(full)
            for (metal::uint j = 0; j < 4; ++j) {
                metal::simdgroup_load(w[j], decoded + k * stride + row_base + 8 * j, stride);
            }
            #pragma clang loop unroll(full)
            for (metal::uint i = 0; i < 4; ++i) {
                metal::simdgroup_float8x8 a;
                const metal::float2 pair =
                    *reinterpret_cast<device const metal::float2 *>(x[i] + offset + k);
                a.thread_elements()[0] = pair.x;
                a.thread_elements()[1] = pair.y;
                #pragma clang loop unroll(full)
                for (metal::uint j = 0; j < 4; ++j) {
                    metal::simdgroup_multiply_accumulate(sums[i][j], a, w[j], sums[i][j]);
                }
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 4; ++i) {
        const metal::uint token = token_base + 8 * i + fm;
        if (token >= tokens) continue;
        #pragma clang loop unroll(full)
        for (metal::uint j = 0; j < 4; ++j) {
            const metal::uint row = first_row + row_base + 8 * j + fn;
            const metal::ulong out = metal::ulong(token) * rows + row;
            if (row < rows) output[out] = sums[i][j].thread_elements()[0];
            if (row + 1 < rows) output[out + 1] = sums[i][j].thread_elements()[1];
        }
    }
}

#endif
