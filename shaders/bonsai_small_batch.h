// Small-batch PTQ1_0 projection for speculative verification blocks and short
// prefill chunks. The 128-token tensor tile costs a full tile for any block of
// 2..128 rows; here one SIMD group covers a few output rows and
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

// One threadgroup is one SIMD group of 32 lanes in sixteen lane pairs
// (lane, lane ^ 1). Lane & 1 selects which `rows_per_lane` of the group's
// 2 * rows_per_lane output rows a lane owns; lane / 16 selects the packed block
// in flight and (lane / 2) & 7 its sixteen-element part, so both lanes of a
// pair walk the same elements. For each pair of tokens a lane loads one input
// row and takes the other from its partner with a shuffle, which halves the
// device loads per product: the kernel issues close to one instruction per
// lane-cycle, and shuffles and loads do not compete for the same slots
// (lane quads that load one token and shuffle three measured slower). On M4
// Pro (17408x5120 / 5120x17408), 2/3/4 tokens went from 140/187/228 and
// 142/189/234 us (eight lanes per block, four rows per group) to 135/175/214
// and 143/181/223 us, the 248320x5120 output head from 1918/2591/3170 to
// 1819/2375/2920 us; 1024-row projections, now covered by half as many
// groups, from 12.7/16.0/19.2 to 16.3/18.9/21.6 us. Dispatch
// ceil(rows / (2 * rows_per_lane)) groups; `first_token` selects the
// activation rows [first_token, first_token + tokens) of the flattened handles.
//
// Each output is the sum over its sixteen lanes of every lane's block-ordered
// sum of scale * (the sixteen trit * input products added in factor order),
// reduced by the same shuffles for every token count, so 2, 3 and 4 tokens give
// the same bits. Rows' factors are derived once per block and held while tokens
// stream through, so register use is 16 * rows_per_lane + 32 + rows_per_lane *
// tokens rather than growing by sixteen per token (token-major arrays measured
// 0.44 ms at four tokens but 2.44 ms at eight on the 17408x5120 projection);
// five rows per lane, or holding every input row and decoding rows one at a
// time, fall off a register cliff.
template<
    metal::uint tokens, metal::uint rows_per_lane,
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
    constexpr metal::uint steps = (tokens + 1) / 2;
    const metal::uint slice = lane & 1;
    const metal::uint pair = lane >> 1;
    const metal::uint part = pair & 7;
    const metal::uint first_row = (2 * group + slice) * rows_per_lane;
    const metal::uint blocks = columns / 128;
    const metal::ulong input_base = metal::ulong(first_token) * columns;
    const metal::ulong output_base = metal::ulong(first_token) * rows;
    // sums[row][2 * step + m]: m = 0 the token this lane loads, m = 1 its partner's.
    float sums[rows_per_lane][2 * steps];
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_lane; ++row) {
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 2 * steps; ++i) sums[row][i] = 0.0f;
    }
    for (metal::uint block = pair / 8; block < blocks; block += 2) {
        // The trits are exact in half; holding them as half halves their
        // registers, and every use widens them back into an F32 FMA, so the
        // outputs are bit-identical to F32 factors. Measured on M4 Pro
        // (17408x5120 / 5120x17408), 2/3/4 tokens went from 200/256/324 and
        // 197/259/344 us to 191/235/281 and 194/237/286 us, and with the
        // half-prefix decoder (same trits) to 152/192/231 and 155/194/238 us.
        half factors[rows_per_lane][16];
        float scales[rows_per_lane];
        #pragma clang loop unroll(full)
        for (metal::uint row = 0; row < rows_per_lane; ++row) {
            // Clamp to a real row: the value is discarded below, nothing is
            // read out of range.
            const metal::uint source = metal::min(first_row + row, rows - 1);
            const metal::ulong packed_block = metal::ulong(source) * blocks + block;
            // Bytes 2 * part, 2 * part + 1 and 16 + part contribute five trits
            // each, exact in half (balanced-prefix decode, bonsai_projection.h),
            // then the qh byte contributes one. The qh trit keeps the F32
            // floor: half prefixes from per-lane constants measured 1-3% slower
            // at four rows, 16-bit integer digits the same.
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
        for (metal::uint step = 0; step < steps; ++step) {
            // An odd last token is loaded by both lanes of the pair.
            const bool paired = 2 * step + 1 < tokens;
            const metal::uint token = 2 * step + (paired ? slice : 0);
            float own[16];
            bonsai_ptq1_values(input, input_base + metal::ulong(token) * columns, block, part, own);
            #pragma clang loop unroll(full)
            for (metal::uint m = 0; m < (paired ? 2u : 1u); ++m) {
                float values[16];
                #pragma clang loop unroll(full)
                for (metal::uint i = 0; i < 16; ++i) {
                    values[i] = m == 0 ? own[i] : metal::simd_shuffle_xor(own[i], 1);
                }
                #pragma clang loop unroll(full)
                for (metal::uint row = 0; row < rows_per_lane; ++row) {
                    float total = 0.0f;
                    #pragma clang loop unroll(full)
                    for (metal::uint i = 0; i < 16; ++i) {
                        total += float(factors[row][i]) * values[i];
                    }
                    sums[row][2 * step + m] += total * scales[row];
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_lane; ++row) {
        #pragma clang loop unroll(full)
        for (metal::uint step = 0; step < steps; ++step) {
            const bool paired = 2 * step + 1 < tokens;
            #pragma clang loop unroll(full)
            for (metal::uint m = 0; m < (paired ? 2u : 1u); ++m) {
                // Sum the sixteen lanes that share this lane's slice.
                float total = sums[row][2 * step + m];
                #pragma clang loop unroll(full)
                for (metal::uint distance = 2; distance < 32; distance *= 2) {
                    total += metal::simd_shuffle_xor(total, distance);
                }
                const metal::uint token = 2 * step + (paired ? slice ^ m : 0);
                if (pair == 0 && first_row + row < rows) {
                    output[output_base + metal::ulong(token) * rows + first_row + row] = total;
                }
            }
        }
    }
}

// Wide variant for 5..8 activation rows, one dispatch instead of two that
// each re-stream the matrix. A threadgroup covers `tiles` eight-row tiles of
// output rows (two, or one for short matrices), with eight SIMD groups
// splitting the packed blocks; each SIMD group multiplies 8x8 simdgroup
// matrices C[token, row] += X[token, k] * W[k, row] for every tile, so one
// activation load feeds `tiles` multiplies. The W tiles hold exact trits
// (each lane decodes rows `fn` and `fn + 1` of each tile at element `fm`),
// so every product is exact before the F32 MMA accumulation; a block's
// partial sums are scaled once, as in the matvec, then the eight SIMD
// groups' partials are added through threadgroup memory.
// Token rows past `tokens` repeat the last row and are not stored: row fm of
// X only reaches row fm of C. Dispatch ceil(rows / (8 * tiles)) groups of
// 256 threads.
// The summation order differs from the scalar kernel, so outputs are not bit
// identical: on random FFN-shaped weights the largest gap measured 1.5e-7 to
// 2.2e-7 of the largest output magnitude. Half-prefix trit decoding took the
// FFN shapes from 341-350 to 280-301 us with bit-identical outputs, and one
// eight-byte activation load per tile (instead of two loads and two selects)
// to 233-237 / 236-281 us on 17408x5120 / 5120x17408.
//
// With one eight-row tile and four SIMD groups per threadgroup, 5120x17408
// cost 235 us at five rows but 281 at eight (248 at seven), against 237 on
// 17408x5120 with the same bytes and multiplies: every threadgroup streams
// all its activations (557 KB at eight tokens of 17,408 columns, 164 KB at
// 5,120) for 28 bytes of weights per 512 activation bytes, and at eight
// distinct token rows the wide matrix's activations apparently stop fitting
// in cache (rows past `tokens` repeat a cached row, hence the knee). Sharing
// each activation load between two row tiles halves that traffic: 5120x17408
// at eight rows went from 281 to 237 us and the other shapes stayed within
// 1%.
// The qh trit's integer decode below then saved about 1% and eight SIMD
// groups instead of four 0.5-3%: on M4 Pro, 5 / 8 rows now take 228 / 229 us
// on both 17408x5120 and 5120x17408 (from 233 / 237 and 235 / 283), 163 / 163
// on 12288x5120 (from 166 / 169) and 86 / 86 on 6144x5120 (from 86 / 88).
// With the trits replaced by constants this layout takes 205 us at eight rows
// on 17408x5120, so the decode adds about 12% to the multiplies and loads.
//
// Inside a block K is permuted: tile j holds elements 8j..8j+7, so the X tile
// is eight contiguous activations per token and the W tile's element 8j + fm
// comes from byte fm + 8 (j & 1) at power j >> 1 (j < 10), byte 16 + fm at
// power j - 10 (j < 15), or qh byte 24 + (fm & 1) at power fm >> 1 (j = 15).
template<metal::uint tiles, typename Packed, typename Input, typename Output>
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
    constexpr metal::uint split = 8;
    // Lane ownership of simdgroup_float8x8 thread_elements(): row fm,
    // columns fn and fn + 1.
    const metal::uint quad = lane / 4;
    const metal::uint fm = (quad & 4) + ((lane / 2) % 4);
    const metal::uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const metal::uint first_row = group * 8 * tiles;
    const metal::uint blocks = columns / 128;
    // Clamp to real rows and tokens: the values are discarded, nothing is read
    // out of range.
    metal::ulong row_base[tiles][2];
    #pragma clang loop unroll(full)
    for (metal::uint t = 0; t < tiles; ++t) {
        #pragma clang loop unroll(full)
        for (metal::uint r = 0; r < 2; ++r) {
            row_base[t][r] =
                metal::ulong(metal::min(first_row + 8 * t + fn + r, rows - 1)) * blocks;
        }
    }
    const metal::ulong input_row =
        metal::ulong(first_token + metal::min(fm, tokens - 1)) * columns + fn;
    // The qh trit at power p is ((q * 3^p) & 255) * 3 >> 8, minus one, in
    // 16-bit integers: the value bonsai_ptq1_trit gives, the floor formula's
    // trit without its two quarter-rate F32 floors.
    const metal::ushort power = metal::ushort4(1, 3, 9, 27)[fm >> 1];
    float sums[tiles][2] = {};
    for (metal::uint block = simd; block < blocks; block += split) {
        half trits[tiles][2][16];
        float scales[tiles][2];
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tiles; ++t) {
            #pragma clang loop unroll(full)
            for (metal::uint r = 0; r < 2; ++r) {
                const metal::ulong packed_block = row_base[t][r] + block;
                scales[t][r] = packed.scale(packed_block);
                #pragma clang loop unroll(full)
                for (metal::uint k = 0; k < 3; ++k) {
                    const metal::uint byte = k == 0 ? fm : (k == 1 ? fm + 8 : 16 + fm);
                    half f[5];
                    bonsai_ptq1_trits(bonsai_ptq1_code(packed.code(packed_block, byte)), f);
                    #pragma clang loop unroll(full)
                    for (metal::uint n = 0; n < 5; ++n) {
                        trits[t][r][k < 2 ? 2 * n + k : 10 + n] = f[n];
                    }
                }
                const metal::ushort high =
                    (metal::ushort(packed.code(packed_block, 24 + (fm & 1))) * power) & 255;
                trits[t][r][15] = half((high * 3) >> 8) - 1.0h;
            }
        }
        const metal::ulong base = input_row + metal::ulong(block) * 128;
        metal::simdgroup_float8x8 partial[tiles];
        #pragma clang loop unroll(full)
        for (metal::uint tile = 0; tile < 16; ++tile) {
            metal::simdgroup_float8x8 x;
            // fn is even and rows are whole blocks, so the pair is aligned.
            const metal::float2 pair = bonsai_ptq1_pair(input, base + tile * 8);
            x.thread_elements()[0] = pair.x;
            x.thread_elements()[1] = pair.y;
            #pragma clang loop unroll(full)
            for (metal::uint t = 0; t < tiles; ++t) {
                metal::simdgroup_float8x8 w;
                w.thread_elements()[0] = float(trits[t][0][tile]);
                w.thread_elements()[1] = float(trits[t][1][tile]);
                if (tile == 0) {
                    metal::simdgroup_multiply(partial[t], x, w);
                } else {
                    metal::simdgroup_multiply_accumulate(partial[t], x, w, partial[t]);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < tiles; ++t) {
            sums[t][0] = metal::fma(partial[t].thread_elements()[0], scales[t][0], sums[t][0]);
            sums[t][1] = metal::fma(partial[t].thread_elements()[1], scales[t][1], sums[t][1]);
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint t = 0; t < tiles; ++t) {
        shared[(t * split + simd) * 64 + lane * 2] = sums[t][0];
        shared[(t * split + simd) * 64 + lane * 2 + 1] = sums[t][1];
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (simd != 0 || fm >= tokens) return;
    #pragma clang loop unroll(full)
    for (metal::uint t = 0; t < tiles; ++t) {
        #pragma clang loop unroll(full)
        for (metal::uint s = 1; s < split; ++s) {
            sums[t][0] += shared[(t * split + s) * 64 + lane * 2];
            sums[t][1] += shared[(t * split + s) * 64 + lane * 2 + 1];
        }
        const metal::uint row = first_row + 8 * t + fn;
        const metal::ulong out = metal::ulong(first_token + fm) * rows + row;
        if (row < rows) output[out] = sums[t][0];
        if (row + 1 < rows) output[out + 1] = sums[t][1];
    }
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
