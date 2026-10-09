// PTQ1_0 layout and signed FWHT adapted from PrismML-Eng/llama.cpp,
// revision 9a9394a895b96003ca842a6041cb28ac49a108f7 (MIT).
// Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
#ifndef BONSAI_PROJECTION_H
#define BONSAI_PROJECTION_H

#include <metal_stdlib>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

struct BonsaiPtq1Block {
    metal::uchar qs[24];
    metal::uchar qh[2];
    half d;
};
static_assert(sizeof(BonsaiPtq1Block) == 28, "PTQ1_0 block layout changed");

// Rotate one 1024-wide block whose thread tid holds elements i * 128 + tid in
// values, as loaded; fused producers compute those values in registers.
template<bool inverse, typename Signs, typename Output>
static inline void bonsai_fwht_values(
    thread float *values,
    Signs signs,
    Output output,
    metal::uint blocks_per_row,
    metal::uint block,
    metal::uint tid,
    threadgroup float *shared
) {
    const metal::ulong base = metal::ulong(block) * 1024;
    const metal::uint sign_base = (block % blocks_per_row) * 1024;
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 8; ++i) {
        const metal::uint local = i * 128 + tid;
        const float sign = inverse ? 1.0f : signs[sign_base + local];
        values[i] = values[i] * sign * (1.0f / 32.0f);
    }
    for (metal::uint distance = 1; distance < 32; distance <<= 1) {
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) {
            const float a = values[i];
            const float b = metal::simd_shuffle_xor(a, distance);
            values[i] = (tid & distance) == 0 ? a + b : b - a;
        }
    }
    for (metal::uint distance = 32; distance < 128; distance <<= 1) {
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) shared[i * 128 + tid] = values[i];
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) {
            const float a = values[i];
            const float b = shared[i * 128 + (tid ^ distance)];
            values[i] = (tid & distance) == 0 ? a + b : b - a;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    for (metal::uint step = 1; step < 8; step <<= 1) {
        for (metal::uint i = 0; i < 8; i += 2 * step) {
            for (metal::uint j = 0; j < step; ++j) {
                const float a = values[i + j];
                const float b = values[i + j + step];
                values[i + j] = a + b;
                values[i + j + step] = a - b;
            }
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 8; ++i) {
        const metal::uint local = i * 128 + tid;
        const float sign = inverse ? signs[sign_base + local] : 1.0f;
        output[base + local] = values[i] * sign;
    }
}

// Indexable arguments may be device pointers or rank-one tensor handles.
template<bool inverse, typename Input, typename Signs, typename Output>
static inline void bonsai_fwht_impl(
    Input input,
    Signs signs,
    Output output,
    metal::uint blocks_per_row,
    metal::uint block,
    metal::uint tid,
    threadgroup float *shared
) {
    const metal::ulong base = metal::ulong(block) * 1024;
    float values[8];
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 8; ++i) values[i] = input[base + i * 128 + tid];
    bonsai_fwht_values<inverse>(values, signs, output, blocks_per_row, block, tid, shared);
}

struct BonsaiNativePacked {
    device const BonsaiPtq1Block *data;

    metal::uchar code(metal::ulong block, metal::uint byte) const thread {
        return byte < 24 ? data[block].qs[byte] : data[block].qh[byte - 24];
    }

    float scale(metal::ulong block) const thread { return float(data[block].d); }
};

template<typename Bytes>
struct BonsaiTensorPacked {
    Bytes data;

    BonsaiTensorPacked(Bytes bytes) thread : data(bytes) {}

    metal::uchar code(metal::ulong block, metal::uint byte) const thread {
        return data[block * 28 + byte];
    }

    float scale(metal::ulong block) const thread {
        const metal::ushort bits = metal::ushort(code(block, 26)) |
            (metal::ushort(code(block, 27)) << 8);
        return float(as_type<half>(bits));
    }
};
#if __METAL_VERSION__ >= 400
template<typename Bytes>
BonsaiTensorPacked(Bytes) -> BonsaiTensorPacked<Bytes>;
#endif

template<typename Packed>
static inline int bonsai_ptq1_trit(Packed packed, metal::ulong block, metal::uint element) {
    const metal::uint powers[5] = {1, 3, 9, 27, 81};
    metal::uint byte_index;
    metal::uint power;
    if (element < 80) {
        byte_index = element & 15;
        power = element >> 4;
    } else if (element < 120) {
        const metal::uint t = element - 80;
        byte_index = 16 + (t & 7);
        power = t >> 3;
    } else {
        const metal::uint t = element - 120;
        byte_index = 24 + (t & 1);
        power = t >> 1;
    }
    const metal::uint byte = packed.code(block, byte_index);
    return int((((byte * powers[power]) & 255) * 3) >> 8) - 1;
}

// Base-3 prefixes of a code byte from one half FMA and one subtraction each,
// with no floor or integer conversion, which are quarter-rate on Apple GPUs:
// on an M4 Pro the floor-based decode left every PTQ1 matvec ALU-bound at
// about 145 GB/s.
// OR-ing a byte q into the mantissa of 1024 gives the half 1024 + q exactly.
// One half FMA then rounds onto [1024, 2048), where halves are integers, so
// it computes a floor: fma(1024 + q, 3^p / 256, a_p) is exactly
// 1024 + floor(q * 3^p / 256) + c_p for every byte. a_p shifts the product by
// (128 + 256 [p even]) * 3^p / 256, half an integer above the floor, so no
// byte rounds to a tie except q = 0, which rounds up because the integer
// below is odd for that choice of offset. Subtracting 1024 + c_p is exact, so
// each prefix is the integer the floor produced and decoded values are
// unchanged (local-metal/tests/bonsai.rs checks every byte at every element).
static inline half bonsai_ptq1_code(metal::uint byte) {
    return as_type<half>(metal::ushort(0x6400u | byte));
}

// floor(q * 3^p / 256) for p = 1..5 (p is a constant once unrolled).
static inline half bonsai_ptq1_prefix(half code, metal::uint p) {
    const half powers[5] = {
        3.0h / 256.0h, 9.0h / 256.0h, 27.0h / 256.0h, 81.0h / 256.0h, 243.0h / 256.0h};
    const half shifts[5] = {1013.5h, 1001.5h, 929.5h, 821.5h, 173.5h};
    const half offsets[5] = {1026.0h, 1038.0h, 1038.0h, 1146.0h, 1146.0h};
    return metal::fma(code, powers[p - 1], shifts[p - 1]) - offsets[p - 1];
}

// The same prefix minus k_p = (3^p - 1) / 2 = 1, 4, 13, 40, 121: the
// balanced-ternary value of the first p trits, an integer in [-121, 121] and
// exact in half. Folding k_p into the offset keeps one FMA and one subtraction.
static inline half bonsai_ptq1_balanced_prefix(half code, metal::uint p) {
    const half powers[5] = {
        3.0h / 256.0h, 9.0h / 256.0h, 27.0h / 256.0h, 81.0h / 256.0h, 243.0h / 256.0h};
    const half shifts[5] = {1013.5h, 1001.5h, 929.5h, 821.5h, 173.5h};
    const half offsets[5] = {1027.0h, 1042.0h, 1051.0h, 1186.0h, 1267.0h};
    return metal::fma(code, powers[p - 1], shifts[p - 1]) - offsets[p - 1];
}

// The five trits of a qs byte, -1/0/+1 in digit order, exactly. With
// balanced prefixes B_p = prefix(p) - k_p and k_{p+1} = 3 k_p + 1, trit n =
// prefix(n + 1) - 3 prefix(n) - 1 = B_{n+1} - 3 B_n, and B_1 is trit 0: one
// FMA per trit after the first instead of a subtraction and an FMA. The
// trits are the same values; on M4 Pro the scalar small-batch kernel went
// from 156.4 / 193.9 / 230.8 to 144.7 / 187.0 / 227.8 us at 2 / 3 / 4 rows
// (17408x5120) with bit-identical outputs.
static inline void bonsai_ptq1_trits(half code, thread half *trits) {
    half previous = 0.0h;
    #pragma clang loop unroll(full)
    for (metal::uint n = 0; n < 5; ++n) {
        const half balanced = bonsai_ptq1_balanced_prefix(code, n + 1);
        trits[n] = n == 0 ? balanced : metal::fma(-3.0h, previous, balanced);
        previous = balanced;
    }
}

// Base pointers of the packed rows one SIMD group reduces. The fused kernels
// choose them per group, so rows may come from different matrices that share
// the input vector; the inner loop never sees which.
template<metal::uint rows_per_group>
struct BonsaiPtq1Rows {
    device const BonsaiPtq1Block *blocks[rows_per_group];
};

// Rows first_row.. of one matrix; rows past the end are clamped to a real row
// rather than branched around, and the caller never writes their sums.
template<metal::uint rows_per_group>
static inline BonsaiPtq1Rows<rows_per_group> bonsai_ptq1_rows(
    device const BonsaiPtq1Block *weights,
    metal::uint first_row,
    metal::uint rows,
    metal::uint blocks
) {
    BonsaiPtq1Rows<rows_per_group> result;
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) {
        result.blocks[row] =
            weights + metal::ulong(metal::min(first_row + row, rows - 1)) * blocks;
    }
    return result;
}

// Four lanes per packed block, eight blocks in flight. Lane p = lane & 3 owns
// qs bytes 4p..4p+3 (elements 16n + 4p + i), qs bytes 16 + 2p + i (elements
// 80 + 8n + 2p + i) and the two qh trits at power p (elements 120 + 2p + i),
// so a row's block arrives in four loads (uchar4, uchar2, uchar2, half) and
// the activations as float4/float2. Each byte's five digit products collapse
// against y[n] - 3*y[n+1] (adapted from the pinned fork's ptq1_0_dot_reg), so
// a trit costs one F32 FMA on a half prefix, and the coefficients are shared
// by every row the SIMD group reduces. Row base pointers are computed once
// and rows past the end are clamped to a real row (their sums are never
// written). The single-token entry points reject a matrix that is not
// 4-byte aligned.
//
// M4 Pro, 17408x5120: 135 us (floor decode, eight lanes per block, four rows)
// to 99 us with half prefixes, 96 us at eight rows per group and 86.5 us with
// this layout; 248320x5120 from 1848 to 1142 us (150 to 243 GB/s). Sixteen
// rows per group spill (250 us). Exact trits dotted with the activations are
// 18 times more accurate but measured 112 against 99 us, so the matvec keeps
// the telescoped form the floor decoder used.
//
// Every lane returns the SIMD-group total of each row in sums.
template<metal::uint rows_per_group>
static inline void bonsai_ptq1_matvec_reuse_impl(
    BonsaiPtq1Rows<rows_per_group> row_blocks,
    device const float *input,
    metal::uint columns,
    metal::uint lane,
    thread float *sums
) {
    const metal::uint blocks = columns / 128;
    const metal::uint part = lane & 3;
    const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[part];
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) sums[row] = 0.0f;
    for (metal::uint block = lane / 4; block < blocks; block += 8) {
        device const float *x = input + block * 128;
        float4 wide[5];
        float2 narrow[5];
        float4 wide_sum = 0.0f;
        float2 narrow_sum = 0.0f;
        {
            float4 v[5];
            float2 u[5];
            #pragma clang loop unroll(full)
            for (metal::uint n = 0; n < 5; ++n) {
                v[n] = *reinterpret_cast<device const float4 *>(x + 16 * n + 4 * part);
                u[n] = *reinterpret_cast<device const float2 *>(x + 80 + 8 * n + 2 * part);
                wide_sum += v[n];
                narrow_sum += u[n];
            }
            #pragma clang loop unroll(full)
            for (metal::uint n = 0; n < 4; ++n) {
                wide[n] = v[n] - 3.0f * v[n + 1];
                narrow[n] = u[n] - 3.0f * u[n + 1];
            }
            wide[4] = v[4];
            narrow[4] = u[4];
        }
        const float2 high_inputs = *reinterpret_cast<device const float2 *>(x + 120 + 2 * part);
        const float input_sum = (wide_sum.x + wide_sum.y) + (wide_sum.z + wide_sum.w) +
            (narrow_sum.x + narrow_sum.y) + (high_inputs.x + high_inputs.y);
        #pragma clang loop unroll(full)
        for (metal::uint row = 0; row < rows_per_group; ++row) {
            device const BonsaiPtq1Block &w = row_blocks.blocks[row][block];
            const metal::uchar4 quad =
                *reinterpret_cast<device const metal::uchar4 *>(&w.qs[4 * part]);
            const metal::uchar2 pair =
                *reinterpret_cast<device const metal::uchar2 *>(&w.qs[16 + 2 * part]);
            const metal::uchar2 high = *reinterpret_cast<device const metal::uchar2 *>(&w.qh[0]);
            const metal::uchar bytes[6] = {quad.x, quad.y, quad.z, quad.w, pair.x, pair.y};
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (metal::uint i = 0; i < 6; ++i) {
                const half code = bonsai_ptq1_code(bytes[i]);
                #pragma clang loop unroll(full)
                for (metal::uint n = 0; n < 5; ++n) {
                    total += float(bonsai_ptq1_prefix(code, n + 1)) *
                        (i < 4 ? wide[n][i] : narrow[n][i - 4]);
                }
            }
            // One trit per qh byte; half prefixes measured slightly slower here.
            #pragma clang loop unroll(full)
            for (metal::uint i = 0; i < 2; ++i) {
                const float code = float(high[i]) * (1.0f / 256.0f) * power;
                total += (metal::floor(3.0f * code) - 3.0f * metal::floor(code)) * high_inputs[i];
            }
            sums[row] += (total - input_sum) * float(w.d);
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) sums[row] = metal::simd_sum(sums[row]);
}

#if __METAL_VERSION__ >= 400
template<int TileTokens, int TileRows, int Simdgroups,
    typename Packed, typename InputTensor, typename OutputTensor>
static inline void bonsai_ptq1_matmul_tensor_impl(
    Packed packed,
    InputTensor input,
    OutputTensor output,
    metal::uint rows,
    metal::uint columns,
    metal::uint tokens,
    metal::uint2 group,
    metal::uint tid,
    threadgroup half *decoded
) {
    constexpr int tile_rows = TileRows, tile_tokens = TileTokens, tile_k = 32;
    const metal::uint first_row = group.x * tile_rows, first_token = group.y * tile_tokens;
    const metal::uint blocks = columns / 128;
    auto weights_tile = metal::tensor(decoded, metal::dextents<int32_t, 2>(tile_k, tile_rows));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            tile_tokens, tile_rows, static_cast<int>(metal::dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroups<Simdgroups>> multiply;
    auto sums = multiply.template get_destination_cooperative_tensor<
        InputTensor, decltype(weights_tile), float>();
    // multiply_accumulate consumes the destination on its first run too.
    // Cooperative tensor storage is not a promise of zero-initialized values.
    #pragma clang loop unroll(full)
    for (metal::ushort i = 0; i < sums.get_capacity(); ++i) {
        if (sums.is_valid_element(i)) sums[i] = 0.0f;
    }
    for (metal::uint start = 0; start < columns; start += tile_k) {
        for (metal::uint i = tid; i < tile_rows * tile_k; i += 128) {
            const metal::uint row = first_row + i / tile_k, column = start + i % tile_k;
            half value = half(0);
            if (row < rows) {
                const metal::ulong block = metal::ulong(row) * blocks + column / 128;
                value = half(bonsai_ptq1_trit(packed, block, column % 128)) *
                    half(packed.scale(block));
            }
            decoded[i] = value;
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        auto input_tile = input.template slice<32, metal::dynamic_extent>(start, first_token);
        multiply.run(input_tile, weights_tile, sums);
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    sums.store(output.slice(first_row, first_token));
}

#endif

#endif
