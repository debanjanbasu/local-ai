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
    const metal::uint sign_base = (block % blocks_per_row) * 1024;
    float values[8];
    #pragma clang loop unroll(full)
    for (metal::uint i = 0; i < 8; ++i) {
        const metal::uint local = i * 128 + tid;
        const float sign = inverse ? 1.0f : signs[sign_base + local];
        values[i] = input[base + local] * sign * (1.0f / 32.0f);
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

template<typename Packed>
static inline float bonsai_ptq1_dot_reuse(
    Packed packed,
    metal::ulong block,
    thread const float *coefficients,
    float input_sum,
    metal::uint lane
) {
    float total = 0.0f;
    #pragma clang loop unroll(full)
    for (metal::uint k = 0; k < 3; ++k) {
        const metal::uint byte = k < 2 ? 2 * lane + k : 16 + lane;
        const float code = float(packed.code(block, byte)) * (1.0f / 256.0f);
        total += metal::floor(3.0f * code) * coefficients[5 * k];
        total += metal::floor(9.0f * code) * coefficients[5 * k + 1];
        total += metal::floor(27.0f * code) * coefficients[5 * k + 2];
        total += metal::floor(81.0f * code) * coefficients[5 * k + 3];
        total += metal::floor(243.0f * code) * coefficients[5 * k + 4];
    }
    const float power = metal::float4(1.0f, 3.0f, 9.0f, 27.0f)[lane >> 1];
    const float high = float(packed.code(block, 24 + (lane & 1))) * (1.0f / 256.0f) * power;
    total += (metal::floor(3.0f * high) - 3.0f * metal::floor(high)) * coefficients[15];
    return (total - input_sum) * packed.scale(block);
}

template<metal::uint rows_per_group, typename Packed, typename Input, typename Output>
static inline void bonsai_ptq1_matvec_reuse_impl(
    Packed packed,
    Input input,
    Output output,
    metal::uint rows,
    metal::uint columns,
    metal::uint group,
    metal::uint lane
) {
    const metal::uint first_row = group * rows_per_group;
    const metal::uint blocks = columns / 128;
    const metal::uint part = lane & 7;
    float sums[rows_per_group] = {0.0f};
    for (metal::uint block = lane / 8; block < blocks; block += 4) {
        float coefficients[16];
        float input_sum = 0.0f;
        #pragma clang loop unroll(full)
        for (metal::uint k = 0; k < 3; ++k) {
            const metal::uint base = k < 2 ? 2 * part + k : 80 + part;
            const metal::uint stride = k < 2 ? 16 : 8;
            float values[5];
            #pragma clang loop unroll(full)
            for (metal::uint n = 0; n < 5; ++n) {
                values[n] = input[block * 128 + base + n * stride];
                input_sum += values[n];
            }
            #pragma clang loop unroll(full)
            for (metal::uint n = 0; n < 4; ++n) {
                coefficients[5 * k + n] = values[n] - 3.0f * values[n + 1];
            }
            coefficients[5 * k + 4] = values[4];
        }
        coefficients[15] = input[block * 128 + 120 + part];
        input_sum += coefficients[15];
        #pragma clang loop unroll(full)
        for (metal::uint row = 0; row < rows_per_group; ++row) {
            if (first_row + row < rows) {
                const metal::ulong packed_block = metal::ulong(first_row + row) * blocks + block;
                sums[row] += bonsai_ptq1_dot_reuse(
                    packed, packed_block, coefficients, input_sum, part);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint row = 0; row < rows_per_group; ++row) {
        const float total = metal::simd_sum(sums[row]);
        if (lane == 0 && first_row + row < rows) output[first_row + row] = total;
    }
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
