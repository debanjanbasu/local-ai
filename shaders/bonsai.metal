// PTQ1_0 layout and signed FWHT adapted from PrismML-Eng/llama.cpp,
// revision 9a9394a895b96003ca842a6041cb28ac49a108f7 (MIT).
// Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include "bonsai_projection.h"
using namespace metal;

kernel void bonsai_fwht_forward(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &blocks_per_row [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[1024];
    bonsai_fwht_impl<false>(input, signs, output, blocks_per_row, block, tid, shared);
}

kernel void bonsai_fwht_inverse(
    device const float *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &blocks_per_row [[buffer(3)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[1024];
    bonsai_fwht_impl<true>(input, signs, output, blocks_per_row, block, tid, shared);
}

// Eight lanes per packed block, four blocks in parallel. Collapse each byte's
// five digit products using y[n] - 3*y[n+1], then reuse those coefficients for
// four output rows. Adapted from the pinned fork's ptq1_0_dot_reg/mul_mv path.
// This trades register-local arithmetic for fewer activation and weight loads;
// it does not materialize a dequantized or differently packed weight cache.
kernel void bonsai_ptq1_matvec_reuse(
    device const BonsaiPtq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    bonsai_ptq1_matvec_reuse_impl<4>(
        BonsaiNativePacked{weights}, input, output, rows, columns, group, lane);
}

// Decode each packed byte once for its four/five trits. A padded column-major
// tile avoids the stride-32 shared-memory stores in the elementwise control.
// All matrix operands remain F32; no whole-matrix unpacking cache is retained.
template<uint tile_tokens, uint tile_columns>
static inline void bonsai_ptq1_matmul_bytewise_impl(
    device const BonsaiPtq1Block *weights, device const float *input, device float *output,
    uint rows, uint columns, uint tokens, uint2 group, uint tid, uint simd_group,
    threadgroup float *activations, threadgroup float *decoded, threadgroup float *results
) {
    const uint first_row = group.x * 32, first_token = group.y * tile_tokens;
    const uint blocks = columns / 128, lane = tid % 32;
    simdgroup_matrix<float, 8, 8> sums[tile_tokens / 8];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < tile_tokens / 8; ++t) {
        sums[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint chunk = 0; chunk < columns / tile_columns; ++chunk) {
        const uint block = chunk / (128 / tile_columns);
        const uint column_start = (chunk % (128 / tile_columns)) * tile_columns;
        for (uint i = tid; i < tile_tokens * tile_columns; i += 128) {
            const uint token = first_token + i / tile_columns, column = i % tile_columns;
            activations[i] = token < tokens
                ? input[ulong(token) * columns + chunk * tile_columns + column] : 0.0f;
        }
        if (lane < 26) {
            const uint base = lane < 16 ? lane : (lane < 24 ? 80 + lane - 16 : 120 + lane - 24);
            const uint stride = lane < 16 ? 16 : (lane < 24 ? 8 : 2);
            for (uint local_row = simd_group; local_row < 32; local_row += 4) {
                const uint row = first_row + local_row;
                device const BonsaiPtq1Block &w = weights[ulong(row) * blocks + block];
                uint q = row < rows ? (lane < 24 ? w.qs[lane] : w.qh[lane - 24]) : 0;
                const float scale = row < rows ? float(w.d) : 0.0f;
                #pragma clang loop unroll(full)
                for (uint n = 0; n < 5; ++n) {
                    if (n < 4 || lane < 24) {
                        const uint next = q * 3;
                        const uint column = base + n * stride;
                        if (column >= column_start && column < column_start + tile_columns) {
                            decoded[(column - column_start) * 33 + local_row] =
                                float(int(next >> 8) - 1) * scale;
                        }
                        q = next & 255;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint column = 0; column < tile_columns; column += 8) {
            simdgroup_matrix<float, 8, 8> b;
            simdgroup_load(b, decoded + column * 33 + simd_group * 8, 33);
            #pragma clang loop unroll(full)
            for (uint t = 0; t < tile_tokens / 8; ++t) {
                simdgroup_matrix<float, 8, 8> a;
                simdgroup_load(a, activations + t * 8 * tile_columns + column, tile_columns);
                simdgroup_multiply_accumulate(sums[t], a, b, sums[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < tile_tokens / 8; ++t) {
        simdgroup_store(sums[t], results + t * 8 * 32 + simd_group * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < tile_tokens * 32; i += 128) {
        const uint token = first_token + i / 32, row = first_row + i % 32;
        if (token < tokens && row < rows) output[ulong(token) * rows + row] = results[i];
    }
}

kernel void bonsai_ptq1_matmul_bytewise_32(
    device const BonsaiPtq1Block *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float activations[32 * 64], decoded[64 * 33];
    // Activation storage is dead after the final multiply; reuse it for results.
    bonsai_ptq1_matmul_bytewise_impl<32, 64>(weights, input, output, rows, columns, tokens,
        group, tid, simd_group, activations, decoded, activations);
}

#if __METAL_VERSION__ >= 400
// Tile-local decoding and device activation operands follow Prism's mul_mm.metal.
// PTQ1 values are exactly representable in half; input and destination stay F32.
// Token tiles of 128 serve prefill chunks; 64 and 32 serve speculative verify
// blocks, which would otherwise pay for a mostly empty 128-token tile.
#define BONSAI_PTQ1_MATMUL_TENSOR(NAME, TILE_TOKENS) \
kernel void NAME( \
    device const BonsaiPtq1Block *weights [[buffer(0)]], device float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]], \
    constant uint &columns [[buffer(4)]], constant uint &tokens [[buffer(5)]], \
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]] \
) { \
    threadgroup half decoded[32 * 32]; \
    auto inputs = tensor(input, dextents<int32_t, 2>(columns, tokens), \
        array<int, 2>({1, int(columns)})); \
    auto destination = tensor(output, dextents<int32_t, 2>(rows, tokens), \
        array<int, 2>({1, int(rows)})); \
    bonsai_ptq1_matmul_tensor_impl<TILE_TOKENS, 32, 4>( \
        BonsaiNativePacked{weights}, inputs, destination, rows, columns, tokens, group, tid, decoded); \
}
BONSAI_PTQ1_MATMUL_TENSOR(bonsai_ptq1_matmul_tensor, 128)
BONSAI_PTQ1_MATMUL_TENSOR(bonsai_ptq1_matmul_tensor_64, 64)
BONSAI_PTQ1_MATMUL_TENSOR(bonsai_ptq1_matmul_tensor_32, 32)
#endif
