// PTQ1_0 layout and signed FWHT adapted from PrismML-Eng/llama.cpp,
// revision 9a9394a895b96003ca842a6041cb28ac49a108f7 (MIT).
// Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include "bonsai_projection.h"
#include "bonsai_mixer.h"
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

// bo_swiglu followed by the forward rotation in one dispatch, for multi-row
// FFN blocks whose gate and up projections are stored. Each thread forms its
// eight elements exactly as bo_swiglu does, in F32 registers, and rotates them
// as bonsai_fwht_forward would after reloading the stored product, so the
// product is neither written nor reread. Every element is read before the
// rotation's first barrier and only its own block is written, so the output
// may alias gate or up.
kernel void bonsai_swiglu_fwht_forward(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &blocks_per_row [[buffer(4)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[1024];
    const ulong base = ulong(block) * 1024;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const ulong index = base + i * 128 + tid;
        values[i] = bonsai_silu(gate[index]) * up[index];
    }
    bonsai_fwht_values<false>(values, signs, output, blocks_per_row, block, tid, shared);
}

// Eight output rows per SIMD group share activation coefficients; see
// bonsai_ptq1_matvec_reuse_impl. Adapted from the pinned fork's
// ptq1_0_dot_reg/mul_mv path. This trades register-local arithmetic for fewer
// activation and weight loads; it does not materialize a dequantized or
// differently packed weight cache.
kernel void bonsai_ptq1_matvec_reuse(
    device const BonsaiPtq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 8;
    float sums[8];
    bonsai_ptq1_matvec_reuse_impl<8>(
        bonsai_ptq1_rows<8>(weights, first_row, rows, columns / 128), input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 8; ++row) {
        if (lane == 0 && first_row + row < rows) output[first_row + row] = sums[row];
    }
}

// Gate and up projections of one input with bo_swiglu folded in: each SIMD
// group reduces four gate rows and the same four up rows, sharing activation
// coefficients across the eight as the plain matvec does, then writes
// silu(gate) * up, so neither projection is stored. With the half-prefix
// decoder four rows of each measured 163.5 us against 170.8 us for two (the
// floor decoder measured 264.5 against 262.1).
kernel void bonsai_ptq1_matvec_swiglu(
    device const BonsaiPtq1Block *gate [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    device const BonsaiPtq1Block *up [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint first_row = group * 4, blocks = columns / 128;
    const BonsaiPtq1Rows<4> gate_rows = bonsai_ptq1_rows<4>(gate, first_row, rows, blocks);
    const BonsaiPtq1Rows<4> up_rows = bonsai_ptq1_rows<4>(up, first_row, rows, blocks);
    BonsaiPtq1Rows<8> both;
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 4; ++row) {
        both.blocks[row] = gate_rows.blocks[row];
        both.blocks[4 + row] = up_rows.blocks[row];
    }
    float sums[8];
    bonsai_ptq1_matvec_reuse_impl<8>(both, input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 4; ++row) {
        if (lane == 0 && first_row + row < rows) {
            output[first_row + row] = bonsai_silu(sums[row]) * sums[4 + row];
        }
    }
}

// Up to three projections of one input in one dispatch. Each SIMD group still
// owns eight rows of a single matrix; the segment only selects base pointers.
static inline void bonsai_ptq1_matvec_concat_impl(
    device const BonsaiPtq1Block *weights0, device const float *input, device float *output0,
    constant uint *segment_rows, uint columns,
    device const BonsaiPtq1Block *weights1, device float *output1,
    device const BonsaiPtq1Block *weights2, device float *output2,
    uint group, uint lane
) {
    const uint groups0 = (segment_rows[0] + 7) / 8, groups1 = (segment_rows[1] + 7) / 8;
    device const BonsaiPtq1Block *weights = weights0;
    device float *output = output0;
    uint rows = segment_rows[0];
    if (group >= groups0 + groups1) {
        group -= groups0 + groups1;
        weights = weights2;
        output = output2;
        rows = segment_rows[2];
    } else if (group >= groups0) {
        group -= groups0;
        weights = weights1;
        output = output1;
        rows = segment_rows[1];
    }
    const uint first_row = group * 8;
    float sums[8];
    bonsai_ptq1_matvec_reuse_impl<8>(
        bonsai_ptq1_rows<8>(weights, first_row, rows, columns / 128), input, columns, lane, sums);
    #pragma clang loop unroll(full)
    for (uint row = 0; row < 8; ++row) {
        if (lane == 0 && first_row + row < rows) output[first_row + row] = sums[row];
    }
}

kernel void bonsai_ptq1_matvec_concat(
    device const BonsaiPtq1Block *weights0 [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output0 [[buffer(2)]],
    constant uint *segment_rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    device const BonsaiPtq1Block *weights1 [[buffer(5)]],
    device float *output1 [[buffer(6)]],
    device const BonsaiPtq1Block *weights2 [[buffer(7)]],
    device float *output2 [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    bonsai_ptq1_matvec_concat_impl(weights0, input, output0, segment_rows, columns,
        weights1, output1, weights2, output2, group, lane);
}

// The concatenated matvec preceded by two BF16 matvecs of a second input, as
// bo_bf16_mv computes them, one SIMD group per BF16 row. Their groups come
// first so their latency overlaps the packed rows instead of trailing them.
kernel void bonsai_ptq1_matvec_concat_bf16(
    device const BonsaiPtq1Block *weights0 [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output0 [[buffer(2)]],
    constant uint *segment_rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    device const BonsaiPtq1Block *weights1 [[buffer(5)]],
    device float *output1 [[buffer(6)]],
    device const BonsaiPtq1Block *weights2 [[buffer(7)]],
    device float *output2 [[buffer(8)]],
    device const ushort *bf16_weights0 [[buffer(9)]],
    device float *bf16_output0 [[buffer(10)]],
    device const ushort *bf16_weights1 [[buffer(11)]],
    device float *bf16_output1 [[buffer(12)]],
    device const float *bf16_input [[buffer(13)]],
    constant uint *bf16_shape [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint bf16_rows = bf16_shape[0], bf16_columns = bf16_shape[1];
    if (group < bf16_rows) {
        bonsai_bf16_mv_impl(bf16_weights0, bf16_input, bf16_output0, bf16_rows, bf16_columns,
            group, lane);
    } else if (group < 2 * bf16_rows) {
        bonsai_bf16_mv_impl(bf16_weights1, bf16_input, bf16_output1, bf16_rows, bf16_columns,
            group - bf16_rows, lane);
    } else {
        bonsai_ptq1_matvec_concat_impl(weights0, input, output0, segment_rows, columns,
            weights1, output1, weights2, output2, group - 2 * bf16_rows, lane);
    }
}

// bo_rms followed by the forward rotation in one dispatch. Every 1024-wide
// threadgroup repeats its row's full 128-thread square sum, in bo_rms's order,
// then normalizes its own block exactly as bo_rms does, stores it (consumers
// such as the BF16 projections read the unrotated row) and rotates it.
kernel void bonsai_rms_fwht_forward(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *normalized [[buffer(2)]],
    device const float *signs [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant uint &blocks_per_row [[buffer(5)]],
    constant float &epsilon [[buffer(6)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[1024];
    const uint dimension = blocks_per_row * 1024;
    const ulong row = block / blocks_per_row;
    float square_sum = 0.0f;
    for (uint i = tid; i < dimension; i += 128) {
        const float value = input[row * dimension + i];
        square_sum = fma(value, value, square_sum);
    }
    const float sum = bonsai_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    const ulong base = ulong(block) * 1024;
    const uint column = (block % blocks_per_row) * 1024;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint local = i * 128 + tid;
        values[i] = input[base + local] * inverse * weights[column + local];
        normalized[base + local] = values[i];
    }
    bonsai_fwht_values<false>(values, signs, output, blocks_per_row, block, tid, shared);
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
