// GPU side of the Neural Engine prefill FFN (local-engine's ane_split). The
// ANE multiplies int8 weights by int8 activations, reading every code as
// code / 128 so fp16 sums cannot overflow; every scale therefore carries a
// factor 128. Adapted from Kapitsa's src/kernels/ane_ffn.metal (nace-ai/kapitsa
// at a6a7bca, Apache-2.0; see THIRD_PARTY_NOTICES.md): the pack, stage and
// join steps and their scale layout. Bonsai's activations and weights are
// already in the checkpoint's Hadamard-rotated basis, so nothing here rotates,
// and its ternary weights convert to per-row int8 by new block scales alone.
#include <metal_stdlib>
using namespace metal;

constant constexpr float kAnePeak = 127.0f;
constant constexpr float kAneUnit = 128.0f;
// Least token peak: fp16's least normal value.
constant constexpr float kAneFloor = 0x1p-14f;

// fp16 infinity or NaN by its exponent bits.
static inline bool bonsai_ane_bad(half value) {
    return (as_type<ushort>(value) & 0x7c00) == 0x7c00;
}

struct BonsaiAneScaleParams {
    uint tokens;
    uint width;
};

// tx[t] = half(max |x[t][:]| / 127 * 128), the ANE's per-token input scale;
// rows t in [tokens, grid) (padding of the ANE function) get 0.
// grid: function rows x 256 threads.
kernel void bonsai_ane_token_scales(
    device const float *x [[buffer(0)]],
    device half *tx [[buffer(1)]],
    constant BonsaiAneScaleParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float peaks[8];
    if (row >= p.tokens) {
        if (tid == 0) {
            tx[row] = half(0.0h);
        }
        return;
    }
    device const float4 *values = (device const float4 *)(x + ulong(row) * p.width);
    float peak = 0.0f;
    for (uint i = tid; i < p.width / 4; i += 256) {
        const float4 v = fabs(values[i]);
        peak = max(peak, max(max(v.x, v.y), max(v.z, v.w)));
    }
    peak = simd_max(peak);
    if (lane == 0) {
        peaks[group] = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            m = max(m, peaks[g]);
        }
        tx[row] = half(max(m, kAneFloor) / kAnePeak * kAneUnit);
    }
}

struct BonsaiAnePackParams {
    uint tokens;
    uint width;
    // Channels per output surface.
    uint split;
    // Bytes per channel row of the output surfaces.
    uint stride;
};

// x[t][c] -> int8 codes rint(x * 128 / tx[t]) channel-major: out0[c][t] for
// c < split, out1[c - split][t] above (zeros for padding tokens). The codes
// use the rounded fp16 scale the ANE multiplies by.
// grid: (function rows / 32, width / 32) x (32, 8) threads.
kernel void bonsai_ane_pack(
    device const float *x [[buffer(0)]],
    device const half *tx [[buffer(1)]],
    device char *out0 [[buffer(2)]],
    device char *out1 [[buffer(3)]],
    constant BonsaiAnePackParams &p [[buffer(4)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    const uint t0 = tile.x * 32;
    const uint c0 = tile.y * 32;
    for (uint j = pos.y; j < 32; j += 8) {
        const uint t = t0 + j;
        float value = 0.0f;
        if (t < p.tokens) {
            value = x[ulong(t) * p.width + c0 + pos.x] * (kAneUnit / float(tx[t]));
        }
        staged[j][pos.x] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device char *out = c0 < p.split ? out0 : out1;
    const uint channel = c0 < p.split ? c0 : c0 - p.split;
    for (uint j = pos.y; j < 32; j += 8) {
        out[ulong(channel + j) * p.stride + t0 + pos.x] =
            char(clamp(rint(staged[pos.x][j]), -kAnePeak, kAnePeak));
    }
}

struct BonsaiAneStageParams {
    // PTQ1 blocks per weight row.
    uint blocks;
    // The output segment's first block and block count.
    uint first;
    uint count;
    // Bytes per output row.
    uint stride;
    // Halves between rows of the scale surface; 0 writes no scales.
    uint scale_stride;
};

static inline float bonsai_ane_block_scale(device const uchar *block) {
    return float(as_type<half>(ushort(ushort(block[26]) | (ushort(block[27]) << 8))));
}

// Trit (-1, 0, +1) of `element` in a PTQ1_0 block: the layout of
// bonsai_projection.h's bonsai_ptq1_trit.
static inline int bonsai_ane_trit(device const uchar *block, uint element) {
    const uint powers[5] = {1, 3, 9, 27, 81};
    uint index;
    uint power;
    if (element < 80) {
        index = element & 15;
        power = element >> 4;
    } else if (element < 120) {
        index = 16 + ((element - 80) & 7);
        power = (element - 80) >> 3;
    } else {
        index = 24 + ((element - 120) & 1);
        power = (element - 120) >> 1;
    }
    return int((((uint(block[index]) * powers[power]) & 255) * 3) >> 8) - 1;
}

// Ternary rows as per-row int8: block b's trits times q_b = rint(127 s_b /
// max |s|), under the row scale max |s| / 127 (times 128) in the scale
// surface. Writes blocks [first, first + count) of each row to `out`.
// grid: rows / 8 x 256 threads, one SIMD group per row.
kernel void bonsai_ane_stage(
    device const uchar *weights [[buffer(0)]],
    device char *out [[buffer(1)]],
    device half *scales [[buffer(2)]],
    constant BonsaiAneStageParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint row = group * 8 + simdgroup;
    device const uchar *packed = weights + ulong(row) * p.blocks * 28;
    float peak = 0.0f;
    for (uint b = lane; b < p.blocks; b += 32) {
        peak = max(peak, fabs(bonsai_ane_block_scale(packed + ulong(b) * 28)));
    }
    peak = simd_max(peak);
    const float inverse = peak > 0.0f ? kAnePeak / peak : 0.0f;
    if (p.scale_stride != 0 && lane == 0) {
        scales[ulong(row) * p.scale_stride] = half(peak / kAnePeak * kAneUnit);
    }
    device char *row_out = out + ulong(row) * p.stride;
    for (uint b = 0; b < p.count; ++b) {
        device const uchar *block = packed + ulong(p.first + b) * 28;
        const int q = int(rint(bonsai_ane_block_scale(block) * inverse));
        char4 codes;
        for (uint e = 0; e < 4; ++e) {
            codes[e] = char(bonsai_ane_trit(block, lane * 4 + e) * q);
        }
        *(device char4 *)(row_out + b * 128 + lane * 4) = codes;
    }
}

struct BonsaiAneJoinParams {
    uint tokens;
    uint width;
    // Halves per row of the partial output.
    uint ystride;
};

// out[t][c] = y[c][t] * y[width][t] * tx[t] in F32 for t < tokens: the ANE's
// down projection before its token scales, its last row the intermediate's
// per-token scale, tx the input's. A non-finite value or scale in a row of
// the block sets *status. grid: (ceil(tokens / 32), width / 32) x (32, 8).
kernel void bonsai_ane_join(
    device const half *y [[buffer(0)]],
    device const half *tx [[buffer(1)]],
    device float *out [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant BonsaiAneJoinParams &p [[buffer(4)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 pos [[thread_position_in_threadgroup]]
) {
    threadgroup float staged[32][33];
    threadgroup float token_scales[32];
    const uint t0 = tile.x * 32;
    const uint c0 = tile.y * 32;
    const uint t = t0 + pos.x;
    const bool inside = t < p.tokens;
    bool bad = false;
    for (uint j = pos.y; j < 32; j += 8) {
        const half value = y[ulong(c0 + j) * p.ystride + t];
        bad |= inside && bonsai_ane_bad(value);
        staged[j][pos.x] = float(value);
    }
    if (pos.y == 0) {
        const half intermediate = y[ulong(p.width) * p.ystride + t];
        const half input = tx[t];
        bad |= inside && (bonsai_ane_bad(intermediate) || bonsai_ane_bad(input));
        token_scales[pos.x] = float(intermediate) * float(input);
    }
    if (bad) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = pos.y; j < 32; j += 8) {
        const uint row = t0 + j;
        if (row < p.tokens) {
            out[ulong(row) * p.width + c0 + pos.x] = staged[pos.x][j] * token_scales[j];
        }
    }
}
