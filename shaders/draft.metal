// GPU-resident greedy drafting: embedding gather from a device-side token id and
// an exact top-2 over draft logits, so chained draft steps never return to the
// CPU. PTQ1_0 decoding mirrors `decode_ptq1_row`; the inverse rotation reuses
// the target's own signed FWHT, so both outputs are bit-identical to the CPU
// decode followed by `bonsai_fwht_inverse`.
#include <metal_stdlib>
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include "bonsai_projection.h"
using namespace metal;

// Element `element` (0..128) of one PTQ1_0 block, exactly as the CPU codec.
static inline float draft_ptq1_element(device const BonsaiPtq1Block &block, uint element) {
    uint byte, power;
    if (element < 80) {
        byte = element % 16;
        power = element / 16;
    } else if (element < 120) {
        byte = 16 + (element - 80) % 8;
        power = (element - 80) / 8;
    } else {
        byte = 24 + (element - 120) % 2;
        power = (element - 120) / 2;
    }
    const uchar code = byte < 24 ? block.qs[byte] : block.qh[byte - 24];
    const uchar pow3[5] = {1, 3, 9, 27, 81};
    const uchar q = uchar(code * pow3[power]);
    const float trit = float((ushort(q) * 3) >> 8) - 1.0f;
    return trit * float(block.d);
}

// One threadgroup of 128 per 1024-wide block of the row: decode row
// `tokens[token_index]` of the packed table and inverse-rotate it, as the CPU
// decode plus `bonsai_fwht_inverse` would. A token outside the table yields zeros.
kernel void draft_embed_inverse(
    device const BonsaiPtq1Block *table [[buffer(0)]],
    device const uint *tokens [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &token_index [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &blocks_per_row [[buffer(6)]],
    uint block [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[1024];
    const uint token = tokens[token_index];
    const bool valid = token < rows;
    // Eight PTQ1 blocks of 128 per rotation block of 1024.
    const ulong first = ulong(valid ? token : 0) * blocks_per_row * 8 + ulong(block) * 8;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        values[i] = valid ? draft_ptq1_element(table[first + i], tid) : 0.0f;
    }
    bonsai_fwht_values<true>(values, signs, output, blocks_per_row, block, tid, shared);
}

struct DraftTopTwo {
    uint best_id;
    float best_logit;
    uint second_id;
    float second_logit;
};

static inline int draft_total_key(float value) {
    int key = as_type<int>(value);
    return key ^ int(uint(key >> 31) >> 1);
}

// `f32::total_cmp` order, ties to the lower token id; `UINT_MAX` is padding.
static inline bool draft_better(float a, uint a_id, float b, uint b_id) {
    if (a_id == UINT_MAX) return false;
    if (b_id == UINT_MAX) return true;
    const int a_key = draft_total_key(a);
    const int b_key = draft_total_key(b);
    return a_key > b_key || (a_key == b_key && a_id < b_id);
}

struct DraftPair {
    uint best_id;
    float best;
    uint second_id;
    float second;
};

static inline DraftPair draft_empty() {
    return DraftPair{UINT_MAX, 0.0f, UINT_MAX, 0.0f};
}

static inline void draft_insert(thread DraftPair &pair, float value, uint id) {
    if (draft_better(value, id, pair.best, pair.best_id)) {
        pair.second = pair.best;
        pair.second_id = pair.best_id;
        pair.best = value;
        pair.best_id = id;
    } else if (draft_better(value, id, pair.second, pair.second_id)) {
        pair.second = value;
        pair.second_id = id;
    }
}

// Merge two disjoint sorted pairs into the top two of their union.
static inline DraftPair draft_merge(DraftPair a, DraftPair b) {
    DraftPair out;
    if (draft_better(a.best, a.best_id, b.best, b.best_id)) {
        out.best = a.best;
        out.best_id = a.best_id;
        const bool keep = draft_better(a.second, a.second_id, b.best, b.best_id);
        out.second = keep ? a.second : b.best;
        out.second_id = keep ? a.second_id : b.best_id;
    } else {
        out.best = b.best;
        out.best_id = b.best_id;
        const bool keep = draft_better(a.best, a.best_id, b.second, b.second_id);
        out.second = keep ? a.best : b.second;
        out.second_id = keep ? a.best_id : b.second_id;
    }
    return out;
}

static inline DraftPair draft_simd_reduce(DraftPair pair) {
    for (ushort distance = 16; distance > 0; distance >>= 1) {
        DraftPair other;
        other.best_id = simd_shuffle_xor(pair.best_id, distance);
        other.best = simd_shuffle_xor(pair.best, distance);
        other.second_id = simd_shuffle_xor(pair.second_id, distance);
        other.second = simd_shuffle_xor(pair.second, distance);
        // Lanes hold disjoint candidates, so the merge is symmetric.
        pair = draft_merge(pair, other);
    }
    return pair;
}

// 256 threads per group, all groups together reduce the threadgroup's slice.
static inline DraftPair draft_group_reduce(
    DraftPair pair,
    threadgroup DraftPair *shared,
    uint tid,
    uint lane,
    uint simd
) {
    pair = draft_simd_reduce(pair);
    if (lane == 0) shared[simd] = pair;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pair = lane < 8 ? shared[lane] : draft_empty();
    return draft_simd_reduce(pair);
}

constant uint DRAFT_GROUP_ELEMENTS = 1024;

// Stage 1: each group of 256 reduces 1024 consecutive logits to a top-two.
kernel void draft_top2_partial(
    device const float *logits [[buffer(0)]],
    device DraftPair *partial [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    DraftPair pair = draft_empty();
    const uint base = group * DRAFT_GROUP_ELEMENTS;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < DRAFT_GROUP_ELEMENTS / 256; ++i) {
        const uint id = base + i * 256 + tid;
        if (id < vocab) draft_insert(pair, logits[id], id);
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    if (tid == 0) partial[group] = pair;
}

// Stage 2: one group of 256 merges every partial, stores the step's top two
// and writes the winner into `tokens[token_index]` for the next chained step.
kernel void draft_top2_final(
    device const DraftPair *partial [[buffer(0)]],
    device uint *tokens [[buffer(1)]],
    device DraftTopTwo *results [[buffer(2)]],
    constant uint &groups [[buffer(3)]],
    constant uint &token_index [[buffer(4)]],
    constant uint &result_index [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    DraftPair pair = draft_empty();
    for (uint index = tid; index < groups; index += 256) {
        pair = draft_merge(pair, partial[index]);
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    if (tid == 0) {
        tokens[token_index] = pair.best_id;
        results[result_index] = DraftTopTwo{pair.best_id, pair.best, pair.second_id, pair.second};
    }
}


// Greedy selection over consecutive logit rows (a verify block, or the one
// row of a decode step): the same `f32::total_cmp` order and ties-to-lower-id
// as `draft_top2_*` and the GPU top-k, plus whether any logit of the row is
// non-finite, which the host would otherwise scan the whole row for.
struct GreedyRowResult {
    uint best_id;
    float best_logit;
    uint nonfinite;
    uint padding;
};

static inline uint greedy_group_or(uint value, threadgroup uint *shared, uint lane, uint simd) {
    value = simd_or(value);
    if (lane == 0) shared[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_or(lane < 8 ? shared[lane] : 0u);
}

// Stage 1: group (x, y) reduces 1024 logits of row y to a top-two and a flag.
kernel void greedy_rows_partial(
    device const float *logits [[buffer(0)]],
    device DraftPair *partial [[buffer(1)]],
    device uint *flags [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    threadgroup uint shared_flags[8];
    device const float *row = logits + ulong(group.y) * vocab;
    DraftPair pair = draft_empty();
    uint nonfinite = 0;
    const uint base = group.x * DRAFT_GROUP_ELEMENTS;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < DRAFT_GROUP_ELEMENTS / 256; ++i) {
        const uint id = base + i * 256 + tid;
        if (id < vocab) {
            const float value = row[id];
            // Exponent bits, not isfinite(): fast math may assume finite values.
            nonfinite |= uint((as_type<uint>(value) & 0x7f800000u) == 0x7f800000u);
            draft_insert(pair, value, id);
        }
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    nonfinite = greedy_group_or(nonfinite, shared_flags, lane, simd);
    if (tid == 0) {
        const uint slot = group.y * groups.x + group.x;
        partial[slot] = pair;
        flags[slot] = nonfinite;
    }
}

// Stage 2: group y merges row y's partials into `results[y]`.
kernel void greedy_rows_final(
    device const DraftPair *partial [[buffer(0)]],
    device const uint *flags [[buffer(1)]],
    device GreedyRowResult *results [[buffer(2)]],
    constant uint &groups [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    threadgroup uint shared_flags[8];
    DraftPair pair = draft_empty();
    uint nonfinite = 0;
    for (uint index = tid; index < groups; index += 256) {
        pair = draft_merge(pair, partial[row * groups + index]);
        nonfinite |= flags[row * groups + index];
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    nonfinite = greedy_group_or(nonfinite, shared_flags, lane, simd);
    if (tid == 0) results[row] = GreedyRowResult{pair.best_id, pair.best, nonfinite, 0};
}

// Several sequences drafting together: one row per sequence per draft step.
// Row y reads its token from `tokens[slots[y]]` and writes its selection back
// into its own slots, so each sequence's chain stays in its own token run.
// Every row computes exactly what the one-row kernels above compute for it.

// Grid (blocks_per_row, rows), 128 threads: `draft_embed_inverse` per row,
// row y landing at row y of `output`.
kernel void draft_embed_inverse_rows(
    device const BonsaiPtq1Block *table [[buffer(0)]],
    device const uint *tokens [[buffer(1)]],
    device const float *signs [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint *slots [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &blocks_per_row [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float shared[1024];
    const uint token = tokens[slots[group.y]];
    const bool valid = token < rows;
    const ulong first = ulong(valid ? token : 0) * blocks_per_row * 8 + ulong(group.x) * 8;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        values[i] = valid ? draft_ptq1_element(table[first + i], tid) : 0.0f;
    }
    bonsai_fwht_values<true>(
        values, signs, output, blocks_per_row, group.y * blocks_per_row + group.x, tid, shared);
}

// Stage 1 of a per-row top two: group (x, y) reduces 1024 logits of row y.
kernel void draft_top2_rows_partial(
    device const float *logits [[buffer(0)]],
    device DraftPair *partial [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    device const float *row = logits + ulong(group.y) * vocab;
    DraftPair pair = draft_empty();
    const uint base = group.x * DRAFT_GROUP_ELEMENTS;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < DRAFT_GROUP_ELEMENTS / 256; ++i) {
        const uint id = base + i * 256 + tid;
        if (id < vocab) draft_insert(pair, row[id], id);
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    if (tid == 0) partial[group.y * groups.x + group.x] = pair;
}

// Stage 2: group y merges row y's partials, writes the winner to
// `tokens[token_slots[y]]` and the pair to `results[result_slots[y]]`.
kernel void draft_top2_rows_final(
    device const DraftPair *partial [[buffer(0)]],
    device uint *tokens [[buffer(1)]],
    device DraftTopTwo *results [[buffer(2)]],
    constant uint &groups [[buffer(3)]],
    constant uint *token_slots [[buffer(4)]],
    constant uint *result_slots [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup DraftPair shared[8];
    DraftPair pair = draft_empty();
    for (uint index = tid; index < groups; index += 256) {
        pair = draft_merge(pair, partial[row * groups + index]);
    }
    pair = draft_group_reduce(pair, shared, tid, lane, simd);
    if (tid == 0) {
        tokens[token_slots[row]] = pair.best_id;
        results[result_slots[row]] =
            DraftTopTwo{pair.best_id, pair.best, pair.second_id, pair.second};
    }
}
