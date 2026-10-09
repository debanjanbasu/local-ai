// Entry points for the small-batch PTQ1_0 projection; the shared
// implementation lives in bonsai_small_batch.h.
#include <metal_stdlib>
#include "bonsai_small_batch.h"

using namespace metal;

#define BONSAI_SMALL_BATCH_KERNEL(tokens, impl) \
kernel void bonsai_ptq1_small_batch_##tokens( \
    device const BonsaiPtq1Block *weights [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant uint &rows [[buffer(3)]], \
    constant uint &columns [[buffer(4)]], \
    constant uint &first_token [[buffer(5)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    impl<tokens, 4>( \
        BonsaiNativePacked{weights}, input, output, rows, columns, first_token, group, lane); \
}

// Measured on M4 Pro (17408x5120): 0.19 / 0.24 / 0.28 ms for 2 / 3 / 4 tokens
// and 0.34 ms at five, where the wide kernel below is no slower; with the
// half-prefix trit decode 0.155 / 0.198 / 0.235 ms against the wide 0.28 ms,
// and with balanced prefixes and the two-token loop order 0.141 / 0.188 / 0.228.
BONSAI_SMALL_BATCH_KERNEL(2, bonsai_ptq1_small_batch_pair_impl)
BONSAI_SMALL_BATCH_KERNEL(3, bonsai_ptq1_small_batch_impl)
BONSAI_SMALL_BATCH_KERNEL(4, bonsai_ptq1_small_batch_impl)

// 0.233-0.237 ms on 17408x5120 and 0.236-0.281 ms on 5120x17408 for 5..8
// tokens (0.28-0.30 ms with two activation loads per tile, 0.34 ms with floor
// decoding); dispatch ceil(rows / 8) threadgroups of 128 threads.
kernel void bonsai_ptq1_small_batch_wide(
    device const BonsaiPtq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &first_token [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float shared[4 * 64];
    bonsai_ptq1_small_batch_wide_impl(
        BonsaiNativePacked{weights}, input, output, rows, columns, first_token, tokens,
        group, simd, lane, shared);
}

// 64 tokens by 64 output rows per threadgroup; see
// bonsai_ptq1_large_batch_impl. Dispatch (ceil(rows / 64), ceil(tokens / 64))
// groups of 128 threads.
kernel void bonsai_ptq1_large_batch(
    device const BonsaiPtq1Block *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup half decoded[128 * bonsai_large_tile];
    bonsai_ptq1_large_batch_impl(
        BonsaiNativePacked{weights}, input, output, rows, columns, tokens,
        group, tid, simd, lane, decoded);
}
