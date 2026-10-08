// Entry points for the small-batch PTQ1_0 projection; the shared
// implementation lives in bonsai_small_batch.h.
#include <metal_stdlib>
#include "bonsai_small_batch.h"

using namespace metal;

#define BONSAI_SMALL_BATCH_KERNEL(tokens) \
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
    bonsai_ptq1_small_batch_impl<tokens, 4>( \
        BonsaiNativePacked{weights}, input, output, rows, columns, first_token, group, lane); \
}

// Measured on M4 Pro (17408x5120): 0.19 / 0.24 / 0.28 ms for 2 / 3 / 4 tokens
// and 0.34 ms at five, where the wide kernel below is no slower; with the
// half-prefix trit decode 0.155 / 0.198 / 0.235 ms against the wide 0.28 ms.
BONSAI_SMALL_BATCH_KERNEL(2)
BONSAI_SMALL_BATCH_KERNEL(3)
BONSAI_SMALL_BATCH_KERNEL(4)

// 0.28-0.30 ms (0.34 ms with floor decoding) for any 1..8 tokens on both FFN
// shapes; dispatch ceil(rows / 8) threadgroups of 128 threads.
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
