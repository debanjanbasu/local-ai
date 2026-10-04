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

// Measured on M4 Pro (17408x5120): 0.19 / 0.26 / 0.35 ms for 2 / 3 / 4 tokens,
// then 0.64 ms at five; larger blocks are cheaper as balanced 3-4 token chunks.
BONSAI_SMALL_BATCH_KERNEL(2)
BONSAI_SMALL_BATCH_KERNEL(3)
BONSAI_SMALL_BATCH_KERNEL(4)
