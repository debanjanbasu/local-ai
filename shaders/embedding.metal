#include <metal_stdlib>
using namespace metal;

/// Embedding lookup: output[token_idx] = table[token_id]
/// table shape: [vocab_size, hidden_dim] (FP16)
/// token_ids: [num_tokens] (uint32)
/// output: [num_tokens, hidden_dim] (FP16)
///
/// Dispatch: grid = (hidden_dim, num_tokens, 1)
kernel void embedding_lookup(
    device const half   *table     [[buffer(0)]],
    device const uint   *token_ids [[buffer(1)]],
    device half         *output    [[buffer(2)]],
    constant uint       &hidden_dim [[buffer(3)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint dim_idx = gid.x;
    uint token_idx = gid.y;

    if (dim_idx >= hidden_dim) return;

    uint token_id = token_ids[token_idx];
    output[token_idx * hidden_dim + dim_idx] = table[token_id * hidden_dim + dim_idx];
}
