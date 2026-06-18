#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 32

/// Tiled matrix multiply: [M, K] × [K, N] → [M, N], all FP16.
/// Used for batch projections during prefill (M = seq_len tokens at once).
/// Accumulates in float32, stores as half.
///
/// Dispatch: threadgroups = (ceil(N/32), ceil(M/32)), threads_per_tg = (32, 32)
kernel void matmul_f16(
    device const half *A  [[buffer(0)]],  // [M, K]
    device const half *B  [[buffer(1)]],  // [K, N]
    device half *C        [[buffer(2)]],  // [M, N]
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Global row/col this thread is responsible for
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;

    // Shared memory tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load A tile
        uint a_col = t * TILE_SIZE + tid.x;
        if (row < M && a_col < K) {
            As[tid.y][tid.x] = float(A[row * K + a_col]);
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load B tile
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < K && col < N) {
            Bs[tid.y][tid.x] = float(B[b_row * N + col]);
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}

/// Tiled matrix multiply with a **transposed** B: [M, K] × [N, K]ᵀ → [M, N].
///
/// B is row-major `[N, K]` — exactly the `[out, in]` layout every weight
/// tensor in this engine uses — so a batch of `M` token rows can be projected
/// through a weight matrix while streaming the weights once (instead of `M`
/// matvec passes). Accumulates in float32, stores as half.
///
/// Dispatch: threadgroups = (ceil(N/32), ceil(M/32)), threads_per_tg = (32, 32)
kernel void matmul_f16_nt(
    device const half *A  [[buffer(0)]],  // [M, K]
    device const half *B  [[buffer(1)]],  // [N, K] row-major (transposed use)
    device half *C        [[buffer(2)]],  // [M, N]
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y * TILE_SIZE + tid.y; // m
    uint col = gid.x * TILE_SIZE + tid.x; // n

    threadgroup float As[TILE_SIZE][TILE_SIZE]; // [m_local][k_local]
    threadgroup float Bs[TILE_SIZE][TILE_SIZE]; // [n_local][k_local]

    float acc = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;

        // B tile: rows indexed by n (this block's columns), cols by k.
        uint b_n = gid.x * TILE_SIZE + tid.y;
        uint b_k = t * TILE_SIZE + tid.x;
        Bs[tid.y][tid.x] = (b_n < N && b_k < K) ? float(B[b_n * K + b_k]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += As[tid.y][k] * Bs[tid.x][k];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}

/// Tiled clipped linear with transposed weights:
/// `Y = clamp(clamp(X, input_min, input_max) @ W^T + bias, output_min, output_max)`.
///
/// Input is row-major `[M, K]`, weight is row-major `[N, K]`. This is the
/// tiled equivalent of `clipped_linear` for Gemma 4 vision projector blocks.
kernel void clipped_linear_f16_nt(
    device const half *A        [[buffer(0)]],  // [M, K]
    device const half *B        [[buffer(1)]],  // [N, K]
    device const half *bias     [[buffer(2)]],
    device half       *C        [[buffer(3)]],  // [M, N]
    constant uint     &M        [[buffer(4)]],
    constant uint     &K        [[buffer(5)]],
    constant uint     &N        [[buffer(6)]],
    constant float    &input_min [[buffer(7)]],
    constant float    &input_max [[buffer(8)]],
    constant float    &output_min [[buffer(9)]],
    constant float    &output_max [[buffer(10)]],
    constant uint     &has_bias [[buffer(11)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    uint row = gid.y * TILE_SIZE + tid.y;
    uint col = gid.x * TILE_SIZE + tid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE_SIZE + tid.x;
        float a = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;
        As[tid.y][tid.x] = clamp(a, input_min, input_max);

        uint b_n = gid.x * TILE_SIZE + tid.y;
        uint b_k = t * TILE_SIZE + tid.x;
        Bs[tid.y][tid.x] = (b_n < N && b_k < K) ? float(B[b_n * K + b_k]) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += As[tid.y][k] * Bs[tid.x][k];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        if (has_bias != 0) {
            acc += float(bias[col]);
        }
        C[row * N + col] = half(clamp(acc, output_min, output_max));
    }
}
