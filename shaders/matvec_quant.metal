#include <metal_stdlib>
using namespace metal;

/// Quantized-weight matvec/matmul kernels.
///
/// Weights stay in their GGUF block format in device memory and are
/// dequantized inside the kernel — weight traffic drops from 16 bits/elem
/// (FP16) to 4.5 (Q4_0) or ~2.06 (TQ2_0), which is the win for the
/// memory-bound decode path.
///
/// Block layouts (little-endian, matching ggml):
///   Q4_0  — 18 B / 32 elems: f16 d, then 16 bytes of nibbles.
///           byte i holds elems i (low nibble) and i+16 (high nibble);
///           value = d * (q - 8).
///   TQ2_0 — 66 B / 256 elems: 64 bytes qs, then f16 d.
///           elem (j>>5)*128 + l*32 + (j&31) comes from bits 2l..2l+1 of
///           qs[j]; value = d * (q - 1).

/// Dequantize a single element of a Q4_0 row. `row` points at the row start.
inline float q4_0_get(device const uchar *row, uint k) {
    device const uchar *bp = row + (k >> 5) * 18;
    float d = float(*(device const half *)bp);
    uchar q = bp[2 + (k & 15)];
    uint nib = ((k & 31) < 16) ? (q & 0x0F) : (q >> 4);
    return d * (float(nib) - 8.0f);
}

/// Dequantize a single element of a TQ2_0 row. `row` points at the row start.
inline float tq2_0_get(device const uchar *row, uint k) {
    device const uchar *bp = row + (k >> 8) * 66;
    uint r = k & 255;
    uint j_iter = r >> 7;
    uint l = (r >> 5) & 3;
    uint m = r & 31;
    float d = float(*(device const half *)(bp + 64));
    uchar q = bp[j_iter * 32 + m];
    return d * (float((q >> (2 * l)) & 3) - 1.0f);
}

/// Output rows handled per threadgroup. Each row is reduced by exactly one
/// SIMD-group (32 lanes), so a threadgroup is `ROWS_PER_TG * 32` threads.
/// Must match `ROWS_PER_TG` in the Rust dispatch (`kernels.rs`).
constant uint ROWS_PER_TG = 4;

/// Q4_0 output rows reduced per SIMD-group. With `ROWS_PER_TG` SIMD-groups per
/// threadgroup this gives `ROWS_PER_TG * Q4_NR0` rows per threadgroup. Must
/// match the `rows_per_sg` passed from `kernels.rs` for the Q4_0 matvec.
constant uint Q4_NR0 = 4;

/// `Q4_NR0` output rows per SIMD-group, ported from llama.cpp's
/// `mul_vec_q_n_f32_impl` / `block_q_n_dot_y` for `block_q4_0`. Three wins over
/// the byte-at-a-time scalar version:
///   1. weights load as `ushort` (2 bytes / instruction), not single `uchar`s;
///   2. the activation `yl` cache is loaded once and reused across all `Q4_NR0`
///      rows (decode is weight-bandwidth bound, but this also halves vector
///      traffic and lets the unpack run from registers);
///   3. no per-element shift/subtract — activations are pre-scaled by
///      `1, 1/16, 1/256, 1/4096` so the raw nibble masks (`0x000F`, `0x00F0`,
///      `0x0F00`, `0xF000`) multiply directly, and the `-8` zero-point is
///      applied once as `d * (sumy * -8 + acc)`.
///
/// Each block (32 elems / 18 B) is split across two lanes (`il = 0` / `8`); the
/// 32 lanes therefore cover 16 blocks per step and stride by 16 blocks.
template <typename OutT>
inline void matvec_q4_0_sg(
    device const uchar *matrix,
    device const half *vector,
    device OutT *output,
    uint rows, uint cols,
    uint tg, uint sgitg, uint lane
) {
    const uint nb = cols >> 5;          // 32-elem blocks per row
    const uint base_row = (tg * ROWS_PER_TG + sgitg) * Q4_NR0;

    // Row base pointers (row stride = nb * 18 bytes). Rows past the end are
    // clamped to the last valid row so the inner load stays in-bounds; their
    // results are simply not written back.
    device const uchar *ax[Q4_NR0];
    for (uint r = 0; r < Q4_NR0; r++) {
        uint row = min(base_row + r, rows - 1);
        ax[r] = matrix + (ulong)row * nb * 18;
    }

    const ushort ix = lane >> 1;        // block group 0..15
    const ushort il = (lane & 1) * 8;   // half-block: 0 or 8

    float yl[16];
    float sumf[Q4_NR0];
    for (uint r = 0; r < Q4_NR0; r++) sumf[r] = 0.0f;

    device const half *yb = vector + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += 16) {
        float sumy = 0.0f;
        for (short i = 0; i < 8; i += 2) {
            float y0  = float(yb[i + 0]);
            float y1  = float(yb[i + 1]);
            float y16 = float(yb[i + 16]);
            float y17 = float(yb[i + 17]);
            sumy += y0 + y1 + y16 + y17;
            yl[i + 0] = y0;
            yl[i + 1] = y1  / 256.0f;
            yl[i + 8] = y16 / 16.0f;
            yl[i + 9] = y17 / 4096.0f;
        }
        for (uint r = 0; r < Q4_NR0; r++) {
            device const uchar *bp = ax[r] + (ulong)ib * 18;
            float d = float(*(device const half *)bp);
            // Skip the f16 scale (1 ushort), then offset to this lane's half.
            device const ushort *qs = (device const ushort *)bp + 1 + (il >> 1);
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (short i = 0; i < 8; i += 2) {
                ushort q = qs[i >> 1];
                acc0 += yl[i + 0] * float(q & 0x000F);
                acc1 += yl[i + 1] * float(q & 0x0F00);
                acc2 += yl[i + 8] * float(q & 0x00F0);
                acc3 += yl[i + 9] * float(q & 0xF000);
            }
            sumf[r] += d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
        }
        yb += 32 * 16;
    }

    for (uint r = 0; r < Q4_NR0; r++) {
        float tot = simd_sum(sumf[r]);
        if (lane == 0 && base_row + r < rows) output[base_row + r] = OutT(tot);
    }
}

kernel void matvec_q4_0(
    device const uchar *matrix [[buffer(0)]],
    device const half *vector  [[buffer(1)]],
    device half *output        [[buffer(2)]],
    constant uint &rows        [[buffer(3)]],
    constant uint &cols        [[buffer(4)]],
    uint tg      [[threadgroup_position_in_grid]],
    uint sgitg   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    matvec_q4_0_sg<half>(matrix, vector, output, rows, cols, tg, sgitg, lane);
}

/// Float32 output variant: tied-embedding logits over the Q4_0 token table.
kernel void matvec_q4_0_f32out(
    device const uchar *matrix [[buffer(0)]],
    device const half *vector  [[buffer(1)]],
    device float *output       [[buffer(2)]],
    constant uint &rows        [[buffer(3)]],
    constant uint &cols        [[buffer(4)]],
    uint tg      [[threadgroup_position_in_grid]],
    uint sgitg   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    matvec_q4_0_sg<float>(matrix, vector, output, rows, cols, tg, sgitg, lane);
}

/// Output rows handled by each SIMD-group (32 lanes). Each lane keeps
/// `ROWS_PER_SG` accumulators; the per-block activation prep (pre-scale by
/// 1/2^p + the raw sum for the `-1` zero point, see the no-shift dequant below)
/// is computed once per block and reused across all `ROWS_PER_SG` rows. With
/// the no-shift form that prep is the dominant non-weight cost, so reusing it
/// across rows now wins (M4 Pro, `matvec_decode_timing`): 1 row ~100 GB/s,
/// 2 ~112, 3 ~114, 4 ~107. `3` is the sweet spot before register pressure cuts
/// occupancy. Must match `MATVEC_TQ2_ROWS_PER_SG` in `kernels.rs`.
constant uint ROWS_PER_SG = 3;

/// `ROWS_PER_SG` output rows per SIMD-group for TQ2_0. Each lane walks whole
/// 256-elem blocks; the strided activation `half2`s are loaded once per block
/// and reused across the SIMD-group's rows, while only the (tiny) quantized
/// weight bytes + scale are re-read per row. A single `simd_sum` reduces each
/// row at the end.
template <typename OutT>
inline void matvec_tq2_0_sg(
    device const uchar *matrix,
    device const half *vector,
    device OutT *output,
    uint rows, uint cols,
    uint tg, uint sgitg, uint lane
) {
    uint base_row = (tg * ROWS_PER_TG + sgitg) * ROWS_PER_SG;
    if (base_row >= rows) return;

    uint blocks = cols >> 8;

    // Each lane owns the two adjacent qs bytes mj0 = 2·lane and mj0+1, so the
    // 32 lanes cover all 64 qs bytes of a block as one coalesced 64-byte read.
    // mj0 is even ⇒ both bytes share j_iter, and the weight load is an aligned
    // `ushort`. Vector reads use aligned `half2` over the adjacent (mj0, mj0+1)
    // pairs, halving vector load instructions.
    uint mj0 = lane * 2;          // 0, 2, …, 62
    uint j = mj0 >> 5;            // 0 or 1 (both bytes share it)
    uint m0 = mj0 & 31;          // even
    uint voff = (j << 7) + m0;   // vector offset within the block

    // Two independent accumulators per row (even/odd blocks). The FMA chain is
    // loop-carried, so a single accumulator serializes on the activation+weight
    // load latency; splitting odd/even blocks doubles the in-flight loads and
    // lets the scheduler overlap them. Summed at the end. (microbench: lifts
    // achieved bandwidth vs the single-accumulator block loop.)
    float acc[ROWS_PER_SG];
    float acc2[ROWS_PER_SG];
    for (uint r = 0; r < ROWS_PER_SG; r++) { acc[r] = 0.0f; acc2[r] = 0.0f; }

    // No-shift dequant (ported from the Q4_0 path): each 2-bit field at bit
    // position p is isolated by masking with `3<<p`, leaving the value scaled by
    // `2^p`. Pre-scaling the activation by `1/2^p` once per block makes the raw
    // masked bits multiply directly — no per-element shift. The `-1` zero point
    // becomes a single `d*(Σ q·a - Σ a)` per row (sumv). This roughly halves the
    // inner-loop ALU vs the shift/subtract form; TQ2 decode was ALU-bound (it
    // hit only ~93 GB/s vs Q4_0's ~190 GB/s at the same shapes).
    const float s1c = 0.25f, s2c = 0.0625f, s3c = 0.015625f;
    const float s4c = 1.0f / 256.0f, s5c = 1.0f / 1024.0f;
    const float s6c = 1.0f / 4096.0f, s7c = 1.0f / 16384.0f;

    uint b = 0;
    for (; b + 1 < blocks; b += 2) {
        uint vbase0 = (b << 8) + voff;
        uint vbase1 = ((b + 1) << 8) + voff;
        float2 a0 = float2(*(device const half2 *)(vector + vbase0));
        float2 a32 = float2(*(device const half2 *)(vector + vbase0 + 32));
        float2 a64 = float2(*(device const half2 *)(vector + vbase0 + 64));
        float2 a96 = float2(*(device const half2 *)(vector + vbase0 + 96));
        float2 c0 = float2(*(device const half2 *)(vector + vbase1));
        float2 c32 = float2(*(device const half2 *)(vector + vbase1 + 32));
        float2 c64 = float2(*(device const half2 *)(vector + vbase1 + 64));
        float2 c96 = float2(*(device const half2 *)(vector + vbase1 + 96));
        // Activations pre-scaled by 1/2^p (p = 0,2,..,14), and the raw sums for
        // the -1 zero point — both block-local, computed once per block.
        float as0_0 = a0.x, as0_1 = a32.x * s1c, as0_2 = a64.x * s2c, as0_3 = a96.x * s3c;
        float as0_4 = a0.y * s4c, as0_5 = a32.y * s5c, as0_6 = a64.y * s6c, as0_7 = a96.y * s7c;
        float sv0 = a0.x + a32.x + a64.x + a96.x + a0.y + a32.y + a64.y + a96.y;
        float as1_0 = c0.x, as1_1 = c32.x * s1c, as1_2 = c64.x * s2c, as1_3 = c96.x * s3c;
        float as1_4 = c0.y * s4c, as1_5 = c32.y * s5c, as1_6 = c64.y * s6c, as1_7 = c96.y * s7c;
        float sv1 = c0.x + c32.x + c64.x + c96.x + c0.y + c32.y + c64.y + c96.y;
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= rows) break;
            device const uchar *bp0 = matrix + ((ulong)row * blocks + b) * 66;
            device const uchar *bp1 = bp0 + 66;
            float d0 = float(*(device const half *)(bp0 + 64));
            float d1 = float(*(device const half *)(bp1 + 64));
            ushort p0 = *(device const ushort *)(bp0 + mj0);
            ushort p1 = *(device const ushort *)(bp1 + mj0);
            float s0 = float(p0 & 0x0003) * as0_0 + float(p0 & 0x000C) * as0_1
                     + float(p0 & 0x0030) * as0_2 + float(p0 & 0x00C0) * as0_3
                     + float(p0 & 0x0300) * as0_4 + float(p0 & 0x0C00) * as0_5
                     + float(p0 & 0x3000) * as0_6 + float(p0 & 0xC000) * as0_7;
            float s1 = float(p1 & 0x0003) * as1_0 + float(p1 & 0x000C) * as1_1
                     + float(p1 & 0x0030) * as1_2 + float(p1 & 0x00C0) * as1_3
                     + float(p1 & 0x0300) * as1_4 + float(p1 & 0x0C00) * as1_5
                     + float(p1 & 0x3000) * as1_6 + float(p1 & 0xC000) * as1_7;
            acc[r] += d0 * (s0 - sv0);
            acc2[r] += d1 * (s1 - sv1);
        }
    }
    // Tail block when `blocks` is odd.
    for (; b < blocks; b++) {
        uint vbase = (b << 8) + voff;
        float2 v0 = float2(*(device const half2 *)(vector + vbase));
        float2 v32 = float2(*(device const half2 *)(vector + vbase + 32));
        float2 v64 = float2(*(device const half2 *)(vector + vbase + 64));
        float2 v96 = float2(*(device const half2 *)(vector + vbase + 96));
        float as0 = v0.x, as1 = v32.x * s1c, as2 = v64.x * s2c, as3 = v96.x * s3c;
        float as4 = v0.y * s4c, as5 = v32.y * s5c, as6 = v64.y * s6c, as7 = v96.y * s7c;
        float sv = v0.x + v32.x + v64.x + v96.x + v0.y + v32.y + v64.y + v96.y;
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            uint row = base_row + r;
            if (row >= rows) break;
            device const uchar *bp = matrix + ((ulong)row * blocks + b) * 66;
            float d = float(*(device const half *)(bp + 64));
            ushort packed = *(device const ushort *)(bp + mj0);
            float bsum = float(packed & 0x0003) * as0 + float(packed & 0x000C) * as1
                       + float(packed & 0x0030) * as2 + float(packed & 0x00C0) * as3
                       + float(packed & 0x0300) * as4 + float(packed & 0x0C00) * as5
                       + float(packed & 0x3000) * as6 + float(packed & 0xC000) * as7;
            acc[r] += d * (bsum - sv);
        }
    }

    for (uint r = 0; r < ROWS_PER_SG; r++) {
        uint row = base_row + r;
        float sum = simd_sum(acc[r] + acc2[r]);
        if (lane == 0 && row < rows) output[row] = OutT(sum);
    }
}

kernel void matvec_tq2_0(
    device const uchar *matrix [[buffer(0)]],
    device const half *vector  [[buffer(1)]],
    device half *output        [[buffer(2)]],
    constant uint &rows        [[buffer(3)]],
    constant uint &cols        [[buffer(4)]],
    uint tg      [[threadgroup_position_in_grid]],
    uint sgitg   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    matvec_tq2_0_sg<half>(matrix, vector, output, rows, cols, tg, sgitg, lane);
}

/// Float32 output variant: tied-embedding logits over the TQ2_0 token table.
kernel void matvec_tq2_0_f32out(
    device const uchar *matrix [[buffer(0)]],
    device const half *vector  [[buffer(1)]],
    device float *output       [[buffer(2)]],
    constant uint &rows        [[buffer(3)]],
    constant uint &cols        [[buffer(4)]],
    uint tg      [[threadgroup_position_in_grid]],
    uint sgitg   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    matvec_tq2_0_sg<float>(matrix, vector, output, rows, cols, tg, sgitg, lane);
}

/// SoA (struct-of-arrays) TQ2_0 matvec. Identical math to `matvec_tq2_0` but
/// reads from a layout where every block's 64 qs bytes are contiguous and
/// 64-byte aligned (`qs[(row*blocks + b)*64]`) and the fp16 scales live in a
/// separate array (`scales[row*blocks + b]`). The interleaved 66-byte block of
/// the standard layout makes the per-block 64-byte simdgroup load straddle two
/// 64-byte sectors (66 % 64 != 0), roughly doubling memory transactions; the
/// aligned SoA load is one clean sector and the scale is read once per block by
/// lane 0 and broadcast (vs 32 redundant loads).
template <typename OutT>
inline void matvec_tq2_0_soa_sg(
    device const uchar *qs,
    device const half *scales,
    device const half *vector,
    device OutT *output,
    uint rows, uint cols,
    uint tg, uint sgitg, uint lane
) {
    uint base_row = tg * ROWS_PER_TG + sgitg;
    if (base_row >= rows) return;

    uint blocks = cols >> 8;
    uint mj0 = lane * 2;
    uint j = mj0 >> 5;
    uint m0 = mj0 & 31;
    uint voff = (j << 7) + m0;

    ulong q_row = (ulong)base_row * blocks * 64;
    ulong s_row = (ulong)base_row * blocks;

    float acc = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        uint vbase = (b << 8) + voff;
        float2 v0  = float2(*(device const half2 *)(vector + vbase));
        float2 v32 = float2(*(device const half2 *)(vector + vbase + 32));
        float2 v64 = float2(*(device const half2 *)(vector + vbase + 64));
        float2 v96 = float2(*(device const half2 *)(vector + vbase + 96));

        device const uchar *bp = qs + q_row + (ulong)b * 64;
        ushort packed = *(device const ushort *)(bp + mj0);
        half dh = (lane == 0) ? scales[s_row + b] : half(0.0h);
        float d = simd_broadcast(float(dh), 0);

        float q0a = float(packed & 3) - 1.0f;
        float q0b = float((packed >> 2) & 3) - 1.0f;
        float q0c = float((packed >> 4) & 3) - 1.0f;
        float q0d = float((packed >> 6) & 3) - 1.0f;
        float q1a = float((packed >> 8) & 3) - 1.0f;
        float q1b = float((packed >> 10) & 3) - 1.0f;
        float q1c = float((packed >> 12) & 3) - 1.0f;
        float q1d = float((packed >> 14) & 3) - 1.0f;
        float bsum = q0a * v0.x + q1a * v0.y
                   + q0b * v32.x + q1b * v32.y
                   + q0c * v64.x + q1c * v64.y
                   + q0d * v96.x + q1d * v96.y;
        acc += d * bsum;
    }

    float sum = simd_sum(acc);
    if (lane == 0) output[base_row] = OutT(sum);
}

kernel void matvec_tq2_0_soa(
    device const uchar *qs      [[buffer(0)]],
    device const half *scales   [[buffer(1)]],
    device const half *vector   [[buffer(2)]],
    device half *output         [[buffer(3)]],
    constant uint &rows         [[buffer(4)]],
    constant uint &cols         [[buffer(5)]],
    uint tg      [[threadgroup_position_in_grid]],
    uint sgitg   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    matvec_tq2_0_soa_sg<half>(qs, scales, vector, output, rows, cols, tg, sgitg, lane);
}

constant uint QTILE = 32;

/// Tiled `[M, K] × [N, K]ᵀ → [M, N]` with quantized B — same tiling as
/// matmul_f16_nt; the B tile load dequantizes one element per thread so the
/// weights stream once per tile regardless of M.
template <float (*GET)(device const uchar *, uint)>
inline void matmul_nt_quant_impl(
    device const half *A,
    device const uchar *B,
    device half *C,
    uint M, uint K, uint N, uint row_bytes,
    uint2 gid, uint2 tid,
    threadgroup float (&As)[QTILE][QTILE],
    threadgroup float (&Bs)[QTILE][QTILE]
) {
    uint row = gid.y * QTILE + tid.y; // m
    uint col = gid.x * QTILE + tid.x; // n

    float acc = 0.0f;
    uint num_tiles = (K + QTILE - 1) / QTILE;

    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * QTILE + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? float(A[row * K + a_col]) : 0.0f;

        uint b_n = gid.x * QTILE + tid.y;
        uint b_k = t * QTILE + tid.x;
        Bs[tid.y][tid.x] = (b_n < N && b_k < K)
            ? GET(B + (ulong)b_n * row_bytes, b_k)
            : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < QTILE; k++) {
            acc += As[tid.y][k] * Bs[tid.x][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = half(acc);
    }
}

kernel void matmul_nt_q4_0(
    device const half *A  [[buffer(0)]],
    device const uchar *B [[buffer(1)]],
    device half *C        [[buffer(2)]],
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float As[QTILE][QTILE];
    threadgroup float Bs[QTILE][QTILE];
    matmul_nt_quant_impl<q4_0_get>(A, B, C, M, K, N, (K >> 5) * 18, gid, tid, As, Bs);
}

// ---------------------------------------------------------------------------
// simdgroup_matrix (MMA) tiled quantized matmul. Same math as
// matmul_nt_quant_impl (C[M,N] = A[M,K] · B[N,K]ᵀ, B quantized) but uses Apple
// matrix units instead of scalar FMA. The naive scalar kernel is compute-bound
// at ~5 GB/s / ~270 GMAC/s on M4 Pro; this targets the matrix pipes.
//
// Tile: 64×32 output per threadgroup (BM=64, BN=32, BK=32), 8 simdgroups (4×2),
// each computing a 16×16 subtile as four 8×8 simdgroup_matrix<float> fragments.
// B is dequantized once per K-tile into threadgroup memory (K-major) so the
// quant decode amortizes across all BM=64 M rows — for a prefill chunk of
// M=256 that is 4 row-tiles instead of 8, halving the redundant 2-bit weight
// decode vs the old 32×32 tile. K/N are multiples of 32 here (Q4_0 K/32,
// TQ2_0 K/256, projection dims all /32), so only M needs bounds checks.
// Float scratch is 64·32 + 32·32 + 64·32 = 20 KiB, within the 32 KiB budget.
#include <metal_simdgroup_matrix>

constant uint MMA_BM = 64;     // output rows per threadgroup
constant uint MMA_BN = 32;     // output cols per threadgroup
constant uint MMA_BK = 32;     // K tile
constant uint MMA_SG_M = 4;    // 64 / 16 subtile rows
constant uint MMA_SG_N = 2;    // 32 / 16 subtile cols
constant uint MMA_SGS = MMA_SG_M * MMA_SG_N; // 8 simdgroups
constant uint MMA_THREADS = MMA_SGS * 32;    // 256 threads

template <float (*GET)(device const uchar *, uint)>
inline void matmul_nt_quant_mma_impl(
    device const half *A,
    device const uchar *B,
    device half *C,
    uint M, uint K, uint N, uint row_bytes,
    uint2 gid, uint tid, uint sgid,
    threadgroup float (&As)[MMA_BM][MMA_BK],   // [m_local][k_local]
    threadgroup float (&Bs)[MMA_BK][MMA_BN],   // [k_local][n_local] (K-major)
    threadgroup float (&Cs)[MMA_BM][MMA_BN]
) {
    uint m_base = gid.y * MMA_BM;
    uint n_base = gid.x * MMA_BN;

    // This simdgroup's 16×16 subtile within the 64×32 output tile (4×2 grid).
    uint sg_m = (sgid / MMA_SG_N) * 16;  // 0, 16, 32, 48
    uint sg_n = (sgid % MMA_SG_N) * 16;  // 0, 16

    simdgroup_matrix<float, 8, 8> acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_tiles = K / MMA_BK;
    for (uint t = 0; t < num_tiles; t++) {
        uint k_base = t * MMA_BK;
        // A load: BM·BK = 2048 half→float values.
        for (uint idx = tid; idx < MMA_BM * MMA_BK; idx += MMA_THREADS) {
            uint lm = idx / MMA_BK;
            uint lk = idx - lm * MMA_BK;
            uint gm = m_base + lm;
            As[lm][lk] = (gm < M) ? float(A[(ulong)gm * K + k_base + lk]) : 0.0f;
        }
        // B load/dequant: BK·BN = 1024 values, independent of BM (decode reused
        // across all BM rows). Stored K-major as Bs[k_local][n_local].
        for (uint idx = tid; idx < MMA_BK * MMA_BN; idx += MMA_THREADS) {
            uint lk = idx / MMA_BN;
            uint ln = idx - lk * MMA_BN;
            uint gn = n_base + ln;
            Bs[lk][ln] = (gn < N) ? GET(B + (ulong)gn * row_bytes, k_base + lk) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < MMA_BK; kk += 8) {
            simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
            simdgroup_load(a0, &As[sg_m + 0][kk], MMA_BK);
            simdgroup_load(a1, &As[sg_m + 8][kk], MMA_BK);
            simdgroup_load(b0, &Bs[kk][sg_n + 0], MMA_BN);
            simdgroup_load(b1, &Bs[kk][sg_n + 8], MMA_BN);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc00, &Cs[sg_m + 0][sg_n + 0], MMA_BN);
    simdgroup_store(acc01, &Cs[sg_m + 0][sg_n + 8], MMA_BN);
    simdgroup_store(acc10, &Cs[sg_m + 8][sg_n + 0], MMA_BN);
    simdgroup_store(acc11, &Cs[sg_m + 8][sg_n + 8], MMA_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < MMA_BM * MMA_BN; idx += MMA_THREADS) {
        uint lm = idx / MMA_BN;
        uint ln = idx - lm * MMA_BN;
        uint gm = m_base + lm;
        uint gn = n_base + ln;
        if (gm < M && gn < N) {
            C[(ulong)gm * N + gn] = half(Cs[lm][ln]);
        }
    }
}

kernel void matmul_nt_q4_0_mma(
    device const half *A  [[buffer(0)]],
    device const uchar *B [[buffer(1)]],
    device half *C        [[buffer(2)]],
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float As[MMA_BM][MMA_BK];
    threadgroup float Bs[MMA_BK][MMA_BN];
    threadgroup float Cs[MMA_BM][MMA_BN];
    matmul_nt_quant_mma_impl<q4_0_get>(A, B, C, M, K, N, (K >> 5) * 18, gid, tid, sgid, As, Bs, Cs);
}

kernel void matmul_nt_tq2_0_mma(
    device const half *A  [[buffer(0)]],
    device const uchar *B [[buffer(1)]],
    device half *C        [[buffer(2)]],
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float As[MMA_BM][MMA_BK];
    threadgroup float Bs[MMA_BK][MMA_BN];
    threadgroup float Cs[MMA_BM][MMA_BN];
    matmul_nt_quant_mma_impl<tq2_0_get>(A, B, C, M, K, N, (K >> 8) * 66, gid, tid, sgid, As, Bs, Cs);
}

// ---------------------------------------------------------------------------
// Small-M Q4_0 matmul (speculative-decode verification, M<=8).
//
// C[M,N] = A[M,K] · B[N,K]ᵀ, B in Q4_0. The 32×32 MMA kernel above pads M up
// to 32, so for the M≈2–7 verify path it wastes ~75% of its matrix-unit work
// AND re-parses each quant block per element via q4_0_get(). This kernel
// instead uses an 8×64 output tile (BM=8, BN=64), exactly one Q4_0 block per
// K-tile (BK=32), and dequantizes each block ONCE into a `half` threadgroup
// tile that is then fed to half×half→float simdgroup_matrix units. Each
// weight byte is read once total; only M is padded (to 8). One simdgroup owns
// one 8×8 output subtile (8 rows × 8 of the 64 N columns).
constant uint SM_BM = 8;
constant uint SM_BN = 64;
constant uint SM_BK = 32;       // one Q4_0 block
constant uint SM_SGS = 8;       // simdgroups per threadgroup (one per 8 N cols)

kernel void matmul_nt_q4_0_smallm(
    device const half *A  [[buffer(0)]],   // [M, K] row-major
    device const uchar *B [[buffer(1)]],   // [N, K] Q4_0
    device half *C        [[buffer(2)]],   // [M, N] row-major
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    threadgroup half  As[SM_BM][SM_BK];   // 0.5 KiB
    threadgroup half  Bs[SM_BK][SM_BN];   // 4.0 KiB
    threadgroup float Cs[SM_BM][SM_BN];   // 2.0 KiB

    uint m_base = gid.y * SM_BM;
    uint n_base = gid.x * SM_BN;
    uint blocks = K >> 5;

    // This simdgroup dequantizes its own 8 N-rows: lane → (row8, part).
    uint row8 = lane >> 2;          // 0..7  (which of the 8 N rows)
    uint part = lane & 3;           // 0..3  (which 4-byte quarter of the block)
    uint n_local = sgid * 8 + row8; // 0..63 column within the 64-wide tile
    uint n = n_base + n_local;
    bool n_ok = n < N;

    simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_tiles = K >> 5;
    for (uint t = 0; t < num_tiles; t++) {
        uint k_base = t << 5;

        // Cooperative A load: 8×32 half (256 threads → one pass).
        for (uint idx = tid; idx < SM_BM * SM_BK; idx += SM_SGS * 32) {
            uint lm = idx >> 5;       // /32
            uint lk = idx & 31;       // %32
            uint gm = m_base + lm;
            As[lm][lk] = (gm < M) ? A[(ulong)gm * K + k_base + lk] : half(0);
        }

        // Dequantize one Q4_0 block for each of this simdgroup's 8 rows.
        device const uchar *bp = B + ((ulong)n * blocks + t) * 18;
        float d_lane = (part == 0 && n_ok) ? float(*(device const half *)bp) : 0.0f;
        float d = simd_shuffle(d_lane, row8 * 4);   // broadcast scale across the 4 lanes
        if (n_ok) {
            uint qoff = 2 + part * 4;                // 4 nibble bytes for this quarter
            ushort q01 = *(device const ushort *)(bp + qoff);
            ushort q23 = *(device const ushort *)(bp + qoff + 2);
            uchar bytes4[4] = { uchar(q01 & 0xff), uchar(q01 >> 8),
                                uchar(q23 & 0xff), uchar(q23 >> 8) };
            for (uint i = 0; i < 4; i++) {
                uint bi = part * 4 + i;             // byte index 0..15
                uchar q = bytes4[i];
                Bs[bi][n_local]      = half(d * (float(q & 0x0f) - 8.0f)); // k = bi
                Bs[bi + 16][n_local] = half(d * (float(q >> 4)  - 8.0f));  // k = bi+16
            }
        } else {
            for (uint i = 0; i < 4; i++) {
                uint bi = part * 4 + i;
                Bs[bi][n_local]      = half(0);
                Bs[bi + 16][n_local] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> a_frag, b_frag;
        for (uint kk = 0; kk < SM_BK; kk += 8) {
            simdgroup_load(a_frag, &As[0][kk], SM_BK);
            simdgroup_load(b_frag, &Bs[kk][sgid * 8], SM_BN);
            simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &Cs[0][sgid * 8], SM_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < SM_BM * SM_BN; idx += SM_SGS * 32) {
        uint lm = idx >> 6;        // /64
        uint ln = idx & 63;        // %64
        uint gm = m_base + lm;
        uint gn = n_base + ln;
        if (gm < M && gn < N) {
            C[(ulong)gm * N + gn] = half(Cs[lm][ln]);
        }
    }
}

// ---------------------------------------------------------------------------
// Small-M TQ2_0 matmul (speculative-decode verification, M<=8).
//
// C[M,N] = A[M,K] · B[N,K]ᵀ, B in TQ2_0 (256 weights / 66-byte block: 64
// packed 2-bit bytes + fp16 scale at byte 64). The 32×32 MMA kernel pads M up
// to 32 AND re-parses each quant block per element via tq2_0_get() (block
// index, scale load, byte load, bit extract, zero-point subtract — K×N times
// redundantly). The multivec kernel reads each block once but its 2-bit decode
// + FMA is compute-bound and scales ~linearly with M. Both are ~6× a single
// matvec for the M≈2–8 verify batch, so MTP loses.
//
// This kernel mirrors matmul_nt_q4_0_smallm: dequantize each TQ2_0 block ONCE
// into a `half` threadgroup tile, then feed half×half→float simdgroup_matrix
// units. Tile BM=8, BN=64, BK=128 (one 128-weight half-block per K-tile — the
// TQ2_0 packing splits a 256-block into two 32-byte halves, so BK=128 reads
// each packed byte exactly once; BK=32 would reread packed bytes 4×). One
// simdgroup owns one 8×8 output subtile (8 rows × 8 of the 64 N columns).
constant uint TQ_BM = 8;
constant uint TQ_BN = 64;
constant uint TQ_BK = 128;      // one half of a 256-weight TQ2_0 block
constant uint TQ_SGS = 8;       // simdgroups per threadgroup (one per 8 N cols)

kernel void matmul_nt_tq2_0_smallm(
    device const half *A  [[buffer(0)]],   // [M, K] row-major
    device const uchar *B [[buffer(1)]],   // [N, K] TQ2_0
    device half *C        [[buffer(2)]],   // [M, N] row-major
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    threadgroup half  As[TQ_BM][TQ_BK];   // 2.0 KiB
    threadgroup half  Bs[TQ_BK][TQ_BN];   // 16.0 KiB
    threadgroup float Cs[TQ_BM][TQ_BN];   // 2.0 KiB

    uint m_base = gid.y * TQ_BM;
    uint n_base = gid.x * TQ_BN;
    uint blocks = K >> 8;               // 256-weight blocks per row

    // This simdgroup dequantizes its own 8 N-rows: lane → (row8, part).
    uint row8 = lane >> 2;             // 0..7  (which of the 8 N rows)
    uint part = lane & 3;             // 0..3  (which 8-byte quarter of the half-block)
    uint n_local = sgid * 8 + row8;   // 0..63 column within the 64-wide tile
    uint n = n_base + n_local;
    bool n_ok = n < N;

    simdgroup_matrix<float, 8, 8> acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_tiles = K >> 7;          // K / 128
    for (uint t = 0; t < num_tiles; t++) {
        uint k_base = t << 7;

        // Cooperative A load: 8×128 half (1024 elems, 256 threads → 4 passes).
        for (uint idx = tid; idx < TQ_BM * TQ_BK; idx += TQ_SGS * 32) {
            uint lm = idx >> 7;        // /128
            uint lk = idx & 127;       // %128
            uint gm = m_base + lm;
            As[lm][lk] = (gm < M) ? A[(ulong)gm * K + k_base + lk] : half(0);
        }

        // Dequantize one TQ2_0 half-block (128 weights) for each of this
        // simdgroup's 8 rows. The 256-block index is t>>1; the low bit of t
        // selects the 32-byte packed half (bytes 0..31 → k 0..127, bytes
        // 32..63 → k 128..255). The fp16 scale is shared by both halves.
        uint block = t >> 1;                       // 256-block index in this row
        uint byte_base = (t & 1) * 32;             // 0 or 32
        device const uchar *bp = B + ((ulong)n * blocks + block) * 66;
        float d_lane = (part == 0 && n_ok) ? float(*(device const half *)(bp + 64)) : 0.0f;
        float d = simd_shuffle(d_lane, row8 * 4);  // broadcast scale across the 4 part lanes
        if (n_ok) {
            for (uint i = 0; i < 8; i++) {
                uint m = part * 8 + i;             // packed byte 0..31 within the half
                uchar q = bp[byte_base + m];
                // byte m encodes k_local = m, 32+m, 64+m, 96+m (bit pairs l=0..3).
                Bs[m +  0][n_local] = half(d * (float((q >> 0) & 3) - 1.0f));
                Bs[m + 32][n_local] = half(d * (float((q >> 2) & 3) - 1.0f));
                Bs[m + 64][n_local] = half(d * (float((q >> 4) & 3) - 1.0f));
                Bs[m + 96][n_local] = half(d * (float((q >> 6) & 3) - 1.0f));
            }
        } else {
            for (uint i = 0; i < 8; i++) {
                uint m = part * 8 + i;
                Bs[m +  0][n_local] = half(0);
                Bs[m + 32][n_local] = half(0);
                Bs[m + 64][n_local] = half(0);
                Bs[m + 96][n_local] = half(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> a_frag, b_frag;
        for (uint kk = 0; kk < TQ_BK; kk += 8) {
            simdgroup_load(a_frag, &As[0][kk], TQ_BK);
            simdgroup_load(b_frag, &Bs[kk][sgid * 8], TQ_BN);
            simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, &Cs[0][sgid * 8], TQ_BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < TQ_BM * TQ_BN; idx += TQ_SGS * 32) {
        uint lm = idx >> 6;        // /64
        uint ln = idx & 63;        // %64
        uint gm = m_base + lm;
        uint gn = n_base + ln;
        if (gm < M && gn < N) {
            C[(ulong)gm * N + gn] = half(Cs[lm][ln]);
        }
    }
}

// ── Batched fused TQ2_0 matvec (small M, 2..8) ──────────────────────────────
// Speculative-verify / continuous-batch GEMM for small M. Keeps
// `matvec_tq2_0_sg`'s fused no-shift dequant and coalesced 64-byte block read
// (the 2.3 GB weight stream is read once and reused across output rows AND
// across the M activation vectors), avoiding the MMA small-M kernel's
// dequant-to-threadgroup round-trip + barriers (which made verify ~4x slower
// than single-token decode for the same weight bytes). This mirrors MLX's
// `qmv` routing for small M rather than its tiled `qmm`.
//
// Each simdgroup owns BATCHM_R output rows (n). For each 256-weight block it
// loads each row's packed qs + scale ONCE, then loops the M activation vectors
// doing only cheap masked FMAs. Accumulators: acc[BATCHM_R][BATCHM_MC] floats
// (2x8 = 16) — low enough to avoid spills. No threadgroup memory, no barriers.
constant uint BATCHM_R  = 2;   // output weight-rows per simdgroup
constant uint BATCHM_MC = 8;   // max M handled in one weight pass

kernel void matmul_nt_tq2_0_batchm(
    device const half  *A  [[buffer(0)]],   // [M, K] row-major
    device const uchar *B  [[buffer(1)]],   // [N, K] TQ2_0
    device half        *C  [[buffer(2)]],   // [M, N] row-major
    constant uint &M       [[buffer(3)]],
    constant uint &K       [[buffer(4)]],
    constant uint &N       [[buffer(5)]],
    uint tg     [[threadgroup_position_in_grid]],
    uint sgitg  [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    uint base_row = (tg * ROWS_PER_TG + sgitg) * BATCHM_R;
    if (base_row >= N) return;

    uint blocks = K >> 8;

    // Lane owns the two adjacent qs bytes mj0 = 2·lane and mj0+1, so the 32
    // lanes cover all 64 qs bytes of a block as one coalesced 64-byte read.
    uint mj0  = lane * 2;         // 0,2,…,62
    uint j    = mj0 >> 5;         // 0 or 1 (both bytes share it)
    uint m0   = mj0 & 31;         // even
    uint voff = (j << 7) + m0;    // activation offset within the block

    // No-shift dequant scales: 2-bit field at bit position 2k is left scaled by
    // 2^(2k); pre-dividing the activation by 2^(2k) makes the masked bits
    // multiply directly. The -1 zero point collapses to d*(Σq·a_scaled - Σa).
    const float s1c = 0.25f,        s2c = 0.0625f;
    const float s3c = 0.015625f,    s4c = 1.0f / 256.0f;
    const float s5c = 1.0f / 1024.0f;
    const float s6c = 1.0f / 4096.0f;
    const float s7c = 1.0f / 16384.0f;

    float acc[BATCHM_R][BATCHM_MC];
    #pragma unroll
    for (uint r = 0; r < BATCHM_R; r++) {
        #pragma unroll
        for (uint mi = 0; mi < BATCHM_MC; mi++) {
            acc[r][mi] = 0.0f;
        }
    }

    for (uint b = 0; b < blocks; b++) {
        // Load each output row's packed weights + scale once for this block.
        ushort p[BATCHM_R];
        float  d[BATCHM_R];
        #pragma unroll
        for (uint r = 0; r < BATCHM_R; r++) {
            uint row = base_row + r;
            if (row < N) {
                device const uchar *bp = B + ((ulong)row * blocks + b) * 66;
                p[r] = *(device const ushort *)(bp + mj0);
                d[r] = float(*(device const half *)(bp + 64));
            } else {
                p[r] = 0;
                d[r] = 0.0f;
            }
        }

        uint vbase = (b << 8) + voff;
        // Stream one activation vector at a time (weights already in registers);
        // keeps only one vector's pre-scaled activations live.
        #pragma unroll
        for (uint mi = 0; mi < BATCHM_MC; mi++) {
            if (mi >= M) {
                break;
            }
            device const half *av = A + (ulong)mi * K;
            float2 v0  = float2(*(device const half2 *)(av + vbase));
            float2 v32 = float2(*(device const half2 *)(av + vbase + 32));
            float2 v64 = float2(*(device const half2 *)(av + vbase + 64));
            float2 v96 = float2(*(device const half2 *)(av + vbase + 96));
            float as0 = v0.x, as1 = v32.x * s1c, as2 = v64.x * s2c, as3 = v96.x * s3c;
            float as4 = v0.y * s4c, as5 = v32.y * s5c, as6 = v64.y * s6c, as7 = v96.y * s7c;
            float sv = v0.x + v32.x + v64.x + v96.x + v0.y + v32.y + v64.y + v96.y;
            #pragma unroll
            for (uint r = 0; r < BATCHM_R; r++) {
                ushort q = p[r];
                float bsum = float(q & 0x0003) * as0 + float(q & 0x000C) * as1
                           + float(q & 0x0030) * as2 + float(q & 0x00C0) * as3
                           + float(q & 0x0300) * as4 + float(q & 0x0C00) * as5
                           + float(q & 0x3000) * as6 + float(q & 0xC000) * as7;
                acc[r][mi] += d[r] * (bsum - sv);
            }
        }
    }

    #pragma unroll
    for (uint mi = 0; mi < BATCHM_MC; mi++) {
        if (mi >= M) {
            break;
        }
        #pragma unroll
        for (uint r = 0; r < BATCHM_R; r++) {
            uint row = base_row + r;
            float total = simd_sum(acc[r][mi]);
            if (lane == 0 && row < N) {
                C[(ulong)mi * N + row] = half(total);
            }
        }
    }
}

kernel void matmul_nt_tq2_0(
    device const half *A  [[buffer(0)]],
    device const uchar *B [[buffer(1)]],
    device half *C        [[buffer(2)]],
    constant uint &M      [[buffer(3)]],
    constant uint &K      [[buffer(4)]],
    constant uint &N      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float As[QTILE][QTILE];
    threadgroup float Bs[QTILE][QTILE];
    matmul_nt_quant_impl<tq2_0_get>(A, B, C, M, K, N, (K >> 8) * 66, gid, tid, As, Bs);
}

/// Maximum batch (number of input vectors) the multi-vector matvec supports.
/// The verify batch is `draft_len + 1 <= 16`; larger batches use the tiled
/// `matmul_nt_*` kernels instead (where weight reuse across 32-row tiles wins).
constant uint MULTIVEC_MAX_M = 16;

/// Small-batch `[M, K] × [N, K]ᵀ → [M, N]` for quantized B. One SIMD-group per
/// weight row (n) — exactly the layout of the SIMD `matvec_*_sg` kernels — so
/// the quantized weight row streams **once** and is dequantized once per
/// element, then reused across all `M` input vectors (each lane keeps `M`
/// accumulators). For small `M` this avoids both the tiled matmul's wasted
/// 32-row tile compute and the M× weight re-reads of looping matvec; it is the
/// fast path for MTP verification (M≈5) and small prefill chunks.
kernel void multivec_nt_q4_0(
    device const half *A       [[buffer(0)]], // [M, K] row-major
    device const uchar *matrix [[buffer(1)]], // [N, K] Q4_0
    device half *C             [[buffer(2)]], // [M, N] row-major
    constant uint &M           [[buffer(3)]],
    constant uint &K           [[buffer(4)]],
    constant uint &N           [[buffer(5)]],
    uint tg    [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    uint row = tg * ROWS_PER_TG + sgitg; // n
    if (row >= N) return;
    uint blocks = K >> 5;
    device const uchar *rowp = matrix + (ulong)row * blocks * 18;
    uint units = blocks << 4;

    float acc[MULTIVEC_MAX_M];
    for (uint mi = 0; mi < M; mi++) acc[mi] = 0.0f;

    for (uint u = lane; u < units; u += 32) {
        uint b = u >> 4;
        uint i = u & 15;
        device const uchar *bp = rowp + b * 18;
        float d = float(*(device const half *)bp);
        uchar q = bp[2 + i];
        float wlo = d * (float(q & 0x0F) - 8.0f);
        float whi = d * (float(q >> 4) - 8.0f);
        uint base = (b << 5) + i;
        for (uint mi = 0; mi < M; mi++) {
            device const half *av = A + (ulong)mi * K;
            acc[mi] += wlo * float(av[base]) + whi * float(av[base + 16]);
        }
    }
    for (uint mi = 0; mi < M; mi++) {
        float s = simd_sum(acc[mi]);
        if (lane == 0) C[(ulong)mi * N + row] = half(s);
    }
}

kernel void multivec_nt_tq2_0(
    device const half *A       [[buffer(0)]], // [M, K] row-major
    device const uchar *matrix [[buffer(1)]], // [N, K] TQ2_0
    device half *C             [[buffer(2)]], // [M, N] row-major
    constant uint &M           [[buffer(3)]],
    constant uint &K           [[buffer(4)]],
    constant uint &N           [[buffer(5)]],
    uint tg    [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]]
) {
    uint row = tg * ROWS_PER_TG + sgitg; // n
    if (row >= N) return;
    uint blocks = K >> 8;
    device const uchar *rowp = matrix + (ulong)row * blocks * 66;
    uint mj0 = lane * 2;       // 0, 2, …, 62
    uint j = mj0 >> 5;         // 0 or 1
    uint m0 = mj0 & 31;        // even
    uint voff = (j << 7) + m0; // vector offset within the 256-elem block

    float acc[MULTIVEC_MAX_M];
    for (uint mi = 0; mi < M; mi++) acc[mi] = 0.0f;

    for (uint b = 0; b < blocks; b++) {
        device const uchar *bp = rowp + b * 66;
        float d = float(*(device const half *)(bp + 64));
        ushort packed = *(device const ushort *)(bp + mj0);
        float q0a = float(packed & 3) - 1.0f;
        float q0b = float((packed >> 2) & 3) - 1.0f;
        float q0c = float((packed >> 4) & 3) - 1.0f;
        float q0d = float((packed >> 6) & 3) - 1.0f;
        float q1a = float((packed >> 8) & 3) - 1.0f;
        float q1b = float((packed >> 10) & 3) - 1.0f;
        float q1c = float((packed >> 12) & 3) - 1.0f;
        float q1d = float((packed >> 14) & 3) - 1.0f;
        uint vbase = (b << 8) + voff;
        for (uint mi = 0; mi < M; mi++) {
            device const half *av = A + (ulong)mi * K;
            float2 v0 = float2(*(device const half2 *)(av + vbase));
            float2 v32 = float2(*(device const half2 *)(av + vbase + 32));
            float2 v64 = float2(*(device const half2 *)(av + vbase + 64));
            float2 v96 = float2(*(device const half2 *)(av + vbase + 96));
            float bsum = q0a * v0.x + q1a * v0.y
                       + q0b * v32.x + q1b * v32.y
                       + q0c * v64.x + q1c * v64.y
                       + q0d * v96.x + q1d * v96.y;
            acc[mi] += d * bsum;
        }
    }
    for (uint mi = 0; mi < M; mi++) {
        float s = simd_sum(acc[mi]);
        if (lane == 0) C[(ulong)mi * N + row] = half(s);
    }
}
