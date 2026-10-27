// DFlash2 block drafter (z-lab `DFlash2DraftModel`, five Qwen3 layers with
// grouped dynamic convolutions and a rank-256 candidate selector) and the
// KFlash best-first tree builder. The dense projections run through the int8
// kernels (`bonsai_int8.metal`) and the shared PTQ1 output matrix through the
// target's own kernels; this file holds everything else of the drafting pass.
//
// Shapes are the pinned drafter's: width 5120, 32 query and 8 K/V heads of
// 128, an 8-row block (the anchor and 7 mask rows), 16 candidates per draft
// position, a 2048-token sliding window. Activations are F32; the drafter's
// context K/V ring is F16.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint DF_WIDTH = 5120;
constant constexpr uint DF_HEAD_DIM = 128;
constant constexpr uint DF_Q_HEADS = 32;
constant constexpr uint DF_KV_HEADS = 8;
constant constexpr uint DF_Q_WIDTH = DF_Q_HEADS * DF_HEAD_DIM;
constant constexpr uint DF_KV_WIDTH = DF_KV_HEADS * DF_HEAD_DIM;
constant constexpr uint DF_GQA = DF_Q_HEADS / DF_KV_HEADS;
constant constexpr uint DF_BLOCK = 8;
constant constexpr uint DF_SLOTS = DF_BLOCK - 1;
constant constexpr uint DF_TOPK = 16;
constant constexpr uint DF_RANK = 256;
constant constexpr uint DF_CONV_GROUP = 16;
constant constexpr uint DF_CONV_GROUPS = DF_WIDTH / DF_CONV_GROUP;
constant constexpr uint DF_WINDOW = 2048;
constant constexpr uint DF_MAX_ROWS = 8;
constant constexpr uint DF_ROPE_PAIRS = DF_HEAD_DIM / 2;
constant constexpr uint DF_ROPE_FINE = 256;
// Keys one attention threadgroup scores.
constant constexpr uint DF_CHUNK = 64;
// Queries sharing one K/V head: 4 heads x 8 block rows.
constant constexpr uint DF_GROUP_QUERIES = DF_GQA * DF_BLOCK;

// ---- elementwise ----

// Row-wise RMSNorm with a plain (Qwen3) weight: y = x * rsqrt(mean(x^2) + eps) * w.
// Grid (rows), 256 threads.
kernel void dflash_rms_rows(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &width [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[8];
    device const float *x = input + ulong(row) * width;
    float sum = 0.0f;
    for (uint i = tid; i < width; i += 256) sum = fma(x[i], x[i], sum);
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float scale = rsqrt(total / float(width) + epsilon);
    device float *y = output + ulong(row) * width;
    for (uint i = tid; i < width; i += 256) y[i] = x[i] * scale * weight[i];
}

// x += y over `count` elements.
kernel void dflash_add(
    device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < count) x[i] += y[i];
}

// out = silu(gate) * up over `count` elements.
kernel void dflash_silu_mul(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) {
        const float g = gate[i];
        output[i] = g / (1.0f + exp(-g)) * up[i];
    }
}

// Target hidden taps, F16 or F32, to the F32 rows the int8 `fc` reads.
kernel void dflash_taps_f16(
    device const half *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = float(input[i]);
}

kernel void dflash_taps_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = input[i];
}

// ---- grouped dynamic convolution ----

// One half of a DFlash2 `GroupedDynamicConv` (kernel 2, groups of 16
// channels) along the block's rows, zero before row 0:
//   out[t, c] = sum_{o < 2} (base[o, c] + dyn[t, o, c / 16]) * x[t - o, c].
// `dyn` rows are the 1280-wide kernel projection, [which][o][group]; the
// caller binds `dyn` and `base` at the half (`which`) it runs.
kernel void dflash_conv(
    device const float *x [[buffer(0)]], device const float *dyn [[buffer(1)]],
    device const float *base [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], uint i [[thread_position_in_grid]]
) {
    if (i >= rows * DF_WIDTH) return;
    const uint t = i / DF_WIDTH, c = i % DF_WIDTH, g = c / DF_CONV_GROUP;
    device const float *d = dyn + ulong(t) * (4 * DF_CONV_GROUPS);
    float acc = (base[c] + d[g]) * x[i];
    if (t >= 1) acc += (base[DF_WIDTH + c] + d[DF_CONV_GROUPS + g]) * x[i - DF_WIDTH];
    output[i] = acc;
}

// ---- per-head RMSNorm and rotary embedding ----

// (cos, sin) of position * inv_freq[i], from host tables computed in F64:
// coarse[a] at position 256 a, fine[b] at position b, combined by the angle sum.
static inline float2 df_rope(
    device const float2 *coarse, device const float2 *fine, uint position, uint i
) {
    const float2 a = coarse[(position / DF_ROPE_FINE) * DF_ROPE_PAIRS + i];
    const float2 b = fine[(position % DF_ROPE_FINE) * DF_ROPE_PAIRS + i];
    return float2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

// One 128-wide head per 128-thread threadgroup: RMSNorm by `weight`, then
// the default (rotate-half) rotary embedding at `position`. Every thread calls it.
static inline float df_head_norm_rope(
    float value, device const float *weight, device const float2 *coarse,
    device const float2 *fine, uint position, float epsilon, uint d, uint lane, uint simd,
    threadgroup float *partial, threadgroup float *shared
) {
    const float sum = simd_sum(value * value);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = partial[0] + partial[1] + partial[2] + partial[3];
    const float normed = value * rsqrt(total / float(DF_HEAD_DIM) + epsilon) * weight[d];
    shared[d] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float2 cs = df_rope(coarse, fine, position, d % DF_ROPE_PAIRS);
    const float other = d < DF_ROPE_PAIRS ? -shared[d + DF_ROPE_PAIRS] : shared[d - DF_ROPE_PAIRS];
    return normed * cs.x + other * cs.y;
}

// The block's queries and keys in place: grid (32 + 8 heads, rows), 128
// threads; row t sits at `position + t`.
kernel void dflash_qk_norm_rope(
    device float *q [[buffer(0)]], device float *k [[buffer(1)]],
    device const float *q_norm [[buffer(2)]], device const float *k_norm [[buffer(3)]],
    device const float2 *coarse [[buffer(4)]], device const float2 *fine [[buffer(5)]],
    constant uint &position [[buffer(6)]], constant float &epsilon [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[DF_HEAD_DIM];
    const uint head = group.x, t = group.y;
    const bool is_q = head < DF_Q_HEADS;
    device float *row = is_q ? q + ulong(t) * DF_Q_WIDTH + head * DF_HEAD_DIM
                             : k + ulong(t) * DF_KV_WIDTH + (head - DF_Q_HEADS) * DF_HEAD_DIM;
    const float value = row[d];
    row[d] = df_head_norm_rope(
        value, is_q ? q_norm : k_norm, coarse, fine, position + t, epsilon, d, lane, simd,
        partial, shared);
}

// Context K/V rows into one layer's F16 ring: grid (8 heads, rows), 128
// threads. Row t is position `position + t`, stored at slot position % capacity.
kernel void dflash_ctx_store(
    device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
    device const float *k_norm [[buffer(2)]], device const float2 *coarse [[buffer(3)]],
    device const float2 *fine [[buffer(4)]], device half *k_cache [[buffer(5)]],
    device half *v_cache [[buffer(6)]], constant uint &position [[buffer(7)]],
    constant uint &capacity [[buffer(8)]], constant float &epsilon [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[DF_HEAD_DIM];
    const uint head = group.x, t = group.y;
    const uint at = position + t;
    const ulong source = ulong(t) * DF_KV_WIDTH + head * DF_HEAD_DIM + d;
    const ulong slot = (ulong(at % capacity) * DF_KV_HEADS + head) * DF_HEAD_DIM + d;
    const float key = df_head_norm_rope(
        k[source], k_norm, coarse, fine, at, epsilon, d, lane, simd, partial, shared);
    k_cache[slot] = half(key);
    v_cache[slot] = half(v[source]);
}

// ---- attention ----

// Block attention over the context window and the block itself, split over
// key chunks (flash decoding). Keys j < nctx are context positions
// P - nctx + j in the F16 ring; keys nctx..nctx+8 are the block's own rows
// (F32, normed and rotated), visible to every row (the block is
// bidirectional). Query row t sees context position p only when
// P + t - p < 2048, the drafter's sliding window.
//
// Grid (chunks, 8 K/V heads), 256 threads: a threadgroup scores its 64 keys
// against the 32 queries sharing its K/V head (4 heads x 8 rows) and writes
// unnormalized partial outputs with their running max and sum.
kernel void dflash_attention_partial(
    device const float *q [[buffer(0)]], device const float *block_k [[buffer(1)]],
    device const float *block_v [[buffer(2)]], device const half *k_cache [[buffer(3)]],
    device const half *v_cache [[buffer(4)]], device float4 *partial_out [[buffer(5)]],
    device float2 *partial_stats [[buffer(6)]], constant uint &position [[buffer(7)]],
    constant uint &nctx [[buffer(8)]], constant uint &capacity [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float4 queries[DF_GROUP_QUERIES * DF_HEAD_DIM / 4];
    threadgroup float scores[DF_GROUP_QUERIES * DF_CHUNK];
    threadgroup float maxima[DF_GROUP_QUERIES];
    threadgroup float sums[DF_GROUP_QUERIES];
    const uint chunk = group.x, g = group.y;
    const uint nkeys = nctx + DF_BLOCK;
    const float scale = rsqrt(float(DF_HEAD_DIM));
    // Query qi = t * 4 + h is head g * 4 + h of block row t.
    for (uint i = tid; i < DF_GROUP_QUERIES * DF_HEAD_DIM / 4; i += 256) {
        const uint qi = i / (DF_HEAD_DIM / 4), d4 = i % (DF_HEAD_DIM / 4);
        const uint t = qi / DF_GQA, h = qi % DF_GQA;
        queries[i] = reinterpret_cast<device const float4 *>(
            q + ulong(t) * DF_Q_WIDTH + (g * DF_GQA + h) * DF_HEAD_DIM)[d4] * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scores: thread owns key kj and queries qg * 8 .. qg * 8 + 8.
    {
        const uint kj = tid % DF_CHUNK, qg = tid / DF_CHUNK;
        const uint j = chunk * DF_CHUNK + kj;
        float acc[8];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 8; ++r) acc[r] = 0.0f;
        if (j < nctx) {
            const uint slot = (position - nctx + j) % capacity;
            device const half4 *key = reinterpret_cast<device const half4 *>(
                k_cache + (ulong(slot) * DF_KV_HEADS + g) * DF_HEAD_DIM);
            for (uint d4 = 0; d4 < DF_HEAD_DIM / 4; ++d4) {
                const float4 kv = float4(key[d4]);
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 8; ++r) {
                    acc[r] += dot(kv, queries[(qg * 8 + r) * (DF_HEAD_DIM / 4) + d4]);
                }
            }
        } else if (j < nkeys) {
            device const float4 *key = reinterpret_cast<device const float4 *>(
                block_k + ulong(j - nctx) * DF_KV_WIDTH + g * DF_HEAD_DIM);
            for (uint d4 = 0; d4 < DF_HEAD_DIM / 4; ++d4) {
                const float4 kv = key[d4];
                #pragma clang loop unroll(full)
                for (uint r = 0; r < 8; ++r) {
                    acc[r] += dot(kv, queries[(qg * 8 + r) * (DF_HEAD_DIM / 4) + d4]);
                }
            }
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 8; ++r) {
            const uint qi = qg * 8 + r, t = qi / DF_GQA;
            // Context position p = P - nctx + j is visible to row t iff p + 2048 > P + t.
            const bool visible =
                j < nkeys && (j >= nctx || position - nctx + j + DF_WINDOW > position + t);
            scores[qi * DF_CHUNK + kj] = visible ? acc[r] : -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax statistics: eight threads per query.
    {
        const uint qi = tid / 8, sub = tid % 8;
        float m = -INFINITY;
        for (uint x = sub; x < DF_CHUNK; x += 8) m = max(m, scores[qi * DF_CHUNK + x]);
        m = max(m, simd_shuffle_xor(m, 1));
        m = max(m, simd_shuffle_xor(m, 2));
        m = max(m, simd_shuffle_xor(m, 4));
        float l = 0.0f;
        for (uint x = sub; x < DF_CHUNK; x += 8) {
            const float p = m == -INFINITY ? 0.0f : exp(scores[qi * DF_CHUNK + x] - m);
            scores[qi * DF_CHUNK + x] = p;
            l += p;
        }
        l += simd_shuffle_xor(l, 1);
        l += simd_shuffle_xor(l, 2);
        l += simd_shuffle_xor(l, 4);
        if (sub == 0) {
            maxima[qi] = m;
            sums[qi] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // P V: thread owns four dimensions of queries qb * 4 .. qb * 4 + 4.
    const uint d4 = tid % (DF_HEAD_DIM / 4), qb = tid / (DF_HEAD_DIM / 4);
    float4 out[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    for (uint kj = 0; kj < DF_CHUNK; ++kj) {
        const uint j = chunk * DF_CHUNK + kj;
        if (j >= nkeys) break;
        float4 value;
        if (j < nctx) {
            const uint slot = (position - nctx + j) % capacity;
            value = float4(reinterpret_cast<device const half4 *>(
                v_cache + (ulong(slot) * DF_KV_HEADS + g) * DF_HEAD_DIM)[d4]);
        } else {
            value = reinterpret_cast<device const float4 *>(
                block_v + ulong(j - nctx) * DF_KV_WIDTH + g * DF_HEAD_DIM)[d4];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) out[r] += scores[(qb * 4 + r) * DF_CHUNK + kj] * value;
    }
    const ulong base = (ulong(chunk) * DF_KV_HEADS + g) * DF_GROUP_QUERIES;
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        partial_out[(base + qb * 4 + r) * (DF_HEAD_DIM / 4) + d4] = out[r];
    }
    if (tid < DF_GROUP_QUERIES) partial_stats[base + tid] = float2(maxima[tid], sums[tid]);
}

// Combine the chunks: grid (32 heads, 8 rows), 128 threads; output [8, 4096].
kernel void dflash_attention_reduce(
    device const float *partial_out [[buffer(0)]], device const float2 *partial_stats [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &chunks [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]
) {
    const uint head = group.x, t = group.y;
    const uint g = head / DF_GQA, qi = t * DF_GQA + head % DF_GQA;
    float m = -INFINITY;
    for (uint c = 0; c < chunks; ++c) {
        m = max(m, partial_stats[(ulong(c) * DF_KV_HEADS + g) * DF_GROUP_QUERIES + qi].x);
    }
    float l = 0.0f, o = 0.0f;
    for (uint c = 0; c < chunks; ++c) {
        const ulong at = (ulong(c) * DF_KV_HEADS + g) * DF_GROUP_QUERIES + qi;
        const float2 stats = partial_stats[at];
        const float w = stats.x == -INFINITY ? 0.0f : exp(stats.x - m);
        l += stats.y * w;
        o += partial_out[at * DF_HEAD_DIM + d] * w;
    }
    output[ulong(t) * DF_Q_WIDTH + head * DF_HEAD_DIM + d] = o / l;
}

// ---- candidate lattice ----

struct DfCandidate {
    uint id;
    float logit;
};

// The GPU-resident lattice (`DflashLattice` on the host).
struct DfLattice {
    uint cand[DF_SLOTS][DF_TOPK];
    float logit[DF_SLOTS][DF_TOPK];
    float edges[DF_SLOTS][DF_TOPK][DF_TOPK];
    float cond_logq[DF_SLOTS][DF_TOPK][DF_TOPK];
};

static inline float4 df_bf16x4(device const ushort4 *row, uint i) {
    const uint4 bits = uint4(row[i]) << 16;
    return as_type<float4>(bits);
}

// Lattice of draft slot s (grid 7, 256 threads, thread j * 16 + k): the
// top-16 candidates and logits of row s, the selector's pairwise score
//   edges[s][j][k] = sum_r A[pred_j, r] * sel[s, r] * B[cand[s][k], r]
// with pred_j the anchor for s = 0 (the same for every j) and candidate j of
// slot s - 1 otherwise, and the conditional log-probability of candidate k
// after candidate j: log_softmax_k(logit[s][k] + edges[s][j][k]).
// `top` rows are `top_stride` candidates apart; codebooks are BF16 bits.
kernel void dflash_lattice(
    device const DfCandidate *top [[buffer(0)]], device const float *selector [[buffer(1)]],
    device const ushort *predecessor [[buffer(2)]], device const ushort *successor [[buffer(3)]],
    device const uint *tokens [[buffer(4)]], device DfLattice *lattice [[buffer(5)]],
    constant uint &top_stride [[buffer(6)]],
    uint s [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    const uint j = tid / DF_TOPK, k = tid % DF_TOPK;
    const uint pred = s == 0 ? tokens[0] : top[(s - 1) * top_stride + j].id;
    const DfCandidate own = top[s * top_stride + k];
    device const ushort4 *a = reinterpret_cast<device const ushort4 *>(predecessor + ulong(pred) * DF_RANK);
    device const ushort4 *b = reinterpret_cast<device const ushort4 *>(successor + ulong(own.id) * DF_RANK);
    device const float4 *w = reinterpret_cast<device const float4 *>(selector + s * DF_RANK);
    float e = 0.0f;
    for (uint r = 0; r < DF_RANK / 4; ++r) e += dot(df_bf16x4(a, r) * w[r], df_bf16x4(b, r));
    lattice->edges[s][j][k] = e;
    if (j == 0) {
        lattice->cand[s][k] = own.id;
        lattice->logit[s][k] = own.logit;
    }
    // The 16 candidates after one predecessor are 16 consecutive lanes.
    const float u = own.logit + e;
    float m = u;
    m = max(m, simd_shuffle_xor(m, 8));
    m = max(m, simd_shuffle_xor(m, 4));
    m = max(m, simd_shuffle_xor(m, 2));
    m = max(m, simd_shuffle_xor(m, 1));
    float z = precise::exp(u - m);
    z += simd_shuffle_xor(z, 8);
    z += simd_shuffle_xor(z, 4);
    z += simd_shuffle_xor(z, 2);
    z += simd_shuffle_xor(z, 1);
    lattice->cond_logq[s][j][k] = u - m - precise::log(z);
}

// ---- best-first tree ----

// Per-depth logistic calibration p = sigmoid(alpha[d] + beta[d] logit(q)),
// d = 1..7 (index 0 unused); alpha 0, beta 1 is p = q.
struct DfCalibration {
    float alpha[DF_MAX_ROWS];
    float beta[DF_MAX_ROWS];
};

// The tree (`DflashTree` on the host).
struct DfTree {
    uint rows;
    uint pad[3];
    uint token[DF_MAX_ROWS];
    int parent[DF_MAX_ROWS];
    uint depth[DF_MAX_ROWS];
    uint slot[DF_MAX_ROWS];
    uint ancestors[DF_MAX_ROWS];
    float log_q[DF_MAX_ROWS];
    float path_log_p[DF_MAX_ROWS];
    float expected[DF_MAX_ROWS];
};

// log of the calibrated acceptance probability of a node of log q `lq`.
static inline float df_lcal(float lq, float a, float b) {
    if (a == 0.0f && b == 1.0f) return lq;
    const float q = min(precise::exp(lq), 0.999999f);
    const float z = a + b * (lq - precise::log(1.0f - q));
    if (z <= -30.0f) return z;
    const float e = precise::exp(-z);
    return -(e < 1e-4f ? e - 0.5f * e * e : precise::log(1.0f + e));
}

// Best-first tree over the lattice: one threadgroup of 32 threads. A node's
// score is its path's summed calibrated log p; the frontier node with the
// highest score (ties: the earlier frontier entry) becomes the next row and
// its 16 children join the frontier, until `max_rows` rows (anchor
// included). A child never scores above its parent, so parents precede
// children and the first n rows are the best n-row tree for every n.
// expected[n - 1] = E(n) = 1 + sum of the path probabilities of rows 1..n-1.
kernel void dflash_tree(
    device const DfLattice *lattice [[buffer(0)]], device const uint *tokens [[buffer(1)]],
    device const DfCalibration *calibration [[buffer(2)]], device DfTree *tree [[buffer(3)]],
    constant uint &max_rows [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint frontier = DF_TOPK * DF_MAX_ROWS;
    threadgroup float fs[frontier];
    threadgroup float fq[frontier];
    // parent row | candidate << 4 | slot << 8
    threadgroup uint fe[frontier];
    threadgroup float ns[DF_MAX_ROWS];
    threadgroup uint anc[DF_MAX_ROWS];
    threadgroup uint nf;
    const uint rows = clamp(max_rows, 1u, DF_MAX_ROWS);
    for (uint c = lane; c < DF_TOPK; c += 32) {
        const float lq = lattice->cond_logq[0][0][c];
        fs[c] = df_lcal(lq, calibration->alpha[1], calibration->beta[1]);
        fq[c] = lq;
        fe[c] = c << 4;
    }
    if (lane == 0) {
        nf = DF_TOPK;
        ns[0] = 0.0f;
        anc[0] = 1;
        tree->token[0] = tokens[0];
        tree->parent[0] = -1;
        tree->depth[0] = 0;
        tree->slot[0] = 0;
        tree->ancestors[0] = 1;
        tree->log_q[0] = 0.0f;
        tree->path_log_p[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint built = 1;
    for (uint r = 1; r < rows; ++r) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        for (uint i = lane; i < nf; i += 32) {
            if (fs[i] > bv) {
                bv = fs[i];
                bi = i;
            }
        }
        for (uint off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(bv, off);
            const uint oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi == 0xffffffffu) break;
        const uint e = fe[bi], pr = e & 15u, k = (e >> 4) & 15u, s = e >> 8;
        const uint depth = s + 1;
        const float lq = fq[bi];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            fs[bi] = -INFINITY;
            ns[r] = bv;
            anc[r] = anc[pr] | (1u << r);
            tree->token[r] = lattice->cand[s][k];
            tree->parent[r] = int(pr);
            tree->depth[r] = depth;
            tree->slot[r] = k;
            tree->ancestors[r] = anc[r];
            tree->log_q[r] = lq;
            tree->path_log_p[r] = bv;
        }
        const bool push = depth < DF_SLOTS && r + 1 < rows;
        const uint base = nf;
        if (push) {
            const float a = calibration->alpha[depth + 1], b = calibration->beta[depth + 1];
            for (uint c = lane; c < DF_TOPK; c += 32) {
                const float child = lattice->cond_logq[depth][k][c];
                fs[base + c] = bv + df_lcal(child, a, b);
                fq[base + c] = child;
                fe[base + c] = r | (c << 4) | (depth << 8);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && push) nf = base + DF_TOPK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        built = r + 1;
    }
    if (lane == 0) {
        tree->rows = built;
        float expected = 1.0f;
        for (uint n = 1; n <= DF_MAX_ROWS; ++n) {
            if (n > 1 && n <= built) expected += precise::exp(ns[n - 1]);
            tree->expected[n - 1] = expected;
        }
        for (uint r = built; r < DF_MAX_ROWS; ++r) {
            tree->token[r] = 0;
            tree->parent[r] = -1;
            tree->depth[r] = 0;
            tree->slot[r] = 0;
            tree->ancestors[r] = 0;
            tree->log_q[r] = 0.0f;
            tree->path_log_p[r] = -INFINITY;
        }
    }
}
