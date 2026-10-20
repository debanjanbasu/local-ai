#include <metal_stdlib>
using namespace metal;

struct TopKCandidate { uint token_id; float logit; };

static inline int total_key(uint bits) {
    int key = as_type<int>(bits);
    return key ^ int(uint(key >> 31) >> 1);
}

static inline bool better(uint a_bits, uint a_id, uint b_bits, uint b_id) {
    if (a_id == UINT_MAX) return false;
    if (b_id == UINT_MAX) return true;
    const int a_key = total_key(a_bits);
    const int b_key = total_key(b_bits);
    return a_key > b_key || (a_key == b_key && a_id < b_id);
}

kernel void sampling_chunk_topk(
    device const uint *logit_bits [[buffer(0)]],
    device TopKCandidate *partial [[buffer(1)]],
    constant uint &vocab [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint bits[256];
    threadgroup uint ids[256];
    const uint id = group * 256 + tid;
    uint current_id = id < vocab ? id : UINT_MAX;
    uint current_bits = id < vocab ? logit_bits[id] : 0;
    ids[tid] = current_id;
    bits[tid] = current_bits;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sort each chunk once, rather than performing k full reductions/reloads.
    for (uint width = 2; width <= 256; width <<= 1) {
        for (uint stride = width >> 1; stride > 0; stride >>= 1) {
            const uint other = tid ^ stride;
            const uint other_bits = bits[other], other_id = ids[other];
            const bool take_better = ((tid & stride) == 0) == ((tid & width) == 0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (take_better ? better(other_bits, other_id, current_bits, current_id)
                            : better(current_bits, current_id, other_bits, other_id)) {
                current_bits = other_bits;
                current_id = other_id;
            }
            bits[tid] = current_bits;
            ids[tid] = current_id;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < k) partial[group * k + tid] = TopKCandidate{current_id, as_type<float>(current_bits)};
}

kernel void sampling_merge_topk(
    device const TopKCandidate *partial [[buffer(0)]],
    device TopKCandidate *output [[buffer(1)]],
    constant uint &groups [[buffer(2)]],
    constant uint &k [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid >= 2 * k) return;
    const bool right = tid >= k;
    const uint own_list = group * 2 + uint(right);
    if (own_list >= groups) return;
    const uint own_rank = tid % k;
    const TopKCandidate item = partial[own_list * k + own_rank];
    const uint other_list = group * 2 + uint(!right);
    uint lo = 0, hi = other_list < groups ? k : 0;
    // Every candidate finds its merged rank in parallel. Prefer the left list
    // for identical sentinels, so even padding has unique, deterministic ranks.
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        const TopKCandidate other = partial[other_list * k + mid];
        const bool precedes = better(as_type<uint>(other.logit), other.token_id,
                                     as_type<uint>(item.logit), item.token_id)
                              || (right && other.token_id == item.token_id);
        if (precedes) lo = mid + 1; else hi = mid;
    }
    const uint rank = own_rank + lo;
    if (rank < k) output[group * k + rank] = item;
}

// Raw log-probabilities of logit rows: each row's log-sum-exp and its `k`
// best logits (k <= 20), in `f32::total_cmp` order with ties to the lower id,
// the order of the top-k and greedy kernels. Logits are only read.
//
// Stage 1, grid (groups, rows), 256 threads: group x of row y reduces its
// 4096 logits to a (max, sum of exp(logit - max)) pair and its own top k.
// Stage 2, grid (rows), 256 threads: row y merges its groups' pairs into the
// log-sum-exp and their candidates into the row's top k.

constant constexpr uint LOGPROB_GROUP = 4096;
constant constexpr uint LOGPROB_PER_THREAD = LOGPROB_GROUP / 256;
constant constexpr uint LOGPROB_MAX_TOP = 20;
constant constexpr uint LOGPROB_FINAL_PER_THREAD = 8;

static inline bool key_better(int a_key, uint a_id, int b_key, uint b_id) {
    if (a_id == UINT_MAX) return false;
    if (b_id == UINT_MAX) return true;
    return a_key > b_key || (a_key == b_key && a_id < b_id);
}

// Fold `other` (a max and its sum of exponentials) into `m`, `s`.
static inline void lse_merge(thread float &m, thread float &s, float other_m, float other_s) {
    if (other_s <= 0.0f) return;
    if (s <= 0.0f) { m = other_m; s = other_s; return; }
    const float top = max(m, other_m);
    s = s * precise::exp(m - top) + other_s * precise::exp(other_m - top);
    m = top;
}

// The threadgroup's (max, sum) from every thread's own.
static inline float2 lse_group(float m, float s, threadgroup float2 *shared, uint lane, uint simd) {
    const float simd_top = simd_max(s > 0.0f ? m : -INFINITY);
    const float simd_total = simd_sum(s > 0.0f ? s * precise::exp(m - simd_top) : 0.0f);
    if (lane == 0) shared[simd] = float2(simd_top, simd_total);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = 0.0f, total = 0.0f;
    for (uint i = 0; i < 8; ++i) lse_merge(top, total, shared[i].x, shared[i].y);
    return float2(top, total);
}

// The threadgroup's best (key, id) from every thread's own. Every thread
// gets the answer; the shared slots are free again when it returns.
static inline void best_group(thread int &key, thread uint &id, threadgroup int *keys,
                              threadgroup uint *ids, uint lane, uint simd) {
    const int simd_key = simd_max(id == UINT_MAX ? INT_MIN : key);
    const uint simd_id = simd_min(id != UINT_MAX && key == simd_key ? id : UINT_MAX);
    if (lane == 0) { keys[simd] = simd_key; ids[simd] = simd_id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    key = keys[0];
    id = ids[0];
    for (uint i = 1; i < 8; ++i) {
        if (key_better(keys[i], ids[i], key, id)) { key = keys[i]; id = ids[i]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void logprob_rows_partial(
    device const float *logits [[buffer(0)]],
    device float2 *stats [[buffer(1)]],
    device TopKCandidate *partial [[buffer(2)]],
    constant uint &vocab [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared_lse[8];
    threadgroup int shared_keys[8];
    threadgroup uint shared_ids[8];
    device const float *row = logits + ulong(group.y) * vocab;
    const uint base = group.x * LOGPROB_GROUP + tid;
    int keys[LOGPROB_PER_THREAD];
    float m = -INFINITY;
    uint valid = 0;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < LOGPROB_PER_THREAD; ++i) {
        const uint id = base + i * 256;
        const bool inside = id < vocab;
        const float value = inside ? row[id] : 0.0f;
        keys[i] = total_key(as_type<uint>(value));
        valid |= uint(inside) << i;
        if (inside) m = max(m, value);
    }
    float s = 0.0f;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < LOGPROB_PER_THREAD; ++i) {
        if (valid & (1u << i)) s += precise::exp(as_type<float>(uint(total_key(uint(keys[i])))) - m);
    }
    const float2 lse = lse_group(m, s, shared_lse, lane, simd);
    const uint slot = group.y * groups.x + group.x;
    if (tid == 0) stats[slot] = lse;
    for (uint round = 0; round < k; ++round) {
        int key = INT_MIN;
        uint id = UINT_MAX;
        uint index = 0;
        // Ascending index is ascending id, so strict > keeps the lowest id.
        #pragma clang loop unroll(full)
        for (uint i = 0; i < LOGPROB_PER_THREAD; ++i) {
            if ((valid & (1u << i)) && (id == UINT_MAX || keys[i] > key)) {
                key = keys[i];
                id = base + i * 256;
                index = i;
            }
        }
        const uint own = id;
        best_group(key, id, shared_keys, shared_ids, lane, simd);
        if (own != UINT_MAX && own == id) valid &= ~(1u << index);
        if (tid == 0) {
            partial[slot * LOGPROB_MAX_TOP + round] =
                TopKCandidate{id, as_type<float>(uint(total_key(uint(key))))};
        }
    }
}

kernel void logprob_rows_final(
    device const float2 *stats [[buffer(0)]],
    device const TopKCandidate *partial [[buffer(1)]],
    device float *lse_out [[buffer(2)]],
    device TopKCandidate *top_out [[buffer(3)]],
    constant uint &groups [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float2 shared_lse[8];
    threadgroup int shared_keys[8];
    threadgroup uint shared_ids[8];
    float m = 0.0f, s = 0.0f;
    for (uint g = tid; g < groups; g += 256) {
        const float2 pair = stats[row * groups + g];
        lse_merge(m, s, pair.x, pair.y);
    }
    const float2 lse = lse_group(m, s, shared_lse, lane, simd);
    if (tid == 0) lse_out[row] = lse.x + precise::log(lse.y);
    int keys[LOGPROB_FINAL_PER_THREAD];
    uint ids[LOGPROB_FINAL_PER_THREAD];
    const uint candidates = groups * k;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < LOGPROB_FINAL_PER_THREAD; ++j) {
        const uint c = tid + j * 256;
        ids[j] = UINT_MAX;
        keys[j] = INT_MIN;
        if (c < candidates) {
            const TopKCandidate item = partial[(row * groups + c / k) * LOGPROB_MAX_TOP + c % k];
            ids[j] = item.token_id;
            keys[j] = total_key(as_type<uint>(item.logit));
        }
    }
    for (uint round = 0; round < k; ++round) {
        int key = INT_MIN;
        uint id = UINT_MAX;
        uint index = 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < LOGPROB_FINAL_PER_THREAD; ++j) {
            if (key_better(keys[j], ids[j], key, id)) {
                key = keys[j];
                id = ids[j];
                index = j;
            }
        }
        const uint own = id;
        best_group(key, id, shared_keys, shared_ids, lane, simd);
        if (own != UINT_MAX && own == id) ids[index] = UINT_MAX;
        if (tid == 0) {
            top_out[row * LOGPROB_MAX_TOP + round] =
                TopKCandidate{id, as_type<float>(uint(total_key(uint(key))))};
        }
    }
}
