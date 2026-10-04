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
