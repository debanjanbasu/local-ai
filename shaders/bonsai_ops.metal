#include <metal_stdlib>
#include "bonsai_gdn.h"
#include "bonsai_mixer.h"
#include "bonsai_projection.h"
#if __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

kernel void bo_rms(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &dimension [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &stride [[buffer(5)]],
    constant float &epsilon [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    bonsai_rms_impl(input, weights, output, dimension, rows, stride, epsilon, row, tid, partial);
}

kernel void bo_add(
    device const float *input [[buffer(0)]], device const float *residual [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = input[i] + residual[i];
}

kernel void bo_swiglu(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = bonsai_silu(gate[i]) * up[i];
}

kernel void bo_sigmoid_mul(
    device const float *input [[buffer(0)]], device const float *gate [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < count) output[i] = input[i] * bonsai_sigmoid(gate[i]);
}

kernel void bo_bf16_mv(
    device const ushort *weights [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    bonsai_bf16_mv_impl(weights, input, output, rows, columns, group, lane);
}

// One or two BF16 matrices of equal shape (a recurrent layer's alpha and
// beta) over a block of rows, four rows per SIMD group, each output bitwise
// `bo_bf16_mv`'s. Groups interleave the matrices, then run over matrix rows,
// then groups of four rows: dispatch matrices * rows * ceil(tokens / 4)
// groups of 32. Two 48x5120 matrices on an M4 Pro: 11.5 us for 2 to 8 rows,
// 18 at 16, 27 at 32, 50 at 64 and 96 at 128, against 26 / 66 / 119 / 222 /
// 437 us for `bo_bf16_mv` per row and 1.18-1.37 ms for the token-tiled
// GEMM this replaced (two threadgroups per 48-row matrix). Eight rows per
// SIMD group measured 15-60 us.
kernel void bo_bf16_mv_tokens(
    device const ushort *weights0 [[buffer(0)]], device const ushort *weights1 [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *output0 [[buffer(3)]],
    device float *output1 [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]], constant uint &tokens [[buffer(7)]],
    constant uint &matrices [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint matrix = group % matrices, rest = group / matrices;
    bonsai_bf16_mv_tokens_impl<4>(
        matrix == 0 ? weights0 : weights1, input, matrix == 0 ? output0 : output1, rows,
        columns, tokens, rest % rows, rest / rows, lane);
}

// `final_history` may alias `history` (in place) or name a separate buffer,
// leaving `history` untouched for a verification block's rollback.
kernel void bo_conv(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *history [[buffer(2)]], device float *output [[buffer(3)]],
    device float *final_history [[buffer(4)]], constant uint &tokens [[buffer(5)]],
    uint channel [[thread_position_in_grid]]
) {
    bonsai_conv_impl(input, weights, history, output, final_history, tokens, channel);
}

kernel void bo_l2_qk(
    device float *qkv [[buffer(0)]], constant float &epsilon [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    bonsai_l2_qk_impl(qkv, qkv, epsilon, head, tid, partial);
}

// bo_conv, bo_l2_qk and bo_decay over a short token block in one dispatch.
// Threadgroup g < 80 convolves channels 128g.. for every token; the first 32
// are Q/K heads and then normalize each token's head in place exactly as
// bo_l2_qk would after reading the stored values. Threadgroup 80 computes the
// decay/beta pairs, which are independent of both.
kernel void bo_conv_l2_decay(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *history [[buffer(2)]], device float *output [[buffer(3)]],
    device const float *a [[buffer(4)]], device const float *alpha [[buffer(5)]],
    device const float *dt [[buffer(6)]], device const float *raw_beta [[buffer(7)]],
    device float *decay [[buffer(8)]], device float *beta [[buffer(9)]],
    device float *final_history [[buffer(10)]],
    constant uint &tokens [[buffer(11)]], constant float &epsilon [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    if (group == 80) {
        for (uint head = tid; head < 48 * tokens; head += 128) {
            bonsai_decay_impl(a, alpha, dt, raw_beta, decay, beta, tokens, head);
        }
        return;
    }
    bonsai_conv_impl(input, weights, history, output, final_history, tokens, group * 128 + tid);
    if (group >= 32) return;
    for (uint token = 0; token < tokens; ++token) {
        bonsai_l2_qk_impl(output, output, epsilon, token * 32 + group, tid, partial);
        // The next reduction reuses partial; finish reading this one first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void bo_decay(
    device const float *a [[buffer(0)]], device const float *alpha [[buffer(1)]],
    device const float *dt [[buffer(2)]], device const float *raw_beta [[buffer(3)]],
    device float *decay [[buffer(4)]], device float *beta [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    uint head [[thread_position_in_grid]]
) {
    bonsai_decay_impl(a, alpha, dt, raw_beta, decay, beta, tokens, head);
}

// Recurrent state is stored as `State` (F32, F16 or BF16) and computed in F32
// registers. `final_state` may alias `state` (in place) or name a separate
// buffer, leaving `state` untouched for a verification block's rollback.
template<typename State>
static inline void bonsai_gdn_row(
    device const float *qkv, device const float *decay, device const float *beta,
    device const State *state, device float *output, device State *final_state,
    uint tokens, uint row, uint lane
) {
    const uint head = row / 128, key_group = head % 16;
    float values[4];
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(state[row * 128 + lane + i * 32]);
    }
    for (uint token = 0; token < tokens; ++token) {
        device const float *qkv_row = qkv + token * 10240;
        float prediction = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint column = lane + i * 32;
            values[i] *= decay[token * 48 + head];
            prediction = fma(values[i], qkv_row[2048 + key_group * 128 + column], prediction);
        }
        const float correction = (qkv_row[4096 + row] - simd_sum(prediction)) * beta[token * 48 + head];
        float result = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint column = lane + i * 32;
            values[i] = fma(qkv_row[2048 + key_group * 128 + column], correction, values[i]);
            result = fma(values[i], qkv_row[key_group * 128 + column], result);
        }
        result = simd_sum(result);
        if (lane == 0) output[token * 6144 + row] = result * 0.08838834764831845f;
    }
    for (uint i = 0; i < 4; ++i) {
        bonsai_store_state(final_state[row * 128 + lane + i * 32], values[i]);
    }
}

// bo_gdn computes one value row per SIMD group, whatever the threadgroup size:
// single-token decode runs four per threadgroup, each exactly as before, since
// 6144 one-SIMD threadgroups took 21.2 us per layer on an M4 Pro (F16 state)
// and 1,536 four-SIMD ones 15.8 us. Row-range callers still dispatch one SIMD.
#define BONSAI_GDN(SUFFIX, STATE) \
kernel void bo_gdn##SUFFIX( \
    device const float *qkv [[buffer(0)]], device const float *decay [[buffer(1)]], \
    device const float *beta [[buffer(2)]], device const STATE *state [[buffer(3)]], \
    device float *output [[buffer(4)]], device STATE *final_state [[buffer(5)]], \
    constant uint &tokens [[buffer(6)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint simds [[simdgroups_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    bonsai_gdn_row<STATE>(qkv, decay, beta, state, output, final_state, tokens, \
                          group * simds + simd, lane); \
} \
kernel void bo_gdn_rows_4##SUFFIX( \
    device const float *qkv [[buffer(0)]], device const float *decay [[buffer(1)]], \
    device const float *beta [[buffer(2)]], device const STATE *state [[buffer(3)]], \
    device float *output [[buffer(4)]], device STATE *final_state [[buffer(5)]], \
    constant uint &tokens [[buffer(6)]], \
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]] \
) { bonsai_gdn_rows<4>(qkv, decay, beta, state, output, final_state, tokens, group, lane); }

BONSAI_GDN(, float)
BONSAI_GDN(_f16, half)
BONSAI_GDN(_bf16, bfloat)
#undef BONSAI_GDN

kernel void bo_gdn_post(
    device const float *input [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *weights [[buffer(2)]], device float *output [[buffer(3)]],
    constant float &epsilon [[buffer(4)]], uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    bonsai_gdn_post_impl(input, gate, weights, output, epsilon, head, tid, partial);
}

// bo_gdn_post followed by the forward 1024-wide rotation of its 6144-wide
// output row, in one dispatch: threadgroup b of a row gathers the eight grouped
// heads 8b.. that bo_gdn_post would store in its block (grouped head g reads
// head (g % 3) * 16 + g / 3), computes each exactly as bo_gdn_post does and
// rotates them as bonsai_fwht_forward would. Dispatch 6 * tokens groups of 128.
kernel void bo_gdn_post_fwht(
    device const float *input [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *weights [[buffer(2)]], device const float *signs [[buffer(3)]],
    device float *output [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8][4];
    threadgroup float shared[1024];
    const uint token = block / 6, first = (block % 6) * 8;
    float values[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint grouped = first + i;
        const uint index = (token * 48 + (grouped % 3) * 16 + grouped / 3) * 128 + tid;
        values[i] = input[index];
        // bonsai_group_sum's per-SIMD partial sums, all eight heads at once.
        const float sum = simd_sum(values[i] * values[i]);
        if (tid % 32 == 0) partial[i][tid / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = weights[tid];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint grouped = first + i;
        const uint index = (token * 48 + (grouped % 3) * 16 + grouped / 3) * 128 + tid;
        float sum = 0.0f;
        for (uint s = 0; s < 4; ++s) sum += partial[i][s];
        values[i] = values[i] * rsqrt(sum / 128.0f + epsilon) * scale * bonsai_silu(gate[index]);
    }
    bonsai_fwht_values<false>(values, signs, output, 6, block, tid, shared);
}

// bo_add (`hidden + branch`) followed by bonsai_rms_fwht_forward, in one
// dispatch. Every 1024-wide threadgroup recomputes its row's sums for the
// square sum, so the sum goes to `sum_out`, which must not alias `hidden` or
// `branch`; the threadgroup stores only its own block of it. Values are
// bitwise those of bo_add, then bonsai_rms_fwht_forward on its output: each
// thread still accumulates elements tid, tid + 128, ... in order, but loads
// a whole 1024-wide block (or, for a known width, the whole row) before its
// FMA chain, so the loads overlap instead of waiting on one another.
template<uint known_blocks>
static inline void bonsai_add_rms_fwht_impl(
    device const float *hidden, device const float *branch, device const float *weights,
    device const float *signs, device float *sum_out, device float *normalized,
    device float *output, uint blocks_per_row, float epsilon, uint block, uint tid,
    threadgroup float *partial, threadgroup float *shared
) {
    const uint blocks = known_blocks != 0 ? known_blocks : blocks_per_row;
    const uint dimension = blocks * 1024;
    const uint own = block % blocks;
    device const float *row_hidden = hidden + ulong(block / blocks) * dimension;
    device const float *row_branch = branch + ulong(block / blocks) * dimension;
    float square_sum = 0.0f;
    float values[8];
    if (known_blocks != 0) {
        float row[known_blocks != 0 ? known_blocks * 8 : 1];
        #pragma clang loop unroll(full)
        for (uint k = 0; k < known_blocks * 8; ++k) {
            row[k] = row_branch[k * 128 + tid] + row_hidden[k * 128 + tid];
        }
        #pragma clang loop unroll(full)
        for (uint k = 0; k < known_blocks * 8; ++k) {
            square_sum = fma(row[k], row[k], square_sum);
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            values[i] = row_branch[own * 1024 + i * 128 + tid] + row_hidden[own * 1024 + i * 128 + tid];
        }
    } else {
        for (uint chunk = 0; chunk < blocks; ++chunk) {
            float chunk_values[8];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                const uint index = chunk * 1024 + i * 128 + tid;
                chunk_values[i] = row_branch[index] + row_hidden[index];
            }
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                square_sum = fma(chunk_values[i], chunk_values[i], square_sum);
                if (chunk == own) values[i] = chunk_values[i];
            }
        }
    }
    const float sum = bonsai_group_sum<128>(square_sum, tid, partial);
    const float inverse = rsqrt(sum / float(dimension) + epsilon);
    const ulong base = ulong(block) * 1024;
    const uint column = own * 1024;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint local = i * 128 + tid;
        sum_out[base + local] = values[i];
        values[i] = values[i] * inverse * weights[column + local];
        normalized[base + local] = values[i];
    }
    bonsai_fwht_values<false>(values, signs, output, blocks, block, tid, shared);
}

kernel void bo_add_rms_fwht(
    device const float *hidden [[buffer(0)]], device const float *branch [[buffer(1)]],
    device const float *weights [[buffer(2)]], device const float *signs [[buffer(3)]],
    device float *sum_out [[buffer(4)]], device float *normalized [[buffer(5)]],
    device float *output [[buffer(6)]], constant uint &blocks_per_row [[buffer(7)]],
    constant float &epsilon [[buffer(8)]],
    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[4];
    threadgroup float shared[1024];
    // The model width (5120) keeps its whole row in registers.
    if (blocks_per_row == 5) {
        bonsai_add_rms_fwht_impl<5>(hidden, branch, weights, signs, sum_out, normalized,
                                    output, blocks_per_row, epsilon, block, tid, partial, shared);
    } else {
        bonsai_add_rms_fwht_impl<0>(hidden, branch, weights, signs, sum_out, normalized,
                                    output, blocks_per_row, epsilon, block, tid, partial, shared);
    }
}

static inline float bonsai_text_rope(
    device const float *input, device const float *weights,
    float inverse, uint coordinate, uint position, float base
) {
    const float value = input[coordinate] * inverse * weights[coordinate];
    if (coordinate >= 64) return value;
    const uint pair = coordinate % 32, other = coordinate < 32 ? coordinate + 32 : coordinate - 32;
    const float other_value = input[other] * inverse * weights[other];
    const float theta = float(position) * pow(base, -float(pair) / 32.0f);
    return coordinate < 32 ? value * cos(theta) - other_value * sin(theta)
                           : other_value * sin(theta) + value * cos(theta);
}

// K/V cache storage formats. A token row is 4 KV heads x 256 dims; the
// quantized formats split each head row into 8 blocks of 32 values (one SIMD
// group per block on write) with one F16 absmax scale per block, elements
// first and the 32 scales after them so every token stays contiguous:
//   F16: 2048 B/token, the historical `half` layout, bitwise unchanged.
//   Q8:  1088 B/token = 1024 int8 + 32 half scales   (value = q * scale)
// Element `lane` of block `block` is dimension block * 32 + lane; readers
// address it exactly as the F16 kernels always have (lane + i * 32). Q8 rows
// are Hadamard-rotated per head (see bonsai_hadamard256); F16 rows are stored
// as computed.
// Format codes: 0 = F16, 1 = Q8.
constant uint BONSAI_KV_F16 = 0;
constant uint BONSAI_KV_Q8 = 1;
constant uint BONSAI_KV_Q8_TOKEN_BYTES = 4 * 256 + 4 * 8 * 2;


// The reader is specialized at compile time: the attention kernels below
// dispatch once on the (key, value) format pair, so the F16 instantiation
// compiles to the historical hoisted half loads with no per-element branch
// (a runtime branch here cost 14 % of decode throughput).
template <uint FORMAT>
static inline float bonsai_kv_load(
    device const uchar *cache, ulong token, uint head, uint block, uint lane
) {
    if (FORMAT == BONSAI_KV_F16) {
        device const half *rows = reinterpret_cast<device const half *>(cache);
        return float(rows[(token * 4 + head) * 256 + block * 32 + lane]);
    }
    if (FORMAT == BONSAI_KV_Q8) {
        device const uchar *row = cache + token * BONSAI_KV_Q8_TOKEN_BYTES;
        const float scale = float(reinterpret_cast<device const half *>(row + 1024)[head * 8 + block]);
        // Rounded to half like the tensor kernels' tiles, so every attention
        // path reads identical operands (as F16 caches always have).
        return float(half(float(reinterpret_cast<device const char *>(row)[head * 256 + block * 32 + lane]) * scale));
    }
    return 0.0f;
}

// Run `IMPL<F, F>(args...)`; key and value always share one format.
#define BONSAI_KV_DISPATCH(IMPL, k_format, v_format, ...)                            \
    switch (k_format) {                                                               \
        case BONSAI_KV_F16: IMPL<0, 0>(__VA_ARGS__); break;                          \
        default: IMPL<1, 1>(__VA_ARGS__); break;                                     \
    }

// All 32 lanes of one SIMD group store one block; `value` is this lane's
// element. The scale is rounded to F16 before quantizing so the stored
// integers match the scale readers will multiply by.
static inline void bonsai_kv_store(
    device uchar *cache, uint format, ulong token, uint head, uint block, uint lane, float value
) {
    if (format == BONSAI_KV_F16) {
        device half *rows = reinterpret_cast<device half *>(cache);
        rows[(token * 4 + head) * 256 + block * 32 + lane] = half(value);
        return;
    }
    const float peak = simd_max(abs(value));
    if (format == BONSAI_KV_Q8) {
        device uchar *row = cache + token * BONSAI_KV_Q8_TOKEN_BYTES;
        const half scale = half(peak / 127.0f);
        const float inverse = scale > half(0.0f) ? 1.0f / float(scale) : 0.0f;
        reinterpret_cast<device char *>(row)[head * 256 + block * 32 + lane] =
            char(clamp(rint(value * inverse), -127.0f, 127.0f));
        if (lane == 0) reinterpret_cast<device half *>(row + 1024)[head * 8 + block] = scale;
        return;
    }
}

// Orthonormal 256-point Walsh-Hadamard transform, one value per thread of a
// 256-thread group. H/16 is symmetric and its own inverse, so (Hq).(Hk) = q.k:
// rotating queries and cached keys leaves every attention score unchanged,
// while spreading each outlier channel's energy over all 256 dimensions before
// quantization (the QuaRot/SpinQuant observation). Values rotate the same way
// and attention outputs are rotated back by `bo_attn_unrotate`.
static inline float bonsai_hadamard256(float value, uint tid, threadgroup float *scratch) {
    for (ushort distance = 1; distance < 32; distance <<= 1) {
        const float other = simd_shuffle_xor(value, distance);
        value = (tid & distance) ? other - value : value + other;
    }
    for (uint distance = 32; distance < 256; distance <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float other = scratch[tid ^ distance];
        value = (tid & distance) ? other - value : value + other;
    }
    return value * (1.0f / 16.0f);
}

// Undo the value rotation of a quantized cache's attention output, one query
// head row per group, then apply the output gate (which does not commute with
// the rotation, so attention kernels run ungated for rotated caches).
kernel void bo_attn_unrotate(
    device float *output [[buffer(0)]], device const float *gate [[buffer(1)]],
    constant uint &gated [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scratch[256];
    const ulong index = ulong(group) * 256 + tid;
    const float value = bonsai_hadamard256(output[index], tid, scratch);
    output[index] = value * (gated ? bonsai_sigmoid(gate[index]) : 1.0f);
}

kernel void bo_attn_prep(
    device const float *qg [[buffer(0)]], device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]], device const float *q_weights [[buffer(3)]],
    device const float *k_weights [[buffer(4)]], device float *query [[buffer(5)]],
    device float *gate [[buffer(6)]], device uchar *k_cache [[buffer(7)]],
    device uchar *v_cache [[buffer(8)]], constant uint &position [[buffer(9)]],
    constant uint &k_format [[buffer(10)]], constant uint &v_format [[buffer(11)]],
    constant float &epsilon [[buffer(12)]], constant float &base [[buffer(13)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8], scratch[256];
    const uint head = group % 24, token = group / 24, absolute_position = position + token;
    // Quantized caches hold Hadamard-rotated rows, so queries rotate with them.
    const bool rotate = k_format != BONSAI_KV_F16;
    device const float *qg_row = qg + token * 12288;
    device const float *key_row = key + token * 1024;
    const float q = qg_row[head * 512 + tid];
    const float q_sum = bonsai_group_sum<256>(q * q, tid, partial);
    float roped = bonsai_text_rope(qg_row + head * 512, q_weights,
        rsqrt(q_sum / 256.0f + epsilon), tid, absolute_position, base);
    if (rotate) roped = bonsai_hadamard256(roped, tid, scratch);
    query[group * 256 + tid] = roped;
    gate[group * 256 + tid] = qg_row[head * 512 + 256 + tid];
    // Every lane must finish reading the shared Q reduction before K reuses it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (head < 4) {
        const float k = key_row[head * 256 + tid];
        const float k_sum = bonsai_group_sum<256>(k * k, tid, partial);
        float rotated = bonsai_text_rope(key_row + head * 256, k_weights,
            rsqrt(k_sum / 256.0f + epsilon), tid, absolute_position, base);
        float v = value[token * 1024 + head * 256 + tid];
        if (rotate) {
            rotated = bonsai_hadamard256(rotated, tid, scratch);
            v = bonsai_hadamard256(v, tid, scratch);
        }
        bonsai_kv_store(k_cache, k_format, absolute_position, head, tid / 32, tid % 32, rotated);
        bonsai_kv_store(v_cache, v_format, absolute_position, head, tid / 32, tid % 32, v);
    }
}

// K/V-only counterpart of bo_attn_prep for rows that only fill a cache (the
// MTP head ingesting committed tokens): one group per token/KV head, the same
// norm and RoPE math, no query or gate work.
kernel void bo_kv_prep(
    device const float *key [[buffer(0)]], device const float *value [[buffer(1)]],
    device const float *k_weights [[buffer(2)]], device uchar *k_cache [[buffer(3)]],
    device uchar *v_cache [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant uint &k_format [[buffer(6)]], constant uint &v_format [[buffer(7)]],
    constant float &epsilon [[buffer(8)]], constant float &base [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partial[8], scratch[256];
    const uint head = group % 4, token = group / 4, absolute_position = position + token;
    device const float *key_row = key + token * 1024;
    const float k = key_row[head * 256 + tid];
    const float k_sum = bonsai_group_sum<256>(k * k, tid, partial);
    float rotated = bonsai_text_rope(key_row + head * 256, k_weights,
        rsqrt(k_sum / 256.0f + epsilon), tid, absolute_position, base);
    float v = value[token * 1024 + head * 256 + tid];
    if (k_format != BONSAI_KV_F16) {
        rotated = bonsai_hadamard256(rotated, tid, scratch);
        v = bonsai_hadamard256(v, tid, scratch);
    }
    bonsai_kv_store(k_cache, k_format, absolute_position, head, tid / 32, tid % 32, rotated);
    bonsai_kv_store(v_cache, v_format, absolute_position, head, tid / 32, tid % 32, v);
}

// Decode attention over one 128-token split. One SIMD group serves
// BONSAI_SPLIT_HEADS of the six query heads that share a KV head, so each
// cached K/V element is fetched from device memory once for all of them
// (the one-head-per-group form re-read every byte six times and held the
// kernel at ~36 GB/s from 65K tokens up). Partials keep one 258-float record
// per (query head, split), so the reduce kernel is unchanged. The group index
// is (kv_head, head slot, split), split fastest.
constant uint BONSAI_SPLIT_HEADS = 2;
constant uint BONSAI_SPLIT_GROUPS_PER_KV_HEAD = 6 / BONSAI_SPLIT_HEADS;

template <uint KF, uint VF>
static inline void bonsai_attn_split_impl(
    device const float *query, device const uchar *key, device const uchar *value,
    device float *partials, uint prefix, uint splits, uint group, uint lane
) {
    const uint split = group % splits, slot = group / splits;
    const uint kv_head = slot / BONSAI_SPLIT_GROUPS_PER_KV_HEAD;
    const uint first_head = kv_head * 6 + (slot % BONSAI_SPLIT_GROUPS_PER_KV_HEAD) * BONSAI_SPLIT_HEADS;
    float q[BONSAI_SPLIT_HEADS][8], acc[BONSAI_SPLIT_HEADS][8];
    float maximum[BONSAI_SPLIT_HEADS], denominator[BONSAI_SPLIT_HEADS];
    for (uint h = 0; h < BONSAI_SPLIT_HEADS; ++h) {
        for (uint i = 0; i < 8; ++i) {
            q[h][i] = query[(first_head + h) * 256 + lane + i * 32];
            acc[h][i] = 0.0f;
        }
        maximum[h] = -INFINITY;
        denominator[h] = 0.0f;
    }
    const uint end = min(split * 128 + 128, prefix);
    for (uint token = split * 128; token < end; ++token) {
        float k[8], score[BONSAI_SPLIT_HEADS];
        for (uint i = 0; i < 8; ++i) k[i] = bonsai_kv_load<KF>(key, token, kv_head, i, lane);
        for (uint h = 0; h < BONSAI_SPLIT_HEADS; ++h) {
            float partial = 0.0f;
            for (uint i = 0; i < 8; ++i) partial = fma(q[h][i], k[i], partial);
            score[h] = partial;
        }
        for (uint h = 0; h < BONSAI_SPLIT_HEADS; ++h) score[h] = simd_sum(score[h]) * (1.0f / 16.0f);
        float v[8];
        for (uint i = 0; i < 8; ++i) v[i] = bonsai_kv_load<VF>(value, token, kv_head, i, lane);
        for (uint h = 0; h < BONSAI_SPLIT_HEADS; ++h) {
            const float next_maximum = max(maximum[h], score[h]);
            const float old_scale = exp(maximum[h] - next_maximum);
            const float weight = exp(score[h] - next_maximum);
            denominator[h] = denominator[h] * old_scale + weight;
            for (uint i = 0; i < 8; ++i) acc[h][i] = fma(weight, v[i], acc[h][i] * old_scale);
            maximum[h] = next_maximum;
        }
    }
    for (uint h = 0; h < BONSAI_SPLIT_HEADS; ++h) {
        device float *out = partials + (ulong(first_head + h) * splits + split) * 258;
        if (lane == 0) { out[0] = maximum[h]; out[1] = denominator[h]; }
        for (uint i = 0; i < 8; ++i) out[2 + lane + i * 32] = acc[h][i];
    }
}

kernel void bo_attn_split(
    device const float *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device float *partials [[buffer(3)]],
    constant uint &prefix [[buffer(4)]], constant uint &splits [[buffer(5)]],
    constant uint &k_format [[buffer(6)]], constant uint &v_format [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    BONSAI_KV_DISPATCH(bonsai_attn_split_impl, k_format, v_format,
        query, key, value, partials, prefix, splits, group, lane);
}

kernel void bo_attn_reduce(
    device const float *partials [[buffer(0)]], device const float *gate [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &splits [[buffer(3)]],
    constant uint &gated [[buffer(4)]], uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    float maximum = -INFINITY;
    for (uint i = 0; i < splits; ++i) maximum = max(maximum, partials[(ulong(head) * splits + i) * 258]);
    float denominator = 0.0f, numerator = 0.0f;
    for (uint i = 0; i < splits; ++i) {
        device const float *part = partials + (ulong(head) * splits + i) * 258;
        const float scale = exp(part[0] - maximum);
        denominator = fma(part[1], scale, denominator);
        numerator = fma(part[2 + tid], scale, numerator);
    }
    const uint index = head * 256 + tid;
    output[index] = (numerator / denominator) * (gated ? bonsai_sigmoid(gate[index]) : 1.0f);
}

// Eight causal queries share a threadgroup. Each SIMD owns one query, keeping
// Q and online-softmax accumulation in F32 registers. Neighboring SIMDs walk
// the same 64-key tiles, enabling cache reuse without a score matrix. This uses
// the tiled online-softmax structure of Prism's fa.metal (MIT).
template <uint KF, uint VF>
static inline void bonsai_attn_block_impl(
    device const float *query, device const uchar *key, device const uchar *value,
    device const float *gate, device float *output, uint position, uint tokens, uint gated,
    uint group, uint simd, uint lane
) {
    const uint head = group % 24;
    const uint row = (group / 24) * 8 + simd;
    const bool active = row < tokens;
    const uint kv_head = head / 6;
    float q[8], acc[8];
    for (uint i = 0; i < 8; ++i) {
        q[i] = active ? query[(ulong(row) * 24 + head) * 256 + lane + i * 32] : 0.0f;
        acc[i] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    const uint prefix = active ? position + row + 1 : 0;
    for (uint tile = 0; tile < prefix; tile += 64) {
        const uint end = min(tile + 64, prefix);
        for (uint token = tile; token < end; ++token) {
            float score = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                score = fma(q[i], bonsai_kv_load<KF>(key, token, kv_head, i, lane), score);
            }
            score = simd_sum(score) * (1.0f / 16.0f);
            const float next_maximum = max(maximum, score);
            const float old_scale = exp(maximum - next_maximum);
            const float weight = exp(score - next_maximum);
            denominator = fma(denominator, old_scale, weight);
            for (uint i = 0; i < 8; ++i) {
                acc[i] = fma(weight, bonsai_kv_load<VF>(value, token, kv_head, i, lane),
                    acc[i] * old_scale);
            }
            maximum = next_maximum;
        }
    }
    if (active) {
        const ulong base = (ulong(row) * 24 + head) * 256 + lane;
        for (uint i = 0; i < 8; ++i) {
            const ulong index = base + i * 32;
            output[index] = (acc[i] / denominator)
                * (gated ? bonsai_sigmoid(gate[index]) : 1.0f);
        }
    }
}

kernel void bo_attn_block(
    device const float *query [[buffer(0)]], device const uchar *key [[buffer(1)]],
    device const uchar *value [[buffer(2)]], device const float *gate [[buffer(3)]],
    device float *output [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &gated [[buffer(7)]],
    constant uint &k_format [[buffer(8)]], constant uint &v_format [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    BONSAI_KV_DISPATCH(bonsai_attn_block_impl, k_format, v_format,
        query, key, value, gate, output, position, tokens, gated, group, simd, lane);
}

#if __METAL_VERSION__ >= 400
// Eight-query/64-key blocked online softmax follows Prism's fa.metal at
// 0781925904391351963d499cb32cd735849b06a5 (MIT). Operand precision is explicit;
// softmax statistics and accumulation stay F32 in every variant. No context-sized
// score matrix or cooperative-fragment lane layout assumptions.
template<typename QueryTensor, typename Probability>
static inline void bonsai_attn_tensor_impl(
    QueryTensor q, device half *key, device half *value,
    device const float *gate, device float *output, uint position, uint tokens,
    uint gated, uint group, uint tid, threadgroup Probability *weights, threadgroup float *scores,
    threadgroup float *result, threadgroup float *maximum, threadgroup float *denominator
) {
    constexpr int nq = 8, nk = 64, dimension = 256;
    const uint head = group % 24, first = (group / 24) * nq, kv_head = head / 6;
    const uint live = min(uint(nq), tokens - first), end = position + first + live;
    const uint row = tid / 16, lane = tid % 16;
    const bool active = row < live;
    if (tid < nq) { maximum[tid] = -INFINITY; denominator[tid] = 0.0f; }
    for (uint i = tid; i < nq * dimension; i += 128) result[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto score_tile = tensor(scores, dextents<int32_t, 2>(nk, nq));
    auto result_tile = tensor(result, dextents<int32_t, 2>(dimension, nq));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, nk, dimension, false, true, false),
        execution_simdgroups<4>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, dimension, static_cast<int>(dynamic_extent),
            false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> pv;

    for (uint start = 0; start < end; start += nk) {
        const uint count = min(uint(nk), end - start);
        auto k = tensor(key + ulong(start) * 1024 + kv_head * dimension,
            dextents<int32_t, 2>(dimension, count), array<int, 2>({1, 1024}));
        auto products = qk.get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
        qk.run(q, k, products);
        products.store(score_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float peak = -INFINITY;
        for (uint column = lane; column < nk; column += 16) {
            const bool visible = active && start + column <= position + first + row;
            const uint i = row * nk + column;
            const float score = visible ? scores[i] * (1.0f / 16.0f) : -INFINITY;
            scores[i] = score;
            peak = max(peak, score);
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) peak = max(peak, simd_shuffle_xor(peak, mask));
        const float next_maximum = active ? max(maximum[row], peak) : 0.0f;
        const float old_scale = active ? exp(maximum[row] - next_maximum) : 0.0f;
        float sum = 0.0f;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float weight = exp(scores[i] - next_maximum);
            weights[i] = Probability(weight);
            sum += weight;
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) sum += simd_shuffle_xor(sum, mask);
        for (uint column = lane; column < dimension; column += 16) {
            result[row * dimension + column] *= old_scale;
        }
        if (lane == 0) {
            maximum[row] = next_maximum;
            denominator[row] = fma(denominator[row], old_scale, sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto probabilities = tensor(weights, dextents<int32_t, 2>(count, live),
            array<int, 2>({1, nk}));
        auto v = tensor(value + ulong(start) * 1024 + kv_head * dimension,
            dextents<int32_t, 2>(dimension, count), array<int, 2>({1, 1024}));
        auto accumulated = pv.get_destination_cooperative_tensor<decltype(probabilities), decltype(v), float>();
        accumulated.load(result_tile);
        pv.run(probabilities, v, accumulated);
        accumulated.store(result_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (uint column = lane; column < dimension; column += 16) {
            const ulong i = (ulong(first + row) * 24 + head) * dimension + column;
            output[i] = result[row * dimension + column] / denominator[row]
                * (gated ? bonsai_sigmoid(gate[i]) : 1.0f);
        }
    }
}

// F32 Q can be read directly with its original stride. Staging it instead costs
// 8 KiB of shared memory per group and was slower in the controlled comparison.
kernel void bo_attn_tensor(
    device float *query [[buffer(0)]], device half *key [[buffer(1)]],
    device half *value [[buffer(2)]], device const float *gate [[buffer(3)]],
    device float *output [[buffer(4)]], constant uint &position [[buffer(5)]],
    constant uint &tokens [[buffer(6)]], constant uint &gated [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    const uint first = (group / 24) * 8, head = group % 24, live = min(8u, tokens - first);
    auto q = tensor(query + ulong(first) * 6144 + head * 256,
        dextents<int32_t, 2>(256, live), array<int, 2>({1, 6144}));
    threadgroup float scores[8 * 64], result[8 * 256], maximum[8], denominator[8];
    bonsai_attn_tensor_impl(q, key, value, gate, output, position, tokens, gated,
        group, tid, scores, scores, result, maximum, denominator);
}

// Decode (single query row) attention on the tensor units for F16 caches.
// One threadgroup per (KV head, `split_tokens`-token split): the Q tile
// is the six query heads that share the KV head at this position (rows 6-7
// idle), so each K/V tile is loaded once for all six heads and the scores
// and P.V products are 8x8 matrix ops instead of per-lane FMA chains. No
// causal mask: every token in the split is visible to the one query row.
// Emits the same (max, denominator, unnormalized numerator) partial record
// per (query head, split) as bo_attn_split, for bo_attn_reduce. `split_tokens`
// is a multiple of 64 chosen by the host: small at short prefixes so enough
// threadgroups exist to fill the GPU, larger at long prefixes to amortize the
// per-split partial record.
kernel void bo_attn_split_tensor(
    device float *query [[buffer(0)]], device half *key [[buffer(1)]],
    device half *value [[buffer(2)]], device float *partials [[buffer(3)]],
    constant uint &prefix [[buffer(4)]], constant uint &splits [[buffer(5)]],
    constant uint &split_tokens [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    constexpr int nq = 8, nk = 64, dimension = 256, live = 6;
    const uint kv_head = group / splits, split = group % splits;
    const uint begin = split * split_tokens, end = min(begin + split_tokens, prefix);
    const uint row = tid / 16, lane = tid % 16;
    const bool active = row < live;
    threadgroup float scores[nq * nk], result[nq * dimension], maximum[nq], denominator[nq];
    if (tid < nq) { maximum[tid] = -INFINITY; denominator[tid] = 0.0f; }
    for (uint i = tid; i < nq * dimension; i += 128) result[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto q = tensor(query + kv_head * 6 * dimension,
        dextents<int32_t, 2>(dimension, live), array<int, 2>({1, dimension}));
    auto score_tile = tensor(scores, dextents<int32_t, 2>(nk, nq));
    auto result_tile = tensor(result, dextents<int32_t, 2>(dimension, nq));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, nk, dimension, false, true, false),
        execution_simdgroups<4>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, dimension, static_cast<int>(dynamic_extent),
            false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> pv;

    for (uint start = begin; start < end; start += nk) {
        const uint count = min(uint(nk), end - start);
        auto k = tensor(key + ulong(start) * 1024 + kv_head * dimension,
            dextents<int32_t, 2>(dimension, count), array<int, 2>({1, 1024}));
        auto products = qk.get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
        qk.run(q, k, products);
        products.store(score_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float peak = -INFINITY;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float score = active && column < count ? scores[i] * (1.0f / 16.0f) : -INFINITY;
            scores[i] = score;
            peak = max(peak, score);
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) peak = max(peak, simd_shuffle_xor(peak, mask));
        const float next_maximum = active ? max(maximum[row], peak) : 0.0f;
        const float old_scale = active ? exp(maximum[row] - next_maximum) : 0.0f;
        float sum = 0.0f;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float weight = exp(scores[i] - next_maximum);
            scores[i] = weight;
            sum += weight;
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) sum += simd_shuffle_xor(sum, mask);
        for (uint column = lane; column < dimension; column += 16) {
            result[row * dimension + column] *= old_scale;
        }
        if (lane == 0) {
            maximum[row] = next_maximum;
            denominator[row] = fma(denominator[row], old_scale, sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto probabilities = tensor(scores, dextents<int32_t, 2>(count, live),
            array<int, 2>({1, nk}));
        auto v = tensor(value + ulong(start) * 1024 + kv_head * dimension,
            dextents<int32_t, 2>(dimension, count), array<int, 2>({1, 1024}));
        auto accumulated = pv.get_destination_cooperative_tensor<decltype(probabilities), decltype(v), float>();
        accumulated.load(result_tile);
        pv.run(probabilities, v, accumulated);
        accumulated.store(result_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        device float *out = partials + (ulong(kv_head * 6 + row) * splits + split) * 258;
        if (lane == 0) { out[0] = maximum[row]; out[1] = denominator[row]; }
        for (uint column = lane; column < dimension; column += 16) {
            out[2 + column] = result[row * dimension + column];
        }
    }
}

// Bytes per cached token row of a quantized format.
// Dequantize `count` (<= NK) tokens of one KV head from a Q8 cache into a half
// tile of NK x 256, eight values per thread step: one 8-byte load of int8s and
// the block's F16 scale. Every load is issued before any
// conversion, so a thread has all of them in flight at once; rows past `count`
// re-read the last valid token and are never consumed by the tensor ops. Eight
// values never straddle a 32-value scale block. Each value is exactly
// bonsai_kv_load<BONSAI_KV_Q8>, rounded once to half for the tensor operands.
template <uint NK>
static inline void bonsai_kv_tile(
    device const uchar *cache, uint start, uint count, uint kv_head,
    threadgroup half *tile, uint tid
) {
    constexpr uint per_thread = NK * 32 / 128;
    uint2 codes[per_thread];
    half scales[per_thread];
    for (uint j = 0; j < per_thread; ++j) {
        const uint chunk = tid + j * 128, token = min(chunk / 32, count - 1);
        const uint first = (chunk % 32) * 8, block = kv_head * 8 + first / 32;
        device const uchar *row = cache + ulong(start + token) * BONSAI_KV_Q8_TOKEN_BYTES;
        codes[j] = *reinterpret_cast<device const uint2 *>(row + kv_head * 256 + first);
        scales[j] = reinterpret_cast<device const half *>(row + 1024)[block];
    }
    for (uint j = 0; j < per_thread; ++j) {
        const uint chunk = tid + j * 128, offset = (chunk / 32) * 256 + (chunk % 32) * 8;
        const float scale = float(scales[j]);
        const float4 low = float4(as_type<char4>(codes[j].x));
        const float4 high = float4(as_type<char4>(codes[j].y));
        threadgroup half4 *out = reinterpret_cast<threadgroup half4 *>(tile + offset);
        out[0] = half4(low * scale);
        out[1] = half4(high * scale);
    }
}

// bo_attn_split_tensor for quantized caches. The tensor operands must be half, so each
// NK-token tile is dequantized into threadgroup memory first: K, then V into
// the same storage once the scores exist, its loads overlapping the softmax
// (two 32-token tiles would exceed 32 KiB of threadgroup memory, and tensor
// tile widths must be multiples of 16; two 16-token tiles measured slower). The P.V
// accumulator stays in registers (a cooperative tensor) for the whole split and
// is rescaled in place by its row's softmax correction, instead of the F16
// kernel's per-tile store/rescale/load of an 8 KiB threadgroup result; that
// round trip is what made small tiles expensive. Softmax statistics, partial
// records and the reduce are the F16 kernel's.
template <uint KF, uint VF, uint NK>
static inline void bonsai_attn_split_tensor_quantized_impl(
    device float *query, device const uchar *key, device const uchar *value,
    device float *partials, uint prefix, uint splits, uint split_tokens,
    uint group, uint tid, threadgroup half *tile, threadgroup float *scores,
    threadgroup float *correction, threadgroup float *maximum, threadgroup float *denominator
) {
    constexpr int nq = 8, nk = NK, dimension = 256, live = 6;
    // KV head fastest: a token's four head rows share the cache lines of its
    // scale block, so the four groups of one split run together and fetch
    // those lines once (5-6 % at 2K-128K tokens over split-major order).
    const uint kv_head = group % 4, split = group / 4;
    const uint begin = split * split_tokens, end = min(begin + split_tokens, prefix);
    const uint row = tid / 16, lane = tid % 16;
    const bool active = row < live;
    if (tid < nq) { maximum[tid] = -INFINITY; denominator[tid] = 0.0f; }

    auto q = tensor(query + kv_head * 6 * dimension,
        dextents<int32_t, 2>(dimension, live), array<int, 2>({1, dimension}));
    auto score_tile = tensor(scores, dextents<int32_t, 2>(nk, nq));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, nk, dimension, false, true, false),
        execution_simdgroups<4>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, dimension, static_cast<int>(dynamic_extent),
            false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> pv;
    auto probabilities_shape = tensor(scores, dextents<int32_t, 2>(nk, live), array<int, 2>({1, nk}));
    auto v_shape = tensor(tile, dextents<int32_t, 2>(dimension, nk), array<int, 2>({1, dimension}));
    auto accumulated = pv.template get_destination_cooperative_tensor<
        decltype(probabilities_shape), decltype(v_shape), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;
    }

    for (uint start = begin; start < end; start += nk) {
        const uint count = min(uint(nk), end - start);
        bonsai_kv_tile<NK>(key, start, count, kv_head, tile, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto k = tensor(tile, dextents<int32_t, 2>(dimension, count),
            array<int, 2>({1, dimension}));
        auto products = qk.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
        qk.run(q, k, products);
        products.store(score_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float peak = -INFINITY;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float score = active && column < count ? scores[i] * (1.0f / 16.0f) : -INFINITY;
            scores[i] = score;
            peak = max(peak, score);
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) peak = max(peak, simd_shuffle_xor(peak, mask));
        const float next_maximum = active ? max(maximum[row], peak) : 0.0f;
        const float old_scale = active ? exp(maximum[row] - next_maximum) : 0.0f;
        float sum = 0.0f;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float weight = exp(scores[i] - next_maximum);
            scores[i] = weight;
            sum += weight;
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) sum += simd_shuffle_xor(sum, mask);
        // K is dead once the scores exist; V replaces it while the softmax runs.
        bonsai_kv_tile<NK>(value, start, count, kv_head, tile, tid);
        if (lane == 0) {
            correction[row] = old_scale;
            maximum[row] = next_maximum;
            denominator[row] = fma(denominator[row], old_scale, sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (accumulated.is_valid_element(i)) {
                accumulated[i] *= correction[accumulated.get_multidimensional_index(i)[1]];
            }
        }
        auto probabilities = tensor(scores, dextents<int32_t, 2>(count, live),
            array<int, 2>({1, nk}));
        auto v = tensor(tile, dextents<int32_t, 2>(dimension, count),
            array<int, 2>({1, dimension}));
        pv.run(probabilities, v, accumulated);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The tile is dead; reuse its storage for the F32 result.
    threadgroup float *result = reinterpret_cast<threadgroup float *>(tile);
    accumulated.store(tensor(result, dextents<int32_t, 2>(dimension, nq)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        device float *out = partials + (ulong(kv_head * 6 + row) * splits + split) * 258;
        if (lane == 0) { out[0] = maximum[row]; out[1] = denominator[row]; }
        for (uint column = lane; column < dimension; column += 16) {
            out[2 + column] = result[row * dimension + column];
        }
    }
}

#define BONSAI_ATTN_SPLIT_TENSOR_QUANTIZED(NAME, KF, VF)                              \
kernel void NAME(                                                                    \
    device float *query [[buffer(0)]], device const uchar *key [[buffer(1)]],        \
    device const uchar *value [[buffer(2)]], device float *partials [[buffer(3)]],   \
    constant uint &prefix [[buffer(4)]], constant uint &splits [[buffer(5)]],        \
    constant uint &split_tokens [[buffer(6)]],                                       \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]] \
) {                                                                                  \
    threadgroup half tile[32 * 256];                                                 \
    threadgroup float scores[8 * 32], correction[8], maximum[8], denominator[8];     \
    bonsai_attn_split_tensor_quantized_impl<KF, VF, 32>(query, key, value, partials, \
        prefix, splits, split_tokens, group, tid, tile, scores, correction, maximum, \
        denominator);                                                                \
}

// bo_attn_tensor for quantized caches: causal prefill over eight query rows of one
// query head, the decode kernel's shared dequantized tile and register
// accumulator, and bo_attn_tensor's causal mask (query row r sees cache
// positions through position + first + r).
template <uint KF, uint VF>
static inline void bonsai_attn_tensor_quantized_impl(
    device float *query, device const uchar *key, device const uchar *value,
    device const float *gate, device float *output, uint position, uint tokens, uint gated,
    uint group, uint tid, threadgroup half *tile, threadgroup float *scores,
    threadgroup float *correction, threadgroup float *maximum, threadgroup float *denominator
) {
    constexpr int nq = 8, nk = 32, dimension = 256;
    const uint first = (group / 24) * nq, head = group % 24, kv_head = head / 6;
    const uint live = min(uint(nq), tokens - first), end = position + first + live;
    const uint row = tid / 16, lane = tid % 16;
    const bool active = row < live;
    if (tid < nq) { maximum[tid] = -INFINITY; denominator[tid] = 0.0f; }

    auto q = tensor(query + ulong(first) * 6144 + head * dimension,
        dextents<int32_t, 2>(dimension, live), array<int, 2>({1, 6144}));
    auto score_tile = tensor(scores, dextents<int32_t, 2>(nk, nq));
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, nk, dimension, false, true, false),
        execution_simdgroups<4>> qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(nq, dimension, static_cast<int>(dynamic_extent),
            false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> pv;
    auto probabilities_shape = tensor(scores, dextents<int32_t, 2>(nk, live), array<int, 2>({1, nk}));
    auto v_shape = tensor(tile, dextents<int32_t, 2>(dimension, nk), array<int, 2>({1, dimension}));
    auto accumulated = pv.template get_destination_cooperative_tensor<
        decltype(probabilities_shape), decltype(v_shape), float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        if (accumulated.is_valid_element(i)) accumulated[i] = 0.0f;
    }

    for (uint start = 0; start < end; start += nk) {
        const uint count = min(uint(nk), end - start);
        bonsai_kv_tile<nk>(key, start, count, kv_head, tile, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto k = tensor(tile, dextents<int32_t, 2>(dimension, count), array<int, 2>({1, dimension}));
        auto products = qk.template get_destination_cooperative_tensor<decltype(q), decltype(k), float>();
        qk.run(q, k, products);
        products.store(score_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float peak = -INFINITY;
        for (uint column = lane; column < nk; column += 16) {
            const bool visible = active && column < count && start + column <= position + first + row;
            const uint i = row * nk + column;
            const float score = visible ? scores[i] * (1.0f / 16.0f) : -INFINITY;
            scores[i] = score;
            peak = max(peak, score);
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) peak = max(peak, simd_shuffle_xor(peak, mask));
        const float next_maximum = active ? max(maximum[row], peak) : 0.0f;
        const float old_scale = active ? exp(maximum[row] - next_maximum) : 0.0f;
        float sum = 0.0f;
        for (uint column = lane; column < nk; column += 16) {
            const uint i = row * nk + column;
            const float weight = exp(scores[i] - next_maximum);
            scores[i] = weight;
            sum += weight;
        }
        for (ushort mask = 8; mask != 0; mask >>= 1) sum += simd_shuffle_xor(sum, mask);
        // K is dead once the scores exist; V replaces it while the softmax runs.
        bonsai_kv_tile<nk>(value, start, count, kv_head, tile, tid);
        if (lane == 0) {
            correction[row] = old_scale;
            maximum[row] = next_maximum;
            denominator[row] = fma(denominator[row], old_scale, sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
            if (accumulated.is_valid_element(i)) {
                accumulated[i] *= correction[accumulated.get_multidimensional_index(i)[1]];
            }
        }
        auto probabilities = tensor(scores, dextents<int32_t, 2>(count, live), array<int, 2>({1, nk}));
        auto v = tensor(tile, dextents<int32_t, 2>(dimension, count), array<int, 2>({1, dimension}));
        pv.run(probabilities, v, accumulated);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *result = reinterpret_cast<threadgroup float *>(tile);
    accumulated.store(tensor(result, dextents<int32_t, 2>(dimension, nq)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        for (uint column = lane; column < dimension; column += 16) {
            const ulong i = (ulong(first + row) * 24 + head) * dimension + column;
            output[i] = result[row * dimension + column] / denominator[row]
                * (gated ? bonsai_sigmoid(gate[i]) : 1.0f);
        }
    }
}

#define BONSAI_ATTN_TENSOR_QUANTIZED(NAME, KF, VF)                                    \
kernel void NAME(                                                                    \
    device float *query [[buffer(0)]], device const uchar *key [[buffer(1)]],        \
    device const uchar *value [[buffer(2)]], device const float *gate [[buffer(3)]], \
    device float *output [[buffer(4)]], constant uint &position [[buffer(5)]],       \
    constant uint &tokens [[buffer(6)]], constant uint &gated [[buffer(7)]],         \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]] \
) {                                                                                  \
    threadgroup half tile[32 * 256];                                                 \
    threadgroup float scores[8 * 32], correction[8], maximum[8], denominator[8];     \
    bonsai_attn_tensor_quantized_impl<KF, VF>(query, key, value, gate, output,       \
        position, tokens, gated, group, tid, tile, scores, correction, maximum,     \
        denominator);                                                                \
}

// Tensor attention instantiations for the quantized layouts.
#define BONSAI_ATTN_QUANTIZED(SUFFIX, FORMAT)                                         \
    BONSAI_ATTN_SPLIT_TENSOR_QUANTIZED(bo_attn_split_tensor_##SUFFIX, FORMAT, FORMAT) \
    BONSAI_ATTN_TENSOR_QUANTIZED(bo_attn_tensor_##SUFFIX, FORMAT, FORMAT)
BONSAI_ATTN_QUANTIZED(q8, 1)

#endif
