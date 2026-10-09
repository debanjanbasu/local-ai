#ifndef BONSAI_MIXER_H
#define BONSAI_MIXER_H

#include <metal_stdlib>

// Qwen35 graph equations: PrismML-Eng/llama.cpp, revision
// 9a9394a895b96003ca842a6041cb28ac49a108f7 (MIT). See THIRD_PARTY_NOTICES.md.
static inline float bonsai_sigmoid(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
static inline float bonsai_silu(float x) { return x * bonsai_sigmoid(x); }

template<metal::uint threads>
static inline float bonsai_group_sum(
    float value, metal::uint tid, threadgroup float *partial
) {
    const float sum = metal::simd_sum(value);
    if (tid % 32 == 0) partial[tid / 32] = sum;
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float result = 0.0f;
    for (metal::uint i = 0; i < threads / 32; ++i) result += partial[i];
    return result;
}

// Indexable arguments may be native device pointers or rank-one tensor handles.
template<typename Input, typename Weights, typename Output>
static inline void bonsai_rms_impl(
    Input input, Weights weights, Output output, metal::uint dimension,
    metal::uint rows, metal::uint stride, float epsilon, metal::uint row,
    metal::uint tid, threadgroup float *partial
) {
    if (row >= rows) return;
    float square_sum = 0.0f;
    for (metal::uint i = tid; i < dimension; i += 128) {
        const float value = input[metal::ulong(row) * stride + i];
        square_sum = metal::fma(value, value, square_sum);
    }
    const float sum = bonsai_group_sum<128>(square_sum, tid, partial);
    const float inverse = metal::rsqrt(sum / float(dimension) + epsilon);
    for (metal::uint i = tid; i < dimension; i += 128) {
        const metal::ulong index = metal::ulong(row) * stride + i;
        output[index] = input[index] * inverse * weights[i];
    }
}

template<typename Weights, typename Input, typename Output>
static inline void bonsai_bf16_mv_impl(
    Weights weights, Input input, Output output, metal::uint rows,
    metal::uint columns, metal::uint group, metal::uint lane
) {
    const metal::uint row = group % rows, token = group / rows;
    const metal::ulong weight_row = metal::ulong(row) * columns;
    const metal::ulong input_row = metal::ulong(token) * columns;
    float sum = 0.0f;
    metal::uint column = lane;
    // Issue eight iterations' loads before their fused multiply-adds, which
    // stay in column order. A 48x5120 row-per-SIMD-group projection was bound
    // by one load latency per iteration: 45 us each on an M4 Pro.
    for (; column + 7 * 32 < columns; column += 8 * 32) {
        float weight[8], value[8];
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) {
            weight[i] = as_type<float>(metal::uint(weights[weight_row + column + i * 32]) << 16);
            value[i] = input[input_row + column + i * 32];
        }
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) sum = metal::fma(weight[i], value[i], sum);
    }
    for (; column < columns; column += 32) {
        const float weight = as_type<float>(metal::uint(weights[weight_row + column]) << 16);
        sum = metal::fma(weight, input[input_row + column], sum);
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) output[group] = sum;
}

// `bonsai_bf16_mv_impl` for up to `group_tokens` activation rows per SIMD
// group: each weight is loaded once and multiplied into one F32 chain per
// row, every chain in the column order of `bonsai_bf16_mv_impl`, so each
// output is bitwise that kernel's whatever the row count. `token_group`
// selects rows [token_group * group_tokens, ...) of `tokens`; rows past the
// end read a real row and are not stored.
template<metal::uint group_tokens, typename Weights, typename Input, typename Output>
static inline void bonsai_bf16_mv_tokens_impl(
    Weights weights, Input input, Output output, metal::uint rows, metal::uint columns,
    metal::uint tokens, metal::uint row, metal::uint token_group, metal::uint lane
) {
    const metal::uint first_token = token_group * group_tokens;
    const metal::ulong weight_row = metal::ulong(row) * columns;
    metal::ulong input_row[group_tokens];
    float sum[group_tokens];
    #pragma clang loop unroll(full)
    for (metal::uint t = 0; t < group_tokens; ++t) {
        input_row[t] = metal::ulong(metal::min(first_token + t, tokens - 1)) * columns;
        sum[t] = 0.0f;
    }
    metal::uint column = lane;
    for (; column + 7 * 32 < columns; column += 8 * 32) {
        float weight[8];
        #pragma clang loop unroll(full)
        for (metal::uint i = 0; i < 8; ++i) {
            weight[i] = as_type<float>(metal::uint(weights[weight_row + column + i * 32]) << 16);
        }
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < group_tokens; ++t) {
            float value[8];
            #pragma clang loop unroll(full)
            for (metal::uint i = 0; i < 8; ++i) value[i] = input[input_row[t] + column + i * 32];
            #pragma clang loop unroll(full)
            for (metal::uint i = 0; i < 8; ++i) sum[t] = metal::fma(weight[i], value[i], sum[t]);
        }
    }
    for (; column < columns; column += 32) {
        const float weight = as_type<float>(metal::uint(weights[weight_row + column]) << 16);
        #pragma clang loop unroll(full)
        for (metal::uint t = 0; t < group_tokens; ++t) {
            sum[t] = metal::fma(weight, input[input_row[t] + column], sum[t]);
        }
    }
    #pragma clang loop unroll(full)
    for (metal::uint t = 0; t < group_tokens; ++t) {
        const float total = metal::simd_sum(sum[t]);
        if (lane == 0 && first_token + t < tokens) {
            output[metal::ulong(first_token + t) * rows + row] = total;
        }
    }
}

template<typename Input, typename Weights, typename History, typename Output,
         typename FinalHistory>
static inline void bonsai_conv_impl(
    Input input, Weights weights, History history, Output output,
    FinalHistory final_history, metal::uint tokens, metal::uint channel
) {
    if (channel >= 10240) return;
    const metal::uint h = channel * 3, w = channel * 4;
    float h0 = history[h], h1 = history[h + 1], h2 = history[h + 2];
    for (metal::uint token = 0; token < tokens; ++token) {
        const metal::uint index = token * 10240 + channel;
        float sum = weights[w] * h0;
        sum = metal::fma(weights[w + 1], h1, sum);
        sum = metal::fma(weights[w + 2], h2, sum);
        sum = metal::fma(weights[w + 3], input[index], sum);
        h0 = h1;
        h1 = h2;
        h2 = input[index];
        output[index] = bonsai_silu(sum);
    }
    final_history[h] = h0;
    final_history[h + 1] = h1;
    final_history[h + 2] = h2;
}

template<typename Qkv, typename Output>
static inline void bonsai_l2_qk_impl(
    Qkv qkv, Output output, float epsilon, metal::uint head,
    metal::uint tid, threadgroup float *partial
) {
    const metal::uint index = (head / 32) * 10240 + (head % 32) * 128 + tid;
    const float value = qkv[index];
    const float sum = bonsai_group_sum<128>(value * value, tid, partial);
    output[index] = value / metal::max(metal::sqrt(sum), epsilon);
}

template<typename A, typename Alpha, typename Dt, typename RawBeta,
         typename Decay, typename Beta>
static inline void bonsai_decay_impl(
    A a, Alpha alpha, Dt dt, RawBeta raw_beta, Decay decay, Beta beta,
    metal::uint tokens, metal::uint head
) {
    if (head >= 48 * tokens) return;
    const float x = alpha[head] + dt[head % 48];
    const float softplus = x > 20.0f ? x : metal::log(1.0f + metal::exp(x));
    decay[head] = metal::exp(a[head % 48] * softplus);
    beta[head] = bonsai_sigmoid(raw_beta[head]);
}

template<typename Input, typename Gate, typename Weights, typename Output>
static inline void bonsai_gdn_post_impl(
    Input input, Gate gate, Weights weights, Output output, float epsilon,
    metal::uint head, metal::uint tid, threadgroup float *partial
) {
    const metal::uint index = head * 128 + tid;
    const float value = input[index];
    const float sum = bonsai_group_sum<128>(value * value, tid, partial);
    const metal::uint grouped_head =
        (head / 48) * 48 + (head % 16) * 3 + (head % 48) / 16;
    output[grouped_head * 128 + tid] = value * metal::rsqrt(sum / 128.0f + epsilon)
        * weights[tid] * bonsai_silu(gate[index]);
}

#endif
