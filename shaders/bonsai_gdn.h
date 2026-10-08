#ifndef BONSAI_GDN_H
#define BONSAI_GDN_H

#include <metal_stdlib>
using namespace metal;

// Round an F32 register to the state's storage type (nearest even).
static inline void bonsai_store_state(device float &state, float value) { state = value; }
// F16 saturates rather than overflowing to infinity.
static inline void bonsai_store_state(device half &state, float value) {
    state = half(clamp(value, -65504.0f, 65504.0f));
}
static inline void bonsai_store_state(device bfloat &state, float value) { state = bfloat(value); }

// Independent value rows share Q/K and scalar gates within one SIMD. Retain
// the original per-row FMA and SIMD reduction order; the state is stored in
// whatever element type the accessors name and computed in F32 registers.
// A tile never crosses a head: each candidate row count divides 128.
// Rank-one tensor accessors keep the equations independent of the storage
// they read. Initial/final state may alias only for in-place use.
template<uint rows, typename Qkv, typename Decay, typename Beta,
         typename State, typename Output, typename FinalState>
static inline void bonsai_gdn_rows(
    Qkv qkv, Decay decay, Beta beta, State initial_state,
    Output output, FinalState final_state, uint tokens, uint group, uint lane
) {
    const uint first_row = group * rows, head = first_row / 128, key_group = head % 16;
    float values[rows][4];
    for (uint r = 0; r < rows; ++r) {
        for (uint i = 0; i < 4; ++i) {
            values[r][i] = float(initial_state[(first_row + r) * 128 + lane + i * 32]);
        }
    }
    for (uint token = 0; token < tokens; ++token) {
        const uint token_offset = token * 10240;
        const float d = decay[token * 48 + head], b = beta[token * 48 + head];
        float q[4], k[4], prediction[rows] = {};
        for (uint i = 0; i < 4; ++i) {
            const uint column = key_group * 128 + lane + i * 32;
            q[i] = qkv[token_offset + column];
            k[i] = qkv[token_offset + 2048 + column];
            for (uint r = 0; r < rows; ++r) {
                values[r][i] *= d;
                prediction[r] = fma(values[r][i], k[i], prediction[r]);
            }
        }
        for (uint r = 0; r < rows; ++r) {
            const float correction = (qkv[token_offset + 4096 + first_row + r] - simd_sum(prediction[r])) * b;
            float result = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                values[r][i] = fma(k[i], correction, values[r][i]);
                result = fma(values[r][i], q[i], result);
            }
            result = simd_sum(result);
            if (lane == 0) output[token * 6144 + first_row + r] = result * 0.08838834764831845f;
        }
    }
    for (uint r = 0; r < rows; ++r) {
        for (uint i = 0; i < 4; ++i) {
            bonsai_store_state(final_state[(first_row + r) * 128 + lane + i * 32], values[r][i]);
        }
    }
}

#endif
