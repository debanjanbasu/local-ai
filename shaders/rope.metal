#include <metal_stdlib>
using namespace metal;

/// Rotary Position Embedding (RoPE) — multi-head, NEOX pairing.
///
/// Implements HF `rope_type: "proportional"` semantics (Gemma 4 full-attention
/// layers): only the first `partial_rotary_factor * head_dim / 2` NEOX pairs
/// rotate, and their frequencies follow the FULL-head schedule
/// `1/theta^(2i/head_dim)`; the remaining pairs have inverse frequency 0
/// (identity). `partial_rotary_factor == 1.0` is standard full NEOX RoPE.
///
/// input/output shape: [num_heads * head_dim] (all heads, single position)
/// Dispatch: threads = (head_dim / 2, num_heads)
kernel void rope(
    device const half *input               [[buffer(0)]],
    device half       *output              [[buffer(1)]],
    constant uint     &head_dim            [[buffer(2)]],
    constant uint     &position            [[buffer(3)]],
    constant float    &theta               [[buffer(4)]],
    constant float    &partial_rotary_factor [[buffer(5)]],
    constant uint     &num_heads           [[buffer(6)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    uint i = gid.x;          // pair index within head (0..head_dim/2)
    uint head = gid.y;       // which head

    uint half_dim = head_dim / 2;
    if (i >= half_dim || head >= num_heads) return;

    uint base = head * head_dim;
    // NEOX half-split pairing over the full head: (i, i + half_dim).
    uint idx0 = base + i;
    uint idx1 = base + i + half_dim;

    uint rope_angles = uint(float(head_dim) * partial_rotary_factor) / 2;
    if (i >= rope_angles) {
        // Zero inverse frequency: pass the pair through unchanged.
        output[idx0] = input[idx0];
        output[idx1] = input[idx1];
        return;
    }

    float x0 = float(input[idx0]);
    float x1 = float(input[idx1]);

    float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    output[idx0] = half(x0 * cos_val - x1 * sin_val);
    output[idx1] = half(x0 * sin_val + x1 * cos_val);
}

/// M-RoPE: Multi-modal Rotary Position Embedding.
///
/// Applies RoPE to only the first `rotary_dim` dimensions (25% of head_dim),
/// split into 3 interleaved sections [s0, s1, s2] for temporal/height/width positions.
/// For text-only inference, all 3 position IDs are the same token position.
///
/// qk is modified in-place: [num_heads * head_dim]
/// Dispatch: threads = (rotary_dim / 2, num_heads)
kernel void mrope(
    device half       *qk           [[buffer(0)]],  // Q or K, in-place [num_heads * head_dim]
    constant uint     &position     [[buffer(1)]],  // token position (same for all 3 axes in text-only)
    constant float    &theta        [[buffer(2)]],  // 10000000.0
    constant uint     &head_dim     [[buffer(3)]],  // e.g. 256
    constant uint     &rotary_dim   [[buffer(4)]],  // e.g. 64 (head_dim * 0.25)
    constant uint     &num_heads    [[buffer(5)]],
    constant uint     *sections     [[buffer(6)]],  // [3] e.g. {11, 11, 10} — dim pairs per section
    uint2 tid [[thread_position_in_grid]]           // (dim_pair, head)
) {
    uint i = tid.x;          // pair index within rotary_dim
    uint head = tid.y;       // which head

    uint half_rotary = rotary_dim / 2;
    if (i >= half_rotary || head >= num_heads) return;

    uint base = head * head_dim;
    uint idx_even = base + i * 2;
    uint idx_odd  = base + i * 2 + 1;

    float x_even = float(qk[idx_even]);
    float x_odd  = float(qk[idx_odd]);

    // For interleaved M-RoPE (mrope_interleaved=true), pairs are
    // [T,H,W,T,H,W,...] so each pair i uses global frequency index i.
    // Section boundaries only matter for multimodal (different positions per section).
    float freq = 1.0f / pow(theta, float(2 * i) / float(rotary_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    qk[idx_even] = half(x_even * cos_val - x_odd * sin_val);
    qk[idx_odd]  = half(x_even * sin_val + x_odd * cos_val);
}

/// Batch NEOX-style RoPE over `seq_len` consecutive token rows starting at
/// `start_pos`. Pairing matches the single-position `rope` kernel above
/// (element `i` with `i + half_dim`, full rotation) so batched prefill /
/// verification produce the same rotation as single-token decode.
///
/// qk is modified in-place: [seq_len, num_heads, head_dim]
/// Dispatch: threads = (head_dim/2, num_heads * seq_len)
kernel void rope_batch(
    device half       *qk           [[buffer(0)]],
    constant uint     &start_pos    [[buffer(1)]],
    constant float    &theta        [[buffer(2)]],
    constant uint     &head_dim     [[buffer(3)]],
    constant uint     &num_heads    [[buffer(4)]],
    constant uint     &seq_len      [[buffer(5)]],
    constant float    &partial_rotary_factor [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint flat = tid.y;
    uint half_dim = head_dim / 2;

    if (i >= half_dim) return;

    uint seq_idx = flat % seq_len;
    uint head = flat / seq_len;

    if (head >= num_heads || seq_idx >= seq_len) return;

    // Proportional RoPE: pairs past `rope_angles` have zero inverse
    // frequency — identity, and the kernel is in-place, so just exit.
    uint rope_angles = uint(float(head_dim) * partial_rotary_factor) / 2;
    if (i >= rope_angles) return;

    uint position = start_pos + seq_idx;
    uint base = (seq_idx * num_heads + head) * head_dim;
    // NEOX half-split pairing: (i, i + half_dim).
    uint idx0 = base + i;
    uint idx1 = base + i + half_dim;

    float x0 = float(qk[idx0]);
    float x1 = float(qk[idx1]);

    float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    qk[idx0] = half(x0 * cos_val - x1 * sin_val);
    qk[idx1] = half(x0 * sin_val + x1 * cos_val);
}

/// Batched-decode RoPE: identical to `rope_batch` but each row (lane) reads its
/// own absolute position from `positions[seq_idx]` instead of `start_pos +
/// seq_idx`. This is the primitive that lets N *independent* sequences be
/// decoded in one batched forward — every lane sits at a different context
/// length, so a single scalar `start_pos` cannot express their positions.
///
/// qk layout: [seq_len(=lanes), num_heads, head_dim], modified in-place.
/// Dispatch: threads = (head_dim/2, num_heads * seq_len).
kernel void rope_batch_decode(
    device half       *qk           [[buffer(0)]],
    constant uint     *positions    [[buffer(1)]],
    constant float    &theta        [[buffer(2)]],
    constant uint     &head_dim     [[buffer(3)]],
    constant uint     &num_heads    [[buffer(4)]],
    constant uint     &seq_len      [[buffer(5)]],
    constant float    &partial_rotary_factor [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint flat = tid.y;
    uint half_dim = head_dim / 2;

    if (i >= half_dim) return;

    uint seq_idx = flat % seq_len;
    uint head = flat / seq_len;

    if (head >= num_heads || seq_idx >= seq_len) return;

    uint rope_angles = uint(float(head_dim) * partial_rotary_factor) / 2;
    if (i >= rope_angles) return;

    uint position = positions[seq_idx];
    uint base = (seq_idx * num_heads + head) * head_dim;
    uint idx0 = base + i;
    uint idx1 = base + i + half_dim;

    float x0 = float(qk[idx0]);
    float x1 = float(qk[idx1]);

    float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    qk[idx0] = half(x0 * cos_val - x1 * sin_val);
    qk[idx1] = half(x0 * sin_val + x1 * cos_val);
}

/// Batch M-RoPE: multi-modal rotary embedding for sequence of tokens.
/// qk is modified in-place: [seq_len, num_heads, head_dim]
///
/// Dispatch: threads = (rotary_dim/2, num_heads * seq_len)
kernel void mrope_batch(
    device half       *qk           [[buffer(0)]],
    constant uint     &start_pos    [[buffer(1)]],
    constant float    &theta        [[buffer(2)]],
    constant uint     &head_dim     [[buffer(3)]],
    constant uint     &rotary_dim   [[buffer(4)]],
    constant uint     &num_heads    [[buffer(5)]],
    constant uint     &seq_len      [[buffer(6)]],
    constant uint     *sections     [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint flat = tid.y;

    uint half_rotary = rotary_dim / 2;
    if (i >= half_rotary) return;

    uint seq_idx = flat % seq_len;
    uint head = flat / seq_len;

    if (head >= num_heads || seq_idx >= seq_len) return;

    uint position = start_pos + seq_idx;
    uint base = (seq_idx * num_heads + head) * head_dim;
    uint idx_even = base + i * 2;
    uint idx_odd  = base + i * 2 + 1;

    float x_even = float(qk[idx_even]);
    float x_odd  = float(qk[idx_odd]);

    // Interleaved M-RoPE: global frequency index (see mrope kernel comment).
    float freq = 1.0f / pow(theta, float(2 * i) / float(rotary_dim));
    float angle = float(position) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    qk[idx_even] = half(x_even * cos_val - x_odd * sin_val);
    qk[idx_odd]  = half(x_even * sin_val + x_odd * cos_val);
}

/// YaRN RoPE — single position, multi-head, with per-dimension frequency scaling.
///
/// Same structure as `rope` but with precomputed freq_scales and attn_factor
/// for NTK-aware interpolation of extended context lengths.
///
/// input/output shape: [num_heads * head_dim] (all heads, single position)
/// Dispatch: threads = (head_dim / 2, num_heads)
kernel void rope_yarn(
    device const half  *input               [[buffer(0)]],
    device half        *output              [[buffer(1)]],
    constant uint      &head_dim            [[buffer(2)]],
    constant uint      &position            [[buffer(3)]],
    constant float     &theta               [[buffer(4)]],
    constant float     &partial_rotary_factor [[buffer(5)]],
    constant uint      &num_heads           [[buffer(6)]],
    device const float *freq_scales         [[buffer(7)]],
    constant float     &attn_factor         [[buffer(8)]],
    uint2 gid                               [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint head = gid.y;

    uint half_dim = head_dim / 2;
    if (i >= half_dim || head >= num_heads) return;

    uint rotary_dim = uint(float(head_dim) * partial_rotary_factor);
    uint half_rotary = rotary_dim / 2;

    uint base = head * head_dim;
    uint idx_even = base + i * 2;
    uint idx_odd = base + i * 2 + 1;

    float x_even = float(input[idx_even]);
    float x_odd = float(input[idx_odd]);

    if (i < half_rotary) {
        float freq = (1.0f / pow(theta, float(2 * i) / float(head_dim))) * freq_scales[i];
        float angle = float(position) * freq;
        float cos_val = cos(angle) * attn_factor;
        float sin_val = sin(angle) * attn_factor;

        output[idx_even] = half(x_even * cos_val - x_odd * sin_val);
        output[idx_odd]  = half(x_even * sin_val + x_odd * cos_val);
    } else {
        output[idx_even] = half(x_even);
        output[idx_odd]  = half(x_odd);
    }
}

/// Batch YaRN RoPE: apply YaRN-scaled rotary embeddings to [seq_len, num_heads, head_dim].
/// Each row i uses position (start_pos + i) for frequency computation.
/// qk is modified in-place.
///
/// Dispatch: threads = (head_dim/2, num_heads * seq_len)
kernel void rope_batch_yarn(
    device half        *qk                  [[buffer(0)]],
    constant uint      &start_pos           [[buffer(1)]],
    constant float     &theta               [[buffer(2)]],
    constant uint      &head_dim            [[buffer(3)]],
    constant uint      &num_heads           [[buffer(4)]],
    constant uint      &seq_len             [[buffer(5)]],
    device const float *freq_scales         [[buffer(6)]],
    constant float     &attn_factor         [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint i = tid.x;
    uint flat = tid.y;
    uint half_dim = head_dim / 2;

    if (i >= half_dim) return;

    uint seq_idx = flat % seq_len;
    uint head = flat / seq_len;

    if (head >= num_heads || seq_idx >= seq_len) return;

    uint position = start_pos + seq_idx;
    uint base_off = (seq_idx * num_heads + head) * head_dim;
    uint idx_even = base_off + i * 2;
    uint idx_odd  = base_off + i * 2 + 1;

    float x_even = float(qk[idx_even]);
    float x_odd  = float(qk[idx_odd]);

    float freq = (1.0f / pow(theta, float(2 * i) / float(head_dim))) * freq_scales[i];
    float angle = float(position) * freq;
    float cos_val = cos(angle) * attn_factor;
    float sin_val = sin(angle) * attn_factor;

    qk[idx_even] = half(x_even * cos_val - x_odd * sin_val);
    qk[idx_odd]  = half(x_even * sin_val + x_odd * cos_val);
}