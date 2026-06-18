#include <metal_stdlib>
using namespace metal;

/// Dequantize 4-bit packed KV segment to FP16.
/// Layout: 2 values per byte, low nibble first.
/// Formula: value = zero_point + quantized_val * scale
///
/// Params buffer layout (passed as constant):
///   uint bits       (offset 0)
///   float zero_point (offset 4)
///   float scale      (offset 8)
///   uint num_elements (offset 12)

struct DequantParams {
    uint bits;
    float zero_point;
    float scale;
    uint num_elements;
};

kernel void dequantize_kv_4bit(
    device const uchar* packed [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant DequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    uint byte_idx = tid / 2;
    uchar raw = packed[byte_idx];
    uchar q = (tid % 2 == 0) ? (raw & 0x0F) : ((raw >> 4) & 0x0F);

    float val = params.zero_point + float(q) * params.scale;
    output[tid] = half(val);
}

kernel void dequantize_kv_6bit(
    device const uchar* packed [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant DequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    uint group = tid / 4;
    uint pos = tid % 4;
    uint base = group * 3;

    uchar q;
    if (pos == 0) {
        q = packed[base] & 0x3F;
    } else if (pos == 1) {
        q = ((packed[base] >> 6) & 0x03) | ((packed[base + 1] & 0x0F) << 2);
    } else if (pos == 2) {
        q = ((packed[base + 1] >> 4) & 0x0F) | ((packed[base + 2] & 0x03) << 4);
    } else {
        q = (packed[base + 2] >> 2) & 0x3F;
    }

    float val = params.zero_point + float(q) * params.scale;
    output[tid] = half(val);
}

kernel void dequantize_kv_2bit(
    device const uchar* packed [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant DequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    uint byte_idx = tid / 4;
    uint pos = tid % 4;
    uchar raw = packed[byte_idx];
    uchar q = (raw >> (pos * 2)) & 0x03;

    float val = params.zero_point + float(q) * params.scale;
    output[tid] = half(val);
}

kernel void dequantize_kv_3bit(
    device const uchar* packed [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant DequantParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    // 8 values per 3 bytes
    uint group = tid / 8;
    uint pos = tid % 8;
    uint base = group * 3;
    uint packed24 = uint(packed[base]) | (uint(packed[base + 1]) << 8) | (uint(packed[base + 2]) << 16);
    uchar q = (packed24 >> (pos * 3)) & 0x07;

    float val = params.zero_point + float(q) * params.scale;
    output[tid] = half(val);
}

/// Dequantize a TurboQuant KV cache (arXiv:2504.19874) to FP16.
///
/// Encode (CPU) stored, per (token, head) vector x of length head_dim d:
///   y = FWHT(sign ⊙ (x / ‖x‖))        — randomized Hadamard rotation
///   codes[i] = argmin_l |y_i - levels[l]|  — Lloyd–Max nearest level
///   norm = ‖x‖ (f16)
/// This kernel inverts it on the fly:
///   ŷ_i = levels[codes_i]              — codebook lookup (levels pre-scaled by 1/√d)
///   x̂ = sign ⊙ FWHT(ŷ) · ‖x‖          — inverse rotation (FWHT is an involution)
///
/// One threadgroup per (token, head); head_dim threads cooperate on an
/// in-place FWHT in threadgroup memory (head_dim ≤ 512, power of two).
///
/// Layout:
///   packed : uchar,  [num_tokens * n_kv_heads * code_bytes]  LSB-first bitstream
///   norms  : half,   [num_tokens * n_kv_heads]
///   levels : float,  [1 << bits]   (pre-scaled by 1/√d on the host)
///   signs  : float,  [head_dim]    (±1 rotation sign flips)
///   output : half,   [num_tokens * n_kv_heads * head_dim]
struct TqDequantParams {
    uint head_dim;   // power of two, == threads per threadgroup
    uint n_kv_heads;
    uint num_tokens;
    uint bits;       // 2, 3, or 4
    uint code_bytes; // head_dim * bits / 8
    uint group_offset; // first (token, head) group to decode; lets SWA layers
                       // dequantize only the active window into absolute slots
    uint ring_capacity; // 0 = absolute storage; >0 = sliding-window ring: the
                        // source (token,head) slot wraps at this many tokens
                        // while the FP16 output stays at the logical slot.
};

kernel void dequantize_kv_turboquant(
    device const uchar* packed [[buffer(0)]],
    device const half* norms [[buffer(1)]],
    device const float* levels [[buffer(2)]],
    device const float* signs [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant TqDequantParams& p [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[512];
    const uint d = p.head_dim;

    // Dispatched threadgroups are relative to the active window; `g` is the
    // logical (token, head) slot so the FP16 output addresses the same absolute
    // positions the full-prefix decode would have.
    uint g = group_id + p.group_offset;

    // The packed codes / norms live at the physical slot, which
    // wraps within `ring_capacity` tokens for sliding-window layers (the FP16
    // output `g` stays logical so attention indexing is unchanged).
    uint src_g = g;
    if (p.ring_capacity != 0u) {
        uint token = g / p.n_kv_heads;
        uint head = g - token * p.n_kv_heads;
        src_g = (token % p.ring_capacity) * p.n_kv_heads + head;
    }

    // Unpack this lane's code from the LSB-first bitstream. A code can
    // straddle a byte boundary (bits == 3); the second byte is always inside
    // this head's code_bytes region, never out of bounds.
    uint bit_off = lane * p.bits;
    uint byte_idx = src_g * p.code_bytes + (bit_off >> 3);
    uint shift = bit_off & 7u;
    uint v = uint(packed[byte_idx]);
    if (shift + p.bits > 8u) {
        v |= uint(packed[byte_idx + 1u]) << 8u;
    }
    uint code = (v >> shift) & ((1u << p.bits) - 1u);

    sh[lane] = levels[code];

    // In-place FWHT butterflies (unnormalized; the 1/√d folds into the final
    // multiply). For pair (j, j+len): new[j] = a+b, new[j+len] = a-b.
    for (uint len = 1; len < d; len <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float self_v = sh[lane];
        float other_v = sh[lane ^ len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[lane] = (lane & len) ? (other_v - self_v) : (self_v + other_v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = float(norms[src_g]);
    float inv_sqrt_d = rsqrt(float(d));
    float val = sh[lane] * inv_sqrt_d * signs[lane] * norm;
    output[g * d + lane] = half(val);
}

/// Encode an FP16 KV segment into a TurboQuant cache (the GPU inverse of
/// `dequantize_kv_turboquant`), eliminating the CPU round-trip on the KV write
/// path. Per (token, head) vector x of length head_dim d:
///   norm = ‖x‖                                    (stored as f16)
///   y    = (1/√d)·FWHT(sign ⊙ (x / norm))         — randomized Hadamard rotation
///   code[i] = argmin_l |y_i - levels[l]|          — Lloyd–Max nearest level
/// then the d codes are packed LSB-first into code_bytes, matching the bit
/// layout the decode kernel reads.
///
/// One threadgroup per (token, head); head_dim threads cooperate on the norm
/// reduction and the in-place FWHT (head_dim ≤ 512, power of two). Input rows
/// are read relative to element 0; packed codes and norms are written to the
/// absolute slot `group_id + group_offset`, so a single-token decode write
/// lands at `position * n_kv_heads` exactly like the CPU `write_kv`.
struct TqEncodeParams {
    uint head_dim;     // power of two, == threads per threadgroup
    uint n_kv_heads;
    uint num_tokens;
    uint bits;         // 2, 3, or 4
    uint code_bytes;   // head_dim * bits / 8
    uint group_offset; // first absolute (token, head) slot to write
    uint ring_capacity; // 0 = absolute storage; >0 = sliding-window ring: the
                        // logical token wraps at this many tokens before write.
};

kernel void encode_kv_turboquant(
    device const half* input [[buffer(0)]],
    device const float* levels [[buffer(1)]],
    device const float* signs [[buffer(2)]],
    device uchar* packed [[buffer(3)]],
    device half* norms [[buffer(4)]],
    constant TqEncodeParams& p [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[512];
    threadgroup float red[512];
    threadgroup uint code_sh[512];
    const uint d = p.head_dim;

    // `group_id` indexes the freshly projected input rows (relative to 0);
    // `g` is the cache slot the codes/norm are written to. For a sliding-window
    // ring (ring_capacity != 0) the logical token wraps within `ring_capacity`
    // rows; `group_offset` is `position * n_kv_heads`, so the base token is
    // recovered as `group_offset / n_kv_heads`.
    uint g;
    if (p.ring_capacity != 0u) {
        uint rel_token = group_id / p.n_kv_heads;
        uint head = group_id - rel_token * p.n_kv_heads;
        uint logical_token = p.group_offset / p.n_kv_heads + rel_token;
        g = (logical_token % p.ring_capacity) * p.n_kv_heads + head;
    } else {
        g = group_id + p.group_offset;
    }
    float xv = float(input[group_id * d + lane]);

    // norm = sqrt(Σ xv²) via tree reduction over the threadgroup.
    red[lane] = xv * xv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = d >> 1; stride > 0u; stride >>= 1) {
        if (lane < stride) {
            red[lane] += red[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(red[0]);
    float inv = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

    // Rotate: sign ⊙ (x / norm), then the unnormalized FWHT butterflies; the
    // 1/√d normalization folds into the post-butterfly multiply below.
    sh[lane] = xv * inv * signs[lane];
    for (uint len = 1u; len < d; len <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float self_v = sh[lane];
        float other_v = sh[lane ^ len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[lane] = (lane & len) ? (other_v - self_v) : (self_v + other_v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = sh[lane] * rsqrt(float(d));

    // Nearest reconstruction level (levels are pre-scaled by 1/√d on the host).
    // Strict `<` with increasing index reproduces the CPU's lowest-index tie
    // break exactly.
    uint k = 1u << p.bits;
    uint best = 0u;
    float bd = INFINITY;
    for (uint i = 0u; i < k; ++i) {
        float dd = fabs(y - levels[i]);
        if (dd < bd) {
            bd = dd;
            best = i;
        }
    }
    code_sh[lane] = best;
    if (lane == 0u) {
        norms[g] = half(norm);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // LSB-first bit packing, one thread per output byte (race-free: each byte
    // is written by exactly one thread). Byte b gathers bits [b*8, b*8+8) from
    // the per-coordinate codes in threadgroup memory.
    if (lane < p.code_bytes) {
        uint byte_val = 0u;
        for (uint bitpos = 0u; bitpos < 8u; ++bitpos) {
            uint global_bit = lane * 8u + bitpos;
            uint code_idx = global_bit / p.bits;
            if (code_idx >= d) {
                break;
            }
            uint within = global_bit - code_idx * p.bits;
            uint bit = (code_sh[code_idx] >> within) & 1u;
            byte_val |= bit << bitpos;
        }
        packed[g * p.code_bytes + lane] = uchar(byte_val);
    }
}

/// Batched per-lane TurboQuant encode + scatter (one new token per lane into a
/// unified by-lane pool). Identical per-vector math as `encode_kv_turboquant`,
/// but each (lane, head) vector is written to the lane's own pool slot
/// `(lane*lane_capacity + positions[lane]) * n_kv_heads + head` instead of a
/// shared absolute offset — the turbo analogue of `write_kv_cache_decode`. One
/// threadgroup per (lane, head); input is `[n_lanes, n_kv_heads*head_dim]` f16.
struct TqEncodeBatchedParams {
    uint head_dim;      // power of two, == threads per threadgroup
    uint n_kv_heads;
    uint lane_capacity; // physical tokens per lane region in the pool
    uint bits;          // 2, 3, or 4
    uint code_bytes;    // head_dim * bits / 8
    uint n_lanes;
    uint ring_capacity; // 0 = absolute; >0 = the lane position wraps at this
                        // many tokens within the lane's physical region.
};

kernel void encode_kv_turboquant_batched(
    device const half* input [[buffer(0)]],
    device const float* levels [[buffer(1)]],
    device const float* signs [[buffer(2)]],
    device uchar* packed [[buffer(3)]],
    device half* norms [[buffer(4)]],
    constant TqEncodeBatchedParams& p [[buffer(5)]],
    constant uint* positions [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    threadgroup float sh[512];
    threadgroup float red[512];
    threadgroup uint code_sh[512];
    const uint d = p.head_dim;

    uint lane_idx = group_id / p.n_kv_heads;
    uint head = group_id - lane_idx * p.n_kv_heads;
    if (lane_idx >= p.n_lanes) { return; }
    // Pool slot for this lane's freshly-projected (head) vector. `lane_capacity`
    // is the physical per-lane stride; the position wraps within `ring_capacity`
    // tokens for sliding-window pools.
    uint pos = positions[lane_idx];
    uint ppos = (p.ring_capacity != 0u) ? (pos % p.ring_capacity) : pos;
    uint g = (lane_idx * p.lane_capacity + ppos) * p.n_kv_heads + head;

    float xv = float(input[group_id * d + lane]);

    red[lane] = xv * xv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = d >> 1; stride > 0u; stride >>= 1) {
        if (lane < stride) {
            red[lane] += red[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(red[0]);
    float inv = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

    sh[lane] = xv * inv * signs[lane];
    for (uint len = 1u; len < d; len <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float self_v = sh[lane];
        float other_v = sh[lane ^ len];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sh[lane] = (lane & len) ? (other_v - self_v) : (self_v + other_v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float y = sh[lane] * rsqrt(float(d));

    uint k = 1u << p.bits;
    uint best = 0u;
    float bd = INFINITY;
    for (uint i = 0u; i < k; ++i) {
        float dd = fabs(y - levels[i]);
        if (dd < bd) {
            bd = dd;
            best = i;
        }
    }
    code_sh[lane] = best;
    if (lane == 0u) {
        norms[g] = half(norm);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane < p.code_bytes) {
        uint byte_val = 0u;
        for (uint bitpos = 0u; bitpos < 8u; ++bitpos) {
            uint global_bit = lane * 8u + bitpos;
            uint code_idx = global_bit / p.bits;
            if (code_idx >= d) {
                break;
            }
            uint within = global_bit - code_idx * p.bits;
            uint bit = (code_sh[code_idx] >> within) & 1u;
            byte_val |= bit << bitpos;
        }
        packed[g * p.code_bytes + lane] = uchar(byte_val);
    }
}
