#include <metal_stdlib>
using namespace metal;

/// Gemma 4 vision patch embedding: Conv2D with kernel_size=stride=patch_size.
///
/// Input pixels are row-major `[3, H, W]` FP16. Weight is GGUF-order
/// `[patch, patch, 3, hidden]` FP16. Output is row-major
/// `[num_patches_h * num_patches_w, hidden]` FP16.
kernel void vision_patch_embed(
    device const half *input        [[buffer(0)]],
    device const half *weight       [[buffer(1)]],
    device const half *bias         [[buffer(2)]],
    device half       *output       [[buffer(3)]],
    constant uint     &hidden_size  [[buffer(4)]],
    constant uint     &patch_size   [[buffer(5)]],
    constant uint     &img_h        [[buffer(6)]],
    constant uint     &img_w        [[buffer(7)]],
    constant uint     &patches_w    [[buffer(8)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint hidden = gid.x;
    uint patch_index = gid.y;
    uint patch_y = patch_index / patches_w;
    uint patch_x = patch_index - patch_y * patches_w;

    float acc = float(bias[hidden]);
    for (uint py = 0; py < patch_size; py++) {
        uint in_y = patch_y * patch_size + py;
        if (in_y >= img_h) continue;
        for (uint px = 0; px < patch_size; px++) {
            uint in_x = patch_x * patch_size + px;
            if (in_x >= img_w) continue;
            for (uint c = 0; c < 3; c++) {
                uint input_idx = (c * img_h + in_y) * img_w + in_x;
                uint weight_idx = (((hidden * 3 + c) * patch_size + py) * patch_size) + px;
                acc += float(input[input_idx]) * float(weight[weight_idx]);
            }
        }
    }
    output[patch_index * hidden_size + hidden] = half(acc);
}

/// Average-pool a sequence laid out as `[rows * cols, hidden]`.
kernel void vision_avg_pool_2d(
    device const half *input        [[buffer(0)]],
    device half       *output       [[buffer(1)]],
    constant uint     &hidden_size  [[buffer(2)]],
    constant uint     &in_rows      [[buffer(3)]],
    constant uint     &in_cols      [[buffer(4)]],
    constant uint     &kernel_size  [[buffer(5)]],
    constant uint     &stride       [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint hidden = gid.x;
    uint out_index = gid.y;
    uint out_cols = (in_cols - kernel_size) / stride + 1;
    uint out_y = out_index / out_cols;
    uint out_x = out_index - out_y * out_cols;

    float acc = 0.0f;
    uint count = 0;
    for (uint ky = 0; ky < kernel_size; ky++) {
        uint in_y = out_y * stride + ky;
        if (in_y >= in_rows) continue;
        for (uint kx = 0; kx < kernel_size; kx++) {
            uint in_x = out_x * stride + kx;
            if (in_x >= in_cols) continue;
            uint in_index = (in_y * in_cols + in_x) * hidden_size + hidden;
            acc += float(input[in_index]);
            count++;
        }
    }
    output[out_index * hidden_size + hidden] = half(acc / max(float(count), 1.0f));
}

/// Add Gemma 4's learned 2D vision position embeddings in-place.
///
/// Hidden states are `[rows * cols, hidden]`; position table is GGUF-order
/// `[hidden, position_count, 2]`, with axis 0 = x and axis 1 = y.
kernel void vision_add_position_embedding(
    device half       *hidden_states  [[buffer(0)]],
    device const half *position_table  [[buffer(1)]],
    constant uint     &hidden_size     [[buffer(2)]],
    constant uint     &position_count  [[buffer(3)]],
    constant uint     &patch_rows      [[buffer(4)]],
    constant uint     &patch_cols      [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    uint hidden = gid.x;
    uint patch_index = gid.y;
    uint y = patch_index / patch_cols;
    uint x = patch_index - y * patch_cols;
    if (hidden >= hidden_size || y >= patch_rows || x >= patch_cols) return;

    uint x_pos = min(x, position_count - 1);
    uint y_pos = min(y, position_count - 1);
    uint x_offset = x_pos * hidden_size + hidden;
    uint y_offset = (position_count + y_pos) * hidden_size + hidden;
    uint state_offset = patch_index * hidden_size + hidden;
    hidden_states[state_offset] = half(float(hidden_states[state_offset])
        + float(position_table[x_offset])
        + float(position_table[y_offset]));
}

static inline void apply_1d_rope_pair(device half *buf,
                                      uint base,
                                      uint i,
                                      uint head_dim,
                                      float theta) {
    float exponent = float(2 * i) / float(head_dim);
    float freq = pow(theta, -exponent);
    float angle = freq;
    float c = cos(angle);
    float s = sin(angle);
    uint even = base + 2 * i;
    uint odd = even + 1;
    float x0 = float(buf[even]);
    float x1 = float(buf[odd]);
    buf[even] = half(x0 * c - x1 * s);
    buf[odd] = half(x0 * s + x1 * c);
}

/// Gemma 4 multidimensional vision RoPE for q/k tensors laid out
/// `[seq_len, num_heads, head_dim]`.
///
/// Gemma 4 splits each head into x/y spatial halves. Within each half it uses
/// the standard rotate-half layout, not interleaved even/odd pairs.
kernel void vision_rope(
    device half    *q          [[buffer(0)]],
    device half    *k          [[buffer(1)]],
    constant uint  &head_dim   [[buffer(2)]],
    constant float &theta      [[buffer(3)]],
    constant uint  &num_heads  [[buffer(4)]],
    constant uint  &patch_rows [[buffer(5)]],
    constant uint  &patch_cols [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]]
) {
    uint pair = gid.x;
    uint head = gid.y;
    uint seq = gid.z;

    if (patch_cols == 0 || seq >= patch_rows * patch_cols) {
        return;
    }

    uint spatial_dim = head_dim / 2;
    uint half_spatial = spatial_dim / 2;
    uint axis = pair / half_spatial;
    uint pair_in_axis = pair - axis * half_spatial;
    if (axis >= 2 || pair_in_axis >= half_spatial) {
        return;
    }

    uint row = seq / patch_cols;
    uint col = seq - row * patch_cols;
    float position = float(axis == 0 ? col : row);
    float inv_freq = pow(theta, -float(pair_in_axis * 2) / float(spatial_dim));
    float angle = position * inv_freq;
    float c = cos(angle);
    float s = sin(angle);

    uint base = (seq * num_heads + head) * head_dim;
    uint axis_base = base + axis * spatial_dim;
    uint lo = axis_base + pair_in_axis;
    uint hi = lo + half_spatial;

    float q0 = float(q[lo]);
    float q1 = float(q[hi]);
    q[lo] = half(q0 * c - q1 * s);
    q[hi] = half(q1 * c + q0 * s);

    float k0 = float(k[lo]);
    float k1 = float(k[hi]);
    k[lo] = half(k0 * c - k1 * s);
    k[hi] = half(k1 * c + k0 * s);
}
