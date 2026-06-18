#include <metal_stdlib>
using namespace metal;

/// Gemma 4 audio subsampling conv layer.
///
/// input:  [in_channels, in_time, in_freq]
/// weight: [3, 3, in_channels, out_channels]
/// norm:   [out_channels]
/// output: [out_channels, out_time, out_freq]
///
/// The kernel fuses Conv2d(stride=2,pad=1,bias=false), LayerNorm over the
/// channel axis at each (time,freq), and ReLU. One thread computes all output
/// channels for one spatial position to keep LayerNorm local and exact.
kernel void audio_subsample_conv2d_ln_relu(
    device const half *input       [[buffer(0)]],
    device const half *weight      [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device half       *output      [[buffer(3)]],
    constant uint &in_channels     [[buffer(4)]],
    constant uint &out_channels    [[buffer(5)]],
    constant uint &in_time         [[buffer(6)]],
    constant uint &in_freq         [[buffer(7)]],
    constant uint &out_time        [[buffer(8)]],
    constant uint &out_freq        [[buffer(9)]],
    constant float &eps            [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint t = gid.x;
    uint f = gid.y;
    if (t >= out_time || f >= out_freq) return;

    float vals[128];
    float mean = 0.0f;
    for (uint oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        for (uint kh = 0; kh < 3; ++kh) {
            int it = int(t * 2 + kh) - 1;
            if (it < 0 || it >= int(in_time)) continue;
            for (uint kw = 0; kw < 3; ++kw) {
                int iff = int(f * 2 + kw) - 1;
                if (iff < 0 || iff >= int(in_freq)) continue;
                for (uint ic = 0; ic < in_channels; ++ic) {
                    uint in_idx = (ic * in_time + uint(it)) * in_freq + uint(iff);
                    uint w_idx = ((kh * 3 + kw) * in_channels + ic) * out_channels + oc;
                    sum += float(input[in_idx]) * float(weight[w_idx]);
                }
            }
        }
        vals[oc] = sum;
        mean += sum;
    }
    mean /= float(out_channels);

    float var = 0.0f;
    for (uint oc = 0; oc < out_channels; ++oc) {
        float d = vals[oc] - mean;
        var += d * d;
    }
    float inv_rms = rsqrt(var / float(out_channels) + eps);

    for (uint oc = 0; oc < out_channels; ++oc) {
        float v = (vals[oc] - mean) * inv_rms * float(norm_weight[oc]);
        v = max(v, 0.0f);
        uint out_idx = (oc * out_time + t) * out_freq + f;
        output[out_idx] = half(v);
    }
}

/// Pack frontend output from `[channels, time, freq]` to HF order
/// `[time, freq * channels]` after `permute(0, 2, 3, 1).reshape(...)`.
kernel void audio_pack_frontend(
    device const half *input    [[buffer(0)]],
    device half       *output   [[buffer(1)]],
    constant uint &channels     [[buffer(2)]],
    constant uint &time         [[buffer(3)]],
    constant uint &freq         [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint t = gid.x;
    uint d = gid.y;
    if (t >= time || d >= freq * channels) return;
    uint f = d / channels;
    uint c = d - f * channels;
    output[t * freq * channels + d] = input[(c * time + t) * freq + f];
}

/// Gemma 4 audio chunked local attention for a single batch.
///
/// q/k/v/output: [seq_len, num_heads * head_dim]
/// rel_k:         [left_context + 1, num_heads * head_dim]
/// per_dim_scale: [head_dim]
kernel void audio_chunked_attention(
    device const half *q             [[buffer(0)]],
    device const half *k             [[buffer(1)]],
    device const half *v             [[buffer(2)]],
    device const half *rel_k         [[buffer(3)]],
    device const half *per_dim_scale [[buffer(4)]],
    device half       *output        [[buffer(5)]],
    constant uint &seq_len           [[buffer(6)]],
    constant uint &num_heads         [[buffer(7)]],
    constant uint &head_dim          [[buffer(8)]],
    constant uint &chunk_size        [[buffer(9)]],
    constant uint &left_context      [[buffer(10)]],
    constant float &softcap          [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x;
    uint token = gid.y;
    uint head = gid.z;
    if (d >= head_dim || token >= seq_len || head >= num_heads) return;

    constexpr uint MAX_CONTEXT = 32;
    float scores[MAX_CONTEXT];
    float max_score = -INFINITY;
    uint context_size = chunk_size + left_context;
    uint block = token / chunk_size;
    uint q_in_block = token - block * chunk_size;
    float q_scale = rsqrt(float(head_dim)) / log2(M_E_F);
    float k_scale = log2(1.0f + M_E_F);

    for (uint c = 0; c < MAX_CONTEXT; ++c) {
        float score = -INFINITY;
        if (c < context_size) {
            int key_idx = int(block * chunk_size + c) - int(left_context);
            bool valid = key_idx >= 0 && uint(key_idx) < seq_len && uint(key_idx) <= token && token - uint(key_idx) <= left_context;
            if (valid) {
                float ac = 0.0f;
                uint q_base = (token * num_heads + head) * head_dim;
                uint k_base = (uint(key_idx) * num_heads + head) * head_dim;
                for (uint rd = 0; rd < head_dim; ++rd) {
                    float qs = float(q[q_base + rd]) * q_scale * log(1.0f + exp(float(per_dim_scale[rd])));
                    float ks = float(k[k_base + rd]) * k_scale;
                    ac += qs * ks;
                }

                uint shifted = q_in_block * context_size + c;
                uint src_q = shifted / (context_size + 1);
                uint rel_pos = shifted - src_q * (context_size + 1);
                float bd = 0.0f;
                if (src_q < chunk_size && rel_pos <= left_context) {
                    uint src_token = block * chunk_size + src_q;
                    if (src_token < seq_len) {
                        uint src_q_base = (src_token * num_heads + head) * head_dim;
                        uint rel_base = (rel_pos * num_heads + head) * head_dim;
                        for (uint rd = 0; rd < head_dim; ++rd) {
                            float qs = float(q[src_q_base + rd]) * q_scale * log(1.0f + exp(float(per_dim_scale[rd])));
                            bd += qs * float(rel_k[rel_base + rd]);
                        }
                    }
                }

                score = tanh((ac + bd) / softcap) * softcap;
            }
        }
        scores[c] = score;
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    for (uint c = 0; c < MAX_CONTEXT; ++c) {
        if (c < context_size) {
            float w = exp(scores[c] - max_score);
            scores[c] = w;
            denom += w;
        }
    }

    float acc = 0.0f;
    if (denom > 0.0f) {
        for (uint c = 0; c < MAX_CONTEXT; ++c) {
            if (c < context_size && scores[c] > 0.0f) {
                int key_idx = int(block * chunk_size + c) - int(left_context);
                uint v_base = (uint(key_idx) * num_heads + head) * head_dim;
                acc += (scores[c] / denom) * float(v[v_base + d]);
            }
        }
    }

    output[(token * num_heads + head) * head_dim + d] = half(acc);
}

/// GLU over the last dimension: output = left * sigmoid(right).
/// input:  [rows, hidden * 2]
/// output: [rows, hidden]
kernel void audio_glu(
    device const half *input  [[buffer(0)]],
    device half       *output [[buffer(1)]],
    constant uint &rows       [[buffer(2)]],
    constant uint &hidden     [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint d = gid.x;
    if (row >= rows || d >= hidden) return;
    uint base = row * hidden * 2;
    float left = float(input[base + d]);
    float right = float(input[base + hidden + d]);
    output[row * hidden + d] = half(left / (1.0f + exp(-right)));
}

/// Causal depthwise 1D conv over `[seq_len, channels]` with left padding.
/// weight is `[channels, kernel_size]` in GGUF row-major order.
kernel void depthwise_conv1d(
    device const half *input  [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half       *output [[buffer(2)]],
    constant uint &seq_len    [[buffer(3)]],
    constant uint &channels   [[buffer(4)]],
    constant uint &kernel_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint c = gid.x;
    uint t = gid.y;
    if (c >= channels || t >= seq_len) return;
    float sum = 0.0f;
    for (uint k = 0; k < kernel_size; ++k) {
        int src = int(t) + int(k) + 1 - int(kernel_size);
        if (src >= 0) {
            sum += float(input[uint(src) * channels + c]) * float(weight[c * kernel_size + k]);
        }
    }
    output[t * channels + c] = half(sum);
}
