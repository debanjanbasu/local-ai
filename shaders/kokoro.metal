// Kokoro-82M speech synthesis kernels (see local-metal/src/kokoro.rs).
//
// Activations are F32 [channels, time] rows (PyTorch's own layout for batch 1),
// each row `row` floats apart so producers can write straight into a channel
// slice of a concatenation. Weights are the checkpoint's F32 values with
// weight normalization folded in by the converter.
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- conv GEMM

// One convolution (or linear map) as a GEMM with gathered inputs:
//   y[co, col * out_stride + out_offset] (+)= scale * (bias[co] +
//       sum_{tap, ci} w[co, tap, ci] * x[ci, col * in_stride + tap * in_dil + in_offset])
// where inputs outside [0, in_len) read as zero. A plain Conv1d is
// (stride, dilation, -padding); a linear map is one tap; each output phase of
// a ConvTranspose1d is a one-stride, negative-dilation conv over its own
// rearranged taps (see the converter). Weights are [cout][taps][cin].
struct KkConv {
    uint cout;
    uint cin;
    uint taps;
    uint columns;
    int in_stride;
    int in_dil;
    int in_offset;
    uint in_len;
    uint in_row;
    uint out_stride;
    uint out_offset;
    uint out_row;
    uint flags;  // 1: add bias, 2: accumulate into y
    float scale;
};

constant constexpr uint KK_TILE = 32;
constant constexpr uint KK_PAD = 4;

// A 128-thread threadgroup owns a 32-row x 32-column tile of y; each SIMD
// group a 16 x 16 quarter as 2 x 2 F32 8x8 simdgroup matrices. Grid:
// (ceil(cout / 32), ceil(columns / 32)).
kernel void kk_conv(
    device const float *w [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]], device const float *bias [[buffer(3)]],
    constant KkConv &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float ws[KK_TILE][KK_TILE + KK_PAD];
    threadgroup float xs[KK_TILE][KK_TILE + KK_PAD];
    threadgroup float cs[KK_TILE][KK_TILE + KK_PAD];
    const uint row0 = group.x * KK_TILE, col0 = group.y * KK_TILE;
    const uint line = tid / 4, k8 = (tid % 4) * 8;
    const uint row = row0 + line;
    const bool row_live = row < p.cout;
    const ulong w_row = ulong(min(row, p.cout - 1)) * p.taps * p.cin;
    const uint sm = (simd / 2) * 16, sn = (simd % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint tap = 0; tap < p.taps; ++tap) {
        const int base = int(tap) * p.in_dil + p.in_offset;
        for (uint ci0 = 0; ci0 < p.cin; ci0 += KK_TILE) {
            float a[8];
            float b[8];
            device const float *wp = w + w_row + ulong(tap) * p.cin + ci0 + k8;
            const uint ci_b = ci0 + line;
            const bool ci_live = ci_b < p.cin;
            device const float *xp = x + ulong(min(ci_b, p.cin - 1)) * p.in_row;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                a[i] = (row_live && ci0 + k8 + i < p.cin) ? wp[i] : 0.0f;
                const uint col = col0 + k8 + i;
                const int index = int(col) * p.in_stride + base;
                b[i] = (ci_live && col < p.columns && index >= 0 && uint(index) < p.in_len)
                           ? xp[index] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                ws[line][k8 + i] = a[i];
                xs[line][k8 + i] = b[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for (uint kk = 0; kk < KK_TILE; kk += 8) {
                simdgroup_float8x8 am[2], bm[2];
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    simdgroup_load(am[i], &ws[sm + 8 * i][kk], KK_TILE + KK_PAD);
                    simdgroup_load(bm[i], &xs[kk][sn + 8 * i], KK_TILE + KK_PAD);
                }
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 2; ++i) {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 2; ++j)
                        simdgroup_multiply_accumulate(acc[i][j], am[i], bm[j], acc[i][j]);
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 2; ++j)
            simdgroup_store(acc[i][j], &cs[sm + 8 * i][sn + 8 * j], KK_TILE + KK_PAD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!row_live) return;
    const float b0 = (p.flags & 1) ? bias[row] : 0.0f;
    device float *yp = y + ulong(row) * p.out_row;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8; ++i) {
        const uint col = col0 + k8 + i;
        if (col >= p.columns) continue;
        const ulong at = ulong(col) * p.out_stride + p.out_offset;
        const float value = p.scale * (cs[line][k8 + i] + b0);
        yp[at] = (p.flags & 2) ? yp[at] + value : value;
    }
}

// ---------------------------------------------------------------- LSTM

// Bidirectional single-layer LSTM recurrence (PyTorch gate order i, f, g, o),
// hidden 256. `gates` holds W_ih x + b_ih + b_hh for both directions,
// [2 * 4H rows][time] (row stride g_row); `whh` is W_hh transposed per
// direction, [2][H][4H]. Output h goes to rows dir * H + j of `out`. One
// threadgroup per direction walks time; threads cover the 4H gate rows.
constant constexpr uint KK_H = 256;

kernel void kk_lstm(
    device const float *gates [[buffer(0)]], device const float *whh [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &steps [[buffer(3)]],
    constant uint &g_row [[buffer(4)]], constant uint &o_row [[buffer(5)]],
    uint dir [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float h[KK_H];
    threadgroup float c[KK_H];
    threadgroup float g[4 * KK_H];
    for (uint j = tid; j < KK_H; j += threads) {
        h[j] = 0.0f;
        c[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float *wd = whh + ulong(dir) * KK_H * 4 * KK_H;
    device const float *gd = gates + ulong(dir) * 4 * KK_H * g_row;
    for (uint s = 0; s < steps; ++s) {
        const uint t = dir == 0 ? s : steps - 1 - s;
        for (uint r = tid; r < 4 * KK_H; r += threads) {
            float acc = gd[ulong(r) * g_row + t];
            for (uint j = 0; j < KK_H; ++j) acc = fma(wd[j * 4 * KK_H + r], h[j], acc);
            g[r] = (r / KK_H == 2) ? precise::tanh(acc) : 1.0f / (1.0f + precise::exp(-acc));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = tid; j < KK_H; j += threads) {
            const float cell = g[KK_H + j] * c[j] + g[j] * g[2 * KK_H + j];
            const float hv = g[3 * KK_H + j] * precise::tanh(cell);
            c[j] = cell;
            h[j] = hv;
            out[ulong(dir * KK_H + j) * o_row + t] = hv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------- norms

// LayerNorm over channels of each column (thread per column), then
// y = norm * scale[c] + shift[c] when flags & 1, then LeakyReLU(slope) when
// flags & 2. In place is allowed.
kernel void kk_channel_norm(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    device const float *scale [[buffer(2)]], device const float *shift [[buffer(3)]],
    constant uint &channels [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &row [[buffer(6)]], constant float &eps [[buffer(7)]],
    constant uint &flags [[buffer(8)]], constant float &slope [[buffer(9)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= columns) return;
    float mean = 0.0f;
    for (uint c = 0; c < channels; ++c) mean += x[ulong(c) * row + t];
    mean /= float(channels);
    float var = 0.0f;
    for (uint c = 0; c < channels; ++c) {
        const float d = x[ulong(c) * row + t] - mean;
        var += d * d;
    }
    const float inv = precise::rsqrt(var / float(channels) + eps);
    for (uint c = 0; c < channels; ++c) {
        float v = (x[ulong(c) * row + t] - mean) * inv;
        if (flags & 1) v = v * scale[c] + shift[c];
        if (flags & 2) v = v > 0.0f ? v : v * slope;
        y[ulong(c) * row + t] = v;
    }
}

// Instance norm of each channel over time (AdaIN1d), one 256-thread
// threadgroup per channel: y = norm * scale[c] + shift[c], then mode 1
// LeakyReLU(slope) or mode 2 Snake (y + sin(alpha[c] y)^2 / alpha[c]).
kernel void kk_instance_norm(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    device const float *scale [[buffer(2)]], device const float *shift [[buffer(3)]],
    device const float *alpha [[buffer(4)]], constant uint &columns [[buffer(5)]],
    constant uint &x_row [[buffer(6)]], constant uint &y_row [[buffer(7)]],
    constant float &eps [[buffer(8)]], constant uint &mode [[buffer(9)]],
    constant float &slope [[buffer(10)]],
    uint c [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]
) {
    threadgroup float partial[8];
    device const float *xp = x + ulong(c) * x_row;
    float s = 0.0f;
    for (uint t = tid; t < columns; t += 256) s += xp[t];
    s = simd_sum(s);
    if (tid % 32 == 0) partial[tid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float mean = total / float(columns);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.0f;
    for (uint t = tid; t < columns; t += 256) {
        const float d = xp[t] - mean;
        v += d * d;
    }
    v = simd_sum(v);
    if (tid % 32 == 0) partial[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float inv = precise::rsqrt(total / float(columns) + eps);
    const float sc = scale[c], sh = shift[c];
    const float a = mode == 2 ? alpha[c] : 1.0f;
    device float *yp = y + ulong(c) * y_row;
    for (uint t = tid; t < columns; t += 256) {
        float value = (xp[t] - mean) * inv * sc + sh;
        if (mode == 1) {
            value = value > 0.0f ? value : value * slope;
        } else if (mode == 2) {
            const float sn = precise::sin(a * value);
            value = value + (1.0f / a) * (sn * sn);
        }
        yp[t] = value;
    }
}

// ---------------------------------------------------------------- elementwise

// y[c, t] = (accumulate ? y[c, t] : 0) + scale * act(x[c, t]) for a
// [channels, columns] block with row strides; act is identity (slope 1) or
// LeakyReLU(slope).
kernel void kk_copy(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &columns [[buffer(2)]], constant uint &x_row [[buffer(3)]],
    constant uint &y_row [[buffer(4)]], constant float &scale [[buffer(5)]],
    constant float &slope [[buffer(6)]], constant uint &accumulate [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]
) {
    if (pos.x >= columns) return;
    float v = x[ulong(pos.y) * x_row + pos.x];
    v = v > 0.0f ? v : v * slope;
    device float *yp = y + ulong(pos.y) * y_row + pos.x;
    *yp = accumulate ? *yp + scale * v : scale * v;
}

// y[c, t] = x[c, index[t]] (duration alignment, or nearest x2 upsampling).
kernel void kk_gather(
    device const float *x [[buffer(0)]], device const uint *index [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &columns [[buffer(3)]],
    constant uint &x_row [[buffer(4)]], constant uint &y_row [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    if (pos.x >= columns) return;
    y[ulong(pos.y) * y_row + pos.x] = x[ulong(pos.y) * x_row + index[pos.x]];
}

// Depthwise ConvTranspose1d(k=3, stride 2, padding 1, output_padding 1):
// y[c, t] = b[c] + sum_k w[c, k] x[c, (t + 1 - k) / 2] over integral, in-range
// sources; columns = 2 * input columns.
kernel void kk_depthwise_up(
    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *b [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &in_columns [[buffer(4)]], constant uint &x_row [[buffer(5)]],
    constant uint &y_row [[buffer(6)]], uint2 pos [[thread_position_in_grid]]
) {
    const uint t = pos.x, c = pos.y;
    if (t >= 2 * in_columns) return;
    float acc = b[c];
    #pragma clang loop unroll(full)
    for (int k = 0; k < 3; ++k) {
        const int num = int(t) + 1 - k;
        if (num >= 0 && (num & 1) == 0 && uint(num / 2) < in_columns)
            acc += w[c * 3 + k] * x[ulong(c) * x_row + uint(num / 2)];
    }
    y[ulong(c) * y_row + t] = acc;
}

// GELU, tanh approximation ("gelu_new"), in place over n values.
kernel void kk_gelu_tanh(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// ---------------------------------------------------------------- attention

// Full bidirectional attention over [heads * 64, tokens] q, k, v (row stride
// `row`), one thread per (token, head), softmax scaled by 1/8.
kernel void kk_attention(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &row [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    const uint t = pos.x, head = pos.y;
    if (t >= tokens) return;
    const ulong base = ulong(head) * 64 * row;
    float qv[64];
    for (uint d = 0; d < 64; ++d) qv[d] = q[base + ulong(d) * row + t] * 0.125f;
    float acc[64];
    for (uint d = 0; d < 64; ++d) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (uint j = 0; j < tokens; ++j) {
        float s = 0.0f;
        for (uint d = 0; d < 64; ++d) s = fma(qv[d], k[base + ulong(d) * row + j], s);
        const float m_new = max(m, s);
        const float correction = precise::exp(m - m_new);
        const float pj = precise::exp(s - m_new);
        l = l * correction + pj;
        for (uint d = 0; d < 64; ++d) acc[d] = acc[d] * correction + pj * v[base + ulong(d) * row + j];
        m = m_new;
    }
    const float inv = 1.0f / l;
    for (uint d = 0; d < 64; ++d) out[base + ulong(d) * row + t] = acc[d] * inv;
}

// ---------------------------------------------------------------- iSTFT

// torch.istft(n_fft 20, hop 5, periodic Hann, center) of the generator head:
// x [22, frames] holds log-magnitudes (rows 0..10) and phase pre-activations
// (rows 11..21, phase = sin(row)). One thread per output sample.
kernel void kk_istft(
    device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
    constant uint &frames [[buffer(2)]], constant uint &row [[buffer(3)]],
    uint n [[thread_position_in_grid]]
) {
    const uint samples = 5 * (frames - 1);
    if (n >= samples) return;
    const int p = int(n) + 10;
    const int first = max(0, (p - 19 + 4) / 5);
    const int last = min(int(frames) - 1, p / 5);
    float acc = 0.0f, env = 0.0f;
    for (int f = first; f <= last; ++f) {
        const int m = p - 5 * f;
        float value = 0.0f;
        for (uint k = 0; k < 11; ++k) {
            const float mag = precise::exp(x[ulong(k) * row + f]);
            const float phase = precise::sin(x[ulong(11 + k) * row + f]);
            const float re = mag * precise::cos(phase);
            const float im = mag * precise::sin(phase);
            const float angle = 2.0f * M_PI_F * float((int(k) * m) % 20) / 20.0f;
            if (k == 0) {
                value += re;
            } else if (k == 10) {
                value += (m & 1) ? -re : re;
            } else {
                value += 2.0f * (re * precise::cos(angle) - im * precise::sin(angle));
            }
        }
        value /= 20.0f;
        const float w = 0.5f - 0.5f * precise::cos(2.0f * M_PI_F * float(m) / 20.0f);
        acc += w * value;
        env += w * w;
    }
    out[n] = env > 1e-11f ? acc / env : 0.0f;
}
