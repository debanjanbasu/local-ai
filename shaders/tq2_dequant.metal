#include <metal_stdlib>
using namespace metal;

kernel void tq2_dequant(
    device const uchar * __restrict__ packed [[buffer(0)]],
    device const half * __restrict__ scale_f16 [[buffer(1)]],
    device half * __restrict__ out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
  uint idx = tid / 4u;
  uint rem = tid % 4u;
  uchar packed_byte = packed[idx];
  uchar val = (packed_byte >> (rem * 2u)) & 0x3u;
  float s = float(scale_f16[idx]);
  out[tid] = half(s * float(val));
}
