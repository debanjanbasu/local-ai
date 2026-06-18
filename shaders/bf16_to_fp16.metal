#include <metal_stdlib>
using namespace metal;

kernel void bf16_to_fp16(
    device const ushort* input  [[buffer(0)]],
    device half*         output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    // BF16 → float → half
    // BF16 bit pattern: same as float32 upper 16 bits
    uint raw = uint(input[id]);
    float f32_val = as_type<float>(raw << 16);
    output[id] = half(f32_val);
}
