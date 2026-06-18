#ifndef COMMON_H
#define COMMON_H

#include <metal_stdlib>
using namespace metal;

#define GLOBAL_ID thread_position_in_grid
#define LOCAL_ID thread_position_in_threadgroup
#define GROUP_ID threadgroup_position_in_grid
#define GROUP_SIZE threads_per_threadgroup

constant half H_PI = 3.14159265h;
constant half H_SQRT_2_OVER_PI = 0.7978845608h;

#endif
