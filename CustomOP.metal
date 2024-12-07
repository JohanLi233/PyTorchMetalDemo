#pragma once

const char *CUSTOM_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void custom_fill(
    device float *data [[buffer(0)]],
    constant float &fill_val [[buffer(1)]],
    constant uint &data_size [[buffer(2)]],
    uint id [[thread_position_in_grid]]
)
{
    if (id < data_size) {
        data[id] = fill_val;
    }
}

kernel void custom_add(device const float* in1 [[buffer(0)]],
                       device const float* in2 [[buffer(1)]],
                       device float* out    [[buffer(2)]],
                       constant uint& data_size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < data_size) {
        out[gid] = in1[gid] + in2[gid];
    }
}
)";