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
)";