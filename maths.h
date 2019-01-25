#pragma once

#include <optixu/optixu_math_namespace.h>

using float3 = optix::float3;
using float2 = optix::float2;

using optix::make_float3;
using optix::make_float2;

constexpr float pi = M_PIf;
constexpr float half_pi = M_PI_2f;
constexpr float inf = 1e20;

using optix::clamp;