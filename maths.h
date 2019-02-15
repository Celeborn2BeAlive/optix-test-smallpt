#pragma once

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

using float4 = optix::float4;
using float3 = optix::float3;
using float2 = optix::float2;

using optix::make_float4;
using optix::make_float3;
using optix::make_float2;

constexpr float pi = M_PIf;
constexpr float half_pi = M_PI_2f;
constexpr float inf = 1e20;

using optix::clamp;

using float4x4 = optix::Matrix4x4;
using float3x4 = optix::Matrix3x4;