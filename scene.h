#pragma once

#include "vector.h"
#include "maths.h"

struct TriMesh {
    Vector<float3> positionBuffer;
    Vector<float3> normalBuffer;
    Vector<uint32_t> indexBuffer;

    size_t triangleCount() const
    {
        return indexBuffer.size() / 3;
    }
};

TriMesh makeSphereTriMesh(const float3 origin, float radius, const uint32_t subdivLongitude = 32);

struct TriangleHit
{
    float dist = inf;
    float u, v;

    TriangleHit() = default;

    TriangleHit(float d, float u, float v) : dist{ d }, u{ u }, v{ v } {}
};

TriangleHit triIntersect(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2);

struct Hit
{
    float dist = inf;
    uint32_t instId;
    uint32_t triId;
    float3 x;
    float3 n;
    optix::float2 uv;

    explicit operator bool() const {
        return dist < inf;
    }
};

struct MeshHit : public TriangleHit
{
    uint32_t triId;

    MeshHit() = default;
    MeshHit(TriangleHit h, uint32_t triId) : TriangleHit{ h }, triId{ triId } {}
};

// Convert a mesh hit to a generic hit
Hit makeHit(uint32_t instId, const TriMesh & mesh, const MeshHit & meshHit);

MeshHit intersect(const float3 ro, const float3 rd, const TriMesh & mesh);

struct Ray {
    float3 o, d;
    Ray() = default;
    Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct SphereHit
{
    float dist = inf;
    float3 x;

    SphereHit() = default;
    SphereHit(float d, float3 x) : dist{ d }, x(x) {}
};

struct Material
{
    float3 emission;
    float3 color;
    Refl_t refl;

    Material(float3 e_, float3 c_, Refl_t refl_) : emission(e_), color(c_), refl(refl_) {}
};

struct Sphere {
    float radius;       // radius
    float3 center;

    Material material;
    TriMesh mesh;

    Sphere(float rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) :
        radius(rad_), center(p_), material{ e_, c_, refl_ }, mesh{ makeSphereTriMesh(p_, rad_) } {}

    Hit makeHit(uint32_t instId, const SphereHit & sphereHit);

    Hit makeHit(uint32_t instId, const MeshHit & meshHit)
    {
        return ::makeHit(instId, mesh, meshHit);
    }

    SphereHit intersectAnalytic(const Ray &r) const;

    MeshHit intersectMesh(const Ray &r) const {
        return ::intersect(r.o, r.d, mesh);
    }

    auto intersect(const Ray &r) const {
        return intersectMesh(r);
    }
};