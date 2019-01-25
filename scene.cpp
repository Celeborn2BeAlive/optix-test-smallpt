#include "scene.h"

TriMesh makeSphereTriMesh(const float3 origin, float radius, const uint32_t subdivLongitude)
{
    const auto discLong = subdivLongitude;
    const auto discLat = 2 * discLong;

    float rcpLat = 1.f / discLat, rcpLong = 1.f / discLong;
    float dPhi = pi * 2.f * rcpLat, dTheta = pi * rcpLong;

    Vector<float3> positionBuffer, normalBuffer;

    for (uint32_t j = 0; j <= discLong; ++j)
    {
        float cosTheta = cos(-half_pi + j * dTheta);
        float sinTheta = sin(-half_pi + j * dTheta);

        for (uint32_t i = 0; i <= discLat; ++i) {
            const float3 coords{
                sin(i * dPhi) * cosTheta,
                sinTheta,
                cos(i * dPhi) * cosTheta
            };

            positionBuffer.emplace_back(origin + radius * coords);
            normalBuffer.emplace_back(coords);
        }
    }

    Vector<uint32_t> indexBuffer;

    for (uint32_t j = 0; j < discLong; ++j)
    {
        uint32_t offset = j * (discLat + 1);
        for (uint32_t i = 0; i < discLat; ++i)
        {
            indexBuffer.push_back(offset + i);
            indexBuffer.push_back(offset + (i + 1));
            indexBuffer.push_back(offset + discLat + 1 + (i + 1));

            indexBuffer.push_back(offset + i);
            indexBuffer.push_back(offset + discLat + 1 + (i + 1));
            indexBuffer.push_back(offset + i + discLat + 1);
        }
    }

    return{ positionBuffer, normalBuffer, indexBuffer };
}

// triangle degined by vertices v0, v1 and  v2
// https://iquilezles.org/www/articles/intersectors/intersectors.htm
TriangleHit triIntersect(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2)
{
    using vec3 = float3;

    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = ro - v0;

    vec3  n = cross(v1v0, v2v0);
    vec3  q = cross(rov0, rd);
    float d = 1.0 / dot(rd, n);
    float u = d*dot(-q, v2v0);
    float v = d*dot(q, v1v0);
    float t = d*dot(-n, rov0);

    if (u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0) t = inf;

    return{ t, u, v };
}

// Convert a mesh hit to a generic hit
Hit makeHit(uint32_t instId, const TriMesh & mesh, const MeshHit & meshHit)
{
    Hit hit;
    hit.dist = meshHit.dist;
    hit.instId = instId;
    hit.triId = meshHit.triId;

    const auto u = meshHit.u;
    const auto v = meshHit.v;
    const auto w = 1.f - u - v;

    const auto i1 = mesh.indexBuffer[meshHit.triId * 3];
    const auto i2 = mesh.indexBuffer[meshHit.triId * 3 + 1];
    const auto i3 = mesh.indexBuffer[meshHit.triId * 3 + 2];

    hit.x = w * mesh.positionBuffer[i1] + u * mesh.positionBuffer[i2] + v * mesh.positionBuffer[i3];
    hit.n = w * mesh.normalBuffer[i1] + u * mesh.normalBuffer[i2] + v * mesh.normalBuffer[i3];
    hit.uv = optix::make_float2(u, v);

    return hit;
}

MeshHit intersect(const float3 ro, const float3 rd, const TriMesh & mesh)
{
    auto minDistance = std::numeric_limits<float>::max();
    uint32_t minIdx;
    TriangleHit minHit;
    for (size_t i = 0; i < mesh.indexBuffer.size(); i += 3) {
        const auto i1 = mesh.indexBuffer[i];
        const auto i2 = mesh.indexBuffer[i + 1];
        const auto i3 = mesh.indexBuffer[i + 2];
        const auto hit = triIntersect(ro, rd, mesh.positionBuffer[i1], mesh.positionBuffer[i2], mesh.positionBuffer[i3]);
        if (hit.dist > 0 && hit.dist < minDistance) {
            minDistance = hit.dist;
            minIdx = i;
            minHit = hit;
        }
    }

    if (minDistance <= 0.f || minDistance == std::numeric_limits<float>::max())
        return{};

    return{ minHit, minIdx / 3 };
}

Hit Sphere::makeHit(uint32_t instId, const SphereHit & sphereHit)
{
    Hit hit;
    hit.dist = sphereHit.dist;
    hit.instId = instId;
    hit.x = sphereHit.x;
    hit.n = optix::normalize(hit.x - center);
    hit.uv = optix::float2{ 0.f, 0.f };
    return hit;
}

SphereHit Sphere::intersectAnalytic(const Ray &r) const
{
    // returns distance, 0 if nohit
    float3 op = center - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, eps = 1e-4, b = dot(op, r.d), det = b*b - dot(op, op) + radius*radius;
    if (det < 0) return{}; else det = sqrt(det);
    const auto dist = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    if (dist > 0) {
        return{ dist, r.o + r.d*dist };
    }
    return{};
}
