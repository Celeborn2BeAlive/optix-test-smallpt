#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>
#include <chrono>
#include <numeric>

#include <optixu/optixu_math_namespace.h>
#include <optix_prime/optix_prime.h>

#include "ThreadUtils.h"

using hr_clock = std::chrono::high_resolution_clock;

inline optix::float3 make_float3(float x = 0.f, float y = 0.f, float z = 0.f)
{
    return optix::float3{ x, y, z };
}

struct TriMesh {
    std::vector<optix::float3> positionBuffer;
    std::vector<optix::float3> normalBuffer;
    std::vector<uint32_t> indexBuffer;

    size_t triangleCount() const
    {
        return indexBuffer.size() / 3;
    }
};

TriMesh makeSphereTriMesh(const optix::float3 origin, float radius, const uint32_t subdivLongitude = 32)
{
    const auto discLong = subdivLongitude;
    const auto discLat = 2 * discLong;

    float rcpLat = 1.f / discLat, rcpLong = 1.f / discLong;
    float dPhi = M_PIf * 2.f * rcpLat, dTheta = M_PIf * rcpLong;

    std::vector<optix::float3> positionBuffer, normalBuffer;

    for (uint32_t j = 0; j <= discLong; ++j)
    {
        float cosTheta = cos(-M_PI_2f + j * dTheta);
        float sinTheta = sin(-M_PI_2f + j * dTheta);

        for (uint32_t i = 0; i <= discLat; ++i) {
            const optix::float3 coords{
                sin(i * dPhi) * cosTheta,
                sinTheta,
                cos(i * dPhi) * cosTheta
            };

            positionBuffer.emplace_back(origin + radius * coords);
            normalBuffer.emplace_back(coords);
        }
    }

    std::vector<uint32_t> indexBuffer;

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

struct TriangleHit
{
    float dist = -1.f;
    float u, v;

    TriangleHit() = default;

    TriangleHit(float d, float u, float v) : dist{ d }, u{ u }, v{ v } {}
};

// triangle degined by vertices v0, v1 and  v2
// https://iquilezles.org/www/articles/intersectors/intersectors.htm
TriangleHit triIntersect(optix::float3 ro, optix::float3 rd, optix::float3 v0, optix::float3 v1, optix::float3 v2)
{
    using vec3 = optix::float3;

    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = ro - v0;

    vec3  n = cross(v1v0, v2v0);
    vec3  q = cross(rov0, rd);
    float d = 1.0 / dot(rd, n);
    float u = d*dot(-q, v2v0);
    float v = d*dot(q, v1v0);
    float t = d*dot(-n, rov0);

    if (u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0) t = -1.0;

    return { t, u, v };
}

struct Hit
{
    static inline float inf() { return 1e20; };
    float dist = inf();
    uint32_t instId;
    optix::float3 x;
    optix::float3 n;
    optix::float2 uv;

    explicit operator bool() const {
        return dist < inf();
    }
};

struct MeshHit : public TriangleHit
{
    uint32_t triId;

    MeshHit() = default;
    MeshHit(TriangleHit h, uint32_t triId) : TriangleHit{ h }, triId{ triId } {}
};

// Convert a mesh hit to a generic hit
Hit makeHit(uint32_t instId, const TriMesh & mesh, const MeshHit & meshHit)
{
    Hit hit;
    hit.dist = meshHit.dist;
    hit.instId = instId;

    const auto u = meshHit.u;
    const auto v = meshHit.v;
    const auto w = 1.f - u - v;

    const auto i1 = mesh.indexBuffer[meshHit.triId];
    const auto i2 = mesh.indexBuffer[meshHit.triId + 1];
    const auto i3 = mesh.indexBuffer[meshHit.triId + 2];

    hit.x = w * mesh.positionBuffer[i1] + u * mesh.positionBuffer[i2] + v * mesh.positionBuffer[i3];
    hit.n = w * mesh.normalBuffer[i1] + u * mesh.normalBuffer[i2] + v * mesh.normalBuffer[i3];
    hit.uv = optix::make_float2(u, v);

    return hit;
}

MeshHit intersect(const optix::float3 ro, const optix::float3 rd, const TriMesh & mesh)
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

    return { minHit, minIdx };
}

struct Ray { 
    optix::float3 o, d; 
    Ray() = default;
    Ray(optix::float3 o_, optix::float3 d_) : o(o_), d(d_) {} 
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Sphere {
    float rad;       // radius
    optix::float3 p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

    TriMesh mesh;

    Sphere(float rad_, optix::float3 p_, optix::float3 e_, optix::float3 c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_), mesh{makeSphereTriMesh(p_, rad_)} {}

    struct SphereHit
    {
        float dist = -1.f;
        optix::float3 x;

        SphereHit() = default;
        SphereHit(float d, optix::float3 x) : dist{ d }, x(x) {}
    };

    Hit makeHit(uint32_t instId, const SphereHit & sphereHit)
    {
        Hit hit;
        hit.dist = sphereHit.dist;
        hit.instId = instId;
        hit.x = sphereHit.x;
        hit.n = optix::normalize(hit.x - p);
        hit.uv = optix::float2{ 0.f, 0.f };
        return hit;
    }

    Hit makeHit(uint32_t instId, const MeshHit & meshHit)
    {
        return ::makeHit(instId, mesh, meshHit);
    }

    SphereHit intersectAnalytic(const Ray &r) const { // returns distance, 0 if nohit
        optix::float3 op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        float t, eps = 1e-4, b = dot(op, r.d), det = b*b - dot(op, op) + rad*rad;
        if (det < 0) return{}; else det = sqrt(det);
        const auto dist = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
        if (dist > 0) {
            return{ dist, r.o + r.d*dist };
        }
        return{};
    }

    MeshHit intersectMesh(const Ray &r) const {
        return ::intersect(r.o, r.d, mesh);
    }

    auto intersect(const Ray &r) const {
        return intersectMesh(r);
    }
};

Sphere spheres[] = {
    Sphere(10, make_float3(50, 40.8, 81.6), make_float3(),make_float3(.75,.25,.25),DIFF),
    Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(1,1,1),  make_float3(), DIFF) //Lite
};

//Sphere spheres[] = {//Scene: radius, position, emission, color, material
//   //Sphere(10, make_float3(50, 40.8, 81.6), make_float3(),make_float3(.75,.25,.25),DIFF),
//  Sphere(1e5, make_float3(1e5 + 1,40.8,81.6), make_float3(),make_float3(.75,.25,.25),DIFF),//Left
//  Sphere(1e5, make_float3(-1e5 + 99,40.8,81.6),make_float3(),make_float3(.25,.25,.75),DIFF),//Rght
//  Sphere(1e5, make_float3(50,40.8, 1e5),     make_float3(),make_float3(.75,.75,.75),DIFF),//Back
//  Sphere(1e5, make_float3(50,40.8,-1e5 + 170), make_float3(),make_float3(),           DIFF),//Frnt
//  Sphere(1e5, make_float3(50, 1e5, 81.6),    make_float3(),make_float3(.75,.75,.75),DIFF),//Botm
//  Sphere(1e5, make_float3(50,-1e5 + 81.6,81.6),make_float3(),make_float3(.75,.75,.75),DIFF),//Top
//  Sphere(16.5,make_float3(27,16.5,47),       make_float3(),make_float3(1,1,1)*.999, SPEC),//Mirr
//  Sphere(16.5,make_float3(73,16.5,78),       make_float3(),make_float3(1,1,1)*.999, REFR),//Glas
//  Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(1,1,1),  make_float3(), DIFF)
//  //Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(1,1,1),  make_float3(), DIFF) //Lite
//};

const size_t sphereCount = sizeof(spheres) / sizeof(spheres[0]);

inline float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }

inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

inline Hit intersect(const Ray &r) {
    using SceneHitT = decltype(spheres[0].intersect(r));
    const size_t count = sizeof(spheres) / sizeof(Sphere);
    SceneHitT nearestHit;
    nearestHit.dist = Hit::inf();
    size_t nearestInst;
    for (size_t i = 0; i < count; ++i) {
        const auto currentHit = spheres[i].intersect(r);
        if (currentHit.dist > 0.f && currentHit.dist < nearestHit.dist) {
            nearestHit = currentHit;
            nearestInst = i;
        }
    }
    if (nearestHit.dist == Hit::inf())
        return{};

    return spheres[nearestInst].makeHit(nearestInst, nearestHit);
}

struct SampleIndex
{
    // Pixel containing the sample
    size_t pixelIdx;
    size_t pixelColumn; // x coordinate
    size_t pixelRow; // y coordinate
    // Group containing the sample; Each pixel is split in multiple disjoint cells
    // due to jittering; Each cells has a group of samples
    size_t groupIdx;
    size_t groupColumn;
    size_t groupRow;
    // Indices of the sample relative to containers
    size_t indexInImage;
    size_t indexInPixel;
    size_t indexInGroup;

    SampleIndex() = default;

    SampleIndex(
        size_t pixelIdx,
        size_t pixelColumn,
        size_t pixelRow,
        size_t groupIdx,
        size_t groupColumn,
        size_t groupRow,
        size_t indexInImage,
        size_t indexInPixel,
        size_t indexInGroup) : 
        pixelIdx{ pixelIdx }, pixelColumn{ pixelColumn }, pixelRow{ pixelRow },
        groupIdx{ groupIdx }, groupColumn{ groupColumn }, groupRow{ groupRow },
        indexInImage{ indexInImage }, indexInPixel{ indexInPixel }, indexInGroup{ indexInGroup }
    {}
};

struct PathContrib
{
    size_t pixelIdx;
    optix::float3 weight;
    Ray currentRay;
    uint32_t depth;

    PathContrib() = default;

    PathContrib(size_t pixelIdx, optix::float3 weight, Ray currentRay, uint32_t depth):
        pixelIdx{ pixelIdx }, weight(weight), currentRay{ currentRay }, depth{ depth }
    {}
};

PathContrib extend(PathContrib path, const Ray & newRay, const optix::float3 & weightFactor)
{
    return{ path.pixelIdx, path.weight * weightFactor, newRay, path.depth + 1 };
}

void flipY(int w, int h, optix::float3 * c)
{
    for (size_t rowIdx = 0; rowIdx < h / 2; ++rowIdx)
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            std::swap(c[(h - rowIdx - 1) * w + colIdx], c[rowIdx * w + colIdx]);
        }
    }
}

void writeImage(int w, int h, const optix::float3 * c)
{
    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w*h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}

void cpuIntersect(const PathContrib * pathBuffer, size_t pathCount, Hit * hits)
{
    for (size_t pathIdx = 0; pathIdx < pathCount; ++pathIdx)
    {
        auto & path = pathBuffer[pathIdx];
        const auto & r = path.currentRay;
        hits[pathIdx] = intersect(r);
    }
}

size_t shadePaths(const PathContrib * pathBuffer, size_t pathCount, const Hit * hitBuffer, std::vector<PathContrib> & nextPaths, optix::float3 * outColor, std::mt19937 & generator)
{
    const std::uniform_real_distribution<float> randFloat(0.0, 1.0);

    size_t currentNextPathIdx = 0;
    for (size_t pathIdx = 0; pathIdx < pathCount; ++pathIdx)
    {
        auto & path = pathBuffer[pathIdx];

        const auto & r = path.currentRay;

        const auto hit = hitBuffer[pathIdx];

        if (!hit) continue; // Here we could accumulate path.weight * envContrib

        const Sphere &obj = spheres[hit.instId];

        optix::float3 x = hit.x + 0.02 * hit.n;
        optix::float3 n = hit.n;
        optix::float3 nl = dot(n, r.d) < 0 ? n : n*-1;
        optix::float3 f = obj.c;

        const float p = optix::max(optix::max(f.x, f.y), f.z);

        outColor[path.pixelIdx] += path.weight * obj.e;

        const auto depth = path.depth;

        // russian roulette to kill paths after too much bounces
        if (depth > 5)
        {
            if (randFloat(generator) < p)
            {
                f *= (1 / p);
            }
            else
            {
                continue;
            }
        }

        // Maximum number of times the path can split
        const auto splitCount = obj.refl != REFR ? 1 : (depth <= 2) ? 2 : 1;

        if (currentNextPathIdx + (splitCount - 1) >= nextPaths.size())
        {
            nextPaths.resize(1 + nextPaths.size() * 2);
        }

        if (obj.refl == DIFF)
        {
            float r1 = 2 * M_PI*randFloat(generator), r2 = randFloat(generator), r2s = sqrt(r2);
            optix::float3 w = nl, u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1) : make_float3(1)), w)), v = cross(w, u);
            optix::float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));

            nextPaths[currentNextPathIdx++] = extend(path, Ray(x, d), f);
            continue;
        }

        const Ray reflRay(x, r.d - n * 2 * dot(n, r.d));
        if (obj.refl == SPEC)
        {
            nextPaths[currentNextPathIdx++] = extend(path, reflRay, f);
            continue;
        }

        const bool into = dot(n, nl) > 0;                // Ray from outside going in?
        const float nc = 1;
        const float nt = 1.5;
        const float nnt = into ? nc / nt : nt / nc;
        const float ddn = dot(r.d, nl);
        const float cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

        if (cos2t < 0)
        {
            nextPaths[currentNextPathIdx++] = extend(path, reflRay, f);
            continue;
        }

        const optix::float3 tdir = normalize(r.d*nnt - n*((into ? 1 : -1)*(ddn * nnt + sqrt(cos2t))));

        const float a = nt - nc;
        const float b = nt + nc;
        const float R0 = a * a / (b * b);
        const float c = 1 - (into ? -ddn : dot(tdir, n));
        const float c2 = c * c;
        const float Re = R0 + (1 - R0) * c2 * c2 * c;
        const float Tr = 1 - Re;

        if (depth <= 2)
        {
            // Split before two bounces
            nextPaths[currentNextPathIdx++] = extend(path, reflRay, f * Re);
            nextPaths[currentNextPathIdx++] = extend(path, Ray(x, tdir), f * Tr);
            continue;
        }

        const float P = .25 + .5 * Re;
        if (randFloat(generator) < P)
        {
            nextPaths[currentNextPathIdx++] = extend(path, reflRay, f * Re / P);
            continue;
        }

        nextPaths[currentNextPathIdx++] = extend(path, Ray(x, tdir), f * Tr / (1.f - P));
    }

    return currentNextPathIdx;
}

int cpuRender(int argc, char *argv[]) {
    const auto start = hr_clock::now();

    fprintf(stderr, "Starting rendering\n");

    const int w = 256;
    const int h = 256;
    const int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    const Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir
    const auto cx = make_float3(w*.5135 / h);
    const auto cy = normalize(cross(cx, cam.d))*.5135;
    const auto threadCount = shn::getSystemThreadCount() - 2;
    const auto pixelCount = w * h;
    std::vector<optix::float3> c(pixelCount, make_float3());

    const int jitterSize = 2;
    const auto sampleCountPerPixel = jitterSize * jitterSize * samps;

    std::uniform_real_distribution<float> randFloat(0.0, 1.0);

    const auto foreachSampleInRow = [&](auto rowIdx, auto functor)
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            const auto pixelIdx = rowIdx * w + colIdx;
            for (size_t sy = 0; sy < jitterSize; ++sy)
            {
                for (size_t sx = 0; sx < jitterSize; ++sx)
                {
                    const auto groupIdx = sy * jitterSize + sx;
                    for (size_t s = 0; s < samps; ++s)
                    {
                        const auto indexInPixel = groupIdx * samps + s;
                        const auto indexInImage = pixelIdx * sampleCountPerPixel + indexInPixel;

                        functor(SampleIndex{ pixelIdx, colIdx, rowIdx, groupIdx, sx, sy, indexInImage, indexInPixel, s });
                    }
                }
            }
        }
    };

    std::atomic_uint pixelCounter = 0;
    const auto renderFuture = shn::asyncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId) // Loop over image rows
    {
        std::mt19937 generator{ uint32_t(rowIdx * rowIdx * rowIdx) };

        std::vector<PathContrib> pathBuffer(w * sampleCountPerPixel);

        // Sample all camera rays to initialize paths
        foreachSampleInRow(rowIdx, [&](SampleIndex sampleIndex)
        {
            const float r1 = 2 * randFloat(generator);
            const float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            const float r2 = 2 * randFloat(generator);
            const float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            const optix::float3 d = cx*(((sampleIndex.groupColumn + .5 + dx) / jitterSize + sampleIndex.pixelColumn) / w - .5) +
                cy*(((sampleIndex.groupRow + .5 + dy) / jitterSize + sampleIndex.pixelRow) / h - .5) + cam.d;
            const Ray cameraRay(Ray(cam.o + d * 140, normalize(d))); // Camera rays are pushed forward to start in interior

            auto & path = pathBuffer[sampleIndex.pixelColumn * sampleCountPerPixel + sampleIndex.indexInPixel];
            path.currentRay = cameraRay;
            path.pixelIdx = sampleIndex.pixelIdx;
            path.weight = optix::float3{ 1,1,1 };
            path.depth = 0;
        });

        std::vector<PathContrib> swapPathBuffer(w * sampleCountPerPixel);
        std::vector<Hit> hitBuffer(pathBuffer.size());

        auto pathCount = pathBuffer.size();

        while (pathCount > 0)
        {
            hitBuffer.resize(pathCount);
            cpuIntersect(pathBuffer.data(), pathCount, hitBuffer.data());

            pathCount = shadePaths(pathBuffer.data(), pathCount, hitBuffer.data(), swapPathBuffer, c.data(), generator);
            std::swap(pathBuffer, swapPathBuffer);
        }

        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            c[rowIdx * w + colIdx] /= sampleCountPerPixel;
        }

        pixelCounter += w;
    });

    while (renderFuture.wait_for(std::chrono::milliseconds(33)) != std::future_status::ready)
    {
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * pixelCounter / pixelCount);
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(hr_clock::now() - start);

    fprintf(stderr, "\nElapsed time: %d ms\n", elapsed.count());

    flipY(w, h, c.data());
    writeImage(w, h, c.data());

    return 0;
}

#define CHK_PRIME( code )                                                      \
{                                                                              \
  RTPresult res__ = code;                                                      \
  if( res__ != RTP_SUCCESS )                                                   \
  {                                                                            \
  const char* err_string;                                                      \
  rtpContextGetLastErrorString( context, &err_string );                        \
  std::cerr << "Error at <<" __FILE__ << "(" << __LINE__ << "): "              \
            << err_string                                                      \
  << "' (" << res__ << ")" << std::endl;                                       \
  exit(1);                                                                     \
  }                                                                            \
}

struct OptixRay
{
    static const RTPbufferformat format = RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX;

    optix::float3 origin;
    float  tmin;
    optix::float3 dir;
    float  tmax;
};

struct OptixHit
{
    static const RTPbufferformat format = RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V;

    float t;
    int   triId;
    int   instId;
    float u;
    float v;
};

std::vector<Hit> convertHits(const OptixHit * hits, size_t count, size_t threadCount)
{
    std::vector<Hit> newHits(count);
    shn::syncParallelLoop(count, threadCount, [&](size_t i, size_t threadId)
    {
        if (hits[i].t <= 0.f)
            return;

        MeshHit meshHit{ TriangleHit{hits[i].t, hits[i].u, hits[i].v}, uint32_t(hits[i].triId) };
        newHits[i] = spheres[hits[i].instId].makeHit(hits[i].instId, meshHit);
    });
    return newHits;
}

int main(int argc, char *argv[]) {
    RTPcontext context;
    rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &context);

    unsigned int device = 0;
    rtpContextSetCudaDeviceNumbers(context, 1, &device);

    std::vector<RTPmodel> models(sphereCount);
    std::vector<optix::float4> matrices(sphereCount * 3);
    std::vector<RTPbufferdesc> vertexBuffers(sphereCount);
    std::vector<RTPbufferdesc> indexBuffers(sphereCount);

    for (size_t i = 0; i < sphereCount; ++i)
    {
        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, spheres[i].mesh.indexBuffer.data(), &indexBuffers[i]);
        rtpBufferDescSetRange(indexBuffers[i], 0, spheres[i].mesh.triangleCount());

        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, spheres[i].mesh.positionBuffer.data(), &vertexBuffers[i]);
        rtpBufferDescSetRange(vertexBuffers[i], 0, spheres[i].mesh.positionBuffer.size());

        rtpModelCreate(context, &models[i]);
        rtpModelSetTriangles(models[i], indexBuffers[i], vertexBuffers[i]);
        rtpModelUpdate(models[i], 0);
        rtpModelFinish(models[i]);

        matrices[i * 3 + 0] = optix::make_float4(1, 0, 0, 0);
        matrices[i * 3 + 1] = optix::make_float4(0, 1, 0, 0);
        matrices[i * 3 + 2] = optix::make_float4(0, 0, 1, 0);
    }

    RTPbufferdesc sceneInstanceBuffer;
    rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_INSTANCE_MODEL, RTP_BUFFER_TYPE_HOST, models.data(), &sceneInstanceBuffer);
    rtpBufferDescSetRange(sceneInstanceBuffer, 0, models.size());

    RTPbufferdesc sceneTransformBuffer;
    rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, matrices.data(), &sceneTransformBuffer);
    rtpBufferDescSetRange(sceneTransformBuffer, 0, models.size());

    RTPmodel sceneModel;
    rtpModelCreate(context, &sceneModel);
    rtpModelSetInstances(sceneModel, sceneInstanceBuffer, sceneTransformBuffer);
    rtpModelUpdate(sceneModel, 0);
    rtpModelFinish(sceneModel);

    const auto start = hr_clock::now();

    fprintf(stderr, "Starting rendering\n");

    const int w = 256;
    const int h = 256;
    const int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    const Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir
    const auto cx = make_float3(w*.5135 / h);
    const auto cy = normalize(cross(cx, cam.d))*.5135;
    const auto threadCount = shn::getSystemThreadCount() - 2;
    const auto pixelCount = w * h;
    std::vector<optix::float3> c(pixelCount, make_float3());

    const int jitterSize = 2;
    const auto sampleCountPerPixel = jitterSize * jitterSize * samps;

    std::uniform_real_distribution<float> randFloat(0.0, 1.0);

    const auto foreachSampleInRow = [&](auto rowIdx, auto functor)
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            const auto pixelIdx = rowIdx * w + colIdx;
            for (size_t sy = 0; sy < jitterSize; ++sy)
            {
                for (size_t sx = 0; sx < jitterSize; ++sx)
                {
                    const auto groupIdx = sy * jitterSize + sx;
                    for (size_t s = 0; s < samps; ++s)
                    {
                        const auto indexInPixel = groupIdx * samps + s;
                        const auto indexInImage = pixelIdx * sampleCountPerPixel + indexInPixel;

                        functor(SampleIndex{ pixelIdx, colIdx, rowIdx, groupIdx, sx, sy, indexInImage, indexInPixel, s });
                    }
                }
            }
        }
    };

    std::vector<std::mt19937> generators(h);
    
    std::vector<PathContrib> pathBuffer(w * h * sampleCountPerPixel);
    std::vector<size_t> pathOffsetPerRow(h);
    std::vector<size_t> pathCountPerRow(h);

    shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId) // Loop over image rows
    {
        auto & generator = generators[rowIdx];

        generator.seed(uint32_t(rowIdx * rowIdx * rowIdx));

        // Sample all camera rays to initialize paths
        foreachSampleInRow(rowIdx, [&](SampleIndex sampleIndex)
        {
            const float r1 = 2 * randFloat(generator);
            const float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            const float r2 = 2 * randFloat(generator);
            const float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            const optix::float3 d = cx*(((sampleIndex.groupColumn + .5 + dx) / jitterSize + sampleIndex.pixelColumn) / w - .5) +
                cy*(((sampleIndex.groupRow + .5 + dy) / jitterSize + sampleIndex.pixelRow) / h - .5) + cam.d;
            const Ray cameraRay(Ray(cam.o + d * 140, normalize(d))); // Camera rays are pushed forward to start in interior

            auto & path = pathBuffer[sampleIndex.indexInImage];
            path.currentRay = cameraRay;
            path.pixelIdx = sampleIndex.pixelIdx;
            path.weight = optix::float3{ 1,1,1 };
            path.depth = 0;
        });

        pathOffsetPerRow[rowIdx] = sampleCountPerPixel * w * rowIdx;
        pathCountPerRow[rowIdx] = sampleCountPerPixel * w;
    });

    size_t pathCount = pathBuffer.size();
    std::vector<std::vector<PathContrib>> nextPathsPerRow(h);
    for (size_t rowIdx = 0; rowIdx < h; ++rowIdx)
        nextPathsPerRow.resize(w * sampleCountPerPixel);

    while (pathCount > 0)
    {
        std::vector<OptixRay> optixRays(pathCount);
        std::vector<OptixHit> optixHits(pathCount);

        shn::syncParallelLoop(pathCount, threadCount, [&](auto pathIdx, auto threadId)
        {
            auto & optixRay = optixRays[pathIdx];
            const auto & path = pathBuffer[pathIdx];
            optixRay.origin = path.currentRay.o;
            optixRay.tmin = 0;
            optixRay.dir = path.currentRay.d;
            optixRay.tmax = Hit::inf();
        });

        RTPbufferdesc raysDesc;
        rtpBufferDescCreate(context, OptixRay::format, RTP_BUFFER_TYPE_HOST, optixRays.data(), &raysDesc);
        rtpBufferDescSetRange(raysDesc, 0, optixRays.size());

        RTPbufferdesc hitsDesc;
        rtpBufferDescCreate(context, OptixHit::format, RTP_BUFFER_TYPE_HOST, optixHits.data(), &hitsDesc);
        rtpBufferDescSetRange(hitsDesc, 0, optixHits.size());

        RTPquery query;
        rtpQueryCreate(sceneModel, RTP_QUERY_TYPE_CLOSEST, &query);
        rtpQuerySetRays(query, raysDesc);
        rtpQuerySetHits(query, hitsDesc);
        rtpQueryExecute(query, 0);

        const auto hits = convertHits(optixHits.data(), pathCount, threadCount);

        shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId)
        {
            const auto pathOffset = pathOffsetPerRow[rowIdx];
            const auto pathCount = pathCountPerRow[rowIdx];
            if (!pathCount)
                return;
            pathCountPerRow[rowIdx] = shadePaths(&pathBuffer[pathOffset], pathCount, &hits[pathOffset], nextPathsPerRow[rowIdx], c.data(), generators[rowIdx]);
        });

        size_t totalPathCount = 0;
        for (size_t rowIdx = 0; rowIdx < h; ++rowIdx)
        {
            pathOffsetPerRow[rowIdx] = totalPathCount;
            totalPathCount += pathCountPerRow[rowIdx];
        }

        pathBuffer.resize(totalPathCount);
        shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId)
        {
            const auto offset = pathOffsetPerRow[rowIdx];
            const auto count = pathCountPerRow[rowIdx];
            std::copy(begin(nextPathsPerRow[rowIdx]), begin(nextPathsPerRow[rowIdx]) + pathCountPerRow[rowIdx], begin(pathBuffer) + offset);
        });

        pathCount = totalPathCount;
    }

    shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId) // Loop over image rows
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            c[rowIdx * w + colIdx] /= sampleCountPerPixel;
        }
    });

    rtpContextDestroy(context);

    flipY(w, h, c.data());
    writeImage(w, h, c.data());
}