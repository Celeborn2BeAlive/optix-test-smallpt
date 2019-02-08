#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>
#include <chrono>
#include <numeric>

#include <optix_prime/optix_prime.h>

#include "ThreadUtils.h"
#include "vector.h"
#include "maths.h"
#include "scene.h"

using hr_clock = std::chrono::high_resolution_clock;

inline float3 int2color(int32_t n)
{
    float3 x = float(n + 1) * optix::make_float3(12.9898f, 78.233f, 56.128f);
    x = optix::make_float3(sin(x.x), sin(x.y), sin(x.z)) * 43758.5453f;
    return optix::make_float3(x.x - int32_t(x.x), x.y - int32_t(x.y), x.z - int32_t(x.z));
}

Sphere spheres[] = {
    Sphere(10, make_float3(50, 40.8, 81.6), make_float3(0, 0, 0),make_float3(.75,.25,.25),DIFF),
    Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(1,1,1),  make_float3(0, 0, 0), DIFF) //Lite
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

inline int toInt(float x) { return int(pow(clamp(x, 0.f, 1.f), 1 / 2.2) * 255 + .5); }

inline Hit intersectGlobalSpheres(const Ray &r) {
    using SceneHitT = decltype(spheres[0].intersect(r));
    const size_t count = sizeof(spheres) / sizeof(Sphere);
    SceneHitT nearestHit;
    size_t nearestInst;
    for (size_t i = 0; i < count; ++i) {
        const auto currentHit = spheres[i].intersect(r);
        if (currentHit.dist > 0.f && currentHit.dist < nearestHit.dist) {
            nearestHit = currentHit;
            nearestInst = i;
        }
    }
    if (nearestHit.dist == inf)
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
    float3 weight;
    Ray currentRay;
    uint32_t depth;

    PathContrib() = default;

    PathContrib(size_t pixelIdx, float3 weight, Ray currentRay, uint32_t depth):
        pixelIdx{ pixelIdx }, weight(weight), currentRay{ currentRay }, depth{ depth }
    {}
};

PathContrib extend(PathContrib path, const Ray & newRay, const float3 & weightFactor)
{
    return{ path.pixelIdx, path.weight * weightFactor, newRay, path.depth + 1 };
}

void flipY(int w, int h, float3 * c)
{
    for (size_t rowIdx = 0; rowIdx < h / 2; ++rowIdx)
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            std::swap(c[(h - rowIdx - 1) * w + colIdx], c[rowIdx * w + colIdx]);
        }
    }
}

void writeImage(int w, int h, const float3 * c)
{
    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w*h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}

void cpuIntersectGlobalSpheres(const PathContrib * pathBuffer, size_t pathCount, Hit * hits)
{
    for (size_t pathIdx = 0; pathIdx < pathCount; ++pathIdx)
    {
        auto & path = pathBuffer[pathIdx];
        const auto & r = path.currentRay;
        hits[pathIdx] = intersectGlobalSpheres(r);
    }
}

size_t shadePaths(const Material * materials, const PathContrib * pathBuffer, size_t pathCount, const Hit * hitBuffer,
    Vector<PathContrib> & nextPaths, float3 * outColor, std::mt19937 & generator)
{
    const std::uniform_real_distribution<float> randFloat(0.0, 1.0);

    size_t currentNextPathIdx = 0;
    for (size_t pathIdx = 0; pathIdx < pathCount; ++pathIdx)
    {
        auto & path = pathBuffer[pathIdx];

        const auto & r = path.currentRay;

        const auto hit = hitBuffer[pathIdx];

        if (!hit) continue; // Here we could accumulate path.weight * envContrib

        const auto & material = materials[hit.instId];

        float3 x = hit.x + 0.02 * hit.n;
        float3 n = hit.n;
        float3 nl = dot(n, r.d) < 0 ? n : n*-1;
        float3 f = material.color;

        const float p = optix::max(optix::max(f.x, f.y), f.z);

        //outColor[path.pixelIdx] += path.weight * material.emission;
        outColor[path.pixelIdx] += nl;
        //outColor[path.pixelIdx] += optix::make_float3(hit.uv.x, hit.uv.y, 0);
        //outColor[path.pixelIdx] += int2color(hit.triId);
        continue;

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
        const auto splitCount = material.refl != REFR ? 1 : (depth <= 2) ? 2 : 1;

        if (currentNextPathIdx + (splitCount - 1) >= nextPaths.size())
        {
            nextPaths.resize(1 + nextPaths.size() * 2);
        }

        if (material.refl == DIFF)
        {
            float r1 = 2 * M_PI*randFloat(generator), r2 = randFloat(generator), r2s = sqrt(r2);
            float3 w = nl, u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w)), v = cross(w, u);
            float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));

            nextPaths[currentNextPathIdx++] = extend(path, Ray(x, d), f);
            continue;
        }

        const Ray reflRay(x, r.d - n * 2 * dot(n, r.d));
        if (material.refl == SPEC)
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

        const float3 tdir = normalize(r.d*nnt - n*((into ? 1 : -1)*(ddn * nnt + sqrt(cos2t))));

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
    Vector<float3> c;
    c.resize(pixelCount, make_float3(0, 0, 0));

    const int jitterSize = 2;
    const auto sampleCountPerPixel = jitterSize * jitterSize * samps;

    Vector<Material> materials;
    for (size_t i = 0; i < sphereCount; ++i)
        materials.emplace_back(spheres[i].material);

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

        Vector<PathContrib> pathBuffer;
        pathBuffer.resize_no_construct(w * sampleCountPerPixel);

        // Sample all camera rays to initialize paths
        foreachSampleInRow(rowIdx, [&](SampleIndex sampleIndex)
        {
            const float r1 = 2 * randFloat(generator);
            const float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            const float r2 = 2 * randFloat(generator);
            const float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            const float3 d = cx*(((sampleIndex.groupColumn + .5 + dx) / jitterSize + sampleIndex.pixelColumn) / w - .5) +
                cy*(((sampleIndex.groupRow + .5 + dy) / jitterSize + sampleIndex.pixelRow) / h - .5) + cam.d;
            const Ray cameraRay(Ray(cam.o + d * 140, normalize(d))); // Camera rays are pushed forward to start in interior

            auto & path = pathBuffer[sampleIndex.pixelColumn * sampleCountPerPixel + sampleIndex.indexInPixel];
            path.currentRay = cameraRay;
            path.pixelIdx = sampleIndex.pixelIdx;
            path.weight = float3{ 1,1,1 };
            path.depth = 0;
        });

        Vector<PathContrib> swapPathBuffer;
        swapPathBuffer.resize_no_construct(w * sampleCountPerPixel);
        Vector<Hit> hitBuffer;
        hitBuffer.resize_no_construct(pathBuffer.size());

        auto pathCount = pathBuffer.size();

        while (pathCount > 0)
        {
            hitBuffer.resize_no_construct(pathCount);
            cpuIntersectGlobalSpheres(pathBuffer.data(), pathCount, hitBuffer.data());

            pathCount = shadePaths(materials.data(), pathBuffer.data(), pathCount, hitBuffer.data(), swapPathBuffer, c.data(), generator);
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

    float3 origin;
    float  tmin;
    float3 dir;
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

template<typename Begin, typename End, typename Functor>
auto map(Begin begin, End && end, Functor && f) -> Vector<decltype(f(*begin))>
{
    using RetT = decltype(f(*begin));
    Vector<RetT> v;
    v.resize_no_construct(end - begin);
    for (size_t i = 0; begin != end; ++begin)
        v[i++] = f(*begin);
    return v;
}

struct CPUIntersector
{
    CPUIntersector(size_t threadCount) : m_threadCount(threadCount)
    {
    }

    void addTriangleMesh(TriMesh mesh)
    {
        m_meshes.emplace_back(std::move(mesh));
    }

    void build()
    {

    }

    Hit intersect(const Ray & r)
    {
        MeshHit nearestHit;
        size_t nearestInst;
        for (size_t i = 0; i < m_meshes.size(); ++i) {
            const auto currentHit = ::intersect(r.o, r.d, m_meshes[i]);
            if (currentHit.dist > 0.f && currentHit.dist < nearestHit.dist) {
                nearestHit = currentHit;
                nearestInst = i;
            }
        }
        if (nearestHit.dist == inf)
            return{};

        return makeHit(nearestInst, m_meshes[nearestInst], nearestHit);
    }

    Vector<Hit> traceRays(const PathContrib * paths, size_t pathCount)
    {
        Vector<Hit> hits;
        hits.resize_no_construct(pathCount);
        shn::syncParallelLoop(pathCount, m_threadCount, [&](auto pathIdx, auto threadIdx)
        {
            hits[pathIdx] = intersect(paths[pathIdx].currentRay);
        });
        return hits;
    }

    size_t m_threadCount;
    Vector<TriMesh> m_meshes;
};

struct OptixIntersector
{
    OptixIntersector(size_t threadCount): m_threadCount(threadCount)
    {
        rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &context);
        unsigned int device = 0;
        rtpContextSetCudaDeviceNumbers(context, 1, &device);
    }

    ~OptixIntersector()
    {
        rtpContextDestroy(context);
    }

    void addTriangleMesh(const TriMesh & mesh)
    {
        m_meshes.emplace_back(&mesh);

        models.emplace_back();
        auto & model = models.back();

        vertexBuffers.emplace_back();
        auto & vertexBuffer = vertexBuffers.back();

        indexBuffers.emplace_back();
        auto & indexBuffer = indexBuffers.back();

        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, (void*) mesh.indexBuffer.data(), &indexBuffer);
        rtpBufferDescSetRange(indexBuffer, 0, mesh.triangleCount());

        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, (void*)mesh.positionBuffer.data(), &vertexBuffer);
        rtpBufferDescSetRange(vertexBuffer, 0, mesh.positionBuffer.size());

        rtpModelCreate(context, &model);
        rtpModelSetTriangles(model, indexBuffer, vertexBuffer);
        rtpModelUpdate(model, 0);
        rtpModelFinish(model);

        matrices.emplace_back(optix::make_float4(1, 0, 0, 0));
        matrices.emplace_back(optix::make_float4(0, 1, 0, 0));
        matrices.emplace_back(optix::make_float4(0, 0, 1, 0));
    }

    void build()
    {
        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_INSTANCE_MODEL, RTP_BUFFER_TYPE_HOST, models.data(), &sceneInstanceBuffer);
        rtpBufferDescSetRange(sceneInstanceBuffer, 0, models.size());

        rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3, RTP_BUFFER_TYPE_HOST, matrices.data(), &sceneTransformBuffer);
        rtpBufferDescSetRange(sceneTransformBuffer, 0, models.size());

        rtpModelCreate(context, &sceneModel);
        rtpModelSetInstances(sceneModel, sceneInstanceBuffer, sceneTransformBuffer);
        rtpModelUpdate(sceneModel, 0);
        rtpModelFinish(sceneModel);
    }

    Vector<Hit> convertHits(const OptixHit * hits, size_t count)
    {
        Vector<Hit> newHits;
        newHits.resize_no_construct(count);
        shn::syncParallelLoop(count, m_threadCount, [&](size_t i, size_t threadId)
        {
            if (hits[i].t <= 0.f)
            {
                newHits[i] = {};
                return;
            }

            // Optix is using the convention P = uA + vB + (1 - u - v)C
            // while our convention is P = (1 - u - v)A + uB + vC
            // so we must use u := optix_v and v := (1 - optix_u - optix_v)
            MeshHit meshHit{ TriangleHit{ hits[i].t, hits[i].v, (1 - hits[i].u - hits[i].v) }, uint32_t(hits[i].triId) };
            newHits[i] = makeHit(hits[i].instId, *m_meshes[hits[i].instId], meshHit);
        });
        return newHits;
    }

    Vector<Hit> traceRays(const PathContrib * paths, size_t pathCount)
    {
        Vector<OptixRay> optixRays;
        optixRays.resize_no_construct(pathCount);
        Vector<OptixHit> optixHits;
        optixHits.resize_no_construct(pathCount);

        shn::syncParallelLoop(pathCount, m_threadCount, [&](auto pathIdx, auto threadId)
        {
            auto & optixRay = optixRays[pathIdx];
            const auto & path = paths[pathIdx];
            optixRay.origin = path.currentRay.o;
            optixRay.tmin = 0;
            optixRay.dir = path.currentRay.d;
            optixRay.tmax = inf;
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

        const auto convertedHits = convertHits(optixHits.data(), pathCount);

        //Vector<Hit> hits;
        //hits.resize_no_construct(pathCount);
        //shn::syncParallelLoop(pathCount, m_threadCount, [&](auto pathIdx, auto threadIdx)
        //{
        //    cpuIntersect(&paths[pathIdx], 1, &hits[pathIdx]);
        //});

        //for (size_t i = 0; i < pathCount; ++i)
        //{
        //    if (hits[i].triId != convertedHits[i].triId)
        //    {
        //        std::cerr << i << ", " << hits[i].triId << ", " << convertedHits[i].triId << std::endl;
        //    }
        //}

        return convertedHits;
    }

    size_t m_threadCount;
    RTPcontext context;

    // Meshs
    Vector<RTPmodel> models;
    Vector<optix::float4> matrices;
    Vector<RTPbufferdesc> vertexBuffers;
    Vector<RTPbufferdesc> indexBuffers;

    RTPbufferdesc sceneInstanceBuffer;
    RTPbufferdesc sceneTransformBuffer;
    RTPmodel sceneModel;

    Vector<const TriMesh*> m_meshes;
};

int main(int argc, char *argv[]) 
{
    TriMesh triangle;
    triangle.positionBuffer = { make_float3(-0.5, -0.5, -1), make_float3(0.5, -0.5, -1), make_float3(0, 0.5, -1) };
    triangle.normalBuffer = { make_float3(1, 0, 0), make_float3(0, 1, 0), make_float3(0, 0, 1) };
    triangle.indexBuffer = { 0, 1, 2 };

    Material material{ make_float3(1, 0, 0), make_float3(0, 0, 0), DIFF };

    const auto threadCount = shn::getSystemThreadCount() - 2;
    OptixIntersector intersector{ threadCount };

    /*for (size_t i = 0; i < sphereCount; ++i)
    {
        intersector.addTriangleMesh(spheres[i].mesh);
    }
    intersector.build();

    Vector<Material> materials;
    for (size_t i = 0; i < sphereCount; ++i)
        materials.emplace_back(spheres[i].material);*/

    intersector.addTriangleMesh(triangle);
    intersector.build();
    Vector<Material> materials = { material };

    const auto start = hr_clock::now();

    fprintf(stderr, "Starting rendering\n");

    const int w = 256;
    const int h = 256;
    const int sampleCountPerJitterCell = argc == 2 ? atoi(argv[1]) / 4 : 16; // # samples
    const Ray cam(make_float3(0, 0, 0), normalize(make_float3(0, 0, -1)));

    const auto cx = make_float3(1, 0, 0);
    const auto cy = normalize(cross(cx, cam.d));

    const auto pixelCount = w * h;
    Vector<float3> c;
    c.resize(pixelCount, make_float3(0, 0, 0));

    const int jitterSize = 2; // each pixel is virtually subdivided in jitterSize * jitterSize square cells, for better sampling
    const auto sampleCountPerPixel = jitterSize * jitterSize * sampleCountPerJitterCell;

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
                    for (size_t s = 0; s < sampleCountPerJitterCell; ++s)
                    {
                        const auto indexInPixel = groupIdx * sampleCountPerJitterCell + s;
                        const auto indexInImage = pixelIdx * sampleCountPerPixel + indexInPixel;

                        functor(SampleIndex{ pixelIdx, colIdx, rowIdx, groupIdx, sx, sy, indexInImage, indexInPixel, s });
                    }
                }
            }
        }
    };

    Vector<std::mt19937> generators;
    generators.resize(h);
    
    Vector<PathContrib> pathBuffer;
    pathBuffer.resize_no_construct(w * h * sampleCountPerPixel);
    Vector<size_t> pathOffsetPerRow;
    pathOffsetPerRow.resize_no_construct(h);
    Vector<size_t> pathCountPerRow;
    pathCountPerRow.resize_no_construct(h);

    shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId) // Loop over image rows
    {
        auto & generator = generators[rowIdx];

        generator.seed(uint32_t(rowIdx * rowIdx * rowIdx));

        // Sample all camera rays to initialize paths
        foreachSampleInRow(rowIdx, [&](SampleIndex sampleIndex)
        {
            const auto jitterCellSize = make_float2(1.f / jitterSize, 1.f / jitterSize); // cell size relative to pixel
            const auto pixelSize = make_float2(1.f / w, 1.f / h); // pixel size relative to image

            const auto randNumbers = make_float2(randFloat(generator), randFloat(generator));

            const auto jitteredRandNumbers = (make_float2(sampleIndex.groupColumn, sampleIndex.groupRow) + randNumbers) * jitterCellSize;

            // Should return numbers centered arround [0, 0]
            const auto samplePixelFilter = [](float r1, float r2) -> float2 {
                return 0.5 * make_float2(2 * r1 - 1, 2 * r2 - 1); // box filter in [-0.5, 0.5]
            };

            // Sample relative to pixel space
            const auto jitteredSample = samplePixelFilter(jitteredRandNumbers.x, jitteredRandNumbers.y);

            const auto pixelCenterInRasterSpace = make_float2(sampleIndex.pixelColumn + 0.5f, sampleIndex.pixelRow + 0.5f);

            const auto samplePositionInRasterSpace = pixelCenterInRasterSpace + jitteredSample; // in [0 - filterExtentW, w + filterExtentH] x [0 - filterExtentW, h + filterExtentH]
            const auto samplePositionInNormalizedRasterSpace = samplePositionInRasterSpace * pixelSize; // in [0 - NormalizedFilterExtendW, 1 + NormalizedFilterExtendW] x [0 - NormalizedFilterExtendH, 1 + NormalizedFilterExtendH]

            const auto sampleInClipSpace = 2.f * samplePositionInNormalizedRasterSpace - make_float2(1.f, 1.f);

            const auto d = normalize(sampleInClipSpace.x * cx + sampleInClipSpace.y * cy + cam.d);
            const Ray cameraRay(cam.o, d);

            auto & path = pathBuffer[sampleIndex.indexInImage];
            path.currentRay = cameraRay;
            path.pixelIdx = sampleIndex.pixelIdx;
            path.weight = float3{ 1,1,1 };
            path.depth = 0;
        });

        pathOffsetPerRow[rowIdx] = sampleCountPerPixel * w * rowIdx;
        pathCountPerRow[rowIdx] = sampleCountPerPixel * w;
    });

    size_t pathCount = pathBuffer.size();
    Vector<Vector<PathContrib>> nextPathsPerRow;
    nextPathsPerRow.resize(h);
    for (size_t rowIdx = 0; rowIdx < h; ++rowIdx)
        nextPathsPerRow.resize(w * sampleCountPerPixel);

    while (pathCount > 0)
    {
        std::clog << "Trace rays" << std::endl;
        const auto hits = intersector.traceRays(pathBuffer.data(), pathCount);

        shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId)
        {
            const auto pathOffset = pathOffsetPerRow[rowIdx];
            const auto pathCount = pathCountPerRow[rowIdx];
            if (!pathCount)
                return;
            pathCountPerRow[rowIdx] = shadePaths(materials.data(), &pathBuffer[pathOffset], pathCount, &hits[pathOffset], nextPathsPerRow[rowIdx], c.data(), generators[rowIdx]);
        });

        pathCount = 0;
        for (size_t rowIdx = 0; rowIdx < h; ++rowIdx)
        {
            pathOffsetPerRow[rowIdx] = pathCount;
            pathCount += pathCountPerRow[rowIdx];
        }

        pathBuffer.resize(pathCount);
        shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId)
        {
            const auto offset = pathOffsetPerRow[rowIdx];
            const auto count = pathCountPerRow[rowIdx];
            std::copy(begin(nextPathsPerRow[rowIdx]), begin(nextPathsPerRow[rowIdx]) + pathCountPerRow[rowIdx], begin(pathBuffer) + offset);
        });
    }

    shn::syncParallelLoop(h, threadCount, [&](auto rowIdx, auto threadId) // Loop over image rows
    {
        for (size_t colIdx = 0; colIdx < w; ++colIdx)
        {
            c[rowIdx * w + colIdx] /= sampleCountPerPixel;
        }
    });

    flipY(w, h, c.data());
    writeImage(w, h, c.data());
}