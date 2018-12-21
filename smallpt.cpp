#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>
#include <chrono>

#include <optixu/optixu_math_namespace.h>

#include "ThreadUtils.h"

using hr_clock = std::chrono::high_resolution_clock;

std::uniform_real_distribution<float> distr(0.0, 1.0);
float rand_float(std::mt19937 & generator) {
    return distr(generator);
}

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
    float distance;
    float u, v;
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

float intersect(const optix::float3 ro, const optix::float3 rd, const TriMesh & mesh, optix::float3 & x, optix::float3 & n, optix::float2 & uv)
{
    auto minDistance = std::numeric_limits<float>::max();
    uint32_t minIdx;
    TriangleHit minHit;
    for (size_t i = 0; i < mesh.indexBuffer.size(); i += 3) {
        const auto i1 = mesh.indexBuffer[i];
        const auto i2 = mesh.indexBuffer[i + 1];
        const auto i3 = mesh.indexBuffer[i + 2];
        const auto hit = triIntersect(ro, rd, mesh.positionBuffer[i1], mesh.positionBuffer[i2], mesh.positionBuffer[i3]);
        if (hit.distance > 0 && hit.distance < minDistance) {
            minDistance = hit.distance;
            minIdx = i;
            minHit = hit;
        }
    }

    if (minDistance <= 0.f || minDistance == std::numeric_limits<float>::max())
        return 0.f;

    x = ro + minDistance * rd;
    const auto i1 = mesh.indexBuffer[minIdx];
    const auto i2 = mesh.indexBuffer[minIdx + 1];
    const auto i3 = mesh.indexBuffer[minIdx + 2];

    n = (1 - minHit.u - minHit.v) * mesh.normalBuffer[i1] + minHit.u * mesh.normalBuffer[i2] + minHit.v * mesh.normalBuffer[i3];
    uv = optix::float2{ minHit.u, minHit.v };

    return minDistance;
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

    float intersectAnalytic(const Ray &r, optix::float3 & x, optix::float3 & n, optix::float2 & uv) const { // returns distance, 0 if nohit
        optix::float3 op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        float t, eps = 1e-4, b = dot(op, r.d), det = b*b - dot(op, op) + rad*rad;
        if (det < 0) return 0; else det = sqrt(det);
        const auto dist = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
        if (dist > 0) {
            x = r.o + r.d*dist;
            n = optix::normalize(x - p);
            uv = optix::float2{0.f, 0.f};
        }
        return dist;
    }

    float intersectMesh(const Ray &r, optix::float3 & x, optix::float3 & n, optix::float2 & uv) const {
        return ::intersect(r.o, r.d, mesh, x, n, uv);
    }

    float intersect(const Ray &r, optix::float3 & x, optix::float3 & n, optix::float2 & uv) const {
        return intersectMesh(r, x, n, uv);
    }
};
Sphere spheres[] = {//Scene: radius, position, emission, color, material
    Sphere(10, make_float3(50, 40.8, 81.6), make_float3(),make_float3(.75,.25,.25),DIFF),
  //Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
  //Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
  //Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
  //Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt
  //Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
  //Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
  //Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
  //Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
  Sphere(600, make_float3(50,681.6 - .27,81.6),make_float3(1,1,1),  make_float3(), DIFF) //Lite
};

inline float clamp(float x) { return x < 0 ? 0 : x>1 ? 1 : x; }

inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

struct Hit
{
    static inline float inf() { return 1e20; };
    float t = inf();
    int id;
    optix::float3 x;
    optix::float3 n;
    optix::float2 uv;

    explicit operator bool() const {
        return t < inf();
    }
};

inline Hit intersect(const Ray &r) {
    Hit hit;
    float count = sizeof(spheres) / sizeof(Sphere), d;
    optix::float3 x, n;
    optix::float2 uv;
    for (int i = int(count); i--;) {
        if ((d = spheres[i].intersect(r, x, n, uv)) && d < hit.t) {
            hit.t = d;
            hit.id = i;
            hit.x = x;
            hit.n = n;
            hit.uv = uv;
        }
    }
    return hit;
}

optix::float3 radiance_rec(const Ray &r, int depth, std::mt19937 & generator) 
{
    const auto hit = intersect(r);

    if (!hit) return make_float3(); // if miss, return black

    const Sphere &obj = spheres[hit.id];        // the hit object

    optix::float3 x = hit.x + 0.01 * hit.n;
    optix::float3 n = hit.n;
    optix::float3 nl = dot(n, r.d) < 0 ? n : n*-1;
    optix::float3 f = obj.c;

    float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl

    if (++depth > 5) 
    {
        if (rand_float(generator) < p && depth < 1) {
            f = f*(1 / p);
        }
        else {
            return obj.e; //R.R.
        }
    }

    if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
        float r1 = 2 * M_PI*rand_float(generator), r2 = rand_float(generator), r2s = sqrt(r2);
        optix::float3 w = nl, u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1) : make_float3(1)), w)), v = cross(w, u);
        optix::float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
        return obj.e + f * radiance_rec(Ray(x, d), depth, generator);
    }

    if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return obj.e + f * radiance_rec(Ray(x, r.d - n * 2 * dot(n, r.d)), depth, generator);

    Ray reflRay(x, r.d - n * 2 * dot(n, r.d));     // Ideal dielectric REFRACTION

    bool into = dot(n, nl) > 0;                // Ray from outside going in?
    float nc = 1;
    float nt = 1.5;
    float nnt = into ? nc / nt : nt / nc;
    float ddn = dot(r.d, nl);
    float cos2t;

    if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)    // Total internal reflection
        return obj.e + f * radiance_rec(reflRay, depth, generator);

    optix::float3 tdir = normalize(r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t))));

    float a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : dot(tdir, n));
    float Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);

    return obj.e + f * (depth > 2 ? (rand_float(generator) < P ?   // Russian roulette
        radiance_rec(reflRay, depth, generator)*RP : radiance_rec(Ray(x, tdir), depth, generator)*TP) : (radiance_rec(reflRay, depth, generator)*Re + radiance_rec(Ray(x, tdir), depth, generator)*Tr));
}

struct PathContrib
{
    uint32_t pixelIdx;
    uint32_t pathIdx;
    optix::float3 weight;
    Ray currentRay;
    uint32_t depth;
};

int main(int argc, char *argv[]) {
    const auto start = hr_clock::now();

    fprintf(stderr, "Starting rendering\n");

    const int w = 256;
    const int h = 256;
    const int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    const Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // cam pos, dir
    const optix::float3 cx = make_float3(w*.5135 / h), cy = normalize(cross(cx, cam.d))*.5135;
    
    const auto threadCount = shn::getSystemThreadCount() - 2;

    std::vector<PathContrib> pathBuffer;
    const int jitterSize = 2;
    pathBuffer.resize(h * w * jitterSize * jitterSize * samps);

    const auto camRaysFuture = shn::asyncParallelLoop(h, threadCount, [&](auto y, auto threadId) // Loop over image rows
    {
        std::mt19937 generator{ uint32_t(y*y*y) };
        for (unsigned short x = 0; x < w; x++) {   // Loop cols
            const int pixelIdx = (h - y - 1)*w + x;
            for (int sy = 0; sy < jitterSize; sy++) {     // 2x2 subpixel rows
                for (int sx = 0; sx < jitterSize; sx++) {        // 2x2 subpixel cols
                    const int jitterIdx = sx + sy * jitterSize;
                    optix::float3 r = make_float3();
                    for (int s = 0; s < samps; s++) {
                        float r1 = 2 * rand_float(generator), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        float r2 = 2 * rand_float(generator), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        optix::float3 d = cx*(((sx + .5 + dx) / 2 + x) / w - .5) +
                            cy*(((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        const Ray cameraRay(Ray(cam.o + d * 140, normalize(d))); // Camera rays are pushed forward to start in interior

                        const auto pathIdx = pixelIdx * jitterSize * jitterSize * samps + jitterIdx * samps + s;
                        auto & path = pathBuffer[pathIdx];
                        path.currentRay = cameraRay;
                        path.pathIdx = pathIdx;
                        path.pixelIdx = pixelIdx;
                        path.weight = optix::float3{ 1,1,1 };
                        path.depth = 0;
                    }
                }
            }
        }
    });

    camRaysFuture.wait();

    const auto pixelCount = w * h;
    std::vector<optix::float3> c(w * h, make_float3());
    std::atomic_uint pixelCounter = 0;
    const auto renderFuture = shn::asyncParallelLoop(pixelCount, threadCount, [&](auto pixelIdx, auto threadId)
    {
        std::mt19937 generator{ uint32_t(pixelIdx) * 12345 };
        const auto pathOffset = pixelIdx * jitterSize * jitterSize * samps;
        optix::float3 r = make_float3();
        for (auto pathIdx = pathOffset; pathIdx < pathBuffer.size() && pathBuffer[pathIdx].pixelIdx == pixelIdx; ++pathIdx)
        {
            auto & path = pathBuffer[pathIdx];
            r = r + radiance_rec(path.currentRay, path.depth, generator);
        }
        c[pixelIdx] = r / float(samps * jitterSize * jitterSize);
        ++pixelCounter;
    });

    while (renderFuture.wait_for(std::chrono::milliseconds(33)) != std::future_status::ready)
    {
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * pixelCounter / pixelCount);
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(hr_clock::now() - start);

    fprintf(stderr, "\nElapsed time: %d ms\n", elapsed.count());

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w*h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
