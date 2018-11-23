#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include <random>
#include <chrono>

#include <optixu/optixu_math_namespace.h>

#include "ThreadUtils.h"

using hr_clock = std::chrono::high_resolution_clock;

std::uniform_real_distribution<double> distr(0.0, 1.0);
double rand_float(std::mt19937 & generator) {
    return distr(generator);
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

TriMesh makeSphere(const optix::float3 origin, float radius, const uint32_t subdivLongitude = 4)
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

struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
    double x, y, z;                  // position, also color (r,g,b)
    Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
    Vec(optix::float3 v) : Vec(v.x, v.y, v.z) {}
    Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
    Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
    Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
    Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
    Vec& norm() { return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
    double dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; }
    Vec operator%(const Vec&b) const { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); } // cross product
    operator optix::float3() const { return optix::float3{ float(x), float(y), float(z) }; }
};

float intersect(const optix::float3 ro, const optix::float3 rd, const TriMesh & mesh, Vec & x, Vec & n)
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
            minIdx = i1;
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

    return minDistance;
}

struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere {
    double rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

    TriMesh mesh;

    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) :
        rad(rad_), p(p_), e(e_), c(c_), refl(refl_), mesh{makeSphere(p_, rad_)} {}

    double intersectAnalytic(const Ray &r, Vec & x, Vec & n) const { // returns distance, 0 if nohit
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b*b - op.dot(op) + rad*rad;
        if (det < 0) return 0; else det = sqrt(det);
        const auto dist = (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
        if (dist > 0) {
            x = r.o + r.d*dist;
            n = (x - p).norm();
        }
        return dist;
    }

    double intersectMesh(const Ray &r, Vec & x, Vec & n) const {
        return ::intersect(r.o, r.d, mesh, x, n);
    }

    double intersect(const Ray &r, Vec & x, Vec & n) const {
        return intersectMesh(r, x, n);
    }
};
Sphere spheres[] = {//Scene: radius, position, emission, color, material
  Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
  Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
  Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
  Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt
  Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
  Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
  Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr
  Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas
  Sphere(600, Vec(50,681.6 - .27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite
};

inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }

inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

struct Hit
{
    static inline double inf() { return 1e20; };
    double t = inf();
    int id;
    Vec x;
    Vec n;

    explicit operator bool() const {
        return t < inf();
    }
};

inline Hit intersect(const Ray &r) {
    Hit hit;
    double count = sizeof(spheres) / sizeof(Sphere), d;
    Vec x, n;
    for (int i = int(count); i--;) {
        if ((d = spheres[i].intersect(r, x, n)) && d < hit.t) {
            hit.t = d;
            hit.id = i;
            hit.x = x;
            hit.n = n;
        }
    }
    return hit;
}

Vec radiance(const Ray &r, int depth, std::mt19937 & generator) {
    const auto hit = intersect(r);
    if (!hit) return Vec(); // if miss, return black
    const Sphere &obj = spheres[hit.id];        // the hit object
    Vec x = hit.x, n = hit.n, nl = n.dot(r.d) < 0 ? n : n*-1, f = obj.c;
    double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
    if (++depth > 5) if (rand_float(generator) < p && depth < 32) f = f*(1 / p); else return obj.e; //R.R.
    if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection
        double r1 = 2 * M_PI*rand_float(generator), r2 = rand_float(generator), r2s = sqrt(r2);
        Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w%u;
        Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2)).norm();
        return obj.e + f.mult(radiance(Ray(x, d), depth, generator));
    }
    else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, generator));
    Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl) > 0;                // Ray from outside going in?
    double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
    if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)    // Total internal reflection
        return obj.e + f.mult(radiance(reflRay, depth, generator));
    Vec tdir = (r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
    double a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : tdir.dot(n));
    double Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);
    return obj.e + f.mult(depth > 2 ? (rand_float(generator) < P ?   // Russian roulette
        radiance(reflRay, depth, generator)*RP : radiance(Ray(x, tdir), depth, generator)*TP) :
        radiance(reflRay, depth, generator)*Re + radiance(Ray(x, tdir), depth, generator)*Tr);
}
int main(int argc, char *argv[]) {
    const auto start = hr_clock::now();

    fprintf(stderr, "Starting rendering\n");

    const int w = 1024;
    const int h = 768;
    const int samps = argc == 2 ? atoi(argv[1]) / 4 : 1; // # samples
    const Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
    const Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135;
    Vec *c = new Vec[w*h];

    std::atomic_uint rowCounter = 0;

    const auto renderFuture = shn::asyncParallelLoop(h, shn::getSystemThreadCount(), [&](auto y, auto threadId) // Loop over image rows
    {
        std::mt19937 generator{ uint32_t(y*y*y) };
        for (unsigned short x = 0; x < w; x++) {   // Loop cols
            for (int sy = 0, i = (h - y - 1)*w + x; sy < 2; sy++) {     // 2x2 subpixel rows
                for (int sx = 0; sx < 2; sx++) {        // 2x2 subpixel cols
                    Vec r;
                    for (int s = 0; s < samps; s++) {
                        double r1 = 2 * rand_float(generator), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        double r2 = 2 * rand_float(generator), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        Vec d = cx*(((sx + .5 + dx) / 2 + x) / w - .5) +
                            cy*(((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                        r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, generator)*(1. / samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
                }
            }
        }
        ++rowCounter;
    });

    while (renderFuture.wait_for(std::chrono::milliseconds(33)) != std::future_status::ready)
    {
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100. * rowCounter / (h - 1));
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(hr_clock::now() - start);

    fprintf(stderr, "\nElapsed time: %d ms\n", elapsed.count());

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i = 0; i < w*h; i++)
        fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}
