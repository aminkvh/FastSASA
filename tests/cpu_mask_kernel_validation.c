/*
 * Bit-identity of the mask-accelerated CPU Shrake-Rupley kernel against the
 * reference kernel, on real structures and on the geometric edge cases the
 * kernel treats specially: coincident centres (equal and unequal radii),
 * large coordinate offsets, rotations, touching pairs, single atoms, dense
 * random packings, and every mask width (1 to 4 words, i.e. up to 255
 * points), at both FP64 and FP32 (each mask variant against its own
 * reference kernel). Also checks the automatic policy and the
 * unsupported-input path.
 */
#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_topology.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static double *
fibonacci_points(int n_points)
{
    const double pi = 3.14159265358979323846;
    const double dlong = pi * (3.0 - sqrt(5.0));
    const double dz = 2.0 / (double)n_points;
    double longitude = 0.0;
    double z = 1.0 - dz / 2.0;
    double *points = (double *)malloc(sizeof(double) * 3u * (size_t)n_points);

    for (int i = 0; i < n_points; ++i) {
        const double r = sqrt(1.0 - z * z);
        points[3 * i] = cos(longitude) * r;
        points[3 * i + 1] = sin(longitude) * r;
        points[3 * i + 2] = z;
        z -= dz;
        longitude += dlong;
    }
    return points;
}

/* Run both kernels; report the first per-atom difference. */
static void
compare_precision(const char *name, int n_atoms, int n_points,
                  const double *x, const double *y, const double *z, const double *expanded,
                  int n_threads, int precision)
{
    double *points = fibonacci_points(n_points);
    double *ref = (double *)calloc((size_t)n_atoms, sizeof(double));
    double *msk = (double *)calloc((size_t)n_atoms, sizeof(double));
    const int sr = fastsasa_cpu_shrake_rupley_precision(n_atoms, n_points, x, y, z, expanded, points, n_threads, precision, ref);
    const int sm = precision == FASTSASA_PRECISION_FP32
        ? fastsasa_cpu_shrake_rupley_mask_fp32(n_atoms, n_points, x, y, z, expanded, points, n_threads, msk)
        : fastsasa_cpu_shrake_rupley_mask(n_atoms, n_points, x, y, z, expanded, points, n_threads, msk);
    int bad = -1;

    /* the reference entry point may itself have routed to the mask kernel
     * (policy); force the reference explicitly for this comparison */
    if (sr == FASTSASA_SUCCESS) {
        double *ref2 = (double *)calloc((size_t)n_atoms, sizeof(double));
        /* reference kernel proper: the policy is bypassed through the
         * environment override, restored afterwards */
#if defined(_WIN32)
        _putenv_s("FASTSASA_CPU_KERNEL", "reference");
#else
        setenv("FASTSASA_CPU_KERNEL", "reference", 1);
#endif
        fastsasa_cpu_shrake_rupley_precision(n_atoms, n_points, x, y, z, expanded, points, n_threads, precision, ref2);
#if defined(_WIN32)
        _putenv_s("FASTSASA_CPU_KERNEL", "");
#else
        unsetenv("FASTSASA_CPU_KERNEL");
#endif
        memcpy(ref, ref2, sizeof(double) * (size_t)n_atoms);
        free(ref2);
    }
    if (sr != sm) {
        printf("FAIL %-40s status reference=%d mask=%d\n", name, sr, sm);
        ++failures;
    } else if (sr == FASTSASA_SUCCESS) {
        for (int a = 0; a < n_atoms; ++a) {
            if (memcmp(&ref[a], &msk[a], sizeof(double)) != 0) { bad = a; break; }
        }
        if (bad >= 0) {
            printf("FAIL %-40s atom %d: reference %.17g mask %.17g\n", name, bad, ref[bad], msk[bad]);
            ++failures;
        } else {
            double total = 0.0;
            for (int a = 0; a < n_atoms; ++a) total += ref[a];
            printf("ok   %-40s %6d atoms %3d points %s total %.6f\n", name, n_atoms, n_points,
                   precision == FASTSASA_PRECISION_FP32 ? "fp32" : "fp64", total);
        }
    } else {
        printf("ok   %-40s both rejected (status %d)\n", name, sr);
    }
    free(points);
    free(ref);
    free(msk);
}

static void
compare(const char *name, int n_atoms, int n_points,
        const double *x, const double *y, const double *z, const double *expanded,
        int n_threads)
{
    compare_precision(name, n_atoms, n_points, x, y, z, expanded, n_threads, FASTSASA_PRECISION_FP64);
    compare_precision(name, n_atoms, n_points, x, y, z, expanded, n_threads, FASTSASA_PRECISION_FP32);
}

static double
default_radius(const char *residue, const char *atom, const char *element, void *userdata)
{
    (void)residue; (void)atom; (void)userdata;
    if (element != NULL) {
        if (strcmp(element, "N") == 0) return 1.55;
        if (strcmp(element, "O") == 0) return 1.52;
        if (strcmp(element, "S") == 0) return 1.80;
        if (strcmp(element, "P") == 0) return 1.80;
    }
    return 1.70;
}

static void
structure_cases(const char *path)
{
    fastsasa_owned_topology topology;
    const int n_points_list[] = {1, 2, 7, 63, 64, 65, 100, 128, 129, 200, 255};
    char name[256];

    /* the reader returns 1 on success */
    if (fastsasa_topology_read_mmcif(path, FASTSASA_TOPOLOGY_INCLUDE_HETATM, default_radius, NULL, &topology) != 1 ||
        topology.n_atoms <= 0) {
        printf("FAIL cannot read %s\n", path);
        ++failures;
        return;
    }
    const int n = topology.n_atoms;
    double *expanded = (double *)malloc(sizeof(double) * (size_t)n);
    double *tx = (double *)malloc(sizeof(double) * (size_t)n);
    double *ty = (double *)malloc(sizeof(double) * (size_t)n);
    double *tz = (double *)malloc(sizeof(double) * (size_t)n);
    for (int a = 0; a < n; ++a) expanded[a] = topology.radii[a] + 1.4;

    for (size_t i = 0; i < sizeof(n_points_list) / sizeof(n_points_list[0]); ++i) {
        snprintf(name, sizeof(name), "%s @%d", path, n_points_list[i]);
        compare(name, n, n_points_list[i], topology.x, topology.y, topology.z, expanded, 1);
    }
    /* threads */
    snprintf(name, sizeof(name), "%s @128 x4 threads", path);
    compare(name, n, 128, topology.x, topology.y, topology.z, expanded, 4);
    /* large offsets */
    const double offsets[] = {1.0e3, 1.0e6, 1.0e9};
    for (size_t i = 0; i < 3; ++i) {
        for (int a = 0; a < n; ++a) { tx[a] = topology.x[a] + offsets[i]; ty[a] = topology.y[a] - offsets[i]; tz[a] = topology.z[a] + offsets[i]; }
        snprintf(name, sizeof(name), "%s offset %.0e", path, offsets[i]);
        compare(name, n, 128, tx, ty, tz, expanded, 1);
    }
    /* rotation about an arbitrary axis, and a mirror */
    {
        const double c = cos(0.7), s = sin(0.7), t = 1.0 - c;
        const double ux = 0.6, uy = 0.48, uz = 0.64;   /* unit axis */
        const double R[9] = {
            t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy,
            t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux,
            t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c};
        for (int a = 0; a < n; ++a) {
            tx[a] = R[0] * topology.x[a] + R[1] * topology.y[a] + R[2] * topology.z[a];
            ty[a] = R[3] * topology.x[a] + R[4] * topology.y[a] + R[5] * topology.z[a];
            tz[a] = R[6] * topology.x[a] + R[7] * topology.y[a] + R[8] * topology.z[a];
        }
        snprintf(name, sizeof(name), "%s rotated", path);
        compare(name, n, 128, tx, ty, tz, expanded, 1);
        for (int a = 0; a < n; ++a) { tx[a] = -topology.x[a]; ty[a] = topology.y[a]; tz[a] = topology.z[a]; }
        snprintf(name, sizeof(name), "%s mirrored", path);
        compare(name, n, 128, tx, ty, tz, expanded, 1);
    }
    /* varied radii */
    for (int a = 0; a < n; ++a) expanded[a] = 1.0 + 3.0 * ((a * 7919) % 1000) / 1000.0;
    snprintf(name, sizeof(name), "%s radii 1..4", path);
    compare(name, n, 128, topology.x, topology.y, topology.z, expanded, 1);

    free(expanded); free(tx); free(ty); free(tz);
    fastsasa_topology_free(&topology);
}

static void
synthetic_cases(void)
{
    /* coincident centres: equal radii is decided by rounding in the reference */
    {
        const double x[2] = {1.0, 1.0}, y[2] = {2.0, 2.0}, z[2] = {3.0, 3.0};
        const double eq[2] = {3.1, 3.1}, gt[2] = {3.0, 3.1}, lt[2] = {3.1, 2.1}, near[2] = {3.1, 3.1 - 1e-12};
        compare("coincident, equal radii", 2, 128, x, y, z, eq, 1);
        compare("coincident, Rj > Ri", 2, 128, x, y, z, gt, 1);
        compare("coincident, Rj < Ri", 2, 128, x, y, z, lt, 1);
        compare("coincident, Rj barely < Ri", 2, 128, x, y, z, near, 1);
        compare("coincident, equal radii, 64 pts", 2, 64, x, y, z, eq, 1);
        compare("coincident, equal radii, 255 pts", 2, 255, x, y, z, eq, 1);
    }
    {
        const double x[4] = {0, 0, 0, 10}, y[4] = {0, 0, 0, 0}, z[4] = {0, 0, 0, 0};
        const double r[4] = {3.0, 3.1, 2.9, 3.0};
        compare("three coincident + one far", 4, 128, x, y, z, r, 1);
    }
    {
        const double x[1] = {0}, y[1] = {0}, z[1] = {0}, r[1] = {3.0};
        compare("single atom", 1, 128, x, y, z, r, 1);
    }
    {
        const double x[2] = {0, 6.0}, y[2] = {0, 0}, z[2] = {0, 0}, r[2] = {3.0, 3.0};
        const double xb[2] = {0, 6.0 - 1e-9};
        const double xf[2] = {0, 100.0};
        compare("touching pair d = Ri + Rj", 2, 128, x, y, z, r, 1);
        compare("pair d just below Ri + Rj", 2, 128, xb, y, z, r, 1);
        compare("two far atoms", 2, 128, xf, y, z, r, 1);
    }
    {
        /* dense random packing: many neighbours, many fully buried atoms */
        enum { N = 3000 };
        static double x[N], y[N], z[N], r[N];
        unsigned long long seed = 12345u;
        for (int a = 0; a < N; ++a) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; x[a] = (double)(seed >> 11) / 9007199254740992.0 * 40.0;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; y[a] = (double)(seed >> 11) / 9007199254740992.0 * 40.0;
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; z[a] = (double)(seed >> 11) / 9007199254740992.0 * 40.0;
            r[a] = 3.2;
        }
        compare("dense random packing", N, 128, x, y, z, r, 1);
        compare("dense random packing x8 threads", N, 100, x, y, z, r, 8);
    }
    {
        /* unsupported point count: mask kernel declines, reference works */
        const double x[2] = {0, 3.0}, y[2] = {0, 0}, z[2] = {0, 0}, r[2] = {3.0, 3.0};
        double *points = fibonacci_points(256);
        double out[2];
        const int st = fastsasa_cpu_shrake_rupley_mask(2, 256, x, y, z, r, points, 1, out);
        if (st != -100) { printf("FAIL 256 points: mask kernel returned %d, expected -100\n", st); ++failures; }
        else printf("ok   256 points declined by the mask kernel (-100)\n");
        if (fastsasa_cpu_mask_policy(2, 256, points) != 0) { printf("FAIL policy accepted 256 points\n"); ++failures; }
        if (fastsasa_cpu_shrake_rupley(2, 256, x, y, z, r, points, 1, out) != FASTSASA_SUCCESS) {
            printf("FAIL reference path failed at 256 points\n"); ++failures;
        }
        free(points);
    }
}

int
main(int argc, char **argv)
{
    /* Make the automatic policy deterministic for this run: the mask kernel
     * is called explicitly, the reference through the override. */
    synthetic_cases();
    for (int i = 1; i < argc; ++i) structure_cases(argv[i]);
    if (failures) {
        printf("fastsasa_cpu_mask_kernel_validation,status,fail,%d\n", failures);
        return 1;
    }
    printf("fastsasa_cpu_mask_kernel_validation,status,pass\n");
    return 0;
}
