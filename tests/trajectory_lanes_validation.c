/*
 * Selection trajectories on GPU frame lanes (batched mask launches, float
 * coordinate upload, several host threads) against one frame per call and
 * against the CPU backend: FP64 selection sums must be bit-identical.
 * Also covers repeated calls that reuse the same host buffers with changed
 * contents (inputs kept on the device are matched by host pointer), point
 * counts above 256 (multi-word masks), and frames that are not exactly
 * representable as floats.
 *
 * The device context comes from FASTSASA_BACKEND (auto, cuda, vulkan).
 */
#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_trajectory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N_ATOMS = 3000, N_FRAMES = 23, N_SELECTIONS = 2 };

static unsigned long long rng_state = 88172645463325252ull;

static double
uniform(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    double radii[N_ATOMS];
    int residue_ids[N_ATOMS];
    unsigned int masks[N_ATOMS];
    double *x, *y, *z;
} system_data;

/* Protein-like packing: ~0.05 atoms per cubic angstrom, radii 1.5-2.0. */
static void
make_system(system_data *s, int float_exact)
{
    const double box = 38.0;
    double base[3 * N_ATOMS];

    for (int a = 0; a < N_ATOMS; ++a) {
        s->radii[a] = 1.5 + 0.5 * uniform();
        s->residue_ids[a] = a / 8;
        for (int k = 0; k < 3; ++k) base[3 * a + k] = box * uniform();
    }
    for (int f = 0; f < N_FRAMES; ++f) {
        for (int a = 0; a < N_ATOMS; ++a) {
            const size_t i = (size_t)f * N_ATOMS + (size_t)a;
            double p[3];
            for (int k = 0; k < 3; ++k) {
                p[k] = base[3 * a + k] + 0.6 * (uniform() - 0.5);
                if (float_exact) p[k] = (double)(float)p[k];
            }
            s->x[i] = p[0];
            s->y[i] = p[1];
            s->z[i] = p[2];
        }
    }
}

/* Selection 0: one atom near the centre; selection 1: atoms in a sphere. */
static void
make_masks(system_data *s, int n_selected)
{
    int best = 0;
    double best_d = 1e300;

    for (int a = 0; a < N_ATOMS; ++a) {
        const double dx = s->x[a] - 19.0, dy = s->y[a] - 19.0, dz = s->z[a] - 19.0;
        const double d = dx * dx + dy * dy + dz * dz;
        s->masks[a] = 0u;
        if (d < best_d) { best_d = d; best = a; }
    }
    s->masks[best] |= 1u;
    for (int a = 0, taken = 0; a < N_ATOMS && taken < n_selected; ++a) {
        const double dx = s->x[a] - 12.0, dy = s->y[a] - 25.0, dz = s->z[a] - 19.0;
        if (dx * dx + dy * dy + dz * dz < 100.0) { s->masks[a] |= 2u; ++taken; }
    }
}

static int
compare(const char *what, const double *got, const double *want, int n)
{
    for (int i = 0; i < n; ++i) {
        if (memcmp(&got[i], &want[i], sizeof(double)) != 0) {
            fprintf(stderr, "%s: value %d (frame %d) %.17g != %.17g\n", what, i, i / N_SELECTIONS, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

/* One frame per call on the same context (no frame lanes). */
static int
per_frame(fastsasa_context *context, const fastsasa_topology *topology, const system_data *s,
          const fastsasa_parameters *parameters, double *out)
{
    for (int f = 0; f < N_FRAMES; ++f) {
        fastsasa_soa_frames one = {s->x + (size_t)f * N_ATOMS, s->y + (size_t)f * N_ATOMS,
                                   s->z + (size_t)f * N_ATOMS, 1};
        const int status = fastsasa_context_calc_trajectory_soa_selection(
            context, topology, &one, s->masks, N_SELECTIONS, parameters, NULL, out + (size_t)f * N_SELECTIONS);
        if (status != FASTSASA_SUCCESS) return status;
    }
    return FASTSASA_SUCCESS;
}

static int
run_case(fastsasa_context *context, system_data *s, int n_points, int float_exact)
{
    fastsasa_topology topology = {s->radii, s->residue_ids, N_ATOMS, N_ATOMS / 8};
    fastsasa_soa_frames frames = {s->x, s->y, s->z, N_FRAMES};
    fastsasa_parameters parameters = {1.4, n_points, FASTSASA_ALGORITHM_SHRAKE_RUPLEY, FASTSASA_PRECISION_FP64};
    double lanes[N_FRAMES * N_SELECTIONS], single[N_FRAMES * N_SELECTIONS], cpu[N_FRAMES * N_SELECTIONS];
    char label[160];
    int status;

    rng_state = 88172645463325252ull + (unsigned long long)n_points;
    make_system(s, float_exact);
    make_masks(s, 150);

    /* two rounds on the same host buffers: the second changes the probe radius,
     * every radius and the frames, which must not reuse anything from the first */
    for (int round = 0; round < 2; ++round) {
        if (round == 1) {
            parameters.probe_radius = 1.2;
            for (int a = 0; a < N_ATOMS; ++a) s->radii[a] += 0.05;
            for (size_t i = 0; i < (size_t)N_FRAMES * N_ATOMS; ++i) {
                s->x[i] += 0.25;
                if (float_exact) s->x[i] = (double)(float)s->x[i];
            }
        }
        snprintf(label, sizeof(label), "points %d, %s frames, round %d", n_points,
                 float_exact ? "float-exact" : "double", round);
        status = fastsasa_context_calc_trajectory_soa_selection(context, &topology, &frames, s->masks, N_SELECTIONS,
                                                              &parameters, NULL, lanes);
        if (status != FASTSASA_SUCCESS) {
            fprintf(stderr, "%s: lanes failed: %s %s\n", label, fastsasa_status_string(status), fastsasa_last_error());
            return 1;
        }
        status = per_frame(context, &topology, s, &parameters, single);
        if (status != FASTSASA_SUCCESS) {
            fprintf(stderr, "%s: per-frame failed: %s\n", label, fastsasa_status_string(status));
            return 1;
        }
        status = fastsasa_cpu_calc_trajectory_soa_selection(&topology, &frames, s->masks, N_SELECTIONS, &parameters,
                                                          0, NULL, cpu);
        if (status != FASTSASA_SUCCESS) {
            fprintf(stderr, "%s: CPU failed: %s\n", label, fastsasa_status_string(status));
            return 1;
        }
        char what[200];
        snprintf(what, sizeof(what), "%s: lanes vs per-frame", label);
        if (compare(what, lanes, single, N_FRAMES * N_SELECTIONS)) return 1;
        snprintf(what, sizeof(what), "%s: lanes vs CPU", label);
        if (compare(what, lanes, cpu, N_FRAMES * N_SELECTIONS)) return 1;
        if (!(lanes[1] > 0.0) || !(lanes[N_SELECTIONS + 1] > 0.0)) {
            fprintf(stderr, "%s: implausible selection SASA %.6f\n", label, lanes[1]);
            return 1;
        }
    }
    return 0;
}

int
main(void)
{
    static const int point_counts[] = {100, 256, 300, 1024};
    fastsasa_context *context = NULL;
    system_data s;
    int status = fastsasa_context_create(&context);

    if (status != FASTSASA_SUCCESS) {
        if (getenv("FASTSASA_REQUIRE_GPU_TESTS") != NULL) {
            fprintf(stderr, "context create failed: %s\n", fastsasa_status_string(status));
            return 1;
        }
        printf("trajectory_lanes_validation,status,skip,reason,no_device\n");
        return 0;
    }
    s.x = (double *)malloc(sizeof(double) * (size_t)N_FRAMES * N_ATOMS);
    s.y = (double *)malloc(sizeof(double) * (size_t)N_FRAMES * N_ATOMS);
    s.z = (double *)malloc(sizeof(double) * (size_t)N_FRAMES * N_ATOMS);
    if (s.x == NULL || s.y == NULL || s.z == NULL) return 1;

    int failed = 0;
    for (size_t p = 0; p < sizeof(point_counts) / sizeof(point_counts[0]) && !failed; ++p) {
        for (int float_exact = 1; float_exact >= 0 && !failed; --float_exact) {
            failed = run_case(context, &s, point_counts[p], float_exact);
        }
    }
    printf("trajectory_lanes_validation,backend,%s,status,%s\n", fastsasa_context_backend(context),
           failed ? "fail" : "pass");
    fastsasa_context_free(context);
    free(s.x);
    free(s.y);
    free(s.z);
    return failed;
}
