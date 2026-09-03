#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "freesasa.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr,
                "usage: %s coords.f64 radii.f64 selection.u32 n_atoms n_frames n_threads panel(complete|selected) n_repeats\n",
                argv[0]);
        return 2;
    }
    const char *coords_path = argv[1];
    const char *radii_path = argv[2];
    const char *selection_path = argv[3];
    int n_atoms = atoi(argv[4]);
    int n_frames = atoi(argv[5]);
    int n_threads = atoi(argv[6]);
    const char *panel = argv[7];
    int n_repeats = atoi(argv[8 <= argc - 1 ? 8 : argc - 1]);
    int selected_panel = strcmp(panel, "selected") == 0;

    double *radii = malloc(sizeof(double) * (size_t)n_atoms);
    FILE *fr = fopen(radii_path, "rb");
    if (!fr || fread(radii, sizeof(double), (size_t)n_atoms, fr) != (size_t)n_atoms) {
        fprintf(stderr, "failed to read radii\n");
        return 1;
    }
    fclose(fr);

    unsigned int *selection_mask = malloc(sizeof(unsigned int) * (size_t)n_atoms);
    FILE *fs = fopen(selection_path, "rb");
    if (!fs || fread(selection_mask, sizeof(unsigned int), (size_t)n_atoms, fs) != (size_t)n_atoms) {
        fprintf(stderr, "failed to read selection mask\n");
        return 1;
    }
    fclose(fs);

    size_t frame_doubles = (size_t)n_atoms * 3;
    double *coords_all = malloc(sizeof(double) * frame_doubles * (size_t)n_frames);
    FILE *fc = fopen(coords_path, "rb");
    if (!fc || fread(coords_all, sizeof(double), frame_doubles * (size_t)n_frames, fc) != frame_doubles * (size_t)n_frames) {
        fprintf(stderr, "failed to read coords\n");
        return 1;
    }
    fclose(fc);

    freesasa_parameters params = freesasa_default_parameters;
    params.alg = FREESASA_SHRAKE_RUPLEY;
    params.probe_radius = 1.4;
    params.shrake_rupley_n_points = 100;
    params.n_threads = n_threads;

    freesasa_set_verbosity(FREESASA_V_SILENT);

    /* One warmup pass (not timed), then n_repeats independent timed passes,
       each over the full trajectory. Only the compute loop (freesasa_calc_coord
       + result summation) is inside the timer; file I/O above is excluded. */
    for (int pass = -1; pass < n_repeats; ++pass) {
        double t0 = now_seconds();
        double total_sum = 0.0;
        for (int f = 0; f < n_frames; ++f) {
            const double *xyz = coords_all + (size_t)f * frame_doubles;
            freesasa_result *result = freesasa_calc_coord(xyz, radii, n_atoms, &params);
            if (!result) {
                fprintf(stderr, "frame %d: freesasa_calc_coord returned NULL\n", f);
                return 1;
            }
            if (selected_panel) {
                double sel_sum = 0.0;
                for (int a = 0; a < n_atoms; ++a) {
                    if (selection_mask[a]) sel_sum += result->sasa[a];
                }
                total_sum += sel_sum;
            } else {
                total_sum += result->total;
            }
            freesasa_result_free(result);
        }
        double t1 = now_seconds();
        double elapsed = t1 - t0;
        if (pass == -1) {
            fprintf(stderr, "warmup: elapsed=%.6f total_sum=%.6f\n", elapsed, total_sum);
        } else {
            printf("repeat=%d n_threads=%d panel=%s frames=%d elapsed=%.6f frames_per_second=%.6f total_sum=%.6f\n",
                   pass, n_threads, panel, n_frames, elapsed, n_frames / elapsed, total_sum);
        }
    }

    free(radii);
    free(selection_mask);
    free(coords_all);
    return 0;
}
