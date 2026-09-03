#include "fastsasa_trajectory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int
near_value(double a,
           double b,
           double tolerance)
{
    const double diff = fabs(a - b);
    const double scale = fmax(1.0, fmax(fabs(a), fabs(b)));

    return diff <= tolerance * scale;
}

int
main(void)
{
    const int n_atoms = 2048;
    fastsasa_context *context = NULL;
    fastsasa_parameters parameters = {1.4, 100, FASTSASA_ALGORITHM_SHRAKE_RUPLEY, FASTSASA_PRECISION_FP64};
    fastsasa_topology topology;
    fastsasa_soa_frames frames;
    double *radii = NULL;
    int *residue_ids = NULL;
    double *x = NULL;
    double *y = NULL;
    double *z = NULL;
    double baseline_total[1] = {0.0};
    double autotuned_total[1] = {0.0};
    int status;

    status = fastsasa_check_device();
    if (status != FASTSASA_SUCCESS) {
        if (getenv("FASTSASA_REQUIRE_GPU_TESTS") != NULL) {
            fprintf(stderr, "CUDA device check failed: %s\n", fastsasa_status_string(status));
            return 1;
        }
        printf("autotune_dispatch_validation,status,skip,reason,no_cuda_device\n");
        return 0;
    }

    radii = (double *)malloc(sizeof(double) * (size_t)n_atoms);
    residue_ids = (int *)malloc(sizeof(int) * (size_t)n_atoms);
    x = (double *)malloc(sizeof(double) * (size_t)n_atoms);
    y = (double *)malloc(sizeof(double) * (size_t)n_atoms);
    z = (double *)malloc(sizeof(double) * (size_t)n_atoms);
    if (radii == NULL || residue_ids == NULL || x == NULL || y == NULL || z == NULL) {
        free(radii);
        free(residue_ids);
        free(x);
        free(y);
        free(z);
        return 1;
    }

    for (int atom = 0; atom < n_atoms; ++atom) {
        radii[atom] = 1.5;
        residue_ids[atom] = atom;
        x[atom] = 3.2 * (double)(atom % 32);
        y[atom] = 3.2 * (double)((atom / 32) % 32);
        z[atom] = 3.2 * (double)(atom / 1024);
    }

    topology.radii = radii;
    topology.residue_ids = residue_ids;
    topology.n_atoms = n_atoms;
    topology.n_residues = n_atoms;
    frames.x = x;
    frames.y = y;
    frames.z = z;
    frames.n_frames = 1;

    status = fastsasa_context_create(&context);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "context create failed: %s\n", fastsasa_status_string(status));
        goto cleanup;
    }

    setenv("FASTSASA_AUTOTUNE", "0", 1);
    status = fastsasa_context_calc_trajectory_soa(context,
                                                &topology,
                                                &frames,
                                                &parameters,
                                                baseline_total,
                                                NULL,
                                                NULL);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "baseline calculation failed: %s\n", fastsasa_status_string(status));
        goto cleanup;
    }

    unsetenv("FASTSASA_AUTOTUNE");
    status = fastsasa_context_calc_trajectory_soa(context,
                                                &topology,
                                                &frames,
                                                &parameters,
                                                autotuned_total,
                                                NULL,
                                                NULL);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "autotuned calculation failed: %s\n", fastsasa_status_string(status));
        goto cleanup;
    }

    if (!near_value(baseline_total[0], autotuned_total[0], 1.0e-5)) {
        fprintf(stderr,
                "autotune dispatch mismatch: baseline=%.12f autotuned=%.12f\n",
                baseline_total[0],
                autotuned_total[0]);
        status = FASTSASA_INVALID_ARGUMENT;
        goto cleanup;
    }

    printf("autotune_dispatch_validation,baseline,%.12f,autotuned,%.12f,status,pass\n",
           baseline_total[0],
           autotuned_total[0]);
    status = FASTSASA_SUCCESS;

cleanup:
    unsetenv("FASTSASA_AUTOTUNE");
    fastsasa_context_free(context);
    free(radii);
    free(residue_ids);
    free(x);
    free(y);
    free(z);
    return status == FASTSASA_SUCCESS ? 0 : 1;
}
