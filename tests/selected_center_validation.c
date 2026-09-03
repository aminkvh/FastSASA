#include "fastsasa.h"
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
    fastsasa_context *context = NULL;
    fastsasa_parameters parameters = {1.4, 100, FASTSASA_ALGORITHM_SHRAKE_RUPLEY, FASTSASA_PRECISION_FP64};
    double radii[3] = {1.5, 1.5, 1.5};
    int residue_ids[3] = {0, 1, 2};
    double x[3] = {0.0, 2.6, 6.0};
    double y[3] = {0.0, 0.0, 0.0};
    double z[3] = {0.0, 0.0, 0.0};
    unsigned int selection_masks[3] = {1u, 0u, 1u};
    double full_selection[1] = {0.0};
    double selected_center[1] = {0.0};
    double selected_center_repeat[1] = {0.0};
    double total_sasa[1] = {0.0};
    fastsasa_topology topology;
    fastsasa_soa_frames frames;
    int status;

    status = fastsasa_check_device();
    if (status != FASTSASA_SUCCESS) {
        if (getenv("FASTSASA_REQUIRE_GPU_TESTS") != NULL) {
            fprintf(stderr, "CUDA device check failed: %s\n", fastsasa_status_string(status));
            return 1;
        }
        printf("selected_center_validation,status,skip,reason,no_cuda_device\n");
        return 0;
    }

    status = fastsasa_context_create(&context);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "context create failed: %s\n", fastsasa_status_string(status));
        return 1;
    }

    topology.radii = radii;
    topology.residue_ids = residue_ids;
    topology.n_atoms = 3;
    topology.n_residues = 3;
    frames.x = x;
    frames.y = y;
    frames.z = z;
    frames.n_frames = 1;

    unsetenv("FASTSASA_SELECTED_CENTER");
    status = fastsasa_context_calc_trajectory_soa_selection(context,
                                                          &topology,
                                                          &frames,
                                                          selection_masks,
                                                          1,
                                                          &parameters,
                                                          total_sasa,
                                                          full_selection);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "full-context selection failed: %s\n", fastsasa_status_string(status));
        fastsasa_context_free(context);
        return 1;
    }

    setenv("FASTSASA_SELECTED_CENTER", "1", 1);
    status = fastsasa_context_calc_trajectory_soa_selection(context,
                                                          &topology,
                                                          &frames,
                                                          selection_masks,
                                                          1,
                                                          &parameters,
                                                          NULL,
                                                          selected_center);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "selected-center calculation failed: %s\n", fastsasa_status_string(status));
        fastsasa_context_free(context);
        return 1;
    }

    status = fastsasa_context_calc_trajectory_soa_selection(context,
                                                          &topology,
                                                          &frames,
                                                          selection_masks,
                                                          1,
                                                          &parameters,
                                                          NULL,
                                                          selected_center_repeat);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "selected-center repeat failed: %s\n", fastsasa_status_string(status));
        fastsasa_context_free(context);
        return 1;
    }

    fastsasa_context_free(context);

    if (!near_value(full_selection[0], selected_center[0], 1.0e-7) ||
        !near_value(selected_center[0], selected_center_repeat[0], 1.0e-12) ||
        !(total_sasa[0] >= full_selection[0])) {
        fprintf(stderr,
                "selected-center validation failed: full_selection=%.12f selected_center=%.12f repeat=%.12f total=%.12f\n",
                full_selection[0],
                selected_center[0],
                selected_center_repeat[0],
                total_sasa[0]);
        return 1;
    }

    printf("selected_center_validation,full_selection,%.12f,selected_center,%.12f,total,%.12f,status,pass\n",
           full_selection[0],
           selected_center[0],
           total_sasa[0]);
    return 0;
}
