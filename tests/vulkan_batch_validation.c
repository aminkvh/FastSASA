#include "fastsasa.h"
#include "fastsasa_trajectory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
near_value(double left, double right)
{
    return fabs(left - right) <= 1.0e-5;
}

static void
select_vulkan_backend(void)
{
#ifdef _WIN32
    _putenv_s("FASTSASA_BACKEND", "vulkan");
    _putenv_s("FASTSASA_SELECTED_CENTER", "1");
#else
    setenv("FASTSASA_BACKEND", "vulkan", 1);
    setenv("FASTSASA_SELECTED_CENTER", "1", 1);
#endif
}

static int
compare_single_frames(fastsasa_context *context,
                      const fastsasa_topology *topology,
                      const fastsasa_soa_frames *frames,
                      const fastsasa_parameters *parameters,
                      const unsigned int *selection_masks,
                      const double *batch_total,
                      const double *batch_selection)
{
    for (int frame = 0; frame < frames->n_frames; ++frame) {
        fastsasa_soa_frames single = {
            frames->x + (size_t)frame * (size_t)topology->n_atoms,
            frames->y + (size_t)frame * (size_t)topology->n_atoms,
            frames->z + (size_t)frame * (size_t)topology->n_atoms,
            1
        };
        double total = 0.0;
        double selection = 0.0;
        int status = fastsasa_context_calc_trajectory_soa(
            context, topology, &single, parameters, &total, NULL, NULL);

        if (status != FASTSASA_SUCCESS || !near_value(total, batch_total[frame])) {
            fprintf(stderr,
                    "Vulkan batch total mismatch at frame %d: %.12f vs %.12f\n",
                    frame, batch_total[frame], total);
            return 1;
        }
        if (selection_masks == NULL) continue;
        status = fastsasa_context_calc_trajectory_soa_selection(
            context, topology, &single, selection_masks, 1, parameters,
            NULL, &selection);
        if (status != FASTSASA_SUCCESS ||
            !near_value(selection, batch_selection[frame])) {
            fprintf(stderr,
                    "Vulkan batch selection mismatch at frame %d: %.12f vs %.12f\n",
                    frame, batch_selection[frame], selection);
            return 1;
        }
    }
    return 0;
}

int
main(void)
{
    const double radii[4] = {1.5, 1.6, 1.4, 1.7};
    const int residue_ids[4] = {0, 1, 2, 3};
    const unsigned int selection_masks[4] = {1u, 0u, 1u, 0u};
    const double x[12] = {
        0.0, 2.8, 0.0, 6.0,
        0.0, 3.1, 0.0, 4.8,
        0.0, 2.6, 0.0, 3.7
    };
    const double y[12] = {
        0.0, 0.0, 2.8, 0.0,
        0.0, 0.0, 3.0, 1.0,
        0.0, 0.0, 2.5, 1.8
    };
    const double z[12] = {0.0};
    const fastsasa_topology topology = {radii, residue_ids, 4, 4};
    const fastsasa_soa_frames frames = {x, y, z, 3};
    fastsasa_parameters parameters = {1.4, 100, FASTSASA_ALGORITHM_SHRAKE_RUPLEY, FASTSASA_PRECISION_FP64};
    double batch_total[3] = {0.0};
    double batch_selection[3] = {0.0};
    fastsasa_context *context = NULL;
    int status;

    select_vulkan_backend();
    status = fastsasa_context_create(&context);
    if (status != FASTSASA_SUCCESS || context == NULL ||
        strcmp(fastsasa_context_backend(context), "vulkan") != 0) {
        printf("fastsasa_vulkan_batch_validation,status,skip,detail,%s\n",
               fastsasa_last_error());
        fastsasa_context_free(context);
        return 77;
    }

    /* Some Vulkan devices (MoltenVK/Metal on Apple GPUs, notably) have no
     * shaderFloat64 support at all - a real, permanent hardware limit, not a
     * transient "no GPU" case. Rather than skip this test outright on such
     * devices, fall back to FP32 so the batch-vs-single-frame consistency
     * property this test checks still gets real coverage. */
    status = fastsasa_context_set_precision(context, FASTSASA_PRECISION_FP64);
    if (status == FASTSASA_DEVICE_UNSUPPORTED) {
        status = fastsasa_context_set_precision(context, FASTSASA_PRECISION_FP32);
        parameters.precision = FASTSASA_PRECISION_FP32;
        printf("fastsasa_vulkan_batch_validation,status,info,detail,shaderFloat64 unsupported; validating at FP32\n");
    }
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "Vulkan precision selection failed: %s\n", fastsasa_last_error());
        fastsasa_context_free(context);
        return 1;
    }

    status = fastsasa_context_calc_trajectory_soa(
        context, &topology, &frames, &parameters, batch_total, NULL, NULL);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "Vulkan SR batch failed: %s\n", fastsasa_last_error());
        fastsasa_context_free(context);
        return 1;
    }
    status = fastsasa_context_calc_trajectory_soa_selection(
        context, &topology, &frames, selection_masks, 1, &parameters,
        NULL, batch_selection);
    if (status != FASTSASA_SUCCESS ||
        compare_single_frames(context, &topology, &frames, &parameters,
                              selection_masks, batch_total, batch_selection) != 0) {
        fastsasa_context_free(context);
        return 1;
    }

    parameters.algorithm = FASTSASA_ALGORITHM_LEE_RICHARDS;
    parameters.n_points = 20;
    status = fastsasa_context_calc_trajectory_soa(
        context, &topology, &frames, &parameters, batch_total, NULL, NULL);
    if (status != FASTSASA_SUCCESS ||
        compare_single_frames(context, &topology, &frames, &parameters,
                              NULL, batch_total, NULL) != 0) {
        fprintf(stderr, "Vulkan LR batch validation failed: %s\n",
                fastsasa_last_error());
        fastsasa_context_free(context);
        return 1;
    }

    /* Batched Lee-Richards with selection centers, the combination that
     * previously had no native multi-frame path. */
    status = fastsasa_context_calc_trajectory_soa_selection(
        context, &topology, &frames, selection_masks, 1, &parameters,
        NULL, batch_selection);
    if (status != FASTSASA_SUCCESS ||
        compare_single_frames(context, &topology, &frames, &parameters,
                              selection_masks, batch_total, batch_selection) != 0) {
        fprintf(stderr, "Vulkan LR selection batch validation failed: %s\n",
                fastsasa_last_error());
        fastsasa_context_free(context);
        return 1;
    }

    fastsasa_context_free(context);
    printf("fastsasa_vulkan_batch_validation,status,pass\n");
    return 0;
}
