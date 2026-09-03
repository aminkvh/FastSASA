#include "fastsasa.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int
same_values(const double *left,
            const double *right,
            int count)
{
    for (int i = 0; i < count; ++i) {
        if (fabs(left[i] - right[i]) > 1.0e-12) return 0;
    }
    return 1;
}

int
main(void)
{
    fastsasa_context *first = NULL;
    fastsasa_context *second = NULL;
    fastsasa_sr_input input = {0};
    const double x[2] = {0.0, 1.5};
    const double y[2] = {0.0, 0.0};
    const double z[2] = {0.0, 0.0};
    const double radii[2] = {1.0, 1.0};
    const double positive_x[3] = {1.0, 0.0, 0.0};
    const double negative_x[3] = {-1.0, 0.0, 0.0};
    double first_initial[2] = {0.0, 0.0};
    double second_result[2] = {0.0, 0.0};
    double first_repeat[2] = {0.0, 0.0};
    int status;

    status = fastsasa_check_device();
    if (status != FASTSASA_SUCCESS) {
        if (getenv("FASTSASA_REQUIRE_GPU_TESTS") != NULL) {
            fprintf(stderr, "CUDA device check failed: %s\n", fastsasa_status_string(status));
            return 1;
        }
        printf("context_isolation_validation,status,skip,reason,no_cuda_device\n");
        return 0;
    }

    if (fastsasa_context_create(&first) != FASTSASA_SUCCESS ||
        fastsasa_context_create(&second) != FASTSASA_SUCCESS) {
        fprintf(stderr, "failed to create CUDA contexts\n");
        fastsasa_context_free(first);
        fastsasa_context_free(second);
        return 1;
    }

    input.n_atoms = 2;
    input.n_points = 1;
    input.x = x;
    input.y = y;
    input.z = z;
    input.radii = radii;
    input.reuse_test_points = 1;
    input.force_double_precision = 1;

    input.test_points = positive_x;
    status = fastsasa_context_shrake_rupley_cell_list(first, &input, first_initial);
    if (status == FASTSASA_SUCCESS) {
        input.test_points = negative_x;
        status = fastsasa_context_shrake_rupley_cell_list(second, &input, second_result);
    }
    if (status == FASTSASA_SUCCESS) {
        input.test_points = positive_x;
        status = fastsasa_context_shrake_rupley_cell_list(first, &input, first_repeat);
    }

    fastsasa_context_free(first);
    fastsasa_context_free(second);

    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "context calculation failed: %s\n", fastsasa_status_string(status));
        return 1;
    }
    if (!same_values(first_initial, first_repeat, 2) ||
        same_values(first_initial, second_result, 2)) {
        fprintf(stderr,
                "context isolation failed: first=(%.12f,%.12f) second=(%.12f,%.12f) repeat=(%.12f,%.12f)\n",
                first_initial[0], first_initial[1],
                second_result[0], second_result[1],
                first_repeat[0], first_repeat[1]);
        return 1;
    }

    printf("context_isolation_validation,status,pass\n");
    return 0;
}
