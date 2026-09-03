#include "fastsasa.h"
#include "fastsasa_cpu.h"

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

static int
run_gpu_case(const char *label,
             const char *float_mode,
             const double *x,
             const double *y,
             const double *z,
             const double *radii,
             const double *test_points,
             const double *cpu_sasa)
{
    fastsasa_context *context = NULL;
    fastsasa_sr_input input = {0};
    double gpu_sasa[2] = {0.0, 0.0};
    int status;

    setenv("FASTSASA_SR_FLOAT", float_mode, 1);

    status = fastsasa_context_create(&context);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "%s context create failed: %s\n", label, fastsasa_status_string(status));
        return 1;
    }

    input.n_atoms = 2;
    input.n_points = 1;
    input.x = x;
    input.y = y;
    input.z = z;
    input.radii = radii;
    input.test_points = test_points;
    input.reuse_test_points = 1;

    status = fastsasa_context_shrake_rupley_cell_list(context, &input, gpu_sasa);
    fastsasa_context_free(context);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "%s GPU calculation failed: %s\n", label, fastsasa_status_string(status));
        return 1;
    }

    if (!near_value(cpu_sasa[0], gpu_sasa[0], 1.0e-10) ||
        !near_value(cpu_sasa[1], gpu_sasa[1], 1.0e-10)) {
        fprintf(stderr,
                "%s boundary mismatch: cpu=(%.12f, %.12f) gpu=(%.12f, %.12f)\n",
                label,
                cpu_sasa[0],
                cpu_sasa[1],
                gpu_sasa[0],
                gpu_sasa[1]);
        return 1;
    }

    printf("cuda_boundary_validation,%s,cpu0,%.12f,gpu0,%.12f,cpu1,%.12f,gpu1,%.12f,status,pass\n",
           label,
           cpu_sasa[0],
           gpu_sasa[0],
           cpu_sasa[1],
           gpu_sasa[1]);
    return 0;
}

int
main(void)
{
    const double x[2] = {0.0, 2.0};
    const double y[2] = {0.0, 0.0};
    const double z[2] = {0.0, 0.0};
    const double radii[2] = {1.0, 1.0};
    const double test_points[3] = {1.0, 0.0, 0.0};
    double cpu_sasa[2] = {0.0, 0.0};
    int status;

    status = fastsasa_check_device();
    if (status != FASTSASA_SUCCESS) {
        if (getenv("FASTSASA_REQUIRE_GPU_TESTS") != NULL) {
            fprintf(stderr, "CUDA device check failed: %s\n", fastsasa_status_string(status));
            return 1;
        }
        printf("cuda_boundary_validation,status,skip,reason,no_cuda_device\n");
        return 0;
    }

    status = fastsasa_cpu_shrake_rupley(2, 1, x, y, z, radii, test_points, 1, cpu_sasa);
    if (status != FASTSASA_SUCCESS) {
        fprintf(stderr, "CPU calculation failed: %s\n", fastsasa_status_string(status));
        return 1;
    }

    if (run_gpu_case("double", "0", x, y, z, radii, test_points, cpu_sasa) != 0) {
        return 1;
    }
    if (run_gpu_case("float", "1", x, y, z, radii, test_points, cpu_sasa) != 0) {
        return 1;
    }

    return 0;
}
