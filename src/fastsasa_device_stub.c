#include "fastsasa_device.h"

#include <stdlib.h>

struct fastsasa_device_context {
    int unused;
};

static const char *fastsasa_no_device_error =
    "FastSASA was built without the CUDA backend";

int
fastsasa_device_context_create(fastsasa_device_context **context)
{
    if (context != NULL) *context = NULL;
    return FASTSASA_NO_DEVICE;
}

void
fastsasa_device_context_free(fastsasa_device_context *context)
{
    (void)context;
}

int
fastsasa_device_context_synchronize(fastsasa_device_context *context)
{
    (void)context;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_enable_profile(fastsasa_device_context *context,
                                     int enabled)
{
    (void)context;
    (void)enabled;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_last_cell_profile(fastsasa_device_context *context,
                                        fastsasa_device_cell_profile *profile)
{
    (void)context;
    (void)profile;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_host_alloc(void **ptr,
                         size_t bytes)
{
    if (ptr == NULL || bytes == 0) return FASTSASA_INVALID_ARGUMENT;
    *ptr = NULL;
    *ptr = malloc(bytes);
    return *ptr != NULL ? FASTSASA_SUCCESS : FASTSASA_MEMORY_ERROR;
}

void
fastsasa_device_host_free(void *ptr)
{
    free(ptr);
}

int
fastsasa_device_shrake_rupley_csr(const fastsasa_device_sr_input *input,
                                double *sasa)
{
    (void)input;
    (void)sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_csr(fastsasa_device_context *context,
                                        const fastsasa_device_sr_input *input,
                                        double *sasa)
{
    (void)context;
    (void)input;
    (void)sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_cell_list(fastsasa_device_context *context,
                                              const fastsasa_device_sr_input *input,
                                              double *sasa)
{
    (void)context;
    (void)input;
    (void)sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_cell_list_async(fastsasa_device_context *context,
                                                    const fastsasa_device_sr_input *input,
                                                    double *sasa)
{
    (void)context;
    (void)input;
    (void)sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_cell_list_total(fastsasa_device_context *context,
                                                    const fastsasa_device_sr_input *input,
                                                    double *total_sasa)
{
    (void)context;
    (void)input;
    (void)total_sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_cell_list_total_async(fastsasa_device_context *context,
                                                          const fastsasa_device_sr_input *input,
                                                          double *total_sasa)
{
    (void)context;
    (void)input;
    (void)total_sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_lee_richards(fastsasa_device_context *context,
                                   const fastsasa_device_sr_input *input,
                                   double *sasa)
{
    (void)context;
    (void)input;
    (void)sasa;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_context_shrake_rupley_csr_batch(fastsasa_device_context *context,
                                              const fastsasa_device_sr_input *inputs,
                                              double *const *sasa_outputs,
                                              int n_inputs)
{
    (void)context;
    (void)inputs;
    (void)sasa_outputs;
    (void)n_inputs;
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_check_device(void)
{
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_device_constant_test_point_limit(void)
{
    return 0;
}

int
fastsasa_device_recommended_trajectory_batch_size(int n_atoms,
                                                int n_frames,
                                                int n_points,
                                                int selection_only)
{
    (void)n_atoms;
    (void)n_frames;
    (void)n_points;
    (void)selection_only;
    return 8;
}

int
fastsasa_device_recommended_parallel_frames(int n_atoms,
                                          int n_points,
                                          int batch_size,
                                          int selection_only)
{
    (void)n_atoms;
    (void)n_points;
    (void)batch_size;
    (void)selection_only;
    return 1;
}

const char *
fastsasa_device_status_string(int status)
{
    switch (status) {
    case FASTSASA_SUCCESS:
        return "success";
    case FASTSASA_INVALID_ARGUMENT:
        return "invalid argument";
    case FASTSASA_CUDA_ERROR:
        return "CUDA error";
    case FASTSASA_MEMORY_ERROR:
        return "memory error";
    case FASTSASA_NO_DEVICE:
        return "no CUDA device";
    case FASTSASA_DEVICE_UNSUPPORTED:
        return "CUDA device unsupported for this launch";
    case FASTSASA_CELL_GRID_TOO_LARGE:
        return "coordinate bounds require an unsupported dense cell grid; wrap or filter input coordinates";
    default:
        return "unknown FastSASA status";
    }
}

const char *
fastsasa_device_last_error(void)
{
    return fastsasa_no_device_error;
}
