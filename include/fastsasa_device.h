#ifndef FASTSASA_DEVICE_H
#define FASTSASA_DEVICE_H

#include "fastsasa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef fastsasa_sr_input fastsasa_device_sr_input;
typedef fastsasa_cell_profile fastsasa_device_cell_profile;
typedef struct fastsasa_device_context fastsasa_device_context;

int fastsasa_device_context_create(fastsasa_device_context **context);
void fastsasa_device_context_free(fastsasa_device_context *context);
int fastsasa_device_context_synchronize(fastsasa_device_context *context);
int fastsasa_device_context_enable_profile(fastsasa_device_context *context,
                                         int enabled);
int fastsasa_device_context_last_cell_profile(fastsasa_device_context *context,
                                            fastsasa_device_cell_profile *profile);

int fastsasa_device_host_alloc(void **ptr,
                             size_t bytes);
void fastsasa_device_host_free(void *ptr);

int fastsasa_device_shrake_rupley_csr(const fastsasa_device_sr_input *input,
                                    double *sasa);
int fastsasa_device_context_shrake_rupley_csr(fastsasa_device_context *context,
                                            const fastsasa_device_sr_input *input,
                                            double *sasa);
int fastsasa_device_context_shrake_rupley_cell_list(fastsasa_device_context *context,
                                                  const fastsasa_device_sr_input *input,
                                                  double *sasa);
int fastsasa_device_context_shrake_rupley_cell_list_async(fastsasa_device_context *context,
                                                        const fastsasa_device_sr_input *input,
                                                        double *sasa);
int fastsasa_device_context_shrake_rupley_cell_list_total(fastsasa_device_context *context,
                                                        const fastsasa_device_sr_input *input,
                                                        double *total_sasa);
int fastsasa_device_context_shrake_rupley_cell_list_total_async(fastsasa_device_context *context,
                                                              const fastsasa_device_sr_input *input,
                                                              double *total_sasa);
int fastsasa_device_context_lee_richards(fastsasa_device_context *context,
                                       const fastsasa_device_sr_input *input,
                                       double *sasa);
int fastsasa_device_context_shrake_rupley_csr_batch(fastsasa_device_context *context,
                                                  const fastsasa_device_sr_input *inputs,
                                                  double *const *sasa_outputs,
                                                  int n_inputs);

int fastsasa_device_check_device(void);
int fastsasa_device_constant_test_point_limit(void);
int fastsasa_device_recommended_trajectory_batch_size(int n_atoms,
                                                    int n_frames,
                                                    int n_points,
                                                    int selection_only);
int fastsasa_device_recommended_parallel_frames(int n_atoms,
                                              int n_points,
                                              int batch_size,
                                              int selection_only);
const char *fastsasa_device_status_string(int status);
const char *fastsasa_device_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
