#ifndef FASTSASA_H
#define FASTSASA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fastsasa_status {
    FASTSASA_SUCCESS = 0,
    FASTSASA_INVALID_ARGUMENT = -1,
    FASTSASA_CUDA_ERROR = -2,
    FASTSASA_MEMORY_ERROR = -3,
    FASTSASA_NO_DEVICE = -4,
    FASTSASA_DEVICE_UNSUPPORTED = -5,
    FASTSASA_CELL_GRID_TOO_LARGE = -6,
    FASTSASA_VULKAN_ERROR = -7
};

enum fastsasa_precision {
    FASTSASA_PRECISION_FP64 = 0,
    FASTSASA_PRECISION_FP32 = 1
};

#define FASTSASA_ABI_VERSION 1u

typedef struct fastsasa_sr_input {
    int n_atoms;
    int n_points;
    const double *xyz;
    const double *x;
    const double *y;
    const double *z;
    const double *radii;
    const double *test_points;
    const int *neighbor_offsets;
    const int *neighbor_indices;
    int n_neighbor_indices;
    int reuse_test_points;
    const int *residue_ids;
    int n_residues;
    double *residue_sasa;
    const unsigned int *selection_masks;
    int n_selections;
    double *selection_sasa;
    /*
     * Advanced/internal SR optimization: when non-zero, only atoms matching
     * these selection-mask bits are used as SASA centers. All atoms still
     * participate as occluders. Use only for selection-only reductions.
     */
    unsigned int active_center_mask;
    /*
     * Optional compact center list for selected-center SR. When supplied, SR
     * kernels launch only these atom indices as centers while all input atoms
     * remain occluders.
     */
    const int *active_center_indices;
    int n_active_centers;
    /*
     * Internal/advanced precision control. Positive forces FP64, negative
     * forces FP32, and zero uses the backend default/environment policy.
     */
    int force_double_precision;
} fastsasa_sr_input;

typedef struct fastsasa_cell_profile {
    double h2d_ms;
    double radius_ms;
    double cell_count_ms;
    double scan_ms;
    double cell_fill_ms;
    double sr_ms;
    double residue_ms;
    double d2h_ms;
    double total_ms;
} fastsasa_cell_profile;

typedef struct fastsasa_context fastsasa_context;

int fastsasa_context_create(fastsasa_context **context);
/* Selects FASTSASA_PRECISION_FP64 or FASTSASA_PRECISION_FP32 for later
 * calculations on this context. Returns FASTSASA_DEVICE_UNSUPPORTED when a
 * Vulkan device lacks shaderFloat64 and FP64 is requested; select FP32 or use
 * the CPU backend in that case. */
int fastsasa_context_set_precision(fastsasa_context *context, int precision);
/* Returns the context's current precision constant (FP64 by default). */
int fastsasa_context_precision(const fastsasa_context *context);
/* Returns "cuda", "vulkan", or "none" for an initialized context. */
const char *fastsasa_context_backend(const fastsasa_context *context);
void fastsasa_context_free(fastsasa_context *context);
int fastsasa_context_synchronize(fastsasa_context *context);
int fastsasa_context_enable_profile(fastsasa_context *context, int enabled);
int fastsasa_context_last_cell_profile(fastsasa_context *context,
                                     fastsasa_cell_profile *profile);

int fastsasa_host_alloc(void **ptr, size_t bytes);
void fastsasa_host_free(void *ptr);

int fastsasa_shrake_rupley_csr(const fastsasa_sr_input *input, double *sasa);
int fastsasa_context_shrake_rupley_csr(fastsasa_context *context,
                                     const fastsasa_sr_input *input,
                                     double *sasa);
int fastsasa_context_shrake_rupley_cell_list(fastsasa_context *context,
                                           const fastsasa_sr_input *input,
                                           double *sasa);
/*
 * Per-point exposed/buried surface mask for every atom in input (VMD-style
 * surface-point export), not a reduced SASA total - the CPU-equivalent
 * function is fastsasa_cpu_exposed_points(). input->radii must already
 * include the probe radius, same convention as the other
 * fastsasa_context_shrake_rupley_* functions. exposed must have room for
 * input->n_atoms * input->n_points bytes, laid out
 * exposed[atom * n_points + point].
 *
 * Vulkan only, and FP64 only, this round: returns FASTSASA_NO_DEVICE for a
 * CUDA-backed context or when Vulkan support was not compiled in, and
 * FASTSASA_DEVICE_UNSUPPORTED when the Vulkan device lacks shaderFloat64.
 * Callers should fall back to fastsasa_cpu_exposed_points() on any non-zero
 * return.
 */
int fastsasa_context_shrake_rupley_exposed_points_cell_list(fastsasa_context *context,
                                                           const fastsasa_sr_input *input,
                                                           unsigned char *exposed);
int fastsasa_context_shrake_rupley_cell_list_async(fastsasa_context *context,
                                                 const fastsasa_sr_input *input,
                                                 double *sasa);
int fastsasa_context_shrake_rupley_cell_list_total(fastsasa_context *context,
                                                 const fastsasa_sr_input *input,
                                                 double *total_sasa);
int fastsasa_context_shrake_rupley_cell_list_total_async(fastsasa_context *context,
                                                       const fastsasa_sr_input *input,
                                                       double *total_sasa);
int fastsasa_context_lee_richards(fastsasa_context *context,
                                const fastsasa_sr_input *input,
                                double *sasa);

int fastsasa_check_device(void);
unsigned int fastsasa_abi_version(void);
size_t fastsasa_sizeof_sr_input(void);
size_t fastsasa_offsetof_sr_input_active_center_mask(void);
size_t fastsasa_offsetof_sr_input_active_center_indices(void);
size_t fastsasa_offsetof_sr_input_n_active_centers(void);
size_t fastsasa_offsetof_sr_input_force_double_precision(void);
int fastsasa_constant_test_point_limit(void);
int fastsasa_recommended_trajectory_batch_size(int n_atoms,
                                             int n_frames,
                                             int n_points,
                                             int selection_only);
int fastsasa_recommended_parallel_frames(int n_atoms,
                                       int n_points,
                                       int batch_size,
                                       int selection_only);

/*
 * Fixed-order Kahan sums shared by every backend and language binding, so
 * totals, residue sums and selection sums are the same double no matter
 * where the per-atom areas were computed.
 */
int fastsasa_sum_atoms(const double *atom_sasa, int n_atoms, double *total);
int fastsasa_sum_residues(const double *atom_sasa,
                        const int *residue_ids,
                        int n_atoms,
                        int n_residues,
                        double *residue_sasa);
int fastsasa_sum_selections(const double *atom_sasa,
                          const unsigned int *selection_masks,
                          int n_atoms,
                          int n_selections,
                          double *selection_sasa);
const char *fastsasa_status_string(int status);
const char *fastsasa_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
