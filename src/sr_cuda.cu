#include "fastsasa_device.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* A macro rather than a file-scope const: MSVC-hosted nvcc rejects host
 * constants referenced from device code. */
#define FASTSASA_PI 3.141592653589793238462643383279502884
static const int FASTSASA_NEIGHBOR_CHUNK = 128;
static const int FASTSASA_LR_ARC_CAPACITY = 64;

#include "fastsasa_exact_math.h"
#include "fastsasa_cpu.h"

/*
 * Bit-reproducible FP64 arithmetic. nvcc fuses a*b+c into FMA by default,
 * which rounds differently from the CPU reference; every operation on the
 * FP64 paths that must match the host bit for bit goes through these
 * non-contractible intrinsics so both sides execute the same correctly
 * rounded sequence.
 */
__device__ __forceinline__ static double fastsasa_dmul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ static double fastsasa_dadd(double a, double b) { return __dadd_rn(a, b); }
__device__ __forceinline__ static double fastsasa_dsub(double a, double b) { return __dsub_rn(a, b); }

/* Per-atom Shrake-Rupley area with the validated CPU association:
 * ((((4*pi)*r)*r)*count)/n. */
__device__ __forceinline__ static double
sr_atom_area(double ri, int exposed, int n_points)
{
    return fastsasa_dmul(fastsasa_dmul(fastsasa_dmul(4.0 * FASTSASA_PI, ri), ri), (double)exposed) / (double)n_points;
}

/* Pending host-side aggregation of one launch: per-atom areas land in pinned
 * memory and totals, residue and selection sums are formed on the host in
 * fixed atom order with Kahan compensation, exactly like the CPU backend,
 * instead of device atomics whose accumulation order is not reproducible.
 * Entries stay queued until the stream is synchronized. */
typedef struct {
    double *h_sasa;
    size_t h_sasa_capacity;
    int n_atoms;
    double *total_sasa;
    const int *residue_ids;
    int n_residues;
    double *residue_sasa;
    const unsigned int *selection_masks;
    int n_selections;
    double *selection_sasa;
} fastsasa_pending_aggregate;

/* Bound on queued launches before the stream is drained, which also bounds
 * the pinned memory held by the queue. */
#define FASTSASA_MAX_PENDING_AGGREGATES 64
static const long long FASTSASA_MAX_DENSE_CELLS = 16LL * 1024LL * 1024LL;
static const long long FASTSASA_MAX_DENSE_CELLS_PER_ATOM = 4096LL;
static thread_local cudaError_t last_cuda_error = cudaSuccess;

__constant__ static double const_test_points[3 * 2048];
__constant__ static int const_neighbor_cell_offsets[81] = {
    0, 0, 0,
    1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1,
    1, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1, 0,
    1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, -1,
    0, 1, 1, 0, 1, -1, 0, -1, 1, 0, -1, -1,
    1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1,
    -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1
};

struct fastsasa_device_context {
    double *d_xyz;
    double *d_x;
    double *d_y;
    double *d_z;
    double *d_radii;
    double *d_radii2;
    double *d_test_points;
    double *d_sasa;
    double *d_total_sasa;
    float *d_sasa_f;
    float *d_xf;
    float *d_yf;
    float *d_zf;
    float *d_radii_f;
    float *d_radii2_f;
    float *d_test_points_f;
    int *d_residue_ids;
    double *d_residue_sasa;
    unsigned int *d_selection_masks;
    double *d_selection_sasa;
    int *d_active_center_indices;
    int *d_neighbor_offsets;
    int *d_neighbor_indices;
    int *d_atom_cells;
    int *d_cell_counts;
    int *d_cell_offsets;
    int *d_cell_fill;
    int *d_cell_atoms;
    int *d_atom_indices;
    int *d_sorted_cells;
    void *d_cell_scan_storage;
    void *d_cell_sort_storage;
    size_t xyz_capacity;
    size_t x_capacity;
    size_t y_capacity;
    size_t z_capacity;
    size_t radii_capacity;
    size_t radii2_capacity;
    size_t test_points_capacity;
    size_t sasa_capacity;
    size_t total_sasa_capacity;
    size_t sasa_f_capacity;
    size_t xf_capacity;
    size_t yf_capacity;
    size_t zf_capacity;
    size_t radii_f_capacity;
    size_t radii2_f_capacity;
    size_t test_points_f_capacity;
    size_t residue_ids_capacity;
    size_t residue_sasa_capacity;
    size_t selection_masks_capacity;
    size_t selection_sasa_capacity;
    size_t active_center_indices_capacity;
    size_t neighbor_offsets_capacity;
    size_t neighbor_indices_capacity;
    size_t atom_cells_capacity;
    size_t cell_counts_capacity;
    size_t cell_offsets_capacity;
    size_t cell_fill_capacity;
    size_t cell_atoms_capacity;
    size_t atom_indices_capacity;
    size_t sorted_cells_capacity;
    size_t cell_scan_storage_capacity;
    size_t cell_sort_storage_capacity;
    int reusable_test_points_n;
    int reusable_const_test_points_n;
    cudaStream_t stream;
    int profile_enabled;
    fastsasa_device_cell_profile last_cell_profile;
    fastsasa_pending_aggregate *pending;
    int n_pending;
    int pending_capacity;
    double *d_lr_slice_areas;
    size_t lr_slice_areas_capacity;
    int *d_lr_overflow;
    size_t lr_overflow_capacity;
};

static int
cuda_status(cudaError_t status)
{
    last_cuda_error = status;
    return status == cudaSuccess ? FASTSASA_SUCCESS : FASTSASA_CUDA_ERROR;
}

static int current_device_properties(cudaDeviceProp *prop);

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
        return "host memory allocation error";
    case FASTSASA_NO_DEVICE:
        return "no CUDA device";
    case FASTSASA_DEVICE_UNSUPPORTED:
        return "CUDA device unsupported for this launch";
    case FASTSASA_CELL_GRID_TOO_LARGE:
        return "coordinate bounds require an unsupported dense cell grid; wrap or filter input coordinates";
    default:
        return "unknown GPU status";
    }
}

const char *
fastsasa_device_last_error(void)
{
    return cudaGetErrorString(last_cuda_error);
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
    cudaDeviceProp prop;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int status;
    int recommended = 8;

    if (n_atoms <= 0 || n_frames == 0 || n_points <= 0) return 8;
    status = current_device_properties(&prop);
    if (status != FASTSASA_SUCCESS) return 8;
    if (cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes)) != FASTSASA_SUCCESS) {
        free_bytes = 0;
    }

    /*
     * Current trajectory batching mainly changes host staging size; device
     * memory is dominated by reusable per-frame context buffers. Keep this
     * conservative so small GPUs such as T600 remain comfortable.
     */
    if (prop.totalGlobalMem <= 5ull * 1024ull * 1024ull * 1024ull) recommended = 4;
    else if (prop.totalGlobalMem >= 12ull * 1024ull * 1024ull * 1024ull) recommended = 8;
    if (selection_only && prop.multiProcessorCount >= 24) recommended *= 2;
    if (n_atoms < 5000 && prop.totalGlobalMem >= 8ull * 1024ull * 1024ull * 1024ull) recommended *= 2;
    if (free_bytes > 0 && free_bytes < 1024ull * 1024ull * 1024ull) recommended = recommended > 2 ? 2 : recommended;
    if (n_frames > 0 && recommended > n_frames) recommended = n_frames;
    if (recommended < 1) recommended = 1;
    if (recommended > 64) recommended = 64;
    return recommended;
}

int
fastsasa_device_recommended_parallel_frames(int n_atoms,
                                          int n_points,
                                          int batch_size,
                                          int selection_only)
{
    cudaDeviceProp prop;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    int status;
    int lanes = 1;
    const char *override_value = getenv("FASTSASA_GPU_FRAME_LANES");

    if (batch_size <= 1 || n_atoms <= 0 || n_points <= 0) return 1;
    if (override_value != NULL && override_value[0] != '\0') {
        char *end = NULL;
        long parsed = strtol(override_value, &end, 10);

        if (end != override_value && *end == '\0' && parsed > 0 && parsed <= 16) {
            return (int)(parsed < batch_size ? parsed : batch_size);
        }
    }
    status = current_device_properties(&prop);
    if (status != FASTSASA_SUCCESS) return 1;
    if (cuda_status(cudaMemGetInfo(&free_bytes, &total_bytes)) != FASTSASA_SUCCESS) {
        free_bytes = 0;
    }

    if (selection_only) {
        if (prop.multiProcessorCount >= 30) lanes = 4;
        else if (prop.multiProcessorCount >= 16) lanes = 2;
    } else if (n_atoms < 8000 && prop.multiProcessorCount >= 24) {
        lanes = 2;
    } else if (n_atoms >= 8000 && n_points <= 128 && prop.multiProcessorCount >= 48) {
        lanes = 2;
    }
    if (prop.totalGlobalMem <= 5ull * 1024ull * 1024ull * 1024ull && lanes > 2) lanes = 2;
    if (free_bytes > 0 && free_bytes < 1536ull * 1024ull * 1024ull) lanes = 1;
    if (lanes > batch_size) lanes = batch_size;
    if (lanes < 1) lanes = 1;
    return lanes;
}

int
fastsasa_device_check_device(void)
{
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);

    if (status != cudaSuccess) return cuda_status(status);
    if (device_count <= 0) return FASTSASA_NO_DEVICE;

    return FASTSASA_SUCCESS;
}

static int
current_device_properties(cudaDeviceProp *prop)
{
    static thread_local cudaDeviceProp cached_prop;
    static thread_local int cached_device = -1;
    int device = 0;
    int status = fastsasa_device_check_device();

    if (status != FASTSASA_SUCCESS) return status;
    if (cuda_status(cudaGetDevice(&device)) != FASTSASA_SUCCESS) return FASTSASA_CUDA_ERROR;
    if (cached_device != device) {
        if (cuda_status(cudaGetDeviceProperties(&cached_prop, device)) != FASTSASA_SUCCESS) {
            return FASTSASA_CUDA_ERROR;
        }
        cached_device = device;
    }
    *prop = cached_prop;
    return FASTSASA_SUCCESS;
}

static int
validate_launch_size(int n_atoms,
                     int threads_per_block)
{
    cudaDeviceProp prop;
    int status = current_device_properties(&prop);

    if (status != FASTSASA_SUCCESS) return status;
    if (threads_per_block > prop.maxThreadsPerBlock || n_atoms > prop.maxGridSize[0]) {
        return FASTSASA_DEVICE_UNSUPPORTED;
    }

    return FASTSASA_SUCCESS;
}

static int
validate_shared_memory_size(size_t dynamic_shared_bytes)
{
    cudaDeviceProp prop;
    int status = current_device_properties(&prop);

    if (status != FASTSASA_SUCCESS) return status;
    if (dynamic_shared_bytes > prop.sharedMemPerBlock) {
        return FASTSASA_DEVICE_UNSUPPORTED;
    }
    return FASTSASA_SUCCESS;
}

static int validate_finite_geometry(const fastsasa_device_sr_input *input,
                                    int validate_test_points);

static int
validate_input(const fastsasa_device_sr_input *input,
               const double *sasa)
{
    if (input == NULL || sasa == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (input->n_atoms <= 0 || input->n_points <= 0 || input->n_points > INT_MAX / 3) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->xyz == NULL || input->radii == NULL || input->test_points == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->neighbor_offsets == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->n_neighbor_indices < 0) return FASTSASA_INVALID_ARGUMENT;
    if (input->n_neighbor_indices > 0 && input->neighbor_indices == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->neighbor_offsets[0] != 0 ||
        input->neighbor_offsets[input->n_atoms] != input->n_neighbor_indices) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    for (int atom = 0; atom < input->n_atoms; ++atom) {
        if (input->neighbor_offsets[atom] < 0 ||
            input->neighbor_offsets[atom] > input->neighbor_offsets[atom + 1] ||
            input->neighbor_offsets[atom + 1] > input->n_neighbor_indices) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    for (int neighbor = 0; neighbor < input->n_neighbor_indices; ++neighbor) {
        if (input->neighbor_indices[neighbor] < 0 ||
            input->neighbor_indices[neighbor] >= input->n_atoms) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (input->residue_ids != NULL || input->residue_sasa != NULL || input->n_residues != 0) {
        if (input->residue_ids == NULL || input->residue_sasa == NULL || input->n_residues <= 0) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (input->selection_masks != NULL || input->selection_sasa != NULL || input->n_selections != 0) {
        if (input->selection_masks == NULL || input->selection_sasa == NULL ||
            input->n_selections <= 0 || input->n_selections > 31) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (input->active_center_mask != 0u) {
        if (input->selection_masks == NULL) return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->active_center_indices != NULL || input->n_active_centers != 0) {
        if (input->active_center_indices == NULL ||
            input->n_active_centers <= 0 ||
            input->n_active_centers > input->n_atoms) {
            return FASTSASA_INVALID_ARGUMENT;
        }
        for (int center = 0; center < input->n_active_centers; ++center) {
            if (input->active_center_indices[center] < 0 ||
                input->active_center_indices[center] >= input->n_atoms) {
                return FASTSASA_INVALID_ARGUMENT;
            }
        }
    }
    return validate_finite_geometry(input, 1);
}

static int
input_has_soa_coords(const fastsasa_device_sr_input *input)
{
    return input->x != NULL && input->y != NULL && input->z != NULL;
}

static int
input_has_partial_soa_coords(const fastsasa_device_sr_input *input)
{
    return input->x != NULL || input->y != NULL || input->z != NULL;
}

static double
input_coord_x(const fastsasa_device_sr_input *input,
              int atom)
{
    return input_has_soa_coords(input) ? input->x[atom] : input->xyz[3 * atom];
}

static double
input_coord_y(const fastsasa_device_sr_input *input,
              int atom)
{
    return input_has_soa_coords(input) ? input->y[atom] : input->xyz[3 * atom + 1];
}

static double
input_coord_z(const fastsasa_device_sr_input *input,
              int atom)
{
    return input_has_soa_coords(input) ? input->z[atom] : input->xyz[3 * atom + 2];
}

static int
validate_finite_geometry(const fastsasa_device_sr_input *input,
                         int validate_test_points)
{
    for (int atom = 0; atom < input->n_atoms; ++atom) {
        if (!isfinite(input_coord_x(input, atom)) ||
            !isfinite(input_coord_y(input, atom)) ||
            !isfinite(input_coord_z(input, atom)) ||
            !isfinite(input->radii[atom]) ||
            input->radii[atom] <= 0.0) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (validate_test_points) {
        for (int point = 0; point < 3 * input->n_points; ++point) {
            if (!isfinite(input->test_points[point])) return FASTSASA_INVALID_ARGUMENT;
        }
    }
    return FASTSASA_SUCCESS;
}

static int
cell_grid_dimension(double span,
                    double cell_size,
                    int *dimension)
{
    const double cells = ceil(span / cell_size);

    if (!isfinite(cells) || cells <= 0.0 || cells > 2147483647.0) {
        return FASTSASA_CELL_GRID_TOO_LARGE;
    }
    *dimension = (int)cells;
    return FASTSASA_SUCCESS;
}

static int
validate_dense_cell_grid(int n_atoms,
                         int nx,
                         int ny,
                         int nz,
                         int *n_cells)
{
    const long long count = (long long)nx * (long long)ny * (long long)nz;
    const long long atom_limit = FASTSASA_MAX_DENSE_CELLS_PER_ATOM * (long long)n_atoms;

    if (nx <= 0 || ny <= 0 || nz <= 0 || count <= 0 ||
        count > 2147483647LL || count > FASTSASA_MAX_DENSE_CELLS ||
        count > atom_limit) {
        return FASTSASA_CELL_GRID_TOO_LARGE;
    }
    *n_cells = (int)count;
    return FASTSASA_SUCCESS;
}

static int
validate_cell_input(const fastsasa_device_sr_input *input,
                    const double *sasa)
{
    if (input == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (input->n_atoms <= 0 || input->n_points <= 0 || input->n_points > INT_MAX / 3) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->xyz == NULL && !input_has_soa_coords(input)) return FASTSASA_INVALID_ARGUMENT;
    if (input->xyz != NULL && input_has_partial_soa_coords(input)) return FASTSASA_INVALID_ARGUMENT;
    if (input->xyz == NULL && input_has_partial_soa_coords(input) && !input_has_soa_coords(input)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->radii == NULL || input->test_points == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->residue_ids != NULL || input->residue_sasa != NULL || input->n_residues != 0) {
        if (input->residue_ids == NULL || input->residue_sasa == NULL || input->n_residues <= 0) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (input->selection_masks != NULL || input->selection_sasa != NULL || input->n_selections != 0) {
        if (input->selection_masks == NULL || input->selection_sasa == NULL ||
            input->n_selections <= 0 || input->n_selections > 31) {
            return FASTSASA_INVALID_ARGUMENT;
        }
    }
    if (input->active_center_mask != 0u) {
        if (input->selection_masks == NULL) return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->active_center_indices != NULL || input->n_active_centers != 0) {
        if (input->active_center_indices == NULL ||
            input->n_active_centers <= 0 ||
            input->n_active_centers > input->n_atoms) {
            return FASTSASA_INVALID_ARGUMENT;
        }
        for (int center = 0; center < input->n_active_centers; ++center) {
            if (input->active_center_indices[center] < 0 ||
                input->active_center_indices[center] >= input->n_atoms) {
                return FASTSASA_INVALID_ARGUMENT;
            }
        }
    }
    return validate_finite_geometry(input, 1);
}

/*
 * Totals, residue and selection sums from a host copy of the per-atom
 * areas, using the same fixed-order Kahan accumulation as the CPU backend
 * (cpu_sum_atoms/cpu_sum_residues/cpu_sum_selections) so every backend
 * reports the same double.
 */
static int
host_aggregate_sasa(const double *sasa,
                    int n_atoms,
                    double *total_sasa,
                    const int *residue_ids,
                    int n_residues,
                    double *residue_sasa,
                    const unsigned int *selection_masks,
                    int n_selections,
                    double *selection_sasa)
{
    if (total_sasa != NULL) {
        double sum = 0.0;
        double compensation = 0.0;

        for (int atom = 0; atom < n_atoms; ++atom) {
            fastsasa_exact_kahan_add(sasa[atom], &sum, &compensation);
        }
        *total_sasa = sum;
    }
    if (residue_sasa != NULL && residue_ids != NULL && n_residues > 0) {
        double *compensation = (double *)calloc((size_t)n_residues, sizeof(double));

        if (compensation == NULL) return FASTSASA_MEMORY_ERROR;
        memset(residue_sasa, 0, sizeof(double) * (size_t)n_residues);
        for (int atom = 0; atom < n_atoms; ++atom) {
            const int residue = residue_ids[atom];

            if (residue >= 0 && residue < n_residues) {
                fastsasa_exact_kahan_add(sasa[atom], &residue_sasa[residue], &compensation[residue]);
            }
        }
        free(compensation);
    }
    if (selection_sasa != NULL && selection_masks != NULL && n_selections > 0) {
        double compensation[32];

        if (n_selections > 31) return FASTSASA_INVALID_ARGUMENT;
        memset(compensation, 0, sizeof(compensation));
        memset(selection_sasa, 0, sizeof(double) * (size_t)n_selections);
        for (int atom = 0; atom < n_atoms; ++atom) {
            for (int selection = 0; selection < n_selections; ++selection) {
                if ((selection_masks[atom] & (1u << selection)) != 0u) {
                    fastsasa_exact_kahan_add(sasa[atom],
                                           &selection_sasa[selection],
                                           &compensation[selection]);
                }
            }
        }
    }
    return FASTSASA_SUCCESS;
}

static int
flush_pending_aggregates(fastsasa_device_context *context)
{
    int status = FASTSASA_SUCCESS;

    for (int i = 0; i < context->n_pending; ++i) {
        const fastsasa_pending_aggregate *entry = &context->pending[i];
        const int entry_status = host_aggregate_sasa(entry->h_sasa,
                                                     entry->n_atoms,
                                                     entry->total_sasa,
                                                     entry->residue_ids,
                                                     entry->n_residues,
                                                     entry->residue_sasa,
                                                     entry->selection_masks,
                                                     entry->n_selections,
                                                     entry->selection_sasa);

        if (entry_status != FASTSASA_SUCCESS && status == FASTSASA_SUCCESS) status = entry_status;
    }
    context->n_pending = 0;
    return status;
}

/* Drains the stream and applies every queued aggregation. */
static int
synchronize_and_flush(fastsasa_device_context *context)
{
    const int status = cuda_status(cudaStreamSynchronize(context->stream));

    if (status != FASTSASA_SUCCESS) {
        context->n_pending = 0;
        return status;
    }
    return flush_pending_aggregates(context);
}

/* Reserves a queue entry (with pinned space for n_atoms areas) for the
 * launch described by input; the caller enqueues the device-to-host copy. */
static int
pending_aggregate_push(fastsasa_device_context *context,
                       const fastsasa_device_sr_input *input,
                       double *total_sasa,
                       fastsasa_pending_aggregate **entry_out)
{
    const size_t bytes = sizeof(double) * (size_t)input->n_atoms;
    fastsasa_pending_aggregate *entry;

    *entry_out = NULL;
    if (context->n_pending >= FASTSASA_MAX_PENDING_AGGREGATES) {
        const int status = synchronize_and_flush(context);

        if (status != FASTSASA_SUCCESS) return status;
    }
    if (context->n_pending == context->pending_capacity) {
        const int grown = context->pending_capacity == 0 ? 8 : context->pending_capacity * 2;
        fastsasa_pending_aggregate *replacement = (fastsasa_pending_aggregate *)realloc(
            context->pending, sizeof(fastsasa_pending_aggregate) * (size_t)grown);

        if (replacement == NULL) return FASTSASA_MEMORY_ERROR;
        memset(replacement + context->pending_capacity, 0,
               sizeof(fastsasa_pending_aggregate) * (size_t)(grown - context->pending_capacity));
        context->pending = replacement;
        context->pending_capacity = grown;
    }
    entry = &context->pending[context->n_pending];
    if (entry->h_sasa_capacity < bytes) {
        cudaFreeHost(entry->h_sasa);
        entry->h_sasa = NULL;
        entry->h_sasa_capacity = 0;
        if (cuda_status(cudaHostAlloc((void **)&entry->h_sasa, bytes, cudaHostAllocDefault)) != FASTSASA_SUCCESS) {
            return FASTSASA_CUDA_ERROR;
        }
        entry->h_sasa_capacity = bytes;
    }
    entry->n_atoms = input->n_atoms;
    entry->total_sasa = total_sasa;
    entry->residue_ids = input->residue_ids;
    entry->n_residues = input->n_residues;
    entry->residue_sasa = input->residue_sasa;
    entry->selection_masks = input->selection_masks;
    entry->n_selections = input->n_selections;
    entry->selection_sasa = input->selection_sasa;
    ++context->n_pending;
    *entry_out = entry;
    return FASTSASA_SUCCESS;
}

static int
ensure_device_capacity(void **ptr,
                       size_t *capacity,
                       size_t bytes)
{
    if (bytes == 0) return FASTSASA_SUCCESS;
    if (*capacity >= bytes) return FASTSASA_SUCCESS;

    cudaFree(*ptr);
    *ptr = NULL;
    *capacity = 0;

    if (cuda_status(cudaMalloc(ptr, bytes)) != FASTSASA_SUCCESS) {
        return FASTSASA_CUDA_ERROR;
    }

    *capacity = bytes;
    return FASTSASA_SUCCESS;
}

static int
ensure_context_capacity(fastsasa_device_context *context,
                        size_t xyz_bytes,
                        size_t radii_bytes,
                        size_t test_point_bytes,
                        size_t offset_bytes,
                        size_t index_bytes,
                        size_t sasa_bytes)
{
    int status;

    status = ensure_device_capacity((void **)&context->d_xyz,
                                    &context->xyz_capacity,
                                    xyz_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_radii,
                                    &context->radii_capacity,
                                    radii_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_radii2,
                                    &context->radii2_capacity,
                                    radii_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_test_points,
                                    &context->test_points_capacity,
                                    test_point_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_neighbor_offsets,
                                    &context->neighbor_offsets_capacity,
                                    offset_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_neighbor_indices,
                                    &context->neighbor_indices_capacity,
                                    index_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_sasa,
                                    &context->sasa_capacity,
                                    sasa_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_residue_capacity(fastsasa_device_context *context,
                        size_t residue_id_bytes,
                        size_t residue_sasa_bytes)
{
    int status;

    status = ensure_device_capacity((void **)&context->d_residue_ids,
                                    &context->residue_ids_capacity,
                                    residue_id_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_residue_sasa,
                                    &context->residue_sasa_capacity,
                                    residue_sasa_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_selection_capacity(fastsasa_device_context *context,
                          size_t selection_mask_bytes,
                          size_t selection_sasa_bytes)
{
    int status;

    status = ensure_device_capacity((void **)&context->d_selection_masks,
                                    &context->selection_masks_capacity,
                                    selection_mask_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_selection_sasa,
                                    &context->selection_sasa_capacity,
                                    selection_sasa_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_active_center_capacity(fastsasa_device_context *context,
                              size_t active_center_bytes)
{
    return ensure_device_capacity((void **)&context->d_active_center_indices,
                                  &context->active_center_indices_capacity,
                                  active_center_bytes);
}

static int
ensure_soa_capacity(fastsasa_device_context *context,
                    size_t coord_bytes)
{
    int status;

    status = ensure_device_capacity((void **)&context->d_x,
                                    &context->x_capacity,
                                    coord_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_y,
                                    &context->y_capacity,
                                    coord_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_z,
                                    &context->z_capacity,
                                    coord_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_float_sr_capacity(fastsasa_device_context *context,
                         size_t coord_bytes,
                         size_t test_point_bytes)
{
    int status;
    const size_t coord_float_bytes = coord_bytes / sizeof(double) * sizeof(float);
    const size_t test_point_float_bytes = test_point_bytes / sizeof(double) * sizeof(float);

    status = ensure_device_capacity((void **)&context->d_xf,
                                    &context->xf_capacity,
                                    coord_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_yf,
                                    &context->yf_capacity,
                                    coord_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_zf,
                                    &context->zf_capacity,
                                    coord_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_radii_f,
                                    &context->radii_f_capacity,
                                    coord_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_radii2_f,
                                    &context->radii2_f_capacity,
                                    coord_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_test_points_f,
                                    &context->test_points_f_capacity,
                                    test_point_float_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_float_sasa_capacity(fastsasa_device_context *context,
                           size_t sasa_bytes)
{
    const size_t sasa_float_bytes = sasa_bytes / sizeof(double) * sizeof(float);

    return ensure_device_capacity((void **)&context->d_sasa_f,
                                  &context->sasa_f_capacity,
                                  sasa_float_bytes);
}

static int
ensure_cell_capacity(fastsasa_device_context *context,
                     size_t atom_cells_bytes,
                     size_t cell_count_bytes,
                     size_t cell_offset_bytes,
                     size_t cell_atoms_bytes)
{
    int status;

    status = ensure_device_capacity((void **)&context->d_atom_cells,
                                    &context->atom_cells_capacity,
                                    atom_cells_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_cell_counts,
                                    &context->cell_counts_capacity,
                                    cell_count_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_cell_offsets,
                                    &context->cell_offsets_capacity,
                                    cell_offset_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_cell_fill,
                                    &context->cell_fill_capacity,
                                    cell_count_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_cell_atoms,
                                    &context->cell_atoms_capacity,
                                    cell_atoms_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_atom_indices,
                                    &context->atom_indices_capacity,
                                    atom_cells_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_device_capacity((void **)&context->d_sorted_cells,
                                    &context->sorted_cells_capacity,
                                    atom_cells_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
ensure_cell_scan_storage(fastsasa_device_context *context,
                         int n_cells)
{
    size_t required_bytes = 0;
    int status;

    if (cuda_status(cub::DeviceScan::InclusiveSum(NULL,
                                                  required_bytes,
                                                  context->d_cell_counts,
                                                  context->d_cell_offsets + 1,
                                                  n_cells,
                                                  context->stream)) != FASTSASA_SUCCESS) {
        return FASTSASA_CUDA_ERROR;
    }
    status = ensure_device_capacity(&context->d_cell_scan_storage,
                                    &context->cell_scan_storage_capacity,
                                    required_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
sort_end_bit(int n_cells)
{
    int bits = 0;
    unsigned int value = (unsigned int)(n_cells > 1 ? n_cells - 1 : 1);

    while (value != 0) {
        ++bits;
        value >>= 1;
    }
    return bits;
}

static int
ensure_cell_sort_storage(fastsasa_device_context *context,
                         int n_atoms,
                         int n_cells)
{
    size_t required_bytes = 0;
    int status;

    if (cuda_status(cub::DeviceRadixSort::SortPairs(NULL,
                                                    required_bytes,
                                                    context->d_atom_cells,
                                                    context->d_sorted_cells,
                                                    context->d_atom_indices,
                                                    context->d_cell_atoms,
                                                    n_atoms,
                                                    0,
                                                    sort_end_bit(n_cells),
                                                    context->stream)) != FASTSASA_SUCCESS) {
        return FASTSASA_CUDA_ERROR;
    }
    status = ensure_device_capacity(&context->d_cell_sort_storage,
                                    &context->cell_sort_storage_capacity,
                                    required_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    return FASTSASA_SUCCESS;
}

static int
create_profile_events(cudaEvent_t *events,
                      int n_events)
{
    for (int i = 0; i < n_events; ++i) {
        events[i] = NULL;
        if (cuda_status(cudaEventCreate(&events[i])) != FASTSASA_SUCCESS) {
            for (int j = 0; j <= i; ++j) {
                if (events[j] != NULL) cudaEventDestroy(events[j]);
                events[j] = NULL;
            }
            return FASTSASA_CUDA_ERROR;
        }
    }
    return FASTSASA_SUCCESS;
}

static void
destroy_profile_events(cudaEvent_t *events,
                       int n_events)
{
    for (int i = 0; i < n_events; ++i) {
        if (events[i] != NULL) cudaEventDestroy(events[i]);
        events[i] = NULL;
    }
}

static double
event_elapsed_ms(cudaEvent_t start,
                 cudaEvent_t end)
{
    float ms = 0.0f;

    if (cudaEventElapsedTime(&ms, start, end) != cudaSuccess) return -1.0;
    return (double)ms;
}

static int
shared_cache_min_points(void)
{
    const char *value = getenv("FASTSASA_SHARED_CACHE_MIN_POINTS");
    char *end = NULL;
    long parsed;

    if (value == NULL || value[0] == '\0') return 2147483647;
    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 2147483647L) return 2147483647;
    return (int)parsed;
}

typedef struct fastsasa_sr_dispatch_policy {
    int sort_cell_list;
    int ordered_cells;
    int compact_active_centers;
    int warp_atom_sr;
    int point_compaction_sr;
    int shared_neighbor_cache;
} fastsasa_sr_dispatch_policy;

static int
env_enabled(const char *name,
            int default_value)
{
    const char *value = getenv(name);

    if (value == NULL || value[0] == '\0') return default_value;
    return strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

static int
env_is_set(const char *name)
{
    const char *value = getenv(name);

    return value != NULL && value[0] != '\0';
}

static int
autotune_enabled(void)
{
    return env_enabled("FASTSASA_AUTOTUNE", 1);
}

static int
sort_cell_list_policy(int n_atoms,
                      int n_cells)
{
    const char *value = getenv("FASTSASA_SORT_CELL_LIST");
    const char *threshold_value = getenv("FASTSASA_SORT_CELL_DENSITY");
    double atoms_per_cell;
    double threshold = 64.0;
    char *end = NULL;

    if (value != NULL) return strcmp(value, "0") != 0 && value[0] != '\0';
    if (n_cells <= 0) return 0;
    if (threshold_value != NULL && threshold_value[0] != '\0') {
        const double parsed = strtod(threshold_value, &end);
        if (end != threshold_value && *end == '\0' && parsed > 0.0) threshold = parsed;
    }
    atoms_per_cell = (double)n_atoms / (double)n_cells;
    return atoms_per_cell >= threshold;
}

static int
use_float_sr(void)
{
    const char *double_value = getenv("FASTSASA_SR_DOUBLE");
    const char *value = getenv("FASTSASA_SR_FLOAT");


    if (double_value != NULL && strcmp(double_value, "0") != 0 && double_value[0] != '\0') {
        return 0;
    }
    if (value != NULL) return strcmp(value, "0") != 0 && value[0] != '\0';
    return 0;
}

static int
use_float_lr(void)
{
    const char *double_value = getenv("FASTSASA_LR_DOUBLE");
    const char *value = getenv("FASTSASA_LR_FLOAT");

    if (double_value != NULL && strcmp(double_value, "0") != 0 && double_value[0] != '\0') {
        return 0;
    }
    if (value != NULL) return strcmp(value, "0") != 0 && value[0] != '\0';
    return 0;
}

static int
use_float_lr_accumulation(void)
{
    return env_enabled("FASTSASA_LR_FLOAT_ACCUM", 0);
}

static int
use_lr_fp64_hybrid(void)
{
    /* FP32-prefiltered FP64 Lee-Richards neighbor rejection; surviving
     * neighbor pairs run the unchanged FP64 arc math, so results stay
     * identical. Set to 0 to disable. */
    return env_enabled("FASTSASA_LR_FP64_HYBRID", 1);
}

static int
use_sr_fp64_hybrid(void)
{
    /* FP32-prefiltered FP64 Shrake-Rupley; results stay identical to the
     * pure FP64 kernel, so this is on by default. Set to 0 to disable. */
    return env_enabled("FASTSASA_SR_FP64_HYBRID", 1);
}

static int
ordered_cell_traversal_policy(void)
{
    return env_enabled("FASTSASA_ORDERED_CELLS", 1);
}

static int
compact_active_centers_policy(void)
{
    return env_enabled("FASTSASA_COMPACT_CENTERS", 1);
}

static int
warp_atom_kernel_policy(int n_points,
                        int n_atoms,
                        int n_cells,
                        int selection_only)
{
    const char *value = getenv("FASTSASA_WARP_ATOMS");
    const char *max_points_value = getenv("FASTSASA_WARP_MAX_POINTS");
    const char *density_value = getenv("FASTSASA_WARP_DENSITY");
    const char *min_atoms_value = getenv("FASTSASA_WARP_MIN_ATOMS");
    int max_points = 256;
    int min_atoms = 2000;
    double density_threshold = 16.0;
    double atoms_per_cell;
    char *end = NULL;
    cudaDeviceProp prop;

    if (value != NULL) return env_enabled("FASTSASA_WARP_ATOMS", 0);
    if (!autotune_enabled()) return 0;
    if (current_device_properties(&prop) == FASTSASA_SUCCESS) {
        if (prop.multiProcessorCount >= 16) max_points = 512;
        if (prop.multiProcessorCount < 12) min_atoms = 4000;
        if (selection_only && prop.multiProcessorCount >= 16) min_atoms = 1000;
    }
    if (max_points_value != NULL && max_points_value[0] != '\0') {
        const long parsed = strtol(max_points_value, &end, 10);
        if (end != max_points_value && *end == '\0' && parsed > 0 && parsed <= 4096) {
            max_points = (int)parsed;
        }
    }
    if (density_value != NULL && density_value[0] != '\0') {
        const double parsed = strtod(density_value, &end);
        if (end != density_value && *end == '\0' && parsed > 0.0) density_threshold = parsed;
    }
    if (min_atoms_value != NULL && min_atoms_value[0] != '\0') {
        const long parsed = strtol(min_atoms_value, &end, 10);
        if (end != min_atoms_value && *end == '\0' && parsed > 0) min_atoms = (int)parsed;
    }
    if (n_points <= 256) return n_atoms >= min_atoms;
    if (n_cells <= 0) return 0;
    atoms_per_cell = (double)n_atoms / (double)n_cells;
    return n_points <= max_points && atoms_per_cell >= density_threshold;
}

static int
point_compaction_kernel_policy(int n_points)
{
    const char *value = getenv("FASTSASA_POINT_COMPACTION");
    const char *threshold_value = getenv("FASTSASA_POINT_COMPACTION_MIN_POINTS");
    int threshold = 500;
    char *end = NULL;

    if (n_points <= 0 || n_points > 2048) return 0;
    if (value != NULL) return env_enabled("FASTSASA_POINT_COMPACTION", 0);
    if (!autotune_enabled()) return 0;
    if (threshold_value != NULL && threshold_value[0] != '\0') {
        const long parsed = strtol(threshold_value, &end, 10);
        if (end != threshold_value && *end == '\0' && parsed > 0 && parsed <= 2048) {
            threshold = (int)parsed;
        }
    }
    return n_points >= threshold;
}

static fastsasa_sr_dispatch_policy
select_sr_dispatch_policy(int n_points,
                          int n_atoms,
                          int n_cells,
                          int float_sr,
                          int active_center,
                          int has_active_center_indices,
                          int n_active_centers)
{
    fastsasa_sr_dispatch_policy policy;
    const int selected_centers = active_center &&
                                 has_active_center_indices &&
                                 n_active_centers > 0;

    memset(&policy, 0, sizeof(policy));
    policy.ordered_cells = ordered_cell_traversal_policy();
    policy.sort_cell_list = autotune_enabled()
                                ? sort_cell_list_policy(n_atoms, n_cells)
                                : env_enabled("FASTSASA_SORT_CELL_LIST", 0);
    if (selected_centers &&
        !env_is_set("FASTSASA_SORT_CELL_LIST") &&
        n_active_centers * 16 < n_atoms) {
        policy.sort_cell_list = 0;
    }
    /* Center compaction (skip non-selected atoms' SR point-sampling
     * entirely) is not FP32-specific: shrake_rupley_cell_kernel and
     * shrake_rupley_cell_hybrid_kernel both accept the same
     * center_indices/n_centers/center_masks/active_center_mask parameters
     * as the float kernels, added for this reason. */
    policy.compact_active_centers = selected_centers &&
                                    compact_active_centers_policy();
    policy.warp_atom_sr = float_sr &&
                          policy.ordered_cells &&
                          warp_atom_kernel_policy(n_points,
                                                  policy.compact_active_centers ? n_active_centers : n_atoms,
                                                  n_cells,
                                                  selected_centers);
    /* A one-atom-per-warp FP64 hybrid kernel was tried and removed: reusing
     * this same density heuristic, it measured (fastsasa_context_enable_
     * profile/last_cell_profile) no change at 35,172 centers (5.11ms vs
     * 5.16ms, noise), a 28% regression at 1,530 centers (0.367ms vs
     * 0.286ms), and a 2.5x regression at 67 centers (0.174ms vs 0.070ms).
     * With n_points<=256 this heuristic's rule is just n_atoms>=min_atoms
     * (no density check), which would have selected it by default at
     * 1,530 centers -- live-regressing this benchmark's own CUDA-selected
     * throughput from 2753.6 to 2517.0 frames/s before it was caught and
     * reverted. Recorded here so the idea isn't retried the same way: the
     * one-atom-per-block hybrid kernel above is already well past the
     * occupancy point where finer-grained scheduling helps on this class
     * of GPU (register-bound at ~82% achieved vs ~83% theoretical
     * occupancy per ncu, not launch-configuration-limited), so warp-level
     * restructuring just adds per-atom loop overhead (ceil(n_points/32)
     * iterations vs ceil(n_points/block_threads)) without buying back
     * concurrency the kernel didn't already have. */
    policy.point_compaction_sr = float_sr &&
                                 policy.ordered_cells &&
                                 !policy.warp_atom_sr &&
                                 point_compaction_kernel_policy(n_points);
    policy.shared_neighbor_cache = n_points >= shared_cache_min_points();
    return policy;
}

int
fastsasa_device_context_create(fastsasa_device_context **context)
{
    int status;

    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    *context = NULL;

    status = fastsasa_device_check_device();
    if (status != FASTSASA_SUCCESS) return status;

    *context = (fastsasa_device_context *)calloc(1, sizeof(fastsasa_device_context));
    if (*context == NULL) return FASTSASA_MEMORY_ERROR;
    (*context)->reusable_test_points_n = -1;
    (*context)->reusable_const_test_points_n = -1;
    if (cuda_status(cudaStreamCreateWithFlags(&(*context)->stream,
                                              cudaStreamNonBlocking)) != FASTSASA_SUCCESS) {
        free(*context);
        *context = NULL;
        return FASTSASA_CUDA_ERROR;
    }

    return FASTSASA_SUCCESS;
}

void
fastsasa_device_context_free(fastsasa_device_context *context)
{
    if (context == NULL) return;

    cudaStreamSynchronize(context->stream);
    cudaFree(context->d_xyz);
    cudaFree(context->d_x);
    cudaFree(context->d_y);
    cudaFree(context->d_z);
    cudaFree(context->d_radii);
    cudaFree(context->d_radii2);
    cudaFree(context->d_test_points);
    cudaFree(context->d_sasa);
    cudaFree(context->d_total_sasa);
    cudaFree(context->d_sasa_f);
    cudaFree(context->d_xf);
    cudaFree(context->d_yf);
    cudaFree(context->d_zf);
    cudaFree(context->d_radii_f);
    cudaFree(context->d_radii2_f);
    cudaFree(context->d_test_points_f);
    cudaFree(context->d_residue_ids);
    cudaFree(context->d_residue_sasa);
    cudaFree(context->d_selection_masks);
    cudaFree(context->d_selection_sasa);
    cudaFree(context->d_active_center_indices);
    cudaFree(context->d_neighbor_offsets);
    cudaFree(context->d_neighbor_indices);
    cudaFree(context->d_atom_cells);
    cudaFree(context->d_cell_counts);
    cudaFree(context->d_cell_offsets);
    cudaFree(context->d_cell_fill);
    cudaFree(context->d_cell_atoms);
    cudaFree(context->d_atom_indices);
    cudaFree(context->d_sorted_cells);
    cudaFree(context->d_cell_scan_storage);
    cudaFree(context->d_cell_sort_storage);
    cudaFree(context->d_lr_slice_areas);
    cudaFree(context->d_lr_overflow);
    for (int i = 0; i < context->pending_capacity; ++i) {
        cudaFreeHost(context->pending[i].h_sasa);
    }
    free(context->pending);
    cudaStreamDestroy(context->stream);
    free(context);
}

int
fastsasa_device_context_synchronize(fastsasa_device_context *context)
{
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    return synchronize_and_flush(context);
}

int
fastsasa_device_context_enable_profile(fastsasa_device_context *context,
                                    int enabled)
{
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    context->profile_enabled = enabled ? 1 : 0;
    return FASTSASA_SUCCESS;
}

int
fastsasa_device_context_last_cell_profile(fastsasa_device_context *context,
                                       fastsasa_device_cell_profile *profile)
{
    if (context == NULL || profile == NULL) return FASTSASA_INVALID_ARGUMENT;
    *profile = context->last_cell_profile;
    return FASTSASA_SUCCESS;
}

int
fastsasa_device_host_alloc(void **ptr,
                        size_t bytes)
{
    if (ptr == NULL || bytes == 0) return FASTSASA_INVALID_ARGUMENT;
    *ptr = NULL;
    return cuda_status(cudaHostAlloc(ptr, bytes, cudaHostAllocDefault));
}

void
fastsasa_device_host_free(void *ptr)
{
    if (ptr != NULL) cudaFreeHost(ptr);
}

__global__ static void
shrake_rupley_kernel(int n_atoms,
                     int n_points,
                     const double *xyz,
                     const double *radii,
                     const double *radii2,
                     const double *test_points,
                     const int *neighbor_offsets,
                     const int *neighbor_indices,
                     double *sasa)
{
    extern __shared__ int exposed_counts[];

    const int atom = blockIdx.x;
    const int tid = threadIdx.x;
    const int first_neighbor = neighbor_offsets[atom];
    const int last_neighbor = neighbor_offsets[atom + 1];
    const double ri = radii[atom];
    const double xi = xyz[3 * atom];
    const double yi = xyz[3 * atom + 1];
    const double zi = xyz[3 * atom + 2];

    int exposed = 0;

    for (int point_base = 0; point_base < n_points; point_base += blockDim.x) {
        const int point = point_base + tid;
        const int active_point = point < n_points;
        const double px = active_point ? fastsasa_dadd(xi, fastsasa_dmul(ri, test_points[3 * point])) : 0.0;
        const double py = active_point ? fastsasa_dadd(yi, fastsasa_dmul(ri, test_points[3 * point + 1])) : 0.0;
        const double pz = active_point ? fastsasa_dadd(zi, fastsasa_dmul(ri, test_points[3 * point + 2])) : 0.0;
        int buried = 0;

        for (int n = first_neighbor; n < last_neighbor; ++n) {
            const int other = neighbor_indices[n];
            const double dx = px - xyz[3 * other];
            const double dy = py - xyz[3 * other + 1];
            const double dz = pz - xyz[3 * other + 2];
            if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy, dy)), fastsasa_dmul(dz, dz)) < radii2[other]) {
                buried = 1;
                break;
            }
        }

        if (!buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sasa[atom] = sr_atom_area(ri, exposed_counts[0], n_points);
    }
}

__global__ static void
shrake_rupley_const_points_kernel(int n_atoms,
                                  int n_points,
                                  const double *xyz,
                                  const double *radii,
                                  const double *radii2,
                                  const int *neighbor_offsets,
                                  const int *neighbor_indices,
                                  double *sasa)
{
    extern __shared__ int exposed_counts[];

    const int atom = blockIdx.x;
    const int tid = threadIdx.x;
    const int first_neighbor = neighbor_offsets[atom];
    const int last_neighbor = neighbor_offsets[atom + 1];
    const double ri = radii[atom];
    const double xi = xyz[3 * atom];
    const double yi = xyz[3 * atom + 1];
    const double zi = xyz[3 * atom + 2];

    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const double px = fastsasa_dadd(xi, fastsasa_dmul(ri, const_test_points[3 * point]));
        const double py = fastsasa_dadd(yi, fastsasa_dmul(ri, const_test_points[3 * point + 1]));
        const double pz = fastsasa_dadd(zi, fastsasa_dmul(ri, const_test_points[3 * point + 2]));
        int buried = 0;

        for (int n = first_neighbor; n < last_neighbor; ++n) {
            const int other = neighbor_indices[n];
            const double dx = px - xyz[3 * other];
            const double dy = py - xyz[3 * other + 1];
            const double dz = pz - xyz[3 * other + 2];
            if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy, dy)), fastsasa_dmul(dz, dz)) < radii2[other]) {
                buried = 1;
                break;
            }
        }

        if (!buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sasa[atom] = sr_atom_area(ri, exposed_counts[0], n_points);
    }
}

__device__ static double
atomic_add_double(double *address,
                  double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

__device__ static int
block_reduce_sum_int(int value,
                     int *warp_sums)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();

    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
    }
    return value;
}

__device__ static int
warp_reduce_sum_int(int value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

static int
sr_block_threads(int n_points)
{
    /* Several legacy shared-memory reductions require a power-of-two block. */
    if (n_points <= 32) return 32;
    if (n_points <= 64) return 64;
    if (n_points <= 128) return 128;
    return 256;
}

/*
 * FP32 prefilter state for the FP64 Lee-Richards path. The two rejection
 * tests below only ever reject neighbor pairs that the FP64 code would also
 * reject (slab miss, or in-plane distance provably beyond touching), using
 * conservative margins, so surviving pairs and therefore all arc decisions
 * are identical to the pure FP64 kernel.
 */
typedef struct {
    const float *xf;
    const float *yf;
    const float *zf;
    const float *rf;
    float xif;
    float yif;
    float slice_zf;
    float ri_prime_ubf;   /* upper bound of ri_prime */
    float linear_margin;
    float squared_margin; /* bounds the error of every f32 squared term */
} lr_prefilter;

__device__ static int
lr_prefilter_rejects(const lr_prefilter *pf, int other)
{
    const float djf = fabsf(pf->zf[other] - pf->slice_zf);
    const float rjf = pf->rf[other];
    float rj_prime2f, dxf, dyf, dij2f, dij_lb2, rj_ub, dij_lb;

    if (djf >= rjf + pf->linear_margin) return 1;
    rj_prime2f = fmaf(rjf, rjf, -(djf * djf));
    dxf = pf->xf[other] - pf->xif;
    dyf = pf->yf[other] - pf->yif;
    dij2f = fmaf(dxf, dxf, dyf * dyf);
    /* Upper-bound rj_prime and lower-bound dij so the comparison is a
     * certain reject; sqrtf is monotone, so padding the argument by the
     * squared-error margin preserves the bound direction. */
    dij_lb2 = dij2f - pf->squared_margin;
    if (dij_lb2 <= 0.0f) return 0;
    rj_ub = sqrtf(fmaxf(rj_prime2f, 0.0f) + pf->squared_margin);
    dij_lb = sqrtf(dij_lb2);
    return dij_lb > pf->ri_prime_ubf + rj_ub;
}

__device__ static int
lr_neighbor_arcs(int atom,
                 int other,
                 double slice_z,
                 double ri_prime,
                 double ri_prime2,
                 double xi,
                 double yi,
                 const double *x,
                 const double *y,
                 const double *z,
                 const double *radii,
                 double *a0,
                 double *b0,
                 double *a1,
                 double *b1,
                 const lr_prefilter *pf)
{
    double rj, dj, rj_prime2, rj_prime, dx, dy, dij;
    double alpha_arg, alpha, beta, inf, sup;

    if (other == atom) return 0;
    if (pf != NULL && lr_prefilter_rejects(pf, other)) return 0;
    /* Same operation sequence as lee_richards_atom_area in fastsasa_cpu.cc. */
    rj = radii[other];
    dj = fabs(fastsasa_dsub(z[other], slice_z));
    if (dj >= rj) return 0;
    rj_prime2 = fastsasa_dsub(fastsasa_dmul(rj, rj), fastsasa_dmul(dj, dj));
    if (rj_prime2 <= 0.0) return 0;
    rj_prime = sqrt(rj_prime2);
    dx = fastsasa_dsub(x[other], xi);
    dy = fastsasa_dsub(y[other], yi);
    dij = sqrt(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy, dy)));

    if (dij >= fastsasa_dadd(ri_prime, rj_prime)) return 0;
    if (dij < 1e-14) return rj_prime >= ri_prime ? -1 : 0;
    if (fastsasa_dadd(dij, ri_prime) < rj_prime) return -1;
    if (fastsasa_dadd(dij, rj_prime) < ri_prime) return 0;

    alpha_arg = fastsasa_dsub(fastsasa_dadd(ri_prime2, fastsasa_dmul(dij, dij)), rj_prime2) /
                fastsasa_dmul(fastsasa_dmul(2.0, ri_prime), dij);
    if (alpha_arg < -1.0) alpha_arg = -1.0;
    if (alpha_arg > 1.0) alpha_arg = 1.0;
    alpha = fastsasa_exact_acos(alpha_arg);
    beta = fastsasa_dadd(fastsasa_exact_atan2(dy, dx), FASTSASA_PI);
    inf = fastsasa_dsub(beta, alpha);
    sup = fastsasa_dadd(beta, alpha);
    if (inf < 0.0) inf = fastsasa_dadd(inf, 2.0 * FASTSASA_PI);
    if (sup > 2.0 * FASTSASA_PI) sup = fastsasa_dsub(sup, 2.0 * FASTSASA_PI);

    if (sup < inf) {
        *a0 = 0.0;
        *b0 = sup;
        *a1 = inf;
        *b1 = 2.0 * FASTSASA_PI;
        return 2;
    }
    *a0 = inf;
    *b0 = sup;
    return 1;
}

__device__ static int
lr_insert_arc_sorted(double begin,
                     double end,
                     double *arc_begin,
                     double *arc_end,
                     int *n_arcs)
{
    int pos;

    if (*n_arcs >= FASTSASA_LR_ARC_CAPACITY) return 0;
    pos = *n_arcs;
    while (pos > 0 && arc_begin[pos - 1] > begin) {
        arc_begin[pos] = arc_begin[pos - 1];
        arc_end[pos] = arc_end[pos - 1];
        --pos;
    }
    arc_begin[pos] = begin;
    arc_end[pos] = end;
    ++(*n_arcs);
    return 1;
}

/*
 * Exposed arc length of one slice circle, or 0 with *buried set when a
 * neighbor covers the whole circle. Arcs are kept sorted by begin angle and
 * the gaps are summed in that order, then closed with (exposed + 2*pi) -
 * covered, exactly like exposed_arc_length in fastsasa_cpu.cc. When more than
 * FASTSASA_LR_ARC_CAPACITY arcs are needed, *overflow is set and the atom is
 * recomputed on the host by the caller.
 */
__device__ static double
lr_exposed_arc_length_local_cell(int atom,
                                 double slice_z,
                                 double ri_prime,
                                 double ri_prime2,
                                 double xi,
                                 double yi,
                                 const double *x,
                                 const double *y,
                                 const double *z,
                                 const double *radii,
                                 int nx,
                                 int ny,
                                 int nz,
                                 const int *atom_cells,
                                 const int *cell_offsets,
                                 const int *cell_atoms,
                                 const lr_prefilter *pf,
                                 int *buried,
                                 int *overflow)
{
    const double two_pi = 2.0 * FASTSASA_PI;
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    double arc_begin[FASTSASA_LR_ARC_CAPACITY];
    double arc_end[FASTSASA_LR_ARC_CAPACITY];
    double exposed;
    double covered_until;
    int n_arcs = 0;

    *buried = 0;
    *overflow = 0;
    for (int offset_i = 0; offset_i < 27; ++offset_i) {
        const int offset_base = 3 * offset_i;
        const int ix = cx + const_neighbor_cell_offsets[offset_base];
        const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
        const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
        int cell, first, last;

        if (ix < 0 || ix >= nx) continue;
        if (iy < 0 || iy >= ny) continue;
        if (iz < 0 || iz >= nz) continue;
        cell = ix + nx * (iy + ny * iz);
        first = cell_offsets[cell];
        last = cell_offsets[cell + 1];

        for (int n = first; n < last; ++n) {
            double a0, b0, a1, b1;
            const int other = cell_atoms[n];
            const int arcs = lr_neighbor_arcs(atom,
                                             other,
                                             slice_z,
                                             ri_prime,
                                             ri_prime2,
                                             xi,
                                             yi,
                                             x,
                                             y,
                                             z,
                                             radii,
                                             &a0,
                                             &b0,
                                             &a1,
                                             &b1,
                                             pf);
            if (arcs < 0) {
                *buried = 1;
                return 0.0;
            }
            if (arcs >= 1 && !lr_insert_arc_sorted(a0, b0, arc_begin, arc_end, &n_arcs)) {
                *overflow = 1;
                return 0.0;
            }
            if (arcs == 2 && !lr_insert_arc_sorted(a1, b1, arc_begin, arc_end, &n_arcs)) {
                *overflow = 1;
                return 0.0;
            }
        }
    }

    if (n_arcs == 0) return two_pi;
    exposed = arc_begin[0];
    covered_until = arc_end[0];
    for (int i = 1; i < n_arcs; ++i) {
        if (covered_until < arc_begin[i]) exposed = fastsasa_dadd(exposed, fastsasa_dsub(arc_begin[i], covered_until));
        if (arc_end[i] > covered_until) covered_until = arc_end[i];
    }
    return fastsasa_dsub(fastsasa_dadd(exposed, two_pi), covered_until);
}

/*
 * One thread per (atom, slice) writes that slice's contribution
 * (delta*ri)*exposed_arc, or 0 for skipped, buried and overflowing slices, to
 * slice_areas[(atom - atom_begin)*n_slices + slice]; lee_richards_slice_reduce_kernel
 * then sums each atom's slices in slice order, reproducing the CPU
 * reference's `area += delta * ri * exposed` loop bit for bit. Overflowing
 * atoms are flagged and recomputed on the host.
 */
__global__ static void
lee_richards_cell_slice_kernel(int n_atoms,
                               int n_slices,
                               int atom_begin,
                               int atom_end,
                               const double *x,
                               const double *y,
                               const double *z,
                               const double *radii,
                               const float *xf,
                               const float *yf,
                               const float *zf,
                               const float *rf,
                               double min_x,
                               double min_y,
                               double min_z,
                               double linear_margin,
                               double squared_margin,
                               int nx,
                               int ny,
                               int nz,
                               const int *atom_cells,
                               const int *cell_offsets,
                               const int *cell_atoms,
                               double *slice_areas,
                               int *overflow_flags)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (atom_end - atom_begin) * n_slices;
    int atom, slice, buried, overflow;
    double ri, xi, yi, zi, delta, slice_z, di, ri_prime2, ri_prime;
    double exposed_arc;

    (void)n_atoms;
    if (index >= total) return;
    atom = atom_begin + index / n_slices;
    slice = index - (atom - atom_begin) * n_slices;
    slice_areas[index] = 0.0;
    ri = radii[atom];
    if (ri <= 0.0) return;
    xi = x[atom];
    yi = y[atom];
    zi = z[atom];
    /* Same operation sequence as lee_richards_atom_area in fastsasa_cpu.cc. */
    delta = fastsasa_dmul(2.0, ri) / (double)n_slices;
    slice_z = fastsasa_dadd(fastsasa_dsub(zi, ri), fastsasa_dmul(delta, (double)slice + 0.5));
    di = fabs(fastsasa_dsub(zi, slice_z));
    ri_prime2 = fastsasa_dsub(fastsasa_dmul(ri, ri), fastsasa_dmul(di, di));
    if (ri_prime2 <= 0.0) return;
    ri_prime = sqrt(ri_prime2);

    lr_prefilter prefilter;
    const lr_prefilter *pf = NULL;
    if (xf != NULL) {
        prefilter.xf = xf;
        prefilter.yf = yf;
        prefilter.zf = zf;
        prefilter.rf = rf;
        prefilter.xif = (float)(xi - min_x);
        prefilter.yif = (float)(yi - min_y);
        prefilter.slice_zf = (float)(slice_z - min_z);
        prefilter.ri_prime_ubf = (float)ri_prime * (1.0f + 4.0f * 5.96e-8f) +
                                 1.0e-6f;
        prefilter.linear_margin = (float)linear_margin;
        prefilter.squared_margin = (float)squared_margin;
        pf = &prefilter;
    }

    exposed_arc = lr_exposed_arc_length_local_cell(atom,
                                                   slice_z,
                                                   ri_prime,
                                                   ri_prime2,
                                                   xi,
                                                   yi,
                                                   x,
                                                   y,
                                                   z,
                                                   radii,
                                                   nx,
                                                   ny,
                                                   nz,
                                                   atom_cells,
                                                   cell_offsets,
                                                   cell_atoms,
                                                   pf,
                                                   &buried,
                                                   &overflow);
    if (overflow) {
        overflow_flags[atom] = 1;
        return;
    }
    if (buried) return;
    slice_areas[index] = fastsasa_dmul(fastsasa_dmul(delta, ri), exposed_arc);
}

__global__ static void
lee_richards_slice_reduce_kernel(int n_slices,
                                 int atom_begin,
                                 int atom_end,
                                 const double *slice_areas,
                                 double *sasa)
{
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const double *row;
    double area = 0.0;

    if (local >= atom_end - atom_begin) return;
    row = slice_areas + (size_t)local * (size_t)n_slices;
    for (int slice = 0; slice < n_slices; ++slice) {
        area = fastsasa_dadd(area, row[slice]);
    }
    sasa[atom_begin + local] = area;
}

__device__ static int
lr_neighbor_arcs_float(int atom,
                       int other,
                       float slice_z,
                       float ri_prime,
                       float ri_prime2,
                       float xi,
                       float yi,
                       const float *x,
                       const float *y,
                       const float *z,
                       const float *radii,
                       float *a0,
                       float *b0,
                       float *a1,
                       float *b1)
{
    const float pi = (float)FASTSASA_PI;
    float rj, dj, rj_prime2, rj_prime, dx, dy, dij;
    float alpha_arg, alpha, beta, inf, sup;

    if (other == atom) return 0;
    rj = radii[other];
    dj = fabsf(z[other] - slice_z);
    if (dj >= rj) return 0;
    rj_prime2 = rj * rj - dj * dj;
    if (rj_prime2 <= 0.0f) return 0;
    rj_prime = sqrtf(rj_prime2);
    dx = x[other] - xi;
    dy = y[other] - yi;
    dij = sqrtf(fmaf(dx, dx, dy * dy));

    if (dij >= ri_prime + rj_prime) return 0;
    if (dij < 1e-6f) return rj_prime >= ri_prime ? -1 : 0;
    if (dij + ri_prime < rj_prime) return -1;
    if (dij + rj_prime < ri_prime) return 0;

    alpha_arg = fmaf(dij, dij, ri_prime2 - rj_prime2) / (2.0f * ri_prime * dij);
    if (alpha_arg < -1.0f) alpha_arg = -1.0f;
    if (alpha_arg > 1.0f) alpha_arg = 1.0f;
    alpha = acosf(alpha_arg);
    beta = atan2f(dy, dx) + pi;
    inf = beta - alpha;
    sup = beta + alpha;
    if (inf < 0.0f) inf += 2.0f * pi;
    if (sup > 2.0f * pi) sup -= 2.0f * pi;

    if (sup < inf) {
        *a0 = 0.0f;
        *b0 = sup;
        *a1 = inf;
        *b1 = 2.0f * pi;
        return 2;
    }
    *a0 = inf;
    *b0 = sup;
    return 1;
}

__device__ static int
lr_angle_covered_cell_float(float angle,
                            int atom,
                            float slice_z,
                            float ri_prime,
                            float ri_prime2,
                            float xi,
                            float yi,
                            const float *x,
                            const float *y,
                            const float *z,
                            const float *radii,
                            int nx,
                            int ny,
                            int nz,
                            const int *atom_cells,
                            const int *cell_offsets,
                            const int *cell_atoms)
{
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;

    for (int offset_i = 0; offset_i < 27; ++offset_i) {
        const int offset_base = 3 * offset_i;
        const int ix = cx + const_neighbor_cell_offsets[offset_base];
        const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
        const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
        int cell, first, last;

        if (ix < 0 || ix >= nx) continue;
        if (iy < 0 || iy >= ny) continue;
        if (iz < 0 || iz >= nz) continue;
        cell = ix + nx * (iy + ny * iz);
        first = cell_offsets[cell];
        last = cell_offsets[cell + 1];

        for (int n = first; n < last; ++n) {
            float a0, b0, a1, b1;
            const int other = cell_atoms[n];
            const int n_arcs = lr_neighbor_arcs_float(atom,
                                                      other,
                                                      slice_z,
                                                      ri_prime,
                                                      ri_prime2,
                                                      xi,
                                                      yi,
                                                      x,
                                                      y,
                                                      z,
                                                      radii,
                                                      &a0,
                                                      &b0,
                                                      &a1,
                                                      &b1);
            if (n_arcs < 0) return 1;
            if (n_arcs >= 1 && angle >= a0 && angle <= b0) return 1;
            if (n_arcs == 2 && angle >= a1 && angle <= b1) return 1;
        }
    }
    return 0;
}

__device__ static float
lr_exposed_arc_length_streaming_cell_float(int atom,
                                           float slice_z,
                                           float ri_prime,
                                           float ri_prime2,
                                           float xi,
                                           float yi,
                                           int n_atoms,
                                           const float *x,
                                           const float *y,
                                           const float *z,
                                           const float *radii,
                                           int nx,
                                           int ny,
                                           int nz,
                                           const int *atom_cells,
                                           const int *cell_offsets,
                                           const int *cell_atoms)
{
    const float two_pi = 2.0f * (float)FASTSASA_PI;
    const float eps = 1e-5f;
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    float current = 0.0f;
    float exposed = 0.0f;

    for (int guard = 0; guard < 4 * n_atoms + 4 && current < two_pi - eps; ++guard) {
        float next = two_pi;

        for (int offset_i = 0; offset_i < 27; ++offset_i) {
            const int offset_base = 3 * offset_i;
            const int ix = cx + const_neighbor_cell_offsets[offset_base];
            const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
            const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
            int cell, first, last;

            if (ix < 0 || ix >= nx) continue;
            if (iy < 0 || iy >= ny) continue;
            if (iz < 0 || iz >= nz) continue;
            cell = ix + nx * (iy + ny * iz);
            first = cell_offsets[cell];
            last = cell_offsets[cell + 1];

            for (int n = first; n < last; ++n) {
                float a0, b0, a1, b1;
                const int other = cell_atoms[n];
                const int n_arcs = lr_neighbor_arcs_float(atom,
                                                          other,
                                                          slice_z,
                                                          ri_prime,
                                                          ri_prime2,
                                                          xi,
                                                          yi,
                                                          x,
                                                          y,
                                                          z,
                                                          radii,
                                                          &a0,
                                                          &b0,
                                                          &a1,
                                                          &b1);
                if (n_arcs < 0) return 0.0f;
                if (n_arcs >= 1) {
                    if (a0 > current + eps && a0 < next) next = a0;
                    if (b0 > current + eps && b0 < next) next = b0;
                }
                if (n_arcs == 2) {
                    if (a1 > current + eps && a1 < next) next = a1;
                    if (b1 > current + eps && b1 < next) next = b1;
                }
            }
        }
        if (next <= current + eps) break;
        if (!lr_angle_covered_cell_float(0.5f * (current + next),
                                         atom,
                                         slice_z,
                                         ri_prime,
                                         ri_prime2,
                                         xi,
                                         yi,
                                         x,
                                         y,
                                         z,
                                         radii,
                                         nx,
                                         ny,
                                         nz,
                                         atom_cells,
                                         cell_offsets,
                                         cell_atoms)) {
            exposed += next - current;
        }
        current = next;
    }
    return exposed;
}

__device__ static int
lr_insert_arc_sorted_float(float begin,
                           float end,
                           float *arc_begin,
                           float *arc_end,
                           int *n_arcs)
{
    int pos;

    if (*n_arcs >= FASTSASA_LR_ARC_CAPACITY) return 0;
    pos = *n_arcs;
    while (pos > 0 && arc_begin[pos - 1] > begin) {
        arc_begin[pos] = arc_begin[pos - 1];
        arc_end[pos] = arc_end[pos - 1];
        --pos;
    }
    arc_begin[pos] = begin;
    arc_end[pos] = end;
    ++(*n_arcs);
    return 1;
}

__device__ static float
lr_exposed_arc_length_local_cell_float(int atom,
                                       float slice_z,
                                       float ri_prime,
                                       float ri_prime2,
                                       float xi,
                                       float yi,
                                       int n_atoms,
                                       const float *x,
                                       const float *y,
                                       const float *z,
                                       const float *radii,
                                       int nx,
                                       int ny,
                                       int nz,
                                       const int *atom_cells,
                                       const int *cell_offsets,
                                       const int *cell_atoms)
{
    const float two_pi = 2.0f * (float)FASTSASA_PI;
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    float arc_begin[FASTSASA_LR_ARC_CAPACITY];
    float arc_end[FASTSASA_LR_ARC_CAPACITY];
    float exposed;
    float covered_until;
    int n_arcs = 0;

    for (int offset_i = 0; offset_i < 27; ++offset_i) {
        const int offset_base = 3 * offset_i;
        const int ix = cx + const_neighbor_cell_offsets[offset_base];
        const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
        const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
        int cell, first, last;

        if (ix < 0 || ix >= nx) continue;
        if (iy < 0 || iy >= ny) continue;
        if (iz < 0 || iz >= nz) continue;
        cell = ix + nx * (iy + ny * iz);
        first = cell_offsets[cell];
        last = cell_offsets[cell + 1];

        for (int n = first; n < last; ++n) {
            float a0, b0, a1, b1;
            const int other = cell_atoms[n];
            const int arcs = lr_neighbor_arcs_float(atom,
                                                   other,
                                                   slice_z,
                                                   ri_prime,
                                                   ri_prime2,
                                                   xi,
                                                   yi,
                                                   x,
                                                   y,
                                                   z,
                                                   radii,
                                                   &a0,
                                                   &b0,
                                                   &a1,
                                                   &b1);
            if (arcs < 0) return 0.0f;
            if (arcs >= 1 && !lr_insert_arc_sorted_float(a0, b0, arc_begin, arc_end, &n_arcs)) {
                return lr_exposed_arc_length_streaming_cell_float(atom,
                                                                 slice_z,
                                                                 ri_prime,
                                                                 ri_prime2,
                                                                 xi,
                                                                 yi,
                                                                 n_atoms,
                                                                 x,
                                                                 y,
                                                                 z,
                                                                 radii,
                                                                 nx,
                                                                 ny,
                                                                 nz,
                                                                 atom_cells,
                                                                 cell_offsets,
                                                                 cell_atoms);
            }
            if (arcs == 2 && !lr_insert_arc_sorted_float(a1, b1, arc_begin, arc_end, &n_arcs)) {
                return lr_exposed_arc_length_streaming_cell_float(atom,
                                                                 slice_z,
                                                                 ri_prime,
                                                                 ri_prime2,
                                                                 xi,
                                                                 yi,
                                                                 n_atoms,
                                                                 x,
                                                                 y,
                                                                 z,
                                                                 radii,
                                                                 nx,
                                                                 ny,
                                                                 nz,
                                                                 atom_cells,
                                                                 cell_offsets,
                                                                 cell_atoms);
            }
        }
    }

    if (n_arcs == 0) return two_pi;
    exposed = arc_begin[0];
    covered_until = arc_end[0];
    for (int i = 1; i < n_arcs; ++i) {
        if (covered_until < arc_begin[i]) exposed += arc_begin[i] - covered_until;
        if (arc_end[i] > covered_until) covered_until = arc_end[i];
    }
    exposed += two_pi - covered_until;
    if (exposed < 0.0f) exposed = 0.0f;
    if (exposed > two_pi) exposed = two_pi;
    return exposed;
}

__global__ static void
lee_richards_cell_slice_float_kernel(int n_atoms,
                                     int n_slices,
                                     const float *x,
                                     const float *y,
                                     const float *z,
                                     const float *radii,
                                     int nx,
                                     int ny,
                                     int nz,
                                     const int *atom_cells,
                                     const int *cell_offsets,
                                     const int *cell_atoms,
                                     double *sasa)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_atoms * n_slices;
    int atom, slice;
    float ri, xi, yi, zi, delta, slice_z, di, ri_prime2, ri_prime;
    float exposed_arc;

    if (index >= total) return;
    atom = index / n_slices;
    slice = index - atom * n_slices;
    ri = __ldg(&radii[atom]);
    if (ri <= 0.0f) return;
    xi = __ldg(&x[atom]);
    yi = __ldg(&y[atom]);
    zi = __ldg(&z[atom]);
    delta = 2.0f * ri / (float)n_slices;
    slice_z = zi - ri + delta * ((float)slice + 0.5f);
    di = fabsf(zi - slice_z);
    ri_prime2 = ri * ri - di * di;
    if (ri_prime2 <= 0.0f) return;
    ri_prime = sqrtf(ri_prime2);
    if (ri_prime <= 0.0f) return;

    exposed_arc = lr_exposed_arc_length_local_cell_float(atom,
                                                         slice_z,
                                                         ri_prime,
                                                         ri_prime2,
                                                         xi,
                                                         yi,
                                                         n_atoms,
                                                         x,
                                                         y,
                                                         z,
                                                         radii,
                                                         nx,
                                                         ny,
                                                         nz,
                                                         atom_cells,
                                                         cell_offsets,
                                                         cell_atoms);
    if (exposed_arc > 0.0f) {
        const double atom_slice_area = (double)delta * (double)ri * (double)exposed_arc;
        atomic_add_double(&sasa[atom], atom_slice_area);
    }
}

__global__ static void
lee_richards_cell_slice_float_accum_kernel(int n_atoms,
                                           int n_slices,
                                           const float *x,
                                           const float *y,
                                           const float *z,
                                           const float *radii,
                                           int nx,
                                           int ny,
                                           int nz,
                                           const int *atom_cells,
                                           const int *cell_offsets,
                                           const int *cell_atoms,
                                           float *sasa)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_atoms * n_slices;
    int atom, slice;
    float ri, xi, yi, zi, delta, slice_z, di, ri_prime2, ri_prime;
    float exposed_arc;

    if (index >= total) return;
    atom = index / n_slices;
    slice = index - atom * n_slices;
    ri = __ldg(&radii[atom]);
    if (ri <= 0.0f) return;
    xi = __ldg(&x[atom]);
    yi = __ldg(&y[atom]);
    zi = __ldg(&z[atom]);
    delta = 2.0f * ri / (float)n_slices;
    slice_z = zi - ri + delta * ((float)slice + 0.5f);
    di = fabsf(zi - slice_z);
    ri_prime2 = ri * ri - di * di;
    if (ri_prime2 <= 0.0f) return;
    ri_prime = sqrtf(ri_prime2);
    if (ri_prime <= 0.0f) return;

    exposed_arc = lr_exposed_arc_length_local_cell_float(atom,
                                                         slice_z,
                                                         ri_prime,
                                                         ri_prime2,
                                                         xi,
                                                         yi,
                                                         n_atoms,
                                                         x,
                                                         y,
                                                         z,
                                                         radii,
                                                         nx,
                                                         ny,
                                                         nz,
                                                         atom_cells,
                                                         cell_offsets,
                                                         cell_atoms);
    if (exposed_arc > 0.0f) {
        atomicAdd(&sasa[atom], delta * ri * exposed_arc);
    }
}

__device__ static int
cell_index_from_xyz(double x,
                    double y,
                    double z,
                    double min_x,
                    double min_y,
                    double min_z,
                    double cell_size,
                    int nx,
                    int ny,
                    int nz)
{
    int ix = (int)floor((x - min_x) / cell_size);
    int iy = (int)floor((y - min_y) / cell_size);
    int iz = (int)floor((z - min_z) / cell_size);

    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;

    return ix + nx * (iy + ny * iz);
}

__global__ static void
split_xyz_kernel(int n_atoms,
                 const double *xyz,
                 double *x,
                 double *y,
                 double *z)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom >= n_atoms) return;
    x[atom] = xyz[3 * atom];
    y[atom] = xyz[3 * atom + 1];
    z[atom] = xyz[3 * atom + 2];
}

__global__ static void
cell_count_kernel(int n_atoms,
                  const double *x,
                  const double *y,
                  const double *z,
                  double min_x,
                  double min_y,
                  double min_z,
                  double cell_size,
                  int nx,
                  int ny,
                  int nz,
                  int *atom_cells,
                  int *cell_counts)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    int cell;

    if (atom >= n_atoms) return;
    cell = cell_index_from_xyz(x[atom],
                               y[atom],
                               z[atom],
                               min_x,
                               min_y,
                               min_z,
                               cell_size,
                               nx,
                               ny,
                               nz);
    atom_cells[atom] = cell;
    atomicAdd(&cell_counts[cell], 1);
}

__global__ static void
sequence_kernel(int n,
                int *values)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) values[i] = i;
}

__global__ static void
double_to_float_kernel(int n,
                       const double *input,
                       float *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) output[i] = (float)input[i];
}

__global__ static void
double_to_shifted_float_kernel(int n,
                               const double *input,
                               double origin,
                               float *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) output[i] = (float)(input[i] - origin);
}

__global__ static void
float_to_double_kernel(int n,
                       const float *input,
                       double *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) output[i] = (double)input[i];
}

__global__ static void
square_float_kernel(int n,
                    const float *input,
                    float *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) output[i] = input[i] * input[i];
}

__global__ static void
cell_fill_kernel(int n_atoms,
                 const int *atom_cells,
                 int *cell_fill,
                 int *cell_atoms)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    int cell;
    int slot;

    if (atom >= n_atoms) return;
    cell = atom_cells[atom];
    slot = atomicAdd(&cell_fill[cell], 1);
    cell_atoms[slot] = atom;
}

__global__ static void
shrake_rupley_cell_kernel(int n_atoms,
                          int n_points,
                          const double *x,
                          const double *y,
                          const double *z,
                          const double *radii,
                          const double *radii2,
                          const double *test_points,
                          int nx,
                          int ny,
                          int nz,
                          const int *atom_cells,
                          const int *cell_offsets,
                          const int *cell_atoms,
                          const int *center_indices,
                          int n_centers,
                          const unsigned int *center_masks,
                          unsigned int active_center_mask,
                          double *sasa,
                          double *total_sasa)
{
    extern __shared__ int exposed_counts[];

    const int center = blockIdx.x;
    if (center_indices != NULL && center >= n_centers) return;
    const int atom = center_indices != NULL ? center_indices[center] : center;
    const int tid = threadIdx.x;

    if (active_center_mask != 0u &&
        ((center_masks[atom] & active_center_mask) == 0u)) {
        if (tid == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const double ri = radii[atom];
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const double px = fastsasa_dadd(xi, fastsasa_dmul(ri, test_points[3 * point]));
        const double py = fastsasa_dadd(yi, fastsasa_dmul(ri, test_points[3 * point + 1]));
        const double pz = fastsasa_dadd(zi, fastsasa_dmul(ri, test_points[3 * point + 2]));
        int buried = 0;

        for (int dz = -1; dz <= 1 && !buried; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1 && !buried; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1 && !buried; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = cell_offsets[cell];
                    last = cell_offsets[cell + 1];

                    for (int n = first; n < last; ++n) {
                        const int other = cell_atoms[n];
                        const double dx = px - x[other];
                        const double dy_atom = py - y[other];
                        const double dz_atom = pz - z[other];

                        if (other == atom) continue;
                        if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy_atom, dy_atom)), fastsasa_dmul(dz_atom, dz_atom)) < radii2[other]) {
                            buried = 1;
                            break;
                        }
                    }
                }
            }
        }

        if (!buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_counts[0], n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

/*
 * FP64-exact Shrake-Rupley with a single-precision prefilter. Consumer GPUs
 * execute FP64 at a small fraction of the FP32 rate, so each point/neighbor
 * test first runs in FP32 on box-local coordinates against a conservative
 * uncertainty margin; only tests that land inside the margin are re-evaluated
 * with the exact FP64 expression of shrake_rupley_cell_kernel. Decisions made
 * outside the margin provably match the FP64 comparison, so exposure counts,
 * and therefore all outputs, are bit-identical to the pure FP64 kernel.
 */
__global__ static void
shrake_rupley_cell_hybrid_kernel(int n_atoms,
                                 int n_points,
                                 const double *x,
                                 const double *y,
                                 const double *z,
                                 const double *radii,
                                 const double *radii2,
                                 const double *test_points,
                                 const float *__restrict__ xf,
                                 const float *__restrict__ yf,
                                 const float *__restrict__ zf,
                                 const float *__restrict__ radii2f,
                                 double min_x,
                                 double min_y,
                                 double min_z,
                                 float margin,
                                 int nx,
                                 int ny,
                                 int nz,
                                 const int *atom_cells,
                                 const int *cell_offsets,
                                 const int *cell_atoms,
                                 const int *center_indices,
                                 int n_centers,
                                 const unsigned int *center_masks,
                                 unsigned int active_center_mask,
                                 double *sasa,
                                 double *total_sasa)
{
    extern __shared__ int exposed_counts[];

    const int center = blockIdx.x;
    if (center_indices != NULL && center >= n_centers) return;
    const int atom = center_indices != NULL ? center_indices[center] : center;
    const int tid = threadIdx.x;

    if (active_center_mask != 0u &&
        ((center_masks[atom] & active_center_mask) == 0u)) {
        if (tid == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const double ri = radii[atom];
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const double px = fastsasa_dadd(xi, fastsasa_dmul(ri, test_points[3 * point]));
        const double py = fastsasa_dadd(yi, fastsasa_dmul(ri, test_points[3 * point + 1]));
        const double pz = fastsasa_dadd(zi, fastsasa_dmul(ri, test_points[3 * point + 2]));
        const float pxf = (float)(px - min_x);
        const float pyf = (float)(py - min_y);
        const float pzf = (float)(pz - min_z);
        int buried = 0;

        for (int dz = -1; dz <= 1 && !buried; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1 && !buried; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1 && !buried; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = cell_offsets[cell];
                    last = cell_offsets[cell + 1];

                    for (int n = first; n < last; ++n) {
                        const int other = cell_atoms[n];

                        if (other == atom) continue;
                        {
                            const float dxf = pxf - xf[other];
                            const float dyf = pyf - yf[other];
                            const float dzf = pzf - zf[other];
                            const float diff = dxf * dxf + dyf * dyf +
                                               dzf * dzf - radii2f[other];

                            if (diff < -margin) {
                                buried = 1;
                                break;
                            }
                            if (diff > margin) continue;
                        }
                        {
                            const double dx = px - x[other];
                            const double dy_atom = py - y[other];
                            const double dz_atom = pz - z[other];

                            if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy_atom, dy_atom)),
                                     fastsasa_dmul(dz_atom, dz_atom)) < radii2[other]) {
                                buried = 1;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (!buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_counts[0], n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_float_ordered_kernel(int n_atoms,
                                        int n_points,
                                        const float *__restrict__ x,
                                        const float *__restrict__ y,
                                        const float *__restrict__ z,
                                        const float *__restrict__ radii,
                                        const float *__restrict__ radii2,
                                        const float *__restrict__ test_points,
                                        int nx,
                                        int ny,
                                        int nz,
                                        const int *__restrict__ atom_cells,
                                        const int *__restrict__ cell_offsets,
                                        const int *__restrict__ cell_atoms,
                                        const int *__restrict__ center_indices,
                                        int n_centers,
                                        const unsigned int *__restrict__ center_masks,
                                        unsigned int active_center_mask,
                                        double *__restrict__ sasa,
                                        double *__restrict__ total_sasa)
{
    extern __shared__ int exposed_counts[];

    const int center = blockIdx.x;
    if (center >= n_centers) return;
    const int atom = center_indices != NULL ? __ldg(&center_indices[center]) : center;
    const int tid = threadIdx.x;

    if (active_center_mask != 0u &&
        ((__ldg(&center_masks[atom]) & active_center_mask) == 0u)) {
        if (tid == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const float ri = __ldg(&radii[atom]);
    const float xi = __ldg(&x[atom]);
    const float yi = __ldg(&y[atom]);
    const float zi = __ldg(&z[atom]);
    const int center_cell = __ldg(&atom_cells[atom]);
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const int point_offset = 3 * point;
        const float px = xi + __ldg(&test_points[point_offset]) * ri;
        const float py = yi + __ldg(&test_points[point_offset + 1]) * ri;
        const float pz = zi + __ldg(&test_points[point_offset + 2]) * ri;
        int buried = 0;

        for (int offset_i = 0; offset_i < 27 && !buried; ++offset_i) {
            const int offset_base = 3 * offset_i;
            const int ix = cx + const_neighbor_cell_offsets[offset_base];
            const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
            const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
            int cell;
            int first;
            int last;

            if (ix < 0 || ix >= nx) continue;
            if (iy < 0 || iy >= ny) continue;
            if (iz < 0 || iz >= nz) continue;
            cell = ix + nx * (iy + ny * iz);
            first = __ldg(&cell_offsets[cell]);
            last = __ldg(&cell_offsets[cell + 1]);

            for (int n = first; n < last; ++n) {
                const int other = __ldg(&cell_atoms[n]);
                float dx, dy_atom, dz_atom;

                if (other == atom) continue;
                dx = px - __ldg(&x[other]);
                dy_atom = py - __ldg(&y[other]);
                dz_atom = pz - __ldg(&z[other]);
                if (dx * dx + dy_atom * dy_atom + dz_atom * dz_atom < __ldg(&radii2[other])) {
                    buried = 1;
                    break;
                }
            }
        }

        if (!buried) ++exposed;
    }

    const int exposed_total = block_reduce_sum_int(exposed, exposed_counts);

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_total, n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_float_warp_ordered_kernel(int n_atoms,
                                             int n_points,
                                             const float *__restrict__ x,
                                             const float *__restrict__ y,
                                             const float *__restrict__ z,
                                             const float *__restrict__ radii,
                                             const float *__restrict__ radii2,
                                             const float *__restrict__ test_points,
                                             int nx,
                                             int ny,
                                             int nz,
                                             const int *__restrict__ atom_cells,
                                             const int *__restrict__ cell_offsets,
                                             const int *__restrict__ cell_atoms,
                                             const int *__restrict__ center_indices,
                                             int n_centers,
                                             const unsigned int *__restrict__ center_masks,
                                             unsigned int active_center_mask,
                                             double *__restrict__ sasa,
                                             double *__restrict__ total_sasa)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int center = blockIdx.x * warps_per_block + warp_in_block;
    if (center >= n_centers) return;
    const int atom = center_indices != NULL ? __ldg(&center_indices[center]) : center;
    int exposed = 0;

    if (active_center_mask != 0u &&
        ((__ldg(&center_masks[atom]) & active_center_mask) == 0u)) {
        if (lane == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const float ri = __ldg(&radii[atom]);
    const float xi = __ldg(&x[atom]);
    const float yi = __ldg(&y[atom]);
    const float zi = __ldg(&z[atom]);
    const int center_cell = __ldg(&atom_cells[atom]);
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;

    for (int point = lane; point < n_points; point += 32) {
        const int point_offset = 3 * point;
        const float px = xi + __ldg(&test_points[point_offset]) * ri;
        const float py = yi + __ldg(&test_points[point_offset + 1]) * ri;
        const float pz = zi + __ldg(&test_points[point_offset + 2]) * ri;
        int buried = 0;

        for (int offset_i = 0; offset_i < 27 && !buried; ++offset_i) {
            const int offset_base = 3 * offset_i;
            const int ix = cx + const_neighbor_cell_offsets[offset_base];
            const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
            const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
            int cell;
            int first;
            int last;

            if (ix < 0 || ix >= nx) continue;
            if (iy < 0 || iy >= ny) continue;
            if (iz < 0 || iz >= nz) continue;
            cell = ix + nx * (iy + ny * iz);
            first = __ldg(&cell_offsets[cell]);
            last = __ldg(&cell_offsets[cell + 1]);

            for (int n = first; n < last; ++n) {
                const int other = __ldg(&cell_atoms[n]);
                float dx, dy_atom, dz_atom;

                if (other == atom) continue;
                dx = px - __ldg(&x[other]);
                dy_atom = py - __ldg(&y[other]);
                dz_atom = pz - __ldg(&z[other]);
                if (dx * dx + dy_atom * dy_atom + dz_atom * dz_atom < __ldg(&radii2[other])) {
                    buried = 1;
                    break;
                }
            }
        }

        if (!buried) ++exposed;
    }

    exposed = warp_reduce_sum_int(exposed);
    if (lane == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed, n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_float_kernel(int n_atoms,
                                int n_points,
                                const float *__restrict__ x,
                                const float *__restrict__ y,
                                const float *__restrict__ z,
                                const float *__restrict__ radii,
                                const float *__restrict__ radii2,
                                const float *__restrict__ test_points,
                                int nx,
                                int ny,
                                int nz,
                                const int *__restrict__ atom_cells,
                                const int *__restrict__ cell_offsets,
                                const int *__restrict__ cell_atoms,
                                const int *__restrict__ center_indices,
                                int n_centers,
                                const unsigned int *__restrict__ center_masks,
                                unsigned int active_center_mask,
                                double *__restrict__ sasa,
                                double *__restrict__ total_sasa)
{
    extern __shared__ int exposed_counts[];

    const int center = blockIdx.x;
    if (center >= n_centers) return;
    const int atom = center_indices != NULL ? __ldg(&center_indices[center]) : center;
    const int tid = threadIdx.x;

    if (active_center_mask != 0u &&
        ((__ldg(&center_masks[atom]) & active_center_mask) == 0u)) {
        if (tid == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const float ri = __ldg(&radii[atom]);
    const float xi = __ldg(&x[atom]);
    const float yi = __ldg(&y[atom]);
    const float zi = __ldg(&z[atom]);
    const int center_cell = __ldg(&atom_cells[atom]);
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const int point_offset = 3 * point;
        const float px = xi + __ldg(&test_points[point_offset]) * ri;
        const float py = yi + __ldg(&test_points[point_offset + 1]) * ri;
        const float pz = zi + __ldg(&test_points[point_offset + 2]) * ri;
        int buried = 0;

        for (int dz = -1; dz <= 1 && !buried; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1 && !buried; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1 && !buried; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = __ldg(&cell_offsets[cell]);
                    last = __ldg(&cell_offsets[cell + 1]);

                    for (int n = first; n < last; ++n) {
                        const int other = __ldg(&cell_atoms[n]);
                        float dx, dy_atom, dz_atom;

                        if (other == atom) continue;
                        dx = px - __ldg(&x[other]);
                        dy_atom = py - __ldg(&y[other]);
                        dz_atom = pz - __ldg(&z[other]);
                        if (dx * dx + dy_atom * dy_atom + dz_atom * dz_atom < __ldg(&radii2[other])) {
                            buried = 1;
                            break;
                        }
                    }
                }
            }
        }

        if (!buried) ++exposed;
    }

    const int exposed_total = block_reduce_sum_int(exposed, exposed_counts);

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_total, n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_float_compact_kernel(int n_atoms,
                                        int n_points,
                                        const float *__restrict__ x,
                                        const float *__restrict__ y,
                                        const float *__restrict__ z,
                                        const float *__restrict__ radii,
                                        const float *__restrict__ radii2,
                                        const float *__restrict__ test_points,
                                        int nx,
                                        int ny,
                                        int nz,
                                        const int *__restrict__ atom_cells,
                                        const int *__restrict__ cell_offsets,
                                        const int *__restrict__ cell_atoms,
                                        const int *__restrict__ center_indices,
                                        int n_centers,
                                        const unsigned int *__restrict__ center_masks,
                                        unsigned int active_center_mask,
                                        double *__restrict__ sasa,
                                        double *__restrict__ total_sasa)
{
    extern __shared__ int shared_points[];

    int *active_points = shared_points;
    int *next_points = shared_points + n_points;
    int *next_count_ptr = shared_points + 2 * n_points;
    const int center = blockIdx.x;
    if (center >= n_centers) return;
    const int atom = center_indices != NULL ? __ldg(&center_indices[center]) : center;
    const int tid = threadIdx.x;

    if (active_center_mask != 0u &&
        ((__ldg(&center_masks[atom]) & active_center_mask) == 0u)) {
        if (tid == 0 && sasa != NULL) sasa[atom] = 0.0;
        return;
    }

    const float ri = __ldg(&radii[atom]);
    const float xi = __ldg(&x[atom]);
    const float yi = __ldg(&y[atom]);
    const float zi = __ldg(&z[atom]);
    const int center_cell = __ldg(&atom_cells[atom]);
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int active_count = n_points;

    for (int point = tid; point < n_points; point += blockDim.x) {
        active_points[point] = point;
    }
    __syncthreads();

    for (int offset_i = 0; offset_i < 27; ++offset_i) {
        const int offset_base = 3 * offset_i;
        const int ix = cx + const_neighbor_cell_offsets[offset_base];
        const int iy = cy + const_neighbor_cell_offsets[offset_base + 1];
        const int iz = cz + const_neighbor_cell_offsets[offset_base + 2];
        int cell;
        int first;
        int last;
        int *tmp_points;

        if (ix < 0 || ix >= nx) continue;
        if (iy < 0 || iy >= ny) continue;
        if (iz < 0 || iz >= nz) continue;
        if (active_count == 0) break;

        cell = ix + nx * (iy + ny * iz);
        first = __ldg(&cell_offsets[cell]);
        last = __ldg(&cell_offsets[cell + 1]);

        if (tid == 0) *next_count_ptr = 0;
        __syncthreads();

        for (int active_index = tid; active_index < active_count; active_index += blockDim.x) {
            const int point = active_points[active_index];
            const int point_offset = 3 * point;
            const float px = xi + __ldg(&test_points[point_offset]) * ri;
            const float py = yi + __ldg(&test_points[point_offset + 1]) * ri;
            const float pz = zi + __ldg(&test_points[point_offset + 2]) * ri;
            int buried = 0;

            for (int n = first; n < last; ++n) {
                const int other = __ldg(&cell_atoms[n]);
                float dx, dy_atom, dz_atom;

                if (other == atom) continue;
                dx = px - __ldg(&x[other]);
                dy_atom = py - __ldg(&y[other]);
                dz_atom = pz - __ldg(&z[other]);
                if (dx * dx + dy_atom * dy_atom + dz_atom * dz_atom < __ldg(&radii2[other])) {
                    buried = 1;
                    break;
                }
            }

            if (!buried) {
                const int out_index = atomicAdd(next_count_ptr, 1);
                next_points[out_index] = point;
            }
        }
        __syncthreads();

        active_count = *next_count_ptr;
        tmp_points = active_points;
        active_points = next_points;
        next_points = tmp_points;
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, active_count, n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_shared_kernel(int n_atoms,
                                 int n_points,
                                 const double *x,
                                 const double *y,
                                 const double *z,
                                 const double *radii,
                                 const double *radii2,
                                 const double *test_points,
                                 int nx,
                                 int ny,
                                 int nz,
                                 const int *atom_cells,
                                 const int *cell_offsets,
                                 const int *cell_atoms,
                                 double *sasa,
                                 double *total_sasa)
{
    extern __shared__ unsigned char shared_raw[];
    double *shared_x = (double *)shared_raw;
    double *shared_y = shared_x + FASTSASA_NEIGHBOR_CHUNK;
    double *shared_z = shared_y + FASTSASA_NEIGHBOR_CHUNK;
    double *shared_radii2 = shared_z + FASTSASA_NEIGHBOR_CHUNK;
    int *exposed_counts = (int *)(shared_radii2 + FASTSASA_NEIGHBOR_CHUNK);
    int *shared_atoms = exposed_counts + blockDim.x;

    const int atom = blockIdx.x;
    const int tid = threadIdx.x;
    const double ri = radii[atom];
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point_base = 0; point_base < n_points; point_base += blockDim.x) {
        const int point = point_base + tid;
        const int active_point = point < n_points;
        const double px = active_point ? fastsasa_dadd(xi, fastsasa_dmul(ri, test_points[3 * point])) : 0.0;
        const double py = active_point ? fastsasa_dadd(yi, fastsasa_dmul(ri, test_points[3 * point + 1])) : 0.0;
        const double pz = active_point ? fastsasa_dadd(zi, fastsasa_dmul(ri, test_points[3 * point + 2])) : 0.0;
        int buried = 0;

        for (int dz = -1; dz <= 1; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = cell_offsets[cell];
                    last = cell_offsets[cell + 1];

                    for (int base = first; base < last; base += FASTSASA_NEIGHBOR_CHUNK) {
                        const int chunk_count = min(FASTSASA_NEIGHBOR_CHUNK, last - base);

                        for (int load = tid; load < chunk_count; load += blockDim.x) {
                            const int other = cell_atoms[base + load];

                            shared_atoms[load] = other;
                            shared_x[load] = x[other];
                            shared_y[load] = y[other];
                            shared_z[load] = z[other];
                            shared_radii2[load] = radii2[other];
                        }
                        __syncthreads();

                        if (active_point && !buried) {
                            for (int n = 0; n < chunk_count; ++n) {
                                const int other = shared_atoms[n];
                                const double dx = px - shared_x[n];
                                const double dy_atom = py - shared_y[n];
                                const double dz_atom = pz - shared_z[n];

                                if (other == atom) continue;
                                if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy_atom, dy_atom)), fastsasa_dmul(dz_atom, dz_atom)) < shared_radii2[n]) {
                                    buried = 1;
                                    break;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
        }

        if (active_point && !buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_counts[0], n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_const_points_kernel(int n_atoms,
                                       int n_points,
                                       const double *x,
                                       const double *y,
                                       const double *z,
                                       const double *radii,
                                       const double *radii2,
                                       int nx,
                                       int ny,
                                       int nz,
                                       const int *atom_cells,
                                       const int *cell_offsets,
                                       const int *cell_atoms,
                                       double *sasa,
                                       double *total_sasa)
{
    extern __shared__ int exposed_counts[];

    const int atom = blockIdx.x;
    const int tid = threadIdx.x;
    const double ri = radii[atom];
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point = tid; point < n_points; point += blockDim.x) {
        const double px = fastsasa_dadd(xi, fastsasa_dmul(ri, const_test_points[3 * point]));
        const double py = fastsasa_dadd(yi, fastsasa_dmul(ri, const_test_points[3 * point + 1]));
        const double pz = fastsasa_dadd(zi, fastsasa_dmul(ri, const_test_points[3 * point + 2]));
        int buried = 0;

        for (int dz = -1; dz <= 1 && !buried; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1 && !buried; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1 && !buried; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = cell_offsets[cell];
                    last = cell_offsets[cell + 1];

                    for (int n = first; n < last; ++n) {
                        const int other = cell_atoms[n];
                        const double dx = px - x[other];
                        const double dy_atom = py - y[other];
                        const double dz_atom = pz - z[other];

                        if (other == atom) continue;
                        if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy_atom, dy_atom)), fastsasa_dmul(dz_atom, dz_atom)) < radii2[other]) {
                            buried = 1;
                            break;
                        }
                    }
                }
            }
        }

        if (!buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_counts[0], n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
shrake_rupley_cell_const_points_shared_kernel(int n_atoms,
                                              int n_points,
                                              const double *x,
                                              const double *y,
                                              const double *z,
                                              const double *radii,
                                              const double *radii2,
                                              int nx,
                                              int ny,
                                              int nz,
                                              const int *atom_cells,
                                              const int *cell_offsets,
                                              const int *cell_atoms,
                                              double *sasa,
                                              double *total_sasa)
{
    extern __shared__ unsigned char shared_raw[];
    double *shared_x = (double *)shared_raw;
    double *shared_y = shared_x + FASTSASA_NEIGHBOR_CHUNK;
    double *shared_z = shared_y + FASTSASA_NEIGHBOR_CHUNK;
    double *shared_radii2 = shared_z + FASTSASA_NEIGHBOR_CHUNK;
    int *exposed_counts = (int *)(shared_radii2 + FASTSASA_NEIGHBOR_CHUNK);
    int *shared_atoms = exposed_counts + blockDim.x;

    const int atom = blockIdx.x;
    const int tid = threadIdx.x;
    const double ri = radii[atom];
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const int center_cell = atom_cells[atom];
    const int cz = center_cell / (nx * ny);
    const int rem = center_cell - cz * nx * ny;
    const int cy = rem / nx;
    const int cx = rem - cy * nx;
    int exposed = 0;

    for (int point_base = 0; point_base < n_points; point_base += blockDim.x) {
        const int point = point_base + tid;
        const int active_point = point < n_points;
        const double px = active_point ? fastsasa_dadd(xi, fastsasa_dmul(ri, const_test_points[3 * point])) : 0.0;
        const double py = active_point ? fastsasa_dadd(yi, fastsasa_dmul(ri, const_test_points[3 * point + 1])) : 0.0;
        const double pz = active_point ? fastsasa_dadd(zi, fastsasa_dmul(ri, const_test_points[3 * point + 2])) : 0.0;
        int buried = 0;

        for (int dz = -1; dz <= 1; ++dz) {
            const int iz = cz + dz;
            if (iz < 0 || iz >= nz) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                const int iy = cy + dy;
                if (iy < 0 || iy >= ny) continue;
                for (int dx_cell = -1; dx_cell <= 1; ++dx_cell) {
                    const int ix = cx + dx_cell;
                    int cell;
                    int first;
                    int last;

                    if (ix < 0 || ix >= nx) continue;
                    cell = ix + nx * (iy + ny * iz);
                    first = cell_offsets[cell];
                    last = cell_offsets[cell + 1];

                    for (int base = first; base < last; base += FASTSASA_NEIGHBOR_CHUNK) {
                        const int chunk_count = min(FASTSASA_NEIGHBOR_CHUNK, last - base);

                        for (int load = tid; load < chunk_count; load += blockDim.x) {
                            const int other = cell_atoms[base + load];

                            shared_atoms[load] = other;
                            shared_x[load] = x[other];
                            shared_y[load] = y[other];
                            shared_z[load] = z[other];
                            shared_radii2[load] = radii2[other];
                        }
                        __syncthreads();

                        if (active_point && !buried) {
                            for (int n = 0; n < chunk_count; ++n) {
                                const int other = shared_atoms[n];
                                const double dx = px - shared_x[n];
                                const double dy_atom = py - shared_y[n];
                                const double dz_atom = pz - shared_z[n];

                                if (other == atom) continue;
                                if (fastsasa_dadd(fastsasa_dadd(fastsasa_dmul(dx, dx), fastsasa_dmul(dy_atom, dy_atom)), fastsasa_dmul(dz_atom, dz_atom)) < shared_radii2[n]) {
                                    buried = 1;
                                    break;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
        }

        if (active_point && !buried) ++exposed;
    }

    exposed_counts[tid] = exposed;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            exposed_counts[tid] += exposed_counts[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const double atom_sasa = sr_atom_area(ri, exposed_counts[0], n_points);
        if (sasa != NULL) sasa[atom] = atom_sasa;
        if (total_sasa != NULL) atomic_add_double(total_sasa, atom_sasa);
    }
}

__global__ static void
square_kernel(int n,
              const double *input,
              double *output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * input[i];
}

int
fastsasa_device_context_shrake_rupley_csr(fastsasa_device_context *context,
                                       const fastsasa_device_sr_input *input,
                                       double *sasa)
{
    int status = validate_input(input, sasa);
    if (status != FASTSASA_SUCCESS) return status;
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;

    const size_t xyz_bytes = sizeof(double) * 3u * (size_t)input->n_atoms;
    const size_t radii_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t test_point_bytes = sizeof(double) * 3u * (size_t)input->n_points;
    const size_t offset_bytes = sizeof(int) * ((size_t)input->n_atoms + 1u);
    const size_t index_bytes = sizeof(int) * (size_t)input->n_neighbor_indices;
    const size_t sasa_bytes = sizeof(double) * (size_t)input->n_atoms;
    const int aggregate_residues = input->residue_ids != NULL;
    /* Constant memory is process-global and unsafe across concurrent contexts. */
    const int use_const_test_points = 0;

    const int prep_threads = 256;
    const int prep_blocks = (input->n_atoms + prep_threads - 1) / prep_threads;
    const int sr_threads = sr_block_threads(input->n_points);

    status = validate_launch_size(input->n_atoms, sr_threads);
    if (status != FASTSASA_SUCCESS) return status;
    status = validate_shared_memory_size(sizeof(int) * (size_t)sr_threads);
    if (status != FASTSASA_SUCCESS) return status;

    status = ensure_context_capacity(context,
                                     xyz_bytes,
                                     radii_bytes,
                                     use_const_test_points ? 0 : test_point_bytes,
                                     offset_bytes,
                                     index_bytes,
                                     sasa_bytes);
    if (status != FASTSASA_SUCCESS) return status;

    if (aggregate_residues) {
        status = ensure_residue_capacity(context,
                                         sizeof(int) * (size_t)input->n_atoms,
                                         sizeof(double) * (size_t)input->n_residues);
        if (status != FASTSASA_SUCCESS) return status;
    }

    if (cuda_status(cudaMemcpyAsync(context->d_xyz, input->xyz, xyz_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (cuda_status(cudaMemcpyAsync(context->d_radii, input->radii, radii_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (use_const_test_points) {
        if (context->reusable_const_test_points_n != input->n_points) {
            if (cuda_status(cudaMemcpyToSymbolAsync(const_test_points, input->test_points, test_point_bytes, 0, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
            context->reusable_const_test_points_n = input->n_points;
        }
        context->reusable_test_points_n = -1;
    } else if (!input->reuse_test_points || context->reusable_test_points_n != input->n_points) {
        if (cuda_status(cudaMemcpyAsync(context->d_test_points, input->test_points, test_point_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        context->reusable_test_points_n = input->reuse_test_points ? input->n_points : -1;
        context->reusable_const_test_points_n = -1;
    }
    if (cuda_status(cudaMemcpyAsync(context->d_neighbor_offsets, input->neighbor_offsets, offset_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (index_bytes > 0 &&
        cuda_status(cudaMemcpyAsync(context->d_neighbor_indices, input->neighbor_indices, index_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (aggregate_residues) {
        if (cuda_status(cudaMemcpyAsync(context->d_residue_ids,
                                        input->residue_ids,
                                        sizeof(int) * (size_t)input->n_atoms,
                                        cudaMemcpyHostToDevice,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
        if (cuda_status(cudaMemsetAsync(context->d_residue_sasa,
                                        0,
                                        sizeof(double) * (size_t)input->n_residues,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    }

    square_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii, context->d_radii2);
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;

    if (use_const_test_points) {
        shrake_rupley_const_points_kernel<<<input->n_atoms, sr_threads, sizeof(int) * sr_threads, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xyz,
            context->d_radii,
            context->d_radii2,
            context->d_neighbor_offsets,
            context->d_neighbor_indices,
            context->d_sasa);
    } else {
        shrake_rupley_kernel<<<input->n_atoms, sr_threads, sizeof(int) * sr_threads, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xyz,
            context->d_radii,
            context->d_radii2,
            context->d_test_points,
            context->d_neighbor_offsets,
            context->d_neighbor_indices,
            context->d_sasa);
    }
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;

    if (cuda_status(cudaMemcpyAsync(sasa, context->d_sasa, sasa_bytes, cudaMemcpyDeviceToHost, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (cuda_status(cudaStreamSynchronize(context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (aggregate_residues) {
        return host_aggregate_sasa(sasa, input->n_atoms, NULL,
                                   input->residue_ids, input->n_residues, input->residue_sasa,
                                   NULL, 0, NULL);
    }

    return FASTSASA_SUCCESS;

cuda_fail:
    return FASTSASA_CUDA_ERROR;
}

int
fastsasa_device_shrake_rupley_csr(const fastsasa_device_sr_input *input,
                               double *sasa)
{
    fastsasa_device_context *context = NULL;
    int status = fastsasa_device_context_create(&context);

    if (status != FASTSASA_SUCCESS) return status;

    status = fastsasa_device_context_shrake_rupley_csr(context, input, sasa);
    fastsasa_device_context_free(context);
    return status;
}

static int
context_shrake_rupley_cell_list_impl(fastsasa_device_context *context,
                                     const fastsasa_device_sr_input *input,
                                     double *sasa,
                                     double *total_sasa,
                                     int synchronize)
{
    int status = validate_cell_input(input, sasa);
    enum {
        PROFILE_START = 0,
        PROFILE_AFTER_H2D,
        PROFILE_AFTER_RADIUS,
        PROFILE_AFTER_COUNT,
        PROFILE_AFTER_SCAN,
        PROFILE_AFTER_FILL,
        PROFILE_AFTER_SR,
        PROFILE_AFTER_RESIDUE,
        PROFILE_AFTER_D2H,
        PROFILE_EVENT_COUNT
    };
    cudaEvent_t profile_events[PROFILE_EVENT_COUNT] = {0};
    int profile_active = 0;
    if (status != FASTSASA_SUCCESS) return status;
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (sasa == NULL && total_sasa == NULL &&
        input->residue_sasa == NULL && input->selection_sasa == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    const int use_soa_coords = input_has_soa_coords(input);
    double min_x = input_coord_x(input, 0);
    double min_y = input_coord_y(input, 0);
    double min_z = input_coord_z(input, 0);
    double max_x = min_x;
    double max_y = min_y;
    double max_z = min_z;
    double max_radius = input->radii[0];

    for (int i = 1; i < input->n_atoms; ++i) {
        const double x = input_coord_x(input, i);
        const double y = input_coord_y(input, i);
        const double z = input_coord_z(input, i);

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (z < min_z) min_z = z;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        if (z > max_z) max_z = z;
        if (input->radii[i] > max_radius) max_radius = input->radii[i];
    }
    if (max_radius <= 0.0) return FASTSASA_INVALID_ARGUMENT;

    const double cell_size = 2.0 * max_radius;
    min_x -= cell_size;
    min_y -= cell_size;
    min_z -= cell_size;
    max_x += cell_size;
    max_y += cell_size;
    max_z += cell_size;

    int nx;
    int ny;
    int nz;
    int n_cells;

    status = cell_grid_dimension(max_x - min_x, cell_size, &nx);
    if (status != FASTSASA_SUCCESS) return status;
    status = cell_grid_dimension(max_y - min_y, cell_size, &ny);
    if (status != FASTSASA_SUCCESS) return status;
    status = cell_grid_dimension(max_z - min_z, cell_size, &nz);
    if (status != FASTSASA_SUCCESS) return status;
    status = validate_dense_cell_grid(input->n_atoms, nx, ny, nz, &n_cells);
    if (status != FASTSASA_SUCCESS) return status;
    const size_t xyz_bytes = use_soa_coords ? 0 : sizeof(double) * 3u * (size_t)input->n_atoms;
    const size_t coord_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t radii_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t test_point_bytes = sizeof(double) * 3u * (size_t)input->n_points;
    const size_t sasa_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t atom_cells_bytes = sizeof(int) * (size_t)input->n_atoms;
    const size_t cell_count_bytes = sizeof(int) * (size_t)n_cells;
    const size_t cell_offset_bytes = sizeof(int) * ((size_t)n_cells + 1u);
    const size_t cell_atoms_bytes = sizeof(int) * (size_t)input->n_atoms;
    const int aggregate_residues = input->residue_ids != NULL;
    const int aggregate_selections = input->selection_masks != NULL;
    const int active_center = input->active_center_mask != 0u && input->selection_masks != NULL;
    const int aggregate_total = total_sasa != NULL;
    const int aggregate_host = aggregate_total || aggregate_residues || aggregate_selections;
    const int float_sr = input->force_double_precision > 0
                             ? 0
                             : (input->force_double_precision < 0
                                    ? 1
                                    : use_float_sr());
    const int hybrid_sr = !float_sr && use_sr_fp64_hybrid();
    const fastsasa_sr_dispatch_policy dispatch = select_sr_dispatch_policy(
        input->n_points,
        input->n_atoms,
        n_cells,
        float_sr,
        active_center,
        input->active_center_indices != NULL,
        input->n_active_centers);
    const int ordered_cells = dispatch.ordered_cells;
    const int compact_active_centers = dispatch.compact_active_centers;
    const int sr_center_count = compact_active_centers
                                    ? input->n_active_centers
                                    : input->n_atoms;
    /* Keep reusable points in context-owned global memory for stream isolation. */
    const int use_const_test_points = 0;
    const int sort_cell_list = dispatch.sort_cell_list;

    const int prep_threads = 256;
    const int prep_blocks = (input->n_atoms + prep_threads - 1) / prep_threads;
    const int sr_threads = sr_block_threads(input->n_points);
    const int sr_warps_per_block = sr_threads / 32;
    const int use_warp_atom_sr = dispatch.warp_atom_sr;
    const int use_point_compaction_sr = dispatch.point_compaction_sr;
    const int sr_blocks = use_warp_atom_sr
                              ? (sr_center_count + sr_warps_per_block - 1) / sr_warps_per_block
                              : sr_center_count;
    const int use_shared_neighbor_cache = dispatch.shared_neighbor_cache;
    const size_t sr_reduce_shared_bytes = sizeof(int) * (size_t)sr_threads;
    const size_t sr_compact_shared_bytes = sizeof(int) * (2u * (size_t)input->n_points + 1u);
    const size_t sr_cached_shared_bytes = sizeof(double) * 4u * (size_t)FASTSASA_NEIGHBOR_CHUNK +
                                          sizeof(int) * ((size_t)sr_threads +
                                                         (size_t)FASTSASA_NEIGHBOR_CHUNK);

    status = validate_launch_size(input->n_atoms, sr_threads);
    if (status != FASTSASA_SUCCESS) return status;
    status = validate_shared_memory_size(use_point_compaction_sr
                                             ? sr_compact_shared_bytes
                                             : (use_shared_neighbor_cache && !float_sr
                                                    ? sr_cached_shared_bytes
                                                    : sr_reduce_shared_bytes));
    if (status != FASTSASA_SUCCESS) return status;
    memset(&context->last_cell_profile, 0, sizeof(context->last_cell_profile));
    if (context->profile_enabled && synchronize) {
        status = create_profile_events(profile_events, PROFILE_EVENT_COUNT);
        if (status != FASTSASA_SUCCESS) return status;
        profile_active = 1;
        if (cuda_status(cudaEventRecord(profile_events[PROFILE_START], context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    }

    status = ensure_context_capacity(context,
                                     xyz_bytes,
                                     radii_bytes,
                                     use_const_test_points ? 0 : test_point_bytes,
                                     0,
                                     0,
                                     sasa != NULL || aggregate_host ? sasa_bytes : 0);
    if (status != FASTSASA_SUCCESS) goto status_fail;
    status = ensure_soa_capacity(context, coord_bytes);
    if (status != FASTSASA_SUCCESS) goto status_fail;
    if (float_sr || hybrid_sr) {
        status = ensure_float_sr_capacity(context, coord_bytes, test_point_bytes);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }
    if (aggregate_total) {
        status = ensure_device_capacity((void **)&context->d_total_sasa,
                                        &context->total_sasa_capacity,
                                        sizeof(double));
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }

    status = ensure_cell_capacity(context,
                                  atom_cells_bytes,
                                  cell_count_bytes,
                                  cell_offset_bytes,
                                  cell_atoms_bytes);
    if (status != FASTSASA_SUCCESS) goto status_fail;
    status = ensure_cell_scan_storage(context, n_cells);
    if (status != FASTSASA_SUCCESS) goto status_fail;
    if (sort_cell_list) {
        status = ensure_cell_sort_storage(context, input->n_atoms, n_cells);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }

    if (aggregate_residues) {
        status = ensure_residue_capacity(context,
                                         sizeof(int) * (size_t)input->n_atoms,
                                         sizeof(double) * (size_t)input->n_residues);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }
    if (aggregate_selections) {
        status = ensure_selection_capacity(context,
                                           sizeof(unsigned int) * (size_t)input->n_atoms,
                                           sizeof(double) * (size_t)input->n_selections);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }
    if (compact_active_centers) {
        status = ensure_active_center_capacity(context,
                                              sizeof(int) * (size_t)input->n_active_centers);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }
    if (use_soa_coords) {
        if (cuda_status(cudaMemcpyAsync(context->d_x, input->x, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        if (cuda_status(cudaMemcpyAsync(context->d_y, input->y, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        if (cuda_status(cudaMemcpyAsync(context->d_z, input->z, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    } else if (cuda_status(cudaMemcpyAsync(context->d_xyz, input->xyz, xyz_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (cuda_status(cudaMemcpyAsync(context->d_radii, input->radii, radii_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (use_const_test_points) {
        if (context->reusable_const_test_points_n != input->n_points) {
            if (cuda_status(cudaMemcpyToSymbolAsync(const_test_points, input->test_points, test_point_bytes, 0, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
            context->reusable_const_test_points_n = input->n_points;
        }
        context->reusable_test_points_n = -1;
    } else if (!input->reuse_test_points || context->reusable_test_points_n != input->n_points) {
        if (cuda_status(cudaMemcpyAsync(context->d_test_points, input->test_points, test_point_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        context->reusable_test_points_n = input->reuse_test_points ? input->n_points : -1;
        context->reusable_const_test_points_n = -1;
    }
    if (aggregate_residues) {
        if (cuda_status(cudaMemcpyAsync(context->d_residue_ids,
                                        input->residue_ids,
                                        sizeof(int) * (size_t)input->n_atoms,
                                        cudaMemcpyHostToDevice,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
        if (cuda_status(cudaMemsetAsync(context->d_residue_sasa,
                                        0,
                                        sizeof(double) * (size_t)input->n_residues,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    }
    if (aggregate_selections) {
        if (cuda_status(cudaMemcpyAsync(context->d_selection_masks,
                                        input->selection_masks,
                                        sizeof(unsigned int) * (size_t)input->n_atoms,
                                        cudaMemcpyHostToDevice,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
        if (cuda_status(cudaMemsetAsync(context->d_selection_sasa,
                                        0,
                                        sizeof(double) * (size_t)input->n_selections,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    }
    if (compact_active_centers &&
        cuda_status(cudaMemcpyAsync(context->d_active_center_indices,
                                    input->active_center_indices,
                                    sizeof(int) * (size_t)input->n_active_centers,
                                    cudaMemcpyHostToDevice,
                                    context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (aggregate_total &&
        cuda_status(cudaMemsetAsync(context->d_total_sasa,
                                    0,
                                    sizeof(double),
                                    context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (!use_soa_coords) {
        split_xyz_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            context->d_xyz,
            context->d_x,
            context->d_y,
            context->d_z);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    }
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_H2D], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if ((active_center || aggregate_host) &&
        cuda_status(cudaMemsetAsync(context->d_sasa, 0, sasa_bytes, context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (float_sr) {
        const int point_values = 3 * input->n_points;
        const int point_blocks = (point_values + prep_threads - 1) / prep_threads;

        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_x, min_x, context->d_xf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_y, min_y, context->d_yf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_z, min_z, context->d_zf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii, context->d_radii_f);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_float_kernel<<<point_blocks, prep_threads, 0, context->stream>>>(point_values, context->d_test_points, context->d_test_points_f);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        square_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii_f, context->d_radii2_f);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    } else {
        square_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii, context->d_radii2);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        if (hybrid_sr) {
            /* Box-local float shadows feed the hybrid kernel's prefilter. */
            double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_x, min_x, context->d_xf);
            if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
            double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_y, min_y, context->d_yf);
            if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
            double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_z, min_z, context->d_zf);
            if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
            double_to_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii2, context->d_radii2_f);
            if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        }
    }
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_RADIUS], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (cuda_status(cudaMemsetAsync(context->d_cell_counts, 0, cell_count_bytes, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    cell_count_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
        input->n_atoms,
        context->d_x,
        context->d_y,
        context->d_z,
        min_x,
        min_y,
        min_z,
        cell_size,
        nx,
        ny,
        nz,
        context->d_atom_cells,
        context->d_cell_counts);
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_COUNT], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (cuda_status(cudaMemsetAsync(context->d_cell_offsets,
                                    0,
                                    sizeof(int),
                                    context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (cuda_status(cub::DeviceScan::InclusiveSum(context->d_cell_scan_storage,
                                                  context->cell_scan_storage_capacity,
                                                  context->d_cell_counts,
                                                  context->d_cell_offsets + 1,
                                                  n_cells,
                                                  context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_SCAN], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (sort_cell_list) {
        sequence_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            context->d_atom_indices);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        if (cuda_status(cub::DeviceRadixSort::SortPairs(context->d_cell_sort_storage,
                                                        context->cell_sort_storage_capacity,
                                                        context->d_atom_cells,
                                                        context->d_sorted_cells,
                                                        context->d_atom_indices,
                                                        context->d_cell_atoms,
                                                        input->n_atoms,
                                                        0,
                                                        sort_end_bit(n_cells),
                                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    } else {
        if (cuda_status(cudaMemcpyAsync(context->d_cell_fill,
                                        context->d_cell_offsets,
                                        cell_count_bytes,
                                        cudaMemcpyDeviceToDevice,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
        cell_fill_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            context->d_atom_cells,
            context->d_cell_fill,
            context->d_cell_atoms);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    }
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_FILL], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (use_warp_atom_sr) {
        shrake_rupley_cell_float_warp_ordered_kernel<<<sr_blocks, sr_threads, 0, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            context->d_radii2_f,
            context->d_test_points_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (use_point_compaction_sr) {
        shrake_rupley_cell_float_compact_kernel<<<sr_center_count, sr_threads, sr_compact_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            context->d_radii2_f,
            context->d_test_points_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (float_sr && ordered_cells) {
        shrake_rupley_cell_float_ordered_kernel<<<sr_center_count, sr_threads, sr_reduce_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            context->d_radii2_f,
            context->d_test_points_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (float_sr) {
        shrake_rupley_cell_float_kernel<<<sr_center_count, sr_threads, sr_reduce_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            context->d_radii2_f,
            context->d_test_points_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (hybrid_sr) {
        /* Conservative FP32-prefilter uncertainty margin. Coordinates are
         * box-local, so magnitudes are bounded by the grid extent; distances
         * that reach the comparison are bounded by the 27-cell block. The
         * factor-of-two slack absorbs the radii2 float conversion. */
        const double eps32 = 5.9604644775390625e-8; /* 2^-24 */
        const int max_dim = nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz);
        const double extent = cell_size * (double)(max_dim + 1);
        const double dmax = 6.0 * cell_size;
        const double delta = 8.0 * eps32 * extent;
        const float sr_hybrid_margin =
            (float)(2.0 * (2.0 * dmax * delta + delta * delta +
                           8.0 * eps32 * dmax * dmax));

        shrake_rupley_cell_hybrid_kernel<<<sr_center_count, sr_threads, sr_reduce_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_x,
            context->d_y,
            context->d_z,
            context->d_radii,
            context->d_radii2,
            context->d_test_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii2_f,
            min_x,
            min_y,
            min_z,
            sr_hybrid_margin,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (use_const_test_points && use_shared_neighbor_cache) {
        shrake_rupley_cell_const_points_shared_kernel<<<input->n_atoms, sr_threads, sr_cached_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_x,
            context->d_y,
            context->d_z,
            context->d_radii,
            context->d_radii2,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (use_const_test_points) {
        shrake_rupley_cell_const_points_kernel<<<input->n_atoms, sr_threads, sr_reduce_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_x,
            context->d_y,
            context->d_z,
            context->d_radii,
            context->d_radii2,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else if (use_shared_neighbor_cache) {
        shrake_rupley_cell_shared_kernel<<<input->n_atoms, sr_threads, sr_cached_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_x,
            context->d_y,
            context->d_z,
            context->d_radii,
            context->d_radii2,
            context->d_test_points,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    } else {
        shrake_rupley_cell_kernel<<<sr_center_count, sr_threads, sr_reduce_shared_bytes, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_x,
            context->d_y,
            context->d_z,
            context->d_radii,
            context->d_radii2,
            context->d_test_points,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            compact_active_centers ? context->d_active_center_indices : NULL,
            sr_center_count,
            active_center ? context->d_selection_masks : NULL,
            active_center ? input->active_center_mask : 0u,
            sasa != NULL || aggregate_host ? context->d_sasa : NULL,
            NULL);
    }
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_SR], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_RESIDUE], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }

    if (sasa != NULL &&
        cuda_status(cudaMemcpyAsync(sasa, context->d_sasa, sasa_bytes, cudaMemcpyDeviceToHost, context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (aggregate_host) {
        fastsasa_pending_aggregate *entry = NULL;

        status = pending_aggregate_push(context, input, total_sasa, &entry);
        if (status != FASTSASA_SUCCESS) goto status_fail;
        if (cuda_status(cudaMemcpyAsync(entry->h_sasa, context->d_sasa, sasa_bytes,
                                        cudaMemcpyDeviceToHost, context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    }
    if (profile_active &&
        cuda_status(cudaEventRecord(profile_events[PROFILE_AFTER_D2H], context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (synchronize) {
        status = synchronize_and_flush(context);
        if (status != FASTSASA_SUCCESS) goto status_fail;
    }
    if (profile_active) {
        context->last_cell_profile.h2d_ms = event_elapsed_ms(profile_events[PROFILE_START],
                                                             profile_events[PROFILE_AFTER_H2D]);
        context->last_cell_profile.radius_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_H2D],
                                                                profile_events[PROFILE_AFTER_RADIUS]);
        context->last_cell_profile.cell_count_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_RADIUS],
                                                                    profile_events[PROFILE_AFTER_COUNT]);
        context->last_cell_profile.scan_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_COUNT],
                                                              profile_events[PROFILE_AFTER_SCAN]);
        context->last_cell_profile.cell_fill_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_SCAN],
                                                                   profile_events[PROFILE_AFTER_FILL]);
        context->last_cell_profile.sr_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_FILL],
                                                            profile_events[PROFILE_AFTER_SR]);
        context->last_cell_profile.residue_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_SR],
                                                                 profile_events[PROFILE_AFTER_RESIDUE]);
        context->last_cell_profile.d2h_ms = event_elapsed_ms(profile_events[PROFILE_AFTER_RESIDUE],
                                                             profile_events[PROFILE_AFTER_D2H]);
        context->last_cell_profile.total_ms = event_elapsed_ms(profile_events[PROFILE_START],
                                                               profile_events[PROFILE_AFTER_D2H]);
        destroy_profile_events(profile_events, PROFILE_EVENT_COUNT);
    }

    return FASTSASA_SUCCESS;

status_fail:
    if (profile_active) destroy_profile_events(profile_events, PROFILE_EVENT_COUNT);
    return status;

cuda_fail:
    if (profile_active) destroy_profile_events(profile_events, PROFILE_EVENT_COUNT);
    return FASTSASA_CUDA_ERROR;
}

static int
validate_lr_input(const fastsasa_device_sr_input *input,
                  const double *sasa)
{
    if (input == NULL || sasa == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (input->n_atoms <= 0 || input->n_points <= 0) return FASTSASA_INVALID_ARGUMENT;
    if (input->xyz == NULL && !input_has_soa_coords(input)) return FASTSASA_INVALID_ARGUMENT;
    if (input->xyz != NULL && input_has_partial_soa_coords(input)) return FASTSASA_INVALID_ARGUMENT;
    if (input->xyz == NULL && input_has_partial_soa_coords(input) && !input_has_soa_coords(input)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (input->radii == NULL) return FASTSASA_INVALID_ARGUMENT;
    return validate_finite_geometry(input, 0);
}

/*
 * Atoms whose slice needed more than FASTSASA_LR_ARC_CAPACITY arcs on the
 * device are recomputed with the CPU reference implementation itself, so
 * the result stays bit-identical without a device-side fallback. This is
 * rare (it needs more than 32 neighbours cutting one slice circle).
 */
static int
lr_recompute_overflow_atoms(fastsasa_device_context *context,
                            const fastsasa_device_sr_input *input,
                            double *sasa)
{
    const int n_atoms = input->n_atoms;
    int *flags = (int *)malloc(sizeof(int) * (size_t)n_atoms);
    const double *x = input->x;
    const double *y = input->y;
    const double *z = input->z;
    double *split = NULL;
    int n_overflow = 0;
    int status = FASTSASA_SUCCESS;

    if (flags == NULL) return FASTSASA_MEMORY_ERROR;
    if (cuda_status(cudaMemcpy(flags, context->d_lr_overflow, sizeof(int) * (size_t)n_atoms,
                               cudaMemcpyDeviceToHost)) != FASTSASA_SUCCESS) {
        free(flags);
        return FASTSASA_CUDA_ERROR;
    }
    for (int atom = 0; atom < n_atoms; ++atom) n_overflow += flags[atom] != 0;
    if (n_overflow == 0) {
        free(flags);
        return FASTSASA_SUCCESS;
    }
    if (!input_has_soa_coords(input)) {
        split = (double *)malloc(sizeof(double) * 3u * (size_t)n_atoms);
        if (split == NULL) {
            free(flags);
            return FASTSASA_MEMORY_ERROR;
        }
        for (int atom = 0; atom < n_atoms; ++atom) {
            split[atom] = input->xyz[3 * atom];
            split[n_atoms + atom] = input->xyz[3 * atom + 1];
            split[2 * n_atoms + atom] = input->xyz[3 * atom + 2];
        }
        x = split;
        y = split + n_atoms;
        z = split + 2 * n_atoms;
    }
    for (int atom = 0; atom < n_atoms && status == FASTSASA_SUCCESS; ++atom) {
        if (flags[atom]) {
            status = fastsasa_cpu_lee_richards_atom(n_atoms, input->n_points, x, y, z,
                                                  input->radii, atom, &sasa[atom]);
        }
    }
    free(split);
    free(flags);
    return status;
}

int
fastsasa_device_context_lee_richards(fastsasa_device_context *context,
                                  const fastsasa_device_sr_input *input,
                                  double *sasa)
{
    int status = validate_lr_input(input, sasa);
    const int prep_threads = 256;
    int prep_blocks;
    int lr_blocks;
    long long total_slices;
    const int use_soa_coords = input_has_soa_coords(input);
    const int float_lr = input->force_double_precision > 0
                             ? 0
                             : (input->force_double_precision < 0
                                    ? 1
                                    : use_float_lr());
    const int float_lr_accum = float_lr && use_float_lr_accumulation();
    int lr_overflow_checked = 0;
    double min_x;
    double min_y;
    double min_z;
    double max_x;
    double max_y;
    double max_z;
    double max_radius;
    double cell_size;
    int nx;
    int ny;
    int nz;
    int n_cells;
    const size_t xyz_bytes = use_soa_coords ? 0 : sizeof(double) * 3u * (size_t)input->n_atoms;
    const size_t coord_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t radii_bytes = sizeof(double) * (size_t)input->n_atoms;
    const size_t sasa_bytes = sizeof(double) * (size_t)input->n_atoms;
    size_t atom_cells_bytes;
    size_t cell_count_bytes;
    size_t cell_offset_bytes;
    size_t cell_atoms_bytes;

    if (status != FASTSASA_SUCCESS) return status;
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    total_slices = (long long)input->n_atoms * (long long)input->n_points;
    if (total_slices <= 0 || total_slices > 2147483647LL) return FASTSASA_INVALID_ARGUMENT;

    min_x = input_coord_x(input, 0);
    min_y = input_coord_y(input, 0);
    min_z = input_coord_z(input, 0);
    max_x = min_x;
    max_y = min_y;
    max_z = min_z;
    max_radius = input->radii[0];
    for (int i = 1; i < input->n_atoms; ++i) {
        const double x = input_coord_x(input, i);
        const double y = input_coord_y(input, i);
        const double z = input_coord_z(input, i);

        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (z < min_z) min_z = z;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        if (z > max_z) max_z = z;
        if (input->radii[i] > max_radius) max_radius = input->radii[i];
    }
    if (max_radius <= 0.0) return FASTSASA_INVALID_ARGUMENT;

    cell_size = 2.0 * max_radius;
    min_x -= cell_size;
    min_y -= cell_size;
    min_z -= cell_size;
    max_x += cell_size;
    max_y += cell_size;
    max_z += cell_size;
    status = cell_grid_dimension(max_x - min_x, cell_size, &nx);
    if (status != FASTSASA_SUCCESS) return status;
    status = cell_grid_dimension(max_y - min_y, cell_size, &ny);
    if (status != FASTSASA_SUCCESS) return status;
    status = cell_grid_dimension(max_z - min_z, cell_size, &nz);
    if (status != FASTSASA_SUCCESS) return status;
    status = validate_dense_cell_grid(input->n_atoms, nx, ny, nz, &n_cells);
    if (status != FASTSASA_SUCCESS) return status;
    atom_cells_bytes = sizeof(int) * (size_t)input->n_atoms;
    cell_count_bytes = sizeof(int) * (size_t)n_cells;
    cell_offset_bytes = sizeof(int) * ((size_t)n_cells + 1u);
    cell_atoms_bytes = sizeof(int) * (size_t)input->n_atoms;

    status = validate_launch_size((int)((total_slices + prep_threads - 1) / prep_threads), prep_threads);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_context_capacity(context, xyz_bytes, radii_bytes, 0, 0, 0, sasa_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_soa_capacity(context, coord_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    if (float_lr || use_lr_fp64_hybrid()) {
        status = ensure_float_sr_capacity(context, coord_bytes, 0);
        if (status != FASTSASA_SUCCESS) return status;
        if (float_lr_accum) {
            status = ensure_float_sasa_capacity(context, sasa_bytes);
            if (status != FASTSASA_SUCCESS) return status;
        }
    }
    status = ensure_cell_capacity(context,
                                  atom_cells_bytes,
                                  cell_count_bytes,
                                  cell_offset_bytes,
                                  cell_atoms_bytes);
    if (status != FASTSASA_SUCCESS) return status;
    status = ensure_cell_scan_storage(context, n_cells);
    if (status != FASTSASA_SUCCESS) return status;

    prep_blocks = (input->n_atoms + prep_threads - 1) / prep_threads;
    lr_blocks = (int)((total_slices + (long long)prep_threads - 1LL) /
                      (long long)prep_threads);

    if (use_soa_coords) {
        if (cuda_status(cudaMemcpyAsync(context->d_x, input->x, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        if (cuda_status(cudaMemcpyAsync(context->d_y, input->y, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        if (cuda_status(cudaMemcpyAsync(context->d_z, input->z, coord_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    } else {
        if (cuda_status(cudaMemcpyAsync(context->d_xyz, input->xyz, xyz_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
        split_xyz_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            context->d_xyz,
            context->d_x,
            context->d_y,
            context->d_z);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    }
    if (cuda_status(cudaMemcpyAsync(context->d_radii, input->radii, radii_bytes, cudaMemcpyHostToDevice, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (float_lr || use_lr_fp64_hybrid()) {
        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_x, min_x, context->d_xf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_y, min_y, context->d_yf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_shifted_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_z, min_z, context->d_zf);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        double_to_float_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(input->n_atoms, context->d_radii, context->d_radii_f);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    }
    if (cuda_status(cudaMemsetAsync(context->d_sasa, 0, sasa_bytes, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (float_lr_accum) {
        if (cuda_status(cudaMemsetAsync(context->d_sasa_f,
                                        0,
                                        sizeof(float) * (size_t)input->n_atoms,
                                        context->stream)) != FASTSASA_SUCCESS) {
            goto cuda_fail;
        }
    }

    if (cuda_status(cudaMemsetAsync(context->d_cell_counts, 0, cell_count_bytes, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    cell_count_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
        input->n_atoms,
        context->d_x,
        context->d_y,
        context->d_z,
        min_x,
        min_y,
        min_z,
        cell_size,
        nx,
        ny,
        nz,
        context->d_atom_cells,
        context->d_cell_counts);
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    if (cuda_status(cudaMemsetAsync(context->d_cell_offsets,
                                    0,
                                    sizeof(int),
                                    context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (cuda_status(cub::DeviceScan::InclusiveSum(context->d_cell_scan_storage,
                                                  context->cell_scan_storage_capacity,
                                                  context->d_cell_counts,
                                                  context->d_cell_offsets + 1,
                                                  n_cells,
                                                  context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    if (cuda_status(cudaMemcpyAsync(context->d_cell_fill,
                                    context->d_cell_offsets,
                                    cell_count_bytes,
                                    cudaMemcpyDeviceToDevice,
                                    context->stream)) != FASTSASA_SUCCESS) {
        goto cuda_fail;
    }
    cell_fill_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
        input->n_atoms,
        context->d_atom_cells,
        context->d_cell_fill,
        context->d_cell_atoms);
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;

    if (float_lr_accum) {
        lee_richards_cell_slice_float_accum_kernel<<<lr_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            context->d_sasa_f);
        if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
        float_to_double_kernel<<<prep_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            context->d_sasa_f,
            context->d_sasa);
    } else if (float_lr) {
        lee_richards_cell_slice_float_kernel<<<lr_blocks, prep_threads, 0, context->stream>>>(
            input->n_atoms,
            input->n_points,
            context->d_xf,
            context->d_yf,
            context->d_zf,
            context->d_radii_f,
            nx,
            ny,
            nz,
            context->d_atom_cells,
            context->d_cell_offsets,
            context->d_cell_atoms,
            context->d_sasa);
    } else {
        {
            const double eps32 = 5.9604644775390625e-8; /* 2^-24 */
            const int max_dim = nx > ny ? (nx > nz ? nx : nz)
                                        : (ny > nz ? ny : nz);
            const double extent = cell_size * (double)(max_dim + 1);
            const double dmax = 6.0 * cell_size;
            const double delta_c = 8.0 * eps32 * extent;
            const int lr_hybrid = use_lr_fp64_hybrid();
            const double linear_margin = 32.0 * eps32 * (extent + cell_size);
            const double squared_margin =
                8.0 * (2.0 * dmax * delta_c + delta_c * delta_c +
                       8.0 * eps32 * dmax * dmax);

            /* Per-slice contributions are staged in a bounded buffer and
             * reduced per atom in slice order; large systems run in atom
             * chunks so the staging buffer stays under the cap. */
            const size_t slice_bytes = sizeof(double) * (size_t)input->n_points;
            const size_t chunk_cap_bytes = (size_t)256 << 20;
            int chunk_atoms = (int)(chunk_cap_bytes / slice_bytes);

            if (chunk_atoms < 1) chunk_atoms = 1;
            if (chunk_atoms > input->n_atoms) chunk_atoms = input->n_atoms;
            status = ensure_device_capacity((void **)&context->d_lr_slice_areas,
                                            &context->lr_slice_areas_capacity,
                                            slice_bytes * (size_t)chunk_atoms);
            if (status != FASTSASA_SUCCESS) return status;
            status = ensure_device_capacity((void **)&context->d_lr_overflow,
                                            &context->lr_overflow_capacity,
                                            sizeof(int) * (size_t)input->n_atoms);
            if (status != FASTSASA_SUCCESS) return status;
            if (cuda_status(cudaMemsetAsync(context->d_lr_overflow, 0,
                                            sizeof(int) * (size_t)input->n_atoms,
                                            context->stream)) != FASTSASA_SUCCESS) {
                goto cuda_fail;
            }
            lr_overflow_checked = 1;
            for (int atom_begin = 0; atom_begin < input->n_atoms; atom_begin += chunk_atoms) {
                const int atom_end = atom_begin + chunk_atoms < input->n_atoms
                                         ? atom_begin + chunk_atoms
                                         : input->n_atoms;
                const long long chunk_slices = (long long)(atom_end - atom_begin) * (long long)input->n_points;
                const int chunk_blocks = (int)((chunk_slices + (long long)prep_threads - 1LL) /
                                               (long long)prep_threads);
                const int reduce_blocks = (atom_end - atom_begin + prep_threads - 1) / prep_threads;

                lee_richards_cell_slice_kernel<<<chunk_blocks, prep_threads, 0, context->stream>>>(
                    input->n_atoms,
                    input->n_points,
                    atom_begin,
                    atom_end,
                    context->d_x,
                    context->d_y,
                    context->d_z,
                    context->d_radii,
                    lr_hybrid ? context->d_xf : NULL,
                    lr_hybrid ? context->d_yf : NULL,
                    lr_hybrid ? context->d_zf : NULL,
                    lr_hybrid ? context->d_radii_f : NULL,
                    min_x,
                    min_y,
                    min_z,
                    linear_margin,
                    squared_margin,
                    nx,
                    ny,
                    nz,
                    context->d_atom_cells,
                    context->d_cell_offsets,
                    context->d_cell_atoms,
                    context->d_lr_slice_areas,
                    context->d_lr_overflow);
                if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
                lee_richards_slice_reduce_kernel<<<reduce_blocks, prep_threads, 0, context->stream>>>(
                    input->n_points,
                    atom_begin,
                    atom_end,
                    context->d_lr_slice_areas,
                    context->d_sasa);
                if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
            }
        }
    }
    if (cuda_status(cudaGetLastError()) != FASTSASA_SUCCESS) goto cuda_fail;
    if (cuda_status(cudaMemcpyAsync(sasa, context->d_sasa, sasa_bytes, cudaMemcpyDeviceToHost, context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (cuda_status(cudaStreamSynchronize(context->stream)) != FASTSASA_SUCCESS) goto cuda_fail;
    if (lr_overflow_checked) {
        status = lr_recompute_overflow_atoms(context, input, sasa);
        if (status != FASTSASA_SUCCESS) return status;
    }

    return FASTSASA_SUCCESS;

cuda_fail:
    return FASTSASA_CUDA_ERROR;
}

int
fastsasa_device_context_shrake_rupley_cell_list(fastsasa_device_context *context,
                                             const fastsasa_device_sr_input *input,
                                             double *sasa)
{
    return context_shrake_rupley_cell_list_impl(context, input, sasa, NULL, 1);
}

int
fastsasa_device_context_shrake_rupley_cell_list_async(fastsasa_device_context *context,
                                                   const fastsasa_device_sr_input *input,
                                                   double *sasa)
{
    return context_shrake_rupley_cell_list_impl(context, input, sasa, NULL, 0);
}

int
fastsasa_device_context_shrake_rupley_cell_list_total(fastsasa_device_context *context,
                                                   const fastsasa_device_sr_input *input,
                                                   double *total_sasa)
{
    return context_shrake_rupley_cell_list_impl(context, input, NULL, total_sasa, 1);
}

int
fastsasa_device_context_shrake_rupley_cell_list_total_async(fastsasa_device_context *context,
                                                         const fastsasa_device_sr_input *input,
                                                         double *total_sasa)
{
    return context_shrake_rupley_cell_list_impl(context, input, NULL, total_sasa, 0);
}

int
fastsasa_device_context_shrake_rupley_csr_batch(fastsasa_device_context *context,
                                             const fastsasa_device_sr_input *inputs,
                                             double *const *sasa_outputs,
                                             int n_inputs)
{
    if (context == NULL || inputs == NULL || sasa_outputs == NULL || n_inputs <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    for (int i = 0; i < n_inputs; ++i) {
        int status;

        if (sasa_outputs[i] == NULL) return FASTSASA_INVALID_ARGUMENT;
        status = fastsasa_device_context_shrake_rupley_csr(context,
                                                        &inputs[i],
                                                        sasa_outputs[i]);
        if (status != FASTSASA_SUCCESS) return status;
    }

    return FASTSASA_SUCCESS;
}
