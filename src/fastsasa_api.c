#include "fastsasa_exact_math.h"
#include "fastsasa.h"

#include "fastsasa_backend_internal.h"
#include "fastsasa_device.h"
#ifdef FASTSASA_HAVE_VULKAN
#include "fastsasa_vulkan_internal.h"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fastsasa_portable.h"

enum fastsasa_backend_kind {
    FASTSASA_BACKEND_CUDA,
    FASTSASA_BACKEND_VULKAN
};

struct fastsasa_context {
    enum fastsasa_backend_kind backend;
    int precision;
    fastsasa_device_context *cuda;
#ifdef FASTSASA_HAVE_VULKAN
    fastsasa_vk_context *vulkan;
    double *vk_xyz;
    double *vk_radii;
    double *vk_points;
    double *vk_areas;
    uint32_t *vk_centers;
    size_t vk_xyz_capacity;
    size_t vk_radii_capacity;
    size_t vk_point_capacity;
    size_t vk_area_capacity;
    size_t vk_center_capacity;
#endif
};

/* Thread-local so contexts running on parallel trajectory lanes cannot race
 * or overwrite each other's diagnostics. */
#if defined(_MSC_VER)
static __declspec(thread) char api_error[512] = "";
#else
static _Thread_local char api_error[512] = "";
#endif

static void
set_api_error(const char *message)
{
    if (message == NULL) message = "";
    snprintf(api_error, sizeof(api_error), "%s", message);
}

static const char *
backend_request(void)
{
    const char *value = getenv("FASTSASA_BACKEND");

    return value == NULL || value[0] == '\0' ? "auto" : value;
}

static int
backend_is(const char *request, const char *name)
{
    return strcmp(request, name) == 0;
}

static int
test_switch_enabled(const char *name)
{
    const char *value = getenv(name);

    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static void
copy_input(fastsasa_device_sr_input *dst,
           const fastsasa_sr_input *src,
           int precision)
{
    dst->n_atoms = src->n_atoms;
    dst->n_points = src->n_points;
    dst->xyz = src->xyz;
    dst->x = src->x;
    dst->y = src->y;
    dst->z = src->z;
    dst->radii = src->radii;
    dst->test_points = src->test_points;
    dst->neighbor_offsets = src->neighbor_offsets;
    dst->neighbor_indices = src->neighbor_indices;
    dst->n_neighbor_indices = src->n_neighbor_indices;
    dst->reuse_test_points = src->reuse_test_points;
    dst->residue_ids = src->residue_ids;
    dst->n_residues = src->n_residues;
    dst->residue_sasa = src->residue_sasa;
    dst->selection_masks = src->selection_masks;
    dst->n_selections = src->n_selections;
    dst->selection_sasa = src->selection_sasa;
    dst->active_center_mask = src->active_center_mask;
    dst->active_center_indices = src->active_center_indices;
    dst->n_active_centers = src->n_active_centers;
    dst->force_double_precision = src->force_double_precision != 0
                                      ? src->force_double_precision
                                      : (precision == FASTSASA_PRECISION_FP64 ? 1 : -1);
}

#ifdef FASTSASA_HAVE_VULKAN
static int
reserve_double(double **buffer, size_t *capacity, size_t count)
{
    double *replacement;

    if (*capacity >= count) return FASTSASA_SUCCESS;
    replacement = (double *)realloc(*buffer, count * sizeof(double));
    if (replacement == NULL) return FASTSASA_MEMORY_ERROR;
    *buffer = replacement;
    *capacity = count;
    return FASTSASA_SUCCESS;
}

static int
reserve_centers(fastsasa_context *context, size_t count)
{
    uint32_t *replacement;

    if (context->vk_center_capacity >= count) return FASTSASA_SUCCESS;
    replacement = (uint32_t *)realloc(context->vk_centers,
                                      count * sizeof(uint32_t));
    if (replacement == NULL) return FASTSASA_MEMORY_ERROR;
    context->vk_centers = replacement;
    context->vk_center_capacity = count;
    return FASTSASA_SUCCESS;
}

static int vulkan_use_fp64(const fastsasa_context *context, const fastsasa_sr_input *input);

static int
prepare_vulkan_input(fastsasa_context *context,
                     const fastsasa_sr_input *input,
                     int need_points,
                     uint32_t *center_count,
                     int *compact_centers)
{
    int status;
    size_t atom_count;
    size_t point_count;
    uint32_t selected = 0;

    if (input == NULL || input->n_atoms <= 0 || input->radii == NULL ||
        (input->xyz == NULL &&
         (input->x == NULL || input->y == NULL || input->z == NULL)) ||
        (need_points && (input->n_points <= 0 || input->test_points == NULL))) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    atom_count = (size_t)input->n_atoms;
    point_count = need_points ? (size_t)input->n_points : 0u;
    status = reserve_double(&context->vk_xyz, &context->vk_xyz_capacity,
                           atom_count * 3u);
    if (status != FASTSASA_SUCCESS) return status;
    status = reserve_double(&context->vk_radii, &context->vk_radii_capacity,
                           atom_count);
    if (status != FASTSASA_SUCCESS) return status;
    if (need_points) {
        status = reserve_double(&context->vk_points, &context->vk_point_capacity,
                               point_count * 3u);
        if (status != FASTSASA_SUCCESS) return status;
    }

    {
        /* FP32 works in box-local coordinates for accuracy. FP64 keeps the
         * absolute coordinates: the shifted differences (a-o)-(b-o) are not
         * the same doubles as a-b, and the FP64 shaders reproduce the CPU
         * reference bit for bit only on the reference's inputs. */
        const int shift = !vulkan_use_fp64(context, input);
        const double origin_x = shift ? (input->xyz != NULL ? input->xyz[0] : input->x[0]) : 0.0;
        const double origin_y = shift ? (input->xyz != NULL ? input->xyz[1] : input->y[0]) : 0.0;
        const double origin_z = shift ? (input->xyz != NULL ? input->xyz[2] : input->z[0]) : 0.0;

        for (size_t atom = 0; atom < atom_count; ++atom) {
            if (input->xyz != NULL) {
                context->vk_xyz[3u * atom] =
                    input->xyz[3u * atom] - origin_x;
                context->vk_xyz[3u * atom + 1u] =
                    input->xyz[3u * atom + 1u] - origin_y;
                context->vk_xyz[3u * atom + 2u] =
                    input->xyz[3u * atom + 2u] - origin_z;
            } else {
                context->vk_xyz[3u * atom] = input->x[atom] - origin_x;
                context->vk_xyz[3u * atom + 1u] =
                    input->y[atom] - origin_y;
                context->vk_xyz[3u * atom + 2u] =
                    input->z[atom] - origin_z;
            }
            context->vk_radii[atom] = input->radii[atom];
        }
    }
    for (size_t point = 0; point < point_count * 3u; ++point) {
        context->vk_points[point] = input->test_points[point];
    }

    *compact_centers = input->active_center_indices != NULL &&
                       input->n_active_centers > 0;
    if (*compact_centers) {
        selected = (uint32_t)input->n_active_centers;
        status = reserve_centers(context, selected);
        if (status != FASTSASA_SUCCESS) return status;
        for (uint32_t center = 0; center < selected; ++center) {
            const int atom = input->active_center_indices[center];
            if (atom < 0 || atom >= input->n_atoms) return FASTSASA_INVALID_ARGUMENT;
            context->vk_centers[center] = (uint32_t)atom;
        }
    } else if (input->active_center_mask != 0u && input->selection_masks != NULL) {
        status = reserve_centers(context, atom_count);
        if (status != FASTSASA_SUCCESS) return status;
        for (size_t atom = 0; atom < atom_count; ++atom) {
            if ((input->selection_masks[atom] & input->active_center_mask) != 0u) {
                context->vk_centers[selected++] = (uint32_t)atom;
            }
        }
        *compact_centers = 1;
    } else {
        selected = (uint32_t)atom_count;
    }

    status = reserve_double(&context->vk_areas, &context->vk_area_capacity,
                           selected > 0u ? selected : 1u);
    if (status != FASTSASA_SUCCESS) return status;
    *center_count = selected;
    return FASTSASA_SUCCESS;
}

static int
finish_vulkan_result(fastsasa_context *context,
                     const fastsasa_sr_input *input,
                     uint32_t center_count,
                     int compact_centers,
                     double *sasa,
                     double *total_sasa)
{
    /* Kahan sums in fixed atom order, like the CPU backend and the CUDA
     * host aggregation, so every backend reports the same double. */
    double total = 0.0;
    double total_compensation = 0.0;
    double *residue_compensation = NULL;
    double selection_compensation[32];

    memset(selection_compensation, 0, sizeof(selection_compensation));
    if (sasa != NULL) memset(sasa, 0, sizeof(double) * (size_t)input->n_atoms);
    if (input->residue_sasa != NULL && input->n_residues > 0) {
        memset(input->residue_sasa, 0,
               sizeof(double) * (size_t)input->n_residues);
        residue_compensation = (double *)calloc((size_t)input->n_residues, sizeof(double));
        if (residue_compensation == NULL) return FASTSASA_MEMORY_ERROR;
    }
    if (input->selection_sasa != NULL && input->n_selections > 0) {
        memset(input->selection_sasa, 0,
               sizeof(double) * (size_t)input->n_selections);
    }

    for (uint32_t center = 0; center < center_count; ++center) {
        const uint32_t atom = compact_centers ? context->vk_centers[center] : center;
        const double area = (double)context->vk_areas[center];

        if (sasa != NULL) sasa[atom] = area;
        fastsasa_exact_kahan_add(area, &total, &total_compensation);
        if (residue_compensation != NULL && input->residue_ids != NULL) {
            const int residue = input->residue_ids[atom];
            if (residue >= 0 && residue < input->n_residues) {
                fastsasa_exact_kahan_add(area, &input->residue_sasa[residue],
                                       &residue_compensation[residue]);
            }
        }
        if (input->selection_sasa != NULL && input->selection_masks != NULL &&
            input->n_selections <= 31) {
            const unsigned int mask = input->selection_masks[atom];
            for (int selection = 0; selection < input->n_selections; ++selection) {
                if ((mask & (1u << selection)) != 0u) {
                    fastsasa_exact_kahan_add(area, &input->selection_sasa[selection],
                                           &selection_compensation[selection]);
                }
            }
        }
    }
    free(residue_compensation);
    if (total_sasa != NULL) *total_sasa = total;
    return FASTSASA_SUCCESS;
}

/* Mirrors the CUDA path's tri-state per-input precision override: positive
 * forces FP64, negative forces FP32, zero uses the context precision. */
static int
vulkan_use_fp64(const fastsasa_context *context, const fastsasa_sr_input *input)
{
    if (input->force_double_precision > 0) return 1;
    if (input->force_double_precision < 0) return 0;
    return context->precision == FASTSASA_PRECISION_FP64;
}

static int
run_vulkan(fastsasa_context *context,
           const fastsasa_sr_input *input,
           double *sasa,
           double *total_sasa,
           int lee_richards)
{
    uint32_t center_count = 0;
    int compact_centers = 0;
    int status = prepare_vulkan_input(context, input, !lee_richards,
                                      &center_count, &compact_centers);
    int vk_status;
    int use_fp64;

    if (status != FASTSASA_SUCCESS) return status;
    if (center_count == 0u) {
        return finish_vulkan_result(context, input, 0u, compact_centers,
                                    sasa, total_sasa);
    }

    use_fp64 = vulkan_use_fp64(context, input);
    if (lee_richards) {
        if (compact_centers) {
            vk_status = fastsasa_vk_lee_richards_centers(
                context->vulkan, context->vk_xyz, context->vk_radii,
                (uint32_t)input->n_atoms, (uint32_t)input->n_points, 0.0,
                context->vk_centers, center_count, use_fp64,
                context->vk_areas);
        } else {
            vk_status = fastsasa_vk_lee_richards(
                context->vulkan, context->vk_xyz, context->vk_radii,
                (uint32_t)input->n_atoms, (uint32_t)input->n_points, 0.0,
                use_fp64, context->vk_areas);
        }
    } else if (compact_centers) {
        vk_status = fastsasa_vk_sr_centers(
            context->vulkan, context->vk_xyz, context->vk_radii,
            (uint32_t)input->n_atoms, context->vk_points,
            (uint32_t)input->n_points, 0.0, context->vk_centers,
            center_count, use_fp64, context->vk_areas);
    } else {
        vk_status = fastsasa_vk_sr(
            context->vulkan, context->vk_xyz, context->vk_radii,
            (uint32_t)input->n_atoms, context->vk_points,
            (uint32_t)input->n_points, 0.0, use_fp64, context->vk_areas);
    }
    if (vk_status != 0) {
        set_api_error(fastsasa_vk_last_error(context->vulkan));
        return FASTSASA_VULKAN_ERROR;
    }
    return finish_vulkan_result(context, input, center_count, compact_centers,
                                sasa, total_sasa);
}

int
fastsasa_context_vulkan_frames(fastsasa_context *context,
                             const fastsasa_sr_input *input,
                             const double *frame_x,
                             const double *frame_y,
                             const double *frame_z,
                             int n_frames,
                             int lee_richards,
                             double *atom_sasa_frames,
                             double *total_sasa_frames,
                             double *residue_sasa_frames,
                             double *selection_sasa_frames)
{
    fastsasa_sr_input first_frame;
    uint32_t center_count = 0;
    int compact_centers = 0;
    int status;
    int vk_status;
    const size_t atom_count = input != NULL ? (size_t)input->n_atoms : 0u;

    if (context == NULL || input == NULL || frame_x == NULL ||
        frame_y == NULL || frame_z == NULL || n_frames <= 0 ||
        context->backend != FASTSASA_BACKEND_VULKAN) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    first_frame = *input;
    first_frame.xyz = NULL;
    first_frame.x = frame_x;
    first_frame.y = frame_y;
    first_frame.z = frame_z;
    status = prepare_vulkan_input(context, &first_frame, !lee_richards,
                                  &center_count, &compact_centers);
    if (status != FASTSASA_SUCCESS) return status;
    if (center_count == 0u) {
        if (atom_sasa_frames != NULL) {
            memset(atom_sasa_frames, 0,
                   sizeof(double) * atom_count * (size_t)n_frames);
        }
        if (total_sasa_frames != NULL) {
            memset(total_sasa_frames, 0, sizeof(double) * (size_t)n_frames);
        }
        if (residue_sasa_frames != NULL && input->n_residues > 0) {
            memset(residue_sasa_frames, 0,
                   sizeof(double) * (size_t)input->n_residues * (size_t)n_frames);
        }
        if (selection_sasa_frames != NULL && input->n_selections > 0) {
            memset(selection_sasa_frames, 0,
                   sizeof(double) * (size_t)input->n_selections * (size_t)n_frames);
        }
        return FASTSASA_SUCCESS;
    }

    status = reserve_double(&context->vk_xyz, &context->vk_xyz_capacity,
                           atom_count * (size_t)n_frames * 3u);
    if (status != FASTSASA_SUCCESS) return status;
    status = reserve_double(&context->vk_areas, &context->vk_area_capacity,
                           (size_t)center_count * (size_t)n_frames);
    if (status != FASTSASA_SUCCESS) return status;
    for (int frame = 0; frame < n_frames; ++frame) {
        const size_t frame_offset = (size_t)frame * atom_count;
        const int shift = !vulkan_use_fp64(context, input);
        const double origin_x = shift ? frame_x[frame_offset] : 0.0;
        const double origin_y = shift ? frame_y[frame_offset] : 0.0;
        const double origin_z = shift ? frame_z[frame_offset] : 0.0;
        for (size_t atom = 0; atom < atom_count; ++atom) {
            const size_t source = frame_offset + atom;
            const size_t target = 3u * source;
            context->vk_xyz[target] = frame_x[source] - origin_x;
            context->vk_xyz[target + 1u] = frame_y[source] - origin_y;
            context->vk_xyz[target + 2u] = frame_z[source] - origin_z;
        }
    }

    {
        const int use_fp64 = vulkan_use_fp64(context, input);

        if (lee_richards) {
            if (compact_centers) {
                vk_status = fastsasa_vk_lee_richards_center_frames(
                    context->vulkan, context->vk_xyz, context->vk_radii,
                    (uint32_t)n_frames, (uint32_t)input->n_atoms,
                    (uint32_t)input->n_points, 0.0,
                    context->vk_centers, center_count, use_fp64,
                    context->vk_areas);
            } else {
                vk_status = fastsasa_vk_lee_richards_frames(
                    context->vulkan, context->vk_xyz, context->vk_radii,
                    (uint32_t)n_frames, (uint32_t)input->n_atoms,
                    (uint32_t)input->n_points, 0.0, use_fp64,
                    context->vk_areas);
            }
        } else if (compact_centers) {
            vk_status = fastsasa_vk_sr_center_frames(
                context->vulkan, context->vk_xyz, context->vk_radii,
                (uint32_t)n_frames, (uint32_t)input->n_atoms,
                context->vk_points, (uint32_t)input->n_points, 0.0,
                context->vk_centers, center_count, use_fp64,
                context->vk_areas);
        } else {
            vk_status = fastsasa_vk_sr_frames(
                context->vulkan, context->vk_xyz, context->vk_radii,
                (uint32_t)n_frames, (uint32_t)input->n_atoms,
                context->vk_points, (uint32_t)input->n_points, 0.0, use_fp64,
                context->vk_areas);
        }
    }
    if (vk_status != 0) {
        set_api_error(fastsasa_vk_last_error(context->vulkan));
        return FASTSASA_VULKAN_ERROR;
    }

    double *residue_compensation = NULL;

    if (residue_sasa_frames != NULL && input->n_residues > 0) {
        residue_compensation = (double *)calloc((size_t)input->n_residues, sizeof(double));
        if (residue_compensation == NULL) return FASTSASA_MEMORY_ERROR;
    }
    for (int frame = 0; frame < n_frames; ++frame) {
        const double *areas = context->vk_areas + (size_t)frame * center_count;
        double *atom_output = atom_sasa_frames != NULL
                                  ? atom_sasa_frames + (size_t)frame * atom_count
                                  : NULL;
        double total_compensation = 0.0;
        double selection_compensation[32];
        double *residue_output = residue_sasa_frames != NULL
                                     ? residue_sasa_frames +
                                           (size_t)frame * (size_t)input->n_residues
                                     : NULL;
        double *selection_output = selection_sasa_frames != NULL
                                       ? selection_sasa_frames +
                                             (size_t)frame * (size_t)input->n_selections
                                       : NULL;
        double total = 0.0;

        memset(selection_compensation, 0, sizeof(selection_compensation));
        if (atom_output != NULL) {
            memset(atom_output, 0, sizeof(double) * atom_count);
        }
        if (residue_output != NULL) {
            memset(residue_output, 0,
                   sizeof(double) * (size_t)input->n_residues);
            memset(residue_compensation, 0,
                   sizeof(double) * (size_t)input->n_residues);
        }
        if (selection_output != NULL) {
            memset(selection_output, 0,
                   sizeof(double) * (size_t)input->n_selections);
        }
        for (uint32_t center = 0; center < center_count; ++center) {
            const uint32_t atom = compact_centers
                                      ? context->vk_centers[center]
                                      : center;
            const double area = (double)areas[center];

            if (atom_output != NULL) atom_output[atom] = area;
            fastsasa_exact_kahan_add(area, &total, &total_compensation);
            if (residue_output != NULL && input->residue_ids != NULL) {
                const int residue = input->residue_ids[atom];
                if (residue >= 0 && residue < input->n_residues) {
                    fastsasa_exact_kahan_add(area, &residue_output[residue],
                                           &residue_compensation[residue]);
                }
            }
            if (selection_output != NULL && input->selection_masks != NULL &&
                input->n_selections <= 31) {
                const unsigned int mask = input->selection_masks[atom];
                for (int selection = 0; selection < input->n_selections; ++selection) {
                    if ((mask & (1u << selection)) != 0u) {
                        fastsasa_exact_kahan_add(area, &selection_output[selection],
                                               &selection_compensation[selection]);
                    }
                }
            }
        }
        if (total_sasa_frames != NULL) total_sasa_frames[frame] = total;
    }
    free(residue_compensation);
    return FASTSASA_SUCCESS;
}
#endif

#ifndef FASTSASA_HAVE_VULKAN
int
fastsasa_context_vulkan_frames(fastsasa_context *context,
                             const fastsasa_sr_input *input,
                             const double *frame_x,
                             const double *frame_y,
                             const double *frame_z,
                             int n_frames,
                             int lee_richards,
                             double *atom_sasa_frames,
                             double *total_sasa_frames,
                             double *residue_sasa_frames,
                             double *selection_sasa_frames)
{
    (void)context;
    (void)input;
    (void)frame_x;
    (void)frame_y;
    (void)frame_z;
    (void)n_frames;
    (void)lee_richards;
    (void)atom_sasa_frames;
    (void)total_sasa_frames;
    (void)residue_sasa_frames;
    (void)selection_sasa_frames;
    return FASTSASA_NO_DEVICE;
}
#endif

int
fastsasa_context_create(fastsasa_context **output)
{
    const char *request = backend_request();
    fastsasa_context *context;
    int vulkan_attempted = 0;

    if (output == NULL) return FASTSASA_INVALID_ARGUMENT;
    *output = NULL;
    if (!backend_is(request, "auto") && !backend_is(request, "cuda") &&
        !backend_is(request, "vulkan") && !backend_is(request, "cpu")) {
        set_api_error("FASTSASA_BACKEND must be auto, vulkan, cuda, or cpu");
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (backend_is(request, "cpu")) {
        set_api_error("CPU backend explicitly requested");
        return FASTSASA_NO_DEVICE;
    }

    context = (fastsasa_context *)calloc(1, sizeof(*context));
    if (context == NULL) return FASTSASA_MEMORY_ERROR;
    context->precision = FASTSASA_PRECISION_FP64;

    /* Vulkan first: it runs on NVIDIA, AMD, and Intel GPUs and is at least
     * as fast as CUDA on every backend/precision combination measured, so
     * it is the default rather than a fallback. CUDA remains available for
     * pinned NVIDIA/HPC deployments via --backend cuda. */
#ifdef FASTSASA_HAVE_VULKAN
    if (backend_is(request, "auto") || backend_is(request, "vulkan")) {
        vulkan_attempted = 1;
        if (fastsasa_vk_context_create(&context->vulkan, -1) == 0) {
            context->backend = FASTSASA_BACKEND_VULKAN;
            *output = context;
            set_api_error("");
            return FASTSASA_SUCCESS;
        }
        if (backend_is(request, "vulkan")) {
            const char *reason = fastsasa_vk_create_error();

            if (reason == NULL || reason[0] == '\0') {
                reason = "failed to initialize a Vulkan compute device";
            }
            set_api_error(reason);
            free(context);
            return FASTSASA_NO_DEVICE;
        }
    }
#else
    if (backend_is(request, "vulkan")) {
        set_api_error("FastSASA was built without the Vulkan backend");
        free(context);
        return FASTSASA_NO_DEVICE;
    }
#endif

    if ((backend_is(request, "auto") || backend_is(request, "cuda")) &&
        !test_switch_enabled("FASTSASA_TEST_DISABLE_CUDA")) {
        int cuda_status = fastsasa_device_context_create(&context->cuda);

        if (cuda_status == FASTSASA_SUCCESS) {
            context->backend = FASTSASA_BACKEND_CUDA;
            *output = context;
            set_api_error("");
            return FASTSASA_SUCCESS;
        }
        if (backend_is(request, "cuda")) {
            free(context);
            return cuda_status;
        }
#ifdef FASTSASA_HAVE_VULKAN
        if (backend_is(request, "auto") && vulkan_attempted) {
            const char *reason = fastsasa_vk_create_error();
            char message[512];

            if (reason == NULL || reason[0] == '\0') {
                reason = "failed to initialize a Vulkan compute device";
            }
            snprintf(message, sizeof(message),
                     "no Vulkan device is available (%s) and the CUDA "
                     "fallback failed", reason);
            set_api_error(message);
        }
#else
        (void)vulkan_attempted;
#endif
        free(context);
        return cuda_status;
    }

    free(context);
    return FASTSASA_NO_DEVICE;
}

int
fastsasa_context_set_precision(fastsasa_context *context, int precision)
{
    if (context == NULL ||
        (precision != FASTSASA_PRECISION_FP64 &&
         precision != FASTSASA_PRECISION_FP32)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
#ifdef FASTSASA_HAVE_VULKAN
    if (context->backend == FASTSASA_BACKEND_VULKAN &&
        precision == FASTSASA_PRECISION_FP64 &&
        !fastsasa_vk_supports_fp64(context->vulkan)) {
        set_api_error("Vulkan device does not support shaderFloat64; select FP32 or use the CPU backend");
        return FASTSASA_DEVICE_UNSUPPORTED;
    }
#endif
    context->precision = precision;
    return FASTSASA_SUCCESS;
}

int
fastsasa_context_precision(const fastsasa_context *context)
{
    return context == NULL ? FASTSASA_PRECISION_FP64 : context->precision;
}

void
fastsasa_context_free(fastsasa_context *context)
{
    if (context == NULL) return;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        fastsasa_device_context_free(context->cuda);
    }
#ifdef FASTSASA_HAVE_VULKAN
    else {
        fastsasa_vk_context_free(context->vulkan);
        free(context->vk_xyz);
        free(context->vk_radii);
        free(context->vk_points);
        free(context->vk_areas);
        free(context->vk_centers);
    }
#endif
    free(context);
}

const char *
fastsasa_context_backend(const fastsasa_context *context)
{
    if (context == NULL) return "none";
    return context->backend == FASTSASA_BACKEND_CUDA ? "cuda" : "vulkan";
}

int
fastsasa_context_synchronize(fastsasa_context *context)
{
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        return fastsasa_device_context_synchronize(context->cuda);
    }
    return FASTSASA_SUCCESS;
}

int
fastsasa_context_enable_profile(fastsasa_context *context, int enabled)
{
    if (context == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        return fastsasa_device_context_enable_profile(context->cuda, enabled);
    }
    (void)enabled;
    return FASTSASA_SUCCESS;
}

int
fastsasa_context_last_cell_profile(fastsasa_context *context,
                                 fastsasa_cell_profile *profile)
{
    if (context == NULL || profile == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        return fastsasa_device_context_last_cell_profile(
            context->cuda, (fastsasa_device_cell_profile *)profile);
    }
    memset(profile, 0, sizeof(*profile));
    return FASTSASA_SUCCESS;
}

int fastsasa_host_alloc(void **ptr, size_t bytes)
{
    return fastsasa_device_host_alloc(ptr, bytes);
}

void fastsasa_host_free(void *ptr)
{
    fastsasa_device_host_free(ptr);
}

int
fastsasa_context_shrake_rupley_csr(fastsasa_context *context,
                                 const fastsasa_sr_input *input,
                                 double *sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_shrake_rupley_csr(context->cuda,
                                                       &device_input, sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, sasa, NULL, 0);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

/*
 * The contextless entry points used to create and destroy a full backend
 * context per call, which costs device initialization every time. They now
 * share one cached context, serialized by a mutex because contexts are not
 * thread safe. The cache is invalidated when FASTSASA_BACKEND changes so
 * in-process backend switches keep working; the context is held for the
 * process lifetime.
 */
static fastsasa_mutex oneshot_lock = FASTSASA_MUTEX_INIT;
static fastsasa_context *oneshot_context = NULL;
static char oneshot_backend[32];

static int
oneshot_context_locked(fastsasa_context **context)
{
    const char *request = backend_request();
    int status = FASTSASA_SUCCESS;

    if (oneshot_context != NULL &&
        strncmp(oneshot_backend, request, sizeof(oneshot_backend) - 1u) != 0) {
        fastsasa_context_free(oneshot_context);
        oneshot_context = NULL;
    }
    if (oneshot_context == NULL) {
        status = fastsasa_context_create(&oneshot_context);
        if (status == FASTSASA_SUCCESS) {
            snprintf(oneshot_backend, sizeof(oneshot_backend), "%s", request);
        }
    }
    *context = oneshot_context;
    return status;
}

int
fastsasa_shrake_rupley_csr(const fastsasa_sr_input *input, double *sasa)
{
    fastsasa_context *context = NULL;
    int status;

    if (input == NULL) return FASTSASA_INVALID_ARGUMENT;
    fastsasa_mutex_lock(&oneshot_lock);
    status = oneshot_context_locked(&context);
    if (status == FASTSASA_SUCCESS) {
        status = fastsasa_context_shrake_rupley_csr(context, input, sasa);
    }
    fastsasa_mutex_unlock(&oneshot_lock);
    return status;
}

int
fastsasa_context_shrake_rupley_cell_list(fastsasa_context *context,
                                       const fastsasa_sr_input *input,
                                       double *sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_shrake_rupley_cell_list(
            context->cuda, &device_input, sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, sasa, NULL, 0);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_context_shrake_rupley_exposed_points_cell_list(fastsasa_context *context,
                                                       const fastsasa_sr_input *input,
                                                       unsigned char *exposed)
{
    if (context == NULL || input == NULL || exposed == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        /* No CUDA surface-point export kernel this round; callers fall back
         * to the CPU implementation for a CUDA-backed context. */
        return FASTSASA_NO_DEVICE;
    }
#ifdef FASTSASA_HAVE_VULKAN
    if (input->n_atoms <= 0 || input->n_points <= 0 || input->radii == NULL ||
        input->test_points == NULL ||
        (input->xyz == NULL && (input->x == NULL || input->y == NULL || input->z == NULL))) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!fastsasa_vk_supports_fp64(context->vulkan)) {
        set_api_error("Vulkan device does not support shaderFloat64; surface-point export "
                     "falls back to the CPU backend");
        return FASTSASA_DEVICE_UNSUPPORTED;
    }
    {
        const int vk_status = fastsasa_vk_sr_exposed_points(
            context->vulkan, input->xyz, input->x, input->y, input->z,
            input->radii, (uint32_t)input->n_atoms, input->test_points,
            (uint32_t)input->n_points, 0.0, exposed);
        if (vk_status != 0) {
            set_api_error(fastsasa_vk_last_error(context->vulkan));
            return FASTSASA_VULKAN_ERROR;
        }
    }
    return FASTSASA_SUCCESS;
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_context_shrake_rupley_cell_list_async(fastsasa_context *context,
                                             const fastsasa_sr_input *input,
                                             double *sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_shrake_rupley_cell_list_async(
            context->cuda, &device_input, sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, sasa, NULL, 0);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_context_shrake_rupley_cell_list_total(fastsasa_context *context,
                                             const fastsasa_sr_input *input,
                                             double *total_sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL || total_sasa == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_shrake_rupley_cell_list_total(
            context->cuda, &device_input, total_sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, NULL, total_sasa, 0);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_context_shrake_rupley_cell_list_total_async(fastsasa_context *context,
                                                   const fastsasa_sr_input *input,
                                                   double *total_sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL || total_sasa == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_shrake_rupley_cell_list_total_async(
            context->cuda, &device_input, total_sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, NULL, total_sasa, 0);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_context_lee_richards(fastsasa_context *context,
                            const fastsasa_sr_input *input,
                            double *sasa)
{
    fastsasa_device_sr_input device_input;
    if (context == NULL || input == NULL || sasa == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (context->backend == FASTSASA_BACKEND_CUDA) {
        copy_input(&device_input, input, context->precision);
        return fastsasa_device_context_lee_richards(context->cuda,
                                                  &device_input, sasa);
    }
#ifdef FASTSASA_HAVE_VULKAN
    return run_vulkan(context, input, sasa, NULL, 1);
#else
    return FASTSASA_NO_DEVICE;
#endif
}

int
fastsasa_check_device(void)
{
    fastsasa_context *context = NULL;
    int status;

    fastsasa_mutex_lock(&oneshot_lock);
    status = oneshot_context_locked(&context);
    fastsasa_mutex_unlock(&oneshot_lock);
    return status;
}

unsigned int fastsasa_abi_version(void) { return FASTSASA_ABI_VERSION; }
size_t fastsasa_sizeof_sr_input(void) { return sizeof(fastsasa_sr_input); }
size_t fastsasa_offsetof_sr_input_active_center_mask(void) { return offsetof(fastsasa_sr_input, active_center_mask); }
size_t fastsasa_offsetof_sr_input_active_center_indices(void) { return offsetof(fastsasa_sr_input, active_center_indices); }
size_t fastsasa_offsetof_sr_input_n_active_centers(void) { return offsetof(fastsasa_sr_input, n_active_centers); }
size_t fastsasa_offsetof_sr_input_force_double_precision(void) { return offsetof(fastsasa_sr_input, force_double_precision); }

int
fastsasa_constant_test_point_limit(void)
{
    return backend_is(backend_request(), "vulkan")
               ? 0
               : fastsasa_device_constant_test_point_limit();
}

int
fastsasa_recommended_trajectory_batch_size(int n_atoms, int n_frames,
                                         int n_points, int selection_only)
{
    if (backend_is(backend_request(), "vulkan")) {
        return n_frames > 0 && n_frames < 8 ? n_frames : 8;
    }
    return fastsasa_device_recommended_trajectory_batch_size(
        n_atoms, n_frames, n_points, selection_only);
}

int
fastsasa_recommended_parallel_frames(int n_atoms, int n_points,
                                   int batch_size, int selection_only)
{
    if (backend_is(backend_request(), "vulkan")) return 1;
    return fastsasa_device_recommended_parallel_frames(
        n_atoms, n_points, batch_size, selection_only);
}

const char *
fastsasa_status_string(int status)
{
    if (status == FASTSASA_VULKAN_ERROR) return "Vulkan error";
    return fastsasa_device_status_string(status);
}

const char *
fastsasa_last_error(void)
{
    return api_error[0] != '\0' ? api_error : fastsasa_device_last_error();
}

int
fastsasa_sum_atoms(const double *atom_sasa, int n_atoms, double *total)
{
    double sum = 0.0;
    double compensation = 0.0;

    if (atom_sasa == NULL || total == NULL || n_atoms < 0) return FASTSASA_INVALID_ARGUMENT;
    for (int atom = 0; atom < n_atoms; ++atom) {
        fastsasa_exact_kahan_add(atom_sasa[atom], &sum, &compensation);
    }
    *total = sum;
    return FASTSASA_SUCCESS;
}

int
fastsasa_sum_residues(const double *atom_sasa,
                    const int *residue_ids,
                    int n_atoms,
                    int n_residues,
                    double *residue_sasa)
{
    double *compensation;

    if (atom_sasa == NULL || residue_ids == NULL || residue_sasa == NULL ||
        n_atoms < 0 || n_residues <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    compensation = (double *)calloc((size_t)n_residues, sizeof(double));
    if (compensation == NULL) return FASTSASA_MEMORY_ERROR;
    memset(residue_sasa, 0, sizeof(double) * (size_t)n_residues);
    for (int atom = 0; atom < n_atoms; ++atom) {
        const int residue = residue_ids[atom];

        if (residue >= 0 && residue < n_residues) {
            fastsasa_exact_kahan_add(atom_sasa[atom], &residue_sasa[residue], &compensation[residue]);
        }
    }
    free(compensation);
    return FASTSASA_SUCCESS;
}

int
fastsasa_sum_selections(const double *atom_sasa,
                      const unsigned int *selection_masks,
                      int n_atoms,
                      int n_selections,
                      double *selection_sasa)
{
    double compensation[32];

    if (atom_sasa == NULL || selection_masks == NULL || selection_sasa == NULL ||
        n_atoms < 0 || n_selections <= 0 || n_selections > 31) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    memset(compensation, 0, sizeof(compensation));
    memset(selection_sasa, 0, sizeof(double) * (size_t)n_selections);
    for (int atom = 0; atom < n_atoms; ++atom) {
        for (int selection = 0; selection < n_selections; ++selection) {
            if ((selection_masks[atom] & (1u << selection)) != 0u) {
                fastsasa_exact_kahan_add(atom_sasa[atom], &selection_sasa[selection],
                                       &compensation[selection]);
            }
        }
    }
    return FASTSASA_SUCCESS;
}
