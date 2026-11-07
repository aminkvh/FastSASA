#include "fastsasa_trajectory.h"
#include "fastsasa_backend_internal.h"
#include "fastsasa_device.h"
#include "fastsasa_portable.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double FASTSASA_TRAJ_PI = 3.141592653589793238462643383279502884;

static int
trajectory_parameters(const fastsasa_parameters *parameters,
                      double *probe_radius,
                      int *n_points,
                      int *algorithm)
{
    *probe_radius = parameters != NULL ? parameters->probe_radius : 1.4;
    *n_points = parameters != NULL ? parameters->n_points : 100;
    *algorithm = parameters != NULL ? parameters->algorithm : FASTSASA_ALGORITHM_SHRAKE_RUPLEY;
    return *probe_radius >= 0.0 && *n_points > 0 &&
           (*algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY ||
            *algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS ||
            *algorithm == FASTSASA_ALGORITHM_EXACT);
}

static double *
expanded_radii_new(const double *radii,
                   int n_atoms,
                   double probe_radius)
{
    double *expanded = (double *)malloc(sizeof(double) * (size_t)n_atoms);

    if (expanded == NULL) return NULL;
    for (int atom = 0; atom < n_atoms; ++atom) expanded[atom] = radii[atom] + probe_radius;
    return expanded;
}

static double *
test_points_new(int n_points)
{
    const double dlong = FASTSASA_TRAJ_PI * (3.0 - sqrt(5.0));
    const double dz = 2.0 / (double)n_points;
    double longitude = 0.0;
    double z = 1.0 - dz / 2.0;
    double *points = (double *)malloc(sizeof(double) * 3u * (size_t)n_points);

    if (points == NULL) return NULL;
    for (int i = 0; i < n_points; ++i) {
        const double r = sqrt(1.0 - z * z);
        points[3 * i] = cos(longitude) * r;
        points[3 * i + 1] = sin(longitude) * r;
        points[3 * i + 2] = z;
        z -= dz;
        longitude += dlong;
    }
    return points;
}

static int
validate_trajectory_input(const fastsasa_topology *topology,
                          const fastsasa_soa_frames *frames,
                          const fastsasa_parameters *parameters,
                          double *probe_radius,
                          int *n_points,
                          int *algorithm)
{
    if (topology == NULL || frames == NULL) return FASTSASA_INVALID_ARGUMENT;
    if (topology->radii == NULL || topology->n_atoms <= 0) return FASTSASA_INVALID_ARGUMENT;
    if (frames->x == NULL || frames->y == NULL || frames->z == NULL ||
        frames->n_frames <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!trajectory_parameters(parameters, probe_radius, n_points, algorithm)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    return FASTSASA_SUCCESS;
}

static int
kahan_add(double value,
          double *sum,
          double *compensation)
{
    const double corrected = value - *compensation;
    const double next = *sum + corrected;

    *compensation = (next - *sum) - corrected;
    *sum = next;
    return FASTSASA_SUCCESS;
}

static int
sum_atoms(const double *atom_sasa,
          int n_atoms,
          double *total)
{
    double compensation = 0.0;

    *total = 0.0;
    for (int atom = 0; atom < n_atoms; ++atom) {
        kahan_add(atom_sasa[atom], total, &compensation);
    }
    return FASTSASA_SUCCESS;
}

static int
sum_residues(const double *atom_sasa,
             const int *residue_ids,
             int n_atoms,
             int n_residues,
             double *residue_sasa)
{
    double *compensation;

    if (residue_ids == NULL || residue_sasa == NULL || n_residues <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    compensation = (double *)calloc((size_t)n_residues, sizeof(double));
    if (compensation == NULL) return FASTSASA_MEMORY_ERROR;
    memset(residue_sasa, 0, sizeof(double) * (size_t)n_residues);
    for (int atom = 0; atom < n_atoms; ++atom) {
        const int residue = residue_ids[atom];
        if (residue >= 0 && residue < n_residues) {
            kahan_add(atom_sasa[atom], &residue_sasa[residue], &compensation[residue]);
        }
    }
    free(compensation);
    return FASTSASA_SUCCESS;
}

static int
sum_selections(const double *atom_sasa,
               const unsigned int *selection_masks,
               int n_atoms,
               int n_selections,
               double *selection_sasa)
{
    double *compensation;

    if (selection_masks == NULL || selection_sasa == NULL ||
        n_selections <= 0 || n_selections > 31) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    compensation = (double *)calloc((size_t)n_selections, sizeof(double));
    if (compensation == NULL) return FASTSASA_MEMORY_ERROR;
    memset(selection_sasa, 0, sizeof(double) * (size_t)n_selections);
    for (int atom = 0; atom < n_atoms; ++atom) {
        for (int selection = 0; selection < n_selections; ++selection) {
            if (selection_masks[atom] & (1u << selection)) {
                kahan_add(atom_sasa[atom],
                          &selection_sasa[selection],
                          &compensation[selection]);
            }
        }
    }
    free(compensation);
    return FASTSASA_SUCCESS;
}

static int
use_selected_center_optimization(void)
{
    const char *value = getenv("FASTSASA_SELECTED_CENTER");

    return value == NULL || value[0] == '\0' ||
           (strcmp(value, "0") != 0 &&
            strcmp(value, "false") != 0 &&
            strcmp(value, "FALSE") != 0);
}


int
fastsasa_context_calc_trajectory_soa(fastsasa_context *context,
                                   const fastsasa_topology *topology,
                                   const fastsasa_soa_frames *frames,
                                   const fastsasa_parameters *parameters,
                                   double *total_sasa,
                                   double *atom_sasa_frames,
                                   double *residue_sasa_frames)
{
    fastsasa_sr_input input;
    double probe_radius;
    int n_points;
    int algorithm;
    double *expanded_radii;
    double *test_points;
    double *scratch_sasa = NULL;
    int status;

    if (context == NULL || total_sasa == NULL) return FASTSASA_INVALID_ARGUMENT;
    status = validate_trajectory_input(topology, frames, parameters, &probe_radius, &n_points, &algorithm);
    if (status != FASTSASA_SUCCESS) return status;
    if (atom_sasa_frames != NULL && residue_sasa_frames != NULL) return FASTSASA_INVALID_ARGUMENT;
    if (residue_sasa_frames != NULL &&
        (topology->residue_ids == NULL || topology->n_residues <= 0)) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    expanded_radii = expanded_radii_new(topology->radii, topology->n_atoms, probe_radius);
    test_points = algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY ? test_points_new(n_points) : NULL;
    if (expanded_radii == NULL ||
        (algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY && test_points == NULL)) {
        free(expanded_radii);
        free(test_points);
        return FASTSASA_MEMORY_ERROR;
    }
    if (residue_sasa_frames != NULL || algorithm != FASTSASA_ALGORITHM_SHRAKE_RUPLEY) {
        scratch_sasa = (double *)malloc(sizeof(double) * (size_t)topology->n_atoms);
        if (scratch_sasa == NULL) {
            free(expanded_radii);
            free(test_points);
            return FASTSASA_MEMORY_ERROR;
        }
    }

    memset(&input, 0, sizeof(input));
    input.n_atoms = topology->n_atoms;
    input.n_points = n_points;
    input.radii = expanded_radii;
    input.test_points = test_points;
    input.reuse_test_points = 1;
    if (residue_sasa_frames != NULL) {
        input.residue_ids = topology->residue_ids;
        input.n_residues = topology->n_residues;
    }

    if (strcmp(fastsasa_context_backend(context), "vulkan") == 0 &&
        algorithm != FASTSASA_ALGORITHM_EXACT &&
        frames->n_frames > 1) {
        status = fastsasa_context_vulkan_frames(
            context, &input, frames->x, frames->y, frames->z,
            frames->n_frames,
            algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS,
            atom_sasa_frames, total_sasa, residue_sasa_frames, NULL);
        if (status == FASTSASA_SUCCESS) {
            free(scratch_sasa);
            free(test_points);
            free(expanded_radii);
            return status;
        }
        /* Batched Vulkan execution is a throughput path; on any failure the
         * validated per-frame loop below remains the correct fallback. */
        status = FASTSASA_SUCCESS;
    }

    if (algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY &&
        atom_sasa_frames == NULL &&
        residue_sasa_frames == NULL &&
        frames->n_frames > 1) {
        int n_lanes = fastsasa_recommended_parallel_frames(topology->n_atoms,
                                                         n_points,
                                                         frames->n_frames,
                                                         0);
        if (strcmp(fastsasa_context_backend(context), "vulkan") == 0) n_lanes = 1;
        if (n_lanes > frames->n_frames) n_lanes = frames->n_frames;
        if (n_lanes > 1) {
            fastsasa_context **lanes = fastsasa_context_frame_lanes(context, frames->n_frames, &n_lanes);

            if (lanes != NULL) {
                for (int frame = 0; frame < frames->n_frames && status == FASTSASA_SUCCESS; frame += n_lanes) {
                    const int chunk = frames->n_frames - frame < n_lanes ? frames->n_frames - frame : n_lanes;
                    int submitted = 0;

                    for (int lane = 0; lane < chunk; ++lane) {
                        const int frame_index = frame + lane;

                        input.x = frames->x + (size_t)frame_index * (size_t)topology->n_atoms;
                        input.y = frames->y + (size_t)frame_index * (size_t)topology->n_atoms;
                        input.z = frames->z + (size_t)frame_index * (size_t)topology->n_atoms;
                        status = fastsasa_context_shrake_rupley_cell_list_total_async(lanes[lane],
                                                                                    &input,
                                                                                    &total_sasa[frame_index]);
                        if (status != FASTSASA_SUCCESS) break;
                        ++submitted;
                    }
                    for (int lane = 0; lane < submitted; ++lane) {
                        const int synchronize_status = fastsasa_context_synchronize(lanes[lane]);

                        if (status == FASTSASA_SUCCESS && synchronize_status != FASTSASA_SUCCESS) {
                            status = synchronize_status;
                        }
                    }
                }
                free(scratch_sasa);
                free(test_points);
                free(expanded_radii);
                return status;
            }
        }
    }

    for (int frame = 0; frame < frames->n_frames; ++frame) {
        double *frame_sasa = atom_sasa_frames != NULL
                                 ? atom_sasa_frames + (size_t)frame * (size_t)topology->n_atoms
                                 : scratch_sasa;

        input.x = frames->x + (size_t)frame * (size_t)topology->n_atoms;
        input.y = frames->y + (size_t)frame * (size_t)topology->n_atoms;
        input.z = frames->z + (size_t)frame * (size_t)topology->n_atoms;
        input.residue_sasa = residue_sasa_frames != NULL
                                  ? residue_sasa_frames + (size_t)frame * (size_t)topology->n_residues
                                  : NULL;

        if (algorithm != FASTSASA_ALGORITHM_SHRAKE_RUPLEY) {
            frame_sasa = atom_sasa_frames != NULL
                             ? atom_sasa_frames + (size_t)frame * (size_t)topology->n_atoms
                             : scratch_sasa;
            status = algorithm == FASTSASA_ALGORITHM_EXACT
                         ? fastsasa_context_exact(context, &input, frame_sasa)
                         : fastsasa_context_lee_richards(context, &input, frame_sasa);
            if (status == FASTSASA_SUCCESS) {
                status = sum_atoms(frame_sasa, topology->n_atoms, &total_sasa[frame]);
            }
            if (status == FASTSASA_SUCCESS && residue_sasa_frames != NULL) {
                status = sum_residues(frame_sasa,
                                      topology->residue_ids,
                                      topology->n_atoms,
                                      topology->n_residues,
                                      input.residue_sasa);
            }
        } else if (frame_sasa != NULL) {
            status = fastsasa_context_shrake_rupley_cell_list(context, &input, frame_sasa);
            if (status == FASTSASA_SUCCESS) {
                status = sum_atoms(frame_sasa, topology->n_atoms, &total_sasa[frame]);
            }
        } else {
            status = fastsasa_context_shrake_rupley_cell_list_total(context,
                                                                  &input,
                                                                  &total_sasa[frame]);
        }
        if (status != FASTSASA_SUCCESS) break;
    }

    free(scratch_sasa);
    free(test_points);
    free(expanded_radii);
    return status;
}

/* x, y, z blocks as floats with their bounding box; 1 when every value
 * converts exactly (frames read from float trajectory files do), so the
 * device can upload half the bytes and widen them back to the same doubles. */
static int
f32_frame(float *fb, const double *fx, const double *fy, const double *fz, int n, double bbox[6])
{
    int exact = 1;
    float lo[3] = {(float)fx[0], (float)fy[0], (float)fz[0]};
    float hi[3] = {lo[0], lo[1], lo[2]};

    for (int a = 0; a < n; ++a) {
        const float vx = (float)fx[a], vy = (float)fy[a], vz = (float)fz[a];
        fb[a] = vx; fb[n + a] = vy; fb[2 * n + a] = vz;
        exact &= (double)vx == fx[a] && (double)vy == fy[a] && (double)vz == fz[a];
        lo[0] = vx < lo[0] ? vx : lo[0]; hi[0] = vx > hi[0] ? vx : hi[0];
        lo[1] = vy < lo[1] ? vy : lo[1]; hi[1] = vy > hi[1] ? vy : hi[1];
        lo[2] = vz < lo[2] ? vz : lo[2]; hi[2] = vz > hi[2] ? vz : hi[2];
    }
    for (int k = 0; k < 3; ++k) { bbox[k] = lo[k]; bbox[3 + k] = hi[k]; }
    return exact;
}

/* Selected-centre frames on GPU lane contexts. The lanes form n_groups groups
 * of group_size; chunk c holds frames [c * group_size, (c + 1) * group_size)
 * and runs on group c % n_groups, so one group's chunk is prepared and
 * launched while the others are still running on the device. Pipeline depth
 * (groups in flight) is what pays, not frames per launch. Host thread t
 * drives the groups g with g % n_threads == t, so the frames and outputs of
 * the threads never overlap. */
#define FASTSASA_SELECTION_MAX_GROUPS 8
#define FASTSASA_SELECTION_HOST_THREADS 2

typedef struct {
    fastsasa_context **lanes;
    int n_groups, group_size, thread, n_threads, batch;
    const fastsasa_soa_frames *frames;
    int n_atoms, n_selections;
    fastsasa_sr_input input;
    float *f32up;            /* per-lane float frames, or NULL */
    double *staging;         /* per-lane page-locked double frames, or NULL */
    double *selection_sasa_frames;
    int status;
    char error[512];         /* fastsasa_last_error() of this thread when status failed */
} selection_worker;

static void
selection_worker_wait(selection_worker *w, int group, int count)
{
    for (int l = 0; l < count; ++l) {
        const int st = fastsasa_context_synchronize(w->lanes[(size_t)group * (size_t)w->group_size + l]);
        if (w->status == FASTSASA_SUCCESS && st != FASTSASA_SUCCESS) w->status = st;
    }
}

static void *
selection_worker_run(void *arg)
{
    selection_worker *w = (selection_worker *)arg;
    const int n = w->n_atoms;
    const size_t coord_bytes = sizeof(double) * (size_t)n;
    const int n_chunks = (w->frames->n_frames + w->group_size - 1) / w->group_size;
    int pending[FASTSASA_SELECTION_MAX_GROUPS] = {0};

    w->status = FASTSASA_SUCCESS;
    for (int c = 0; c < n_chunks && w->status == FASTSASA_SUCCESS; ++c) {
        const int group = c % w->n_groups;
        if (group % w->n_threads != w->thread) continue;
        fastsasa_context **group_lanes = w->lanes + (size_t)group * (size_t)w->group_size;
        int submitted = 0;

        selection_worker_wait(w, group, pending[group]);   /* this group's previous chunk */
        pending[group] = 0;
        if (w->status != FASTSASA_SUCCESS) break;
        for (int l = 0; l < w->group_size; ++l) {
            const int frame = c * w->group_size + l;
            const int lane = group * w->group_size + l;
            if (frame >= w->frames->n_frames) break;
            const double *fx = w->frames->x + (size_t)frame * (size_t)n;
            const double *fy = w->frames->y + (size_t)frame * (size_t)n;
            const double *fz = w->frames->z + (size_t)frame * (size_t)n;
            w->input.x = fx;
            w->input.y = fy;
            w->input.z = fz;
            if (w->f32up != NULL) {
                float *fb = w->f32up + (size_t)lane * 3u * (size_t)n;
                double bbox[6];
                const int exact = f32_frame(fb, fx, fy, fz, n, bbox);
                fastsasa_context_set_f32_coords(group_lanes[l], exact ? fb : NULL, bbox);
            } else if (w->staging != NULL) {
                double *sx = w->staging + (size_t)lane * 3u * (size_t)n;
                memcpy(sx, fx, coord_bytes);
                memcpy(sx + n, fy, coord_bytes);
                memcpy(sx + 2 * n, fz, coord_bytes);
                w->input.x = sx;
                w->input.y = sx + n;
                w->input.z = sx + 2 * n;
            }
            w->input.selection_sasa = w->selection_sasa_frames + (size_t)frame * (size_t)w->n_selections;
            w->status = fastsasa_context_shrake_rupley_cell_list_async(group_lanes[l], &w->input, NULL);
            if (w->status != FASTSASA_SUCCESS) break;
            ++submitted;
        }
        if (w->batch && w->status == FASTSASA_SUCCESS && submitted > 0) {
            /* every lane stopped before its SR launch: one mask launch for the chunk */
            w->status = fastsasa_context_batched_sr(group_lanes, submitted);
        }
        pending[group] = submitted;
    }
    for (int g = w->thread; g < w->n_groups; g += w->n_threads) selection_worker_wait(w, g, pending[g]);
    if (w->status != FASTSASA_SUCCESS) snprintf(w->error, sizeof(w->error), "%s", fastsasa_last_error());
    return NULL;
}

static int
selection_lanes_run(fastsasa_context *context,
                    fastsasa_context **lanes,
                    int n_lanes,
                    const fastsasa_soa_frames *frames,
                    int n_atoms,
                    const fastsasa_sr_input *input,
                    int n_selections,
                    double *selection_sasa_frames)
{
    const int cuda = strcmp(fastsasa_context_backend(context), "cuda") == 0;
    const int batch = cuda && n_lanes <= 16;
    const size_t coord_bytes = sizeof(double) * (size_t)n_atoms;
    float *f32up = NULL;
    double *staging = NULL;
    selection_worker workers[FASTSASA_SELECTION_HOST_THREADS];
    fastsasa_thread threads[FASTSASA_SELECTION_HOST_THREADS];
    int n_groups = batch ? n_lanes : 1;
    int n_threads = FASTSASA_SELECTION_HOST_THREADS;
    int started = 0;
    int status = FASTSASA_SUCCESS;

    /* Constant inputs stay on the device (reuse flag 2). They are matched by
     * host pointer, so nothing may carry over from an earlier call. */
    for (int l = 0; l < n_lanes; ++l) fastsasa_context_reset_static(lanes[l]);
    if (cuda && fastsasa_device_host_alloc((void **)&f32up, sizeof(float) * 3u * (size_t)n_atoms * (size_t)n_lanes) != FASTSASA_SUCCESS) {
        f32up = NULL;
    }
    if (cuda && f32up == NULL &&
        fastsasa_device_host_alloc((void **)&staging, 3u * coord_bytes * (size_t)n_lanes) != FASTSASA_SUCCESS) {
        staging = NULL;
    }
    if (batch) for (int l = 0; l < n_lanes; ++l) fastsasa_context_set_defer_sr(lanes[l], 1);
    if (n_groups > FASTSASA_SELECTION_MAX_GROUPS) n_groups = FASTSASA_SELECTION_MAX_GROUPS;
    if (n_groups > n_lanes) n_groups = n_lanes;
    if (!batch) n_threads = 1;
    if (n_threads > n_groups) n_threads = n_groups;

    for (int t = 0; t < n_threads; ++t) {
        selection_worker *w = &workers[t];
        w->lanes = lanes; w->n_groups = n_groups; w->group_size = n_lanes / n_groups;
        w->thread = t; w->n_threads = n_threads; w->batch = batch;
        w->frames = frames; w->n_atoms = n_atoms; w->n_selections = n_selections;
        w->input = *input; w->input.reuse_test_points |= 2;
        w->f32up = f32up; w->staging = staging;
        w->selection_sasa_frames = selection_sasa_frames;
        w->status = FASTSASA_SUCCESS;
        w->error[0] = '\0';
    }
    for (int t = 1; t < n_threads; ++t, ++started)
        if (fastsasa_thread_create(&threads[t], selection_worker_run, &workers[t]) != 0) break;
    if (started == n_threads - 1) {
        selection_worker_run(&workers[0]);
    } else {
        workers[0].status = FASTSASA_MEMORY_ERROR;
    }
    for (int t = 1; t <= started; ++t) fastsasa_thread_join(&threads[t]);
    for (int t = 0; t < n_threads; ++t)
        if (status == FASTSASA_SUCCESS && workers[t].status != FASTSASA_SUCCESS) {
            status = workers[t].status;
            if (t > 0) fastsasa_set_last_error(workers[t].error);
        }
    if (batch) for (int l = 0; l < n_lanes; ++l) fastsasa_context_set_defer_sr(lanes[l], 0);
    for (int l = 0; l < n_lanes; ++l) fastsasa_context_reset_static(lanes[l]);   /* before f32up goes */
    fastsasa_device_host_free(f32up);
    fastsasa_device_host_free(staging);
    return status;
}

int
fastsasa_context_calc_trajectory_soa_selection(fastsasa_context *context,
                                             const fastsasa_topology *topology,
                                             const fastsasa_soa_frames *frames,
                                             const unsigned int *selection_masks,
                                             int n_selections,
                                             const fastsasa_parameters *parameters,
                                             double *total_sasa,
                                             double *selection_sasa_frames)
{
    fastsasa_sr_input input;
    double probe_radius;
    int n_points;
    int algorithm;
    double *expanded_radii;
    double *test_points;
    double *scratch_sasa = NULL;
    int *active_center_indices = NULL;
    int n_active_centers = 0;
    int need_scratch_sasa;
    int selected_center;
    int status;

    if (context == NULL || selection_sasa_frames == NULL) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    status = validate_trajectory_input(topology, frames, parameters, &probe_radius, &n_points, &algorithm);
    if (status != FASTSASA_SUCCESS) return status;
    if (selection_masks == NULL || n_selections <= 0 || n_selections > 31) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    expanded_radii = expanded_radii_new(topology->radii, topology->n_atoms, probe_radius);
    test_points = algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY ? test_points_new(n_points) : NULL;
    selected_center = total_sasa == NULL &&
                      algorithm != FASTSASA_ALGORITHM_LEE_RICHARDS &&
                      use_selected_center_optimization();
    need_scratch_sasa = algorithm != FASTSASA_ALGORITHM_SHRAKE_RUPLEY ||
                        total_sasa != NULL ||
                        !selected_center;
    if (need_scratch_sasa) {
        scratch_sasa = (double *)malloc(sizeof(double) * (size_t)topology->n_atoms);
    }
    if (expanded_radii == NULL ||
        (algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY && test_points == NULL) ||
        (need_scratch_sasa && scratch_sasa == NULL)) {
        free(scratch_sasa);
        free(expanded_radii);
        free(test_points);
        return FASTSASA_MEMORY_ERROR;
    }
    if (selected_center) {
        active_center_indices = (int *)malloc(sizeof(int) * (size_t)topology->n_atoms);
        if (active_center_indices == NULL) {
            free(scratch_sasa);
            free(expanded_radii);
            free(test_points);
            return FASTSASA_MEMORY_ERROR;
        }
        for (int atom = 0; atom < topology->n_atoms; ++atom) {
            if (selection_masks[atom] != 0u) {
                active_center_indices[n_active_centers++] = atom;
            }
        }
    }

    memset(&input, 0, sizeof(input));
    input.n_atoms = topology->n_atoms;
    input.n_points = n_points;
    input.radii = expanded_radii;
    input.test_points = test_points;
    input.reuse_test_points = 1;
    input.selection_masks = selection_masks;
    input.n_selections = n_selections;
    if (selected_center) {
        input.active_center_mask = (1u << n_selections) - 1u;
        if (n_active_centers > 0) {
            input.active_center_indices = active_center_indices;
            input.n_active_centers = n_active_centers;
        }
    }

    if (strcmp(fastsasa_context_backend(context), "vulkan") == 0 &&
        algorithm != FASTSASA_ALGORITHM_EXACT &&
        frames->n_frames > 1) {
        status = fastsasa_context_vulkan_frames(
            context, &input, frames->x, frames->y, frames->z,
            frames->n_frames,
            algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS,
            NULL, total_sasa, NULL, selection_sasa_frames);
        if (status == FASTSASA_SUCCESS) {
            free(active_center_indices);
            free(scratch_sasa);
            free(test_points);
            free(expanded_radii);
            return status;
        }
        /* Batched Vulkan execution is a throughput path; on any failure the
         * validated per-frame loop below remains the correct fallback. */
        status = FASTSASA_SUCCESS;
    }

    if (selected_center && algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY && frames->n_frames > 1) {
        int n_lanes = fastsasa_recommended_parallel_frames(topology->n_atoms,
                                                         n_points,
                                                         frames->n_frames,
                                                         1);
        if (strcmp(fastsasa_context_backend(context), "vulkan") == 0) n_lanes = 1;
        if (n_lanes > frames->n_frames) n_lanes = frames->n_frames;
        if (n_lanes > 1) {
            fastsasa_context **lanes = fastsasa_context_frame_lanes(context, frames->n_frames, &n_lanes);

            if (lanes != NULL) {
                status = selection_lanes_run(context, lanes, n_lanes, frames, topology->n_atoms, &input,
                                             n_selections, selection_sasa_frames);
                free(active_center_indices);
                free(scratch_sasa);
                free(test_points);
                free(expanded_radii);
                return status;
            }
        }
    }

    for (int frame = 0; frame < frames->n_frames; ++frame) {
        input.x = frames->x + (size_t)frame * (size_t)topology->n_atoms;
        input.y = frames->y + (size_t)frame * (size_t)topology->n_atoms;
        input.z = frames->z + (size_t)frame * (size_t)topology->n_atoms;
        input.selection_sasa = selection_sasa_frames + (size_t)frame * (size_t)n_selections;
        if (algorithm != FASTSASA_ALGORITHM_SHRAKE_RUPLEY) {
            status = algorithm == FASTSASA_ALGORITHM_EXACT
                         ? fastsasa_context_exact(context, &input, scratch_sasa)
                         : fastsasa_context_lee_richards(context, &input, scratch_sasa);
            if (status == FASTSASA_SUCCESS) {
                status = sum_selections(scratch_sasa,
                                        selection_masks,
                                        topology->n_atoms,
                                        n_selections,
                                        input.selection_sasa);
            }
        } else if (selected_center) {
            status = fastsasa_context_shrake_rupley_cell_list(context, &input, NULL);
        } else {
            status = fastsasa_context_shrake_rupley_cell_list(context, &input, scratch_sasa);
        }
        if (status != FASTSASA_SUCCESS) break;
        if (total_sasa != NULL) {
            status = sum_atoms(scratch_sasa, topology->n_atoms, &total_sasa[frame]);
        }
        if (status != FASTSASA_SUCCESS) break;
    }

    free(active_center_indices);
    free(scratch_sasa);
    free(test_points);
    free(expanded_radii);
    return status;
}
