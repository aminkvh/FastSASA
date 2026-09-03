#include "fastsasa_trajectory.h"
#include "fastsasa_backend_internal.h"

#include <math.h>
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
            *algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS);
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

static fastsasa_context **
trajectory_context_lanes_new(fastsasa_context *primary,
                             int n_lanes)
{
    fastsasa_context **lanes;

    if (primary == NULL || n_lanes <= 1) return NULL;
    lanes = (fastsasa_context **)calloc((size_t)n_lanes, sizeof(fastsasa_context *));
    if (lanes == NULL) return NULL;
    lanes[0] = primary;
    for (int lane = 1; lane < n_lanes; ++lane) {
        if (fastsasa_context_create(&lanes[lane]) != FASTSASA_SUCCESS) {
            for (int cleanup = 1; cleanup < lane; ++cleanup) {
                fastsasa_context_free(lanes[cleanup]);
            }
            free(lanes);
            return NULL;
        }
        if (fastsasa_context_set_precision(
                lanes[lane], fastsasa_context_precision(primary)) != FASTSASA_SUCCESS) {
            fastsasa_context_free(lanes[lane]);
            for (int cleanup = 1; cleanup < lane; ++cleanup) {
                fastsasa_context_free(lanes[cleanup]);
            }
            free(lanes);
            return NULL;
        }
    }
    return lanes;
}

static void
trajectory_context_lanes_free(fastsasa_context **lanes,
                              int n_lanes)
{
    if (lanes == NULL) return;
    for (int lane = 1; lane < n_lanes; ++lane) {
        fastsasa_context_free(lanes[lane]);
    }
    free(lanes);
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
    if (residue_sasa_frames != NULL || algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) {
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
            fastsasa_context **lanes = trajectory_context_lanes_new(context, n_lanes);

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
                trajectory_context_lanes_free(lanes, n_lanes);
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

        if (algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) {
            frame_sasa = atom_sasa_frames != NULL
                             ? atom_sasa_frames + (size_t)frame * (size_t)topology->n_atoms
                             : scratch_sasa;
            status = fastsasa_context_lee_richards(context, &input, frame_sasa);
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
                      algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY &&
                      use_selected_center_optimization();
    need_scratch_sasa = algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS ||
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

    if (selected_center && frames->n_frames > 1) {
        int n_lanes = fastsasa_recommended_parallel_frames(topology->n_atoms,
                                                         n_points,
                                                         frames->n_frames,
                                                         1);
        if (strcmp(fastsasa_context_backend(context), "vulkan") == 0) n_lanes = 1;
        if (n_lanes > frames->n_frames) n_lanes = frames->n_frames;
        if (n_lanes > 1) {
            fastsasa_context **lanes = trajectory_context_lanes_new(context, n_lanes);

            if (lanes != NULL) {
                for (int frame = 0; frame < frames->n_frames && status == FASTSASA_SUCCESS; frame += n_lanes) {
                    const int chunk = frames->n_frames - frame < n_lanes ? frames->n_frames - frame : n_lanes;
                    int submitted = 0;

                    for (int lane = 0; lane < chunk; ++lane) {
                        const int frame_index = frame + lane;

                        input.x = frames->x + (size_t)frame_index * (size_t)topology->n_atoms;
                        input.y = frames->y + (size_t)frame_index * (size_t)topology->n_atoms;
                        input.z = frames->z + (size_t)frame_index * (size_t)topology->n_atoms;
                        input.selection_sasa = selection_sasa_frames + (size_t)frame_index * (size_t)n_selections;
                        status = fastsasa_context_shrake_rupley_cell_list_async(lanes[lane],
                                                                              &input,
                                                                              NULL);
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
                trajectory_context_lanes_free(lanes, n_lanes);
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
        if (algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) {
            status = fastsasa_context_lee_richards(context, &input, scratch_sasa);
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
