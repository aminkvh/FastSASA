#ifndef FASTSASA_TRAJECTORY_H
#define FASTSASA_TRAJECTORY_H

#include "fastsasa.h"

#ifdef __cplusplus
extern "C" {
#endif

enum fastsasa_algorithm {
    FASTSASA_ALGORITHM_SHRAKE_RUPLEY = 0,
    FASTSASA_ALGORITHM_LEE_RICHARDS = 1
};

typedef struct fastsasa_parameters {
    double probe_radius;
    int n_points;
    int algorithm;
    /* FASTSASA_PRECISION_FP64 (0, default) or FASTSASA_PRECISION_FP32 (see
     * fastsasa.h). Only affects the CPU Shrake-Rupley path; CPU Lee-Richards
     * is FP64-only regardless of this field. */
    int precision;
} fastsasa_parameters;

typedef struct fastsasa_topology {
    const double *radii;
    const int *residue_ids;
    int n_atoms;
    int n_residues;
} fastsasa_topology;

typedef struct fastsasa_soa_frames {
    const double *x;
    const double *y;
    const double *z;
    int n_frames;
} fastsasa_soa_frames;

int fastsasa_context_calc_trajectory_soa(fastsasa_context *context,
                                       const fastsasa_topology *topology,
                                       const fastsasa_soa_frames *frames,
                                       const fastsasa_parameters *parameters,
                                       double *total_sasa,
                                       double *atom_sasa_frames,
                                       double *residue_sasa_frames);

int fastsasa_context_calc_trajectory_soa_selection(fastsasa_context *context,
                                                 const fastsasa_topology *topology,
                                                 const fastsasa_soa_frames *frames,
                                                 const unsigned int *selection_masks,
                                                 int n_selections,
                                                 const fastsasa_parameters *parameters,
                                                 double *total_sasa,
                                                 double *selection_sasa_frames);

#ifdef __cplusplus
}
#endif

#endif
