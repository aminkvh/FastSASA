#ifndef FASTSASA_BACKEND_INTERNAL_H
#define FASTSASA_BACKEND_INTERNAL_H

#include "fastsasa.h"

int fastsasa_context_vulkan_frames(fastsasa_context *context,
                                 const fastsasa_sr_input *input,
                                 const double *frame_x,
                                 const double *frame_y,
                                 const double *frame_z,
                                 int n_frames,
                                 int lee_richards,
                                 double *atom_sasa_frames,
                                 double *total_sasa_frames,
                                 double *residue_sasa_frames,
                                 double *selection_sasa_frames);

/* Contexts for parallel trajectory frames: lanes[0] is primary, the rest
 * are owned by it and reused across calls. *n_lanes is the count wanted on
 * entry and the count granted on return; NULL when only one is granted. */
fastsasa_context **fastsasa_context_frame_lanes(fastsasa_context *primary, int n_frames, int *n_lanes);

#endif
