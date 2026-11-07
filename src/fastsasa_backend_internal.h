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
/* Deferred SR, batched mask launch and float upload: CUDA lanes only */
int fastsasa_context_set_defer_sr(fastsasa_context *context, int on);
int fastsasa_context_batched_sr(fastsasa_context **lanes, int n);
int fastsasa_context_set_f32_coords(fastsasa_context *context, const float *xyz_blocks, const double *bbox);
/* Forget constant inputs kept by an earlier trajectory call; no-op off CUDA */
int fastsasa_context_reset_static(fastsasa_context *context);
/* Error text is per thread; carry a worker thread's to the caller */
void fastsasa_set_last_error(const char *message);

#endif
