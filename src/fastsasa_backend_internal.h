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

#endif
