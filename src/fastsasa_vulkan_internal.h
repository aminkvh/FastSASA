#ifndef FASTSASA_VULKAN_H
#define FASTSASA_VULKAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fastsasa_vk_context fastsasa_vk_context;

/* device_index < 0 selects a discrete GPU when available. */
int fastsasa_vk_context_create(fastsasa_vk_context **context, int device_index);
/* Returns this thread's most recent fastsasa_vk_context_create failure reason. */
const char *fastsasa_vk_create_error(void);
void fastsasa_vk_context_free(fastsasa_vk_context *context);
const char *fastsasa_vk_device_name(const fastsasa_vk_context *context);
const char *fastsasa_vk_last_error(const fastsasa_vk_context *context);
uint32_t fastsasa_vk_subgroup_size(const fastsasa_vk_context *context);
uint32_t fastsasa_vk_sr_workgroup_size(const fastsasa_vk_context *context);
int fastsasa_vk_supports_fp64(const fastsasa_vk_context *context);

/*
 * Calculate Shrake-Rupley SASA for every atom. Coordinates and sphere points
 * are tightly packed xyz arrays. The effective test radius per atom is
 * radii[i] + probe_radius; FastSASA's API layer passes probe-expanded radii with
 * probe_radius 0. For FP32 calculations the caller should translate
 * coordinates to a frame-local origin first, as the API layer does, because
 * the backend converts the given coordinates to the selected precision
 * without shifting them.
 */
int fastsasa_vk_sr(fastsasa_vk_context *context,
                 const double *xyz,
                 const double *radii,
                 uint32_t atom_count,
                 const double *sphere_xyz,
                 uint32_t point_count,
                 double probe_radius,
                 int use_fp64,
                 double *sasa);

/*
 * Per-point exposed/buried mask for every atom (VMD-style surface-point
 * export), not a reduced SASA area. FP64 only - the caller must check
 * fastsasa_vk_supports_fp64() first, or simply try the call and fall back to
 * the CPU backend on failure. xyz may be nullptr if x/y/z are given instead
 * (planar SoA), matching fastsasa_cpu_exposed_points()'s calling convention;
 * radii must already include probe_radius (probe_radius is passed as 0 by
 * the API layer, matching fastsasa_vk_sr() above). exposed must have room
 * for atom_count * point_count bytes, laid out exposed[atom * point_count +
 * point], identical to fastsasa_cpu_exposed_points()'s output layout.
 */
int fastsasa_vk_sr_exposed_points(fastsasa_vk_context *context,
                                const double *xyz,
                                const double *x,
                                const double *y,
                                const double *z,
                                const double *radii,
                                uint32_t atom_count,
                                const double *sphere_xyz,
                                uint32_t point_count,
                                double probe_radius,
                                unsigned char *exposed);

/*
 * Calculate SASA only for center_indices while retaining every atom in xyz as
 * an occluder. Output order follows center_indices.
 */
int fastsasa_vk_sr_centers(fastsasa_vk_context *context,
                         const double *xyz,
                         const double *radii,
                         uint32_t atom_count,
                         const double *sphere_xyz,
                         uint32_t point_count,
                         double probe_radius,
                         const uint32_t *center_indices,
                         uint32_t center_count,
                         int use_fp64,
                         double *sasa);

int fastsasa_vk_lee_richards(fastsasa_vk_context *context,
                           const double *xyz,
                           const double *radii,
                           uint32_t atom_count,
                           uint32_t slice_count,
                           double probe_radius,
                           int use_fp64,
                           double *sasa);

int fastsasa_vk_lee_richards_centers(fastsasa_vk_context *context,
                                   const double *xyz,
                                   const double *radii,
                                   uint32_t atom_count,
                                   uint32_t slice_count,
                                   double probe_radius,
                                   const uint32_t *center_indices,
                                   uint32_t center_count,
                                   int use_fp64,
                                   double *sasa);

/* Frame-major xyz input and frame-major SASA output. The context and buffers
 * are reused across frames; frames are not retained after calculation. */
int fastsasa_vk_sr_frames(fastsasa_vk_context *context,
                        const double *frame_xyz,
                        const double *radii,
                        uint32_t frame_count,
                        uint32_t atom_count,
                        const double *sphere_xyz,
                        uint32_t point_count,
                        double probe_radius,
                        int use_fp64,
                        double *frame_sasa);

int fastsasa_vk_sr_center_frames(fastsasa_vk_context *context,
                               const double *frame_xyz,
                               const double *radii,
                               uint32_t frame_count,
                               uint32_t atom_count,
                               const double *sphere_xyz,
                               uint32_t point_count,
                               double probe_radius,
                               const uint32_t *center_indices,
                               uint32_t center_count,
                               int use_fp64,
                               double *frame_sasa);

int fastsasa_vk_lee_richards_frames(fastsasa_vk_context *context,
                                  const double *frame_xyz,
                                  const double *radii,
                                  uint32_t frame_count,
                                  uint32_t atom_count,
                                  uint32_t slice_count,
                                  double probe_radius,
                                  int use_fp64,
                                  double *frame_sasa);

int fastsasa_vk_lee_richards_center_frames(fastsasa_vk_context *context,
                                         const double *frame_xyz,
                                         const double *radii,
                                         uint32_t frame_count,
                                         uint32_t atom_count,
                                         uint32_t slice_count,
                                         double probe_radius,
                                         const uint32_t *center_indices,
                                         uint32_t center_count,
                                         int use_fp64,
                                         double *frame_sasa);

#ifdef __cplusplus
}
#endif

#endif
