#ifndef FASTSASA_CPU_H
#define FASTSASA_CPU_H

#include "fastsasa_trajectory.h"

#ifdef __cplusplus
extern "C" {
#endif

int fastsasa_cpu_default_threads(void);

int fastsasa_cpu_shrake_rupley(int n_atoms,
                            int n_points,
                            const double *x,
                            const double *y,
                            const double *z,
                            const double *expanded_radii,
                            const double *test_points,
                            int n_threads,
                            double *sasa);

/*
 * Same as fastsasa_cpu_shrake_rupley, but precision selects
 * FASTSASA_PRECISION_FP64 (default, identical to fastsasa_cpu_shrake_rupley)
 * or FASTSASA_PRECISION_FP32 (see fastsasa.h) for faster reduced-precision
 * compute. Input and output arrays stay double either way; only the internal
 * arithmetic changes precision.
 */
int fastsasa_cpu_shrake_rupley_precision(int n_atoms,
                                       int n_points,
                                       const double *x,
                                       const double *y,
                                       const double *z,
                                       const double *expanded_radii,
                                       const double *test_points,
                                       int n_threads,
                                       int precision,
                                       double *sasa);

/*
 * Writes a row-major n_atoms x n_points mask marking which Shrake-Rupley
 * test points on each atom's probe-expanded sphere are solvent accessible.
 * Uses the identical point test as fastsasa_cpu_shrake_rupley, so
 * (row sum / n_points) * 4 * pi * expanded_radius^2 equals the atom SASA.
 * Intended for surface visualization exports.
 */
int fastsasa_cpu_exposed_points(int n_atoms,
                              int n_points,
                              const double *x,
                              const double *y,
                              const double *z,
                              const double *expanded_radii,
                              const double *test_points,
                              int n_threads,
                              unsigned char *exposed);

int fastsasa_cpu_lee_richards(int n_atoms,
                           int n_slices,
                           const double *x,
                           const double *y,
                           const double *z,
                           const double *expanded_radii,
                           int n_threads,
                           double *sasa);

/*
 * Lee-Richards area of one atom with a brute-force neighbour scan; used by
 * the CUDA backend to recompute the rare atoms whose slice exceeds the
 * device arc buffer, so GPU results stay bit-identical to this reference.
 */
int fastsasa_cpu_lee_richards_atom(int n_atoms,
                                 int n_slices,
                                 const double *x,
                                 const double *y,
                                 const double *z,
                                 const double *expanded_radii,
                                 int atom,
                                 double *area);

int fastsasa_cpu_calc_trajectory_soa(const fastsasa_topology *topology,
                                   const fastsasa_soa_frames *frames,
                                   const fastsasa_parameters *parameters,
                                   int n_threads,
                                   double *total_sasa,
                                   double *atom_sasa_frames,
                                   double *residue_sasa_frames);

int fastsasa_cpu_calc_trajectory_soa_selection(const fastsasa_topology *topology,
                                             const fastsasa_soa_frames *frames,
                                             const unsigned int *selection_masks,
                                             int n_selections,
                                             const fastsasa_parameters *parameters,
                                             int n_threads,
                                             double *total_sasa,
                                             double *selection_sasa_frames);

#ifdef __cplusplus
}
#endif

#endif
