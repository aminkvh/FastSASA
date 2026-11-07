/* Exact SASA on CUDA (exact_cuda.cu), driven by sr_cuda.cu's device contexts. */
#ifndef FASTSASA_EXACT_CUDA_H
#define FASTSASA_EXACT_CUDA_H

#include "fastsasa.h"

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct fastsasa_exact_cuda;

/* Areas of all atoms, or of centers[0..n_centers) only (sasa[centers[c]]), on stream; *state holds the
 * device buffers between calls (created on first use). Synchronizes the stream before returning. */
int fastsasa_exact_cuda_run(struct fastsasa_exact_cuda **state, cudaStream_t stream, int n_atoms, const double *x,
                            const double *y, const double *z, const double *expanded_radii, const int *centers,
                            int n_centers, int mixed, double *sasa);
void fastsasa_exact_cuda_free(struct fastsasa_exact_cuda *state);

#ifdef __cplusplus
}
#endif

#endif
