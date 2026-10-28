#ifndef FASTSASA_XTC_H
#define FASTSASA_XTC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GROMACS XTC reader built on the vendored xdrfile library
 * (third-party/xdrfile); no external plugin or runtime is needed. */
typedef struct fastsasa_xtc {
    void *file;              /* XDRFILE * */
    int64_t *frame_offsets;  /* byte offset of every frame, from one header scan */
    float *coords;
    int n_atoms;
    int n_frames;
    int current_frame;
} fastsasa_xtc;

int fastsasa_xtc_open(fastsasa_xtc *xtc,
                    const char *path);
void fastsasa_xtc_close(fastsasa_xtc *xtc);
int fastsasa_xtc_read_frame_soa(fastsasa_xtc *xtc,
                              double *x,
                              double *y,
                              double *z);
int fastsasa_xtc_seek_frame(fastsasa_xtc *xtc,
                          int frame);

#ifdef __cplusplus
}
#endif

#endif
