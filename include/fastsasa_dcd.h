#ifndef FASTSASA_DCD_H
#define FASTSASA_DCD_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fastsasa_dcd {
    FILE *file;
    int n_atoms;
    int n_frames;
    int has_unit_cell;
    int reverse_endian;
    int64_t frame_start_offset;
    int64_t frame_stride;
    int current_frame;
    float *scratch;
} fastsasa_dcd;

int fastsasa_dcd_open(fastsasa_dcd *dcd,
                    const char *path);
void fastsasa_dcd_close(fastsasa_dcd *dcd);
int fastsasa_dcd_read_frame(fastsasa_dcd *dcd,
                          double *xyz);
int fastsasa_dcd_read_frame_soa(fastsasa_dcd *dcd,
                              double *x,
                              double *y,
                              double *z);
int fastsasa_dcd_seek_frame(fastsasa_dcd *dcd,
                          int frame);

#ifdef __cplusplus
}
#endif

#endif
