#ifndef FASTSASA_XTC_H
#define FASTSASA_XTC_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fastsasa_xtc {
    void *plugin_library;
    void *plugin;
    void *file_handle;
    char *path;
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
