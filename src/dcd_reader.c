#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "fastsasa_dcd.h"
#include "fastsasa_portable.h"

#include <limits.h>
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
swap_u32(uint32_t value)
{
    return ((value & 0x000000ffu) << 24) |
           ((value & 0x0000ff00u) << 8) |
           ((value & 0x00ff0000u) >> 8) |
           ((value & 0xff000000u) >> 24);
}

static int32_t
maybe_swap_i32(int32_t value,
               int reverse_endian)
{
    uint32_t raw;

    if (!reverse_endian) return value;
    memcpy(&raw, &value, sizeof(raw));
    raw = swap_u32(raw);
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static int
read_i32(FILE *file,
         int reverse_endian,
         int32_t *value)
{
    if (fread(value, sizeof(*value), 1, file) != 1) return 0;
    *value = maybe_swap_i32(*value, reverse_endian);
    return 1;
}

static int
skip_bytes(FILE *file,
           int32_t n)
{
    return fastsasa_fseek(file, (fastsasa_file_offset)n, SEEK_CUR) == 0;
}

static int
read_record_payload(FILE *file,
                    int reverse_endian,
                    void *payload,
                    int32_t expected_size)
{
    int32_t start_size, end_size;

    if (!read_i32(file, reverse_endian, &start_size)) return 0;
    if (start_size != expected_size) return 0;
    if (fread(payload, 1, (size_t)expected_size, file) != (size_t)expected_size) return 0;
    if (!read_i32(file, reverse_endian, &end_size)) return 0;
    return end_size == expected_size;
}

static int
skip_record(FILE *file,
            int reverse_endian)
{
    int32_t start_size, end_size;

    if (!read_i32(file, reverse_endian, &start_size)) return 0;
    if (start_size < 0) return 0;
    if (!skip_bytes(file, start_size)) return 0;
    if (!read_i32(file, reverse_endian, &end_size)) return 0;
    return end_size == start_size;
}

static int
read_float_record(FILE *file,
                  int reverse_endian,
                  int n,
                  double *dst,
                  int stride,
                  int offset,
                  float *tmp)
{
    int32_t start_size, end_size;
    int32_t expected_size;

    if (n <= 0 || (size_t)n > (size_t)INT32_MAX / sizeof(float)) return 0;
    expected_size = (int32_t)(sizeof(float) * (size_t)n);
    if (!read_i32(file, reverse_endian, &start_size)) return 0;
    if (start_size != expected_size) return 0;
    if (tmp == NULL) return 0;

    if (fread(tmp, 1, (size_t)start_size, file) != (size_t)start_size) return 0;
    if (!read_i32(file, reverse_endian, &end_size) || end_size != start_size) {
        return 0;
    }

    if (reverse_endian) {
        for (int i = 0; i < n; ++i) {
            uint32_t raw;
            memcpy(&raw, &tmp[i], sizeof(raw));
            raw = swap_u32(raw);
            memcpy(&tmp[i], &raw, sizeof(tmp[i]));
        }
    }

    for (int i = 0; i < n; ++i) {
        dst[(size_t)stride * (size_t)i + (size_t)offset] = tmp[i];
    }

    return 1;
}

static int
detect_frame_layout(fastsasa_dcd *dcd)
{
    const int32_t coord_size = (int32_t)(sizeof(float) * (size_t)dcd->n_atoms);
    int32_t first_size, end_size, x_size;
    fastsasa_file_offset frame_start;

    frame_start = fastsasa_ftell(dcd->file);
    if (frame_start < 0) return 0;
    dcd->frame_start_offset = frame_start;

    if (!read_i32(dcd->file, dcd->reverse_endian, &first_size)) return 0;
    if (first_size == coord_size) {
        dcd->has_unit_cell = 0;
        dcd->frame_stride = 3 * ((int64_t)coord_size + 8);
        return fastsasa_fseek(dcd->file, frame_start, SEEK_SET) == 0;
    }

    if (first_size <= 0) return 0;
    if (!skip_bytes(dcd->file, first_size)) return 0;
    if (!read_i32(dcd->file, dcd->reverse_endian, &end_size) || end_size != first_size) return 0;
    if (!read_i32(dcd->file, dcd->reverse_endian, &x_size) || x_size != coord_size) return 0;

    dcd->has_unit_cell = 1;
    dcd->frame_stride = ((int64_t)first_size + 8) + 3 * ((int64_t)coord_size + 8);
    return fastsasa_fseek(dcd->file, frame_start, SEEK_SET) == 0;
}

int
fastsasa_dcd_open(fastsasa_dcd *dcd,
                      const char *path)
{
    int32_t rec_size, rec_end, n_atoms_record;
    unsigned char header[84];
    char cord[4];

    if (dcd == NULL || path == NULL) return 0;
    memset(dcd, 0, sizeof(*dcd));

    dcd->file = fopen(path, "rb");
    if (dcd->file == NULL) return 0;

    if (fread(&rec_size, sizeof(rec_size), 1, dcd->file) != 1) goto fail;
    if (rec_size == 84) {
        dcd->reverse_endian = 0;
    } else if (maybe_swap_i32(rec_size, 1) == 84) {
        dcd->reverse_endian = 1;
    } else {
        goto fail;
    }

    if (fread(header, 1, sizeof(header), dcd->file) != sizeof(header)) goto fail;
    memcpy(cord, header, sizeof(cord));
    if (memcmp(cord, "CORD", 4) != 0) goto fail;
    memcpy(&dcd->n_frames, header + 4, sizeof(int32_t));
    dcd->n_frames = maybe_swap_i32(dcd->n_frames, dcd->reverse_endian);

    if (!read_i32(dcd->file, dcd->reverse_endian, &rec_end) || rec_end != 84) goto fail;
    if (!skip_record(dcd->file, dcd->reverse_endian)) goto fail;
    if (!read_record_payload(dcd->file, dcd->reverse_endian, &n_atoms_record, sizeof(n_atoms_record))) goto fail;
    dcd->n_atoms = maybe_swap_i32(n_atoms_record, dcd->reverse_endian);
    if (dcd->n_atoms <= 0 ||
        (size_t)dcd->n_atoms > (size_t)INT32_MAX / sizeof(float)) goto fail;
    dcd->scratch = malloc(sizeof(float) * (size_t)dcd->n_atoms);
    if (dcd->scratch == NULL) goto fail;

    if (!detect_frame_layout(dcd)) goto fail;
    dcd->current_frame = 0;
    return 1;

fail:
    fastsasa_dcd_close(dcd);
    return 0;
}

void
fastsasa_dcd_close(fastsasa_dcd *dcd)
{
    if (dcd == NULL) return;
    if (dcd->file != NULL) fclose(dcd->file);
    free(dcd->scratch);
    memset(dcd, 0, sizeof(*dcd));
}

int
fastsasa_dcd_read_frame(fastsasa_dcd *dcd,
                            double *xyz)
{
    if (dcd == NULL || dcd->file == NULL || xyz == NULL) return 0;

    if (dcd->has_unit_cell && !skip_record(dcd->file, dcd->reverse_endian)) return 0;

    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, xyz, 3, 0, dcd->scratch)) return 0;
    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, xyz, 3, 1, dcd->scratch)) return 0;
    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, xyz, 3, 2, dcd->scratch)) return 0;
    ++dcd->current_frame;
    return 1;
}

int
fastsasa_dcd_read_frame_soa(fastsasa_dcd *dcd,
                                double *x,
                                double *y,
                                double *z)
{
    if (dcd == NULL || dcd->file == NULL || x == NULL || y == NULL || z == NULL) return 0;

    if (dcd->has_unit_cell && !skip_record(dcd->file, dcd->reverse_endian)) return 0;

    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, x, 1, 0, dcd->scratch)) return 0;
    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, y, 1, 0, dcd->scratch)) return 0;
    if (!read_float_record(dcd->file, dcd->reverse_endian, dcd->n_atoms, z, 1, 0, dcd->scratch)) return 0;
    ++dcd->current_frame;
    return 1;
}

int
fastsasa_dcd_seek_frame(fastsasa_dcd *dcd,
                      int frame)
{
    int64_t offset;

    if (dcd == NULL || dcd->file == NULL || frame < 0 || dcd->frame_stride <= 0) return 0;
    if (dcd->n_frames > 0 && frame >= dcd->n_frames) return 0;
    if (frame > 0 && dcd->frame_stride >
        (INT64_MAX - dcd->frame_start_offset) / (int64_t)frame) return 0;
    offset = dcd->frame_start_offset + dcd->frame_stride * (int64_t)frame;
    if (fastsasa_fseek(dcd->file, (fastsasa_file_offset)offset, SEEK_SET) != 0) return 0;
    dcd->current_frame = frame;
    return 1;
}
