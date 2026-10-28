#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L

#include "fastsasa_xtc.h"
#include "fastsasa_portable.h"

#include "xdrfile.h"
#include "xdrfile_xtc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* XTC frame layout (big-endian XDR): magic 1995, natoms, step, time, a 3x3
 * box, natoms again, then either natoms x 3 raw floats (natoms <= 9) or the
 * compressed block: precision, minint[3], maxint[3], smallidx, a byte count,
 * and that many bytes padded to a multiple of four. */
enum {
    kXtcMagic = 1995,
    kXtcHeaderBytes = 4 * (4 + 9 + 1),
    kXtcCompressedPreambleBytes = 4 * (1 + 3 + 3 + 1 + 1),
    kXtcSmallIdxOffset = kXtcHeaderBytes + 4 * (1 + 3 + 3),
    /* Valid range of smallidx, as checked by the vendored decoder: FIRSTIDX
     * up to the 73-entry magicints table size minus 9 (it reads smallidx + 8). */
    kXtcFirstIdx = 9,
    kXtcLastIdx = 64,
    kScanBufferBytes = 8 * 1024 * 1024
};

/* Largest compressed block xdrfile can decode for n_atoms: its buffer holds
 * 1.2 * 3 * n_atoms ints, of which the first three are reserved. */
static int64_t
max_compressed_bytes(int n_atoms)
{
    const int64_t ints = (int64_t)(3 * (int64_t)n_atoms * 1.2) - 3;

    return ints > 0 ? 4 * ints : 0;
}

static int
read_be_int32(FILE *file,
              int32_t *value)
{
    unsigned char bytes[4];

    if (fread(bytes, 1u, 4u, file) != 4u) return 0;
    *value = (int32_t)(((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
                       ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3]);
    return 1;
}

/* Records every frame's offset by reading only the headers and skipping the
 * coordinate blocks, so frame counts and random access cost one quick pass
 * instead of decoding the whole file. */
static int
scan_frame_offsets(const char *path,
                   int n_atoms,
                   int64_t **offsets,
                   int *n_frames)
{
    FILE *file = fopen(path, "rb");
    int64_t *list = NULL;
    int capacity = 0;
    int count = 0;
    fastsasa_file_offset offset = 0;
    fastsasa_file_offset file_size;

    *offsets = NULL;
    *n_frames = 0;
    if (file == NULL) return 0;
    if (fastsasa_fseek(file, 0, SEEK_END) != 0 || (file_size = fastsasa_ftell(file)) < 0) {
        fclose(file);
        return 0;
    }
    /* A large buffer keeps the forward skips inside buffered data, so the
     * scan streams through the file instead of seeking per frame (which
     * costs seconds per thousand frames on a spinning disk). */
    setvbuf(file, NULL, _IOFBF, kScanBufferBytes);
    for (;;) {
        int32_t magic, frame_atoms;
        fastsasa_file_offset next;

        if (fastsasa_fseek(file, offset, SEEK_SET) != 0 || !read_be_int32(file, &magic)) break;
        if (magic != kXtcMagic || !read_be_int32(file, &frame_atoms) || frame_atoms != n_atoms) {
            free(list);
            fclose(file);
            return 0;
        }
        if (n_atoms <= 9) {
            next = offset + kXtcHeaderBytes + (fastsasa_file_offset)n_atoms * 12;
        } else {
            int32_t small_idx, byte_count;

            /* Reject frames xdrfile could not decode safely: an out-of-range
             * smallidx indexes past its table, and an oversized block would
             * overrun its decompression buffer. */
            if (fastsasa_fseek(file, offset + kXtcSmallIdxOffset, SEEK_SET) != 0 ||
                !read_be_int32(file, &small_idx) || !read_be_int32(file, &byte_count) ||
                small_idx < kXtcFirstIdx || small_idx > kXtcLastIdx ||
                byte_count < 0 || byte_count > max_compressed_bytes(n_atoms)) {
                free(list);
                fclose(file);
                return 0;
            }
            next = offset + kXtcHeaderBytes + kXtcCompressedPreambleBytes +
                   (((fastsasa_file_offset)byte_count + 3) / 4) * 4;
        }
        /* A frame cut off by the end of the file (an interrupted run) is
         * not counted, so every listed frame can be read. */
        if (next > file_size) break;
        if (count == capacity) {
            const int grown = capacity > 0 ? 2 * capacity : 1024;
            int64_t *resized = (int64_t *)realloc(list, sizeof(int64_t) * (size_t)grown);

            if (resized == NULL) {
                free(list);
                fclose(file);
                return 0;
            }
            list = resized;
            capacity = grown;
        }
        list[count++] = (int64_t)offset;
        offset = next;
    }
    fclose(file);
    if (count == 0) {
        free(list);
        return 0;
    }
    *offsets = list;
    *n_frames = count;
    return 1;
}

int
fastsasa_xtc_open(fastsasa_xtc *xtc,
                  const char *path)
{
    int n_atoms = 0;

    if (xtc == NULL || path == NULL) return 0;
    memset(xtc, 0, sizeof(*xtc));
    if (read_xtc_natoms((char *)path, &n_atoms) != exdrOK || n_atoms <= 0) return 0;
    if (!scan_frame_offsets(path, n_atoms, &xtc->frame_offsets, &xtc->n_frames)) return 0;
    xtc->coords = (float *)malloc(sizeof(float) * 3u * (size_t)n_atoms);
    xtc->file = xdrfile_open(path, "r");
    if (xtc->coords == NULL || xtc->file == NULL) {
        fastsasa_xtc_close(xtc);
        return 0;
    }
    xtc->n_atoms = n_atoms;
    xtc->current_frame = 0;
    return 1;
}

void
fastsasa_xtc_close(fastsasa_xtc *xtc)
{
    if (xtc == NULL) return;
    if (xtc->file != NULL) xdrfile_close((XDRFILE *)xtc->file);
    free(xtc->frame_offsets);
    free(xtc->coords);
    memset(xtc, 0, sizeof(*xtc));
}

int
fastsasa_xtc_read_frame_soa(fastsasa_xtc *xtc,
                            double *x,
                            double *y,
                            double *z)
{
    int step;
    float time;
    float precision;
    matrix box;

    if (xtc == NULL || xtc->file == NULL || xtc->coords == NULL ||
        x == NULL || y == NULL || z == NULL) {
        return 0;
    }
    if (read_xtc((XDRFILE *)xtc->file, xtc->n_atoms, &step, &time, box,
                 (rvec *)xtc->coords, &precision) != exdrOK) {
        return 0;
    }
    ++xtc->current_frame;
    /* XTC stores nanometres; the float scaling matches what VMD's molfile
     * XTC plugin produced, so coordinates are unchanged from earlier
     * releases. */
    for (int atom = 0; atom < xtc->n_atoms; ++atom) {
        x[atom] = (double)(xtc->coords[3 * atom] * 10.0f);
        y[atom] = (double)(xtc->coords[3 * atom + 1] * 10.0f);
        z[atom] = (double)(xtc->coords[3 * atom + 2] * 10.0f);
    }
    return 1;
}

int
fastsasa_xtc_seek_frame(fastsasa_xtc *xtc,
                        int frame)
{
    FILE *file;

    if (xtc == NULL || xtc->file == NULL || frame < 0 || frame >= xtc->n_frames) return 0;
    file = xdrfile_get_fp((XDRFILE *)xtc->file);
    if (file == NULL ||
        fastsasa_fseek(file, (fastsasa_file_offset)xtc->frame_offsets[frame], SEEK_SET) != 0) {
        return 0;
    }
    xtc->current_frame = frame;
    return 1;
}
