#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_radius.h"
#include "fastsasa_topology.h"
#include "fastsasa_trajectory.h"

#include "fastsasa_dcd.h"
#include "fastsasa_xtc.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include "fastsasa_portable.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef FASTSASA_VERSION
#define FASTSASA_VERSION "unknown"
#endif

typedef struct radius_config {
    char **keys;
    double *radii;
    int *classes;
    int n;
} radius_config;

enum {
    FASTSASA_CLASS_UNKNOWN = -1,
    FASTSASA_CLASS_APOLAR = 0,
    FASTSASA_CLASS_POLAR = 1
};

typedef enum trajectory_format {
    TRAJECTORY_FORMAT_DCD,
    TRAJECTORY_FORMAT_XTC
} trajectory_format;

typedef struct trajectory_reader {
    trajectory_format format;
    fastsasa_dcd dcd;
    fastsasa_xtc xtc;
    int n_atoms;
} trajectory_reader;

typedef struct selection_plan {
    unsigned int *atom_masks;
    char (*names)[FASTSASA_MAX_SELECTION_NAME + 1];
    int n_selections;
} selection_plan;

typedef struct frame_plan {
    int *indices;
    int n_indices;
    int stream_all;
    int stream_start;
    int stream_step;
} frame_plan;

static double
now_seconds(void)
{
    return fastsasa_monotonic_seconds();
}

static void
kahan_add(double value,
          double *sum,
          double *compensation)
{
    const double corrected = value - *compensation;
    const double next = *sum + corrected;

    *compensation = (next - *sum) - corrected;
    *sum = next;
}

static char *
copy_string(const char *text)
{
    size_t n = text != NULL ? strlen(text) : 0u;
    char *copy = (char *)malloc(n + 1u);

    if (copy == NULL) return NULL;
    if (n > 0u) memcpy(copy, text, n);
    copy[n] = '\0';
    return copy;
}

static void
trim_copy(char *dst,
          size_t dst_size,
          const char *src,
          size_t src_size)
{
    size_t begin = 0u;
    size_t end = src_size;
    size_t n;

    if (dst_size == 0u) return;
    while (begin < end && isspace((unsigned char)src[begin])) ++begin;
    while (end > begin && isspace((unsigned char)src[end - 1u])) --end;
    n = end - begin;
    if (n >= dst_size) n = dst_size - 1u;
    memcpy(dst, src + begin, n);
    dst[n] = '\0';
}

static void
upper_inplace(char *text)
{
    if (text == NULL) return;
    for (; *text != '\0'; ++text) *text = (char)toupper((unsigned char)*text);
}

static int
has_suffix(const char *path,
           const char *suffix)
{
    size_t path_len = strlen(path);
    size_t suffix_len = strlen(suffix);

    return path_len >= suffix_len && strcmp(path + path_len - suffix_len, suffix) == 0;
}

static void
radius_config_free(radius_config *config)
{
    if (config == NULL) return;
    for (int i = 0; i < config->n; ++i) free(config->keys[i]);
    free(config->keys);
    free(config->radii);
    free(config->classes);
    memset(config, 0, sizeof(*config));
}

static int
class_from_text(const char *text)
{
    char key[32];

    trim_copy(key, sizeof(key), text != NULL ? text : "", strlen(text != NULL ? text : ""));
    upper_inplace(key);
    if (strcmp(key, "POLAR") == 0) return FASTSASA_CLASS_POLAR;
    if (strcmp(key, "APOLAR") == 0) return FASTSASA_CLASS_APOLAR;
    return FASTSASA_CLASS_UNKNOWN;
}

static int
radius_config_add(radius_config *config,
                  const char *residue,
                  const char *atom,
                  double radius,
                  int atom_class)
{
    char key[160];
    char **new_keys;
    double *new_radii;
    int *new_classes;

    snprintf(key, sizeof(key), "%s:%s", residue, atom);
    upper_inplace(key);
    new_keys = (char **)realloc(config->keys, sizeof(char *) * (size_t)(config->n + 1));
    if (new_keys == NULL) return 0;
    config->keys = new_keys;
    new_radii = (double *)realloc(config->radii, sizeof(double) * (size_t)(config->n + 1));
    if (new_radii == NULL) return 0;
    config->radii = new_radii;
    new_classes = (int *)realloc(config->classes, sizeof(int) * (size_t)(config->n + 1));
    if (new_classes == NULL) return 0;
    config->classes = new_classes;
    config->keys[config->n] = copy_string(key);
    if (config->keys[config->n] == NULL) return 0;
    config->radii[config->n] = radius;
    config->classes[config->n] = atom_class;
    ++config->n;
    return 1;
}

static int
load_radius_config(const char *path,
                   radius_config *config)
{
    FILE *file = fopen(path, "r");
    char line[512];
    char section[32] = "";
    char type_names[256][64];
    double type_radii[256];
    int type_classes[256];
    int n_types = 0;

    if (file == NULL) return 0;
    memset(config, 0, sizeof(*config));
    while (fgets(line, sizeof(line), file) != NULL) {
        char *hash = strchr(line, '#');
        char first[64], second[64], third[64];

        if (hash != NULL) *hash = '\0';
        first[0] = '\0';
        second[0] = '\0';
        third[0] = '\0';
        if (sscanf(line, "%63s", first) != 1) continue;
        if (strcmp(first, "types:") == 0) {
            snprintf(section, sizeof(section), "%s", "types");
            continue;
        }
        if (strcmp(first, "atoms:") == 0) {
            snprintf(section, sizeof(section), "%s", "atoms");
            continue;
        }
        if (first[strlen(first) - 1u] == ':') {
            section[0] = '\0';
            continue;
        }
        if (strcmp(section, "types") == 0 && n_types < 256 &&
            sscanf(line, "%63s %lf %63s", first, &type_radii[n_types], second) >= 2) {
            snprintf(type_names[n_types], sizeof(type_names[n_types]), "%s", first);
            upper_inplace(type_names[n_types]);
            type_classes[n_types] = class_from_text(second);
            ++n_types;
        } else if (strcmp(section, "atoms") == 0 &&
                   sscanf(line, "%63s %63s %63s", first, second, third) == 3) {
            double radius = -1.0;
            int atom_class = FASTSASA_CLASS_UNKNOWN;

            upper_inplace(third);
            for (int i = 0; i < n_types; ++i) {
                if (strcmp(type_names[i], third) == 0) {
                    radius = type_radii[i];
                    atom_class = type_classes[i];
                    break;
                }
            }
            if (radius > 0.0 && !radius_config_add(config, first, second, radius, atom_class)) {
                fclose(file);
                return 0;
            }
        }
    }
    fclose(file);
    return 1;
}

static int
load_default_radius_config(radius_config *config)
{
    const char *env_path = getenv("FASTSASA_DEFAULT_CONFIG");

    if (env_path != NULL && env_path[0] != '\0' && load_radius_config(env_path, config)) return 1;
    for (size_t i = 0;; ++i) {
        const char *candidate = fastsasa_default_config_candidate(i);

        if (candidate == NULL) break;
        if (load_radius_config(candidate, config)) return 1;
    }
    return 0;
}

static int
config_index_for_key(const radius_config *config,
                     const char *residue,
                     const char *atom)
{
    char key[160];

    snprintf(key, sizeof(key), "%s:%s", residue != NULL ? residue : "", atom != NULL ? atom : "");
    upper_inplace(key);
    for (int i = 0; i < config->n; ++i) {
        if (strcmp(config->keys[i], key) == 0) return i;
    }
    return -1;
}

/* Exact residue:atom entry first; then the same atom under the canonical
 * residue name for MD variants (HIE -> HIS, CYX -> CYS, ...). */
static int
config_lookup(const radius_config *config,
              const char *residue,
              const char *atom)
{
    int index;
    const char *canonical;

    if (config == NULL) return -1;
    index = config_index_for_key(config, residue, atom);
    if (index >= 0) return index;
    canonical = fastsasa_canonical_residue(residue);
    if (canonical == NULL) return -1;
    return config_index_for_key(config, canonical, atom);
}

static double
configured_radius(const char *residue,
                  const char *atom,
                  const char *element,
                  void *userdata)
{
    const radius_config *config = (const radius_config *)userdata;
    const int index = config_lookup(config, residue, atom);

    if (index >= 0) return config->radii[index];
    return fastsasa_element_radius(element);
}

static int
configured_class(const char *residue,
                 const char *atom,
                 const char *element,
                 const radius_config *config)
{
    if (config != NULL && config->n > 0) {
        const int index = config_lookup(config, residue, atom);

        return index >= 0 ? config->classes[index] : FASTSASA_CLASS_UNKNOWN;
    }
    return fastsasa_element_class(element);
}

static int
allocate_topology(fastsasa_owned_topology *topology,
                  int n_atoms)
{
    memset(topology, 0, sizeof(*topology));
    topology->x = (double *)calloc((size_t)n_atoms, sizeof(double));
    topology->y = (double *)calloc((size_t)n_atoms, sizeof(double));
    topology->z = (double *)calloc((size_t)n_atoms, sizeof(double));
    topology->radii = (double *)calloc((size_t)n_atoms, sizeof(double));
    topology->residue_ids = (int *)calloc((size_t)n_atoms, sizeof(int));
    topology->residue_numbers = (int *)calloc((size_t)n_atoms, sizeof(int));
    topology->residue_number_strings = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->atom_names = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->residue_names = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->chain_ids = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->segment_ids = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->elements = (char **)calloc((size_t)n_atoms, sizeof(char *));
    topology->atom_flags = (unsigned char *)calloc((size_t)n_atoms, sizeof(unsigned char));
    return topology->x != NULL && topology->y != NULL && topology->z != NULL &&
           topology->radii != NULL && topology->residue_ids != NULL &&
           topology->residue_numbers != NULL && topology->residue_number_strings != NULL &&
           topology->atom_names != NULL && topology->residue_names != NULL &&
           topology->chain_ids != NULL && topology->segment_ids != NULL &&
           topology->elements != NULL && topology->atom_flags != NULL;
}

static int
set_topology_atom(fastsasa_owned_topology *topology,
                  int atom,
                  const char *atom_name,
                  const char *residue_name,
                  const char *chain_id,
                  const char *segment_id,
                  const char *residue_number,
                  const char *element)
{
    topology->atom_names[atom] = copy_string(atom_name);
    topology->residue_names[atom] = copy_string(residue_name);
    topology->chain_ids[atom] = copy_string(chain_id);
    topology->segment_ids[atom] = copy_string(segment_id);
    topology->residue_number_strings[atom] = copy_string(residue_number);
    topology->elements[atom] = copy_string(element);
    return topology->atom_names[atom] != NULL &&
           topology->residue_names[atom] != NULL &&
           topology->chain_ids[atom] != NULL &&
           topology->segment_ids[atom] != NULL &&
           topology->residue_number_strings[atom] != NULL &&
           topology->elements[atom] != NULL;
}

static int
read_pdb_topology(const char *path,
                  radius_config *config,
                  fastsasa_owned_topology *topology)
{
    FILE *file = fopen(path, "r");
    char line[256];
    int n_atoms = 0;
    int atom = 0;
    int residue = -1;
    char last_chain[16] = "";
    char last_residue_number[32] = "";
    char last_residue_name[32] = "";

    if (file == NULL) return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strncmp(line, "ATOM  ", 6) == 0 || strncmp(line, "HETATM", 6) == 0) ++n_atoms;
    }
    if (n_atoms <= 0 || !allocate_topology(topology, n_atoms)) {
        fclose(file);
        fastsasa_topology_free(topology);
        return 0;
    }
    rewind(file);
    while (fgets(line, sizeof(line), file) != NULL) {
        char atom_name[32], residue_name[32], chain_id[16], segment_id[16], residue_number[32], element[8], column_element[8];
        double x, y, z, radius;

        if (strncmp(line, "ATOM  ", 6) != 0 && strncmp(line, "HETATM", 6) != 0) continue;
        trim_copy(atom_name, sizeof(atom_name), line + 12, strlen(line) >= 16u ? 4u : 0u);
        trim_copy(residue_name, sizeof(residue_name), line + 17, strlen(line) >= 20u ? 3u : 0u);
        trim_copy(chain_id, sizeof(chain_id), line + 21, strlen(line) >= 22u ? 1u : 0u);
        trim_copy(residue_number, sizeof(residue_number), line + 22, strlen(line) >= 26u ? 4u : 0u);
        trim_copy(segment_id, sizeof(segment_id), line + 72, strlen(line) >= 76u ? 4u : 0u);
        trim_copy(column_element, sizeof(column_element), line + 76, strlen(line) >= 78u ? 2u : 0u);
        fastsasa_infer_element(element, sizeof(element), atom_name, column_element);
        if (strlen(line) < 54u || sscanf(line + 30, "%lf %lf %lf", &x, &y, &z) != 3) {
            fprintf(stderr, "invalid PDB coordinate record in %s\n", path);
            fclose(file);
            fastsasa_topology_free(topology);
            return 0;
        }

        radius = configured_radius(residue_name, atom_name, element, config);
        if (radius <= 0.0) radius = 1.70;
        if (atom == 0 ||
            strcmp(chain_id, last_chain) != 0 ||
            strcmp(residue_number, last_residue_number) != 0 ||
            strcmp(residue_name, last_residue_name) != 0) {
            ++residue;
            snprintf(last_chain, sizeof(last_chain), "%s", chain_id);
            snprintf(last_residue_number, sizeof(last_residue_number), "%s", residue_number);
            snprintf(last_residue_name, sizeof(last_residue_name), "%s", residue_name);
        }
        topology->x[atom] = x;
        topology->y[atom] = y;
        topology->z[atom] = z;
        topology->radii[atom] = radius;
        topology->residue_ids[atom] = residue;
        topology->residue_numbers[atom] = atoi(residue_number);
        topology->atom_flags[atom] =
            (strncmp(line, "HETATM", 6) == 0 ? FASTSASA_ATOM_HETATM : 0) |
            ((strcmp(element, "H") == 0 || strcmp(element, "D") == 0) ? FASTSASA_ATOM_HYDROGEN : 0);
        if (!set_topology_atom(topology, atom, atom_name, residue_name, chain_id, segment_id, residue_number, element)) {
            fclose(file);
            fastsasa_topology_free(topology);
            return 0;
        }
        ++atom;
    }
    fclose(file);
    topology->n_atoms = atom;
    topology->n_residues = residue + 1;
    return atom > 0;
}

static int
read_psf_topology(const char *path,
                  radius_config *config,
                  fastsasa_owned_topology *topology)
{
    FILE *file = fopen(path, "r");
    char line[512];
    int n_atoms = 0;
    int residue = -1;
    char last_segment[64] = "";
    char last_residue_number[64] = "";
    char last_residue_name[64] = "";

    if (file == NULL) return 0;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strstr(line, "!NATOM") != NULL) {
            if (sscanf(line, "%d", &n_atoms) != 1 || n_atoms <= 0) {
                fclose(file);
                return 0;
            }
            break;
        }
    }
    if (n_atoms <= 0 || !allocate_topology(topology, n_atoms)) {
        fclose(file);
        fastsasa_topology_free(topology);
        return 0;
    }
    for (int atom = 0; atom < n_atoms; ++atom) {
        int atom_id;
        char segment[64], residue_number[64], residue_name[64], atom_name[64], atom_type[64], element[8];
        double charge, mass, radius;

        if (fgets(line, sizeof(line), file) == NULL ||
            sscanf(line, "%d %63s %63s %63s %63s %63s %lf %lf",
                   &atom_id, segment, residue_number, residue_name, atom_name, atom_type, &charge, &mass) != 8) {
            fclose(file);
            fastsasa_topology_free(topology);
            return 0;
        }
        fastsasa_infer_element(element, sizeof(element), atom_name, atom_type);
        radius = configured_radius(residue_name, atom_name, element, config);
        if (radius <= 0.0) radius = fastsasa_element_radius(element);
        if (radius <= 0.0) radius = 1.70;
        if (atom == 0 ||
            strcmp(segment, last_segment) != 0 ||
            strcmp(residue_number, last_residue_number) != 0 ||
            strcmp(residue_name, last_residue_name) != 0) {
            ++residue;
            snprintf(last_segment, sizeof(last_segment), "%s", segment);
            snprintf(last_residue_number, sizeof(last_residue_number), "%s", residue_number);
            snprintf(last_residue_name, sizeof(last_residue_name), "%s", residue_name);
        }
        topology->radii[atom] = radius;
        topology->residue_ids[atom] = residue;
        topology->residue_numbers[atom] = atoi(residue_number);
        topology->atom_flags[atom] =
            (strcmp(element, "H") == 0 || strcmp(element, "D") == 0) ? FASTSASA_ATOM_HYDROGEN : 0;
        if (!set_topology_atom(topology, atom, atom_name, residue_name, "", segment, residue_number, element)) {
            fclose(file);
            fastsasa_topology_free(topology);
            return 0;
        }
    }
    fclose(file);
    topology->n_atoms = n_atoms;
    topology->n_residues = residue + 1;
    return 1;
}

static int
read_topology(const char *path,
              radius_config *config,
              fastsasa_owned_topology *topology)
{
    if (has_suffix(path, ".psf") || has_suffix(path, ".PSF")) {
        return read_psf_topology(path, config, topology);
    }
    if (has_suffix(path, ".cif") || has_suffix(path, ".CIF") ||
        has_suffix(path, ".mmcif") || has_suffix(path, ".MMCIF")) {
        return fastsasa_topology_read_mmcif(path,
                                          FASTSASA_TOPOLOGY_INCLUDE_HETATM | FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN,
                                          configured_radius,
                                          config,
                                          topology);
    }
    return read_pdb_topology(path, config, topology);
}

static void
selection_plan_release(selection_plan *plan)
{
    if (plan == NULL) return;
    free(plan->atom_masks);
    free(plan->names);
    memset(plan, 0, sizeof(*plan));
}

static void
frame_plan_release(frame_plan *plan)
{
    if (plan == NULL) return;
    free(plan->indices);
    memset(plan, 0, sizeof(*plan));
}

static void
selection_warning_to_stderr(const char *message,
                            void *userdata)
{
    (void)userdata;
    fprintf(stderr, "%s\n", message);
}

static int
parse_int_strict(const char *text,
                 int *value)
{
    char *end = NULL;
    long parsed;

    if (text == NULL || text[0] == '\0') return 0;
    parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < -2147483647L || parsed > 2147483647L) return 0;
    *value = (int)parsed;
    return 1;
}

static int
parse_double_strict(const char *text,
                    double *value)
{
    char *end = NULL;
    double parsed;

    if (text == NULL || text[0] == '\0') return 0;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static int
normalize_frame_index(int index,
                      int known_frames,
                      int *normalized)
{
    if (index < 0) {
        if (known_frames <= 0) return 0;
        index = known_frames + index;
    }
    if (index < 0) return 0;
    if (known_frames > 0 && index >= known_frames) return 0;
    *normalized = index;
    return 1;
}

static int
append_frame_index(frame_plan *plan,
                   int frame)
{
    int *new_indices = (int *)realloc(plan->indices, sizeof(int) * (size_t)(plan->n_indices + 1));

    if (new_indices == NULL) return 0;
    plan->indices = new_indices;
    plan->indices[plan->n_indices++] = frame;
    return 1;
}

static int
frame_plan_from_count(frame_plan *plan,
                      int count,
                      int known_frames)
{
    int limit = count;

    if (count < 0) return 0;
    if (count == 0) {
        if (known_frames <= 0) {
            plan->stream_all = 1;
            plan->stream_start = 0;
            plan->stream_step = 1;
            return 1;
        }
        limit = known_frames;
    } else if (known_frames > 0 && limit > known_frames) {
        limit = known_frames;
    }
    for (int frame = 0; frame < limit; ++frame) {
        if (!append_frame_index(plan, frame)) return 0;
    }
    return 1;
}

static int
frame_plan_from_spec(frame_plan *plan,
                     const char *spec,
                     int known_frames)
{
    char text[128];
    char *first_colon;
    int value;

    trim_copy(text, sizeof(text), spec != NULL ? spec : ":", strlen(spec != NULL ? spec : ":"));
    if (text[0] == '\0' || strcmp(text, ":") == 0) {
        return frame_plan_from_count(plan, 0, known_frames);
    }

    first_colon = strchr(text, ':');
    if (first_colon == NULL) {
        int normalized;

        if (!parse_int_strict(text, &value) ||
            !normalize_frame_index(value, known_frames, &normalized)) {
            return 0;
        }
        return append_frame_index(plan, normalized);
    }

    {
        char *second_colon = strchr(first_colon + 1, ':');
        char start_text[64] = "";
        char stop_text[64] = "";
        char step_text[64] = "";
        int start = 0;
        int stop = known_frames;
        int step = 1;

        if (second_colon != NULL && strchr(second_colon + 1, ':') != NULL) return 0;
        trim_copy(start_text, sizeof(start_text), text, (size_t)(first_colon - text));
        if (second_colon != NULL) {
            trim_copy(stop_text, sizeof(stop_text), first_colon + 1, (size_t)(second_colon - first_colon - 1));
            trim_copy(step_text, sizeof(step_text), second_colon + 1, strlen(second_colon + 1));
        } else {
            trim_copy(stop_text, sizeof(stop_text), first_colon + 1, strlen(first_colon + 1));
        }
        if (step_text[0] != '\0' && (!parse_int_strict(step_text, &step) || step <= 0)) return 0;
        if (start_text[0] != '\0' && (!parse_int_strict(start_text, &start) ||
                                      !normalize_frame_index(start, known_frames, &start))) {
            return 0;
        }
        if (stop_text[0] != '\0') {
            if (!parse_int_strict(stop_text, &stop)) return 0;
            if (stop < 0) {
                if (known_frames <= 0) return 0;
                stop = known_frames + stop;
            }
        } else if (known_frames <= 0) {
            plan->stream_all = 1;
            plan->stream_start = start;
            plan->stream_step = step;
            return 1;
        }
        if (known_frames > 0) {
            if (start < 0) start = 0;
            if (start > known_frames) start = known_frames;
            if (stop < 0) stop = 0;
            if (stop > known_frames) stop = known_frames;
        } else if (start < 0 || stop < 0) {
            return 0;
        }
        for (int frame = start; frame < stop; frame += step) {
            if (!append_frame_index(plan, frame)) return 0;
        }
    }
    return 1;
}

static int
frame_plan_next(const frame_plan *plan,
                int position,
                int *frame)
{
    if (plan->stream_all) {
        *frame = plan->stream_start + position * plan->stream_step;
        return 1;
    }
    if (position < 0 || position >= plan->n_indices) return 0;
    *frame = plan->indices[position];
    return 1;
}

static int
build_selection_plan(selection_plan *plan,
                     const fastsasa_owned_topology *topology,
                     const char *const *commands,
                     int n_selections)
{
    memset(plan, 0, sizeof(*plan));
    if (n_selections <= 0) return 1;
    if (n_selections > 31 || topology == NULL || topology->n_atoms <= 0) return 0;
    plan->atom_masks = (unsigned int *)calloc((size_t)topology->n_atoms, sizeof(unsigned int));
    plan->names = (char (*)[FASTSASA_MAX_SELECTION_NAME + 1])calloc((size_t)n_selections, sizeof(*plan->names));
    if (plan->atom_masks == NULL || plan->names == NULL) {
        selection_plan_release(plan);
        return 0;
    }
    for (int selection = 0; selection < n_selections; ++selection) {
        if (!fastsasa_topology_selection_mask_ex(commands[selection],
                                               topology,
                                               1u << selection,
                                               plan->atom_masks,
                                               plan->names[selection],
                                               sizeof(plan->names[selection]),
                                               selection_warning_to_stderr,
                                               NULL,
                                               NULL)) {
            selection_plan_release(plan);
            return 0;
        }
    }
    plan->n_selections = n_selections;
    return 1;
}

static int
build_expression_plan(selection_plan *plan,
                      const fastsasa_owned_topology *topology,
                      const char *command,
                      const char *default_name)
{
    char *named_command = NULL;
    const char *commands[1];
    int ok;

    if (command == NULL) {
        memset(plan, 0, sizeof(*plan));
        return 1;
    }
    if (strchr(command, ',') != NULL) {
        commands[0] = command;
        return build_selection_plan(plan, topology, commands, 1);
    }
    named_command = (char *)malloc(strlen(default_name) + strlen(command) + 3u);
    if (named_command == NULL) return 0;
    snprintf(named_command, strlen(default_name) + strlen(command) + 3u, "%s, %s", default_name, command);
    commands[0] = named_command;
    ok = build_selection_plan(plan, topology, commands, 1);
    free(named_command);
    return ok;
}

static int
selection_count_for_bit(const unsigned int *masks,
                        int n_atoms,
                        unsigned int bit)
{
    int count = 0;

    for (int atom = 0; atom < n_atoms; ++atom) {
        if ((masks[atom] & bit) != 0u) ++count;
    }
    return count;
}

static int
compact_selection_plan(selection_plan *plan,
                       const int *indices,
                       int n_filtered,
                       int original_n_atoms,
                       const char *kind)
{
    unsigned int *compact_masks;

    if (plan == NULL || plan->n_selections <= 0 || indices == NULL) return 1;
    compact_masks = (unsigned int *)calloc((size_t)n_filtered, sizeof(unsigned int));
    if (compact_masks == NULL) return 0;
    for (int atom = 0; atom < n_filtered; ++atom) {
        compact_masks[atom] = plan->atom_masks[indices[atom]];
    }
    for (int selection = 0; selection < plan->n_selections; ++selection) {
        unsigned int bit = 1u << selection;
        int before = selection_count_for_bit(plan->atom_masks, original_n_atoms, bit);
        int after = selection_count_for_bit(compact_masks, n_filtered, bit);

        if (before > 0 && after == 0) {
            fprintf(stderr,
                    "FastSASA: warning: %s '%s' matched atoms outside the calculation universe; none remain\n",
                    kind,
                    plan->names[selection]);
        } else if (after < before) {
            fprintf(stderr,
                    "FastSASA: warning: %s '%s' was intersected with the calculation universe; %d of %d atoms remain\n",
                    kind,
                    plan->names[selection],
                    after,
                    before);
        }
    }
    free(plan->atom_masks);
    plan->atom_masks = compact_masks;
    return 1;
}

static int
build_filter_indices(const selection_plan *plan,
                     const fastsasa_owned_topology *topology,
                     int include_hydrogen,
                     int include_hetatm,
                     int **indices,
                     int *n_filtered)
{
    int count = 0;
    int out = 0;
    const int n_atoms = topology->n_atoms;

    *indices = NULL;
    *n_filtered = n_atoms;
    for (int atom = 0; atom < n_atoms; ++atom) {
        unsigned char flags = topology->atom_flags != NULL ? topology->atom_flags[atom] : 0;
        int keep = (include_hydrogen || (flags & FASTSASA_ATOM_HYDROGEN) == 0) &&
                   (include_hetatm || (flags & FASTSASA_ATOM_HETATM) == 0);

        if (keep && plan != NULL && plan->n_selections > 0) {
            keep = (plan->atom_masks[atom] & 1u) != 0u;
        }
        if (keep) ++count;
    }
    if (count <= 0) return 0;
    if (count == n_atoms) return 1;
    *indices = (int *)malloc(sizeof(int) * (size_t)count);
    if (*indices == NULL) return 0;
    for (int atom = 0; atom < n_atoms; ++atom) {
        unsigned char flags = topology->atom_flags != NULL ? topology->atom_flags[atom] : 0;
        int keep = (include_hydrogen || (flags & FASTSASA_ATOM_HYDROGEN) == 0) &&
                   (include_hetatm || (flags & FASTSASA_ATOM_HETATM) == 0);

        if (keep && plan != NULL && plan->n_selections > 0) {
            keep = (plan->atom_masks[atom] & 1u) != 0u;
        }
        if (keep) (*indices)[out++] = atom;
    }
    *n_filtered = count;
    return 1;
}

static int
build_class_selection_masks(const fastsasa_owned_topology *topology,
                            const selection_plan *selections,
                            const radius_config *config,
                            const int *source_indices,
                            int n_mask_atoms,
                            unsigned int **class_masks,
                            int *n_class_masks)
{
    unsigned int *masks;
    int n_masks;

    *class_masks = NULL;
    *n_class_masks = 0;
    if (topology == NULL || n_mask_atoms <= 0) return 0;

    n_masks = selections != NULL && selections->n_selections > 0 ? selections->n_selections * 4 : 3;
    if (n_masks > 31) return 0;
    masks = (unsigned int *)calloc((size_t)n_mask_atoms, sizeof(unsigned int));
    if (masks == NULL) return 0;

    for (int atom = 0; atom < n_mask_atoms; ++atom) {
        int source = source_indices != NULL ? source_indices[atom] : atom;
        int atom_class = configured_class(topology->residue_names[source],
                                          topology->atom_names[source],
                                          topology->elements[source],
                                          config);

        if (selections != NULL && selections->n_selections > 0) {
            for (int selection = 0; selection < selections->n_selections; ++selection) {
                const int base = selection * 4;

                if ((selections->atom_masks[atom] & (1u << selection)) == 0u) continue;
                masks[atom] |= 1u << base;
                if (atom_class == FASTSASA_CLASS_POLAR) masks[atom] |= 1u << (base + 1);
                else if (atom_class == FASTSASA_CLASS_APOLAR) masks[atom] |= 1u << (base + 2);
                else masks[atom] |= 1u << (base + 3);
            }
        } else {
            if (atom_class == FASTSASA_CLASS_POLAR) masks[atom] |= 1u;
            else if (atom_class == FASTSASA_CLASS_APOLAR) masks[atom] |= 1u << 1;
            else masks[atom] |= 1u << 2;
        }
    }

    *class_masks = masks;
    *n_class_masks = n_masks;
    return 1;
}

static int
build_filtered_topology(const fastsasa_owned_topology *topology,
                        const int *indices,
                        int n_filtered,
                        fastsasa_topology *filtered,
                        double **filtered_radii,
                        int **filtered_residue_ids)
{
    *filtered_radii = NULL;
    *filtered_residue_ids = NULL;
    if (indices == NULL) {
        filtered->radii = topology->radii;
        filtered->residue_ids = topology->residue_ids;
        filtered->n_atoms = topology->n_atoms;
        filtered->n_residues = topology->n_residues;
        return 1;
    }
    *filtered_radii = (double *)malloc(sizeof(double) * (size_t)n_filtered);
    *filtered_residue_ids = (int *)malloc(sizeof(int) * (size_t)n_filtered);
    if (*filtered_radii == NULL || *filtered_residue_ids == NULL) {
        free(*filtered_radii);
        free(*filtered_residue_ids);
        *filtered_radii = NULL;
        *filtered_residue_ids = NULL;
        return 0;
    }
    for (int atom = 0; atom < n_filtered; ++atom) {
        int source = indices[atom];

        (*filtered_radii)[atom] = topology->radii[source];
        (*filtered_residue_ids)[atom] = topology->residue_ids[source];
    }
    filtered->radii = *filtered_radii;
    filtered->residue_ids = *filtered_residue_ids;
    filtered->n_atoms = n_filtered;
    filtered->n_residues = topology->n_residues;
    return 1;
}

static int
trajectory_reader_open(trajectory_reader *reader,
                       const char *path)
{
    memset(reader, 0, sizeof(*reader));
    if (has_suffix(path, ".dcd") || has_suffix(path, ".DCD")) {
        if (!fastsasa_dcd_open(&reader->dcd, path)) return 0;
        reader->format = TRAJECTORY_FORMAT_DCD;
        reader->n_atoms = reader->dcd.n_atoms;
        return 1;
    }
    if (has_suffix(path, ".xtc") || has_suffix(path, ".XTC")) {
        if (!fastsasa_xtc_open(&reader->xtc, path)) return 0;
        reader->format = TRAJECTORY_FORMAT_XTC;
        reader->n_atoms = reader->xtc.n_atoms;
        return 1;
    }
    return 0;
}

static void
trajectory_reader_close(trajectory_reader *reader)
{
    if (reader == NULL) return;
    if (reader->format == TRAJECTORY_FORMAT_DCD) fastsasa_dcd_close(&reader->dcd);
    else if (reader->format == TRAJECTORY_FORMAT_XTC) fastsasa_xtc_close(&reader->xtc);
    memset(reader, 0, sizeof(*reader));
}

static int
trajectory_reader_read_frame_soa(trajectory_reader *reader,
                                 double *x,
                                 double *y,
                                 double *z)
{
    if (reader->format == TRAJECTORY_FORMAT_DCD) return fastsasa_dcd_read_frame_soa(&reader->dcd, x, y, z);
    if (reader->format == TRAJECTORY_FORMAT_XTC) return fastsasa_xtc_read_frame_soa(&reader->xtc, x, y, z);
    return 0;
}

static int
trajectory_reader_seek_frame(trajectory_reader *reader,
                             int frame)
{
    if (reader->format == TRAJECTORY_FORMAT_DCD) return fastsasa_dcd_seek_frame(&reader->dcd, frame);
    if (reader->format == TRAJECTORY_FORMAT_XTC) return fastsasa_xtc_seek_frame(&reader->xtc, frame);
    return 0;
}

static int
trajectory_reader_frame_count(const trajectory_reader *reader)
{
    if (reader->format == TRAJECTORY_FORMAT_DCD) return reader->dcd.n_frames;
    if (reader->format == TRAJECTORY_FORMAT_XTC) return reader->xtc.n_frames;
    return 0;
}

/*
 * One prefetchable batch read. The reader thread fills the next batch's
 * buffers while the compute backend works on the current batch, hiding both
 * file I/O latency and per-frame decode/filter cost behind the calculation.
 * Only the producer touches the reader and the frame cursor between spawn
 * and join, so no locking is needed.
 */
typedef struct {
    trajectory_reader *reader;
    const frame_plan *plan;
    int *frame_position;
    int *frames_processed;
    const int *filter_indices;
    int n_calc_atoms;
    int batch_size;
    double *x_full;
    double *y_full;
    double *z_full;
    double *x;
    double *y;
    double *z;
    int *frame_indices;
    int batch_count;
    int seek_failed_frame;
} batch_read_job;

static void
fill_batch(batch_read_job *job)
{
    job->batch_count = 0;
    job->seek_failed_frame = -1;
    while (job->batch_count < job->batch_size &&
           (job->plan->stream_all ||
            *job->frame_position < job->plan->n_indices)) {
        double *x = job->x + (size_t)job->n_calc_atoms * (size_t)job->batch_count;
        double *y = job->y + (size_t)job->n_calc_atoms * (size_t)job->batch_count;
        double *z = job->z + (size_t)job->n_calc_atoms * (size_t)job->batch_count;
        int requested_frame;

        if (!frame_plan_next(job->plan, *job->frame_position, &requested_frame)) break;
        if (!trajectory_reader_seek_frame(job->reader, requested_frame)) {
            if (job->plan->stream_all) break;
            job->seek_failed_frame = requested_frame;
            return;
        }
        if (job->filter_indices != NULL) {
            if (!trajectory_reader_read_frame_soa(job->reader, job->x_full,
                                                  job->y_full, job->z_full)) {
                break;
            }
            for (int atom = 0; atom < job->n_calc_atoms; ++atom) {
                int source = job->filter_indices[atom];

                x[atom] = job->x_full[source];
                y[atom] = job->y_full[source];
                z[atom] = job->z_full[source];
            }
        } else if (!trajectory_reader_read_frame_soa(job->reader, x, y, z)) {
            break;
        }
        job->frame_indices[job->batch_count] = requested_frame;
        ++job->batch_count;
        ++*job->frame_position;
        ++*job->frames_processed;
    }
}

static void *
fill_batch_thread(void *argument)
{
    fill_batch((batch_read_job *)argument);
    return NULL;
}

/* Same deterministic Fibonacci ordering as the trajectory engine and the
 * structure CLI, so exported surface points reproduce the reported SASA. */
static double *
surface_test_points_new(int n_points)
{
    const double pi = 3.14159265358979323846;
    const double dlong = pi * (3.0 - sqrt(5.0));
    const double dz = 2.0 / (double)n_points;
    double longitude = 0.0;
    double z = 1.0 - dz / 2.0;
    double *points = (double *)malloc(sizeof(double) * 3u * (size_t)n_points);

    if (points == NULL) return NULL;
    for (int i = 0; i < n_points; ++i) {
        const double r = sqrt(1.0 - z * z);
        points[3 * i] = cos(longitude) * r;
        points[3 * i + 1] = sin(longitude) * r;
        points[3 * i + 2] = z;
        z -= dz;
        longitude += dlong;
    }
    return points;
}

static int
write_i32(FILE *out, int32_t value)
{
    return fwrite(&value, sizeof(value), 1, out) == 1;
}

/* Writes the buffered surface points as a CHARMM-style DCD with a fixed
 * point count per frame (frames shorter than the widest one repeat their
 * first point). VMD loads it onto an empty "mol new atoms N" molecule via
 * its compiled DCD reader, which is far faster than parsing text XYZ. */
static int
write_surface_dcd(FILE *out,
                  const float *coords,
                  const long *counts,
                  int frames)
{
    const float origin[3] = {0.0f, 0.0f, 0.0f};
    int32_t icntrl[20] = {0};
    char title[80];
    long slots = 0;
    size_t offset = 0;
    float *plane;

    for (int frame = 0; frame < frames; ++frame) {
        if (counts[frame] > slots) slots = counts[frame];
    }
    if (slots <= 0 || slots > INT32_MAX / 4) return 0;
    plane = (float *)malloc(sizeof(float) * (size_t)slots);
    if (plane == NULL) return 0;

    icntrl[0] = frames;   /* NSET */
    icntrl[1] = 0;        /* ISTART */
    icntrl[2] = 1;        /* NSAVC */
    icntrl[3] = frames;   /* NSTEP */
    icntrl[19] = 24;      /* CHARMM version */
    memset(title, ' ', sizeof(title));
    memcpy(title, "FastSASA accessible surface points", 34);

    if (!write_i32(out, 84) || fwrite("CORD", 1, 4, out) != 4 ||
        fwrite(icntrl, sizeof(int32_t), 20, out) != 20 || !write_i32(out, 84) ||
        !write_i32(out, 4 + 80) || !write_i32(out, 1) ||
        fwrite(title, 1, 80, out) != 80 || !write_i32(out, 4 + 80) ||
        !write_i32(out, 4) || !write_i32(out, (int32_t)slots) || !write_i32(out, 4)) {
        free(plane);
        return 0;
    }
    for (int frame = 0; frame < frames; ++frame) {
        const long count = counts[frame];
        const float *first = count > 0 ? coords + 3u * offset : origin;

        for (int axis = 0; axis < 3; ++axis) {
            for (long point = 0; point < slots; ++point) {
                const float *p = point < count ? coords + 3u * (offset + (size_t)point) : first;

                plane[point] = p[axis];
            }
            if (!write_i32(out, (int32_t)(4 * slots)) ||
                fwrite(plane, sizeof(float), (size_t)slots, out) != (size_t)slots ||
                !write_i32(out, (int32_t)(4 * slots))) {
                free(plane);
                return 0;
            }
        }
        offset += (size_t)count;
    }
    free(plane);
    return 1;
}

static void
print_usage(const char *program,
            FILE *out)
{
    fprintf(out,
            "usage: %s trajectory --topology FILE --trajectory FILE [options]\n"
            "       %s topology.pdb|topology.cif|topology.psf trajectory.dcd|trajectory.xtc [options]\n"
            "options: --frames spec --batch-size n --probe-radius n --resolution n --precision fp64|fp32 --output file\n"
            "         --residue|--summary --config-file file --hydrogen --hetatm --threads n\n"
            "         --backend auto|vulkan|cuda|cpu --cpu --surface-points file --surface-resolution n\n"
            "         --shrake-rupley|--lee-richards --classes --filter expression\n"
            "         --select 'expression|name, expression' --help --version\n",
            program,
            program);
}

static int
trajectory_host_alloc(void **ptr,
                      size_t bytes,
                      int pageable_host)
{
    if (!pageable_host) return fastsasa_host_alloc(ptr, bytes);
    *ptr = malloc(bytes);
    return *ptr != NULL ? FASTSASA_SUCCESS : FASTSASA_MEMORY_ERROR;
}

static void
trajectory_host_free(void *ptr,
                     int pageable_host)
{
    if (ptr == NULL) return;
    if (pageable_host) free(ptr);
    else fastsasa_host_free(ptr);
}

static int
run_topology_trajectory(const char *topology_path,
                        const char *trajectory_path,
                        const char *frame_spec,
                        int batch_size,
                        double probe_radius,
                        int resolution,
                        int residue_output,
                        int summary_output,
                        int class_output,
                        radius_config *config,
                        const char *filter_command,
                        const char *const *selection_commands,
                        int n_selections,
                        int algorithm,
                        int include_hydrogen,
                        int include_hetatm,
                        int force_cpu,
                        int precision,
                        int n_threads,
                        const char *output_path,
                        const char *surface_points_path,
                        int surface_resolution)
{
    FILE *out = stdout;
    FILE *surface_out = NULL;
    int surface_gpu_fallback_noted = 0;
    double *surface_test_points = NULL;
    double *surface_expanded_radii = NULL;
    unsigned char *surface_exposed = NULL;
    int surface_n_points = 0;
    int surface_xyz = 0;
    int surface_dcd = 0;
    float *surface_xyz_coords = NULL;   /* all frames' points, packed */
    size_t surface_xyz_capacity = 0;
    size_t surface_xyz_count = 0;
    long *surface_frame_counts = NULL;  /* points per exported frame */
    int surface_frames = 0;
    int surface_frames_capacity = 0;
    fastsasa_owned_topology topology;
    fastsasa_context *context = NULL;
    trajectory_reader reader;
    fastsasa_parameters parameters = {probe_radius, 100, FASTSASA_ALGORITHM_SHRAKE_RUPLEY, FASTSASA_PRECISION_FP64};
    fastsasa_topology sasa_topology;
    double *x_batch = NULL;
    double *y_batch = NULL;
    double *z_batch = NULL;
    double *x_full = NULL;
    double *y_full = NULL;
    double *z_full = NULL;
    double *total_sasa = NULL;
    double *residue_sasa = NULL;
    double *selection_sasa = NULL;
    double *selection_sasa_sum = NULL;
    double *selection_sasa_sum_compensation = NULL;
    unsigned int *calc_selection_masks = NULL;
    int calc_n_selections = 0;
    selection_plan selections;
    selection_plan filter;
    frame_plan frames_to_read;
    int *filter_indices = NULL;
    int n_calc_atoms = 0;
    double *filtered_radii = NULL;
    int *filtered_residue_ids = NULL;
    int frame_position = 0;
    int frames_processed = 0;
    int *batch_frame_indices = NULL;
    int status;
    double wall_start;
    double wall_seconds;
    double gpu_seconds = 0.0;
    double total_sasa_sum = 0.0;
    double total_sasa_sum_compensation = 0.0;
    int known_frames;
    int need_total_sasa;
    int pageable_host = force_cpu;

    memset(&topology, 0, sizeof(topology));
    memset(&selections, 0, sizeof(selections));
    memset(&filter, 0, sizeof(filter));
    memset(&frames_to_read, 0, sizeof(frames_to_read));

    if (output_path != NULL) {
        out = fopen(output_path, "w");
        if (out == NULL) {
            fprintf(stderr, "failed to open output file %s\n", output_path);
            return 1;
        }
    }
    if (!read_topology(topology_path, config, &topology)) {
        fprintf(stderr, "failed to read topology: %s\n", topology_path);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (!trajectory_reader_open(&reader, trajectory_path)) {
        fprintf(stderr, "failed to read trajectory: %s\n", trajectory_path);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (reader.n_atoms != topology.n_atoms) {
        fprintf(stderr, "atom count mismatch: topology=%d trajectory=%d\n", topology.n_atoms, reader.n_atoms);
        trajectory_reader_close(&reader);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (n_selections > 0 && residue_output) {
        fprintf(stderr, "selection and residue output modes are separate CSV modes\n");
        trajectory_reader_close(&reader);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (class_output && residue_output) {
        fprintf(stderr, "trajectory --classes is not supported with residue output\n");
        trajectory_reader_close(&reader);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (filter_command != NULL) {
        if (!build_expression_plan(&filter, &topology, filter_command, "filter")) {
            fprintf(stderr, "failed to build trajectory atom filter\n");
            selection_plan_release(&filter);
            trajectory_reader_close(&reader);
            fastsasa_topology_free(&topology);
            if (out != stdout) fclose(out);
            return 1;
        }
    }
    if (!build_filter_indices(&filter,
                              &topology,
                              include_hydrogen,
                              include_hetatm,
                              &filter_indices,
                              &n_calc_atoms)) {
        fprintf(stderr, "trajectory calculation universe contains no atoms\n");
        selection_plan_release(&filter);
        trajectory_reader_close(&reader);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (!build_selection_plan(&selections, &topology, selection_commands, n_selections)) {
        fprintf(stderr, "failed to build reusable selection masks\n");
        selection_plan_release(&filter);
        trajectory_reader_close(&reader);
        fastsasa_topology_free(&topology);
        if (out != stdout) fclose(out);
        return 1;
    }
    if (batch_size < 0) {
        fprintf(stderr, "invalid trajectory --batch-size value: %d\n", batch_size);
        status = FASTSASA_INVALID_ARGUMENT;
        goto cleanup_before_context;
    }
    if (resolution <= 0) {
        fprintf(stderr, "invalid trajectory --resolution value: %d\n", resolution);
        status = FASTSASA_INVALID_ARGUMENT;
        goto cleanup_before_context;
    }
    known_frames = trajectory_reader_frame_count(&reader);
    if (!frame_plan_from_spec(&frames_to_read, frame_spec != NULL ? frame_spec : ":", known_frames)) {
        fprintf(stderr, "invalid trajectory --frames spec: %s\n", frame_spec != NULL ? frame_spec : ":");
        status = FASTSASA_INVALID_ARGUMENT;
        goto cleanup_before_context;
    }
    {
        const char *requested_backend = getenv("FASTSASA_BACKEND");

        if (requested_backend != NULL && strcmp(requested_backend, "cpu") == 0) {
            force_cpu = 1;
            pageable_host = 1;
        }
        if (!force_cpu) {
            const int context_status = fastsasa_context_create(&context);

            if (context_status != FASTSASA_SUCCESS) {
                const int automatic = requested_backend == NULL ||
                                      requested_backend[0] == '\0' ||
                                      strcmp(requested_backend, "auto") == 0;
                if (automatic) {
                    fprintf(stderr,
                            "no GPU backend available; using threaded CPU with %d thread(s)\n",
                            n_threads);
                    force_cpu = 1;
                    pageable_host = 1;
                } else {
                    fprintf(stderr, "failed to create FastSASA context: %s (%s)\n",
                            fastsasa_status_string(context_status),
                            fastsasa_last_error());
                    selection_plan_release(&selections);
                    selection_plan_release(&filter);
                    frame_plan_release(&frames_to_read);
                    free(filter_indices);
                    trajectory_reader_close(&reader);
                    fastsasa_topology_free(&topology);
                    if (out != stdout) fclose(out);
                    return 1;
                }
            } else if (strcmp(fastsasa_context_backend(context), "vulkan") == 0) {
                pageable_host = 1;
            }
            if (!force_cpu && context != NULL) {
                const int precision_status =
                    fastsasa_context_set_precision(context, precision);

                if (precision_status != FASTSASA_SUCCESS) {
                    const int automatic = requested_backend == NULL ||
                                          requested_backend[0] == '\0' ||
                                          strcmp(requested_backend, "auto") == 0;
                    if (automatic) {
                        fprintf(stderr,
                                "requested GPU precision is unavailable; using threaded CPU with %d thread(s)\n",
                                n_threads);
                        fastsasa_context_free(context);
                        context = NULL;
                        force_cpu = 1;
                        pageable_host = 1;
                    } else {
                        fprintf(stderr, "failed to set GPU precision: %s (%s)\n",
                                fastsasa_status_string(precision_status),
                                fastsasa_last_error());
                        status = precision_status;
                        goto cleanup_before_context;
                    }
                }
            }
        }
    }
    if (batch_size == 0) {
        if (force_cpu) {
            batch_size = 8;
        } else {
            batch_size = fastsasa_recommended_trajectory_batch_size(n_calc_atoms,
                                                                  known_frames,
                                                                  resolution,
                                                                  n_selections > 0);
            if (batch_size <= 0) batch_size = 8;
        }
    }

    if (!build_filtered_topology(&topology,
                                 filter_indices,
                                 n_calc_atoms,
                                 &sasa_topology,
                                 &filtered_radii,
                                 &filtered_residue_ids)) {
        fprintf(stderr, "failed to build filtered topology\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (filter_indices != NULL &&
        !compact_selection_plan(&selections, filter_indices, n_calc_atoms, topology.n_atoms, "selection")) {
        fprintf(stderr, "failed to remap selection masks onto filtered topology\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (class_output) {
        if (!build_class_selection_masks(&topology,
                                         &selections,
                                         config,
                                         filter_indices,
                                         n_calc_atoms,
                                         &calc_selection_masks,
                                         &calc_n_selections)) {
            fprintf(stderr, "failed to build SASA class masks\n");
            status = FASTSASA_MEMORY_ERROR;
            goto cleanup;
        }
    } else if (n_selections > 0) {
        calc_selection_masks = selections.atom_masks;
        calc_n_selections = n_selections;
    }

    need_total_sasa = n_selections == 0;
    /* Two batch-buffer sets: the reader thread prefetches the next batch
     * into one set while the backend computes on the other. */
    if (trajectory_host_alloc((void **)&x_batch, sizeof(double) * 2u * (size_t)n_calc_atoms * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS ||
        trajectory_host_alloc((void **)&y_batch, sizeof(double) * 2u * (size_t)n_calc_atoms * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS ||
        trajectory_host_alloc((void **)&z_batch, sizeof(double) * 2u * (size_t)n_calc_atoms * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS ||
        (need_total_sasa &&
         trajectory_host_alloc((void **)&total_sasa, sizeof(double) * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS)) {
        fprintf(stderr, "failed to allocate trajectory batch\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (filter_indices != NULL &&
        (trajectory_host_alloc((void **)&x_full, sizeof(double) * (size_t)topology.n_atoms, pageable_host) != FASTSASA_SUCCESS ||
         trajectory_host_alloc((void **)&y_full, sizeof(double) * (size_t)topology.n_atoms, pageable_host) != FASTSASA_SUCCESS ||
         trajectory_host_alloc((void **)&z_full, sizeof(double) * (size_t)topology.n_atoms, pageable_host) != FASTSASA_SUCCESS)) {
        fprintf(stderr, "failed to allocate filtered trajectory staging buffers\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (calc_n_selections > 0 &&
        trajectory_host_alloc((void **)&selection_sasa, sizeof(double) * (size_t)calc_n_selections * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS) {
        fprintf(stderr, "failed to allocate selection trajectory batch\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (calc_n_selections > 0) {
        selection_sasa_sum = (double *)calloc((size_t)calc_n_selections, sizeof(double));
        selection_sasa_sum_compensation = (double *)calloc((size_t)calc_n_selections, sizeof(double));
        if (selection_sasa_sum == NULL || selection_sasa_sum_compensation == NULL) {
            fprintf(stderr, "failed to allocate selection summary\n");
            status = FASTSASA_MEMORY_ERROR;
            goto cleanup;
        }
    }
    batch_frame_indices = (int *)malloc(sizeof(int) * 2u * (size_t)batch_size);
    if (batch_frame_indices == NULL) {
        fprintf(stderr, "failed to allocate trajectory frame index batch\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }
    if (residue_output &&
        trajectory_host_alloc((void **)&residue_sasa, sizeof(double) * (size_t)topology.n_residues * (size_t)batch_size, pageable_host) != FASTSASA_SUCCESS) {
        fprintf(stderr, "failed to allocate residue trajectory batch\n");
        status = FASTSASA_MEMORY_ERROR;
        goto cleanup;
    }

    parameters.algorithm = algorithm;
    parameters.n_points = resolution;
    parameters.precision = precision;
    if (force_cpu && algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS &&
        precision == FASTSASA_PRECISION_FP32) {
        fprintf(stderr,
                "FastSASA: warning: CPU Lee-Richards is FP64-only; --precision fp32 ignored for this algorithm\n");
    }

    if (!summary_output && class_output && n_selections > 0) fprintf(out, "frame,selection,total_sasa,polar_sasa,apolar_sasa,unknown_sasa,seconds\n");
    else if (!summary_output && class_output) fprintf(out, "frame,total_sasa,polar_sasa,apolar_sasa,unknown_sasa,seconds\n");
    else if (!summary_output && n_selections > 0) fprintf(out, "frame,selection,selection_sasa,seconds\n");
    else if (!summary_output && residue_output) fprintf(out, "frame,residue,total_sasa,residue_sasa,seconds\n");
    else if (!summary_output) fprintf(out, "frame,total_sasa,seconds\n");

    if (surface_points_path != NULL) {
        /* Lee-Richards resolutions count slices; sample the surface with the
         * default 100 Shrake-Rupley points in that case. surface_resolution
         * lets a caller decouple the exported point density from the
         * reported SASA's --resolution, so one pass can compute high-density
         * totals and a lighter-density surface together. */
        if (surface_resolution > 0) {
            surface_n_points = surface_resolution;
        } else {
            surface_n_points = algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS ? 100 : resolution;
        }
        {
            /* A .xyz target writes a VMD-loadable multi-frame XYZ file and a
             * .dcd target the equivalent binary DCD (load it onto "mol new
             * atoms N"), both with a fixed slot count; unused slots repeat
             * the frame's first point so they render invisibly. Any other
             * name streams "x y z atom" text blocks per frame. */
            const size_t path_length = strlen(surface_points_path);
            surface_xyz = path_length >= 4 &&
                          strcmp(surface_points_path + path_length - 4, ".xyz") == 0;
            surface_dcd = path_length >= 4 &&
                          strcmp(surface_points_path + path_length - 4, ".dcd") == 0;
        }
        surface_out = fopen(surface_points_path, surface_dcd ? "wb" : "w");
        surface_test_points = surface_test_points_new(surface_n_points);
        surface_expanded_radii = (double *)malloc(sizeof(double) * (size_t)n_calc_atoms);
        surface_exposed = (unsigned char *)malloc((size_t)n_calc_atoms * (size_t)surface_n_points);
        if (surface_out == NULL || surface_test_points == NULL ||
            surface_expanded_radii == NULL || surface_exposed == NULL) {
            fprintf(stderr, "failed to prepare surface-point export %s\n", surface_points_path);
            status = FASTSASA_MEMORY_ERROR;
            goto cleanup;
        }
        for (int atom = 0; atom < n_calc_atoms; ++atom) {
            surface_expanded_radii[atom] = sasa_topology.radii[atom] + probe_radius;
        }
    }

    wall_start = now_seconds();
    {
        const size_t batch_stride = (size_t)n_calc_atoms * (size_t)batch_size;
        batch_read_job jobs[2];
        fastsasa_thread reader_thread;
        int reader_thread_active = 0;
        int current = 0;

        for (int side = 0; side < 2; ++side) {
            jobs[side].reader = &reader;
            jobs[side].plan = &frames_to_read;
            jobs[side].frame_position = &frame_position;
            jobs[side].frames_processed = &frames_processed;
            jobs[side].filter_indices = filter_indices;
            jobs[side].n_calc_atoms = n_calc_atoms;
            jobs[side].batch_size = batch_size;
            jobs[side].x_full = x_full;
            jobs[side].y_full = y_full;
            jobs[side].z_full = z_full;
            jobs[side].x = x_batch + (size_t)side * batch_stride;
            jobs[side].y = y_batch + (size_t)side * batch_stride;
            jobs[side].z = z_batch + (size_t)side * batch_stride;
            jobs[side].frame_indices = batch_frame_indices + (size_t)side * (size_t)batch_size;
            jobs[side].batch_count = 0;
            jobs[side].seek_failed_frame = -1;
        }
        fill_batch(&jobs[current]);

    while (1) {
        batch_read_job *job = &jobs[current];
        int batch_count;
        double start, end, seconds_per_frame;
        fastsasa_soa_frames frames;

        if (job->seek_failed_frame >= 0) {
            fprintf(stderr, "failed to seek trajectory frame: %d\n",
                    job->seek_failed_frame);
            status = FASTSASA_INVALID_ARGUMENT;
            goto cleanup;
        }
        batch_count = job->batch_count;
        if (batch_count == 0) break;
        if (frames_to_read.stream_all || frame_position < frames_to_read.n_indices) {
            if (fastsasa_thread_create(&reader_thread, fill_batch_thread,
                                       &jobs[1 - current]) == 0) {
                reader_thread_active = 1;
            } else {
                /* Fall back to synchronous reading for the next batch. */
                fill_batch(&jobs[1 - current]);
            }
        } else {
            jobs[1 - current].batch_count = 0;
            jobs[1 - current].seek_failed_frame = -1;
        }
        frames.x = job->x;
        frames.y = job->y;
        frames.z = job->z;
        frames.n_frames = batch_count;
        start = now_seconds();
        if (calc_n_selections > 0) {
            if (force_cpu) {
                status = fastsasa_cpu_calc_trajectory_soa_selection(&sasa_topology,
                                                                  &frames,
                                                                  calc_selection_masks,
                                                                  calc_n_selections,
                                                                  &parameters,
                                                                  n_threads,
                                                                  total_sasa,
                                                                  selection_sasa);
            } else {
                status = fastsasa_context_calc_trajectory_soa_selection(context,
                                                                      &sasa_topology,
                                                                      &frames,
                                                                      calc_selection_masks,
                                                                      calc_n_selections,
                                                                      &parameters,
                                                                      total_sasa,
                                                                      selection_sasa);
            }
        } else {
            if (force_cpu) {
                status = fastsasa_cpu_calc_trajectory_soa(&sasa_topology,
                                                        &frames,
                                                        &parameters,
                                                        n_threads,
                                                        total_sasa,
                                                        NULL,
                                                        residue_output ? residue_sasa : NULL);
            } else {
                status = fastsasa_context_calc_trajectory_soa(context,
                                                            &sasa_topology,
                                                            &frames,
                                                            &parameters,
                                                            total_sasa,
                                                            NULL,
                                                            residue_output ? residue_sasa : NULL);
            }
        }
        end = now_seconds();
        {
        const int *frame_ids = job->frame_indices;
        if (status != FASTSASA_SUCCESS) {
            if (reader_thread_active) {
                fastsasa_thread_join(&reader_thread);
                reader_thread_active = 0;
            }
            fprintf(stderr, "FastSASA failed on frame batch starting at %d: %s (%s)\n",
                    frame_ids[0],
                    fastsasa_status_string(status),
                    fastsasa_last_error());
            goto cleanup;
        }
        gpu_seconds += end - start;
        seconds_per_frame = (end - start) / (double)batch_count;
        for (int i = 0; i < batch_count; ++i) {
            if (total_sasa != NULL) {
                kahan_add(total_sasa[i], &total_sasa_sum, &total_sasa_sum_compensation);
            }
            if (calc_n_selections > 0) {
                for (int selection = 0; selection < calc_n_selections; ++selection) {
                    kahan_add(selection_sasa[(size_t)i * (size_t)calc_n_selections + (size_t)selection],
                              &selection_sasa_sum[selection],
                              &selection_sasa_sum_compensation[selection]);
                }
            }
            if (summary_output) continue;
            if (class_output && n_selections > 0) {
                for (int selection = 0; selection < n_selections; ++selection) {
                    const size_t base = (size_t)i * (size_t)calc_n_selections + (size_t)selection * 4u;

                    fprintf(out,
                            "%d,%s,%.12f,%.12f,%.12f,%.12f,%.9f\n",
                            frame_ids[i],
                            selections.names[selection],
                            selection_sasa[base],
                            selection_sasa[base + 1u],
                            selection_sasa[base + 2u],
                            selection_sasa[base + 3u],
                            seconds_per_frame);
                }
            } else if (class_output) {
                const size_t base = (size_t)i * (size_t)calc_n_selections;
                const double polar = selection_sasa[base];
                const double apolar = selection_sasa[base + 1u];
                const double unknown = selection_sasa[base + 2u];

                fprintf(out,
                        "%d,%.12f,%.12f,%.12f,%.12f,%.9f\n",
                        frame_ids[i],
                        total_sasa[i],
                        polar,
                        apolar,
                        unknown,
                        seconds_per_frame);
            } else if (n_selections > 0) {
                for (int selection = 0; selection < n_selections; ++selection) {
                    fprintf(out,
                            "%d,%s,%.12f,%.9f\n",
                            frame_ids[i],
                            selections.names[selection],
                            selection_sasa[(size_t)i * (size_t)calc_n_selections + (size_t)selection],
                            seconds_per_frame);
                }
            } else if (residue_output) {
                for (int residue = 0; residue < topology.n_residues; ++residue) {
                    fprintf(out,
                            "%d,%d,%.12f,%.12f,%.9f\n",
                            frame_ids[i],
                            residue,
                            total_sasa[i],
                            residue_sasa[(size_t)i * (size_t)topology.n_residues + (size_t)residue],
                            seconds_per_frame);
                }
            } else {
                fprintf(out, "%d,%.12f,%.9f\n", frame_ids[i], total_sasa[i], seconds_per_frame);
            }
        }
        }
        if (surface_out != NULL) {
            /* Vulkan surface-point export is FP64 only (surface_expanded_radii/
             * surface_test_points are always double). It falls back to the
             * threaded CPU kernel below on any failure - a CUDA-backed
             * context (no exposed-points kernel this round), a Vulkan device
             * without shaderFloat64, or any other runtime error. */
            const int try_gpu_surface = !force_cpu && context != NULL &&
                                        strcmp(fastsasa_context_backend(context), "vulkan") == 0;

            for (int i = 0; i < batch_count; ++i) {
                const double *fx = job->x + (size_t)n_calc_atoms * (size_t)i;
                const double *fy = job->y + (size_t)n_calc_atoms * (size_t)i;
                const double *fz = job->z + (size_t)n_calc_atoms * (size_t)i;
                long exposed_total = 0;
                int surface_status = FASTSASA_INVALID_ARGUMENT;

                if (try_gpu_surface) {
                    fastsasa_sr_input surface_input;

                    memset(&surface_input, 0, sizeof(surface_input));
                    surface_input.n_atoms = n_calc_atoms;
                    surface_input.n_points = surface_n_points;
                    surface_input.x = fx;
                    surface_input.y = fy;
                    surface_input.z = fz;
                    surface_input.radii = surface_expanded_radii;
                    surface_input.test_points = surface_test_points;
                    surface_status = fastsasa_context_shrake_rupley_exposed_points_cell_list(
                        context, &surface_input, surface_exposed);
                    if (surface_status != FASTSASA_SUCCESS && !surface_gpu_fallback_noted) {
                        /* Results stay correct either way; say once why the
                         * documented GPU speedup is missing. */
                        const char *reason = fastsasa_last_error();
                        fprintf(stderr,
                                "FastSASA: note: Vulkan surface-point export unavailable (%s); using the threaded CPU kernel\n",
                                reason != NULL && reason[0] != '\0' ? reason : "no detail");
                        surface_gpu_fallback_noted = 1;
                    }
                }
                if (surface_status != FASTSASA_SUCCESS) {
                    surface_status = fastsasa_cpu_exposed_points(
                        n_calc_atoms, surface_n_points, fx, fy, fz,
                        surface_expanded_radii, surface_test_points,
                        n_threads, surface_exposed);
                }
                if (surface_status != FASTSASA_SUCCESS) {
                    status = FASTSASA_INVALID_ARGUMENT;
surface_export_failed:
                    fprintf(stderr, "surface-point export failed on frame %d\n", job->frame_indices[i]);
                    if (reader_thread_active) {
                        fastsasa_thread_join(&reader_thread);
                        reader_thread_active = 0;
                    }
                    goto cleanup;
                }
                for (size_t k = 0; k < (size_t)n_calc_atoms * (size_t)surface_n_points; ++k) {
                    exposed_total += surface_exposed[k];
                }
                if (surface_xyz || surface_dcd) {
                    if (surface_frames == surface_frames_capacity) {
                        const int grown = surface_frames_capacity == 0 ? 64 : surface_frames_capacity * 2;
                        long *replacement = (long *)realloc(surface_frame_counts, sizeof(long) * (size_t)grown);

                        if (replacement == NULL) {
                            status = FASTSASA_MEMORY_ERROR;
                            goto surface_export_failed;
                        }
                        surface_frame_counts = replacement;
                        surface_frames_capacity = grown;
                    }
                    if (surface_xyz_count + (size_t)exposed_total > surface_xyz_capacity) {
                        size_t grown = surface_xyz_capacity == 0 ? 1u << 20 : surface_xyz_capacity;
                        float *replacement;

                        while (grown < surface_xyz_count + (size_t)exposed_total) grown *= 2u;
                        replacement = (float *)realloc(surface_xyz_coords, sizeof(float) * 3u * grown);
                        if (replacement == NULL) {
                            status = FASTSASA_MEMORY_ERROR;
                            goto surface_export_failed;
                        }
                        surface_xyz_coords = replacement;
                        surface_xyz_capacity = grown;
                    }
                    surface_frame_counts[surface_frames++] = exposed_total;
                } else {
                    fprintf(surface_out, "# frame %d points %ld\n", job->frame_indices[i], exposed_total);
                }
                for (int atom = 0; atom < n_calc_atoms; ++atom) {
                    const double radius = surface_expanded_radii[atom];
                    const unsigned char *row = surface_exposed + (size_t)atom * (size_t)surface_n_points;

                    for (int point = 0; point < surface_n_points; ++point) {
                        /* Round once in double so the streamed text and the
                         * float-buffered XYZ forms print identical coordinates. */
                        const double px = round((fx[atom] + radius * surface_test_points[3 * point]) * 1000.0) / 1000.0;
                        const double py = round((fy[atom] + radius * surface_test_points[3 * point + 1]) * 1000.0) / 1000.0;
                        const double pz = round((fz[atom] + radius * surface_test_points[3 * point + 2]) * 1000.0) / 1000.0;

                        if (!row[point]) continue;
                        if (surface_xyz || surface_dcd) {
                            float *slot = surface_xyz_coords + 3u * surface_xyz_count++;
                            slot[0] = (float)px;
                            slot[1] = (float)py;
                            slot[2] = (float)pz;
                        } else {
                            fprintf(surface_out, "%.3f %.3f %.3f %d\n", px, py, pz, atom);
                        }
                    }
                }
            }
        }
        if (reader_thread_active) {
            fastsasa_thread_join(&reader_thread);
            reader_thread_active = 0;
        }
        current = 1 - current;
    }
        if (reader_thread_active) fastsasa_thread_join(&reader_thread);
    }
    wall_seconds = now_seconds() - wall_start;
    if (summary_output) {
        double gpu_fps = gpu_seconds > 0.0 ? (double)frames_processed / gpu_seconds : 0.0;
        double wall_fps = wall_seconds > 0.0 ? (double)frames_processed / wall_seconds : 0.0;

        if (class_output && n_selections > 0) {
            fprintf(out, "frames,known_frames,atoms,batch_size,selection,total_sasa_sum,polar_sasa_sum,apolar_sasa_sum,unknown_sasa_sum,gpu_seconds,wall_seconds,gpu_frames_per_second,wall_frames_per_second\n");
            for (int selection = 0; selection < n_selections; ++selection) {
                const size_t base = (size_t)selection * 4u;

                fprintf(out,
                        "%d,%d,%d,%d,%s,%.12f,%.12f,%.12f,%.12f,%.9f,%.9f,%.6f,%.6f\n",
                        frames_processed, known_frames, n_calc_atoms, batch_size, selections.names[selection],
                        selection_sasa_sum[base],
                        selection_sasa_sum[base + 1u],
                        selection_sasa_sum[base + 2u],
                        selection_sasa_sum[base + 3u],
                        gpu_seconds, wall_seconds, gpu_fps, wall_fps);
            }
        } else if (class_output) {
            fprintf(out, "frames,known_frames,atoms,batch_size,total_sasa_sum,polar_sasa_sum,apolar_sasa_sum,unknown_sasa_sum,gpu_seconds,wall_seconds,gpu_frames_per_second,wall_frames_per_second\n");
            fprintf(out,
                    "%d,%d,%d,%d,%.12f,%.12f,%.12f,%.12f,%.9f,%.9f,%.6f,%.6f\n",
                    frames_processed, known_frames, n_calc_atoms, batch_size, total_sasa_sum,
                    selection_sasa_sum[0],
                    selection_sasa_sum[1],
                    selection_sasa_sum[2],
                    gpu_seconds, wall_seconds, gpu_fps, wall_fps);
        } else if (n_selections > 0) {
            fprintf(out, "frames,known_frames,atoms,batch_size,selection,selection_sasa_sum,gpu_seconds,wall_seconds,gpu_frames_per_second,wall_frames_per_second\n");
            for (int selection = 0; selection < n_selections; ++selection) {
                fprintf(out,
                        "%d,%d,%d,%d,%s,%.12f,%.9f,%.9f,%.6f,%.6f\n",
                        frames_processed, known_frames, n_calc_atoms, batch_size, selections.names[selection],
                        selection_sasa_sum[selection],
                        gpu_seconds, wall_seconds, gpu_fps, wall_fps);
            }
        } else {
            fprintf(out, "frames,known_frames,atoms,batch_size,total_sasa_sum,gpu_seconds,wall_seconds,gpu_frames_per_second,wall_frames_per_second\n");
            fprintf(out,
                    "%d,%d,%d,%d,%.12f,%.9f,%.9f,%.6f,%.6f\n",
                    frames_processed, known_frames, n_calc_atoms, batch_size, total_sasa_sum,
                    gpu_seconds, wall_seconds, gpu_fps, wall_fps);
        }
    }
    status = FASTSASA_SUCCESS;

cleanup:
    if (surface_out != NULL && surface_dcd && status == FASTSASA_SUCCESS && surface_frames > 0) {
        if (!write_surface_dcd(surface_out, surface_xyz_coords, surface_frame_counts, surface_frames)) {
            fprintf(stderr, "failed to write surface-point DCD %s\n", surface_points_path);
            status = FASTSASA_MEMORY_ERROR;
        }
    }
    if (surface_out != NULL && surface_xyz && status == FASTSASA_SUCCESS && surface_frames > 0) {
        const float origin[3] = {0.0f, 0.0f, 0.0f};
        long slots = 0;
        size_t offset = 0;

        for (int frame = 0; frame < surface_frames; ++frame) {
            if (surface_frame_counts[frame] > slots) slots = surface_frame_counts[frame];
        }
        for (int frame = 0; frame < surface_frames; ++frame) {
            const long count = surface_frame_counts[frame];
            const float *first = count > 0 ? surface_xyz_coords + 3u * offset : origin;

            fprintf(surface_out, "%ld\nFastSASA accessible surface frame %d\n", slots, frame);
            for (long point = 0; point < slots; ++point) {
                const float *p = point < count ? surface_xyz_coords + 3u * (offset + (size_t)point) : first;

                fprintf(surface_out, "X %.3f %.3f %.3f\n", p[0], p[1], p[2]);
            }
            offset += (size_t)count;
        }
    }
    if (surface_out != NULL) fclose(surface_out);
    free(surface_test_points);
    free(surface_expanded_radii);
    free(surface_exposed);
    free(surface_xyz_coords);
    free(surface_frame_counts);
    trajectory_host_free(x_batch, pageable_host);
    trajectory_host_free(y_batch, pageable_host);
    trajectory_host_free(z_batch, pageable_host);
    trajectory_host_free(x_full, pageable_host);
    trajectory_host_free(y_full, pageable_host);
    trajectory_host_free(z_full, pageable_host);
    trajectory_host_free(total_sasa, pageable_host);
    trajectory_host_free(selection_sasa, pageable_host);
    trajectory_host_free(residue_sasa, pageable_host);
    free(batch_frame_indices);
    free(selection_sasa_sum);
    free(selection_sasa_sum_compensation);
    fastsasa_context_free(context);
    selection_plan_release(&selections);
    selection_plan_release(&filter);
    frame_plan_release(&frames_to_read);
    if (class_output) free(calc_selection_masks);
    free(filter_indices);
    free(filtered_radii);
    free(filtered_residue_ids);
    trajectory_reader_close(&reader);
    fastsasa_topology_free(&topology);
    if (out != stdout) fclose(out);
    return status == FASTSASA_SUCCESS ? 0 : 1;

cleanup_before_context:
    selection_plan_release(&selections);
    selection_plan_release(&filter);
    frame_plan_release(&frames_to_read);
    free(filter_indices);
    trajectory_reader_close(&reader);
    fastsasa_topology_free(&topology);
    if (out != stdout) fclose(out);
    return status == FASTSASA_SUCCESS ? 0 : 1;
}

int
fastsasa_trajectory_cli_main(int argc,
                           char **argv)
{
    const char *frame_spec = NULL;
    int batch_size = 0;
    double probe_radius = 1.4;
    int resolution = 100;
    int resolution_set = 0;
    int residue_output = 0;
    int summary_output = 0;
    int class_output = 0;
    int positional = 0;
    const char *config_path = NULL;
    const char *filter_command = NULL;
    const char **selection_commands = NULL;
    int n_selections = 0;
    int algorithm = FASTSASA_ALGORITHM_SHRAKE_RUPLEY;
    int force_cpu = 0;
    int include_hydrogen = 0;
    int include_hetatm = 0;
    int precision = FASTSASA_PRECISION_FP64;
    int n_threads = fastsasa_cpu_default_threads();
    const char *output_path = NULL;
    const char *surface_points_path = NULL;
    int surface_resolution = 0;
    radius_config config;
    int ret;

    memset(&config, 0, sizeof(config));
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0], stdout);
            return 0;
        }
        if (strcmp(argv[i], "--version") == 0) {
            fprintf(stdout, "FastSASA %s\n", FASTSASA_VERSION);
            return 0;
        }
    }
    if (argc < 3) {
        print_usage(argv[0], stderr);
        return 1;
    }
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--config-file") == 0 || strcmp(argv[i], "-c") == 0) {
            if (++i >= argc) goto usage_fail;
            config_path = argv[i];
        } else if (strcmp(argv[i], "--select") == 0) {
            const char **new_commands;

            if (++i >= argc) goto usage_fail;
            new_commands = (const char **)realloc(selection_commands, sizeof(char *) * (size_t)(n_selections + 1));
            if (new_commands == NULL) {
                free(selection_commands);
                return 1;
            }
            selection_commands = new_commands;
            selection_commands[n_selections++] = argv[i];
        } else if (strcmp(argv[i], "--filter") == 0) {
            if (++i >= argc) goto usage_fail;
            filter_command = argv[i];
        } else if (strcmp(argv[i], "--frames") == 0) {
            if (++i >= argc) goto usage_fail;
            frame_spec = argv[i];
        } else if (strcmp(argv[i], "--batch-size") == 0) {
            if (++i >= argc || !parse_int_strict(argv[i], &batch_size)) goto usage_fail;
        } else if (strcmp(argv[i], "--probe-radius") == 0 || strcmp(argv[i], "-p") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "missing --probe-radius value\n");
                goto usage_fail;
            }
            if (!parse_double_strict(argv[i], &probe_radius) || probe_radius < 0.0) {
                fprintf(stderr, "invalid --probe-radius value: %s\n", argv[i]);
                goto usage_fail;
            }
        } else if (strcmp(argv[i], "--resolution") == 0 || strcmp(argv[i], "-n") == 0) {
            if (++i >= argc || !parse_int_strict(argv[i], &resolution)) goto usage_fail;
            resolution_set = 1;
        } else if (strcmp(argv[i], "--precision") == 0) {
            if (++i >= argc) goto usage_fail;
            if (strcmp(argv[i], "fp64") == 0) precision = FASTSASA_PRECISION_FP64;
            else if (strcmp(argv[i], "fp32") == 0) precision = FASTSASA_PRECISION_FP32;
            else {
                fprintf(stderr, "--precision must be fp64 or fp32\n");
                goto usage_fail;
            }
        } else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
            if (++i >= argc) goto usage_fail;
            output_path = argv[i];
        } else if (strcmp(argv[i], "--surface-points") == 0) {
            if (++i >= argc) goto usage_fail;
            surface_points_path = argv[i];
        } else if (strcmp(argv[i], "--surface-resolution") == 0) {
            if (++i >= argc || !parse_int_strict(argv[i], &surface_resolution)) goto usage_fail;
        } else if (strcmp(argv[i], "--shrake-rupley") == 0 || strcmp(argv[i], "-S") == 0) {
            algorithm = FASTSASA_ALGORITHM_SHRAKE_RUPLEY;
        } else if (strcmp(argv[i], "--lee-richards") == 0 || strcmp(argv[i], "-L") == 0) {
            algorithm = FASTSASA_ALGORITHM_LEE_RICHARDS;
        } else if (strcmp(argv[i], "--hydrogen") == 0 || strcmp(argv[i], "-Y") == 0) {
            include_hydrogen = 1;
        } else if (strcmp(argv[i], "--hetatm") == 0 || strcmp(argv[i], "-H") == 0) {
            include_hetatm = 1;
        } else if (strcmp(argv[i], "--classes") == 0) {
            class_output = 1;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            force_cpu = 1;
        } else if (strcmp(argv[i], "--backend") == 0) {
            if (++i >= argc) goto usage_fail;
            if (strcmp(argv[i], "cpu") == 0) {
                force_cpu = 1;
            } else if (strcmp(argv[i], "auto") == 0 ||
                       strcmp(argv[i], "cuda") == 0 ||
                       strcmp(argv[i], "vulkan") == 0) {
                /* The library dispatches on FASTSASA_BACKEND; the flag is the
                 * user-facing spelling of the same request. */
                if (fastsasa_setenv("FASTSASA_BACKEND", argv[i]) != 0) goto usage_fail;
            } else {
                fprintf(stderr, "--backend must be auto, vulkan, cuda, or cpu\n");
                goto usage_fail;
            }
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc || !parse_int_strict(argv[i], &n_threads) || n_threads <= 0) goto usage_fail;
        } else if (strcmp(argv[i], "residue") == 0 || strcmp(argv[i], "--residue") == 0) {
            residue_output = 1;
        } else if (strcmp(argv[i], "summary") == 0 || strcmp(argv[i], "--summary") == 0) {
            summary_output = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown trajectory option: %s\n", argv[i]);
            goto usage_fail;
        } else if (positional == 0) {
            frame_spec = argv[i];
            ++positional;
        } else if (positional == 1) {
            if (!parse_int_strict(argv[i], &batch_size)) goto usage_fail;
            ++positional;
        } else {
            goto usage_fail;
        }
    }
    if (summary_output && residue_output) {
        fprintf(stderr, "trajectory mode accepts only one of --summary and --residue\n");
        goto usage_fail;
    }
    if (filter_command == NULL) {
        fprintf(stderr,
                "FastSASA: warning: trajectory input has no --filter; all eligible topology atoms will participate in SASA. "
                "Hydrogen and PDB/mmCIF HETATM records are excluded unless requested. "
                "For standard protein SASA use --filter protein.\n");
    }
    if (algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS && !resolution_set) {
        resolution = 20;
    }
    if (config_path != NULL) {
        if (!load_radius_config(config_path, &config)) {
            fprintf(stderr, "failed to read config file: %s\n", config_path);
            free(selection_commands);
            return 1;
        }
    } else {
        (void)load_default_radius_config(&config);
    }
    ret = run_topology_trajectory(argv[1],
                                  argv[2],
                                  frame_spec,
                                  batch_size,
                                  probe_radius,
                                  resolution,
                                  residue_output,
                                  summary_output,
                                  class_output,
                                  &config,
                                  filter_command,
                                  selection_commands,
                                  n_selections,
                                  algorithm,
                                  include_hydrogen,
                                  include_hetatm,
                                  force_cpu,
                                  precision,
                                  n_threads,
                                  output_path,
                                  surface_points_path,
                                  surface_resolution);
    radius_config_free(&config);
    free(selection_commands);
    return ret;

usage_fail:
    print_usage(argv[0], stderr);
    radius_config_free(&config);
    free(selection_commands);
    return 1;
}
