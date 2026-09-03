#ifndef FASTSASA_TOPOLOGY_H
#define FASTSASA_TOPOLOGY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fastsasa_topology_options {
    FASTSASA_TOPOLOGY_INCLUDE_HETATM = 1,
    FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN = 2,
    FASTSASA_TOPOLOGY_SKIP_UNKNOWN = 4,
    FASTSASA_TOPOLOGY_HALT_AT_UNKNOWN = 8,
    FASTSASA_TOPOLOGY_JOIN_MODELS = 16
};

enum fastsasa_atom_flags {
    FASTSASA_ATOM_HETATM = 1,
    FASTSASA_ATOM_HYDROGEN = 2
};

typedef double (*fastsasa_radius_callback)(const char *residue_name,
                                         const char *atom_name,
                                         const char *element,
                                         void *userdata);

#define FASTSASA_MAX_SELECTION_NAME 50

typedef void (*fastsasa_selection_warning_callback)(const char *message,
                                                  void *userdata);

typedef struct fastsasa_owned_topology {
    double *x;
    double *y;
    double *z;
    double *radii;
    int *residue_ids;
    int *residue_numbers;
    char **residue_number_strings;
    char **atom_names;
    char **residue_names;
    char **chain_ids;
    char **segment_ids;
    char **elements;
    int n_atoms;
    int n_residues;
    /* Record/element metadata retained for trajectory-safe calculation masks. */
    unsigned char *atom_flags;
} fastsasa_owned_topology;

int fastsasa_topology_read_mmcif(const char *path,
                               int options,
                               fastsasa_radius_callback radius_callback,
                               void *userdata,
                               fastsasa_owned_topology *topology);

void fastsasa_topology_free(fastsasa_owned_topology *topology);

size_t fastsasa_sizeof_owned_topology(void);
size_t fastsasa_offsetof_owned_topology_atom_flags(void);

int fastsasa_topology_selection_mask(const char *command,
                                   const fastsasa_owned_topology *topology,
                                   unsigned int bit,
                                   unsigned int *atom_masks,
                                   char *name,
                                   size_t name_size);

int fastsasa_topology_selection_mask_ex(const char *command,
                                      const fastsasa_owned_topology *topology,
                                      unsigned int bit,
                                      unsigned int *atom_masks,
                                      char *name,
                                      size_t name_size,
                                      fastsasa_selection_warning_callback warning_callback,
                                      void *warning_userdata,
                                      int *warning_count);

#ifdef __cplusplus
}
#endif

#endif
