#ifndef FASTSASA_RADIUS_H
#define FASTSASA_RADIUS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

double fastsasa_element_radius(const char *element);
int fastsasa_infer_element(char *dst,
                         size_t dst_size,
                         const char *atom_name,
                         const char *column_element);
int fastsasa_element_class(const char *element);
const char *fastsasa_default_config_candidate(size_t index);
/* Maps common MD force-field residue names (protonation states, tautomers,
 * disulfide cysteine) to the standard residue they parameterize, e.g. HIE ->
 * HIS, CYX -> CYS, ASH -> ASP. Returns NULL when the name has no alias. */
const char *fastsasa_canonical_residue(const char *residue_name);

#ifdef __cplusplus
}
#endif

#endif
