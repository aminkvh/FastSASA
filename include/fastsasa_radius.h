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
/* The built-in default radius table (share/protor.config, ProtOr), as text
 * that is not NUL-terminated; *length receives its size in bytes. */
const char *fastsasa_default_config_text(size_t *length);
/* Maps common MD force-field residue names (protonation states, tautomers,
 * disulfide cysteine) to the standard residue they parameterize, e.g. HIE ->
 * HIS, CYX -> CYS, ASH -> ASP. Returns NULL when the name has no alias. */
const char *fastsasa_canonical_residue(const char *residue_name);
/* Maps CHARMM/GROMACS atom names to the PDB names used by the radius tables
 * (ILE CD -> CD1, OT1/OT2 -> O/OXT, ACE cap CAY/CY/OY, nucleic O1P/O2P).
 * Returns 1 when an alias exists; *canonical_residue is then the residue to
 * look up, or NULL for the caller's own residue name. */
int fastsasa_canonical_atom(const char *residue_name,
                            const char *atom_name,
                            const char **canonical_residue,
                            const char **canonical_atom);
/* Element symbol from a topology mass (PSF and other force-field formats,
 * whose type column is not an element). Masses below 4.5 are hydrogen,
 * including repartitioned hydrogens; heavy atoms that gave mass to their
 * hydrogens (HMR) are matched too, with atom_name (may be NULL) deciding
 * between candidate elements. Returns 0 when the mass names no element
 * (virtual sites). */
int fastsasa_element_from_mass(char *dst,
                               size_t dst_size,
                               double mass,
                               const char *atom_name);

#ifdef __cplusplus
}
#endif

#endif
