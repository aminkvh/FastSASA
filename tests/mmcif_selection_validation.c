#include "fastsasa_topology.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
count_selection_warning(const char *message,
                        void *userdata)
{
    int *count = (int *)userdata;

    (void)message;
    ++*count;
}

static int
count_mask(const unsigned int *masks,
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
validate_segment_selection_aliases(void)
{
    double x[1] = {0.0};
    double y[1] = {0.0};
    double z[1] = {0.0};
    double radii[1] = {1.7};
    int residue_ids[1] = {0};
    int residue_numbers[1] = {677};
    char *residue_number_strings[1] = {(char *)"677"};
    char *atom_names[1] = {(char *)"CA"};
    char *residue_names[1] = {(char *)"LEU"};
    char *chain_ids[1] = {(char *)"A"};
    char *segment_ids[1] = {(char *)"AP"};
    char *elements[1] = {(char *)"C"};
    fastsasa_owned_topology topology = {
        x, y, z, radii, residue_ids, residue_numbers,
        residue_number_strings, atom_names, residue_names,
        chain_ids, segment_ids, elements, 1, 1, NULL
    };
    unsigned int masks[1] = {0u};
    char name[64];

    if (!fastsasa_topology_selection_mask("chain_a, chain A", &topology, 1u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("chain_ap, chain AP", &topology, 2u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("segid_ap, segid AP", &topology, 4u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("segname_ap, segname AP", &topology, 8u, masks, name, sizeof(name))) return 0;

    return count_mask(masks, 1, 1u) == 1 &&
           count_mask(masks, 1, 2u) == 0 &&
           count_mask(masks, 1, 4u) == 1 &&
           count_mask(masks, 1, 8u) == 1;
}

static int
validate_negative_residue_selection_ranges(void)
{
    double x[7] = {0.0};
    double y[7] = {0.0};
    double z[7] = {0.0};
    double radii[7] = {1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7};
    int residue_ids[7] = {0, 1, 2, 3, 4, 5, 6};
    int residue_numbers[7] = {-20, -17, -10, 0, 5, 10, 15};
    char *residue_number_strings[7] = {
        (char *)"-20", (char *)"-17", (char *)"-10", (char *)"0",
        (char *)"5", (char *)"10", (char *)"15"
    };
    char *atom_names[7] = {
        (char *)"CA", (char *)"CA", (char *)"CA", (char *)"CA",
        (char *)"CA", (char *)"CA", (char *)"CA"
    };
    char *residue_names[7] = {
        (char *)"ALA", (char *)"ALA", (char *)"ALA", (char *)"ALA",
        (char *)"ALA", (char *)"ALA", (char *)"ALA"
    };
    char *chain_ids[7] = {
        (char *)"A", (char *)"A", (char *)"A", (char *)"A",
        (char *)"A", (char *)"A", (char *)"A"
    };
    char *segment_ids[7] = {
        (char *)"", (char *)"", (char *)"", (char *)"",
        (char *)"", (char *)"", (char *)""
    };
    char *elements[7] = {
        (char *)"C", (char *)"C", (char *)"C", (char *)"C",
        (char *)"C", (char *)"C", (char *)"C"
    };
    fastsasa_owned_topology topology = {
        x, y, z, radii, residue_ids, residue_numbers,
        residue_number_strings, atom_names, residue_names,
        chain_ids, segment_ids, elements, 7, 7, NULL
    };
    unsigned int masks[7] = {0u};
    char name[64];

    if (!fastsasa_topology_selection_mask("neg_single, resi \\-10", &topology, 1u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("neg_range, resi \\-20-\\-15+\\-10-5", &topology, 2u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("open_left, resi -10", &topology, 4u, masks, name, sizeof(name))) return 0;
    if (!fastsasa_topology_selection_mask("open_right, resi 10-", &topology, 8u, masks, name, sizeof(name))) return 0;

    return count_mask(masks, 7, 1u) == 1 &&
           count_mask(masks, 7, 2u) == 5 &&
           count_mask(masks, 7, 4u) == 6 &&
           count_mask(masks, 7, 8u) == 2;
}

static int
validate_selection_edge_behavior(void)
{
    double x[4] = {0.0};
    double y[4] = {0.0};
    double z[4] = {0.0};
    double radii[4] = {1.7, 1.7, 1.7, 1.7};
    int residue_ids[4] = {0, 1, 2, 3};
    int residue_numbers[4] = {1, 1, 2, 10};
    char *residue_number_strings[4] = {
        (char *)"1", (char *)"1A", (char *)"2", (char *)"10"
    };
    char *atom_names[4] = {
        (char *)"CA", (char *)"CB", (char *)"OXT", (char *)"O5'"
    };
    char *residue_names[4] = {
        (char *)"ALA", (char *)"ALA", (char *)"GLY", (char *)"DA"
    };
    char *chain_ids[4] = {
        (char *)"A", (char *)"A", (char *)"B", (char *)"1"
    };
    char *segment_ids[4] = {
        (char *)"", (char *)"", (char *)"", (char *)""
    };
    char *elements[4] = {
        (char *)"C", (char *)"C", (char *)"O", (char *)"P"
    };
    fastsasa_owned_topology topology = {
        x, y, z, radii, residue_ids, residue_numbers,
        residue_number_strings, atom_names, residue_names,
        chain_ids, segment_ids, elements, 4, 4, NULL
    };
    unsigned int masks[4] = {0u};
    char name[128];
    int warnings = 0;
    const char *long_name = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzzzz, name CA";

    if (!fastsasa_topology_selection_mask_ex("icode, resi 1A",
                                           &topology, 1u, masks, name, sizeof(name),
                                           count_selection_warning, &warnings, NULL)) {
        fprintf(stderr, "edge behavior failed: insertion-code selection rejected\n");
        return 0;
    }
    if (warnings != 0 || count_mask(masks, 4, 1u) != 1) {
        fprintf(stderr, "edge behavior failed: insertion-code warnings=%d count=%d\n", warnings, count_mask(masks, 4, 1u));
        return 0;
    }

    warnings = 0;
    if (!fastsasa_topology_selection_mask_ex("warns, name ABCDE+CA+ZZ",
                                           &topology, 2u, masks, name, sizeof(name),
                                           count_selection_warning, &warnings, NULL)) {
        fprintf(stderr, "edge behavior failed: missing-name selection rejected\n");
        return 0;
    }
    if (warnings != 2 || count_mask(masks, 4, 2u) != 1) {
        fprintf(stderr, "edge behavior failed: missing-name warnings=%d count=%d\n", warnings, count_mask(masks, 4, 2u));
        return 0;
    }

    warnings = 0;
    if (!fastsasa_topology_selection_mask_ex("bad_resi, resi A+1AA+1A",
                                           &topology, 4u, masks, name, sizeof(name),
                                           count_selection_warning, &warnings, NULL)) {
        fprintf(stderr, "edge behavior failed: bad-resi selection rejected\n");
        return 0;
    }
    if (warnings != 2 || count_mask(masks, 4, 4u) != 1) {
        fprintf(stderr, "edge behavior failed: bad-resi warnings=%d count=%d\n", warnings, count_mask(masks, 4, 4u));
        return 0;
    }

    if (fastsasa_topology_selection_mask("missing_comma", &topology, 8u, masks, name, sizeof(name))) {
        fprintf(stderr, "edge behavior failed: missing comma accepted\n");
        return 0;
    }
    if (!fastsasa_topology_selection_mask(", name CA", &topology, 8u, masks, name, sizeof(name))) {
        fprintf(stderr, "edge behavior failed: optional selection label rejected\n");
        return 0;
    }
    if (count_mask(masks, 4, 8u) != 1 || strcmp(name, "name_CA") != 0) {
        fprintf(stderr, "edge behavior failed: optional label count=%d name=%s\n", count_mask(masks, 4, 8u), name);
        return 0;
    }
    if (fastsasa_topology_selection_mask("bad, name CA-CB", &topology, 32u, masks, name, sizeof(name))) {
        fprintf(stderr, "edge behavior failed: bad atom-name range accepted\n");
        return 0;
    }

    if (!fastsasa_topology_selection_mask(long_name, &topology, 16u, masks, name, sizeof(name))) {
        fprintf(stderr, "edge behavior failed: long-name selection rejected\n");
        return 0;
    }
    return strlen(name) == FASTSASA_MAX_SELECTION_NAME &&
           strncmp(name, long_name, FASTSASA_MAX_SELECTION_NAME) == 0;
}

int
main(int argc,
     char **argv)
{
    const char *path = argc > 1 ? argv[1] : "tests/data/2isk.cif";
    fastsasa_owned_topology topology;
    unsigned int *masks = NULL;
    char name[64];
    int ca_count;
    int chain_count;
    int segid_no_match_count;
    int ca_or_n_count;
    int not_ca_count;
    int residue_range_count;
    int no_match_count;
    int protein_count;
    int segment_aliases_ok;
    int negative_ranges_ok;
    int edge_behavior_ok;

    if (!fastsasa_topology_read_mmcif(path, FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN, NULL, NULL, &topology)) {
        fprintf(stderr, "failed to read mmCIF topology: %s\n", path);
        return 1;
    }
    masks = calloc((size_t)topology.n_atoms, sizeof(unsigned int));
    if (masks == NULL) {
        fastsasa_topology_free(&topology);
        return 1;
    }

    if (!fastsasa_topology_selection_mask("ca, name ca", &topology, 1u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to select CA atoms from mmCIF metadata\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    ca_count = count_mask(masks, topology.n_atoms, 1u);

    if (!fastsasa_topology_selection_mask("chain_a, chain A", &topology, 2u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to select chain A atoms from mmCIF metadata\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    chain_count = count_mask(masks, topology.n_atoms, 2u);

    if (!fastsasa_topology_selection_mask("segid_a, segid A", &topology, 128u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to evaluate segid selection on mmCIF metadata\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    segid_no_match_count = count_mask(masks, topology.n_atoms, 128u);

    if (!fastsasa_topology_selection_mask("ca_or_n, (name ca or name n) and chain A", &topology, 4u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to evaluate boolean mmCIF selection\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    ca_or_n_count = count_mask(masks, topology.n_atoms, 4u);

    if (!fastsasa_topology_selection_mask("not_ca, not name ca", &topology, 8u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to evaluate negated mmCIF selection\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    not_ca_count = count_mask(masks, topology.n_atoms, 8u);

    if (!fastsasa_topology_selection_mask("residue_range, resi 1-999999", &topology, 16u, masks, name, sizeof(name))) {
        fprintf(stderr, "failed to evaluate residue range mmCIF selection\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    residue_range_count = count_mask(masks, topology.n_atoms, 16u);

    if (!fastsasa_topology_selection_mask("no_match, name definitely_not_an_atom", &topology, 32u, masks, name, sizeof(name))) {
        fprintf(stderr, "valid no-match mmCIF selection should not fail\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    no_match_count = count_mask(masks, topology.n_atoms, 32u);

    if (!fastsasa_topology_selection_mask("protein, protein", &topology, 64u, masks, name, sizeof(name))) {
        fprintf(stderr, "valid protein selection should not fail\n");
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }
    protein_count = count_mask(masks, topology.n_atoms, 64u);
    segment_aliases_ok = validate_segment_selection_aliases();
    negative_ranges_ok = validate_negative_residue_selection_ranges();
    edge_behavior_ok = validate_selection_edge_behavior();

    if (topology.n_atoms <= 0 || topology.n_residues <= 0 || ca_count <= 0 || chain_count <= 0 ||
        ca_or_n_count <= 0 || ca_or_n_count > chain_count || not_ca_count + ca_count != topology.n_atoms ||
        residue_range_count <= 0 || no_match_count != 0 || protein_count <= 0 ||
        segid_no_match_count != 0 || !segment_aliases_ok ||
        !negative_ranges_ok ||
        !edge_behavior_ok) {
        fprintf(stderr,
                "mmCIF metadata validation failed: atoms=%d residues=%d ca=%d chain=%d segid=%d ca_or_n=%d not_ca=%d range=%d no_match=%d protein=%d segment_aliases=%d negative_ranges=%d edge_behavior=%d\n",
                topology.n_atoms,
                topology.n_residues,
                ca_count,
                chain_count,
                segid_no_match_count,
                ca_or_n_count,
                not_ca_count,
                residue_range_count,
                no_match_count,
                protein_count,
                segment_aliases_ok,
                negative_ranges_ok,
                edge_behavior_ok);
        free(masks);
        fastsasa_topology_free(&topology);
        return 1;
    }

    printf("mmcif_selection_validation,atoms,%d,residues,%d,ca,%d,chain_a,%d,segid_a,%d,ca_or_n,%d,not_ca,%d,range,%d,no_match,%d,protein,%d,status,pass\n",
           topology.n_atoms,
           topology.n_residues,
           ca_count,
           chain_count,
           segid_no_match_count,
           ca_or_n_count,
           not_ca_count,
           residue_range_count,
           no_match_count,
           protein_count);
    free(masks);
    fastsasa_topology_free(&topology);
    return 0;
}
