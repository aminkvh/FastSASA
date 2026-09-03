#include "fastsasa_topology.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

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
validate_one(const char *path)
{
    fastsasa_owned_topology default_topology;
    fastsasa_owned_topology include_topology;
    fastsasa_owned_topology joined_topology;
    std::vector<unsigned int> masks;
    char name[64];
    int ca_count;
    int no_match_count;
    std::set<std::string> chains;

    std::memset(&default_topology, 0, sizeof(default_topology));
    std::memset(&include_topology, 0, sizeof(include_topology));
    std::memset(&joined_topology, 0, sizeof(joined_topology));

    if (!fastsasa_topology_read_mmcif(path, 0, nullptr, nullptr, &default_topology)) {
        std::fprintf(stderr, "failed default mmCIF read: %s\n", path);
        return 0;
    }
    if (!fastsasa_topology_read_mmcif(path,
                                    FASTSASA_TOPOLOGY_INCLUDE_HETATM | FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN,
                                    nullptr,
                                    nullptr,
                                    &include_topology)) {
        std::fprintf(stderr, "failed include-all mmCIF read: %s\n", path);
        fastsasa_topology_free(&default_topology);
        return 0;
    }
    if (!fastsasa_topology_read_mmcif(path,
                                    FASTSASA_TOPOLOGY_JOIN_MODELS,
                                    nullptr,
                                    nullptr,
                                    &joined_topology)) {
        std::fprintf(stderr, "failed join-model mmCIF read: %s\n", path);
        fastsasa_topology_free(&default_topology);
        fastsasa_topology_free(&include_topology);
        return 0;
    }

    masks.assign((size_t)default_topology.n_atoms, 0u);
    if (!fastsasa_topology_selection_mask("ca, name ca", &default_topology, 1u, masks.data(), name, sizeof(name))) {
        std::fprintf(stderr, "failed CA selection: %s\n", path);
        goto fail;
    }
    if (!fastsasa_topology_selection_mask("none, name definitely_not_an_atom", &default_topology, 2u, masks.data(), name, sizeof(name))) {
        std::fprintf(stderr, "failed no-match selection: %s\n", path);
        goto fail;
    }
    ca_count = count_mask(masks.data(), default_topology.n_atoms, 1u);
    no_match_count = count_mask(masks.data(), default_topology.n_atoms, 2u);

    for (int atom = 0; atom < default_topology.n_atoms; ++atom) {
        if (default_topology.chain_ids[atom] != nullptr) chains.insert(default_topology.chain_ids[atom]);
    }

    if (default_topology.n_atoms <= 0 ||
        default_topology.n_residues <= 0 ||
        include_topology.n_atoms < default_topology.n_atoms ||
        joined_topology.n_atoms < default_topology.n_atoms ||
        no_match_count != 0 ||
        chains.empty()) {
        std::fprintf(stderr,
                     "mmCIF corpus validation failed for %s: atoms=%d include=%d joined=%d residues=%d ca=%d no_match=%d chains=%zu\n",
                     path,
                     default_topology.n_atoms,
                     include_topology.n_atoms,
                     joined_topology.n_atoms,
                     default_topology.n_residues,
                     ca_count,
                     no_match_count,
                     chains.size());
        goto fail;
    }

    std::printf("mmcif_corpus,path,%s,atoms,%d,include_atoms,%d,joined_atoms,%d,residues,%d,ca,%d,chains,%zu,status,pass\n",
                path,
                default_topology.n_atoms,
                include_topology.n_atoms,
                joined_topology.n_atoms,
                default_topology.n_residues,
                ca_count,
                chains.size());
    fastsasa_topology_free(&default_topology);
    fastsasa_topology_free(&include_topology);
    fastsasa_topology_free(&joined_topology);
    return 1;

fail:
    fastsasa_topology_free(&default_topology);
    fastsasa_topology_free(&include_topology);
    fastsasa_topology_free(&joined_topology);
    return 0;
}

int
main(int argc,
     char **argv)
{
    const char *default_paths[] = {
        "tests/data/1d3z.cif",
        "tests/data/1sui.cif",
        "tests/data/1ubq.cif",
        "tests/data/2isk.cif",
        "tests/data/2jo4.cif",
        "tests/data/3bkr.cif",
        "tests/data/3gnn.cif",
        "tests/data/5dx9.cif",
        "tests/data/5hdn.cif",
        "tests/data/7cma-assembly1.cif"
    };
    int ok = 1;

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) ok = validate_one(argv[i]) && ok;
    } else {
        for (size_t i = 0; i < sizeof(default_paths) / sizeof(default_paths[0]); ++i) {
            ok = validate_one(default_paths[i]) && ok;
        }
    }
    return ok ? 0 : 1;
}
