#include "fastsasa_portable.h"
#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_radius.h"
#include "fastsasa_topology.h"
#include "fastsasa_trajectory.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gemmi/cif.hpp>
#include <gemmi/to_cif.hpp>

#ifndef FASTSASA_VERSION
#define FASTSASA_VERSION "unknown"
#endif

extern "C" int fastsasa_trajectory_cli_main(int argc, char **argv);

struct radius_config {
    std::string name;
    std::unordered_map<std::string, double> type_radius;
    std::unordered_map<std::string, std::string> type_class;
    std::unordered_map<std::string, std::string> atom_type;
};

struct rsa_area {
    double total;
    double main_chain;
    double side_chain;
    double polar;
    double apolar;
    double unknown;
};

struct residue_reference {
    const char *name;
    rsa_area area;
};

static rsa_area
zero_rsa_area(void)
{
    rsa_area area = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    return area;
}

/*
 * First atom of a residue, computed once per topology. The writers used to
 * rescan the atom array from index zero for every residue, which is
 * quadratic for large structures (seconds for a 290k-atom assembly). The
 * CLI is single threaded and processes one topology per run, so a static
 * cache keyed on the topology address is safe. Returns n_atoms when the
 * residue has no atoms, matching the old scan's end value.
 */
static int
residue_first_atom(const fastsasa_owned_topology &topology, int residue)
{
    static const fastsasa_owned_topology *cached_topology = nullptr;
    static std::vector<int> first_atom;

    if (cached_topology != &topology ||
        (int)first_atom.size() != topology.n_residues) {
        cached_topology = &topology;
        first_atom.assign(topology.n_residues > 0 ? topology.n_residues : 0, -1);
        for (int atom = 0; atom < topology.n_atoms; ++atom) {
            const int id = topology.residue_ids[atom];
            if (id >= 0 && id < (int)first_atom.size() && first_atom[id] < 0) {
                first_atom[id] = atom;
            }
        }
    }
    if (residue < 0 || residue >= (int)first_atom.size() ||
        first_atom[residue] < 0) {
        return topology.n_atoms;
    }
    return first_atom[residue];
}

struct cli_state {
    double probe_radius = 1.4;
    int n_points = 100;
    int resolution_set = 0;
    int lee_richards = 0;
    int force_cpu = 0;
    int cpu_fallback = 1;
    int precision = FASTSASA_PRECISION_FP64;
    int precision_set = 0;
    const char *surface_points_path = nullptr;
    int n_threads = fastsasa_cpu_default_threads();
    int topology_options = 0;
    int force_cif_input = 0;
    int format_log = 1;
    int format_residue = 0;
    int format_seq = 0;
    int format_rsa = 0;
    int format_pdb = 0;
    int format_cif = 0;
    int format_json = 0;
    int format_xml = 0;
    int format_set = 0;
    int report_classes = 0;
    const char *output_path = nullptr;
    const char *config_path = nullptr;
    int config_loaded = 0;
    int reference_available = 0;
    std::string active_config_path;
    std::vector<std::string> selections;
    radius_config config;
};

static std::string
trim(const std::string &value)
{
    size_t begin = 0;
    size_t end = value.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(begin, end - begin);
}

static std::string
upper(std::string value)
{
    for (char &c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return value;
}

static int
has_suffix(const char *path,
           const char *suffix)
{
    const size_t path_len = std::strlen(path);
    const size_t suffix_len = std::strlen(suffix);

    return path_len >= suffix_len && std::strcmp(path + path_len - suffix_len, suffix) == 0;
}

static void
usage(const char *program,
      FILE *out = stderr)
{
    std::fprintf(out,
                 "usage: %s [options] structure.pdb|structure.cif\n"
                 "       %s trajectory --topology FILE --trajectory FILE [trajectory-options]\n"
                 "options: --shrake-rupley --lee-richards --probe-radius N --resolution N\n"
                 "         --threads N --precision fp64|fp32\n"
                 "         --backend auto|vulkan|cuda|cpu --cpu --no-cpu-fallback\n"
                 "         --config-file FILE --hetatm --hydrogen --unknown guess|skip|halt\n"
                 "         --cif --join-models --classes --select 'expression|name, expression'\n"
                 "         --format log|res|seq|rsa|pdb|cif|json|xml --output FILE\n"
                 "         --surface-points FILE\n"
                 "trajectory-options: --frames SPEC --batch-size N --probe-radius N --resolution N --output FILE --summary --residue\n"
                 "                    --backend auto|vulkan|cuda|cpu --cpu --threads N --precision fp64|fp32\n"
                 "                    --shrake-rupley --lee-richards --config-file FILE --hetatm --hydrogen --filter COMMAND\n"
                 "                    --select COMMAND --classes --help --version\n",
                 program,
                 program);
}

static int
parse_int_strict_cpp(const char *text,
                     int *value)
{
    char *end = nullptr;
    long parsed;

    if (text == nullptr || text[0] == '\0') return 0;
    errno = 0;
    parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) return 0;
    *value = static_cast<int>(parsed);
    return 1;
}

static int
parse_double_strict_cpp(const char *text,
                        double *value)
{
    char *end = nullptr;
    double parsed;

    if (text == nullptr || text[0] == '\0') return 0;
    errno = 0;
    parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}

static int
launch_trajectory_cli(const char *program,
                      const std::vector<std::string> &arguments)
{
    std::vector<char *> trajectory_arguments;
    trajectory_arguments.reserve(arguments.size() + 2u);
    trajectory_arguments.push_back(const_cast<char *>(program));
    for (const std::string &argument : arguments) {
        trajectory_arguments.push_back(const_cast<char *>(argument.c_str()));
    }
    trajectory_arguments.push_back(nullptr);
    return fastsasa_trajectory_cli_main(static_cast<int>(trajectory_arguments.size() - 1u),
                                      trajectory_arguments.data());
}

static int
maybe_launch_trajectory_cli(int argc,
                            char **argv)
{
    const int subcommand = argc > 1 && std::strcmp(argv[1], "trajectory") == 0;
    int has_trajectory_flag = 0;

    for (int i = subcommand ? 2 : 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trajectory") == 0) {
            has_trajectory_flag = 1;
            break;
        }
    }
    if (!subcommand && !has_trajectory_flag) return -1;

    if (subcommand && argc == 3 &&
        (std::strcmp(argv[2], "--help") == 0 || std::strcmp(argv[2], "-h") == 0)) {
        return launch_trajectory_cli(argv[0], {"--help"});
    }
    if (subcommand && argc == 3 && std::strcmp(argv[2], "--version") == 0) {
        std::fprintf(stdout, "FastSASA %s\n", FASTSASA_VERSION);
        return 0;
    }

    if (subcommand && argc > 2 && argv[2][0] != '-') {
        std::vector<std::string> arguments;
        for (int i = 2; i < argc; ++i) arguments.emplace_back(argv[i]);
        return launch_trajectory_cli(argv[0], arguments);
    }

    const char *topology = nullptr;
    const char *trajectory = nullptr;
    const char *frames = nullptr;
    const char *batch_size = nullptr;
    const char *resolution = nullptr;
    const char *output = nullptr;
    int residue = 0;
    int summary = 0;
    std::vector<std::string> forwarded;

    for (int i = subcommand ? 2 : 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--topology") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --topology value\n");
                return 1;
            }
            topology = argv[i];
        } else if (std::strcmp(argv[i], "--trajectory") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --trajectory value\n");
                return 1;
            }
            trajectory = argv[i];
        } else if (std::strcmp(argv[i], "--frames") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --frames value\n");
                return 1;
            }
            frames = argv[i];
        } else if (std::strcmp(argv[i], "--batch-size") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --batch-size value\n");
                return 1;
            }
            batch_size = argv[i];
        } else if (std::strcmp(argv[i], "--resolution") == 0 || std::strcmp(argv[i], "-n") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --resolution value\n");
                return 1;
            }
            resolution = argv[i];
        } else if (std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --output value\n");
                return 1;
            }
            output = argv[i];
        } else if (std::strcmp(argv[i], "--residue") == 0) {
            residue = 1;
        } else if (std::strcmp(argv[i], "--summary") == 0) {
            summary = 1;
        } else {
            forwarded.emplace_back(argv[i]);
        }
    }

    if (topology == nullptr || trajectory == nullptr) {
        std::fprintf(stderr, "trajectory mode requires --topology FILE and --trajectory FILE\n");
        usage(argv[0]);
        return 1;
    }
    if (residue && summary) {
        std::fprintf(stderr, "trajectory mode accepts only one of --summary and --residue\n");
        return 1;
    }

    std::vector<std::string> arguments;
    arguments.emplace_back(topology);
    arguments.emplace_back(trajectory);
    if (frames != nullptr) {
        arguments.emplace_back("--frames");
        arguments.emplace_back(frames);
    }
    if (batch_size != nullptr) {
        arguments.emplace_back("--batch-size");
        arguments.emplace_back(batch_size);
    }
    if (resolution != nullptr) {
        arguments.emplace_back("--resolution");
        arguments.emplace_back(resolution);
    }
    if (output != nullptr) {
        arguments.emplace_back("--output");
        arguments.emplace_back(output);
    }
    if (residue) arguments.emplace_back("--residue");
    if (summary) arguments.emplace_back("--summary");
    arguments.insert(arguments.end(), forwarded.begin(), forwarded.end());
    return launch_trajectory_cli(argv[0], arguments);
}

static double
element_radius(const std::string &element)
{
    return fastsasa_element_radius(element.c_str());
}

static std::string
infer_element(const std::string &atom_name,
              const std::string &column_element)
{
    char element[8];

    fastsasa_infer_element(element, sizeof(element), atom_name.c_str(), column_element.c_str());
    return element;
}

static int
load_radius_config(const char *path,
                   radius_config *config)
{
    std::ifstream input(path);
    std::string line;
    std::string section;

    if (!input) return 0;
    while (std::getline(input, line)) {
        std::string::size_type hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        line = trim(line);
        if (line.empty()) continue;
        if (line == "types:") {
            section = "types";
            continue;
        }
        if (line == "atoms:") {
            section = "atoms";
            continue;
        }
        if (line.back() == ':') {
            section.clear();
            continue;
        }

        std::vector<std::string> fields;
        size_t start = 0;
        while (start < line.size()) {
            while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) ++start;
            size_t end = start;
            while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) ++end;
            if (end > start) fields.push_back(line.substr(start, end - start));
            start = end;
        }

        if (section.empty() && fields.size() >= 2 && upper(fields[0]) == "NAME:") {
            config->name = fields[1];
            continue;
        }
        if (section == "types" && fields.size() >= 2) {
            config->type_radius[upper(fields[0])] = std::atof(fields[1].c_str());
            if (fields.size() >= 3) config->type_class[upper(fields[0])] = upper(fields[2]);
        } else if (section == "atoms" && fields.size() >= 3) {
            config->atom_type[upper(fields[0]) + ":" + upper(fields[1])] = upper(fields[2]);
        }
    }
    return 1;
}

/* The relative-SASA references are ProtOr max-ASA values, so they apply
 * exactly when the loaded table declares itself ProtOr (`name: ProtOr`),
 * whether it was auto-discovered, named by FASTSASA_DEFAULT_CONFIG, or
 * passed with --config-file. Any other table, including the glycan
 * extension (ProtOr-Glycans), reports N/A. */
static void
note_config_loaded(cli_state *state,
                   const char *path)
{
    state->config_loaded = 1;
    state->reference_available = state->config.name == "ProtOr";
    state->active_config_path = path;
}

static int
load_default_radius_config(cli_state *state)
{
    const char *env_path = std::getenv("FASTSASA_DEFAULT_CONFIG");

    if (env_path != nullptr && env_path[0] != '\0' && load_radius_config(env_path, &state->config)) {
        note_config_loaded(state, env_path);
        return 1;
    }
    for (size_t i = 0;; ++i) {
        const char *candidate = fastsasa_default_config_candidate(i);

        if (candidate == nullptr) break;
        if (load_radius_config(candidate, &state->config)) {
            note_config_loaded(state, candidate);
            return 1;
        }
    }
    return 0;
}

static double
configured_radius(const char *residue_name,
                  const char *atom_name,
                  const char *element,
                  void *userdata)
{
    const radius_config *config = static_cast<const radius_config *>(userdata);

    if (config != nullptr) {
        const std::string atom = upper(trim(atom_name ? atom_name : ""));
        const std::string residue = upper(trim(residue_name ? residue_name : ""));
        const std::string key = residue + ":" + atom;
        const std::string any_key = "ANY:" + atom;
        auto atom_it = config->atom_type.find(key);
        if (atom_it == config->atom_type.end()) {
            const char *canonical = fastsasa_canonical_residue(residue.c_str());
            if (canonical != nullptr) atom_it = config->atom_type.find(std::string(canonical) + ":" + atom);
        }
        if (atom_it == config->atom_type.end()) atom_it = config->atom_type.find(any_key);
        if (atom_it != config->atom_type.end()) {
            const auto type_it = config->type_radius.find(atom_it->second);
            if (type_it != config->type_radius.end()) return type_it->second;
        }
        if (!residue.empty() && residue != "?" && residue != ".") {
            const std::string guessed_element = upper(trim(element ? element : ""));
            const double guessed = element_radius(guessed_element);

            if (guessed >= 0.0) {
                std::fprintf(stderr,
                             "FastSASA: warning: atom '%s %s' unknown, guessing element is '%s', and radius %.3f A\n",
                             residue.c_str(),
                             atom.c_str(),
                             guessed_element.c_str(),
                             guessed);
            }
            return guessed;
        }
    }
    return element_radius(element ? element : "");
}

static int
push_string(char ***array,
            int n,
            const std::string &value)
{
    (*array)[n] = static_cast<char *>(std::malloc(value.size() + 1u));
    if ((*array)[n] == nullptr) return 0;
    std::memcpy((*array)[n], value.c_str(), value.size() + 1u);
    return 1;
}

static int
reserve_topology(fastsasa_owned_topology *topology,
                 int capacity)
{
    topology->x = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(capacity)));
    topology->y = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(capacity)));
    topology->z = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(capacity)));
    topology->radii = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(capacity)));
    topology->residue_ids = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(capacity)));
    topology->residue_numbers = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(capacity)));
    topology->residue_number_strings = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    topology->atom_names = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    topology->residue_names = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    topology->chain_ids = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    topology->segment_ids = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    topology->elements = static_cast<char **>(std::calloc(static_cast<size_t>(capacity), sizeof(char *)));
    return topology->x && topology->y && topology->z && topology->radii &&
           topology->residue_ids && topology->residue_numbers && topology->residue_number_strings &&
           topology->atom_names && topology->residue_names && topology->chain_ids &&
           topology->segment_ids && topology->elements;
}

static int
read_pdb_topology(const char *path,
                  const cli_state &state,
                  fastsasa_owned_topology *topology)
{
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::unordered_set<std::string> seen_altlocs;
    std::string line;
    int in_first_model = 1;
    int saw_model = 0;

    if (!input) return 0;
    std::memset(topology, 0, sizeof(*topology));
    while (std::getline(input, line)) lines.push_back(line);
    if (!reserve_topology(topology, static_cast<int>(lines.size()))) return 0;

    std::string last_residue_key;
    int residue_id = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        line = lines[i];
        if (line.compare(0, 5, "MODEL") == 0) {
            if (saw_model && (state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) break;
            saw_model = 1;
            in_first_model = 1;
            continue;
        }
        if (line.compare(0, 6, "ENDMDL") == 0) {
            if ((state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) break;
            in_first_model = 0;
            continue;
        }
        if (!in_first_model && (state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) continue;
        if (line.compare(0, 4, "ATOM") != 0 && line.compare(0, 6, "HETATM") != 0) continue;
        if (line.compare(0, 6, "HETATM") == 0 && (state.topology_options & FASTSASA_TOPOLOGY_INCLUDE_HETATM) == 0) continue;

        const std::string atom_name = trim(line.size() >= 16 ? line.substr(12, 4) : "");
        const std::string altloc = line.size() >= 17 ? line.substr(16, 1) : " ";
        const std::string residue_name = trim(line.size() >= 20 ? line.substr(17, 3) : "");
        const std::string chain = trim(line.size() >= 22 ? line.substr(21, 1) : "");
        const std::string resi = trim(line.size() >= 27 ? line.substr(22, 5) : "");
        const std::string segment = trim(line.size() >= 76 ? line.substr(72, 4) : "");
        const std::string element = infer_element(atom_name, line.size() >= 78 ? line.substr(76, 2) : "");
        const std::string atom_key = chain + ":" + resi + ":" + residue_name + ":" + atom_name;
        char *end = nullptr;
        double x, y, z;
        double radius;

        if ((element == "H" || element == "D") && (state.topology_options & FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN) == 0) continue;
        if (altloc != " " && altloc != "" && !seen_altlocs.insert(atom_key).second) continue;

        if (line.size() < 54 ||
            !parse_double_strict_cpp(trim(line.substr(30, 8)).c_str(), &x) ||
            !parse_double_strict_cpp(trim(line.substr(38, 8)).c_str(), &y) ||
            !parse_double_strict_cpp(trim(line.substr(46, 8)).c_str(), &z)) {
            std::fprintf(stderr, "invalid PDB coordinate record in %s\n", path);
            return 0;
        }

        radius = configured_radius(residue_name.c_str(), atom_name.c_str(), element.c_str(),
                                   state.config_loaded ? const_cast<radius_config *>(&state.config) : nullptr);
        if (radius < 0.0) {
            if (state.topology_options & FASTSASA_TOPOLOGY_HALT_AT_UNKNOWN) return 0;
            if (state.topology_options & FASTSASA_TOPOLOGY_SKIP_UNKNOWN) continue;
            radius = 0.0;
        }

        const int atom = topology->n_atoms;
        const std::string residue_key = chain + ":" + resi + ":" + residue_name;
        if (residue_key != last_residue_key) {
            ++residue_id;
            last_residue_key = residue_key;
        }
        topology->x[atom] = x;
        topology->y[atom] = y;
        topology->z[atom] = z;
        topology->radii[atom] = radius;
        topology->residue_ids[atom] = residue_id;
        topology->residue_numbers[atom] = static_cast<int>(std::strtol(resi.c_str(), &end, 10));
        if (!push_string(&topology->residue_number_strings, atom, resi) ||
            !push_string(&topology->atom_names, atom, atom_name) ||
            !push_string(&topology->residue_names, atom, residue_name) ||
            !push_string(&topology->chain_ids, atom, chain) ||
            !push_string(&topology->segment_ids, atom, segment) ||
            !push_string(&topology->elements, atom, element)) {
            return 0;
        }
        ++topology->n_atoms;
    }
    topology->n_residues = residue_id + 1;
    return topology->n_atoms > 0;
}

static void
make_test_points(int n_points,
                 std::vector<double> *points)
{
    const double pi = 3.14159265358979323846;
    const double dz = 2.0 / n_points;
    const double longitude_step = pi * (3.0 - std::sqrt(5.0));
    double z = 1.0 - dz / 2.0;
    double longitude = 0.0;

    points->resize(static_cast<size_t>(n_points) * 3u);
    for (int i = 0; i < n_points; ++i) {
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        (*points)[3 * i] = std::cos(longitude) * r;
        (*points)[3 * i + 1] = std::sin(longitude) * r;
        (*points)[3 * i + 2] = z;
        z -= dz;
        longitude += longitude_step;
    }
}

static int
run_lee_richards_gpu(const fastsasa_owned_topology &topology,
                     const cli_state &state,
                     std::vector<double> *atom_sasa)
{
    fastsasa_context *context = nullptr;
    fastsasa_sr_input input;
    std::vector<double> expanded_radii(static_cast<size_t>(topology.n_atoms));
    int status;

    if (state.n_points <= 0 || topology.n_atoms <= 0) return 0;
    atom_sasa->assign(static_cast<size_t>(topology.n_atoms), 0.0);
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        expanded_radii[atom] = topology.radii[atom] + state.probe_radius;
    }
    std::memset(&input, 0, sizeof(input));
    input.n_atoms = topology.n_atoms;
    input.n_points = state.n_points;
    input.x = topology.x;
    input.y = topology.y;
    input.z = topology.z;
    input.radii = expanded_radii.data();

    status = fastsasa_context_create(&context);
    if (status != FASTSASA_SUCCESS) return 0;
    status = fastsasa_context_set_precision(context, state.precision);
    if (status != FASTSASA_SUCCESS) {
        fastsasa_context_free(context);
        return 0;
    }
    status = fastsasa_context_lee_richards(context, &input, atom_sasa->data());
    fastsasa_context_free(context);
    return status == FASTSASA_SUCCESS;
}

static int
run_cpu(const fastsasa_owned_topology &topology,
        const cli_state &state,
        std::vector<double> *atom_sasa)
{
    std::vector<double> expanded_radii(static_cast<size_t>(topology.n_atoms));
    std::vector<double> points;
    int status;

    if (topology.n_atoms <= 0) return 0;
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        expanded_radii[atom] = topology.radii[atom] + state.probe_radius;
    }
    atom_sasa->assign(static_cast<size_t>(topology.n_atoms), 0.0);
    if (state.lee_richards) {
        if (state.precision == FASTSASA_PRECISION_FP32) {
            std::fprintf(stderr,
                         "FastSASA: warning: CPU Lee-Richards is FP64-only; --precision fp32 ignored for this algorithm\n");
        }
        status = fastsasa_cpu_lee_richards(topology.n_atoms,
                                         state.n_points,
                                         topology.x,
                                         topology.y,
                                         topology.z,
                                         expanded_radii.data(),
                                         state.n_threads,
                                         atom_sasa->data());
    } else {
        make_test_points(state.n_points, &points);
        status = fastsasa_cpu_shrake_rupley_precision(topology.n_atoms,
                                          state.n_points,
                                          topology.x,
                                          topology.y,
                                          topology.z,
                                          expanded_radii.data(),
                                          points.data(),
                                          state.n_threads,
                                          state.precision,
                                          atom_sasa->data());
    }
    return status == FASTSASA_SUCCESS;
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

static double
compensated_sum(const std::vector<double> &values)
{
    double sum = 0.0;
    double compensation = 0.0;

    for (size_t i = 0; i < values.size(); ++i) {
        kahan_add(values[i], &sum, &compensation);
    }
    return sum;
}

static void
reduce_atom_sasa(const fastsasa_owned_topology &topology,
                 const std::vector<double> &atom_sasa,
                 const std::vector<unsigned int> &selection_masks,
                 const std::vector<std::string> &selection_names,
                 double *total_sasa,
                 std::vector<double> *residue_sasa,
                 std::vector<double> *selection_sasa)
{
    double total_compensation = 0.0;
    std::vector<double> residue_compensation(residue_sasa->size(), 0.0);
    std::vector<double> selection_compensation(selection_sasa->size(), 0.0);

    *total_sasa = 0.0;
    std::fill(residue_sasa->begin(), residue_sasa->end(), 0.0);
    std::fill(selection_sasa->begin(), selection_sasa->end(), 0.0);

    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const double sasa = atom_sasa[static_cast<size_t>(atom)];

        kahan_add(sasa, total_sasa, &total_compensation);
        if (topology.residue_ids[atom] >= 0 &&
            topology.residue_ids[atom] < topology.n_residues) {
            const size_t residue = static_cast<size_t>(topology.residue_ids[atom]);
            kahan_add(sasa, &(*residue_sasa)[residue], &residue_compensation[residue]);
        }
        if (!selection_masks.empty()) {
            for (int selection = 0; selection < static_cast<int>(selection_names.size()); ++selection) {
                if (selection_masks[static_cast<size_t>(atom)] & (1u << selection)) {
                    const size_t selection_index = static_cast<size_t>(selection);
                    kahan_add(sasa,
                              &(*selection_sasa)[selection_index],
                              &selection_compensation[selection_index]);
                }
            }
        }
    }
}

static void
selection_warning_to_stderr(const char *message,
                            void *userdata)
{
    (void)userdata;
    std::fprintf(stderr, "%s\n", message);
}

static int
run_sasa(const fastsasa_owned_topology &topology,
         const cli_state &state,
         std::vector<double> *atom_sasa,
         double *total_sasa,
         std::vector<double> *residue_sasa,
         std::vector<double> *selection_sasa,
         std::vector<std::string> *selection_names,
         std::vector<unsigned int> *selection_masks_out)
{
    fastsasa_context *context = nullptr;
    fastsasa_sr_input input;
    std::vector<double> expanded_radii;
    std::vector<unsigned int> selection_masks;
    int status;

    atom_sasa->assign(static_cast<size_t>(topology.n_atoms), 0.0);
    residue_sasa->assign(static_cast<size_t>(topology.n_residues), 0.0);
    selection_sasa->assign(state.selections.size(), 0.0);
    selection_names->clear();
    selection_masks_out->clear();

    if (!state.selections.empty()) {
        selection_masks.assign(static_cast<size_t>(topology.n_atoms), 0u);
        for (size_t i = 0; i < state.selections.size(); ++i) {
            char name[128];
            if (!fastsasa_topology_selection_mask_ex(state.selections[i].c_str(),
                                                   &topology,
                                                   1u << i,
                                                   selection_masks.data(),
                                                   name,
                                                   sizeof(name),
                                                   selection_warning_to_stderr,
                                                   nullptr,
                                                   nullptr)) {
                std::fprintf(stderr, "failed to parse selection: %s\n", state.selections[i].c_str());
                return 0;
            }
            selection_names->push_back(name);
        }
        *selection_masks_out = selection_masks;
    }

    if (state.lee_richards) {
        int ok = 0;

        if (state.force_cpu) {
            ok = run_cpu(topology, state, atom_sasa);
        } else {
            ok = run_lee_richards_gpu(topology, state, atom_sasa);
            if (!ok && state.cpu_fallback) {
                std::fprintf(stderr, "GPU Lee-Richards failed; trying CPU fallback with %d thread(s)\n", state.n_threads);
                ok = run_cpu(topology, state, atom_sasa);
            }
        }
        if (!ok) {
            return 0;
        }
        reduce_atom_sasa(topology, *atom_sasa, selection_masks, *selection_names,
                         total_sasa, residue_sasa, selection_sasa);
        return 1;
    }

    if (state.force_cpu) {
        if (!run_cpu(topology, state, atom_sasa)) {
            return 0;
        }
        reduce_atom_sasa(topology, *atom_sasa, selection_masks, *selection_names,
                         total_sasa, residue_sasa, selection_sasa);
        return 1;
    }

    std::vector<double> points;
    make_test_points(state.n_points, &points);
    expanded_radii.resize(static_cast<size_t>(topology.n_atoms));
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        expanded_radii[atom] = topology.radii[atom] + state.probe_radius;
    }
    std::memset(&input, 0, sizeof(input));
    input.n_atoms = topology.n_atoms;
    input.n_points = state.n_points;
    input.x = topology.x;
    input.y = topology.y;
    input.z = topology.z;
    input.radii = expanded_radii.data();
    input.test_points = points.data();
    input.reuse_test_points = 1;
    input.residue_ids = topology.residue_ids;
    input.n_residues = topology.n_residues;
    input.residue_sasa = residue_sasa->data();
    input.selection_masks = selection_masks.empty() ? nullptr : selection_masks.data();
    input.n_selections = static_cast<int>(selection_names->size());
    input.selection_sasa = selection_sasa->empty() ? nullptr : selection_sasa->data();
    input.force_double_precision = 0;

    status = fastsasa_context_create(&context);
    if (status == FASTSASA_SUCCESS) {
        status = fastsasa_context_set_precision(context, state.precision);
        if (status == FASTSASA_SUCCESS) {
            status = fastsasa_context_shrake_rupley_cell_list(context, &input,
                                                            atom_sasa->data());
        }
        fastsasa_context_free(context);
    }
    if (status != FASTSASA_SUCCESS) {
        if (!state.cpu_fallback) return 0;
        std::fprintf(stderr, "No usable GPU backend; using CPU with %d thread(s)", state.n_threads);
        if (fastsasa_last_error()[0] != '\0') std::fprintf(stderr, " (%s)", fastsasa_last_error());
        std::fprintf(stderr, "\n");
        if (!run_cpu(topology, state, atom_sasa)) return 0;
        reduce_atom_sasa(topology, *atom_sasa, selection_masks, *selection_names,
                         total_sasa, residue_sasa, selection_sasa);
        return 1;
    }

    *total_sasa = compensated_sum(*atom_sasa);
    return 1;
}

static const char *
output_algorithm_name(const cli_state &state)
{
    return state.lee_richards ? "Lee & Richards" : "Shrake & Rupley";
}

static const char *
fastsasa_algorithm_name(const cli_state &state)
{
    return state.lee_richards ? "Lee-Richards" : "Shrake-Rupley";
}

static const char *
resolution_key(const cli_state &state)
{
    return state.lee_richards ? "slices" : "testpoints";
}

static const char *
safe_text(const char *value)
{
    return value ? value : "";
}

static double
pdb_b_factor(double value)
{
    if (value < 0.0) return 0.0;
    if (value > 999.99) return 999.99;
    return value;
}

static const char *
generator_name(const cli_state &state)
{
    (void)state;
    return "FastSASA";
}

static std::string
json_escape(const char *value)
{
    std::string escaped;
    const char *text = safe_text(value);

    for (const char *p = text; *p; ++p) {
        const unsigned char ch = static_cast<unsigned char>(*p);

        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
            escaped.push_back(static_cast<char>(ch));
        } else if (ch == '\b') escaped += "\\b";
        else if (ch == '\f') escaped += "\\f";
        else if (ch == '\n') escaped += "\\n";
        else if (ch == '\r') escaped += "\\r";
        else if (ch == '\t') escaped += "\\t";
        else if (ch < 0x20u) {
            char encoded[7];
            std::snprintf(encoded, sizeof(encoded), "\\u%04x", ch);
            escaped += encoded;
        } else escaped.push_back(static_cast<char>(ch));
    }
    return escaped;
}

static std::string
xml_escape(const char *value)
{
    std::string escaped;
    const char *text = safe_text(value);

    for (const char *p = text; *p; ++p) {
        if (*p == '&') escaped += "&amp;";
        else if (*p == '<') escaped += "&lt;";
        else if (*p == '>') escaped += "&gt;";
        else if (*p == '"') escaped += "&quot;";
        else escaped.push_back(*p);
    }
    return escaped;
}

static std::string
cif_token(const char *value)
{
    std::string text = safe_text(value);
    int quote = text.empty();

    for (char &c : text) {
        if (c == '\'') c = '"';
        if (std::isspace(static_cast<unsigned char>(c)) || c == '#' || c == ';') quote = 1;
    }
    if (text.empty()) text = "?";
    if (!quote) return text;
    return "'" + text + "'";
}

 static int
is_backbone_atom(const char *atom_name)
{
    static const char *backbone[] = {"CA", "N", "O", "C", "OXT",
                                     "P", "OP1", "OP2", "O5'", "C5'", "C4'",
                                     "O4'", "C3'", "O3'", "C2'", "C1'"};
    const std::string atom = upper(trim(atom_name ? atom_name : ""));

    for (size_t i = 0; i < sizeof(backbone) / sizeof(backbone[0]); ++i) {
        if (atom == backbone[i]) return 1;
    }
    return 0;
}

static int
atom_class_from_config(const radius_config *config,
                       const char *residue_name,
                       const char *atom_name)
{
    if (config == nullptr) return -1;

    const std::string atom = upper(trim(atom_name ? atom_name : ""));
    const std::string residue = upper(residue_name ? residue_name : "");
    auto atom_it =
        config->atom_type.find(residue + ":" + atom);

    if (atom_it == config->atom_type.end()) {
        const char *canonical = fastsasa_canonical_residue(residue.c_str());
        if (canonical != nullptr) atom_it = config->atom_type.find(std::string(canonical) + ":" + atom);
    }
    if (atom_it == config->atom_type.end()) atom_it = config->atom_type.find("ANY:" + atom);
    if (atom_it == config->atom_type.end()) return -1;

    const auto class_it = config->type_class.find(atom_it->second);
    if (class_it == config->type_class.end()) return -1;
    if (class_it->second == "POLAR") return 1;
    if (class_it->second == "APOLAR") return 0;
    return -1;
}

static int
atom_class_fallback(const char *element)
{
    const std::string key = upper(trim(element ? element : ""));

    if (key == "C") return 0;
    if (!key.empty() && key != "H" && key != "D") return 1;
    return -1;
}

/* Per-residue maximum-ASA reference values for relative SASA (--format rsa),
 * one entry per standard amino acid: {total, main-chain, side-chain, polar,
 * apolar} (the rsa_area field order). These are FreeSASA's own ProtOr reference values
 * (classifier_protor.c, Copyright (c) 2016 Simon Mitternacht, MIT license;
 * see licenses/FreeSASA-radii-table-MIT.txt), reproduced here unchanged so
 * relative SASA is comparable across tools -- not an independent
 * computation. */
static rsa_area
default_protor_reference(const char *residue_name)
{
    static const residue_reference references[] = {
        {"ALA", {108.76, 43.96, 64.80, 37.75, 71.01, 0.0}},
        {"ARG", {238.17, 42.00, 196.17, 165.00, 73.17, 0.0}},
        {"ASN", {145.01, 41.53, 103.48, 103.46, 41.55, 0.0}},
        {"ASP", {142.76, 42.29, 100.47, 100.27, 42.49, 0.0}},
        {"CYS", {132.20, 42.55, 89.66, 92.74, 39.47, 0.0}},
        {"GLN", {178.83, 42.00, 136.83, 131.85, 46.98, 0.0}},
        {"GLU", {174.18, 42.00, 132.18, 122.48, 51.70, 0.0}},
        {"GLY", {81.09, 81.09, 0.00, 44.65, 36.44, 0.0}},
        {"HIS", {182.97, 39.09, 143.87, 85.94, 97.03, 0.0}},
        {"ILE", {175.73, 41.49, 134.23, 36.85, 138.87, 0.0}},
        {"LEU", {179.56, 39.78, 139.78, 37.16, 142.39, 0.0}},
        {"LYS", {204.98, 42.00, 162.98, 93.88, 111.10, 0.0}},
        {"MET", {193.10, 42.00, 151.10, 75.48, 117.62, 0.0}},
        {"PHE", {199.88, 38.43, 161.45, 34.94, 164.94, 0.0}},
        {"PRO", {137.21, 27.51, 109.70, 16.09, 121.12, 0.0}},
        {"SER", {118.34, 43.41, 74.93, 71.38, 46.96, 0.0}},
        {"THR", {140.60, 41.96, 98.64, 66.15, 74.45, 0.0}},
        {"TRP", {249.19, 42.59, 206.60, 61.64, 187.55, 0.0}},
        {"TYR", {214.19, 38.43, 175.76, 81.12, 133.07, 0.0}},
        {"VAL", {151.97, 41.50, 110.46, 36.87, 115.09, 0.0}},
    };
    std::string residue = upper(residue_name ? residue_name : "");
    const char *canonical = fastsasa_canonical_residue(residue.c_str());

    if (canonical != nullptr) residue = canonical;
    for (size_t i = 0; i < sizeof(references) / sizeof(references[0]); ++i) {
        if (residue == references[i].name) return references[i].area;
    }
    return zero_rsa_area();
}

static std::vector<rsa_area>
compute_residue_areas(const fastsasa_owned_topology &topology,
                      const cli_state &state,
                      const std::vector<double> &atom_sasa)
{
    std::vector<rsa_area> areas(static_cast<size_t>(topology.n_residues), zero_rsa_area());
    const radius_config *config = state.config_loaded ? &state.config : nullptr;

    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const int residue = topology.residue_ids[atom];
        int atom_class;

        if (residue < 0 || residue >= topology.n_residues) continue;
        rsa_area &area = areas[static_cast<size_t>(residue)];
        const double sasa = atom_sasa[static_cast<size_t>(atom)];

        area.total += sasa;
        if (is_backbone_atom(topology.atom_names[atom])) area.main_chain += sasa;
        else area.side_chain += sasa;

        atom_class = atom_class_from_config(config, topology.residue_names[atom], topology.atom_names[atom]);
        if (atom_class < 0 && config == nullptr) atom_class = atom_class_fallback(topology.elements[atom]);
        if (atom_class == 1) area.polar += sasa;
        else if (atom_class == 0) area.apolar += sasa;
        else area.unknown += sasa;
    }

    return areas;
}

static rsa_area
sum_rsa_areas(const std::vector<rsa_area> &areas)
{
    rsa_area sum = zero_rsa_area();

    for (size_t i = 0; i < areas.size(); ++i) {
        sum.total += areas[i].total;
        sum.main_chain += areas[i].main_chain;
        sum.side_chain += areas[i].side_chain;
        sum.polar += areas[i].polar;
        sum.apolar += areas[i].apolar;
        sum.unknown += areas[i].unknown;
    }
    return sum;
}

static std::vector<rsa_area>
compute_selection_areas(const fastsasa_owned_topology &topology,
                        const cli_state &state,
                        const std::vector<double> &atom_sasa,
                        const std::vector<unsigned int> &selection_masks,
                        size_t n_selections)
{
    std::vector<rsa_area> areas(n_selections, zero_rsa_area());
    const radius_config *config = state.config_loaded ? &state.config : nullptr;

    if (selection_masks.empty()) return areas;
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const double sasa = atom_sasa[static_cast<size_t>(atom)];
        int atom_class = atom_class_from_config(config, topology.residue_names[atom], topology.atom_names[atom]);

        if (atom_class < 0 && config == nullptr) atom_class = atom_class_fallback(topology.elements[atom]);
        for (size_t selection = 0; selection < n_selections; ++selection) {
            if ((selection_masks[static_cast<size_t>(atom)] & (1u << selection)) == 0u) continue;
            areas[selection].total += sasa;
            if (atom_class == 1) areas[selection].polar += sasa;
            else if (atom_class == 0) areas[selection].apolar += sasa;
            else areas[selection].unknown += sasa;
        }
    }
    return areas;
}

static void
add_rsa_area(rsa_area *sum,
             const rsa_area &area)
{
    sum->total += area.total;
    sum->main_chain += area.main_chain;
    sum->side_chain += area.side_chain;
    sum->polar += area.polar;
    sum->apolar += area.apolar;
    sum->unknown += area.unknown;
}

static void
write_rsa_abs_rel(FILE *out,
                  double absolute,
                  double reference)
{
    std::fprintf(out, "%7.2f", absolute);
    if (reference > 0.0) std::fprintf(out, "%6.1f", 100.0 * absolute / reference);
    else std::fprintf(out, "   N/A");
}

static void
write_rsa_output(FILE *out,
                 const char *source,
                 const fastsasa_owned_topology &topology,
                 const cli_state &state,
                 const std::vector<rsa_area> &residue_areas)
{
    std::map<std::string, rsa_area> chain_areas;
    std::vector<std::string> chain_order;
    rsa_area total = zero_rsa_area();

    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom >= topology.n_atoms) continue;
        const std::string chain = safe_text(topology.chain_ids[atom]);
        if (chain_areas.find(chain) == chain_areas.end()) {
            chain_areas[chain] = zero_rsa_area();
            chain_order.push_back(chain);
        }
        add_rsa_area(&chain_areas[chain], residue_areas[static_cast<size_t>(residue)]);
        add_rsa_area(&total, residue_areas[static_cast<size_t>(residue)]);
    }

    std::fprintf(out, "REM  FastSASA %s\n", FASTSASA_VERSION);
    std::fprintf(out, "REM  Absolute and relative SASAs for %s\n", source);
    if (!state.reference_available) {
        std::fprintf(out, "REM  No reference values available for custom radii\n");
    } else {
        std::fprintf(out, "REM  Atomic radii and reference values for relative SASA: ProtOr\n");
    }
    std::fprintf(out, "REM  Chains:");
    for (size_t i = 0; i < chain_order.size(); ++i) std::fprintf(out, " %s", chain_order[i].c_str());
    std::fprintf(out, "\nREM  Algorithm: %s\n", output_algorithm_name(state));
    std::fprintf(out, "REM  Probe-radius: %.2f\n", state.probe_radius);
    std::fprintf(out, "REM  %s: %d\n",
                 state.lee_richards ? "Slices" : "Test-points",
                 state.n_points);
    std::fprintf(out, "REM RES _ NUM      All-atoms   Total-Side   Main-Chain    Non-polar    All polar\n");
    std::fprintf(out, "REM                ABS   REL    ABS   REL    ABS   REL    ABS   REL    ABS   REL\n");

    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom >= topology.n_atoms) continue;
        const rsa_area &area = residue_areas[static_cast<size_t>(residue)];
        const rsa_area reference = state.reference_available
                                       ? default_protor_reference(topology.residue_names[atom])
                                       : zero_rsa_area();

        std::fprintf(out, "RES %-3s %3s%-4s ",
                     safe_text(topology.residue_names[atom]),
                     safe_text(topology.chain_ids[atom]),
                     safe_text(topology.residue_number_strings[atom]));
        write_rsa_abs_rel(out, area.total, reference.total);
        write_rsa_abs_rel(out, area.side_chain, reference.side_chain);
        write_rsa_abs_rel(out, area.main_chain, reference.main_chain);
        write_rsa_abs_rel(out, area.apolar, reference.apolar);
        write_rsa_abs_rel(out, area.polar, reference.polar);
        std::fprintf(out, "\n");
    }

    std::fprintf(out, "END  Absolute sums over single chains surface\n");
    for (size_t i = 0; i < chain_order.size(); ++i) {
        const rsa_area &area = chain_areas[chain_order[i]];

        std::fprintf(out, "CHAIN%3zu %3s %10.1f   %10.1f   %10.1f   %10.1f   %10.1f\n",
                     i + 1u,
                     chain_order[i].c_str(),
                     area.total,
                     area.side_chain,
                     area.main_chain,
                     area.apolar,
                     area.polar);
    }
    std::fprintf(out, "END  Absolute sums over all chains\n");
    std::fprintf(out, "TOTAL        %10.1f   %10.1f   %10.1f   %10.1f   %10.1f\n",
                 total.total,
                 total.side_chain,
                 total.main_chain,
                 total.apolar,
                 total.polar);
}

static void
write_pdb_output(FILE *out,
                 const fastsasa_owned_topology &topology,
                 const std::vector<double> &atom_sasa)
{
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const char *chain = safe_text(topology.chain_ids[atom]);
        const int residue_number = topology.residue_numbers ? topology.residue_numbers[atom] : atom + 1;

        std::fprintf(out,
                     "ATOM  %5d %-4.4s %-3.3s %1.1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2.2s\n",
                     atom + 1,
                     safe_text(topology.atom_names[atom]),
                     safe_text(topology.residue_names[atom]),
                     chain,
                     residue_number,
                     topology.x[atom],
                     topology.y[atom],
                     topology.z[atom],
                     topology.radii[atom],
                     pdb_b_factor(atom_sasa[atom]),
                     safe_text(topology.elements[atom]));
    }
    std::fprintf(out, "END\n");
}

static int
write_pdb_compatible_output(FILE *out,
                            const char *source,
                            const fastsasa_owned_topology &topology,
                            const cli_state &state,
                            const std::vector<double> &atom_sasa)
{
    std::ifstream input(source);
    std::vector<std::string> lines;
    std::unordered_set<std::string> seen_altlocs;
    int atom = 0;
    std::string last_residue_name;
    std::string last_chain;
    std::string last_residue_number;
    int last_serial = 0;
    int saw_model = 0;
    int in_first_model = 1;

    if (!input) return 0;

    std::fprintf(out, "REMARK 999 This PDB file was generated by %s.\n",
                 generator_name(state));
    std::fprintf(out, "REMARK 999 In the ATOM records temperature factors have been\n"
                      "REMARK 999 replaced by the SASA of the atom, and the occupancy\n"
                      "REMARK 999 by the radius used in the calculation.\n");
    std::fprintf(out, "MODEL        1\n");

    for (std::string line; std::getline(input, line);) {
        if (line.compare(0, 5, "MODEL") == 0) {
            if (saw_model && (state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) break;
            saw_model = 1;
            in_first_model = 1;
            continue;
        }
        if (line.compare(0, 6, "ENDMDL") == 0) {
            if ((state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) break;
            in_first_model = 0;
            continue;
        }
        if (!in_first_model && (state.topology_options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0) continue;
        if (line.compare(0, 4, "ATOM") != 0 && line.compare(0, 6, "HETATM") != 0) continue;
        if (line.compare(0, 6, "HETATM") == 0 && (state.topology_options & FASTSASA_TOPOLOGY_INCLUDE_HETATM) == 0) continue;

        const std::string atom_name = trim(line.size() >= 16 ? line.substr(12, 4) : "");
        const std::string altloc = line.size() >= 17 ? line.substr(16, 1) : " ";
        const std::string residue_name = trim(line.size() >= 20 ? line.substr(17, 3) : "");
        const std::string chain = trim(line.size() >= 22 ? line.substr(21, 1) : "");
        const std::string raw_residue_number = line.size() >= 27 ? line.substr(22, 5) : "";
        const std::string residue_number = trim(raw_residue_number);
        const std::string element = infer_element(atom_name, line.size() >= 78 ? line.substr(76, 2) : "");
        const std::string atom_key = chain + ":" + residue_number + ":" + residue_name + ":" + atom_name;

        if ((element == "H" || element == "D") && (state.topology_options & FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN) == 0) continue;
        if (altloc != " " && altloc != "" && !seen_altlocs.insert(atom_key).second) continue;
        if (atom >= topology.n_atoms) return 0;

        if (line.size() < 66) line.resize(66, ' ');
        {
            char values[32];
            std::snprintf(values, sizeof(values), "%6.2f%6.2f",
                          topology.radii[atom],
                          pdb_b_factor(atom_sasa[atom]));
            line.replace(54, 12, values, 12);
            line.resize(66);
        }
        std::fprintf(out, "%s\n", line.c_str());
        last_residue_name = safe_text(topology.residue_names[atom]);
        last_chain = safe_text(topology.chain_ids[atom]);
        last_residue_number = raw_residue_number;
        last_serial = atom + 1;
        ++atom;
    }

    if (atom != topology.n_atoms || last_chain.empty()) return 0;
    std::fprintf(out, "TER   %5d     %4s %c%5s\nENDMDL\n",
                 last_serial + 1,
                 last_residue_name.c_str(),
                 last_chain[0],
                 last_residue_number.c_str());
    return 1;
}

 static void
write_fastsasa_cif_output(FILE *out,
                        const char *source,
                        const fastsasa_owned_topology &topology,
                        const cli_state &state,
                        double total_sasa,
                        const std::vector<double> &atom_sasa,
                        const std::vector<double> &residue_sasa)
{
    std::ifstream input(source);
    std::string line;
    rsa_area structure_area = zero_rsa_area();

    if (state.report_classes) {
        structure_area = sum_rsa_areas(compute_residue_areas(topology, state, atom_sasa));
    }

    if (input) {
        while (std::getline(input, line)) std::fprintf(out, "%s\n", line.c_str());
        std::fprintf(out, "\n");
    } else {
        std::fprintf(out, "data_fastsasa\n\n");
    }

    std::fprintf(out, "#\n");
    std::fprintf(out, "_FastSASA_results.algorithm '%s'\n", fastsasa_algorithm_name(state));
    std::fprintf(out, "_FastSASA_results.probe_radius %.12g\n", state.probe_radius);
    std::fprintf(out, "_FastSASA_results.%s %d\n",
                 state.lee_richards ? "n_slices" : "n_points",
                 state.n_points);
    std::fprintf(out, "_FastSASA_results.total_sasa %.12f\n", total_sasa);
    if (state.report_classes) {
        std::fprintf(out, "_FastSASA_results.polar_sasa %.12f\n", structure_area.polar);
        std::fprintf(out, "_FastSASA_results.apolar_sasa %.12f\n", structure_area.apolar);
        std::fprintf(out, "_FastSASA_results.unknown_sasa %.12f\n", structure_area.unknown);
    }
    std::fprintf(out, "#\nloop_\n");
    std::fprintf(out, "_FastSASA_atom_sasa.id\n");
    std::fprintf(out, "_FastSASA_atom_sasa.atom_name\n");
    std::fprintf(out, "_FastSASA_atom_sasa.residue_name\n");
    std::fprintf(out, "_FastSASA_atom_sasa.chain_id\n");
    std::fprintf(out, "_FastSASA_atom_sasa.residue_number\n");
    std::fprintf(out, "_FastSASA_atom_sasa.sasa\n");
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        std::fprintf(out, "%d %s %s %s %s %.12f\n",
                     atom + 1,
                     cif_token(topology.atom_names[atom]).c_str(),
                     cif_token(topology.residue_names[atom]).c_str(),
                     cif_token(topology.chain_ids[atom]).c_str(),
                     cif_token(topology.residue_number_strings[atom]).c_str(),
                     atom_sasa[atom]);
    }
    std::fprintf(out, "#\nloop_\n");
    std::fprintf(out, "_FastSASA_residue_sasa.id\n");
    std::fprintf(out, "_FastSASA_residue_sasa.chain_id\n");
    std::fprintf(out, "_FastSASA_residue_sasa.residue_number\n");
    std::fprintf(out, "_FastSASA_residue_sasa.residue_name\n");
    std::fprintf(out, "_FastSASA_residue_sasa.sasa\n");
    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom < topology.n_atoms) {
            std::fprintf(out, "%d %s %s %s %.12f\n",
                         residue + 1,
                         cif_token(topology.chain_ids[atom]).c_str(),
                         cif_token(topology.residue_number_strings[atom]).c_str(),
                         cif_token(topology.residue_names[atom]).c_str(),
                         residue_sasa[residue]);
        }
    }
    std::fprintf(out, "#\n");
}

static void
write_cif_output(FILE *out,
                 const char *source,
                 const fastsasa_owned_topology &topology,
                 const cli_state &state,
                 double total_sasa,
                 const std::vector<double> &atom_sasa,
                 const std::vector<double> &residue_sasa)
{
    write_fastsasa_cif_output(out, source, topology, state, total_sasa, atom_sasa, residue_sasa);
}

static void
write_json_output(FILE *out,
                  const char *source,
                  const fastsasa_owned_topology &topology,
                  const cli_state &state,
                  double total_sasa,
                  const std::vector<double> &atom_sasa,
                  const std::vector<double> &residue_sasa,
                  const std::vector<double> &selection_sasa,
                  const std::vector<std::string> &selection_names,
                  const std::vector<unsigned int> &selection_masks)
{
    std::vector<rsa_area> residue_areas;
    std::vector<rsa_area> selection_areas;
    rsa_area structure_area = zero_rsa_area();

    if (state.report_classes) {
        residue_areas = compute_residue_areas(topology, state, atom_sasa);
        selection_areas = compute_selection_areas(topology, state, atom_sasa, selection_masks, selection_names.size());
        structure_area = sum_rsa_areas(residue_areas);
    }
    std::fprintf(out, "{\n");
    std::fprintf(out, "  \"source\": \"%s\",\n", json_escape(source).c_str());
    std::fprintf(out, "  \"parameters\": {\"algorithm\": \"%s\", \"probe_radius\": %.12g, \"%s\": %d},\n",
                 fastsasa_algorithm_name(state),
                 state.probe_radius,
                 state.lee_richards ? "n_slices" : "n_points",
                 state.n_points);
    std::fprintf(out, "  \"total_sasa\": %.12f,\n", total_sasa);
    if (state.report_classes) {
        std::fprintf(out,
                     "  \"classes\": {\"polar_sasa\": %.12f, \"apolar_sasa\": %.12f, \"unknown_sasa\": %.12f},\n",
                     structure_area.polar,
                     structure_area.apolar,
                     structure_area.unknown);
    }
    std::fprintf(out, "  \"atoms\": [\n");
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        std::fprintf(out,
                     "    {\"id\": %d, \"atom_name\": \"%s\", \"residue_name\": \"%s\", \"chain\": \"%s\", \"residue_number\": \"%s\", \"sasa\": %.12f}%s\n",
                     atom + 1,
                     json_escape(topology.atom_names[atom]).c_str(),
                     json_escape(topology.residue_names[atom]).c_str(),
                     json_escape(topology.chain_ids[atom]).c_str(),
                     json_escape(topology.residue_number_strings[atom]).c_str(),
                     atom_sasa[atom],
                     atom + 1 == topology.n_atoms ? "" : ",");
    }
    std::fprintf(out, "  ],\n  \"residues\": [\n");
    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom < topology.n_atoms) {
            if (state.report_classes) {
                const rsa_area &area = residue_areas[static_cast<size_t>(residue)];

                std::fprintf(out,
                             "    {\"id\": %d, \"chain\": \"%s\", \"residue_number\": \"%s\", \"residue_name\": \"%s\", \"sasa\": %.12f, \"polar_sasa\": %.12f, \"apolar_sasa\": %.12f, \"unknown_sasa\": %.12f}%s\n",
                             residue + 1,
                             json_escape(topology.chain_ids[atom]).c_str(),
                             json_escape(topology.residue_number_strings[atom]).c_str(),
                             json_escape(topology.residue_names[atom]).c_str(),
                             residue_sasa[residue],
                             area.polar,
                             area.apolar,
                             area.unknown,
                             residue + 1 == topology.n_residues ? "" : ",");
            } else {
                std::fprintf(out,
                             "    {\"id\": %d, \"chain\": \"%s\", \"residue_number\": \"%s\", \"residue_name\": \"%s\", \"sasa\": %.12f}%s\n",
                             residue + 1,
                             json_escape(topology.chain_ids[atom]).c_str(),
                             json_escape(topology.residue_number_strings[atom]).c_str(),
                             json_escape(topology.residue_names[atom]).c_str(),
                             residue_sasa[residue],
                             residue + 1 == topology.n_residues ? "" : ",");
            }
        }
    }
    std::fprintf(out, "  ],\n  \"selections\": [\n");
    for (size_t i = 0; i < selection_names.size(); ++i) {
        if (state.report_classes) {
            const rsa_area &area = selection_areas[i];

            std::fprintf(out,
                         "    {\"name\": \"%s\", \"sasa\": %.12f, \"polar_sasa\": %.12f, \"apolar_sasa\": %.12f, \"unknown_sasa\": %.12f}%s\n",
                         json_escape(selection_names[i].c_str()).c_str(),
                         selection_sasa[i],
                         area.polar,
                         area.apolar,
                         area.unknown,
                         i + 1 == selection_names.size() ? "" : ",");
        } else {
            std::fprintf(out,
                         "    {\"name\": \"%s\", \"sasa\": %.12f}%s\n",
                         json_escape(selection_names[i].c_str()).c_str(),
                         selection_sasa[i],
                         i + 1 == selection_names.size() ? "" : ",");
        }
    }
    std::fprintf(out, "  ]\n}\n");
}

static void
write_xml_output(FILE *out,
                 const char *source,
                 const fastsasa_owned_topology &topology,
                 const cli_state &state,
                 double total_sasa,
                 const std::vector<double> &atom_sasa,
                 const std::vector<double> &residue_sasa,
                 const std::vector<double> &selection_sasa,
                 const std::vector<std::string> &selection_names,
                 const std::vector<unsigned int> &selection_masks)
{
    std::vector<rsa_area> residue_areas;
    std::vector<rsa_area> selection_areas;
    rsa_area structure_area = zero_rsa_area();

    if (state.report_classes) {
        residue_areas = compute_residue_areas(topology, state, atom_sasa);
        selection_areas = compute_selection_areas(topology, state, atom_sasa, selection_masks, selection_names.size());
        structure_area = sum_rsa_areas(residue_areas);
    }
    std::fprintf(out, "<FastSASA source=\"%s\">\n", xml_escape(source).c_str());
    std::fprintf(out, "  <parameters algorithm=\"%s\" probe_radius=\"%.12g\" %s=\"%d\" />\n",
                 fastsasa_algorithm_name(state),
                 state.probe_radius,
                 state.lee_richards ? "n_slices" : "n_points",
                 state.n_points);
    std::fprintf(out, "  <total_sasa>%.12f</total_sasa>\n", total_sasa);
    if (state.report_classes) {
        std::fprintf(out,
                     "  <classes polar_sasa=\"%.12f\" apolar_sasa=\"%.12f\" unknown_sasa=\"%.12f\" />\n",
                     structure_area.polar,
                     structure_area.apolar,
                     structure_area.unknown);
    }
    std::fprintf(out, "  <atoms>\n");
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        std::fprintf(out,
                     "    <atom id=\"%d\" atom_name=\"%s\" residue_name=\"%s\" chain=\"%s\" residue_number=\"%s\" sasa=\"%.12f\" />\n",
                     atom + 1,
                     xml_escape(topology.atom_names[atom]).c_str(),
                     xml_escape(topology.residue_names[atom]).c_str(),
                     xml_escape(topology.chain_ids[atom]).c_str(),
                     xml_escape(topology.residue_number_strings[atom]).c_str(),
                     atom_sasa[atom]);
    }
    std::fprintf(out, "  </atoms>\n  <residues>\n");
    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom < topology.n_atoms) {
            if (state.report_classes) {
                const rsa_area &area = residue_areas[static_cast<size_t>(residue)];

                std::fprintf(out,
                             "    <residue id=\"%d\" chain=\"%s\" residue_number=\"%s\" residue_name=\"%s\" sasa=\"%.12f\" polar_sasa=\"%.12f\" apolar_sasa=\"%.12f\" unknown_sasa=\"%.12f\" />\n",
                             residue + 1,
                             xml_escape(topology.chain_ids[atom]).c_str(),
                             xml_escape(topology.residue_number_strings[atom]).c_str(),
                             xml_escape(topology.residue_names[atom]).c_str(),
                             residue_sasa[residue],
                             area.polar,
                             area.apolar,
                             area.unknown);
            } else {
                std::fprintf(out,
                             "    <residue id=\"%d\" chain=\"%s\" residue_number=\"%s\" residue_name=\"%s\" sasa=\"%.12f\" />\n",
                             residue + 1,
                             xml_escape(topology.chain_ids[atom]).c_str(),
                             xml_escape(topology.residue_number_strings[atom]).c_str(),
                             xml_escape(topology.residue_names[atom]).c_str(),
                             residue_sasa[residue]);
            }
        }
    }
    std::fprintf(out, "  </residues>\n  <selections>\n");
    for (size_t i = 0; i < selection_names.size(); ++i) {
        if (state.report_classes) {
            const rsa_area &area = selection_areas[i];

            std::fprintf(out,
                         "    <selection name=\"%s\" sasa=\"%.12f\" polar_sasa=\"%.12f\" apolar_sasa=\"%.12f\" unknown_sasa=\"%.12f\" />\n",
                         xml_escape(selection_names[i].c_str()).c_str(),
                         selection_sasa[i],
                         area.polar,
                         area.apolar,
                         area.unknown);
        } else {
            std::fprintf(out,
                         "    <selection name=\"%s\" sasa=\"%.12f\" />\n",
                         xml_escape(selection_names[i].c_str()).c_str(),
                         selection_sasa[i]);
        }
    }
    std::fprintf(out, "  </selections>\n</FastSASA>\n");
}

/*
 * Writes the solvent-accessible Shrake-Rupley surface points as
 * "x y z atom_index" lines. The exposed-point test is identical to the SASA
 * calculation, so the emitted points sample exactly the reported surface.
 * Lee-Richards runs sample with the default 100 Shrake-Rupley points because
 * LR resolution counts slices, not sphere points.
 */
static int
write_surface_points(const char *path,
                     const fastsasa_owned_topology &topology,
                     const cli_state &state)
{
    const int n_points = state.lee_richards ? 100 : state.n_points;
    std::vector<double> expanded_radii(static_cast<size_t>(topology.n_atoms));
    std::vector<double> points;
    std::vector<unsigned char> exposed(
        static_cast<size_t>(topology.n_atoms) * static_cast<size_t>(n_points));
    FILE *out;
    int status;

    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        expanded_radii[static_cast<size_t>(atom)] =
            topology.radii[atom] + state.probe_radius;
    }
    make_test_points(n_points, &points);
    status = fastsasa_cpu_exposed_points(topology.n_atoms, n_points,
                                       topology.x, topology.y, topology.z,
                                       expanded_radii.data(), points.data(),
                                       state.n_threads, exposed.data());
    if (status != FASTSASA_SUCCESS) {
        std::fprintf(stderr, "surface-point export failed\n");
        return 0;
    }
    out = std::fopen(path, "w");
    if (out == nullptr) {
        std::fprintf(stderr, "failed to open surface-points file %s: %s\n",
                     path, std::strerror(errno));
        return 0;
    }
    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const double radius = expanded_radii[static_cast<size_t>(atom)];
        const unsigned char *row = exposed.data() +
            static_cast<size_t>(atom) * static_cast<size_t>(n_points);

        for (int point = 0; point < n_points; ++point) {
            if (!row[point]) continue;
            std::fprintf(out, "%.3f %.3f %.3f %d\n",
                         topology.x[atom] + radius * points[3 * point],
                         topology.y[atom] + radius * points[3 * point + 1],
                         topology.z[atom] + radius * points[3 * point + 2],
                         atom);
        }
    }
    std::fclose(out);
    return 1;
}

static void
write_output(FILE *out,
             const char *source,
             const fastsasa_owned_topology &topology,
             const cli_state &state,
             double total_sasa,
             const std::vector<double> &atom_sasa,
             const std::vector<double> &residue_sasa,
             const std::vector<double> &selection_sasa,
             const std::vector<std::string> &selection_names,
             const std::vector<unsigned int> &selection_masks)
{
    static const char *standard_residues[] = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    };
    const std::vector<rsa_area> residue_areas = compute_residue_areas(topology, state, atom_sasa);
    std::map<std::string, double> chain_sasa;
    std::map<std::string, double> residue_type_sasa;
    std::set<std::string> chains;
    rsa_area structure_area = zero_rsa_area();
    std::vector<rsa_area> selection_areas;

    for (int atom = 0; atom < topology.n_atoms; ++atom) {
        const std::string chain = safe_text(topology.chain_ids[atom]);
        chains.insert(chain);
        chain_sasa[chain] += atom_sasa[atom];
    }
    for (int residue = 0; residue < topology.n_residues; ++residue) {
        int atom = residue_first_atom(topology, residue);
        if (atom < topology.n_atoms) {
            residue_type_sasa[safe_text(topology.residue_names[atom])] += residue_sasa[residue];
        }
        structure_area.total += residue_areas[static_cast<size_t>(residue)].total;
        structure_area.main_chain += residue_areas[static_cast<size_t>(residue)].main_chain;
        structure_area.side_chain += residue_areas[static_cast<size_t>(residue)].side_chain;
        structure_area.polar += residue_areas[static_cast<size_t>(residue)].polar;
        structure_area.apolar += residue_areas[static_cast<size_t>(residue)].apolar;
        structure_area.unknown += residue_areas[static_cast<size_t>(residue)].unknown;
    }
    if (state.report_classes) {
        selection_areas = compute_selection_areas(topology, state, atom_sasa, selection_masks, selection_names.size());
    }

    if (state.format_log) {
        std::fprintf(out, "## %s ##\n\n", generator_name(state));
        std::fprintf(out, "PARAMETERS\nalgorithm    : %s\nprobe-radius : %.3f\n%s%s: %d\n\n",
                     output_algorithm_name(state),
                     state.probe_radius,
                     resolution_key(state),
                     state.lee_richards ? "       " : "   ",
                     state.n_points);
        std::fprintf(out, "INPUT\nsource  : %s\nchains  : ", source);
        for (std::set<std::string>::const_iterator it = chains.begin(); it != chains.end(); ++it) {
            std::fprintf(out, "%s", it->c_str());
        }
        std::fprintf(out, "\nmodel   : 1\natoms   : %d\n\n", topology.n_atoms);
        const double apolar_output = structure_area.unknown == 0.0
            ? (std::round(total_sasa * 100.0) - std::round(structure_area.polar * 100.0)) / 100.0
            : structure_area.apolar;
        std::fprintf(out,
                     "RESULTS (A^2)\nTotal     : %10.2f\nApolar    : %10.2f\nPolar     : %10.2f\n",
                     total_sasa,
                     apolar_output,
                     structure_area.polar);
        if (structure_area.unknown > 0.0) {
            std::fprintf(out, "Unknown   : %10.2f\n", structure_area.unknown);
        }
        for (std::map<std::string, double>::const_iterator it = chain_sasa.begin(); it != chain_sasa.end(); ++it) {
            std::fprintf(out, "CHAIN %3s : %10.2f\n", it->first.c_str(), it->second);
        }
        if (!selection_names.empty()) {
            std::fprintf(out, "\nSELECTIONS\n");
            for (size_t i = 0; i < selection_names.size(); ++i) {
                if (state.report_classes) {
                    const rsa_area &area = selection_areas[i];

                    std::fprintf(out,
                                 "%s : %10.2f  polar %10.2f  apolar %10.2f",
                                 selection_names[i].c_str(),
                                 selection_sasa[i],
                                 area.polar,
                                 area.apolar);
                    if (area.unknown > 0.0) std::fprintf(out, "  unknown %10.2f", area.unknown);
                    std::fprintf(out, "\n");
                } else {
                    std::fprintf(out, "%s : %10.2f\n", selection_names[i].c_str(), selection_sasa[i]);
                }
            }
        }
    }
    if (state.format_residue) {
        std::fprintf(out, "# Residue types in %s\n", source);
        for (size_t i = 0; i < sizeof(standard_residues) / sizeof(standard_residues[0]); ++i) {
            const std::map<std::string, double>::const_iterator found = residue_type_sasa.find(standard_residues[i]);
            const double value = found == residue_type_sasa.end() ? 0.0 : found->second;
            std::fprintf(out, "RES %-3s : %10.2f\n", standard_residues[i], value);
        }
        std::fprintf(out, "\n");
    }
    if (state.format_seq) {
        std::fprintf(out, "# Residues in %s\n", source);
        for (int residue = 0; residue < topology.n_residues; ++residue) {
            int atom = residue_first_atom(topology, residue);
            if (atom < topology.n_atoms) {
                const char *residue_number = safe_text(topology.residue_number_strings[atom]);
                int has_insertion_code = 0;

                for (const char *p = residue_number; *p; ++p) {
                    if (!std::isdigit(static_cast<unsigned char>(*p)) && *p != '-' && *p != '+') {
                        has_insertion_code = 1;
                        break;
                    }
                }
                if (has_insertion_code) {
                    std::fprintf(out, "SEQ %s %5s %-3s : %7.2f\n",
                                 safe_text(topology.chain_ids[atom]),
                                 residue_number,
                                 safe_text(topology.residue_names[atom]),
                                 residue_sasa[residue]);
                } else {
                    std::fprintf(out, "SEQ %s %4s  %-3s : %7.2f\n",
                                 safe_text(topology.chain_ids[atom]),
                                 residue_number,
                                 safe_text(topology.residue_names[atom]),
                                 residue_sasa[residue]);
                }
            }
        }
        std::fprintf(out, "\n");
    }
    if (state.format_rsa) {
        write_rsa_output(out, source, topology, state, residue_areas);
    }
    if (state.format_pdb) {
        if (has_suffix(source, ".pdb") || has_suffix(source, ".PDB")) {
            if (!write_pdb_compatible_output(out, source, topology, state, atom_sasa)) {
                write_pdb_output(out, topology, atom_sasa);
            }
        } else {
            write_pdb_output(out, topology, atom_sasa);
        }
    }
    if (state.format_cif) {
        write_cif_output(out, source, topology, state, total_sasa, atom_sasa, residue_sasa);
    }
    if (state.format_json) {
        write_json_output(out, source, topology, state, total_sasa, atom_sasa, residue_sasa,
                          selection_sasa, selection_names, selection_masks);
    }
    if (state.format_xml) {
        write_xml_output(out, source, topology, state, total_sasa, atom_sasa, residue_sasa,
                         selection_sasa, selection_names, selection_masks);
    }
}

static int
parse_args(int argc,
           char **argv,
           cli_state *state,
           const char **input_path)
{
    *input_path = nullptr;
    /* FASTSASA_BACKEND=cpu is an explicit CPU request, the same as --cpu, so
     * it must not be reported as a failed GPU attempt with a fallback. */
    if (const char *env_backend = std::getenv("FASTSASA_BACKEND")) {
        if (std::strcmp(env_backend, "cpu") == 0 || std::strcmp(env_backend, "CPU") == 0) {
            state->force_cpu = 1;
        }
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shrake-rupley") == 0 || std::strcmp(argv[i], "-S") == 0) {
            state->lee_richards = 0;
            continue;
        } else if (std::strcmp(argv[i], "--cif") == 0) {
            state->force_cif_input = 1;
        } else if (std::strcmp(argv[i], "--lee-richards") == 0 || std::strcmp(argv[i], "-L") == 0) {
            state->lee_richards = 1;
        } else if (std::strcmp(argv[i], "--probe-radius") == 0 || std::strcmp(argv[i], "-p") == 0) {
            if (++i >= argc || !parse_double_strict_cpp(argv[i], &state->probe_radius)) {
                std::fprintf(stderr, "invalid --probe-radius value\n");
                return 0;
            }
        } else if (std::strcmp(argv[i], "--resolution") == 0 || std::strcmp(argv[i], "-n") == 0) {
            if (++i >= argc || !parse_int_strict_cpp(argv[i], &state->n_points)) {
                std::fprintf(stderr, "invalid --resolution value\n");
                return 0;
            }
            state->resolution_set = 1;
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc || !parse_int_strict_cpp(argv[i], &state->n_threads)) {
                std::fprintf(stderr, "invalid --threads value\n");
                return 0;
            }
        } else if (std::strcmp(argv[i], "--precision") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --precision value\n");
                return 0;
            }
            if (std::strcmp(argv[i], "fp64") == 0) {
                state->precision = FASTSASA_PRECISION_FP64;
                state->precision_set = 1;
            } else if (std::strcmp(argv[i], "fp32") == 0) {
                state->precision = FASTSASA_PRECISION_FP32;
                state->precision_set = 1;
            } else {
                std::fprintf(stderr, "--precision must be fp64 or fp32\n");
                return 0;
            }
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            state->force_cpu = 1;
        } else if (std::strcmp(argv[i], "--surface-points") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --surface-points value\n");
                return 0;
            }
            state->surface_points_path = argv[i];
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --backend value\n");
                return 0;
            }
            if (std::strcmp(argv[i], "cpu") == 0) {
                state->force_cpu = 1;
            } else if (std::strcmp(argv[i], "auto") == 0 ||
                       std::strcmp(argv[i], "cuda") == 0 ||
                       std::strcmp(argv[i], "vulkan") == 0) {
                /* The library dispatches on FASTSASA_BACKEND; the flag is the
                 * user-facing spelling of the same request. */
                if (fastsasa_setenv("FASTSASA_BACKEND", argv[i]) != 0) {
                    std::fprintf(stderr, "failed to select --backend %s\n", argv[i]);
                    return 0;
                }
            } else {
                std::fprintf(stderr, "--backend must be auto, vulkan, cuda, or cpu\n");
                return 0;
            }
        } else if (std::strcmp(argv[i], "--no-cpu-fallback") == 0) {
            state->cpu_fallback = 0;
        } else if (std::strcmp(argv[i], "--config-file") == 0 || std::strcmp(argv[i], "-c") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --config-file value\n");
                return 0;
            }
            state->config_path = argv[i];
        } else if (std::strcmp(argv[i], "--hetatm") == 0 || std::strcmp(argv[i], "-H") == 0) {
            state->topology_options |= FASTSASA_TOPOLOGY_INCLUDE_HETATM;
        } else if (std::strcmp(argv[i], "--hydrogen") == 0 || std::strcmp(argv[i], "-Y") == 0) {
            state->topology_options |= FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN;
        } else if (std::strcmp(argv[i], "--join-models") == 0 || std::strcmp(argv[i], "-m") == 0) {
            state->topology_options |= FASTSASA_TOPOLOGY_JOIN_MODELS;
        } else if (std::strcmp(argv[i], "--unknown") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --unknown value\n");
                return 0;
            }
            const char *mode = argv[i];
            if (std::strcmp(mode, "skip") == 0) state->topology_options |= FASTSASA_TOPOLOGY_SKIP_UNKNOWN;
            else if (std::strcmp(mode, "halt") == 0) state->topology_options |= FASTSASA_TOPOLOGY_HALT_AT_UNKNOWN;
            else if (std::strcmp(mode, "guess") != 0) {
                std::fprintf(stderr, "invalid --unknown value: %s\n", mode);
                return 0;
            }
        } else if (std::strcmp(argv[i], "--select") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --select value\n");
                return 0;
            }
            if (state->selections.size() >= 31u) {
                std::fprintf(stderr, "FastSASA supports at most 31 selections per calculation\n");
                return 0;
            }
            state->selections.push_back(argv[i]);
        } else if (std::strcmp(argv[i], "--classes") == 0) {
            state->report_classes = 1;
        } else if (((std::strcmp(argv[i], "--format") == 0 || std::strcmp(argv[i], "-f") == 0) && i + 1 < argc) ||
                   std::strncmp(argv[i], "--format=", 9) == 0) {
            const char *format = std::strncmp(argv[i], "--format=", 9) == 0 ? argv[i] + 9 : argv[++i];
            if (!state->format_set) {
                state->format_log = 0;
                state->format_set = 1;
            }
            if (std::strcmp(format, "log") == 0) state->format_log = 1;
            else if (std::strcmp(format, "res") == 0) state->format_residue = 1;
            else if (std::strcmp(format, "seq") == 0) state->format_seq = 1;
            else if (std::strcmp(format, "rsa") == 0) state->format_rsa = 1;
            else if (std::strcmp(format, "pdb") == 0) state->format_pdb = 1;
            else if (std::strcmp(format, "cif") == 0) state->format_cif = 1;
            else if (std::strcmp(format, "json") == 0) state->format_json = 1;
            else if (std::strcmp(format, "xml") == 0) state->format_xml = 1;
            else {
                std::fprintf(stderr, "native FastSASA CLI currently supports --format=log|res|seq|rsa|pdb|cif|json|xml\n");
                return 0;
            }
        } else if (std::strcmp(argv[i], "--format") == 0 || std::strcmp(argv[i], "-f") == 0) {
            std::fprintf(stderr, "missing --format value\n");
            return 0;
        } else if (std::strcmp(argv[i], "--output") == 0 || std::strcmp(argv[i], "-o") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "missing --output value\n");
                return 0;
            }
            state->output_path = argv[i];
        } else if (argv[i][0] != '-') {
            if (*input_path != nullptr) {
                std::fprintf(stderr, "multiple structure inputs are not supported\n");
                return 0;
            }
            *input_path = argv[i];
        } else {
            return 0;
        }
    }
    if (state->lee_richards && !state->resolution_set) {
        state->n_points = 20;
    }
    if (*input_path == nullptr) return 0;
    if (state->probe_radius < 0.0) {
        std::fprintf(stderr, "--probe-radius must be non-negative\n");
        return 0;
    }
    if (state->n_points <= 0) {
        std::fprintf(stderr, "--resolution must be positive\n");
        return 0;
    }
    if (state->n_threads <= 0) {
        std::fprintf(stderr, "--threads must be positive\n");
        return 0;
    }
    return 1;
}

int
main(int argc,
     char **argv)
{
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        usage(argv[0], stdout);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::fprintf(stdout, "FastSASA %s\n", FASTSASA_VERSION);
        return 0;
    }

    const int trajectory_status = maybe_launch_trajectory_cli(argc, argv);

    if (trajectory_status >= 0) return trajectory_status;

    cli_state state;
    const char *input_path = nullptr;
    fastsasa_owned_topology topology;
    std::vector<double> atom_sasa;
    std::vector<double> residue_sasa;
    std::vector<double> selection_sasa;
    std::vector<std::string> selection_names;
    std::vector<unsigned int> selection_masks;
    double total_sasa = 0.0;
    FILE *out = stdout;
    int ok;

    if (!parse_args(argc, argv, &state, &input_path)) {
        usage(argv[0]);
        return 1;
    }
    if (state.config_path != nullptr) {
        if (!load_radius_config(state.config_path, &state.config)) {
            std::fprintf(stderr, "failed to read config file: %s\n", state.config_path);
            return 1;
        }
        note_config_loaded(&state, state.config_path);
    } else {
        load_default_radius_config(&state);
    }

    std::memset(&topology, 0, sizeof(topology));
    if (state.force_cif_input ||
        has_suffix(input_path, ".cif") || has_suffix(input_path, ".CIF") ||
        has_suffix(input_path, ".mmcif") || has_suffix(input_path, ".MMCIF")) {
        ok = fastsasa_topology_read_mmcif(input_path,
                                        state.topology_options,
                                        configured_radius,
                                        state.config_loaded ? &state.config : nullptr,
                                        &topology);
    } else {
        ok = read_pdb_topology(input_path, state, &topology);
    }
    if (!ok) {
        std::fprintf(stderr, "failed to read topology: %s\n", input_path);
        fastsasa_topology_free(&topology);
        return 1;
    }

    if (!run_sasa(topology, state, &atom_sasa, &total_sasa, &residue_sasa, &selection_sasa, &selection_names, &selection_masks)) {
        std::fprintf(stderr, "%s SASA calculation failed",
                     state.lee_richards ? "Lee-Richards" : "Shrake-Rupley");
        if (fastsasa_last_error()[0] != '\0') std::fprintf(stderr, ": %s", fastsasa_last_error());
        std::fprintf(stderr, "\n");
        fastsasa_topology_free(&topology);
        return 1;
    }

    if (state.output_path != nullptr) {
        out = std::fopen(state.output_path, "w");
        if (out == nullptr) {
            std::fprintf(stderr, "failed to open output file %s: %s\n", state.output_path, std::strerror(errno));
            fastsasa_topology_free(&topology);
            return 1;
        }
    }
    write_output(out, input_path, topology, state, total_sasa, atom_sasa, residue_sasa,
                 selection_sasa, selection_names, selection_masks);
    if (out != stdout) std::fclose(out);
    if (state.surface_points_path != nullptr &&
        !write_surface_points(state.surface_points_path, topology, state)) {
        fastsasa_topology_free(&topology);
        return 1;
    }
    fastsasa_topology_free(&topology);
    return 0;
}
