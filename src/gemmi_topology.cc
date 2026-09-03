#include "fastsasa_topology.h"
#include "fastsasa_radius.h"

#include <gemmi/cif.hpp>
#include <gemmi/mmcif.hpp>

#include <cstdlib>
#include <cstddef>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static char *
copy_string(const std::string &value)
{
    char *copy = static_cast<char *>(std::malloc(value.size() + 1u));

    if (copy == nullptr) return nullptr;
    std::memcpy(copy, value.c_str(), value.size() + 1u);
    return copy;
}

static char *
copy_cstring(const char *value)
{
    if (value == nullptr) value = "";
    return copy_string(std::string(value));
}

static std::string
upper_copy(const char *value)
{
    std::string out = value != nullptr ? value : "";

    for (char &c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

static double
parse_double(const std::string &value)
{
    return std::strtod(value.c_str(), nullptr);
}

static int
is_hydrogen_symbol(const std::string &element)
{
    std::string key = upper_copy(element.c_str());

    return key == "H" || key == "D";
}

static int
allocate_topology(fastsasa_owned_topology *topology,
                  int n_atoms)
{
    topology->x = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(n_atoms)));
    topology->y = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(n_atoms)));
    topology->z = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(n_atoms)));
    topology->radii = static_cast<double *>(std::malloc(sizeof(double) * static_cast<size_t>(n_atoms)));
    topology->residue_ids = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(n_atoms)));
    topology->residue_numbers = static_cast<int *>(std::malloc(sizeof(int) * static_cast<size_t>(n_atoms)));
    topology->residue_number_strings = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->atom_names = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->residue_names = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->chain_ids = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->segment_ids = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->elements = static_cast<char **>(std::calloc(static_cast<size_t>(n_atoms), sizeof(char *)));
    topology->atom_flags = static_cast<unsigned char *>(std::calloc(static_cast<size_t>(n_atoms), sizeof(unsigned char)));

    return topology->x != nullptr && topology->y != nullptr && topology->z != nullptr &&
           topology->radii != nullptr && topology->residue_ids != nullptr &&
           topology->residue_numbers != nullptr && topology->residue_number_strings != nullptr &&
           topology->atom_names != nullptr && topology->residue_names != nullptr &&
           topology->chain_ids != nullptr && topology->segment_ids != nullptr &&
           topology->elements != nullptr && topology->atom_flags != nullptr;
}

static int
keep_atom_site_row(const gemmi::cif::Table::Row &site,
                   const std::string &first_model,
                   int options)
{
    if (site[0] != "ATOM" && (options & FASTSASA_TOPOLOGY_INCLUDE_HETATM) == 0) return 0;
    if (is_hydrogen_symbol(site[7]) && (options & FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN) == 0) return 0;
    if ((options & FASTSASA_TOPOLOGY_JOIN_MODELS) == 0 && site[11] != first_model) return 0;
    return 1;
}

static int
read_mmcif_atom_site_table(const char *path,
                           int options,
                           fastsasa_radius_callback radius_callback,
                           void *userdata,
                           fastsasa_owned_topology *topology)
{
    static const std::vector<std::string> columns{
        "group_PDB",
        "auth_asym_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
        "auth_comp_id",
        "auth_atom_id",
        "label_alt_id",
        "type_symbol",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "pdbx_PDB_model_num"
    };
    gemmi::cif::Document doc;
    int n_atoms = 0;
    int atom_index = 0;
    int residue = 0;
    std::string first_model;
    std::unordered_set<std::string> seen_atom_sites;
    std::unordered_map<std::string, std::string> selected_altlocs;
    std::string last_residue_key;

    try {
        doc = gemmi::cif::read_file(path);
    } catch (...) {
        return 0;
    }

    for (gemmi::cif::Block &block : doc.blocks) {
        gemmi::cif::Table table = block.find("_atom_site.", columns);
        if (!table.ok()) continue;
        for (gemmi::cif::Table::Row site : table) {
            const std::string residue_site_key = std::string(site[11]) + ":" + std::string(site[1]) + ":" +
                                                 std::string(site[2]) + ":" + std::string(site[3]);
            const std::string altloc = site[6] == "." || site[6] == "?" ? "" : std::string(site[6]);

            if (first_model.empty()) first_model = site[11];
            if (!keep_atom_site_row(site, first_model, options)) continue;
            if (!altloc.empty()) {
                const std::unordered_map<std::string, std::string>::const_iterator selected = selected_altlocs.find(residue_site_key);

                if (selected == selected_altlocs.end()) selected_altlocs[residue_site_key] = altloc;
                else if (selected->second != altloc) continue;
            }
            const std::string atom_key = std::string(site[11]) + ":" + std::string(site[1]) + ":" +
                                         std::string(site[2]) + ":" + std::string(site[3]) + ":" +
                                         std::string(site[5]);
            if (!seen_atom_sites.insert(atom_key).second) continue;
            ++n_atoms;
        }
    }
    if (n_atoms <= 0) return 0;
    if (!allocate_topology(topology, n_atoms)) {
        fastsasa_topology_free(topology);
        return 0;
    }

    seen_atom_sites.clear();
    for (gemmi::cif::Block &block : doc.blocks) {
        gemmi::cif::Table table = block.find("_atom_site.", columns);
        if (!table.ok()) continue;
        for (gemmi::cif::Table::Row site : table) {
            const std::string residue_site_key = std::string(site[11]) + ":" + std::string(site[1]) + ":" +
                                                 std::string(site[2]) + ":" + std::string(site[3]);
            const std::string altloc = site[6] == "." || site[6] == "?" ? "" : std::string(site[6]);
            const std::string atom_key = std::string(site[11]) + ":" + std::string(site[1]) + ":" +
                                         std::string(site[2]) + ":" + std::string(site[3]) + ":" +
                                         std::string(site[5]);
            const std::string residue_number = site[3] == "?" || site[3] == "."
                                                   ? std::string(site[2])
                                                   : std::string(site[2]) + std::string(site[3]);
            const std::string residue_key = std::string(site[11]) + ":" + std::string(site[1]) + ":" +
                                            residue_number + ":" + std::string(site[4]);
            const char *element = site[7].c_str();
            double radius;

            if (!keep_atom_site_row(site, first_model, options)) continue;
            if (!altloc.empty() && selected_altlocs[residue_site_key] != altloc) continue;
            if (!seen_atom_sites.insert(atom_key).second) continue;
            topology->n_atoms = atom_index;
            if (residue_key != last_residue_key) {
                ++residue;
                last_residue_key = residue_key;
            }

            radius = radius_callback != nullptr
                         ? radius_callback(site[4].c_str(), site[5].c_str(), element, userdata)
                         : -1.0;
            if (radius < 0.0) radius = fastsasa_element_radius(element);
            if (radius < 0.0) {
                if ((options & FASTSASA_TOPOLOGY_HALT_AT_UNKNOWN) != 0) {
                    fastsasa_topology_free(topology);
                    return 0;
                }
                if ((options & FASTSASA_TOPOLOGY_SKIP_UNKNOWN) != 0) continue;
                radius = 0.0;
            }

            topology->x[atom_index] = parse_double(site[8]);
            topology->y[atom_index] = parse_double(site[9]);
            topology->z[atom_index] = parse_double(site[10]);
            topology->radii[atom_index] = radius;
            topology->residue_ids[atom_index] = residue - 1;
            topology->residue_numbers[atom_index] = std::atoi(site[2].c_str());
            topology->n_atoms = atom_index + 1;
            topology->residue_number_strings[atom_index] = copy_string(residue_number);
            topology->atom_names[atom_index] = copy_string(site[5]);
            topology->residue_names[atom_index] = copy_string(site[4]);
            topology->chain_ids[atom_index] = copy_string(site[1]);
            topology->segment_ids[atom_index] = copy_string("");
            topology->elements[atom_index] = copy_string(site[7]);
            topology->atom_flags[atom_index] =
                (site[0] != "ATOM" ? FASTSASA_ATOM_HETATM : 0) |
                (is_hydrogen_symbol(site[7]) ? FASTSASA_ATOM_HYDROGEN : 0);
            if (topology->residue_number_strings[atom_index] == nullptr ||
                topology->atom_names[atom_index] == nullptr ||
                topology->residue_names[atom_index] == nullptr ||
                topology->chain_ids[atom_index] == nullptr ||
                topology->segment_ids[atom_index] == nullptr ||
                topology->elements[atom_index] == nullptr) {
                fastsasa_topology_free(topology);
                return 0;
            }
            ++atom_index;
        }
    }

    topology->n_atoms = atom_index;
    topology->n_residues = residue;
    return atom_index > 0;
}

static int
append_atom(fastsasa_owned_topology *topology,
            int index,
            int residue,
            const gemmi::Chain &chain,
            const gemmi::Atom &atom,
            const gemmi::Residue &gemmi_residue,
            int options,
            fastsasa_radius_callback radius_callback,
            void *userdata)
{
    const char *element = atom.element.uname();
    double radius = radius_callback != nullptr
                        ? radius_callback(gemmi_residue.name.c_str(), atom.name.c_str(), element, userdata)
                        : -1.0;

    if (radius < 0.0) radius = fastsasa_element_radius(element);
    if (radius < 0.0) {
        if ((options & FASTSASA_TOPOLOGY_HALT_AT_UNKNOWN) != 0) return 0;
        if ((options & FASTSASA_TOPOLOGY_SKIP_UNKNOWN) != 0) return 2;
        radius = 0.0;
    }
    topology->x[index] = atom.pos.x;
    topology->y[index] = atom.pos.y;
    topology->z[index] = atom.pos.z;
    topology->radii[index] = radius;
    topology->residue_ids[index] = residue;
    topology->residue_numbers[index] = gemmi_residue.seqid.num.has_value()
                                           ? *gemmi_residue.seqid.num
                                           : 0;
    topology->residue_number_strings[index] = copy_string(gemmi_residue.seqid.str());
    topology->atom_names[index] = copy_string(atom.name);
    topology->residue_names[index] = copy_string(gemmi_residue.name);
    topology->chain_ids[index] = copy_string(chain.name);
    topology->segment_ids[index] = copy_string("");
    topology->elements[index] = copy_cstring(element);
    topology->atom_flags[index] =
        (gemmi_residue.het_flag == 'H' ? FASTSASA_ATOM_HETATM : 0) |
        (atom.is_hydrogen() ? FASTSASA_ATOM_HYDROGEN : 0);
    if (topology->residue_number_strings[index] == nullptr ||
        topology->atom_names[index] == nullptr ||
        topology->residue_names[index] == nullptr ||
        topology->chain_ids[index] == nullptr ||
        topology->segment_ids[index] == nullptr ||
        topology->elements[index] == nullptr) {
        return 0;
    }
    return 1;
}

void
fastsasa_topology_free(fastsasa_owned_topology *topology)
{
    if (topology == nullptr) return;
    std::free(topology->x);
    std::free(topology->y);
    std::free(topology->z);
    std::free(topology->radii);
    std::free(topology->residue_ids);
    std::free(topology->residue_numbers);
    if (topology->residue_number_strings != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->residue_number_strings[atom]);
    }
    if (topology->atom_names != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->atom_names[atom]);
    }
    if (topology->residue_names != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->residue_names[atom]);
    }
    if (topology->chain_ids != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->chain_ids[atom]);
    }
    if (topology->segment_ids != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->segment_ids[atom]);
    }
    if (topology->elements != nullptr) {
        for (int atom = 0; atom < topology->n_atoms; ++atom) std::free(topology->elements[atom]);
    }
    std::free(topology->residue_number_strings);
    std::free(topology->atom_names);
    std::free(topology->residue_names);
    std::free(topology->chain_ids);
    std::free(topology->segment_ids);
    std::free(topology->elements);
    std::free(topology->atom_flags);
    std::memset(topology, 0, sizeof(*topology));
}

size_t
fastsasa_sizeof_owned_topology(void)
{
    return sizeof(fastsasa_owned_topology);
}

size_t
fastsasa_offsetof_owned_topology_atom_flags(void)
{
    return offsetof(fastsasa_owned_topology, atom_flags);
}

int
fastsasa_topology_read_mmcif(const char *path,
                                 int options,
                                 fastsasa_radius_callback radius_callback,
                                 void *userdata,
                                 fastsasa_owned_topology *topology)
{
    if (path == nullptr || topology == nullptr) return 0;
    std::memset(topology, 0, sizeof(*topology));

    try {
        gemmi::Structure structure = gemmi::make_structure(gemmi::cif::read_file(path));
        size_t n_models = (options & FASTSASA_TOPOLOGY_JOIN_MODELS) != 0
                              ? structure.models.size()
                              : 1u;
        int n_atoms = 0;
        int residue = 0;

        if (structure.models.empty()) return 0;
        if (n_models > structure.models.size()) n_models = structure.models.size();

        for (size_t model_index = 0; model_index < n_models; ++model_index) {
            const gemmi::Model &model = structure.models[model_index];
            std::unordered_map<std::string, char> selected_altlocs;
            std::unordered_set<std::string> seen_atom_sites;

            for (const gemmi::Chain &chain : model.chains) {
                for (const gemmi::Residue &gemmi_residue : chain.residues) {
                    if (gemmi_residue.het_flag == 'H' && (options & FASTSASA_TOPOLOGY_INCLUDE_HETATM) == 0) {
                        continue;
                    }
                    for (const gemmi::Atom &atom : gemmi_residue.atoms) {
                        std::string atom_key;
                        const std::string residue_site_key = model.name + ":" + chain.name + ":" +
                                                             gemmi_residue.seqid.str();

                        if (atom.is_hydrogen() && (options & FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN) == 0) continue;
                        if (atom.altloc != '\0') {
                            const std::unordered_map<std::string, char>::const_iterator selected = selected_altlocs.find(residue_site_key);

                            if (selected == selected_altlocs.end()) selected_altlocs[residue_site_key] = atom.altloc;
                            else if (selected->second != atom.altloc) continue;
                        }
                        atom_key = model.name + ":" + chain.name + ":" + gemmi_residue.seqid.str() + ":" +
                                   gemmi_residue.name + ":" + atom.name;
                        if (!seen_atom_sites.insert(atom_key).second) continue;
                        ++n_atoms;
                    }
                }
            }
        }
        if (n_atoms <= 0) return 0;

        if (!allocate_topology(topology, n_atoms)) {
            fastsasa_topology_free(topology);
            return 0;
        }

        int atom_index = 0;
        for (size_t model_index = 0; model_index < n_models; ++model_index) {
            const gemmi::Model &model = structure.models[model_index];
            std::unordered_map<std::string, char> selected_altlocs;
            std::unordered_set<std::string> seen_atom_sites;

            for (const gemmi::Chain &chain : model.chains) {
                for (const gemmi::Residue &gemmi_residue : chain.residues) {
                    int residue_has_atoms = 0;

                    if (gemmi_residue.het_flag == 'H' && (options & FASTSASA_TOPOLOGY_INCLUDE_HETATM) == 0) {
                        continue;
                    }
                    for (const gemmi::Atom &atom : gemmi_residue.atoms) {
                        std::string atom_key;
                        const std::string residue_site_key = model.name + ":" + chain.name + ":" +
                                                             gemmi_residue.seqid.str();

                        if (atom.is_hydrogen() && (options & FASTSASA_TOPOLOGY_INCLUDE_HYDROGEN) == 0) continue;
                        if (atom.altloc != '\0') {
                            const std::unordered_map<std::string, char>::const_iterator selected = selected_altlocs.find(residue_site_key);

                            if (selected == selected_altlocs.end()) selected_altlocs[residue_site_key] = atom.altloc;
                            else if (selected->second != atom.altloc) continue;
                        }
                        atom_key = model.name + ":" + chain.name + ":" + gemmi_residue.seqid.str() + ":" +
                                   gemmi_residue.name + ":" + atom.name;
                        if (!seen_atom_sites.insert(atom_key).second) continue;
                        topology->n_atoms = atom_index + 1;
                        int append_status = append_atom(topology,
                                                        atom_index,
                                                        residue,
                                                        chain,
                                                        atom,
                                                        gemmi_residue,
                                                        options,
                                                        radius_callback,
                                                        userdata);
                        if (append_status == 0) {
                            fastsasa_topology_free(topology);
                            return 0;
                        }
                        if (append_status == 2) continue;
                        ++atom_index;
                        residue_has_atoms = 1;
                    }
                    if (residue_has_atoms) ++residue;
                }
            }
        }

        topology->n_atoms = atom_index;
        topology->n_residues = residue;
        return atom_index > 0;
    } catch (...) {
        fastsasa_topology_free(topology);
        std::memset(topology, 0, sizeof(*topology));
        try {
            return read_mmcif_atom_site_table(path, options, radius_callback, userdata, topology);
        } catch (...) {
            fastsasa_topology_free(topology);
            return 0;
        }
    }
}

static std::string
trim_copy(const std::string &value)
{
    size_t begin = 0;
    size_t end = value.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(begin, end - begin);
}

static std::string
selection_name_from_expression(const std::string &expression)
{
    std::string name;
    bool previous_separator = false;

    for (size_t i = 0; i < expression.size() && name.size() < FASTSASA_MAX_SELECTION_NAME; ++i) {
        unsigned char c = static_cast<unsigned char>(expression[i]);

        if (std::isalnum(c)) {
            name.push_back(static_cast<char>(c));
            previous_separator = false;
        } else if (!previous_separator && !name.empty()) {
            name.push_back('_');
            previous_separator = true;
        }
    }
    while (!name.empty() && name[name.size() - 1u] == '_') name.resize(name.size() - 1u);
    if (name.empty()) name = "selection";
    return name;
}

static int
equals_ci(const char *left,
          const std::string &right)
{
    return upper_copy(left) == upper_copy(right.c_str());
}

static int
parse_integer(const std::string &value,
              int *out)
{
    char *end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);

    if (end == value.c_str() || *end != '\0') return 0;
    *out = static_cast<int>(parsed);
    return 1;
}

enum selection_token_type {
    TOK_END,
    TOK_VALUE,
    TOK_AND,
    TOK_OR,
    TOK_NOT,
    TOK_NAME,
    TOK_SYMBOL,
    TOK_RESN,
    TOK_RESI,
    TOK_CHAIN,
    TOK_SEGID,
    TOK_PROTEIN,
    TOK_PLUS,
    TOK_MINUS,
    TOK_LPAREN,
    TOK_RPAREN
};

struct selection_token {
    selection_token_type type;
    std::string value;
};

static std::vector<selection_token>
tokenize_selection(const std::string &text)
{
    std::vector<selection_token> tokens;
    size_t i = 0;

    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (std::isspace(c) || text[i] == '=') {
            ++i;
            continue;
        }
        if (text[i] == '+') {
            tokens.push_back({TOK_PLUS, "+"});
            ++i;
            continue;
        }
        if (text[i] == '-') {
            tokens.push_back({TOK_MINUS, "-"});
            ++i;
            continue;
        }
        if (text[i] == '(') {
            tokens.push_back({TOK_LPAREN, "("});
            ++i;
            continue;
        }
        if (text[i] == ')') {
            tokens.push_back({TOK_RPAREN, ")"});
            ++i;
            continue;
        }
        if (text[i] == '\'' || text[i] == '"') {
            char quote = text[i++];
            size_t start = i;

            while (i < text.size() && text[i] != quote) ++i;
            tokens.push_back({TOK_VALUE, text.substr(start, i - start)});
            if (i < text.size()) ++i;
            continue;
        }

        std::string value;
        while (i < text.size()) {
            unsigned char value_char = static_cast<unsigned char>(text[i]);

            if (text[i] == '\\' && i + 1 < text.size()) {
                value.push_back(text[i + 1]);
                i += 2;
                continue;
            }
            if (std::isspace(value_char) || text[i] == '=' || text[i] == '+' ||
                text[i] == '-' || text[i] == '(' || text[i] == ')') {
                break;
            }
            value.push_back(text[i]);
            ++i;
        }

        std::string upper = upper_copy(value.c_str());
        if (upper == "AND" || upper == "&") tokens.push_back({TOK_AND, value});
        else if (upper == "OR" || upper == "|") tokens.push_back({TOK_OR, value});
        else if (upper == "NOT" || upper == "!") tokens.push_back({TOK_NOT, value});
        else if (upper == "NAME" || upper == "ATOM") tokens.push_back({TOK_NAME, value});
        else if (upper == "SYMBOL" || upper == "ELEMENT" || upper == "ELEM") tokens.push_back({TOK_SYMBOL, value});
        else if (upper == "RESN" || upper == "RESNAME" || upper == "RESIDUE") tokens.push_back({TOK_RESN, value});
        else if (upper == "RESI" || upper == "RESID" || upper == "RESNUM") tokens.push_back({TOK_RESI, value});
        else if (upper == "CHAIN") tokens.push_back({TOK_CHAIN, value});
        else if (upper == "SEGID" || upper == "SEGNAME" || upper == "SEGMENT") tokens.push_back({TOK_SEGID, value});
        else if (upper == "PROTEIN") tokens.push_back({TOK_PROTEIN, value});
        else tokens.push_back({TOK_VALUE, value});
    }
    tokens.push_back({TOK_END, ""});
    return tokens;
}

enum selection_node_type {
    NODE_SELECTOR,
    NODE_AND,
    NODE_OR,
    NODE_NOT
};

enum selector_type {
    SELECT_NAME,
    SELECT_SYMBOL,
    SELECT_RESN,
    SELECT_RESI,
    SELECT_CHAIN,
    SELECT_SEGID,
    SELECT_PROTEIN
};

struct selector_item {
    std::string left;
    std::string right;
    int is_range;
    int open_left;
    int open_right;
    int valid;
};

struct selection_node {
    selection_node_type type;
    selector_type selector;
    std::vector<selector_item> items;
    selection_node *left;
    selection_node *right;
};

static selection_node *
make_node(selection_node_type type)
{
    selection_node *node = new selection_node;

    node->type = type;
    node->selector = SELECT_NAME;
    node->left = nullptr;
    node->right = nullptr;
    return node;
}

static void
free_node(selection_node *node)
{
    if (node == nullptr) return;
    free_node(node->left);
    free_node(node->right);
    delete node;
}

class selection_parser {
public:
    explicit selection_parser(const std::string &text)
        : tokens(tokenize_selection(text)), pos(0), ok(true)
    {
    }

    selection_node *parse()
    {
        selection_node *node = parse_or();

        if (!ok || peek().type != TOK_END) {
            free_node(node);
            return nullptr;
        }
        return node;
    }

private:
    std::vector<selection_token> tokens;
    size_t pos;
    bool ok;

    const selection_token &peek() const { return tokens[pos]; }

    const selection_token &take()
    {
        return tokens[pos++];
    }

    selection_node *parse_or()
    {
        selection_node *left = parse_and();

        while (ok && peek().type == TOK_OR) {
            take();
            selection_node *right = parse_and();
            selection_node *parent = make_node(NODE_OR);
            parent->left = left;
            parent->right = right;
            left = parent;
        }
        return left;
    }

    selection_node *parse_and()
    {
        selection_node *left = parse_unary();

        while (ok && peek().type == TOK_AND) {
            take();
            selection_node *right = parse_unary();
            selection_node *parent = make_node(NODE_AND);
            parent->left = left;
            parent->right = right;
            left = parent;
        }
        return left;
    }

    selection_node *parse_unary()
    {
        if (peek().type == TOK_NOT) {
            take();
            selection_node *node = make_node(NODE_NOT);
            node->right = parse_unary();
            return node;
        }
        return parse_primary();
    }

    selection_node *parse_primary()
    {
        if (peek().type == TOK_LPAREN) {
            selection_node *node;

            take();
            node = parse_or();
            if (peek().type != TOK_RPAREN) {
                ok = false;
                free_node(node);
                return nullptr;
            }
            take();
            return node;
        }
        return parse_selector();
    }

    selection_node *parse_selector()
    {
        selector_type selector;

        switch (peek().type) {
        case TOK_NAME:
            selector = SELECT_NAME;
            break;
        case TOK_SYMBOL:
            selector = SELECT_SYMBOL;
            break;
        case TOK_RESN:
            selector = SELECT_RESN;
            break;
        case TOK_RESI:
            selector = SELECT_RESI;
            break;
        case TOK_CHAIN:
            selector = SELECT_CHAIN;
            break;
        case TOK_SEGID:
            selector = SELECT_SEGID;
            break;
        case TOK_PROTEIN: {
            take();
            selection_node *node = make_node(NODE_SELECTOR);
            node->selector = SELECT_PROTEIN;
            return node;
        }
        default:
            ok = false;
            return nullptr;
        }
        take();

        selection_node *node = make_node(NODE_SELECTOR);
        node->selector = selector;
        if (!parse_item_list(selector, node->items)) {
            free_node(node);
            ok = false;
            return nullptr;
        }
        return node;
    }

    bool parse_item_list(selector_type selector,
                         std::vector<selector_item> &items)
    {
        selector_item item;

        if (!parse_item(selector, item)) return false;
        items.push_back(item);
        while (peek().type == TOK_PLUS) {
            take();
            if (!parse_item(selector, item)) return false;
            items.push_back(item);
        }
        return true;
    }

    bool parse_item(selector_type selector,
                    selector_item &item)
    {
        item.left.clear();
        item.right.clear();
        item.is_range = 0;
        item.open_left = 0;
        item.open_right = 0;
        item.valid = 1;

        if (selector == SELECT_RESI && peek().type == TOK_MINUS) {
            take();
            if (peek().type != TOK_VALUE) return false;
            item.right = take().value;
            item.is_range = 1;
            item.open_left = 1;
            return true;
        }

        if (peek().type != TOK_VALUE) return false;
        item.left = take().value;

        if ((selector == SELECT_RESI || selector == SELECT_CHAIN) && peek().type == TOK_MINUS) {
            take();
            item.is_range = 1;
            if (peek().type == TOK_VALUE) {
                item.right = take().value;
            } else if (selector == SELECT_RESI) {
                item.open_right = 1;
            } else {
                return false;
            }
        }
        return true;
    }
};

static void
selection_warn(fastsasa_selection_warning_callback callback,
               void *userdata,
               int *warning_count,
               const std::string &message)
{
    if (warning_count != nullptr) ++*warning_count;
    if (callback != nullptr) {
        std::string branded = "FastSASA: warning: " + message;
        callback(branded.c_str(), userdata);
    }
}

static const char *
selector_name(selector_type selector)
{
    switch (selector) {
    case SELECT_NAME:
        return "name";
    case SELECT_SYMBOL:
        return "symbol";
    case SELECT_RESN:
        return "resn";
    case SELECT_RESI:
        return "resi";
    case SELECT_CHAIN:
        return "chain";
    case SELECT_SEGID:
        return "segid";
    case SELECT_PROTEIN:
        return "protein";
    }
    return "selection";
}

static int
is_letters_only(const std::string &value)
{
    if (value.empty()) return 0;
    for (std::string::const_iterator it = value.begin(); it != value.end(); ++it) {
        if (!std::isalpha(static_cast<unsigned char>(*it))) return 0;
    }
    return 1;
}

static int
is_residue_insertion_code(const std::string &value)
{
    size_t n = value.size();

    if (n < 2 || n > 5) return 0;
    if (!std::isalpha(static_cast<unsigned char>(value[n - 1]))) return 0;
    for (size_t i = 0; i + 1 < n; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(value[i]))) return 0;
    }
    return 1;
}

static int
is_chain_range_valid(const selector_item &item)
{
    int left_number;
    int right_number;
    int left_is_number = parse_integer(item.left, &left_number);
    int right_is_number = parse_integer(item.right, &right_number);

    if (left_is_number || right_is_number) return left_is_number && right_is_number;
    return item.left.size() == 1 && item.right.size() == 1;
}

static int
validate_selector_item(selector_type selector,
                       selector_item &item,
                       fastsasa_selection_warning_callback callback,
                       void *userdata,
                       int *warning_count)
{
    int parsed;

    if (selector == SELECT_PROTEIN) return 1;
    if (selector == SELECT_SEGID) return 1;

    if (item.is_range) {
        if (selector == SELECT_RESI) {
            if ((item.open_left || parse_integer(item.left, &parsed)) &&
                (item.open_right || parse_integer(item.right, &parsed))) {
                return 1;
            }
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid resi range '" +
                               item.left + "-" + item.right + "'; residue ranges need numeric bounds");
            item.valid = 0;
            return 0;
        }
        if (selector == SELECT_CHAIN && is_chain_range_valid(item)) return 1;
        selection_warn(callback, userdata, warning_count,
                       "selection: ignoring invalid chain range '" +
                           item.left + "-" + item.right +
                           "'; chain ranges need two single-character chains or two numbers");
        item.valid = 0;
        return 0;
    }

    switch (selector) {
    case SELECT_NAME:
        if (item.left.size() > 4u) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid atom name '" +
                               item.left + "'; atom names are limited to 4 characters");
            item.valid = 0;
            return 0;
        }
        break;
    case SELECT_SYMBOL:
        if (!is_letters_only(item.left)) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid element symbol '" +
                               item.left + "'; symbols must be one or two letters");
            item.valid = 0;
            return 0;
        }
        if (item.left.size() > 2u) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid element symbol '" +
                               item.left + "'; symbols are limited to two characters");
            item.valid = 0;
            return 0;
        }
        break;
    case SELECT_RESN:
        if (item.left.size() > 3u) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid residue name '" +
                               item.left + "'; residue names are limited to 3 characters");
            item.valid = 0;
            return 0;
        }
        break;
    case SELECT_RESI:
        if (!parse_integer(item.left, &parsed) && !is_residue_insertion_code(item.left)) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid residue id '" +
                               item.left + "'; use a number or a number plus one insertion code such as 125A");
            item.valid = 0;
            return 0;
        }
        break;
    case SELECT_CHAIN:
        if (item.left.size() > 1u) {
            selection_warn(callback, userdata, warning_count,
                           "selection: ignoring invalid chain id '" +
                               item.left + "'; chain ids are limited to one character");
            item.valid = 0;
            return 0;
        }
        break;
    case SELECT_SEGID:
    case SELECT_PROTEIN:
        break;
    }
    return 1;
}

static int
residue_number_at(const fastsasa_owned_topology *topology,
                  int atom)
{
    return topology->residue_numbers != nullptr ? topology->residue_numbers[atom] : 0;
}

static int
item_matches_value(selector_type selector,
                   const fastsasa_owned_topology *topology,
                   int atom,
                   const selector_item &item)
{
    int parsed;

    switch (selector) {
    case SELECT_NAME:
        return equals_ci(topology->atom_names[atom], item.left);
    case SELECT_SYMBOL:
        return equals_ci(topology->elements[atom], item.left);
    case SELECT_RESN:
        return equals_ci(topology->residue_names[atom], item.left);
    case SELECT_RESI:
        if (topology->residue_number_strings != nullptr &&
            equals_ci(topology->residue_number_strings[atom], item.left)) {
            return 1;
        }
        return parse_integer(item.left, &parsed) && residue_number_at(topology, atom) == parsed;
    case SELECT_CHAIN:
        return equals_ci(topology->chain_ids[atom], item.left);
    case SELECT_SEGID:
        return equals_ci(topology->segment_ids[atom], item.left);
    case SELECT_PROTEIN:
        return 0;
    }
    return 0;
}

static int
item_matches_range(selector_type selector,
                   const fastsasa_owned_topology *topology,
                   int atom,
                   const selector_item &item)
{
    int lower = 0;
    int upper = 0;
    int value = 0;

    if (selector == SELECT_RESI) {
        if (topology->n_atoms <= 0) return 0;
        if (item.open_left) {
            lower = residue_number_at(topology, 0);
            if (!parse_integer(item.right, &upper)) return 0;
        } else if (item.open_right) {
            if (!parse_integer(item.left, &lower)) return 0;
            upper = residue_number_at(topology, topology->n_atoms - 1);
        } else {
            if (!parse_integer(item.left, &lower) || !parse_integer(item.right, &upper)) return 0;
        }
        value = residue_number_at(topology, atom);
        return value >= lower && value <= upper;
    }

    if (selector == SELECT_CHAIN) {
        int left_number;
        int right_number;

        if (parse_integer(item.left, &left_number) && parse_integer(item.right, &right_number)) {
            if (!parse_integer(topology->chain_ids[atom], &value)) return 0;
            return value >= left_number && value <= right_number;
        }
        if (item.left.size() != 1 || item.right.size() != 1 ||
            topology->chain_ids[atom] == nullptr || std::strlen(topology->chain_ids[atom]) != 1) {
            return 0;
        }
        value = std::toupper(static_cast<unsigned char>(topology->chain_ids[atom][0]));
        lower = std::toupper(static_cast<unsigned char>(item.left[0]));
        upper = std::toupper(static_cast<unsigned char>(item.right[0]));
        return value >= lower && value <= upper;
    }

    return 0;
}

static void
validate_selection_tree(selection_node *node,
                        const fastsasa_owned_topology *topology,
                        fastsasa_selection_warning_callback callback,
                        void *userdata,
                        int *warning_count)
{
    if (node == nullptr) return;
    validate_selection_tree(node->left, topology, callback, userdata, warning_count);
    validate_selection_tree(node->right, topology, callback, userdata, warning_count);

    if (node->type != NODE_SELECTOR || node->selector == SELECT_PROTEIN) return;

    for (std::vector<selector_item>::iterator it = node->items.begin(); it != node->items.end(); ++it) {
        int matches = 0;

        if (!validate_selector_item(node->selector, *it, callback, userdata, warning_count)) continue;
        if (it->is_range) continue;

        for (int atom = 0; atom < topology->n_atoms; ++atom) {
            if (item_matches_value(node->selector, topology, atom, *it)) ++matches;
        }
        if (matches == 0) {
            selection_warn(callback, userdata, warning_count,
                           std::string("selection: no atoms matched ") +
                               selector_name(node->selector) + " '" + it->left + "'");
        }
    }
}

static int
is_standard_protein_residue(const char *residue_name)
{
    static const char *residues[] = {
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL",
        "SEC", "PYL", "ASX", "GLX",
        "HSD", "HSE", "HSP", "HID", "HIE", "HIP"
    };

    if (residue_name == nullptr) return 0;
    for (size_t i = 0; i < sizeof(residues) / sizeof(residues[0]); ++i) {
        if (equals_ci(residue_name, residues[i])) return 1;
    }
    return 0;
}

static int
atom_matches_selector(const fastsasa_owned_topology *topology,
                      int atom,
                      const selection_node *node)
{
    if (node->selector == SELECT_PROTEIN) {
        return is_standard_protein_residue(topology->residue_names[atom]);
    }

    for (std::vector<selector_item>::const_iterator it = node->items.begin(); it != node->items.end(); ++it) {
        if (!it->valid) continue;
        if (it->is_range) {
            if (item_matches_range(node->selector, topology, atom, *it)) return 1;
        } else if (item_matches_value(node->selector, topology, atom, *it)) {
            return 1;
        }
    }
    return 0;
}

static int
atom_matches_node(const fastsasa_owned_topology *topology,
                  int atom,
                  const selection_node *node)
{
    if (node == nullptr) return 0;

    switch (node->type) {
    case NODE_SELECTOR:
        return atom_matches_selector(topology, atom, node);
    case NODE_AND:
        return atom_matches_node(topology, atom, node->left) &&
               atom_matches_node(topology, atom, node->right);
    case NODE_OR:
        return atom_matches_node(topology, atom, node->left) ||
               atom_matches_node(topology, atom, node->right);
    case NODE_NOT:
        return !atom_matches_node(topology, atom, node->right);
    }
    return 0;
}

static int
topology_selection_mask_ex_impl(const char *command,
                                const fastsasa_owned_topology *topology,
                                unsigned int bit,
                                unsigned int *atom_masks,
                                char *name,
                                size_t name_size,
                                fastsasa_selection_warning_callback warning_callback,
                                void *warning_userdata,
                                int *warning_count)
{
    std::string text;
    std::string expression;
    selection_node *parsed = nullptr;
    size_t comma;
    int n_selected = 0;
    size_t copy_length;

    if (command == nullptr || topology == nullptr || atom_masks == nullptr ||
        name == nullptr || name_size == 0 || bit == 0u ||
        topology->atom_names == nullptr || topology->residue_names == nullptr ||
        topology->chain_ids == nullptr || topology->segment_ids == nullptr ||
        topology->elements == nullptr ||
        topology->residue_numbers == nullptr || topology->residue_number_strings == nullptr) {
        return 0;
    }
    if (warning_count != nullptr) *warning_count = 0;

    text = trim_copy(command);
    comma = text.find(',');
    if (comma == std::string::npos) {
        std::string generated_name;

        expression = text;
        generated_name = selection_name_from_expression(expression);
        copy_length = generated_name.size();
        if (copy_length >= name_size) copy_length = name_size - 1u;
        std::memcpy(name, generated_name.c_str(), copy_length);
        name[copy_length] = '\0';
    } else {
        std::string selection_name = trim_copy(text.substr(0, comma));
        expression = trim_copy(text.substr(comma + 1));
        if (selection_name.empty()) selection_name = selection_name_from_expression(expression);
        copy_length = selection_name.size();
        if (copy_length > FASTSASA_MAX_SELECTION_NAME) copy_length = FASTSASA_MAX_SELECTION_NAME;
        if (copy_length >= name_size) copy_length = name_size - 1u;
        std::memcpy(name, selection_name.c_str(), copy_length);
        name[copy_length] = '\0';
    }
    if (expression.empty()) return 0;

    parsed = selection_parser(expression).parse();
    if (parsed == nullptr) return 0;

    validate_selection_tree(parsed, topology, warning_callback, warning_userdata, warning_count);

    for (int atom = 0; atom < topology->n_atoms; ++atom) {
        if (atom_matches_node(topology, atom, parsed)) {
            atom_masks[atom] |= bit;
            ++n_selected;
        }
    }
    free_node(parsed);
    return 1;
}

int
fastsasa_topology_selection_mask_ex(const char *command,
                                  const fastsasa_owned_topology *topology,
                                  unsigned int bit,
                                  unsigned int *atom_masks,
                                  char *name,
                                  size_t name_size,
                                  fastsasa_selection_warning_callback warning_callback,
                                  void *warning_userdata,
                                  int *warning_count)
{
    try {
        return topology_selection_mask_ex_impl(command, topology, bit, atom_masks,
                                               name, name_size, warning_callback,
                                               warning_userdata, warning_count);
    } catch (...) {
        return 0;
    }
}

int
fastsasa_topology_selection_mask(const char *command,
                               const fastsasa_owned_topology *topology,
                               unsigned int bit,
                               unsigned int *atom_masks,
                               char *name,
                               size_t name_size)
{
    return fastsasa_topology_selection_mask_ex(command,
                                            topology,
                                            bit,
                                            atom_masks,
                                            name,
                                            name_size,
                                            nullptr,
                                            nullptr,
                                            nullptr);
}
