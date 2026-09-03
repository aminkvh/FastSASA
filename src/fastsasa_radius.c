#include "fastsasa_radius.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

enum {
    FASTSASA_RADIUS_CLASS_UNKNOWN = -1,
    FASTSASA_RADIUS_CLASS_APOLAR = 0,
    FASTSASA_RADIUS_CLASS_POLAR = 1
};

static void
trim_copy(char *dst,
          size_t dst_size,
          const char *src)
{
    size_t begin = 0u;
    size_t end = src != NULL ? strlen(src) : 0u;
    size_t n;

    if (dst_size == 0u) return;
    while (begin < end && isspace((unsigned char)src[begin])) ++begin;
    while (end > begin && isspace((unsigned char)src[end - 1u])) --end;
    n = end - begin;
    if (n >= dst_size) n = dst_size - 1u;
    if (n > 0u) memcpy(dst, src + begin, n);
    dst[n] = '\0';
}

static void
upper_inplace(char *text)
{
    if (text == NULL) return;
    for (; *text != '\0'; ++text) *text = (char)toupper((unsigned char)*text);
}

/* Bondi, A. van der Waals Volumes and Radii. J Phys Chem. 1964;68(3):441-451. */
double
fastsasa_element_radius(const char *element)
{
    char key[8];

    trim_copy(key, sizeof(key), element);
    upper_inplace(key);
    if (strcmp(key, "H") == 0 || strcmp(key, "D") == 0) return 1.10;
    if (strcmp(key, "C") == 0) return 1.70;
    if (strcmp(key, "N") == 0) return 1.55;
    if (strcmp(key, "O") == 0) return 1.52;
    if (strcmp(key, "F") == 0) return 1.47;
    if (strcmp(key, "P") == 0) return 1.80;
    if (strcmp(key, "S") == 0) return 1.80;
    if (strcmp(key, "CL") == 0) return 1.75;
    if (strcmp(key, "BR") == 0) return 1.85;
    if (strcmp(key, "I") == 0) return 1.98;
    if (strcmp(key, "NA") == 0) return 2.27;
    if (strcmp(key, "MG") == 0) return 1.73;
    if (strcmp(key, "K") == 0) return 2.75;
    if (strcmp(key, "CA") == 0) return 2.31;
    if (strcmp(key, "FE") == 0) return 1.80;
    if (strcmp(key, "ZN") == 0) return 1.39;
    return -1.0;
}

int
fastsasa_infer_element(char *dst,
                     size_t dst_size,
                     const char *atom_name,
                     const char *column_element)
{
    char element[8];

    if (dst_size == 0u) return 0;
    trim_copy(element, sizeof(element), column_element);
    if (element[0] != '\0') {
        upper_inplace(element);
        snprintf(dst, dst_size, "%s", element);
        return 1;
    }
    for (const char *p = atom_name; p != NULL && *p != '\0'; ++p) {
        if (isalpha((unsigned char)*p)) {
            char first = (char)toupper((unsigned char)*p);
            char second = '\0';

            if (p[1] != '\0' && isalpha((unsigned char)p[1])) second = (char)toupper((unsigned char)p[1]);
            if ((first == 'C' && second == 'L') ||
                (first == 'B' && second == 'R') ||
                (first == 'N' && second == 'A') ||
                (first == 'M' && second == 'G') ||
                (first == 'F' && second == 'E') ||
                (first == 'Z' && second == 'N') ||
                (first == 'C' && second == 'A')) {
                snprintf(dst, dst_size, "%c%c", first, second);
                return 1;
            }
            snprintf(dst, dst_size, "%c", first);
            return 1;
        }
    }
    dst[0] = '\0';
    return 0;
}

int
fastsasa_element_class(const char *element)
{
    char key[8];

    trim_copy(key, sizeof(key), element);
    upper_inplace(key);
    if (strcmp(key, "C") == 0) return FASTSASA_RADIUS_CLASS_APOLAR;
    if (key[0] != '\0' && strcmp(key, "H") != 0 && strcmp(key, "D") != 0) {
        return FASTSASA_RADIUS_CLASS_POLAR;
    }
    return FASTSASA_RADIUS_CLASS_UNKNOWN;
}

const char *
fastsasa_canonical_residue(const char *residue_name)
{
    static const struct {
        const char *alias;
        const char *canonical;
    } aliases[] = {
        {"HID", "HIS"}, {"HIE", "HIS"}, {"HIP", "HIS"},
        {"HSD", "HIS"}, {"HSE", "HIS"}, {"HSP", "HIS"},
        {"CYX", "CYS"}, {"CYM", "CYS"},
        {"ASH", "ASP"}, {"GLH", "GLU"}, {"LYN", "LYS"}, {"ARN", "ARG"},
    };
    char key[8];
    size_t i;

    trim_copy(key, sizeof(key), residue_name);
    for (i = 0u; key[i] != '\0'; ++i) key[i] = (char)toupper((unsigned char)key[i]);
    for (i = 0u; i < sizeof(aliases) / sizeof(aliases[0]); ++i) {
        if (strcmp(key, aliases[i].alias) == 0) return aliases[i].canonical;
    }
    return NULL;
}

const char *
fastsasa_default_config_candidate(size_t index)
{
    static const char *candidates[] = {
#ifdef FASTSASA_INSTALL_DEFAULT_CONFIG
        FASTSASA_INSTALL_DEFAULT_CONFIG,
#endif
        "share/protor.config",
        "../share/protor.config",
        "../../share/protor.config",
        "/usr/local/share/fastsasa/protor.config",
        "/usr/share/fastsasa/protor.config"
    };

    if (index >= sizeof(candidates) / sizeof(candidates[0])) return NULL;
    return candidates[index];
}
