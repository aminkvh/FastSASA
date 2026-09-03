#include "fastsasa_xtc.h"

#include <stdlib.h>
#include <string.h>

#ifdef FASTSASA_HAVE_MOLFILE
#include <dlfcn.h>
#include <molfile_plugin.h>

typedef int (*molfile_runtime_init_fn)(void);
typedef int (*molfile_runtime_register_fn)(void *, vmdplugin_register_cb);

typedef struct plugin_search {
    molfile_plugin_t *plugin;
} plugin_search;

static int
register_plugin(void *data,
                vmdplugin_t *plugin)
{
    plugin_search *search = (plugin_search *)data;

    if (plugin == NULL || search == NULL) return VMDPLUGIN_ERROR;
    if (strcmp(plugin->type, MOLFILE_PLUGIN_TYPE) == 0 &&
        strcmp(plugin->name, "xtc") == 0) {
        search->plugin = (molfile_plugin_t *)plugin;
    }
    return VMDPLUGIN_SUCCESS;
}

static const char *
plugin_path(int index)
{
    const char *env_path = getenv("FASTSASA_MOLFILE_GROMACS_PLUGIN");


    if (index == 0 && env_path != NULL && env_path[0] != '\0') return env_path;
    if (env_path != NULL && env_path[0] != '\0') --index;

    switch (index) {
    case 0:
        return "/usr/local/lib/v" "md/plugins/LINUXAMD64/molfile/gromacsplugin.so";
    case 1:
        return "/usr/local/lib/v" "md2/plugins/LINUXAMD64/molfile/gromacsplugin.so";
    default:
        return NULL;
    }
}

static int
load_xtc_plugin(fastsasa_xtc *xtc)
{
    for (int i = 0;; ++i) {
        const char *path = plugin_path(i);
        void *library;
        molfile_runtime_init_fn init;
        molfile_runtime_register_fn register_fn;
        plugin_search search = {NULL};

        if (path == NULL) return 0;

        library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (library == NULL) continue;

        init = (molfile_runtime_init_fn)dlsym(library, "vmdplugin_init");
        register_fn = (molfile_runtime_register_fn)dlsym(library, "vmdplugin_register");
        if (init == NULL || register_fn == NULL || init() != VMDPLUGIN_SUCCESS ||
            register_fn(&search, register_plugin) != VMDPLUGIN_SUCCESS ||
            search.plugin == NULL) {
            dlclose(library);
            continue;
        }

        xtc->plugin_library = library;
        xtc->plugin = search.plugin;
        return 1;
    }
}
#endif

int
fastsasa_xtc_open(fastsasa_xtc *xtc,
                      const char *path)
{
#ifdef FASTSASA_HAVE_MOLFILE
    molfile_plugin_t *plugin;
    int n_atoms = 0;
    size_t path_len;

    if (xtc == NULL || path == NULL) return 0;
    memset(xtc, 0, sizeof(*xtc));

    if (!load_xtc_plugin(xtc)) return 0;
    plugin = (molfile_plugin_t *)xtc->plugin;
    if (plugin->open_file_read == NULL || plugin->read_next_timestep == NULL ||
        plugin->close_file_read == NULL) {
        fastsasa_xtc_close(xtc);
        return 0;
    }

    xtc->file_handle = plugin->open_file_read(path, "xtc", &n_atoms);
    if (xtc->file_handle == NULL || n_atoms <= 0) {
        fastsasa_xtc_close(xtc);
        return 0;
    }

    xtc->coords = malloc(sizeof(float) * 3u * (size_t)n_atoms);
    if (xtc->coords == NULL) {
        fastsasa_xtc_close(xtc);
        return 0;
    }
    path_len = strlen(path);
    xtc->path = (char *)malloc(path_len + 1u);
    if (xtc->path == NULL) {
        fastsasa_xtc_close(xtc);
        return 0;
    }
    memcpy(xtc->path, path, path_len + 1u);
    xtc->n_atoms = n_atoms;
    xtc->current_frame = 0;

    if (plugin->read_timestep_metadata != NULL) {
        molfile_timestep_metadata_t metadata;

        memset(&metadata, 0, sizeof(metadata));
        if (plugin->read_timestep_metadata(xtc->file_handle, &metadata) == MOLFILE_SUCCESS) {
            xtc->n_frames = (int)metadata.count;
        }
    }
    return 1;
#else
    (void)xtc;
    (void)path;
    return 0;
#endif
}

void
fastsasa_xtc_close(fastsasa_xtc *xtc)
{
    if (xtc == NULL) return;
#ifdef FASTSASA_HAVE_MOLFILE
    if (xtc->plugin != NULL && xtc->file_handle != NULL) {
        molfile_plugin_t *plugin = (molfile_plugin_t *)xtc->plugin;

        if (plugin->close_file_read != NULL) plugin->close_file_read(xtc->file_handle);
    }
    free(xtc->path);
    free(xtc->coords);
    if (xtc->plugin_library != NULL) dlclose(xtc->plugin_library);
#endif
    memset(xtc, 0, sizeof(*xtc));
}

int
fastsasa_xtc_read_frame_soa(fastsasa_xtc *xtc,
                                double *x,
                                double *y,
                                double *z)
{
#ifdef FASTSASA_HAVE_MOLFILE
    molfile_plugin_t *plugin;
    molfile_timestep_t timestep;
    int status;

    if (xtc == NULL || xtc->file_handle == NULL || xtc->coords == NULL ||
        x == NULL || y == NULL || z == NULL) {
        return 0;
    }

    plugin = (molfile_plugin_t *)xtc->plugin;
    memset(&timestep, 0, sizeof(timestep));
    timestep.coords = xtc->coords;

    status = plugin->read_next_timestep(xtc->file_handle, xtc->n_atoms, &timestep);
    if (status != MOLFILE_SUCCESS) return 0;
    ++xtc->current_frame;

    for (int atom = 0; atom < xtc->n_atoms; ++atom) {
        x[atom] = (double)xtc->coords[3 * atom];
        y[atom] = (double)xtc->coords[3 * atom + 1];
        z[atom] = (double)xtc->coords[3 * atom + 2];
    }
    return 1;
#else
    (void)xtc;
    (void)x;
    (void)y;
    (void)z;
    return 0;
#endif
}

int
fastsasa_xtc_seek_frame(fastsasa_xtc *xtc,
                      int frame)
{
#ifdef FASTSASA_HAVE_MOLFILE
    molfile_plugin_t *plugin;
    molfile_timestep_t timestep;

    if (xtc == NULL || xtc->plugin == NULL || xtc->coords == NULL ||
        xtc->path == NULL || frame < 0) {
        return 0;
    }
    if (xtc->n_frames > 0 && frame >= xtc->n_frames) return 0;

    plugin = (molfile_plugin_t *)xtc->plugin;
    if (plugin->read_next_timestep == NULL || plugin->open_file_read == NULL ||
        plugin->close_file_read == NULL) {
        return 0;
    }

    if (frame < xtc->current_frame || xtc->file_handle == NULL) {
        int n_atoms = 0;
        void *file_handle;

        if (xtc->file_handle != NULL) {
            plugin->close_file_read(xtc->file_handle);
            xtc->file_handle = NULL;
        }
        file_handle = plugin->open_file_read(xtc->path, "xtc", &n_atoms);
        if (file_handle == NULL) return 0;
        if (n_atoms != xtc->n_atoms) {
            plugin->close_file_read(file_handle);
            return 0;
        }
        xtc->file_handle = file_handle;
        xtc->current_frame = 0;
    }

    memset(&timestep, 0, sizeof(timestep));
    timestep.coords = xtc->coords;
    while (xtc->current_frame < frame) {
        if (plugin->read_next_timestep(xtc->file_handle, xtc->n_atoms, &timestep) != MOLFILE_SUCCESS) {
            return 0;
        }
        ++xtc->current_frame;
    }
    return 1;
#else
    (void)xtc;
    (void)frame;
    return 0;
#endif
}
