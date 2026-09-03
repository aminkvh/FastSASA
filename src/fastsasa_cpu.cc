#include "fastsasa.h"
#include "fastsasa_cpu.h"
#include "fastsasa_exact_math.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

static const double FASTSASA_CPU_PI = 3.141592653589793238462643383279502884;

static void
join_threads(std::vector<std::thread> *threads) noexcept
{
    for (std::thread &thread : *threads) {
        if (thread.joinable()) thread.join();
    }
}

struct cell_key {
    int x;
    int y;
    int z;

    bool operator==(const cell_key &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct cell_hash {
    size_t operator()(const cell_key &key) const
    {
        size_t hash = static_cast<size_t>(key.x) * 73856093u;
        hash ^= static_cast<size_t>(key.y) * 19349663u;
        hash ^= static_cast<size_t>(key.z) * 83492791u;
        return hash;
    }
};

class cell_list {
public:
    cell_list(int n_atoms,
              const double *x,
              const double *y,
              const double *z,
              double cell_size)
        : cell_size_(cell_size)
    {
        for (int atom = 0; atom < n_atoms; ++atom) {
            cells_[key(x[atom], y[atom], z[atom])].push_back(atom);
        }
    }

    template <typename Visitor>
    void visit(double x,
               double y,
               double z,
               Visitor visitor) const
    {
        const cell_key center = key(x, y, z);

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const cell_key neighbor = {center.x + dx, center.y + dy, center.z + dz};
                    const auto found = cells_.find(neighbor);

                    if (found == cells_.end()) continue;
                    for (int atom : found->second) visitor(atom);
                }
            }
        }
    }

    void collect(double x,
                 double y,
                 double z,
                 std::vector<int> &atoms) const
    {
        const cell_key center = key(x, y, z);

        atoms.clear();
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const cell_key neighbor = {center.x + dx, center.y + dy, center.z + dz};
                    const auto found = cells_.find(neighbor);

                    if (found == cells_.end()) continue;
                    atoms.insert(atoms.end(), found->second.begin(), found->second.end());
                }
            }
        }
    }

private:
    cell_key key(double x,
                 double y,
                 double z) const
    {
        const cell_key result = {
            static_cast<int>(std::floor(x / cell_size_)),
            static_cast<int>(std::floor(y / cell_size_)),
            static_cast<int>(std::floor(z / cell_size_))
        };
        return result;
    }

    double cell_size_;
    std::unordered_map<cell_key, std::vector<int>, cell_hash> cells_;
};

static double
maximum_radius(const double *radii,
               int n_atoms)
{
    double radius = 0.0;

    for (int atom = 0; atom < n_atoms; ++atom) radius = std::max(radius, radii[atom]);
    return radius;
}

static bool
finite_geometry(int n_atoms,
                const double *x,
                const double *y,
                const double *z,
                const double *expanded_radii)
{
    for (int atom = 0; atom < n_atoms; ++atom) {
        if (!std::isfinite(x[atom]) || !std::isfinite(y[atom]) ||
            !std::isfinite(z[atom]) || !std::isfinite(expanded_radii[atom]) ||
            expanded_radii[atom] <= 0.0) {
            return false;
        }
    }
    return true;
}

static bool
finite_test_points(int n_points,
                   const double *test_points)
{
    for (int point = 0; point < 3 * n_points; ++point) {
        if (!std::isfinite(test_points[point])) return false;
    }
    return true;
}

static bool
cell_keys_fit(int n_atoms,
              const double *x,
              const double *y,
              const double *z,
              double cell_size)
{
    const double minimum = static_cast<double>(std::numeric_limits<int>::min()) + 1.0;
    const double maximum = static_cast<double>(std::numeric_limits<int>::max()) - 1.0;

    for (int atom = 0; atom < n_atoms; ++atom) {
        const double cx = std::floor(x[atom] / cell_size);
        const double cy = std::floor(y[atom] / cell_size);
        const double cz = std::floor(z[atom] / cell_size);

        if (cx < minimum || cx > maximum ||
            cy < minimum || cy > maximum ||
            cz < minimum || cz > maximum) {
            return false;
        }
    }
    return true;
}

static int
thread_count(int requested,
             int n_atoms)
{
    int count = requested > 0 ? requested : fastsasa_cpu_default_threads();

    if (count < 1) count = 1;
    if (count > n_atoms) count = n_atoms;
    return count;
}

int
fastsasa_cpu_default_threads(void)
{
    const unsigned int available = std::thread::hardware_concurrency();

    return available > 1u ? static_cast<int>(available - 1u) : 1;
}

static int
env_enabled(const char *name,
            int default_value)
{
    const char *value = std::getenv(name);

    if (value == nullptr || value[0] == '\0') return default_value;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

static int
normalized_thread_count(int requested)
{
    int count = requested > 0 ? requested : fastsasa_cpu_default_threads();

    return count > 0 ? count : 1;
}

static int
cpu_frame_thread_count(int total_threads,
                       int n_frames,
                       int n_atoms)
{
    const char *value = std::getenv("FASTSASA_CPU_FRAME_THREADS");
    int requested = 0;

    if (n_frames <= 1 || total_threads <= 1) return 1;
    if (value != nullptr && value[0] != '\0') requested = std::atoi(value);
    if (requested <= 0) {
        if (n_atoms >= 15000) requested = 1;
        else if (n_atoms >= 5000) requested = std::max(1, total_threads / 3);
        else requested = std::max(1, total_threads / 2);
    }
    if (requested > total_threads) requested = total_threads;
    if (requested > n_frames) requested = n_frames;
    return requested > 0 ? requested : 1;
}

static int
cpu_shrake_rupley_impl(int n_atoms,
                       int n_points,
                       const double *x,
                       const double *y,
                       const double *z,
                       const double *expanded_radii,
                       const double *test_points,
                       int n_threads,
                       double *sasa)
{
    std::vector<std::thread> threads;
    std::vector<double> radius2;
    std::atomic<int> worker_status(FASTSASA_SUCCESS);
    int count;

    if (n_atoms <= 0 || n_points <= 0 || x == nullptr || y == nullptr ||
        z == nullptr || expanded_radii == nullptr || test_points == nullptr ||
        sasa == nullptr) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!finite_geometry(n_atoms, x, y, z, expanded_radii) ||
        !finite_test_points(n_points, test_points)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    const double max_radius = maximum_radius(expanded_radii, n_atoms);
    if (max_radius <= 0.0 || !cell_keys_fit(n_atoms, x, y, z, max_radius)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    radius2.resize(static_cast<size_t>(n_atoms));
    for (int atom = 0; atom < n_atoms; ++atom) {
        radius2[static_cast<size_t>(atom)] = expanded_radii[atom] * expanded_radii[atom];
    }
    count = thread_count(n_threads, n_atoms);
    threads.reserve(static_cast<size_t>(count));
    if (!env_enabled("FASTSASA_CPU_SIMD", 1)) {
        const cell_list cells(n_atoms, x, y, z, max_radius);

        try {
            for (int thread_id = 0; thread_id < count; ++thread_id) {
                const int begin = static_cast<int>(static_cast<size_t>(thread_id) *
                                                   static_cast<size_t>(n_atoms) /
                                                   static_cast<size_t>(count));
                const int end = static_cast<int>(static_cast<size_t>(thread_id + 1) *
                                                 static_cast<size_t>(n_atoms) /
                                                 static_cast<size_t>(count));

                threads.emplace_back([=, &cells, &radius2, &worker_status]() {
                    try {
                for (int atom = begin; atom < end; ++atom) {
                    const double radius = expanded_radii[atom];
                    int accessible = 0;

                    for (int point = 0; point < n_points; ++point) {
                        const double px = x[atom] + radius * test_points[3 * point];
                        const double py = y[atom] + radius * test_points[3 * point + 1];
                        const double pz = z[atom] + radius * test_points[3 * point + 2];
                        int buried = 0;

                        cells.visit(px, py, pz, [&](int other) {
                            if (buried) return;
                            const double dx = px - x[other];
                            const double dy = py - y[other];
                            const double dz = pz - z[other];

                            if (other != atom && dx * dx + dy * dy + dz * dz < radius2[static_cast<size_t>(other)]) {
                                buried = 1;
                            }
                        });
                        if (!buried) ++accessible;
                    }
                    sasa[atom] = 4.0 * FASTSASA_CPU_PI * radius * radius *
                                 static_cast<double>(accessible) / static_cast<double>(n_points);
                }
                    } catch (const std::bad_alloc &) {
                        worker_status.store(FASTSASA_MEMORY_ERROR);
                    } catch (...) {
                        worker_status.store(FASTSASA_INVALID_ARGUMENT);
                    }
                });
            }
        } catch (...) {
            join_threads(&threads);
            throw;
        }
        join_threads(&threads);
        return worker_status.load();
    }

    const cell_list cells(n_atoms, x, y, z, 2.0 * max_radius);
    const int point_simd = env_enabled("FASTSASA_CPU_POINT_SIMD", 0);
    try {
        for (int thread_id = 0; thread_id < count; ++thread_id) {
            const int begin = static_cast<int>(static_cast<size_t>(thread_id) *
                                               static_cast<size_t>(n_atoms) /
                                               static_cast<size_t>(count));
            const int end = static_cast<int>(static_cast<size_t>(thread_id + 1) *
                                             static_cast<size_t>(n_atoms) /
                                             static_cast<size_t>(count));

            threads.emplace_back([=, &cells, &radius2, &worker_status]() {
                try {
            std::vector<int> candidates;
            std::vector<int> neighbors;
            candidates.reserve(128u);
            neighbors.reserve(64u);

            for (int atom = begin; atom < end; ++atom) {
                const double radius = expanded_radii[atom];
                int accessible = 0;

                cells.collect(x[atom], y[atom], z[atom], candidates);
                neighbors.clear();
                for (int candidate : candidates) {
                    const double neighbor_radius = expanded_radii[candidate];
                    const double max_distance = radius + neighbor_radius;
                    const double dx = x[atom] - x[candidate];
                    const double dy = y[atom] - y[candidate];
                    const double dz = z[atom] - z[candidate];

                    if (candidate != atom &&
                        dx * dx + dy * dy + dz * dz < max_distance * max_distance) {
                        neighbors.push_back(candidate);
                    }
                }
                for (int point = 0; point < n_points; ++point) {
                    const double px = x[atom] + radius * test_points[3 * point];
                    const double py = y[atom] + radius * test_points[3 * point + 1];
                    const double pz = z[atom] + radius * test_points[3 * point + 2];
                    int buried = 0;
                    const int n_candidates = static_cast<int>(neighbors.size());
                    const int *candidate_atoms = neighbors.data();

                    if (!point_simd) {
                        for (int candidate = 0; candidate < n_candidates; ++candidate) {
                            const int other = candidate_atoms[candidate];
                            const double dx = px - x[other];
                            const double dy = py - y[other];
                            const double dz = pz - z[other];

                            if (dx * dx + dy * dy + dz * dz < radius2[static_cast<size_t>(other)]) {
                                buried = 1;
                                break;
                            }
                        }
                        if (!buried) ++accessible;
                        continue;
                    }
#if defined(__GNUC__) || defined(__clang__)
#pragma omp simd reduction(|:buried)
#endif
                    for (int candidate = 0; candidate < n_candidates; ++candidate) {
                        const int other = candidate_atoms[candidate];
                        const double dx = px - x[other];
                        const double dy = py - y[other];
                        const double dz = pz - z[other];
                        const int occludes = dx * dx + dy * dy + dz * dz < radius2[static_cast<size_t>(other)];

                        buried |= occludes;
                    }
                    if (!buried) ++accessible;
                }
                sasa[atom] = 4.0 * FASTSASA_CPU_PI * radius * radius *
                             static_cast<double>(accessible) / static_cast<double>(n_points);
            }
                } catch (const std::bad_alloc &) {
                    worker_status.store(FASTSASA_MEMORY_ERROR);
                } catch (...) {
                    worker_status.store(FASTSASA_INVALID_ARGUMENT);
                }
            });
        }
    } catch (...) {
        join_threads(&threads);
        throw;
    }
    join_threads(&threads);
    return worker_status.load();
}

int
fastsasa_cpu_shrake_rupley(int n_atoms,
                         int n_points,
                         const double *x,
                         const double *y,
                         const double *z,
                         const double *expanded_radii,
                         const double *test_points,
                         int n_threads,
                         double *sasa)
{
    try {
        return cpu_shrake_rupley_impl(n_atoms, n_points, x, y, z,
                                      expanded_radii, test_points, n_threads, sasa);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

/*
 * FP32 Shrake-Rupley is a deliberately separate implementation, not a
 * template instantiation shared with the FP64 path above: the FP64 CPU
 * kernel is relied on elsewhere to be bit-identical to the CUDA/Vulkan FP64
 * backends (see fastsasa_backend_bit_identity), and CUDA's rare-atom
 * recompute depends on the FP64 CPU reference being unchanged. Duplicating
 * the (small, portable, non-SIMD-intrinsic) hot loop in float keeps that
 * guarantee at zero risk instead of threading a precision template through
 * code other correctness guarantees depend on.
 */
class cell_list32 {
public:
    cell_list32(int n_atoms,
                const float *x,
                const float *y,
                const float *z,
                float cell_size)
        : cell_size_(cell_size)
    {
        for (int atom = 0; atom < n_atoms; ++atom) {
            cells_[key(x[atom], y[atom], z[atom])].push_back(atom);
        }
    }

    void collect(float x,
                 float y,
                 float z,
                 std::vector<int> &atoms) const
    {
        const cell_key center = key(x, y, z);

        atoms.clear();
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const cell_key neighbor = {center.x + dx, center.y + dy, center.z + dz};
                    const auto found = cells_.find(neighbor);

                    if (found == cells_.end()) continue;
                    atoms.insert(atoms.end(), found->second.begin(), found->second.end());
                }
            }
        }
    }

private:
    cell_key key(float x,
                float y,
                float z) const
    {
        const cell_key result = {
            static_cast<int>(std::floor(x / cell_size_)),
            static_cast<int>(std::floor(y / cell_size_)),
            static_cast<int>(std::floor(z / cell_size_))
        };
        return result;
    }

    float cell_size_;
    std::unordered_map<cell_key, std::vector<int>, cell_hash> cells_;
};

static int
cpu_shrake_rupley_impl_fp32(int n_atoms,
                            int n_points,
                            const double *x,
                            const double *y,
                            const double *z,
                            const double *expanded_radii,
                            const double *test_points,
                            int n_threads,
                            double *sasa)
{
    std::vector<std::thread> threads;
    std::atomic<int> worker_status(FASTSASA_SUCCESS);
    int count;

    if (n_atoms <= 0 || n_points <= 0 || x == nullptr || y == nullptr ||
        z == nullptr || expanded_radii == nullptr || test_points == nullptr ||
        sasa == nullptr) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!finite_geometry(n_atoms, x, y, z, expanded_radii) ||
        !finite_test_points(n_points, test_points)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    /* Bucketing and the max-radius scan stay in double: they only run once
     * (not per point) and cell_keys_fit's overflow check is about the
     * double coordinate range, not compute precision. */
    const double max_radius = maximum_radius(expanded_radii, n_atoms);
    if (max_radius <= 0.0 || !cell_keys_fit(n_atoms, x, y, z, max_radius)) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    /* The FP32 coordinates are box-local, matching the CUDA and Vulkan FP32
     * shadows: subtracting the per-axis minimum in double before the float
     * conversion keeps the float magnitudes at the size of the system, so a
     * large absolute offset (a structure far from the origin, or a
     * translated trajectory) does not eat the float mantissa. Only
     * differences are ever consumed below, so the shift cancels exactly. */
    double min_x = x[0];
    double min_y = y[0];
    double min_z = z[0];
    for (int atom = 1; atom < n_atoms; ++atom) {
        min_x = std::min(min_x, x[atom]);
        min_y = std::min(min_y, y[atom]);
        min_z = std::min(min_z, z[atom]);
    }
    std::vector<float> xf(static_cast<size_t>(n_atoms));
    std::vector<float> yf(static_cast<size_t>(n_atoms));
    std::vector<float> zf(static_cast<size_t>(n_atoms));
    std::vector<float> radius_f(static_cast<size_t>(n_atoms));
    std::vector<float> radius2_f(static_cast<size_t>(n_atoms));
    for (int atom = 0; atom < n_atoms; ++atom) {
        xf[static_cast<size_t>(atom)] = static_cast<float>(x[atom] - min_x);
        yf[static_cast<size_t>(atom)] = static_cast<float>(y[atom] - min_y);
        zf[static_cast<size_t>(atom)] = static_cast<float>(z[atom] - min_z);
        radius_f[static_cast<size_t>(atom)] = static_cast<float>(expanded_radii[atom]);
        radius2_f[static_cast<size_t>(atom)] = radius_f[static_cast<size_t>(atom)] *
                                               radius_f[static_cast<size_t>(atom)];
    }
    std::vector<float> points_f(static_cast<size_t>(n_points) * 3u);
    for (int i = 0; i < 3 * n_points; ++i) points_f[static_cast<size_t>(i)] = static_cast<float>(test_points[i]);

    count = thread_count(n_threads, n_atoms);
    threads.reserve(static_cast<size_t>(count));
    const cell_list32 cells(n_atoms, xf.data(), yf.data(), zf.data(), 2.0f * static_cast<float>(max_radius));

    try {
        for (int thread_id = 0; thread_id < count; ++thread_id) {
            const int begin = static_cast<int>(static_cast<size_t>(thread_id) *
                                               static_cast<size_t>(n_atoms) /
                                               static_cast<size_t>(count));
            const int end = static_cast<int>(static_cast<size_t>(thread_id + 1) *
                                             static_cast<size_t>(n_atoms) /
                                             static_cast<size_t>(count));

            threads.emplace_back([=, &cells, &worker_status,
                                  &xf, &yf, &zf, &radius_f, &radius2_f, &points_f]() {
                try {
                    std::vector<int> candidates;
                    std::vector<int> neighbors;
                    candidates.reserve(128u);
                    neighbors.reserve(64u);

                    for (int atom = begin; atom < end; ++atom) {
                        const float radius = radius_f[static_cast<size_t>(atom)];
                        int accessible = 0;

                        cells.collect(xf[static_cast<size_t>(atom)], yf[static_cast<size_t>(atom)],
                                     zf[static_cast<size_t>(atom)], candidates);
                        neighbors.clear();
                        for (int candidate : candidates) {
                            const float neighbor_radius = radius_f[static_cast<size_t>(candidate)];
                            const float max_distance = radius + neighbor_radius;
                            const float dx = xf[static_cast<size_t>(atom)] - xf[static_cast<size_t>(candidate)];
                            const float dy = yf[static_cast<size_t>(atom)] - yf[static_cast<size_t>(candidate)];
                            const float dz = zf[static_cast<size_t>(atom)] - zf[static_cast<size_t>(candidate)];

                            if (candidate != atom &&
                                dx * dx + dy * dy + dz * dz < max_distance * max_distance) {
                                neighbors.push_back(candidate);
                            }
                        }
                        for (int point = 0; point < n_points; ++point) {
                            const float px = xf[static_cast<size_t>(atom)] + radius * points_f[3 * point];
                            const float py = yf[static_cast<size_t>(atom)] + radius * points_f[3 * point + 1];
                            const float pz = zf[static_cast<size_t>(atom)] + radius * points_f[3 * point + 2];
                            int buried = 0;

                            for (int other : neighbors) {
                                const float dx = px - xf[static_cast<size_t>(other)];
                                const float dy = py - yf[static_cast<size_t>(other)];
                                const float dz = pz - zf[static_cast<size_t>(other)];

                                if (dx * dx + dy * dy + dz * dz < radius2_f[static_cast<size_t>(other)]) {
                                    buried = 1;
                                    break;
                                }
                            }
                            if (!buried) ++accessible;
                        }
                        /* Only the point tests run in float. The area is the
                         * exact exposed count scaled in FP64 with the double
                         * radius, in the same operation order as the CUDA
                         * sr_atom_area and Vulkan sr_count_to_area formulas,
                         * so CPU FP32 matches the Vulkan FP32 readback exactly
                         * whenever the per-atom counts agree (CUDA FP32 can
                         * still differ at the 1e-6 A^2 level per atom from its
                         * own device-side rounding). */
                        const double expanded = expanded_radii[atom];
                        sasa[atom] = 4.0 * FASTSASA_CPU_PI * expanded * expanded *
                                     static_cast<double>(accessible) / static_cast<double>(n_points);
                    }
                } catch (const std::bad_alloc &) {
                    worker_status.store(FASTSASA_MEMORY_ERROR);
                } catch (...) {
                    worker_status.store(FASTSASA_INVALID_ARGUMENT);
                }
            });
        }
    } catch (...) {
        join_threads(&threads);
        throw;
    }
    join_threads(&threads);
    return worker_status.load();
}

int
fastsasa_cpu_shrake_rupley_precision(int n_atoms,
                                    int n_points,
                                    const double *x,
                                    const double *y,
                                    const double *z,
                                    const double *expanded_radii,
                                    const double *test_points,
                                    int n_threads,
                                    int precision,
                                    double *sasa)
{
    if (precision != FASTSASA_PRECISION_FP64 && precision != FASTSASA_PRECISION_FP32) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    try {
        if (precision == FASTSASA_PRECISION_FP32) {
            return cpu_shrake_rupley_impl_fp32(n_atoms, n_points, x, y, z,
                                               expanded_radii, test_points, n_threads, sasa);
        }
        return cpu_shrake_rupley_impl(n_atoms, n_points, x, y, z,
                                      expanded_radii, test_points, n_threads, sasa);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

static int
cpu_exposed_points_impl(int n_atoms,
                        int n_points,
                        const double *x,
                        const double *y,
                        const double *z,
                        const double *expanded_radii,
                        const double *test_points,
                        int n_threads,
                        unsigned char *exposed)
{
    std::vector<std::thread> threads;
    std::vector<double> radius2;
    std::atomic<int> worker_status(FASTSASA_SUCCESS);
    int count;

    if (n_atoms <= 0 || n_points <= 0 || x == nullptr || y == nullptr ||
        z == nullptr || expanded_radii == nullptr || test_points == nullptr ||
        exposed == nullptr) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!finite_geometry(n_atoms, x, y, z, expanded_radii) ||
        !finite_test_points(n_points, test_points)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    const double max_radius = maximum_radius(expanded_radii, n_atoms);
    if (max_radius <= 0.0 || !cell_keys_fit(n_atoms, x, y, z, max_radius)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    radius2.resize(static_cast<size_t>(n_atoms));
    for (int atom = 0; atom < n_atoms; ++atom) {
        radius2[static_cast<size_t>(atom)] = expanded_radii[atom] * expanded_radii[atom];
    }
    count = thread_count(n_threads, n_atoms);
    threads.reserve(static_cast<size_t>(count));
    const cell_list cells(n_atoms, x, y, z, 2.0 * max_radius);
    try {
        for (int thread_id = 0; thread_id < count; ++thread_id) {
            const int begin = static_cast<int>(static_cast<size_t>(thread_id) *
                                               static_cast<size_t>(n_atoms) /
                                               static_cast<size_t>(count));
            const int end = static_cast<int>(static_cast<size_t>(thread_id + 1) *
                                             static_cast<size_t>(n_atoms) /
                                             static_cast<size_t>(count));

            threads.emplace_back([=, &cells, &radius2, &worker_status]() {
                try {
                    for (int atom = begin; atom < end; ++atom) {
                        const double radius = expanded_radii[atom];
                        unsigned char *row = exposed +
                            static_cast<size_t>(atom) * static_cast<size_t>(n_points);

                        for (int point = 0; point < n_points; ++point) {
                            const double px = x[atom] + radius * test_points[3 * point];
                            const double py = y[atom] + radius * test_points[3 * point + 1];
                            const double pz = z[atom] + radius * test_points[3 * point + 2];
                            int buried = 0;

                            cells.visit(px, py, pz, [&](int other) {
                                if (buried) return;
                                const double dx = px - x[other];
                                const double dy = py - y[other];
                                const double dz = pz - z[other];

                                if (other != atom &&
                                    dx * dx + dy * dy + dz * dz <
                                        radius2[static_cast<size_t>(other)]) {
                                    buried = 1;
                                }
                            });
                            row[point] = buried ? 0u : 1u;
                        }
                    }
                } catch (const std::bad_alloc &) {
                    worker_status.store(FASTSASA_MEMORY_ERROR);
                } catch (...) {
                    worker_status.store(FASTSASA_INVALID_ARGUMENT);
                }
            });
        }
    } catch (...) {
        join_threads(&threads);
        throw;
    }
    join_threads(&threads);
    return worker_status.load();
}

int
fastsasa_cpu_exposed_points(int n_atoms,
                          int n_points,
                          const double *x,
                          const double *y,
                          const double *z,
                          const double *expanded_radii,
                          const double *test_points,
                          int n_threads,
                          unsigned char *exposed)
{
    try {
        return cpu_exposed_points_impl(n_atoms, n_points, x, y, z,
                                       expanded_radii, test_points, n_threads,
                                       exposed);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

static double
exposed_arc_length(std::vector<std::pair<double, double> > *arcs)
{
    const double two_pi = 2.0 * FASTSASA_CPU_PI;
    double exposed;
    double covered_until;

    if (arcs->empty()) return two_pi;
    std::sort(arcs->begin(), arcs->end());
    exposed = (*arcs)[0].first;
    covered_until = (*arcs)[0].second;
    for (size_t i = 1; i < arcs->size(); ++i) {
        if (covered_until < (*arcs)[i].first) exposed += (*arcs)[i].first - covered_until;
        if ((*arcs)[i].second > covered_until) covered_until = (*arcs)[i].second;
    }
    return exposed + two_pi - covered_until;
}

/*
 * Per-atom Lee-Richards area. This is the arithmetic reference that the
 * CUDA FP64 path reproduces bit for bit: acos/atan2 come from
 * fastsasa_exact_math.h (the same code runs on the device) and the host
 * build disables FMA contraction. `visit(f)` calls f(other) for every
 * candidate neighbour; the arcs are sorted before use, so the visiting
 * order does not affect the result.
 */
template <typename Visitor>
static double
lee_richards_atom_area_visit(int n_slices,
                             const double *x,
                             const double *y,
                             const double *z,
                             const double *expanded_radii,
                             int atom,
                             Visitor visit)
{
    const double two_pi = 2.0 * FASTSASA_CPU_PI;
    const double xi = x[atom];
    const double yi = y[atom];
    const double zi = z[atom];
    const double ri = expanded_radii[atom];
    const double delta = 2.0 * ri / n_slices;
    std::vector<std::pair<double, double> > arcs;
    double area = 0.0;

    for (int slice = 0; slice < n_slices; ++slice) {
        const double slice_z = zi - ri + delta * (slice + 0.5);
        const double di = std::fabs(zi - slice_z);
        const double ri_prime2 = ri * ri - di * di;
        int buried = 0;

        if (ri_prime2 <= 0.0) continue;
        const double ri_prime = std::sqrt(ri_prime2);
        arcs.clear();
        visit([&](int other) {
            if (buried) return;
            const double rj = expanded_radii[other];
            const double dj = std::fabs(z[other] - slice_z);

            if (other == atom || dj >= rj) return;
            const double rj_prime2 = rj * rj - dj * dj;
            if (rj_prime2 <= 0.0) return;
            const double rj_prime = std::sqrt(rj_prime2);
            const double dx = x[other] - xi;
            const double dy = y[other] - yi;
            const double dij = std::sqrt(dx * dx + dy * dy);

            if (dij >= ri_prime + rj_prime) return;
            if (dij < 1e-14) {
                if (rj_prime >= ri_prime) {
                    buried = 1;
                }
                return;
            }
            if (dij + ri_prime < rj_prime) {
                buried = 1;
                return;
            }
            if (dij + rj_prime < ri_prime) return;

            const double argument = (ri_prime2 + dij * dij - rj_prime2) /
                                    (2.0 * ri_prime * dij);
            const double alpha = fastsasa_exact_acos(std::max(-1.0, std::min(1.0, argument)));
            const double beta = fastsasa_exact_atan2(dy, dx) + FASTSASA_CPU_PI;
            double lower = beta - alpha;
            double upper = beta + alpha;

            if (lower < 0.0) lower += two_pi;
            if (upper > two_pi) upper -= two_pi;
            if (upper < lower) {
                arcs.emplace_back(0.0, upper);
                arcs.emplace_back(lower, two_pi);
            } else {
                arcs.emplace_back(lower, upper);
            }
        });
        if (!buried) area += delta * ri * exposed_arc_length(&arcs);
    }
    return area;
}

static double
lee_richards_atom_area(int n_atoms,
                       int n_slices,
                       const double *x,
                       const double *y,
                       const double *z,
                       const double *expanded_radii,
                       const cell_list &cells,
                       int atom)
{
    (void)n_atoms;
    return lee_richards_atom_area_visit(
        n_slices, x, y, z, expanded_radii, atom,
        [&](auto &&callback) { cells.visit(x[atom], y[atom], z[atom], callback); });
}

static int
cpu_lee_richards_impl(int n_atoms,
                      int n_slices,
                      const double *x,
                      const double *y,
                      const double *z,
                      const double *expanded_radii,
                      int n_threads,
                      double *sasa)
{
    std::vector<std::thread> threads;
    std::atomic<int> worker_status(FASTSASA_SUCCESS);
    int count;

    if (n_atoms <= 0 || n_slices <= 0 || x == nullptr || y == nullptr ||
        z == nullptr || expanded_radii == nullptr || sasa == nullptr) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!finite_geometry(n_atoms, x, y, z, expanded_radii)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    const double max_radius = maximum_radius(expanded_radii, n_atoms);
    if (max_radius <= 0.0 || !cell_keys_fit(n_atoms, x, y, z, max_radius)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    const cell_list cells(n_atoms, x, y, z, 2.0 * max_radius);
    count = thread_count(n_threads, n_atoms);
    threads.reserve(static_cast<size_t>(count));
    try {
        for (int thread_id = 0; thread_id < count; ++thread_id) {
            const int begin = static_cast<int>(static_cast<size_t>(thread_id) *
                                               static_cast<size_t>(n_atoms) /
                                               static_cast<size_t>(count));
            const int end = static_cast<int>(static_cast<size_t>(thread_id + 1) *
                                             static_cast<size_t>(n_atoms) /
                                             static_cast<size_t>(count));

            threads.emplace_back([=, &cells, &worker_status]() {
                try {
            for (int atom = begin; atom < end; ++atom) {
                sasa[atom] = lee_richards_atom_area(n_atoms,
                                                    n_slices,
                                                    x,
                                                    y,
                                                    z,
                                                    expanded_radii,
                                                    cells,
                                                    atom);
            }
                } catch (const std::bad_alloc &) {
                    worker_status.store(FASTSASA_MEMORY_ERROR);
                } catch (...) {
                    worker_status.store(FASTSASA_INVALID_ARGUMENT);
                }
            });
        }
    } catch (...) {
        join_threads(&threads);
        throw;
    }
    join_threads(&threads);
    return worker_status.load();
}

int
fastsasa_cpu_lee_richards_atom(int n_atoms,
                             int n_slices,
                             const double *x,
                             const double *y,
                             const double *z,
                             const double *expanded_radii,
                             int atom,
                             double *area)
{
    if (n_atoms <= 0 || n_slices <= 0 || x == nullptr || y == nullptr ||
        z == nullptr || expanded_radii == nullptr || area == nullptr ||
        atom < 0 || atom >= n_atoms) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    try {
        /* Brute-force neighbour scan: identical result to the cell-list
         * path because arcs are sorted before use and non-neighbours yield
         * no arcs. */
        *area = lee_richards_atom_area_visit(
            n_slices, x, y, z, expanded_radii, atom,
            [&](auto &&callback) {
                for (int other = 0; other < n_atoms; ++other) callback(other);
            });
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    return FASTSASA_SUCCESS;
}

int
fastsasa_cpu_lee_richards(int n_atoms,
                        int n_slices,
                        const double *x,
                        const double *y,
                        const double *z,
                        const double *expanded_radii,
                        int n_threads,
                        double *sasa)
{
    try {
        return cpu_lee_richards_impl(n_atoms, n_slices, x, y, z,
                                     expanded_radii, n_threads, sasa);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

static int
cpu_trajectory_parameters(const fastsasa_parameters *parameters,
                          double *probe_radius,
                          int *resolution,
                          int *algorithm,
                          int *precision)
{
    *probe_radius = parameters != nullptr ? parameters->probe_radius : 1.4;
    *resolution = parameters != nullptr ? parameters->n_points : 100;
    *algorithm = parameters != nullptr ? parameters->algorithm : FASTSASA_ALGORITHM_SHRAKE_RUPLEY;
    *precision = parameters != nullptr ? parameters->precision : FASTSASA_PRECISION_FP64;
    return *probe_radius >= 0.0 &&
           *resolution > 0 &&
           (*algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY ||
            *algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) &&
           (*precision == FASTSASA_PRECISION_FP64 || *precision == FASTSASA_PRECISION_FP32);
}

static std::vector<double>
cpu_expanded_radii(const double *radii,
                   int n_atoms,
                   double probe_radius)
{
    std::vector<double> expanded(static_cast<size_t>(n_atoms));

    for (int atom = 0; atom < n_atoms; ++atom) {
        expanded[static_cast<size_t>(atom)] = radii[atom] + probe_radius;
    }
    return expanded;
}

static std::vector<double>
cpu_test_points(int n_points)
{
    const double dlong = FASTSASA_CPU_PI * (3.0 - std::sqrt(5.0));
    const double dz = 2.0 / static_cast<double>(n_points);
    double longitude = 0.0;
    double z = 1.0 - dz / 2.0;
    std::vector<double> points(3u * static_cast<size_t>(n_points));

    for (int point = 0; point < n_points; ++point) {
        const double r = std::sqrt(1.0 - z * z);

        points[static_cast<size_t>(3 * point)] = std::cos(longitude) * r;
        points[static_cast<size_t>(3 * point + 1)] = std::sin(longitude) * r;
        points[static_cast<size_t>(3 * point + 2)] = z;
        z -= dz;
        longitude += dlong;
    }
    return points;
}

static double
cpu_sum_atoms(const double *atom_sasa,
              int n_atoms)
{
    double sum = 0.0;
    double compensation = 0.0;

    for (int atom = 0; atom < n_atoms; ++atom) {
        const double corrected = atom_sasa[atom] - compensation;
        const double next = sum + corrected;

        compensation = (next - sum) - corrected;
        sum = next;
    }
    return sum;
}

static int
cpu_sum_residues(const double *atom_sasa,
                 const int *residue_ids,
                 int n_atoms,
                 int n_residues,
                 double *residue_sasa)
{
    std::vector<double> compensation(static_cast<size_t>(n_residues), 0.0);

    if (residue_ids == nullptr || residue_sasa == nullptr || n_residues <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    std::fill(residue_sasa, residue_sasa + n_residues, 0.0);
    for (int atom = 0; atom < n_atoms; ++atom) {
        const int residue = residue_ids[atom];

        if (residue >= 0 && residue < n_residues) {
            const double corrected = atom_sasa[atom] - compensation[static_cast<size_t>(residue)];
            const double next = residue_sasa[residue] + corrected;

            compensation[static_cast<size_t>(residue)] = (next - residue_sasa[residue]) - corrected;
            residue_sasa[residue] = next;
        }
    }
    return FASTSASA_SUCCESS;
}

static int
cpu_sum_selections(const double *atom_sasa,
                   const unsigned int *selection_masks,
                   int n_atoms,
                   int n_selections,
                   double *selection_sasa)
{
    std::vector<double> compensation(static_cast<size_t>(n_selections), 0.0);

    if (selection_masks == nullptr || selection_sasa == nullptr ||
        n_selections <= 0 || n_selections > 31) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    std::fill(selection_sasa, selection_sasa + n_selections, 0.0);
    for (int atom = 0; atom < n_atoms; ++atom) {
        for (int selection = 0; selection < n_selections; ++selection) {
            if ((selection_masks[atom] & (1u << selection)) != 0u) {
                const size_t index = static_cast<size_t>(selection);
                const double corrected = atom_sasa[atom] - compensation[index];
                const double next = selection_sasa[selection] + corrected;

                compensation[index] = (next - selection_sasa[selection]) - corrected;
                selection_sasa[selection] = next;
            }
        }
    }
    return FASTSASA_SUCCESS;
}

static int
validate_cpu_trajectory_input(const fastsasa_topology *topology,
                              const fastsasa_soa_frames *frames,
                              const fastsasa_parameters *parameters,
                              double *probe_radius,
                              int *resolution,
                              int *algorithm,
                              int *precision)
{
    if (topology == nullptr || frames == nullptr) return FASTSASA_INVALID_ARGUMENT;
    if (topology->radii == nullptr || topology->n_atoms <= 0) return FASTSASA_INVALID_ARGUMENT;
    if (frames->x == nullptr || frames->y == nullptr || frames->z == nullptr ||
        frames->n_frames <= 0) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    if (!cpu_trajectory_parameters(parameters, probe_radius, resolution, algorithm, precision)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    return FASTSASA_SUCCESS;
}

static int
cpu_calc_trajectory_soa_impl(const fastsasa_topology *topology,
                             const fastsasa_soa_frames *frames,
                             const fastsasa_parameters *parameters,
                             int n_threads,
                             double *total_sasa,
                             double *atom_sasa_frames,
                             double *residue_sasa_frames)
{
    double probe_radius;
    int resolution;
    int algorithm;
    int precision;
    int status = validate_cpu_trajectory_input(topology, frames, parameters, &probe_radius, &resolution, &algorithm, &precision);

    if (status != FASTSASA_SUCCESS) return status;
    if (total_sasa == nullptr) return FASTSASA_INVALID_ARGUMENT;
    if (atom_sasa_frames != nullptr && residue_sasa_frames != nullptr) return FASTSASA_INVALID_ARGUMENT;
    if (residue_sasa_frames != nullptr &&
        (topology->residue_ids == nullptr || topology->n_residues <= 0)) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    const int total_threads = normalized_thread_count(n_threads);
    const int frame_threads = cpu_frame_thread_count(total_threads, frames->n_frames, topology->n_atoms);
    const int atom_threads = std::max(1, total_threads / frame_threads);
    const std::vector<double> expanded = cpu_expanded_radii(topology->radii, topology->n_atoms, probe_radius);
    const std::vector<double> points = algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY
                                           ? cpu_test_points(resolution)
                                           : std::vector<double>();
    std::atomic<int> next_frame(0);
    std::vector<int> statuses(static_cast<size_t>(frame_threads), FASTSASA_SUCCESS);
    std::vector<std::thread> workers;

    try {
        workers.reserve(static_cast<size_t>(frame_threads));
        for (int worker = 0; worker < frame_threads; ++worker) {
            workers.emplace_back([&, worker]() {
            try {
                std::vector<double> scratch(static_cast<size_t>(topology->n_atoms));

                for (;;) {
                    const int frame = next_frame.fetch_add(1);
                    if (frame >= frames->n_frames) break;

                    const double *x = frames->x + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    const double *y = frames->y + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    const double *z = frames->z + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    double *frame_sasa = atom_sasa_frames != nullptr
                                             ? atom_sasa_frames + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms)
                                             : scratch.data();
                    int frame_status;

                    if (algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) {
                        frame_status = fastsasa_cpu_lee_richards(topology->n_atoms,
                                                               resolution,
                                                               x,
                                                               y,
                                                               z,
                                                               expanded.data(),
                                                               atom_threads,
                                                               frame_sasa);
                    } else {
                        frame_status = fastsasa_cpu_shrake_rupley_precision(topology->n_atoms,
                                                                resolution,
                                                                x,
                                                                y,
                                                                z,
                                                                expanded.data(),
                                                                points.data(),
                                                                atom_threads,
                                                                precision,
                                                                frame_sasa);
                    }
                    if (frame_status != FASTSASA_SUCCESS) {
                        statuses[static_cast<size_t>(worker)] = frame_status;
                        break;
                    }
                    total_sasa[frame] = cpu_sum_atoms(frame_sasa, topology->n_atoms);
                    if (residue_sasa_frames != nullptr) {
                        frame_status = cpu_sum_residues(frame_sasa,
                                                        topology->residue_ids,
                                                        topology->n_atoms,
                                                        topology->n_residues,
                                                        residue_sasa_frames + static_cast<size_t>(frame) *
                                                        static_cast<size_t>(topology->n_residues));
                        if (frame_status != FASTSASA_SUCCESS) {
                            statuses[static_cast<size_t>(worker)] = frame_status;
                            break;
                        }
                    }
                }
            } catch (...) {
                statuses[static_cast<size_t>(worker)] = FASTSASA_MEMORY_ERROR;
            }
            });
        }
    } catch (...) {
        join_threads(&workers);
        throw;
    }
    for (std::thread &worker : workers) worker.join();
    for (int worker_status : statuses) {
        if (worker_status != FASTSASA_SUCCESS) return worker_status;
    }
    return FASTSASA_SUCCESS;
}

static int
cpu_calc_trajectory_soa_selection_impl(const fastsasa_topology *topology,
                                       const fastsasa_soa_frames *frames,
                                       const unsigned int *selection_masks,
                                       int n_selections,
                                       const fastsasa_parameters *parameters,
                                       int n_threads,
                                       double *total_sasa,
                                       double *selection_sasa_frames)
{
    double probe_radius;
    int resolution;
    int algorithm;
    int precision;
    int status = validate_cpu_trajectory_input(topology, frames, parameters, &probe_radius, &resolution, &algorithm, &precision);

    if (status != FASTSASA_SUCCESS) return status;
    if (selection_masks == nullptr || selection_sasa_frames == nullptr ||
        n_selections <= 0 || n_selections > 31) {
        return FASTSASA_INVALID_ARGUMENT;
    }

    const int total_threads = normalized_thread_count(n_threads);
    const int frame_threads = cpu_frame_thread_count(total_threads, frames->n_frames, topology->n_atoms);
    const int atom_threads = std::max(1, total_threads / frame_threads);
    const std::vector<double> expanded = cpu_expanded_radii(topology->radii, topology->n_atoms, probe_radius);
    const std::vector<double> points = algorithm == FASTSASA_ALGORITHM_SHRAKE_RUPLEY
                                           ? cpu_test_points(resolution)
                                           : std::vector<double>();
    std::atomic<int> next_frame(0);
    std::vector<int> statuses(static_cast<size_t>(frame_threads), FASTSASA_SUCCESS);
    std::vector<std::thread> workers;

    try {
        workers.reserve(static_cast<size_t>(frame_threads));
        for (int worker = 0; worker < frame_threads; ++worker) {
            workers.emplace_back([&, worker]() {
            try {
                std::vector<double> scratch(static_cast<size_t>(topology->n_atoms));

                for (;;) {
                    const int frame = next_frame.fetch_add(1);
                    if (frame >= frames->n_frames) break;

                    const double *x = frames->x + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    const double *y = frames->y + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    const double *z = frames->z + static_cast<size_t>(frame) * static_cast<size_t>(topology->n_atoms);
                    double *frame_selection = selection_sasa_frames +
                                              static_cast<size_t>(frame) * static_cast<size_t>(n_selections);
                    int frame_status;

                    if (algorithm == FASTSASA_ALGORITHM_LEE_RICHARDS) {
                        frame_status = fastsasa_cpu_lee_richards(topology->n_atoms,
                                                               resolution,
                                                               x,
                                                               y,
                                                               z,
                                                               expanded.data(),
                                                               atom_threads,
                                                               scratch.data());
                    } else {
                        frame_status = fastsasa_cpu_shrake_rupley_precision(topology->n_atoms,
                                                                resolution,
                                                                x,
                                                                y,
                                                                z,
                                                                expanded.data(),
                                                                points.data(),
                                                                atom_threads,
                                                                precision,
                                                                scratch.data());
                    }
                    if (frame_status != FASTSASA_SUCCESS) {
                        statuses[static_cast<size_t>(worker)] = frame_status;
                        break;
                    }
                    frame_status = cpu_sum_selections(scratch.data(),
                                                      selection_masks,
                                                      topology->n_atoms,
                                                      n_selections,
                                                      frame_selection);
                    if (frame_status != FASTSASA_SUCCESS) {
                        statuses[static_cast<size_t>(worker)] = frame_status;
                        break;
                    }
                    if (total_sasa != nullptr) {
                        total_sasa[frame] = cpu_sum_atoms(scratch.data(), topology->n_atoms);
                    }
                }
            } catch (...) {
                statuses[static_cast<size_t>(worker)] = FASTSASA_MEMORY_ERROR;
            }
            });
        }
    } catch (...) {
        join_threads(&workers);
        throw;
    }
    for (std::thread &worker : workers) worker.join();
    for (int worker_status : statuses) {
        if (worker_status != FASTSASA_SUCCESS) return worker_status;
    }
    return FASTSASA_SUCCESS;
}

int
fastsasa_cpu_calc_trajectory_soa(const fastsasa_topology *topology,
                               const fastsasa_soa_frames *frames,
                               const fastsasa_parameters *parameters,
                               int n_threads,
                               double *total_sasa,
                               double *atom_sasa_frames,
                               double *residue_sasa_frames)
{
    try {
        return cpu_calc_trajectory_soa_impl(topology,
                                            frames,
                                            parameters,
                                            n_threads,
                                            total_sasa,
                                            atom_sasa_frames,
                                            residue_sasa_frames);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

int
fastsasa_cpu_calc_trajectory_soa_selection(const fastsasa_topology *topology,
                                         const fastsasa_soa_frames *frames,
                                         const unsigned int *selection_masks,
                                         int n_selections,
                                         const fastsasa_parameters *parameters,
                                         int n_threads,
                                         double *total_sasa,
                                         double *selection_sasa_frames)
{
    try {
        return cpu_calc_trajectory_soa_selection_impl(topology,
                                                      frames,
                                                      selection_masks,
                                                      n_selections,
                                                      parameters,
                                                      n_threads,
                                                      total_sasa,
                                                      selection_sasa_frames);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}
