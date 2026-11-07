/*
 * Analytic (exact) SASA on the CPU: the per-atom routine of
 * fastsasa_exact_core.h over atoms in grid order, threaded over atoms in
 * dynamic chunks. Atoms that overflow the small per-thread workspace are
 * recomputed with the large one; the result never depends on the thread
 * count or the split.
 */
#include "fastsasa_cpu.h"
#include "fastsasa_cpu_pool.h"
#include "fastsasa_exact_host.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <new>
#include <vector>

namespace {

typedef fastsasa_exact::Work<fastsasa_exact::SmallCaps> SmallWork;
typedef fastsasa_exact::Work<fastsasa_exact::LargeCaps> LargeWork;

int
default_thread_count(int requested)
{
    const int count = requested > 0 ? requested : fastsasa_cpu_default_threads();
    return count > 0 ? count : 1;
}

/* Areas of all atoms (centers == nullptr: out[i] for atom i) or of the
 * listed ones (out[c] for centers[c]). */
int
cpu_exact_impl(int n_atoms, const double *x, const double *y, const double *z, const double *radii, int n_threads,
               const int *centers, int n_centers, double *out)
{
    if (n_atoms < 0 || (n_atoms > 0 && (x == nullptr || y == nullptr || z == nullptr || radii == nullptr)) ||
        (centers != nullptr && n_centers < 0)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    const int n_work = centers != nullptr ? n_centers : n_atoms;
    if (n_work == 0) return FASTSASA_SUCCESS;
    if (out == nullptr) return FASTSASA_INVALID_ARGUMENT;
    fastsasa_exact::SortedFrame f;
    if (!fastsasa_exact::sort_frame(n_atoms, x, y, z, radii, f)) return FASTSASA_INVALID_ARGUMENT;
    /* work item w: sorted atom q (all atoms, in grid order) or the w-th centre */
    std::vector<int> work_atom;
    if (centers != nullptr) {
        std::vector<int> sorted_of(static_cast<size_t>(n_atoms));
        for (int q = 0; q < n_atoms; ++q) sorted_of[static_cast<size_t>(f.orig[static_cast<size_t>(q)])] = q;
        work_atom.resize(static_cast<size_t>(n_centers));
        for (int c = 0; c < n_centers; ++c) {
            if (centers[c] < 0 || centers[c] >= n_atoms) return FASTSASA_INVALID_ARGUMENT;
            work_atom[static_cast<size_t>(c)] = sorted_of[static_cast<size_t>(centers[c])];
        }
    }
    const fastsasa_exact::GridView g = f.g.view();
    double est[3 * fastsasa_exact::kEstimatePoints];
    fastsasa_exact::estimate_points(est);

    const int count = std::min(default_thread_count(n_threads), std::max(1, n_work / 64));
    std::vector<std::unique_ptr<SmallWork> > work(static_cast<size_t>(count));
    for (int t = 0; t < count; ++t) work[static_cast<size_t>(t)].reset(new SmallWork());
    std::vector<int> spill;   /* work items that overflowed the small workspace */
    std::mutex spill_lock;
    const double *xs = f.xs.data(), *Rs = f.Rs.data();
    const auto atom_of = [&](int w) { return centers != nullptr ? work_atom[static_cast<size_t>(w)] : w; };
    const auto out_of = [&](int w) -> double & {
        return centers != nullptr ? out[w] : out[f.orig[static_cast<size_t>(w)]];
    };

    fastsasa_cpu::WorkerPool::run_chunked(count, n_work, [&](int begin, int end, int task) {
        SmallWork &W = *work[static_cast<size_t>(task)];
        for (int w = begin; w < end; ++w) {
            const int q = atom_of(w);
            int retries = 0, overflow = 0;
            out_of(w) = Rs[q] > 0.0 ? fastsasa_exact::atom_area(q, xs, Rs, g, est, W, &retries, &overflow) : 0.0;
            if (overflow) {
                std::lock_guard<std::mutex> hold(spill_lock);
                spill.push_back(w);
            }
        }
    });
    if (!spill.empty()) {
        std::unique_ptr<LargeWork> big(new LargeWork());
        for (size_t s = 0; s < spill.size(); ++s) {
            int retries = 0, overflow = 0;
            out_of(spill[s]) = fastsasa_exact::atom_area(atom_of(spill[s]), xs, Rs, g, est, *big, &retries, &overflow);
        }
    }
    for (int w = 0; w < n_work; ++w) {
        if (std::isnan(out_of(w))) return FASTSASA_DEGENERATE_GEOMETRY;
    }
    return FASTSASA_SUCCESS;
}

}  // namespace

extern "C" int
fastsasa_cpu_exact(int n_atoms, const double *x, const double *y, const double *z, const double *expanded_radii,
                   int n_threads, double *sasa)
{
    try {
        return cpu_exact_impl(n_atoms, x, y, z, expanded_radii, n_threads, nullptr, 0, sasa);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}

extern "C" int
fastsasa_cpu_exact_centers(int n_atoms, const double *x, const double *y, const double *z,
                           const double *expanded_radii, int n_threads, const int *centers, int n_centers,
                           double *areas)
{
    try {
        if (n_centers == 0) return FASTSASA_SUCCESS;
        if (centers == nullptr) return FASTSASA_INVALID_ARGUMENT;
        return cpu_exact_impl(n_atoms, x, y, z, expanded_radii, n_threads, centers, n_centers, areas);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}
