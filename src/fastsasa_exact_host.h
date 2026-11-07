/*
 * Host-side frame preparation for the analytic SASA kernels
 * (fastsasa_exact_core.h): atoms sorted into grid order, cell starts and the
 * spherical neighbour stencil. Shared by the CPU backend and the GPU paths.
 */
#ifndef FASTSASA_EXACT_HOST_H
#define FASTSASA_EXACT_HOST_H

#include "fastsasa_exact_core.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace fastsasa_exact {

/* Cell starts over atoms in grid order (cell = rmax, so the 2 rmax reach
 * spans two cells), and per (dz, dy) offset the largest |dx| whose cells can
 * hold a neighbour. */
struct Grid {
    double lo[3];
    double cell;
    int dims[3];
    int K;
    std::vector<int> start;
    std::vector<int> dxmax;

    GridView view() const
    {
        GridView g;
        for (int a = 0; a < 3; ++a) {
            g.lo[a] = lo[a];
            g.dims[a] = dims[a];
        }
        g.cell = cell;
        g.start = start.data();
        g.K = K;
        g.dxmax = dxmax.data();
        return g;
    }
};

/* One frame in grid order: xs (x, y, z interleaved) and Rs sorted,
 * orig[q] = input index of sorted atom q. */
struct SortedFrame {
    std::vector<double> xs, Rs;
    std::vector<int> orig;
    std::vector<int> cell_of;
    Grid g;
};

inline void
estimate_points(double *pts)
{
    for (int p = 0; p < kEstimatePoints; ++p) {
        const double z = 1.0 - (2.0 * p + 1.0) / kEstimatePoints;
        const double r = std::sqrt(1.0 - z * z);
        const double th = p * kPi * (3.0 - std::sqrt(5.0));
        pts[3 * p] = r * std::cos(th);
        pts[3 * p + 1] = r * std::sin(th);
        pts[3 * p + 2] = z;
    }
}

/* cells at offset d can hold a neighbour only if the box gap
 * cell * sqrt(sum (|d| - 1)+^2) is below the reach */
inline void
build_stencil(Grid &g, double reach)
{
    g.K = static_cast<int>(std::ceil(reach / g.cell - 1e-12));
    if (g.K < 1) g.K = 1;
    const int w = 2 * g.K + 1;
    const double lim = (reach / g.cell) * (reach / g.cell);
    g.dxmax.assign(static_cast<size_t>(w) * static_cast<size_t>(w), -1);
    for (int dz = -g.K; dz <= g.K; ++dz)
        for (int dy = -g.K; dy <= g.K; ++dy) {
            const double ez = std::max(std::abs(dz) - 1, 0), ey = std::max(std::abs(dy) - 1, 0);
            int best = -1;
            for (int dx = 0; dx <= g.K; ++dx) {
                const double ex = std::max(dx - 1, 0);
                if (ex * ex + ey * ey + ez * ez < lim) best = dx;
            }
            g.dxmax[static_cast<size_t>((dz + g.K) * w + dy + g.K)] = best;
        }
}

/* Sorts one frame (SoA input, radii already including the probe) into grid
 * order. Returns false when a coordinate or radius is not finite. */
inline bool
sort_frame(int n, const double *x, const double *y, const double *z, const double *R, SortedFrame &f)
{
    Grid &g = f.g;
    double hi[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
    double rmax = 0.0;
    for (int a = 0; a < 3; ++a) g.lo[a] = HUGE_VAL;
    for (int i = 0; i < n; ++i) {
        const double p[3] = {x[i], y[i], z[i]};
        for (int a = 0; a < 3; ++a) {
            if (!std::isfinite(p[a])) return false;
            g.lo[a] = std::min(g.lo[a], p[a]);
            hi[a] = std::max(hi[a], p[a]);
        }
        if (!std::isfinite(R[i]) || R[i] < 0.0) return false;
        rmax = std::max(rmax, R[i]);
    }
    if (n == 0) {
        for (int a = 0; a < 3; ++a) g.lo[a] = hi[a] = 0.0;
    }
    /* cell = rmax; coarser for sparse systems, so the grid stays O(n) */
    double cell = rmax > 0.0 ? rmax : 1.0;
    const double max_cells = 4.0 * n + 64.0;
    for (;;) {
        double cells = 1.0;
        for (int a = 0; a < 3; ++a) cells *= std::floor((hi[a] - g.lo[a]) / cell) + 1.0;
        if (cells <= max_cells) break;
        cell *= 1.5;
    }
    g.cell = cell;
    for (int a = 0; a < 3; ++a) g.dims[a] = std::max(1, static_cast<int>(std::floor((hi[a] - g.lo[a]) / cell)) + 1);
    const size_t nc = static_cast<size_t>(g.dims[0]) * static_cast<size_t>(g.dims[1]) * static_cast<size_t>(g.dims[2]);
    f.cell_of.resize(static_cast<size_t>(n));
    g.start.assign(nc + 1u, 0);
    for (int i = 0; i < n; ++i) {
        const double p[3] = {x[i], y[i], z[i]};
        int c[3];
        for (int a = 0; a < 3; ++a) c[a] = std::min(g.dims[a] - 1, static_cast<int>((p[a] - g.lo[a]) / cell));
        f.cell_of[static_cast<size_t>(i)] = (c[2] * g.dims[1] + c[1]) * g.dims[0] + c[0];
        ++g.start[static_cast<size_t>(f.cell_of[static_cast<size_t>(i)]) + 1u];
    }
    for (size_t c = 0; c < nc; ++c) g.start[c + 1u] += g.start[c];
    std::vector<int> fill(g.start.begin(), g.start.end() - 1);
    f.orig.resize(static_cast<size_t>(n));
    f.xs.resize(3u * static_cast<size_t>(n));
    f.Rs.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int q = fill[static_cast<size_t>(f.cell_of[static_cast<size_t>(i)])]++;
        f.orig[static_cast<size_t>(q)] = i;
        f.xs[3u * static_cast<size_t>(q)] = x[i];
        f.xs[3u * static_cast<size_t>(q) + 1u] = y[i];
        f.xs[3u * static_cast<size_t>(q) + 2u] = z[i];
        f.Rs[static_cast<size_t>(q)] = R[i];
    }
    build_stencil(g, 2.0 * rmax);
    return true;
}

}  // namespace fastsasa_exact

#endif
