/*
 * Mask-accelerated exact Shrake-Rupley for the CPU backend.
 *
 * Same result as the reference kernel in fastsasa_cpu.cc, bit for bit. The
 * reference decides, for every (atom, test point, neighbour) triple, whether
 *
 *     |p - c_j|^2 < R_j^2,   p = c_i + R_i * s
 *
 * in double precision. Rearranged, the point is blocked by neighbour j when
 * s.u > t with u = (c_j - c_i)/d and t = (R_i^2 + d^2 - R_j^2) / (2 R_i d):
 * every neighbour hides one spherical cap, described by a direction and a
 * threshold. This kernel quantizes only the direction, into an octahedral
 * grid of bins. For every bin it stores the test points sorted by their dot
 * product with the bin's centre direction, as prefix masks in point-id
 * space, plus a table mapping a threshold to a prefix length. The threshold
 * itself stays exact (its lookup in the table is rounded conservatively in
 * both directions).
 *
 * A point whose sorted dot product exceeds t by more than the bin's angular
 * radius (delta, the largest chord from the bin centre to any direction in
 * the bin) is certainly blocked; one that falls short of t by more than delta
 * is certainly clear. Both are settled with table lookups and a few 64-bit
 * operations. Only points within delta of the cap boundary are handed to the
 * reference's own distance test, evaluated in the same operation order, so
 * the decision for every point is either mathematically certain or made by
 * the exact reference arithmetic. The area formula is the reference's.
 *
 * Grid resolution is therefore a performance knob (finer bins -> fewer exact
 * tests, larger table), never an accuracy knob.
 *
 * Neighbour search: atoms are counting-sorted into a dense cell grid with
 * structure-of-arrays coordinates, so every cell is a contiguous slice and
 * the candidate scan is a plain vectorizable loop. Each pair is examined
 * once (half stencil) and recorded for both atoms.
 */
#include "fastsasa_cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

/* Portability: GCC/Clang builtins and attributes, with MSVC equivalents. */
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define FASTSASA_ALWAYS_INLINE __forceinline
#define FASTSASA_RESTRICT __restrict
static inline int fastsasa_ctz64(std::uint64_t v) { unsigned long i; _BitScanForward64(&i, v); return static_cast<int>(i); }
static inline int fastsasa_ctz32(unsigned v) { unsigned long i; _BitScanForward(&i, v); return static_cast<int>(i); }
static inline int fastsasa_popcount64(std::uint64_t v) { return static_cast<int>(__popcnt64(v)); }
#else
#define FASTSASA_ALWAYS_INLINE inline __attribute__((always_inline))
#define FASTSASA_RESTRICT __restrict__
static inline int fastsasa_ctz64(std::uint64_t v) { return __builtin_ctzll(v); }
static inline int fastsasa_ctz32(unsigned v) { return __builtin_ctz(v); }
static inline int fastsasa_popcount64(std::uint64_t v) { return __builtin_popcountll(v); }
#endif

namespace {

constexpr int kMaxWords = 4;                 /* up to 256 test points */
constexpr int kMaxPoints = 255;              /* uint8 prefix counts */
constexpr int kUnsupported = -100;
#ifndef FASTSASA_MASK_TBINS
#define FASTSASA_MASK_TBINS 64
#endif
constexpr int kThresholdBins = FASTSASA_MASK_TBINS; /* t in [-1, 1] -> table index */
constexpr double kPi = 3.141592653589793238462643383279502884;
/* Absolute pad on the dot-product margin, far above double rounding of a
 * distance test (~1e-15) and far below the geometric margin (>= ~0.02). */
constexpr double kDotPad = 1.0e-7;

/* Octahedral direction encoding (unit vector -> (u, v) in [-1, 1]^2). */
inline void octa_encode(double x, double y, double z, double &u, double &v)
{
    const double inv_l1 = 1.0 / (std::fabs(x) + std::fabs(y) + std::fabs(z));
    double px = x * inv_l1;
    double py = y * inv_l1;
    const double pz = z * inv_l1;

    if (pz < 0.0) {
        const double sx = px >= 0.0 ? 1.0 : -1.0;
        const double sy = py >= 0.0 ? 1.0 : -1.0;
        const double ax = std::fabs(px);
        const double ay = std::fabs(py);
        px = (1.0 - ay) * sx;
        py = (1.0 - ax) * sy;
    }
    u = px;
    v = py;
}

/* Bin indices for a direction v and for -v, from one L1 normalization.
 * The length of (x, y, z) is irrelevant. For the octahedral map, a point
 * (px, py, pz) on the L1 sphere encodes as (px, py) when pz >= 0 and as
 * the fold (sign(px)(1-|py|), sign(py)(1-|px|)) when pz < 0. */
inline void octa_bins(double x, double y, double z, int resolution, double res_half, int &bin_pos, int &bin_neg)
{
    const double inv_l1 = 1.0 / (std::fabs(x) + std::fabs(y) + std::fabs(z));
    const double px = x * inv_l1;
    const double py = y * inv_l1;
    const double pz = z * inv_l1;
    const double sx = px >= 0.0 ? 1.0 : -1.0;
    const double sy = py >= 0.0 ? 1.0 : -1.0;
    const double fx = (1.0 - std::fabs(py)) * sx;     /* fold of (px, py) */
    const double fy = (1.0 - std::fabs(px)) * sy;
    /* +v: (px, py) if pz >= 0 else fold.  -v: (-px, -py) if -pz >= 0
     * i.e. pz <= 0, else the fold of (-px, -py) = -(fold of (px, py)). */
    double u, v, nu, nv;
    if (pz >= 0.0) { u = px; v = py; nu = -fx; nv = -fy; }
    else           { u = fx; v = fy; nu = -px; nv = -py; }
    int bx = std::clamp(static_cast<int>((u + 1.0) * res_half), 0, resolution - 1);
    int by = std::clamp(static_cast<int>((v + 1.0) * res_half), 0, resolution - 1);
    bin_pos = by * resolution + bx;
    bx = std::clamp(static_cast<int>((nu + 1.0) * res_half), 0, resolution - 1);
    by = std::clamp(static_cast<int>((nv + 1.0) * res_half), 0, resolution - 1);
    bin_neg = by * resolution + bx;
}

inline void octa_decode(double u, double v, double &x, double &y, double &z)
{
    x = u;
    y = v;
    z = 1.0 - std::fabs(u) - std::fabs(v);
    if (z < 0.0) {
        const double sx = x >= 0.0 ? 1.0 : -1.0;
        const double sy = y >= 0.0 ? 1.0 : -1.0;
        const double ax = std::fabs(x);
        const double ay = std::fabs(y);
        x = (1.0 - ay) * sx;
        y = (1.0 - ax) * sy;
    }
    const double norm = std::sqrt(x * x + y * y + z * z);
    x /= norm;
    y /= norm;
    z /= norm;
}

struct MaskTable {
    int n_points = 0;
    int words = 0;
    int resolution = 0;
    int n_bins = 0;
    double max_delta = 0.0;
    std::vector<double> points;          /* copy of the test points, for cache identity */
    std::vector<double> delta;           /* per bin */
    /* prefix[(bin * (n_points + 1) + k) * words + w]: bits of the first k
     * sorted points. At resolution 64 and 128 points this is 8.4 MB; it
     * lives in L3, and one cap touches one or two adjacent rows. A stride-8
     * compressed variant (coarse rows plus order bytes) was measured slower:
     * the bit fix-up loop costs more than the misses it saves. */
    std::vector<std::uint64_t> prefix;
    /* entry[(bin * kThresholdBins + q) * 2 * words]: for every t in
     * [edge(q), edge(q+1)), edge(q) = -1 + 2 q / kThresholdBins,
     *   blocked = bits of points with dot > edge(q+1) + delta + pad (t rounded up)
     *   band    = bits of points with dot in (edge(q) - delta - pad, that]
     *             (the possibly-blocked band; t rounded down)
     * stored as two consecutive rows, so one cap costs one address
     * computation and one contiguous 2*words load. Both bounds are
     * conservative. The full prefix table is only needed while building. */
    std::vector<std::uint64_t> entry;

    static int resolution_from_env()
    {
        const char *value = std::getenv("FASTSASA_CPU_MASK_RESOLUTION");
        int resolution = value != nullptr && value[0] != '\0' ? std::atoi(value) : 64;
        if (resolution < 4) resolution = 4;
        if (resolution > 128) resolution = 128;
        if (resolution & 1) ++resolution;   /* even: bins never straddle a fold line */
        return resolution;
    }

    bool matches(int n, const double *test_points) const
    {
        return n == n_points &&
               std::memcmp(points.data(), test_points, sizeof(double) * 3u * static_cast<size_t>(n)) == 0;
    }

    inline const std::uint64_t *prefix_at(int bin, int k) const
    {
        return prefix.data() + (static_cast<size_t>(bin) * static_cast<size_t>(n_points + 1) + static_cast<size_t>(k)) * static_cast<size_t>(words);
    }

    void build(int n, const double *test_points, int res)
    {
        n_points = n;
        words = (n + 63) / 64;
        resolution = res;
        n_bins = res * res;
        points.assign(test_points, test_points + 3 * n);
        delta.assign(static_cast<size_t>(n_bins), 0.0);
        prefix.assign(static_cast<size_t>(n_bins) * static_cast<size_t>(n + 1) * static_cast<size_t>(words), 0u);
        entry.assign(static_cast<size_t>(n_bins) * static_cast<size_t>(kThresholdBins) * 2u * static_cast<size_t>(words), 0u);
        max_delta = 0.0;

        std::vector<std::pair<double, int>> scratch(static_cast<size_t>(n));
        for (int iy = 0; iy < res; ++iy) {
            for (int ix = 0; ix < res; ++ix) {
                const int bin = iy * res + ix;
                const double u0 = -1.0 + 2.0 * ix / res;
                const double v0 = -1.0 + 2.0 * iy / res;
                const double u1 = u0 + 2.0 / res;
                const double v1 = v0 + 2.0 / res;
                double cx, cy, cz;
                octa_decode(0.5 * (u0 + u1), 0.5 * (v0 + v1), cx, cy, cz);

                /* The cell is a spherical polygon with geodesic edges (a
                 * straight segment on an octahedron face maps to a great
                 * circle), so the farthest direction from an interior point
                 * is a corner. Sample the edges too, for safety. */
                double d_max = 0.0;
                for (int a = 0; a <= 8; ++a) {
                    for (int b = 0; b <= 8; ++b) {
                        if (a != 0 && a != 8 && b != 0 && b != 8) continue;
                        double dx, dy, dz;
                        octa_decode(u0 + (u1 - u0) * a / 8.0, v0 + (v1 - v0) * b / 8.0, dx, dy, dz);
                        const double chord = std::sqrt((dx - cx) * (dx - cx) + (dy - cy) * (dy - cy) + (dz - cz) * (dz - cz));
                        d_max = std::max(d_max, chord);
                    }
                }
                delta[static_cast<size_t>(bin)] = d_max * (1.0 + 1.0e-6) + 1.0e-9;
                max_delta = std::max(max_delta, delta[static_cast<size_t>(bin)]);

                for (int p = 0; p < n; ++p) {
                    const double dot = test_points[3 * p] * cx + test_points[3 * p + 1] * cy + test_points[3 * p + 2] * cz;
                    scratch[static_cast<size_t>(p)] = {dot, p};
                }
                std::sort(scratch.begin(), scratch.end(),
                          [](const std::pair<double, int> &l, const std::pair<double, int> &r) {
                              return l.first > r.first || (l.first == r.first && l.second < r.second);
                          });
                std::uint64_t *row = prefix.data() + static_cast<size_t>(bin) * static_cast<size_t>(n + 1) * static_cast<size_t>(words);
                for (int k = 0; k < n; ++k) {
                    std::memcpy(row + static_cast<size_t>(k + 1) * words, row + static_cast<size_t>(k) * words, sizeof(std::uint64_t) * static_cast<size_t>(words));
                    const int id = scratch[static_cast<size_t>(k)].second;
                    row[static_cast<size_t>(k + 1) * words + (id >> 6)] |= std::uint64_t(1) << (id & 63);
                }
                const double d_bin = delta[static_cast<size_t>(bin)] + kDotPad;
                auto count_above = [&](double v) {
                    int k = 0;
                    while (k < n && scratch[static_cast<size_t>(k)].first > v) ++k;
                    return k;
                };
                for (int q = 0; q < kThresholdBins; ++q) {
                    const double lo_edge = -1.0 + 2.0 * q / kThresholdBins;
                    const double hi_edge = -1.0 + 2.0 * (q + 1) / kThresholdBins;
                    const int kb = count_above(hi_edge + d_bin);
                    const int ka = count_above(lo_edge - d_bin);
                    std::uint64_t *e = entry.data() + (static_cast<size_t>(bin) * kThresholdBins + static_cast<size_t>(q)) * 2u * static_cast<size_t>(words);
                    const std::uint64_t *pb = row + static_cast<size_t>(kb) * words;
                    const std::uint64_t *pa = row + static_cast<size_t>(ka) * words;
                    for (int w = 0; w < words; ++w) {
                        e[w] = pb[w];
                        e[words + w] = pa[w] & ~pb[w];
                    }
                }
            }
        }
        prefix.clear();
        prefix.shrink_to_fit();
    }

    /* Table entry (blocked row, then band row) for threshold t in
     * (-1-delta, 1+delta) and direction bin. */
    inline const std::uint64_t *entry_at(int bin, double t) const
    {
        int q = static_cast<int>((t + 1.0) * (0.5 * kThresholdBins));
        if (q < 0) q = 0;
        if (q >= kThresholdBins) q = kThresholdBins - 1;
        return entry.data() + (static_cast<size_t>(bin) * kThresholdBins + static_cast<size_t>(q)) * 2u * static_cast<size_t>(words);
    }
};


/* Vector types for the fused neighbour loop (GCC/Clang). Elsewhere the
 * scalar path below is used. Both compute identical values: the same
 * operations in the same order, with contraction disabled. */
#if defined(__GNUC__) || defined(__clang__)
typedef double v4d __attribute__((vector_size(32)));
typedef long long v4l __attribute__((vector_size(32)));
#define FASTSASA_HAVE_V4D 1
#endif

/* Scalar geometry for one neighbour: cap threshold and octahedral (u, v)
 * scaled to bin units. */
static FASTSASA_ALWAYS_INLINE void geometry_one(double vx, double vy, double vz, double d2, double rj2,
                         double ri2, double inv_two_ri, double res_half,
                         double &t, double &su, double &sv)
{
    const double inv_d = 1.0 / std::sqrt(d2);
    t = (ri2 + d2 - rj2) * inv_two_ri * inv_d;
    const double inv_l1 = 1.0 / (std::fabs(vx) + std::fabs(vy) + std::fabs(vz));
    const double px = vx * inv_l1;
    const double py = vy * inv_l1;
    const double pz = vz * inv_l1;
    const double sx = px >= 0.0 ? 1.0 : -1.0;
    const double sy = py >= 0.0 ? 1.0 : -1.0;
    const double fx = (1.0 - std::fabs(py)) * sx;
    const double fy = (1.0 - std::fabs(px)) * sy;
    const bool up = pz >= 0.0;
    su = ((up ? px : fx) + 1.0) * res_half;
    sv = ((up ? py : fy) + 1.0) * res_half;
}


std::mutex g_table_mutex;
std::shared_ptr<const MaskTable> g_table;
/* Atom-evaluations requested from the CPU Shrake-Rupley path so far in this
 * process (across calls, frames, and threads). The default policy builds
 * the table once this passes the break-even where the build cost is repaid
 * by the kernel's per-atom saving; see fastsasa_cpu_mask_policy(). */
std::atomic<long long> g_atom_work{0};

bool table_ready(int n_points, const double *test_points)
{
    const int resolution = MaskTable::resolution_from_env();
    std::lock_guard<std::mutex> lock(g_table_mutex);
    return g_table && g_table->resolution == resolution && g_table->matches(n_points, test_points);
}

std::shared_ptr<const MaskTable> acquire_table(int n_points, const double *test_points)
{
    const int resolution = MaskTable::resolution_from_env();
    std::lock_guard<std::mutex> lock(g_table_mutex);
    if (g_table && g_table->resolution == resolution && g_table->matches(n_points, test_points)) return g_table;
    auto table = std::make_shared<MaskTable>();
    table->build(n_points, test_points, resolution);
    g_table = table;
    return table;
}

/* Dense cell grid; atoms counting-sorted by cell with SoA coordinates. */
struct Grid {
    double origin[3];
    double inv_cell;
    int dims[3];
    int stencil;                          /* cells to each side covering the reach */
    std::vector<std::pair<int, int>> rows;  /* (dz, dy) offsets, nearest first */
    std::vector<int> cell_start;          /* size cells + 1 */
    std::vector<int> atom_of;             /* sorted slot -> atom id */
    std::vector<double> sx, sy, sz, sr;   /* sorted coordinates and expanded radii */
    std::vector<double> sr2, inv_two_r;   /* R^2 and 1/(2R), sorted */

    bool build(int n_atoms, const double *x, const double *y, const double *z, const double *r,
               double reach, double cell_size)
    {
        double lo[3] = {x[0], y[0], z[0]};
        double hi[3] = {x[0], y[0], z[0]};
        for (int a = 1; a < n_atoms; ++a) {
            lo[0] = std::min(lo[0], x[a]); hi[0] = std::max(hi[0], x[a]);
            lo[1] = std::min(lo[1], y[a]); hi[1] = std::max(hi[1], y[a]);
            lo[2] = std::min(lo[2], z[a]); hi[2] = std::max(hi[2], z[a]);
        }
        inv_cell = 1.0 / cell_size;
        stencil = static_cast<int>(std::ceil(reach * inv_cell));
        rows.clear();
        for (int dz = -stencil; dz <= stencil; ++dz)
            for (int dy = -stencil; dy <= stencil; ++dy) rows.push_back({dz, dy});
        std::sort(rows.begin(), rows.end(), [](const std::pair<int, int> &l, const std::pair<int, int> &r) {
            return l.first * l.first + l.second * l.second < r.first * r.first + r.second * r.second;
        });
        std::size_t cells = 1;
        for (int k = 0; k < 3; ++k) {
            origin[k] = lo[k];
            const double span = (hi[k] - lo[k]) * inv_cell;
            if (!(span < 1.0e7)) return false;
            dims[k] = static_cast<int>(span) + 1;
            cells *= static_cast<std::size_t>(dims[k]);
            if (cells > (std::size_t(1) << 26)) return false;
        }
        cell_start.assign(cells + 1u, 0);
        std::vector<int> cell_of(static_cast<size_t>(n_atoms));
        for (int a = 0; a < n_atoms; ++a) {
            const int c = cell_index(x[a], y[a], z[a]);
            cell_of[static_cast<size_t>(a)] = c;
            ++cell_start[static_cast<size_t>(c) + 1u];
        }
        for (std::size_t c = 0; c < cells; ++c) cell_start[c + 1u] += cell_start[c];
        atom_of.resize(static_cast<size_t>(n_atoms));
        sx.resize(static_cast<size_t>(n_atoms));
        sy.resize(static_cast<size_t>(n_atoms));
        sz.resize(static_cast<size_t>(n_atoms));
        sr.resize(static_cast<size_t>(n_atoms));
        sr2.resize(static_cast<size_t>(n_atoms));
        inv_two_r.resize(static_cast<size_t>(n_atoms));
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int a = 0; a < n_atoms; ++a) {
            const int slot = fill[static_cast<size_t>(cell_of[static_cast<size_t>(a)])]++;
            atom_of[static_cast<size_t>(slot)] = a;
            sx[static_cast<size_t>(slot)] = x[a];
            sy[static_cast<size_t>(slot)] = y[a];
            sz[static_cast<size_t>(slot)] = z[a];
            sr[static_cast<size_t>(slot)] = r[a];
            sr2[static_cast<size_t>(slot)] = r[a] * r[a];
            inv_two_r[static_cast<size_t>(slot)] = 1.0 / (2.0 * r[a]);
        }
        return true;
    }

    inline int coord(double v, int k) const
    {
        int c = static_cast<int>((v - origin[k]) * inv_cell);
        if (c < 0) c = 0;
        if (c >= dims[k]) c = dims[k] - 1;
        return c;
    }
    inline int cell_index(double x, double y, double z) const
    {
        return (coord(z, 2) * dims[1] + coord(y, 1)) * dims[0] + coord(x, 0);
    }
};

/* A neighbour at the same centre (d = 0) has no cap direction, and the
 * reference's per-point distance test |R_i s|^2 < R_j^2 is decided by
 * rounding when R_i == R_j. Every point is therefore treated as ambiguous
 * and handed to the exact pass, which reproduces the reference bit for bit. */
template <int W>
static FASTSASA_ALWAYS_INLINE void park_all_points(std::uint64_t (&visible)[W], int b,
                                   std::uint64_t *FASTSASA_RESTRICT pending, int *FASTSASA_RESTRICT pending_cap, int &n_pending)
{
    std::uint64_t *dst = pending + static_cast<size_t>(n_pending) * kMaxWords;
    std::uint64_t any = 0;
    for (int w = 0; w < W; ++w) { dst[w] = visible[w]; any |= visible[w]; }
    if (any) { pending_cap[n_pending] = b; ++n_pending; }
}

struct Stats {
    long atoms = 0, neighbours = 0, exact_tests = 0, early_break = 0, fully_buried = 0, cleared_by_test = 0;
};

struct WorkerArgs {
    const Grid &grid;
    const MaskTable &lut;
    const double *test_points;
    double *sasa;
    Stats *stats;
    std::atomic<int> *status;
    std::atomic<long> *candidates;
};

/* One neighbour's cap applied to the visibility mask. Returns false when
 * the atom is settled (fully buried, or nothing visible remains). */
template <int W>
static FASTSASA_ALWAYS_INLINE bool apply_cap(const MaskTable &lut, std::uint64_t (&visible)[W],
                             int b, double t, double su, double sv,
                             double lim_lo, double lim_hi, int res_max,
                             std::uint64_t *FASTSASA_RESTRICT pending, int *FASTSASA_RESTRICT pending_cap, int &n_pending,
                             bool &fully_buried, Stats &st)
{
    if (t >= lim_hi) return true;
    if (t <= lim_lo) { fully_buried = true; return false; }
    ++st.neighbours;
    const int bx = std::clamp(static_cast<int>(su), 0, res_max);
    const int by = std::clamp(static_cast<int>(sv), 0, res_max);
    const int bin = by * lut.resolution + bx;
    const std::uint64_t *e = lut.entry_at(bin, t);
    const std::uint64_t *blocked = e;
    const std::uint64_t *band = e + W;
    std::uint64_t any = 0;
    std::uint64_t amb_any = 0;
    std::uint64_t amb[W];
    for (int w = 0; w < W; ++w) {
        const std::uint64_t vis = visible[w] & ~blocked[w];
        amb[w] = vis & band[w];
        amb_any |= amb[w];
        visible[w] = vis;
        any |= vis;
    }
    if (amb_any) {
        std::uint64_t *dst = pending + static_cast<size_t>(n_pending) * kMaxWords;
        for (int w = 0; w < W; ++w) dst[w] = amb[w];
        pending_cap[n_pending] = b;
        ++n_pending;
    }
    if (any == 0) { ++st.early_break; return false; }
    return true;
}

/* The worker is compiled twice on x86-64 GCC/Clang: once for the baseline
 * ISA and once with AVX2+FMA codegen (FMA contraction stays off, so the
 * arithmetic is identical; only the vector width and instruction selection
 * change). The AVX2 copy is chosen at run time when the CPU supports it. */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && !defined(__AVX2__)
#define FASTSASA_MASK_MULTIVERSION 1
#endif

template <int W>
static FASTSASA_ALWAYS_INLINE void worker_body(const WorkerArgs &A, int begin, int end, int tid);

template <int W>
static void worker_impl_base(const WorkerArgs &A, int begin, int end, int tid)
{
    worker_body<W>(A, begin, end, tid);
}

#if FASTSASA_MASK_MULTIVERSION
template <int W>
__attribute__((target("avx2,fma")))
static void worker_impl_avx2(const WorkerArgs &A, int begin, int end, int tid)
{
    worker_body<W>(A, begin, end, tid);
}
#endif

template <int W>
static void worker_impl(const WorkerArgs &A, int begin, int end, int tid)
{
#if FASTSASA_MASK_MULTIVERSION
    static const bool have_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (have_avx2) {
        worker_impl_avx2<W>(A, begin, end, tid);
        return;
    }
#endif
    worker_impl_base<W>(A, begin, end, tid);
}

template <int W>
static FASTSASA_ALWAYS_INLINE void worker_body(const WorkerArgs &A, int begin, int end, int tid)
{
    const Grid &grid = A.grid;
    const MaskTable &lut = A.lut;
    const double *test_points = A.test_points;
    double *sasa = A.sasa;
    const int n_points = lut.n_points;
    Stats st;
    long candidates = 0;
    try {
        std::uint64_t full[W];
        for (int w = 0; w < W; ++w) full[w] = 0;
        for (int p = 0; p < n_points; ++p) full[p >> 6] |= std::uint64_t(1) << (p & 63);
        const double res_half = 0.5 * lut.resolution;
        const int res_max = lut.resolution - 1;
        const double lim_hi = 1.0 + lut.max_delta + kDotPad;    /* t >= lim_hi: no effect */
        const double lim_lo = -1.0 - lut.max_delta - kDotPad;   /* t <= lim_lo: buries */
        const int s_ = grid.stencil;
        const double *const FASTSASA_RESTRICT gsx = grid.sx.data();
        const double *const FASTSASA_RESTRICT gsy = grid.sy.data();
        const double *const FASTSASA_RESTRICT gsz = grid.sz.data();
        const double *const FASTSASA_RESTRICT gsr = grid.sr.data();
        const double *const FASTSASA_RESTRICT gsr2 = grid.sr2.data();
        const int *const FASTSASA_RESTRICT gcell = grid.cell_start.data();
        alignas(64) int bidx[kMaxPoints];
        int pending_cap_storage_size = 1024;
        std::vector<std::uint64_t> pending_storage(static_cast<size_t>(kMaxWords) * 1024u);
        std::vector<int> pending_cap_storage(1024u);
        std::uint64_t *pending = pending_storage.data();
        int *pending_cap = pending_cap_storage.data();
#if FASTSASA_HAVE_V4D
        const v4d one = {1.0, 1.0, 1.0, 1.0}, zero = {0.0, 0.0, 0.0, 0.0};
        const v4l signbit = {(long long)0x8000000000000000ULL, (long long)0x8000000000000000ULL,
                             (long long)0x8000000000000000ULL, (long long)0x8000000000000000ULL};
#endif

        for (int slot = begin; slot < end; ++slot) {
            const int atom = grid.atom_of[static_cast<size_t>(slot)];
            const double ri = gsr[slot];
            const double ri2 = gsr2[slot];
            const double inv_two_ri = grid.inv_two_r[static_cast<size_t>(slot)];
            const double xi = gsx[slot];
            const double yi = gsy[slot];
            const double zi = gsz[slot];
            ++st.atoms;

            /* Growth for pathological densities: a stencil can hold at most
             * the atoms of 125 cells; size pending for the worst case seen. */
            {
                const int gx0 = grid.coord(xi, 0), gy0 = grid.coord(yi, 1), gz0 = grid.coord(zi, 2);
                int upper = 0;
                for (int dz = std::max(gz0 - s_, 0); dz <= std::min(gz0 + s_, grid.dims[2] - 1); ++dz)
                    for (int dy = std::max(gy0 - s_, 0); dy <= std::min(gy0 + s_, grid.dims[1] - 1); ++dy) {
                        const int row = (dz * grid.dims[1] + dy) * grid.dims[0];
                        upper += gcell[row + std::min(gx0 + s_, grid.dims[0] - 1) + 1] - gcell[row + std::max(gx0 - s_, 0)];
                    }
                if (upper + 1 > pending_cap_storage_size) {
                    pending_cap_storage_size = 2 * (upper + 1);
                    pending_cap_storage.resize(static_cast<size_t>(pending_cap_storage_size));
                    pending_storage.resize(static_cast<size_t>(pending_cap_storage_size) * kMaxWords);
                    pending = pending_storage.data();
                    pending_cap = pending_cap_storage.data();
                }
            }

            /* One fused pass over the stencil, nearest rows first.
             * Candidates go four at a time: overlap test and cap geometry
             * in vector registers, then table and mask work for the lanes
             * that overlap. */
            std::uint64_t visible[W];
            for (int w = 0; w < W; ++w) visible[w] = full[w];
            int n_pending = 0;
            bool fully_buried = false;
            bool go = true;
            const int gx = grid.coord(xi, 0), gy = grid.coord(yi, 1), gz = grid.coord(zi, 2);
#if FASTSASA_HAVE_V4D
            const v4d vxi = {xi, xi, xi, xi}, vyi = {yi, yi, yi, yi}, vzi = {zi, zi, zi, zi}, vri = {ri, ri, ri, ri};
            const v4d vri2 = {ri2, ri2, ri2, ri2}, vinv2 = {inv_two_ri, inv_two_ri, inv_two_ri, inv_two_ri};
            const v4d vhalf = {res_half, res_half, res_half, res_half};
#endif
            for (std::size_t ri_ = 0; ri_ < grid.rows.size() && go; ++ri_) {
                const int dz = gz + grid.rows[ri_].first;
                const int dy = gy + grid.rows[ri_].second;
                if (dz < 0 || dz >= grid.dims[2] || dy < 0 || dy >= grid.dims[1]) continue;
                const int row = (dz * grid.dims[1] + dy) * grid.dims[0];
                const int c0 = row + std::max(gx - s_, 0);
                const int c1 = row + std::min(gx + s_, grid.dims[0] - 1);
                const int b0 = gcell[c0];
                const int b1 = gcell[c1 + 1];
                candidates += b1 - b0;
                int b = b0;
#if FASTSASA_HAVE_V4D
                for (; b + 4 <= b1 && go; b += 4) {
                    v4d px, py, pz, pr, pr2;
                    __builtin_memcpy(&px, gsx + b, 32); __builtin_memcpy(&py, gsy + b, 32);
                    __builtin_memcpy(&pz, gsz + b, 32); __builtin_memcpy(&pr, gsr + b, 32);
                    __builtin_memcpy(&pr2, gsr2 + b, 32);
                    const v4d vx = px - vxi, vy = py - vyi, vz = pz - vzi;
                    const v4d d2 = vx * vx + vy * vy + vz * vz;
                    const v4d reach = vri + pr;
                    const v4l hit = d2 < reach * reach;
                    unsigned m = 0;
                    for (int l = 0; l < 4; ++l) m |= static_cast<unsigned>(hit[l] != 0) << l;
                    if (b <= slot && slot < b + 4) m &= ~(1u << (slot - b));
                    if (!m) continue;
                    /* Geometry for all four lanes. d2 == 0 lanes (self, or a
                     * coincident atom) get a harmless synthetic direction;
                     * self is masked out above, and a coincident neighbour
                     * yields t = (ri2 - rj2) / 0 -> +-inf, which the lim
                     * tests turn into "no effect" or "buries", the same as
                     * the reference's d2 < rj2 test at the sphere centre. */
                    const v4l zero_d = d2 == zero;
                    const v4d d2s = (v4d)(((v4l)d2 & ~zero_d) | ((v4l)one & zero_d));
                    v4d sq = zero;
                    for (int l = 0; l < 4; ++l) sq[l] = __builtin_sqrt(d2s[l]);
                    const v4d inv_d = one / sq;
                    v4d t = (vri2 + d2 - pr2) * vinv2 * inv_d;
                    const v4d ax = (v4d)((v4l)vx & ~signbit);
                    const v4d ay = (v4d)((v4l)vy & ~signbit);
                    const v4d az = (v4d)((v4l)vz & ~signbit);
                    const v4d l1 = ax + ay + az;
                    const v4d l1s = (v4d)(((v4l)l1 & ~zero_d) | ((v4l)one & zero_d));
                    const v4d inv_l1 = one / l1s;
                    const v4d qx = vx * inv_l1, qy = vy * inv_l1, qz = vz * inv_l1;
                    const v4l sx = (v4l)qx & signbit, sy = (v4l)qy & signbit;
                    const v4d aqx = (v4d)((v4l)qx & ~signbit), aqy = (v4d)((v4l)qy & ~signbit);
                    const v4d fx = (v4d)((v4l)(one - aqy) | sx);
                    const v4d fy = (v4d)((v4l)(one - aqx) | sy);
                    const v4l down = qz < zero;
                    const v4d u = (v4d)(((v4l)fx & down) | ((v4l)qx & ~down));
                    const v4d v = (v4d)(((v4l)fy & down) | ((v4l)qy & ~down));
                    const v4d su = (u + one) * vhalf;
                    const v4d sv = (v + one) * vhalf;
                    /* Batched cap application: classify the four lanes,
                     * issue all table loads independently (kpair, then
                     * both prefix rows), and only then touch the mask, so
                     * the dependent load chains of up to four neighbours
                     * overlap instead of serializing. */
                    {
                        const std::uint64_t *rows_b[4];
                        int slots[4];
                        int n_lanes = 0;
                        bool buried_now = false;
                        unsigned mm = m;
                        while (mm) {
                            const int l = fastsasa_ctz32(mm);
                            mm &= mm - 1;
                            if (zero_d[l]) { park_all_points<W>(visible, b + l, pending, pending_cap, n_pending); ++st.neighbours; continue; }
                            const double tl = t[l];
                            if (tl >= lim_hi) continue;
                            if (tl <= lim_lo) { buried_now = true; break; }
                            const int bx = std::clamp(static_cast<int>(su[l]), 0, res_max);
                            const int by = std::clamp(static_cast<int>(sv[l]), 0, res_max);
                            const int bin = by * lut.resolution + bx;
                            rows_b[n_lanes] = lut.entry_at(bin, tl);
                            slots[n_lanes] = b + l;
                            ++n_lanes;
                        }
                        if (buried_now) { fully_buried = true; go = false; break; }
                        st.neighbours += n_lanes;
                        for (int q = 0; q < n_lanes; ++q) {
                            std::uint64_t any = 0;
                            std::uint64_t amb_any = 0;
                            std::uint64_t amb[W];
                            const std::uint64_t *blocked = rows_b[q];
                            const std::uint64_t *band = blocked + W;
                            for (int w = 0; w < W; ++w) {
                                const std::uint64_t vis = visible[w] & ~blocked[w];
                                amb[w] = vis & band[w];
                                amb_any |= amb[w];
                                visible[w] = vis;
                                any |= vis;
                            }
                            if (amb_any) {
                                std::uint64_t *dst = pending + static_cast<size_t>(n_pending) * kMaxWords;
                                for (int w = 0; w < W; ++w) dst[w] = amb[w];
                                pending_cap[n_pending] = slots[q];
                                ++n_pending;
                            }
                            if (any == 0) { ++st.early_break; go = false; break; }
                        }
                    }
                }
#endif
                for (; b < b1 && go; ++b) {
                    const double vx = gsx[b] - xi;
                    const double vy = gsy[b] - yi;
                    const double vz = gsz[b] - zi;
                    const double d2 = vx * vx + vy * vy + vz * vz;
                    const double reach = ri + gsr[b];
                    if (!(d2 < reach * reach) || b == slot) continue;
                    double t, su, sv;
                    if (d2 == 0.0) { park_all_points<W>(visible, b, pending, pending_cap, n_pending); ++st.neighbours; continue; }
                    geometry_one(vx, vy, vz, d2, gsr2[b], ri2, inv_two_ri, res_half, t, su, sv);
                    go = apply_cap<W>(lut, visible, b, t, su, sv, lim_lo, lim_hi, res_max,
                                      pending, pending_cap, n_pending, fully_buried, st);
                }
            }

            int accessible = 0;
            if (fully_buried) {
                ++st.fully_buried;
            } else {
                /* Exact pass for parked bits that are still visible. */
                for (int q = 0; q < n_pending; ++q) {
                    const std::uint64_t *amb = pending + static_cast<size_t>(q) * kMaxWords;
                    int n_amb = 0;
                    for (int w = 0; w < W; ++w) {
                        std::uint64_t a = amb[w] & visible[w];
                        while (a) {
                            const int bit = fastsasa_ctz64(a);
                            a &= a - 1;
                            bidx[n_amb++] = (w << 6) | bit;
                        }
                    }
                    if (!n_amb) continue;
                    const int js = pending_cap[q];
                    const double rj2 = gsr2[js];
                    const double xj = gsx[js];
                    const double yj = gsy[js];
                    const double zj = gsz[js];
                    st.exact_tests += n_amb;
                    /* The reference's exact test, same operation order. */
                    for (int k = 0; k < n_amb; ++k) {
                        const int p = bidx[k];
                        const double px = xi + ri * test_points[3 * p];
                        const double py = yi + ri * test_points[3 * p + 1];
                        const double pz = zi + ri * test_points[3 * p + 2];
                        const double ddx = px - xj;
                        const double ddy = py - yj;
                        const double ddz = pz - zj;
                        if (ddx * ddx + ddy * ddy + ddz * ddz < rj2) {
                            visible[p >> 6] &= ~(std::uint64_t(1) << (p & 63));
                            ++st.cleared_by_test;
                        }
                    }
                }
                for (int w = 0; w < W; ++w) accessible += fastsasa_popcount64(visible[w]);
            }
            sasa[atom] = 4.0 * kPi * ri * ri * static_cast<double>(accessible) / static_cast<double>(n_points);
        }
    } catch (const std::bad_alloc &) {
        A.status->store(FASTSASA_MEMORY_ERROR);
    } catch (...) {
        A.status->store(FASTSASA_INVALID_ARGUMENT);
    }
    A.stats[tid] = st;
    A.candidates->fetch_add(candidates, std::memory_order_relaxed);
}



/* A cap as seen from one endpoint: the neighbour, its bin, and the two
 * prefix lengths (certain / possible). kind: 0 normal, 1 no effect, 2 buries. */
struct Cap {
    int slot;
    int bin;
    std::uint16_t k_block;
    std::uint16_t k_amb;
};


} // namespace

/* Default policy for the CPU Shrake-Rupley path. Returns 1 when the mask
 * kernel should run, 0 for the reference kernel.
 *
 *   FASTSASA_CPU_KERNEL=reference  always the reference kernel
 *   FASTSASA_CPU_KERNEL=mask       always the mask kernel (builds the table)
 *   unset / auto                   mask kernel once the cumulative work in
 *                                  this process repays the one-time table
 *                                  build (about 43 ms at 128 points: the
 *                                  equivalent of ~20k atom-evaluations at
 *                                  the ~2.3 us/atom the kernel saves), or
 *                                  immediately if the table already exists.
 * Point counts above 255 are always the reference kernel. */
extern "C" int
fastsasa_cpu_mask_policy(int n_atoms, int n_points, const double *test_points)
{
    if (n_points > kMaxPoints || n_points <= 0 || n_atoms <= 0) return 0;
    const char *kernel = std::getenv("FASTSASA_CPU_KERNEL");
    if (kernel != nullptr && kernel[0] != '\0') {
        if (std::strcmp(kernel, "mask") == 0) return 1;
        if (std::strcmp(kernel, "reference") == 0) return 0;
    }
    if (table_ready(n_points, test_points)) return 1;
    /* Break-even in atom-evaluations, scaled with the build cost (linear in
     * n_points at fixed resolution: ~0.33 ms per point at res 64). */
    const long long break_even = 160LL * static_cast<long long>(n_points);
    const long long work = g_atom_work.fetch_add(n_atoms, std::memory_order_relaxed) + n_atoms;
    return work >= break_even ? 1 : 0;
}

extern "C" int
fastsasa_cpu_shrake_rupley_mask(int n_atoms,
                                int n_points,
                                const double *x,
                                const double *y,
                                const double *z,
                                const double *expanded_radii,
                                const double *test_points,
                                int n_threads,
                                double *sasa)
{
    if (n_points > kMaxPoints) return kUnsupported;
    if (n_points <= 0 || n_atoms <= 0) return FASTSASA_INVALID_ARGUMENT;

    double max_radius = 0.0;
    for (int a = 0; a < n_atoms; ++a) max_radius = std::max(max_radius, expanded_radii[a]);
    if (max_radius <= 0.0) return FASTSASA_INVALID_ARGUMENT;

    const std::shared_ptr<const MaskTable> table = acquire_table(n_points, test_points);
    const MaskTable &lut = *table;
    const int words = lut.words;

    /* Cell edge = max radius, stencil 2: covers reach 2 * max_radius with a
     * 5x5x5 block, 42% less volume than 3x3x3 cells of twice the edge. */
    Grid grid;
    if (!grid.build(n_atoms, x, y, z, expanded_radii, 2.0 * max_radius, max_radius)) return kUnsupported;

    /* Threads: at ~1 us per atom, a thread launch (~20-40 us) only pays
     * for itself above a few hundred atoms per thread; measured optimum on
     * a 602-atom structure was 4 threads, so cap at one thread per 256
     * atoms when the caller did not ask for a specific count. */
    int count = n_threads > 0 ? n_threads : fastsasa_cpu_default_threads();
    if (count < 1) count = 1;
    if (n_threads <= 0) {
        const int by_size = n_atoms / 256 + 1;
        if (count > by_size) count = by_size;
    }
    if (count > n_atoms) count = n_atoms;

    const bool want_stats = std::getenv("FASTSASA_CPU_MASK_STATS") != nullptr;
    std::vector<Stats> stats(static_cast<size_t>(count));
    std::atomic<int> worker_status(FASTSASA_SUCCESS);
    std::atomic<long> candidates{0};
    const std::size_t n_pairs = 0;

    /* Atom-major: each atom scans its own stencil (every pair is seen twice,
     * once per endpoint, so there is no scatter and no cross-atom traffic;
     * atoms are processed in sorted slot order so their neighbourhoods are
     * spatially and cache local). The worker is instantiated per mask width
     * so the visibility mask is a fixed-size register array. */
    WorkerArgs args{grid, lut, test_points, sasa, stats.data(), &worker_status, &candidates};
    auto run = [&](int begin, int end, int tid) {
        switch (words) {
        case 1: worker_impl<1>(args, begin, end, tid); break;
        case 2: worker_impl<2>(args, begin, end, tid); break;
        case 3: worker_impl<3>(args, begin, end, tid); break;
        default: worker_impl<4>(args, begin, end, tid); break;
        }
    };

    if (count == 1) {
        run(0, n_atoms, 0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(count));
        for (int tid = 0; tid < count; ++tid) {
            const int begin = static_cast<int>(static_cast<size_t>(tid) * static_cast<size_t>(n_atoms) / static_cast<size_t>(count));
            const int end = static_cast<int>(static_cast<size_t>(tid + 1) * static_cast<size_t>(n_atoms) / static_cast<size_t>(count));
            threads.emplace_back(run, begin, end, tid);
        }
        for (std::thread &thread : threads) thread.join();
    }
    if (want_stats) {
        Stats total;
        for (const Stats &s : stats) {
            total.atoms += s.atoms; total.neighbours += s.neighbours; total.exact_tests += s.exact_tests;
            total.early_break += s.early_break; total.fully_buried += s.fully_buried; total.cleared_by_test += s.cleared_by_test;
        }
        std::fprintf(stderr,
                     "mask-stats res=%d atoms=%ld cand/atom=%.1f nbr/atom=%.1f exact/atom=%.1f cleared/exact=%.2f early_break=%.2f fully_buried=%.3f pairs=%zu\n",
                     lut.resolution, total.atoms, static_cast<double>(candidates.load()) / n_atoms,
                     static_cast<double>(total.neighbours) / total.atoms, static_cast<double>(total.exact_tests) / total.atoms,
                     static_cast<double>(total.cleared_by_test) / std::max(1L, total.exact_tests),
                     static_cast<double>(total.early_break) / total.atoms, static_cast<double>(total.fully_buried) / total.atoms, n_pairs);
    }
    return worker_status.load();
}
