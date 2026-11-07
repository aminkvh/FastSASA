/*
 * Analytic (exact) SASA of one atom, shared by the CPU backend and the GPU
 * recompute path: fixed-size workspaces, no allocation, host and device.
 *
 * Atom i's exposed surface is its sphere cut by the radical planes
 * n_k . p <= h_k of its overlapping neighbours. The power cell (a box clipped
 * by those planes, kept as a degree-3 vertex graph) gives each circle's
 * bounding edges, and Gauss-Bonnet gives
 *     A = R^2 (2 pi chi - sum of exterior angles) + R sum_k h_k phi_k.
 * Planes are cut nearest first, so buried atoms stop early. Near-degenerate
 * configurations (a cell vertex on the sphere, an edge tangent to it,
 * coincident centres) are retried with a deterministic per-atom jitter of
 * 1e-8 to 1e-4 A; an atom that no attempt resolves is returned as NaN,
 * never as a silently wrong value.
 */
#ifndef FASTSASA_EXACT_CORE_H
#define FASTSASA_EXACT_CORE_H
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define FASTSASA_EXACT_HD __host__ __device__ __forceinline__
#else
#define FASTSASA_EXACT_HD inline
#endif

namespace fastsasa_exact {

constexpr double kPi = 3.14159265358979323846;
constexpr int kEstimatePoints = 64;
constexpr int kBuckets = 32;
constexpr double kTolRel = 1e-13;  // near-degeneracy threshold, relative to the atom radius

/* Neighbour search over atoms stored in grid order. */
struct GridView {
    double lo[3], cell;
    int dims[3];
    const int *start;    // cell -> first atom (size cells + 1)
    int K;               // stencil reach in cells
    const int *dxmax;    // per (dz, dy) offset: largest |dx|, -1 none
};

struct Plane {
    double nx, ny, nz, h;
};

/* Workspace capacities: planes (overlapping neighbours), cell vertices (< 255), edge-sphere crossings.
 * Small for the GPU; an atom that overflows is recomputed with the large one. */
template <int kMaxPlanes_, int kMaxGV_, int kMaxCross_>
struct Caps {
    static constexpr int kMaxPlanes = kMaxPlanes_, kMaxGV = kMaxGV_, kMaxCross = kMaxCross_;
};
using SmallCaps = Caps<128, 96, 128>;
using LargeCaps = Caps<2048, 254, 1024>;

template <class C>
struct Graph {
    static constexpr int kMaxGV = C::kMaxGV;
    int nv;
    double x[kMaxGV], y[kMaxGV], z[kMaxGV];
    int16_t pl[kMaxGV][3];   // the three planes at the vertex (negative: box)
    uint8_t nb[kMaxGV][3];   // neighbour along the edge not on pl[c]
};

template <class C>
struct Work {
    static constexpr int kMaxPlanes = C::kMaxPlanes, kMaxGV = C::kMaxGV, kMaxCross = C::kMaxCross;
    Plane P[kMaxPlanes];
    uint32_t order[kMaxPlanes];   // bucket << 16 | plane
    Graph<C> G[2];
    int16_t slot[kMaxPlanes + 6];
    // crossings, as seen from each of the two faces of their edge
    double X[kMaxCross][3];
    int16_t xc_head[kMaxPlanes], xc_next[2 * kMaxCross], xc_point[2 * kMaxCross];
    uint8_t xc_start[2 * kMaxCross];
    int16_t lab_head[kMaxPlanes], lab_next[3 * kMaxGV], lab_plane[3 * kMaxGV];
    int16_t next_cross[kMaxCross];
    uint8_t active[kMaxPlanes];
};

enum Status { kOk = 0, kDegenerate = 1, kOverflow = 2 };
enum CutResult { kCutNone, kCutDone, kCutEmpty, kCutFail };

FASTSASA_EXACT_HD double jitter(int j, int a)
{
    uint64_t z = (uint64_t)j * 3 + a + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

FASTSASA_EXACT_HD double dmax(double a, double b) { return a > b ? a : b; }

/* Cut cell A by n.p <= h into B in one pass: survivors keep their order, the new vertices (one per
 * crossing edge) follow; the new face is closed through the per-plane slot table. */
template <class C>
FASTSASA_EXACT_HD CutResult graph_cut(const Graph<C> &A, Graph<C> &B, int16_t *slot, const double *n, double h, int id,
                             double &rho2, double tol)
{
    constexpr int kMaxGV = C::kMaxGV;
    double S[kMaxGV];
    int map[kMaxGV];
    const int nv = A.nv;
    double smax = -1e300;
    for (int q = 0; q < nv; ++q) {
        S[q] = n[0] * A.x[q] + n[1] * A.y[q] + n[2] * A.z[q] - h;
        smax = dmax(smax, S[q]);
    }
    if (smax < -tol) return kCutNone;
    int ns = 0;
    bool zero = false;
    for (int q = 0; q < nv; ++q) {
        zero |= fabs(S[q]) <= tol;
        map[q] = S[q] > 0 ? -1 : ns++;
    }
    if (zero) return kCutFail;
    if (ns == 0) return kCutEmpty;
    int nf = ns;
    double r2 = 0.0;
    for (int q = 0; q < nv; ++q) {
        const int t = map[q];
        if (t < 0) continue;
        B.x[t] = A.x[q]; B.y[t] = A.y[q]; B.z[t] = A.z[q];
        r2 = dmax(r2, A.x[q] * A.x[q] + A.y[q] * A.y[q] + A.z[q] * A.z[q]);
        for (int j = 0; j < 3; ++j) {
            B.pl[t][j] = A.pl[q][j];
            const int u = A.nb[q][j];
            if (map[u] >= 0) { B.nb[t][j] = (uint8_t)map[u]; continue; }
            if (nf >= kMaxGV) return kCutFail;
            const int x = nf++;
            const double f = S[q] / (S[q] - S[u]);
            B.x[x] = A.x[q] + f * (A.x[u] - A.x[q]);
            B.y[x] = A.y[q] + f * (A.y[u] - A.y[q]);
            B.z[x] = A.z[q] + f * (A.z[u] - A.z[q]);
            r2 = dmax(r2, B.x[x] * B.x[x] + B.y[x] * B.y[x] + B.z[x] * B.z[x]);
            B.pl[x][0] = A.pl[q][(j + 1) % 3];   // the edge q-u lies on these two planes
            B.pl[x][1] = A.pl[q][(j + 2) % 3];
            B.pl[x][2] = (int16_t)id;
            B.nb[x][0] = B.nb[x][1] = 0xff;
            B.nb[x][2] = (uint8_t)t;
            B.nb[t][j] = (uint8_t)x;
        }
    }
    // the two new vertices on an old plane p are joined by the edge on (p, id), which is opposite
    // each one's other old plane
    bool ok = true;
    for (int x = ns; x < nf; ++x)
        for (int c = 0; c < 2; ++c) {
            int16_t &sl = slot[B.pl[x][c] + 6];
            if (sl < 0) { sl = (int16_t)(2 * x + c); continue; }
            const int y = sl >> 1, cy = sl & 1;
            sl = -1;
            if (B.nb[x][1 - c] != 0xff || B.nb[y][1 - cy] != 0xff) ok = false;   // a plane met more than twice
            B.nb[x][1 - c] = (uint8_t)y;
            B.nb[y][1 - cy] = (uint8_t)x;
        }
    for (int x = ns; x < nf; ++x)
        for (int c = 0; c < 2; ++c) {
            int16_t &sl = slot[B.pl[x][c] + 6];
            if (sl >= 0) { ok = false; sl = -1; }
        }
    if (!ok) return kCutFail;
    B.nv = nf;
    rho2 = r2;
    return kCutDone;
}

/* Area of atom i (atoms in grid order). Returns kOk, or kDegenerate / kOverflow (no area). */
template <class C>
FASTSASA_EXACT_HD int atom_area_once(int i, const double *x, const double *R, const GridView &g, const double *est_pts,
                            Work<C> &W, double eps, double &area)
{
    constexpr int kMaxPlanes = C::kMaxPlanes, kMaxCross = C::kMaxCross;
    const double ci[3] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
    const double Ri = R[i], R2 = Ri * Ri;
    int m = 0;
    int c0[3];
    for (int a = 0; a < 3; ++a) {
        const int c = (int)((ci[a] - g.lo[a]) / g.cell);
        c0[a] = c < g.dims[a] - 1 ? c : g.dims[a] - 1;
    }
    const int K = g.K, w = 2 * K + 1;
    for (int dz = -K; dz <= K; ++dz) {
        const int cz = c0[2] + dz;
        if (cz < 0 || cz >= g.dims[2]) continue;
        for (int dy = -K; dy <= K; ++dy) {
            const int cy = c0[1] + dy;
            const int dxm = g.dxmax[(dz + K) * w + dy + K];
            if (dxm < 0 || cy < 0 || cy >= g.dims[1]) continue;
            const int row = (cz * g.dims[1] + cy) * g.dims[0];
            const int x0 = c0[0] - dxm > 0 ? c0[0] - dxm : 0;
            const int x1 = c0[0] + dxm < g.dims[0] - 1 ? c0[0] + dxm : g.dims[0] - 1;
            const int b0 = g.start[row + x0], b1 = g.start[row + x1 + 1];
            for (int j = b0; j < b1; ++j) {
                if (j == i) continue;
                double vx = x[3 * j] - ci[0], vy = x[3 * j + 1] - ci[1], vz = x[3 * j + 2] - ci[2];
                if (eps != 0.0) {
                    vx += eps * (jitter(j, 0) - jitter(i, 0));
                    vy += eps * (jitter(j, 1) - jitter(i, 1));
                    vz += eps * (jitter(j, 2) - jitter(i, 2));
                }
                const double d2 = vx * vx + vy * vy + vz * vz;
                const double reach = Ri + R[j];
                if (d2 >= reach * reach) continue;
                if (d2 < kTolRel * kTolRel * R2) return kDegenerate;
                const double d = sqrt(d2);
                const double h = (d2 + R2 - R[j] * R[j]) / (2.0 * d);
                if (h <= -Ri) { area = 0.0; return kOk; }
                if (h >= Ri) continue;
                if (m >= kMaxPlanes) return kOverflow;
                W.P[m++] = {vx / d, vy / d, vz / d, h};
            }
        }
    }
    if (m == 0) { area = 4.0 * kPi * R2; return kOk; }
    const Plane *P = W.P;
    {   // nearest planes first: counting sort into kBuckets slices of [-R, R]
        int cnt[kBuckets + 1];
        for (int b = 0; b <= kBuckets; ++b) cnt[b] = 0;
        uint8_t bk[kMaxPlanes];
        for (int k = 0; k < m; ++k) {
            int b = (int)((P[k].h + Ri) * (0.5 / Ri) * kBuckets);
            b = b < 0 ? 0 : b >= kBuckets ? kBuckets - 1 : b;
            bk[k] = (uint8_t)b;
            ++cnt[b + 1];
        }
        for (int b = 0; b < kBuckets; ++b) cnt[b + 1] += cnt[b];
        for (int k = 0; k < m; ++k) W.order[cnt[bk[k]]++] = (uint32_t)bk[k] << 16 | (uint32_t)k;
    }
    for (int k = 0; k < m + 6; ++k) W.slot[k] = -1;
    Graph<C> *Gc = &W.G[0], *Gn = &W.G[1];
    const double L = Ri * (1.0 + 1e-9);
    Gc->nv = 8;
    for (int c = 0; c < 8; ++c) {
        Gc->x[c] = c & 1 ? L : -L; Gc->y[c] = c & 2 ? L : -L; Gc->z[c] = c & 4 ? L : -L;
        Gc->pl[c][0] = c & 1 ? -2 : -1; Gc->pl[c][1] = c & 2 ? -4 : -3; Gc->pl[c][2] = c & 4 ? -6 : -5;
        Gc->nb[c][0] = (uint8_t)(c ^ 1); Gc->nb[c][1] = (uint8_t)(c ^ 2); Gc->nb[c][2] = (uint8_t)(c ^ 4);
    }
    double rho2 = 3.0 * L * L;
    for (int q = 0; q < m; ++q) {
        const int k = (int)(W.order[q] & 0xffffu);
        // every later plane has h >= lo; once lo is past the cell's farthest vertex none can cut
        const double lo = -Ri + (double)(W.order[q] >> 16) * (2.0 * Ri / kBuckets) - 1e-9 * Ri;
        if (lo >= 0.0 && lo * lo >= rho2) break;
        const double nk[3] = {P[k].nx, P[k].ny, P[k].nz};
        const CutResult r = graph_cut(*Gc, *Gn, W.slot, nk, P[k].h, k, rho2, kTolRel * Ri);
        if (r == kCutFail) return kDegenerate;
        if (r == kCutEmpty) { area = 0.0; return kOk; }
        if (r == kCutNone) continue;
        Graph<C> *t = Gc; Gc = Gn; Gn = t;
        if (rho2 < R2) { area = 0.0; return kOk; }   // cell inside the sphere: buried
    }
    const Graph<C> &G = *Gc;
    // cell edges: where one pierces the sphere the boundary passes from one face circle to the other
    for (int k = 0; k < m; ++k) { W.xc_head[k] = -1; W.lab_head[k] = -1; W.active[k] = 0; }
    double sum_ext = 0.0;
    int ncross = 0, nxc = 0, nlab = 0;
    for (int u = 0; u < G.nv; ++u) {
        for (int c = 0; c < 3; ++c) {
            const int v = G.nb[u][c];
            if (v < u) continue;
            const int pa = G.pl[u][(c + 1) % 3], pb = G.pl[u][(c + 2) % 3];
            if (pa >= 0) { W.active[pa] = 1; W.lab_plane[nlab] = (int16_t)pb; W.lab_next[nlab] = W.lab_head[pa]; W.lab_head[pa] = (int16_t)nlab++; }
            if (pb >= 0) { W.active[pb] = 1; W.lab_plane[nlab] = (int16_t)pa; W.lab_next[nlab] = W.lab_head[pb]; W.lab_head[pb] = (int16_t)nlab++; }
            const double A[3] = {G.x[u], G.y[u], G.z[u]};
            const double D[3] = {G.x[v] - A[0], G.y[v] - A[1], G.z[v] - A[2]};
            const double qa = D[0] * D[0] + D[1] * D[1] + D[2] * D[2];
            const double qb = 2.0 * (A[0] * D[0] + A[1] * D[1] + A[2] * D[2]);
            const double qc = A[0] * A[0] + A[1] * A[1] + A[2] * A[2] - R2;
            const double disc = qb * qb - 4.0 * qa * qc;
            if (disc <= 0.0) continue;
            const double sq = sqrt(disc);
            const double ts[2] = {(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)};
            // near-degenerate: an edge almost tangent to the sphere, or a cell vertex almost on it
            const double tol_t = kTolRel * Ri / sqrt(qa);
            if (ts[1] - ts[0] < tol_t && ts[1] > -tol_t && ts[0] < 1.0 + tol_t) return kDegenerate;
            for (int r = 0; r < 2; ++r) {
                const double t = ts[r];
                if (fabs(t) < tol_t || fabs(1.0 - t) < tol_t) return kDegenerate;
                if (!(t > 0.0 && t < 1.0)) continue;
                if (pa < 0 || pb < 0) return kDegenerate;   // a box edge cannot reach the sphere
                if (ncross >= kMaxCross) return kOverflow;
                double *X = W.X[ncross];
                X[0] = A[0] + t * D[0]; X[1] = A[1] + t * D[1]; X[2] = A[2] + t * D[2];
                const double na[3] = {P[pa].nx, P[pa].ny, P[pa].nz}, nb[3] = {P[pb].nx, P[pb].ny, P[pb].nz};
                const double tau[3] = {na[1] * X[2] - na[2] * X[1], na[2] * X[0] - na[0] * X[2], na[0] * X[1] - na[1] * X[0]};
                const double dn = nb[0] * tau[0] + nb[1] * tau[1] + nb[2] * tau[2];
                if (fabs(dn) < kTolRel * Ri) return kDegenerate;
                // exterior angle between the two circles at X
                const double uu[3] = {X[0] / Ri, X[1] / Ri, X[2] / Ri};
                const double da = na[0] * uu[0] + na[1] * uu[1] + na[2] * uu[2];
                const double db = nb[0] * uu[0] + nb[1] * uu[1] + nb[2] * uu[2];
                double va[3], vb[3];
                for (int e = 0; e < 3; ++e) { va[e] = na[e] - da * uu[e]; vb[e] = nb[e] - db * uu[e]; }
                const double cx[3] = {va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2], va[0] * vb[1] - va[1] * vb[0]};
                sum_ext += atan2(sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]), va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]);
                // seen from face a the arc starts here when dn < 0; from face b the sense is reversed
                W.xc_point[nxc] = (int16_t)ncross; W.xc_start[nxc] = dn < 0.0; W.xc_next[nxc] = W.xc_head[pa]; W.xc_head[pa] = (int16_t)nxc++;
                W.xc_point[nxc] = (int16_t)ncross; W.xc_start[nxc] = dn > 0.0; W.xc_next[nxc] = W.xc_head[pb]; W.xc_head[pb] = (int16_t)nxc++;
                W.next_cross[ncross] = -1;
                ++ncross;
            }
        }
    }
    double sum_h_phi = 0.0;
    bool any_arc = false;
    int full_circles = 0;
    for (int k = 0; k < m; ++k) {
        if (!W.active[k]) continue;
        const Plane &pk = P[k];
        const double nk[3] = {pk.nx, pk.ny, pk.nz};
        double e1[3];
        if (fabs(pk.nx) < 0.9) { e1[0] = 0; e1[1] = pk.nz; e1[2] = -pk.ny; }
        else { e1[0] = -pk.nz; e1[1] = 0; e1[2] = pk.nx; }
        const double l1 = 1.0 / sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        e1[0] *= l1; e1[1] *= l1; e1[2] *= l1;
        const double e2[3] = {nk[1] * e1[2] - nk[2] * e1[1], nk[2] * e1[0] - nk[0] * e1[2], nk[0] * e1[1] - nk[1] * e1[0]};
        double phi = 0.0;
        if (W.xc_head[k] < 0) {
            // circle wholly inside or outside the face: test one circle point against the face's edges
            const double rk = sqrt(R2 - pk.h * pk.h);
            const double X[3] = {pk.h * nk[0] + rk * e1[0], pk.h * nk[1] + rk * e1[1], pk.h * nk[2] + rk * e1[2]};
            bool inside = true;
            for (int s = W.lab_head[k]; s >= 0 && inside; s = W.lab_next[s]) {
                const int q = W.lab_plane[s];
                if (q < 0) continue;
                inside = P[q].nx * X[0] + P[q].ny * X[1] + P[q].nz * X[2] <= P[q].h;
            }
            if (inside) { phi = 2.0 * kPi; ++full_circles; }
        } else {
            double key[kMaxCross], cu[kMaxCross], cv[kMaxCross];
            int id[kMaxCross], st[kMaxCross];
            int nc = 0;
            for (int s = W.xc_head[k]; s >= 0; s = W.xc_next[s]) {
                const double *X = W.X[W.xc_point[s]];
                const double u = X[0] * e1[0] + X[1] * e1[1] + X[2] * e1[2];
                const double v = X[0] * e2[0] + X[1] * e2[1] + X[2] * e2[2];
                const double l = fabs(u) + fabs(v);
                const double kk = v >= 0 ? 1.0 - u / l : 3.0 + u / l;   // monotone in the angle, no trig
                int b = nc++;   // insertion by angle
                while (b > 0 && key[b - 1] > kk) {
                    key[b] = key[b - 1]; cu[b] = cu[b - 1]; cv[b] = cv[b - 1]; id[b] = id[b - 1]; st[b] = st[b - 1];
                    --b;
                }
                key[b] = kk; cu[b] = u; cv[b] = v; id[b] = W.xc_point[s]; st[b] = W.xc_start[s];
            }
            if (nc & 1) return kDegenerate;
            for (int a = 0; a < nc; ++a) {
                if (!st[a]) continue;
                const int e = a + 1 < nc ? a + 1 : 0;
                if (st[e]) return kDegenerate;
                double d = atan2(cu[a] * cv[e] - cv[a] * cu[e], cu[a] * cu[e] + cv[a] * cv[e]);
                if (d <= 0) d += 2.0 * kPi;
                phi += d;
                W.next_cross[id[a]] = (int16_t)id[e];
            }
        }
        if (phi > 0.0) any_arc = true;
        sum_h_phi += pk.h * phi;
    }
    if (!any_arc) { area = 0.0; return kOk; }
    const double a0 = Ri * sum_h_phi - R2 * sum_ext;
    const double unit = 2.0 * kPi * R2, full = 4.0 * kPi * R2;
    // boundary loops: a single loop bounds a disc, so chi = 1 without sampling
    int loops = full_circles;
    for (int c = 0; c < ncross && loops <= 1; ++c) {
        if (W.next_cross[c] < 0) continue;
        ++loops;
        int v = c;
        while (W.next_cross[v] >= 0) { const int nx = W.next_cross[v]; W.next_cross[v] = -1; v = nx; }
    }
    double chi = 1.0;
    if (loops != 1) {   // chi from a coarse point estimate (the area is in [0, 4 pi R^2])
        int inside = 0;
        for (int p = 0; p < kEstimatePoints; ++p) {
            const double *u = est_pts + 3 * p;
            bool ok = true;
            for (int k = 0; k < m && ok; ++k) ok = !W.active[k] || Ri * (P[k].nx * u[0] + P[k].ny * u[1] + P[k].nz * u[2]) <= P[k].h;
            inside += ok;
        }
        chi = round((full * inside / kEstimatePoints - a0) / unit);
    }
    const double a = a0 + unit * chi;
    area = a < 0.0 ? 0.0 : a > full ? full : a;
    return kOk;
}

/* With retries off exact degeneracies (deterministic jitter). NaN if none succeeds; *overflow is set
 * when the workspace is too small (recompute with LargeCaps). */
template <class C>
FASTSASA_EXACT_HD double atom_area(int i, const double *x, const double *R, const GridView &g, const double *est_pts, Work<C> &W,
                          int *retries, int *overflow)
{
    const double eps_ladder[6] = {0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4};
    for (int e = 0; e < 6; ++e) {
        double a;
        const int s = atom_area_once(i, x, R, g, est_pts, W, eps_ladder[e], a);
        if (s == kOk) return a;
        if (s == kOverflow) { *overflow = 1; break; }
        ++*retries;
    }
    return NAN;
}

}  // namespace fastsasa_exact

#endif
