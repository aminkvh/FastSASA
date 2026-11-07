/*
 * Exact SASA on CUDA, one warp per atom: the method and plane order of
 * fastsasa_exact_core.h, with the 32 lanes sharing the atom's work:
 * neighbour search, the bucket sort (lane = bucket), each cut's vertex pass
 * (survivor and new-vertex slots from warp prefix sums, so the cell matches
 * the CPU's vertex for vertex), edge crossings, and one face per lane for the
 * arcs. The cell lives in shared memory. Real = float is the mixed-precision
 * pass (FP32 geometry, FP64 sums); it flags near-degenerate or overflowing
 * atoms for the FP64 pass instead of retrying.
 */
#ifndef FASTSASA_EXACT_WARP_CUH
#define FASTSASA_EXACT_WARP_CUH
#include "fastsasa_exact_core.h"

namespace fastsasa_exact {

template <int kMP_, int kGV_, int kMC_>
struct WarpCaps {
    static constexpr int kMP = kMP_, kGV = kGV_, kMC = kMC_;
    static constexpr int kVPL = (kGV + 31) / 32;   // vertices per lane
};

template <class C, class Real = double>
struct WarpWork {
    Real nx[C::kMP], ny[C::kMP], nz[C::kMP], h[C::kMP];
    Real x[2][C::kGV], y[2][C::kGV], z[2][C::kGV];
    Real S[C::kGV];
    Real X[C::kMC][3];
    int16_t pl[2][C::kGV][3];
    uint8_t nb[2][C::kGV][3];
    int16_t order[C::kMP];
    int16_t slot[C::kMP + 6];
    int16_t xpa[C::kMC], xpb[C::kMC], next_cross[C::kMC];
    uint8_t xstart[C::kMC];   // 1: the arc on face xpa starts here
    uint8_t active[C::kMP];
    int nv[2];
    int m, ncross;
};

template <class T>
__device__ __forceinline__ T warp_sum(T v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <class T>
__device__ __forceinline__ T warp_max(T v)
{
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int warp_isum(int v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int warp_excl_scan(int v, int lane)
{
    int s = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, s, o);
        if (lane >= o) s += t;
    }
    return s - v;
}

/* One attempt for atom i. Returns kOk / kDegenerate / kOverflow; area valid on kOk (all lanes). */
template <class Real> __device__ __forceinline__ Real tol_rel();
template <> __device__ __forceinline__ double tol_rel<double>() { return kTolRel; }
/* FP32 pass: anything within 4e-6 R of a degeneracy goes to the FP64 pass */
template <> __device__ __forceinline__ float tol_rel<float>() { return 4e-6f; }

template <class C, class Real>
__device__ int warp_atom_once(int i, const double *__restrict__ x, const double *__restrict__ R, const float4 *__restrict__ xf,
                              const GridView &g, const double *__restrict__ est_pts, WarpWork<C, Real> &W, double eps,
                              double &area)
{
    const float4 xfi = xf != nullptr ? xf[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    constexpr int kMP = C::kMP, kGV = C::kGV, kMC = C::kMC, kVPL = C::kVPL;
    const int lane = threadIdx.x & 31;
    const unsigned below = (1u << lane) - 1u;
    const double ci[3] = {x[3 * i], x[3 * i + 1], x[3 * i + 2]};
    const double Rid = R[i];
    const Real Ri = (Real)Rid, R2 = Ri * Ri;
    const Real kTol = tol_rel<Real>();
    // ---- neighbour planes (in the CPU's candidate order: ballot compaction keeps lane order)
    int m = 0;
    bool degenerate = false, buried = false, overflow = false;
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
            for (int j0 = b0; j0 < b1; j0 += 32) {
                const int j = j0 + lane;
                bool keep = false;
                Real pnx = 0, pny = 0, pnz = 0, ph = 0;
                if (j < b1 && j != i) {
                    Real vx, vy, vz, Rj;
                    bool near;
                    if (sizeof(Real) == 4 && xf != nullptr) {
                        // FP32: prefilter on box-local floats (with a margin), exact vector only if kept
                        const float4 pj = xf[j];
                        const float fx = pj.x - xfi.x, fy = pj.y - xfi.y, fz = pj.z - xfi.z;
                        const float rr = xfi.w + pj.w;
                        near = fx * fx + fy * fy + fz * fz < rr * rr * 1.0001f + 1e-4f;
                        Rj = (Real)pj.w;
                    } else {
                        near = true;
                        Rj = (Real)R[j];
                    }
                    Real d2 = 0;
                    if (near) {
                        double vxd = x[3 * j] - ci[0], vyd = x[3 * j + 1] - ci[1], vzd = x[3 * j + 2] - ci[2];
                        if (eps != 0.0) {
                            vxd += eps * (jitter(j, 0) - jitter(i, 0));
                            vyd += eps * (jitter(j, 1) - jitter(i, 1));
                            vzd += eps * (jitter(j, 2) - jitter(i, 2));
                        }
                        vx = (Real)vxd; vy = (Real)vyd; vz = (Real)vzd;
                        d2 = vx * vx + vy * vy + vz * vz;
                    }
                    const Real reach = Ri + Rj;
                    if (near && d2 < reach * reach) {
                        if (d2 < kTol * kTol * R2) degenerate = true;
                        else {
                            const Real d = sqrt(d2);
                            const Real hh = (d2 + R2 - Rj * Rj) / ((Real)2.0 * d);
                            if (hh <= -Ri) buried = true;
                            else if (hh < Ri) { keep = true; pnx = vx / d; pny = vy / d; pnz = vz / d; ph = hh; }
                        }
                    }
                }
                const unsigned bal = __ballot_sync(0xffffffffu, keep);
                if (keep) {
                    const int slotq = m + __popc(bal & below);
                    if (slotq < kMP) { W.nx[slotq] = pnx; W.ny[slotq] = pny; W.nz[slotq] = pnz; W.h[slotq] = ph; }
                }
                m += __popc(bal);
            }
        }
    }
    if (__any_sync(0xffffffffu, degenerate)) return kDegenerate;
    if (__any_sync(0xffffffffu, buried)) { area = 0.0; return kOk; }
    if (m > kMP) return kOverflow;
    if (m == 0) { area = 4.0 * kPi * Rid * Rid; return kOk; }
    __syncwarp();
    // ---- order: counting sort into 32 h-buckets, lane = bucket, index order inside a bucket
    static_assert(kBuckets == 32, "one lane per bucket");
    {
        int cnt = 0;
        for (int k = 0; k < m; ++k) {
            int b = (int)((W.h[k] + Ri) * ((Real)0.5 / Ri) * kBuckets);
            b = b < 0 ? 0 : b >= kBuckets ? kBuckets - 1 : b;
            cnt += b == lane;
        }
        int pos = warp_excl_scan(cnt, lane);
        for (int k = 0; k < m && cnt > 0; ++k) {
            int b = (int)((W.h[k] + Ri) * ((Real)0.5 / Ri) * kBuckets);
            b = b < 0 ? 0 : b >= kBuckets ? kBuckets - 1 : b;
            if (b == lane) { W.order[pos++] = (int16_t)k; --cnt; }
        }
    }
    for (int k = lane; k < m + 6; k += 32) W.slot[k] = -1;
    const Real L = Ri * ((Real)1 + 64 * kTol);
    if (lane < 8) {
        const int c = lane;
        W.x[0][c] = c & 1 ? L : -L; W.y[0][c] = c & 2 ? L : -L; W.z[0][c] = c & 4 ? L : -L;
        W.pl[0][c][0] = c & 1 ? -2 : -1; W.pl[0][c][1] = c & 2 ? -4 : -3; W.pl[0][c][2] = c & 4 ? -6 : -5;
        W.nb[0][c][0] = (uint8_t)(c ^ 1); W.nb[0][c][1] = (uint8_t)(c ^ 2); W.nb[0][c][2] = (uint8_t)(c ^ 4);
    }
    int cur = 0, nv = 8;
    Real rho2 = (Real)3.0 * L * L;
    const Real tol = kTol * Ri;
    __syncwarp();
    // ---- cuts
    for (int q = 0; q < m; ++q) {
        const int k = W.order[q];
        const Real hk = W.h[k];
        int bk = (int)((hk + Ri) * ((Real)0.5 / Ri) * kBuckets);
        bk = bk < 0 ? 0 : bk >= kBuckets ? kBuckets - 1 : bk;
        const Real lo = -Ri + (Real)bk * (2 * Ri / kBuckets) - 8 * kTol * Ri;
        if (lo >= (Real)0.0 && lo * lo >= rho2) break;
        const Real n0 = W.nx[k], n1 = W.ny[k], n2 = W.nz[k];
        const int A = cur, B = cur ^ 1;
        Real s_own[kVPL];
        Real smax = -(Real)1e30;
        bool zero = false;
        for (int t = 0; t < kVPL; ++t) {
            const int v = lane + 32 * t;
            s_own[t] = -(Real)1e30;
            if (v < nv) {
                const Real s = n0 * W.x[A][v] + n1 * W.y[A][v] + n2 * W.z[A][v] - hk;
                s_own[t] = s;
                W.S[v] = s;
                smax = fmax(smax, s);
                zero |= fabs(s) <= tol;
            }
        }
        smax = warp_max(smax);
        if (smax < -tol) continue;
        if (__any_sync(0xffffffffu, zero)) return kDegenerate;
        __syncwarp();
        // survivor slots and new-vertex slots, in vertex order (as on the CPU)
        int map_own[kVPL], newbase[kVPL], newcnt[kVPL];
        int ns = 0;
        for (int t = 0; t < kVPL; ++t) {
            const int v = lane + 32 * t;
            const bool surv = v < nv && s_own[t] <= 0;
            const unsigned bal = __ballot_sync(0xffffffffu, surv);
            map_own[t] = surv ? ns + __popc(bal & below) : -1;
            ns += __popc(bal);
        }
        if (ns == 0) { area = 0.0; return kOk; }
        int nnew = 0;
        for (int t = 0; t < kVPL; ++t) {
            const int v = lane + 32 * t;
            int c = 0;
            if (map_own[t] >= 0)
                for (int j = 0; j < 3; ++j) c += W.S[W.nb[A][v][j]] > 0;
            newcnt[t] = c;
            const int pre = warp_excl_scan(c, lane);
            newbase[t] = ns + nnew + pre;
            nnew += warp_isum(c);
        }
        const int nf = ns + nnew;
        if (nf > kGV) return kOverflow;
        // map of every old vertex, for the survivors' neighbour pointers
        int *mapS = reinterpret_cast<int *>(W.X);   // scratch: X is unused during the cuts
        for (int t = 0; t < kVPL; ++t) {
            const int v = lane + 32 * t;
            if (v < nv) mapS[v] = map_own[t];
        }
        __syncwarp();
        Real r2 = 0;
        for (int t = 0; t < kVPL; ++t) {
            const int v = lane + 32 * t;
            const int tt = map_own[t];
            if (tt < 0) continue;
            const Real vxA = W.x[A][v], vyA = W.y[A][v], vzA = W.z[A][v];
            W.x[B][tt] = vxA; W.y[B][tt] = vyA; W.z[B][tt] = vzA;
            r2 = fmax(r2, vxA * vxA + vyA * vyA + vzA * vzA);
            int xn = newbase[t];
            for (int j = 0; j < 3; ++j) {
                W.pl[B][tt][j] = W.pl[A][v][j];
                const int u = W.nb[A][v][j];
                const int mu = mapS[u];
                if (mu >= 0) { W.nb[B][tt][j] = (uint8_t)mu; continue; }
                const int xv = xn++;
                const Real sq = s_own[t], su = W.S[u];
                const Real f = sq / (sq - su);
                W.x[B][xv] = vxA + f * (W.x[A][u] - vxA);
                W.y[B][xv] = vyA + f * (W.y[A][u] - vyA);
                W.z[B][xv] = vzA + f * (W.z[A][u] - vzA);
                r2 = fmax(r2, W.x[B][xv] * W.x[B][xv] + W.y[B][xv] * W.y[B][xv] + W.z[B][xv] * W.z[B][xv]);
                W.pl[B][xv][0] = W.pl[A][v][(j + 1) % 3];
                W.pl[B][xv][1] = W.pl[A][v][(j + 2) % 3];
                W.pl[B][xv][2] = (int16_t)k;
                W.nb[B][xv][0] = W.nb[B][xv][1] = 0xff;
                W.nb[B][xv][2] = (uint8_t)tt;
                W.nb[B][tt][j] = (uint8_t)xv;
            }
        }
        __syncwarp();
        // close the new face (short and sequential: lane 0)
        int ok = 1;
        if (lane == 0) {
            for (int xv = ns; xv < nf; ++xv)
                for (int c = 0; c < 2; ++c) {
                    int16_t &sl = W.slot[W.pl[B][xv][c] + 6];
                    if (sl < 0) { sl = (int16_t)(2 * xv + c); continue; }
                    const int y = sl >> 1, cy = sl & 1;
                    sl = -1;
                    if (W.nb[B][xv][1 - c] != 0xff || W.nb[B][y][1 - cy] != 0xff) ok = 0;
                    W.nb[B][xv][1 - c] = (uint8_t)y;
                    W.nb[B][y][1 - cy] = (uint8_t)xv;
                }
            for (int xv = ns; xv < nf; ++xv)
                for (int c = 0; c < 2; ++c) {
                    int16_t &sl = W.slot[W.pl[B][xv][c] + 6];
                    if (sl >= 0) { ok = 0; sl = -1; }
                }
        }
        ok = __shfl_sync(0xffffffffu, ok, 0);
        if (!ok) return kDegenerate;
        rho2 = warp_max(r2);
        cur = B;
        nv = nf;
        __syncwarp();
        if (rho2 < R2) { area = 0.0; return kOk; }   // cell inside the sphere: buried
    }
    // ---- edges: crossings with the sphere (any order; ids are per-atom)
    const int G = cur;
    for (int k = lane; k < m; k += 32) W.active[k] = 0;
    if (lane == 0) W.ncross = 0;
    __syncwarp();
    double ext_part = 0.0;
    for (int t = 0; t < kVPL; ++t) {
        const int u = lane + 32 * t;
        if (u >= nv) continue;
        for (int c = 0; c < 3; ++c) {
            const int v = W.nb[G][u][c];
            if (v < u) continue;
            const int pa = W.pl[G][u][(c + 1) % 3], pb = W.pl[G][u][(c + 2) % 3];
            if (pa >= 0) W.active[pa] = 1;
            if (pb >= 0) W.active[pb] = 1;
            const Real Ax = W.x[G][u], Ay = W.y[G][u], Az = W.z[G][u];
            const Real Dx = W.x[G][v] - Ax, Dy = W.y[G][v] - Ay, Dz = W.z[G][v] - Az;
            const Real qa = Dx * Dx + Dy * Dy + Dz * Dz;
            const Real qb = (Real)2.0 * (Ax * Dx + Ay * Dy + Az * Dz);
            const Real qc = Ax * Ax + Ay * Ay + Az * Az - R2;
            const Real disc = qb * qb - (Real)4.0 * qa * qc;
            if (disc <= (Real)0.0) continue;
            const Real sq = sqrt(disc);
            const Real ts[2] = {(-qb - sq) / ((Real)2.0 * qa), (-qb + sq) / ((Real)2.0 * qa)};
            const Real tol_t = kTol * Ri / sqrt(qa);
            if (ts[1] - ts[0] < tol_t && ts[1] > -tol_t && ts[0] < (Real)1.0 + tol_t) { degenerate = true; continue; }
            for (int r = 0; r < 2; ++r) {
                const Real tr = ts[r];
                if (fabs(tr) < tol_t || fabs((Real)1.0 - tr) < tol_t) { degenerate = true; continue; }
                if (!(tr > (Real)0.0 && tr < (Real)1.0)) continue;
                if (pa < 0 || pb < 0) { degenerate = true; continue; }
                const Real X0 = Ax + tr * Dx, X1 = Ay + tr * Dy, X2 = Az + tr * Dz;
                const Real na[3] = {W.nx[pa], W.ny[pa], W.nz[pa]}, nbv[3] = {W.nx[pb], W.ny[pb], W.nz[pb]};
                const Real tau[3] = {na[1] * X2 - na[2] * X1, na[2] * X0 - na[0] * X2, na[0] * X1 - na[1] * X0};
                const Real dn = nbv[0] * tau[0] + nbv[1] * tau[1] + nbv[2] * tau[2];
                if (fabs(dn) < kTol * Ri) { degenerate = true; continue; }
                const Real uu[3] = {X0 / Ri, X1 / Ri, X2 / Ri};
                const Real da = na[0] * uu[0] + na[1] * uu[1] + na[2] * uu[2];
                const Real db = nbv[0] * uu[0] + nbv[1] * uu[1] + nbv[2] * uu[2];
                Real va[3], vb[3];
                for (int e = 0; e < 3; ++e) { va[e] = na[e] - da * uu[e]; vb[e] = nbv[e] - db * uu[e]; }
                const Real cx[3] = {va[1] * vb[2] - va[2] * vb[1], va[2] * vb[0] - va[0] * vb[2], va[0] * vb[1] - va[1] * vb[0]};
                ext_part += (double)atan2(sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]), va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]);
                const int id = atomicAdd(&W.ncross, 1);
                if (id >= kMC) { overflow = true; continue; }
                W.X[id][0] = X0; W.X[id][1] = X1; W.X[id][2] = X2;
                W.xpa[id] = (int16_t)pa; W.xpb[id] = (int16_t)pb; W.xstart[id] = dn < (Real)0.0;
                W.next_cross[id] = -1;
            }
        }
    }
    if (__any_sync(0xffffffffu, degenerate)) return kDegenerate;
    if (__any_sync(0xffffffffu, overflow)) return kOverflow;
    const double sum_ext = warp_sum(ext_part);
    __syncwarp();
    const int ncross = W.ncross;
    // ---- faces: one per lane
    constexpr int kFaceCross = 32;
    double hphi_part = 0.0;
    int full_part = 0;
    bool arc_part = false;
    for (int k = lane; k < m; k += 32) {
        if (!W.active[k]) continue;
        const Real nk[3] = {W.nx[k], W.ny[k], W.nz[k]};
        const Real hk = W.h[k];
        Real e1[3];
        if (fabs(nk[0]) < (Real)0.9) { e1[0] = 0; e1[1] = nk[2]; e1[2] = -nk[1]; }
        else { e1[0] = -nk[2]; e1[1] = 0; e1[2] = nk[0]; }
        const Real l1 = (Real)1.0 / sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        e1[0] *= l1; e1[1] *= l1; e1[2] *= l1;
        const Real e2[3] = {nk[1] * e1[2] - nk[2] * e1[1], nk[2] * e1[0] - nk[0] * e1[2], nk[0] * e1[1] - nk[1] * e1[0]};
        Real key[kFaceCross], cu[kFaceCross], cv[kFaceCross];
        int idl[kFaceCross], st[kFaceCross];
        int nc = 0;
        for (int s = 0; s < ncross; ++s) {
            const int a = W.xpa[s], b = W.xpb[s];
            if (a != k && b != k) continue;
            if (nc >= kFaceCross) { overflow = true; break; }
            const Real u = W.X[s][0] * e1[0] + W.X[s][1] * e1[1] + W.X[s][2] * e1[2];
            const Real v = W.X[s][0] * e2[0] + W.X[s][1] * e2[1] + W.X[s][2] * e2[2];
            const Real l = fabs(u) + fabs(v);
            const Real kk = v >= 0 ? (Real)1.0 - u / l : (Real)3.0 + u / l;
            const int stv = a == k ? W.xstart[s] : !W.xstart[s];
            int p = nc++;
            while (p > 0 && key[p - 1] > kk) {
                key[p] = key[p - 1]; cu[p] = cu[p - 1]; cv[p] = cv[p - 1]; idl[p] = idl[p - 1]; st[p] = st[p - 1];
                --p;
            }
            key[p] = kk; cu[p] = u; cv[p] = v; idl[p] = s; st[p] = stv;
        }
        double phi = 0.0;
        if (nc == 0) {
            // circle wholly inside or outside the face: test one circle point against the face's edges
            const Real rk = sqrt(R2 - hk * hk);
            const Real P0 = hk * nk[0] + rk * e1[0], P1 = hk * nk[1] + rk * e1[1], P2 = hk * nk[2] + rk * e1[2];
            bool inside = true;
            for (int u = 0; u < nv && inside; ++u)
                for (int c = 0; c < 3 && inside; ++c) {
                    const int v = W.nb[G][u][c];
                    if (v < u) continue;
                    const int pa = W.pl[G][u][(c + 1) % 3], pb = W.pl[G][u][(c + 2) % 3];
                    const int qo = pa == k ? pb : pb == k ? pa : -1;
                    if (qo < 0) continue;
                    inside = W.nx[qo] * P0 + W.ny[qo] * P1 + W.nz[qo] * P2 <= W.h[qo];
                }
            if (inside) { phi = 2.0 * kPi; ++full_part; }
        } else {
            if (nc & 1) degenerate = true;
            for (int a = 0; a < nc && !degenerate; ++a) {
                if (!st[a]) continue;
                const int e = a + 1 < nc ? a + 1 : 0;
                if (st[e]) { degenerate = true; break; }
                double d = (double)atan2(cu[a] * cv[e] - cv[a] * cu[e], cu[a] * cu[e] + cv[a] * cv[e]);
                if (d <= 0) d += 2.0 * kPi;
                phi += d;
                W.next_cross[idl[a]] = (int16_t)idl[e];
            }
        }
        if (phi > 0.0) arc_part = true;
        hphi_part += (double)hk * phi;
    }
    if (__any_sync(0xffffffffu, degenerate)) return kDegenerate;
    if (__any_sync(0xffffffffu, overflow)) return kOverflow;
    const bool any_arc = __any_sync(0xffffffffu, arc_part);
    if (!any_arc) { area = 0.0; return kOk; }
    const double sum_h_phi = warp_sum(hphi_part);
    const int full_circles = warp_isum(full_part);
    __syncwarp();
    const double a0 = Rid * sum_h_phi - Rid * Rid * sum_ext;
    const double unit = 2.0 * kPi * Rid * Rid, full = 4.0 * kPi * Rid * Rid;
    int loops = full_circles;
    if (lane == 0)
        for (int c = 0; c < ncross && loops <= 1; ++c) {
            if (W.next_cross[c] < 0) continue;
            ++loops;
            int v = c;
            while (W.next_cross[v] >= 0) { const int nx = W.next_cross[v]; W.next_cross[v] = -1; v = nx; }
        }
    loops = __shfl_sync(0xffffffffu, loops, 0);
    double chi = 1.0;
    if (loops != 1) {
        int inside = 0;
        for (int p = lane; p < kEstimatePoints; p += 32) {
            const Real u0 = (Real)est_pts[3 * p], u1 = (Real)est_pts[3 * p + 1], u2 = (Real)est_pts[3 * p + 2];
            bool ok = true;
            for (int k = 0; k < m && ok; ++k) ok = !W.active[k] || Ri * (W.nx[k] * u0 + W.ny[k] * u1 + W.nz[k] * u2) <= W.h[k];
            inside += ok;
        }
        inside = warp_isum(inside);
        chi = round((full * inside / kEstimatePoints - a0) / unit);
    }
    const double a = a0 + unit * chi;
    area = a < 0.0 ? 0.0 : a > full ? full : a;
    return kOk;
}

/* FP64: retries off exact degeneracies with a tiny jitter. FP32 (try_only): one attempt; a degenerate or
 * overflowing atom returns NaN with *overflow set, for the FP64 pass. */
template <class C, class Real>
__device__ double warp_atom_area(int i, const double *x, const double *R, const float4 *xf, const GridView &g,
                                 const double *est_pts, WarpWork<C, Real> &W, int *retries, int *overflow)
{
    const double eps_ladder[6] = {0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4};
    const int tries = sizeof(Real) == 4 ? 1 : 6;
    for (int e = 0; e < tries; ++e) {
        double a = 0.0;
        __syncwarp();
        const int s = warp_atom_once(i, x, R, xf, g, est_pts, W, eps_ladder[e], a);
        if (s == kOk) return a;
        if (s == kOverflow) { *overflow = 1; break; }
        ++*retries;
    }
    if (sizeof(Real) == 4) *overflow = 1;
    return NAN;
}

}  // namespace fastsasa_exact

#endif
