/*
 * Exact SASA on CUDA (fastsasa_device_context_exact): the frame is sorted
 * into grid order on the host, then one warp per atom computes the areas
 * (fastsasa_exact_warp.cuh). FP64: FP64 throughout, with the CPU's jitter
 * retries. Mixed: an FP32-geometry pass with FP64 sums, then the atoms it
 * flags (near-degenerate or overflowing) again in FP64. Atoms that overflow
 * the FP64 pass's workspace are recomputed on the host with the large
 * workspace, so no atom is left without an area.
 */
#include "fastsasa_exact_cuda.h"
#include "fastsasa_exact_host.h"
#include "fastsasa_exact_warp.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <memory>
#include <new>
#include <vector>

namespace {

typedef fastsasa_exact::WarpCaps<128, 96, 128> Caps64;   /* FP64 pass */
typedef fastsasa_exact::WarpCaps<80, 64, 48> Caps32;     /* FP32 pass: overflow goes to the FP64 pass */
template <class Real> struct CapsFor { typedef Caps64 type; };
template <> struct CapsFor<float> { typedef Caps32 type; };

constexpr int kWarpsPerBlock = 2;

__constant__ double c_estimate_points[3 * fastsasa_exact::kEstimatePoints];

/* atoms list[0..count) (sorted indices; all atoms when list is null); one WarpWork per warp in
 * dynamic shared memory */
template <class Real>
__global__ void
exact_warp_kernel(int count, const int *__restrict__ list, const double *__restrict__ xs, const double *__restrict__ Rs,
                  const float4 *__restrict__ xf, fastsasa_exact::GridView g, double *__restrict__ area,
                  unsigned char *__restrict__ flag)
{
    extern __shared__ __align__(16) unsigned char smem[];
    typedef fastsasa_exact::WarpWork<typename CapsFor<Real>::type, Real> Work;
    Work &W = reinterpret_cast<Work *>(smem)[threadIdx.x >> 5];
    const int warps = gridDim.x * (blockDim.x >> 5);
    for (int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); w < count; w += warps) {
        const int q = list != nullptr ? list[w] : w;
        double a = 0.0;
        int retries = 0, over = 0;
        if (Rs[q] > 0.0) a = fastsasa_exact::warp_atom_area(q, xs, Rs, xf, g, c_estimate_points, W, &retries, &over);
        if ((threadIdx.x & 31) == 0) {
            area[q] = a;
            flag[q] = (unsigned char)over;
        }
    }
}

template <class Real>
int
kernel_setup(size_t *smem, int *blocks)
{
    static int cached_blocks = 0;
    *smem = sizeof(fastsasa_exact::WarpWork<typename CapsFor<Real>::type, Real>) * (size_t)kWarpsPerBlock;
    if (cached_blocks > 0) {
        *blocks = cached_blocks;
        return 0;
    }
    int device = 0, per_sm = 0, sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaFuncSetAttribute(exact_warp_kernel<Real>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)*smem) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, exact_warp_kernel<Real>, 32 * kWarpsPerBlock, *smem) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || per_sm <= 0) {
        return -1;
    }
    cached_blocks = per_sm * sms;
    *blocks = cached_blocks;
    return 0;
}

template <class T>
bool
ensure(T **ptr, size_t *capacity, size_t count)
{
    if (count <= *capacity && *ptr != nullptr) return true;
    cudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    const size_t want = count + count / 4 + 64;
    if (cudaMalloc(reinterpret_cast<void **>(ptr), sizeof(T) * want) != cudaSuccess) return false;
    *capacity = want;
    return true;
}

}  // namespace

struct fastsasa_exact_cuda {
    double *d_xs = nullptr, *d_Rs = nullptr, *d_area = nullptr;
    float4 *d_xf = nullptr;
    int *d_start = nullptr, *d_dxmax = nullptr, *d_list = nullptr;
    unsigned char *d_flag = nullptr;
    size_t xs_cap = 0, Rs_cap = 0, area_cap = 0, xf_cap = 0, start_cap = 0, dxmax_cap = 0, list_cap = 0, flag_cap = 0;
    bool points_uploaded = false;
    fastsasa_exact::SortedFrame frame;
    std::vector<float4> xf;
    std::vector<double> area;
    std::vector<unsigned char> flag;
    std::vector<int> list;
};

extern "C" void
fastsasa_exact_cuda_free(struct fastsasa_exact_cuda *state)
{
    if (state == nullptr) return;
    cudaFree(state->d_xs);
    cudaFree(state->d_Rs);
    cudaFree(state->d_area);
    cudaFree(state->d_xf);
    cudaFree(state->d_start);
    cudaFree(state->d_dxmax);
    cudaFree(state->d_list);
    cudaFree(state->d_flag);
    delete state;
}

static int
exact_cuda_run(fastsasa_exact_cuda *s, cudaStream_t stream, int n, const double *x, const double *y, const double *z,
               const double *radii, const int *centers, int n_centers, int mixed, double *sasa)
{
    fastsasa_exact::SortedFrame &f = s->frame;
    if (!fastsasa_exact::sort_frame(n, x, y, z, radii, f)) return FASTSASA_INVALID_ARGUMENT;
    const int count = centers != nullptr ? n_centers : n;
    if (count == 0) return FASTSASA_SUCCESS;
    const size_t un = (size_t)n;
    if (!ensure(&s->d_xs, &s->xs_cap, 3u * un) || !ensure(&s->d_Rs, &s->Rs_cap, un) ||
        !ensure(&s->d_area, &s->area_cap, un) || !ensure(&s->d_flag, &s->flag_cap, un) ||
        !ensure(&s->d_list, &s->list_cap, un) || !ensure(&s->d_start, &s->start_cap, f.g.start.size()) ||
        !ensure(&s->d_dxmax, &s->dxmax_cap, f.g.dxmax.size()) || (mixed && !ensure(&s->d_xf, &s->xf_cap, un))) {
        return FASTSASA_MEMORY_ERROR;
    }
    if (!s->points_uploaded) {
        double est[3 * fastsasa_exact::kEstimatePoints];
        fastsasa_exact::estimate_points(est);
        if (cudaMemcpyToSymbol(c_estimate_points, est, sizeof(est)) != cudaSuccess) return FASTSASA_CUDA_ERROR;
        s->points_uploaded = true;
    }
    /* the work list holds sorted indices */
    const int *d_list = nullptr;
    if (centers != nullptr) {
        std::vector<int> sorted_of(un);
        for (int q = 0; q < n; ++q) sorted_of[(size_t)f.orig[(size_t)q]] = q;
        s->list.resize((size_t)n_centers);
        for (int c = 0; c < n_centers; ++c) {
            if (centers[c] < 0 || centers[c] >= n) return FASTSASA_INVALID_ARGUMENT;
            s->list[(size_t)c] = sorted_of[(size_t)centers[c]];
        }
    }
    cudaError_t e = cudaMemcpyAsync(s->d_xs, f.xs.data(), sizeof(double) * 3u * un, cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s->d_Rs, f.Rs.data(), sizeof(double) * un, cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s->d_start, f.g.start.data(), sizeof(int) * f.g.start.size(), cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess) e = cudaMemcpyAsync(s->d_dxmax, f.g.dxmax.data(), sizeof(int) * f.g.dxmax.size(), cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess && centers != nullptr) {
        e = cudaMemcpyAsync(s->d_list, s->list.data(), sizeof(int) * (size_t)n_centers, cudaMemcpyHostToDevice, stream);
        d_list = s->d_list;
    }
    if (e == cudaSuccess && mixed) {
        /* box-local float shadow (x, y, z, R) for the FP32 neighbour prefilter */
        s->xf.resize(un);
        for (int q = 0; q < n; ++q) {
            s->xf[(size_t)q] = make_float4((float)(f.xs[3u * q] - f.g.lo[0]), (float)(f.xs[3u * q + 1u] - f.g.lo[1]),
                                           (float)(f.xs[3u * q + 2u] - f.g.lo[2]), (float)f.Rs[(size_t)q]);
        }
        e = cudaMemcpyAsync(s->d_xf, s->xf.data(), sizeof(float4) * un, cudaMemcpyHostToDevice, stream);
    }
    if (e != cudaSuccess) return FASTSASA_CUDA_ERROR;
    fastsasa_exact::GridView gv = f.g.view();
    gv.start = s->d_start;
    gv.dxmax = s->d_dxmax;
    size_t smem64 = 0, smem32 = 0;
    int blocks64 = 0, blocks32 = 0;
    if (kernel_setup<double>(&smem64, &blocks64) != 0 || (mixed && kernel_setup<float>(&smem32, &blocks32) != 0)) {
        return FASTSASA_CUDA_ERROR;
    }
    s->flag.resize(un);
    if (mixed) {
        exact_warp_kernel<float><<<blocks32, 32 * kWarpsPerBlock, smem32, stream>>>(count, d_list, s->d_xs, s->d_Rs, s->d_xf, gv,
                                                                                s->d_area, s->d_flag);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(s->flag.data(), s->d_flag, un, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return FASTSASA_CUDA_ERROR;
        }
        std::vector<int> again;
        for (int w = 0; w < count; ++w) {
            const int q = centers != nullptr ? s->list[(size_t)w] : w;
            if (s->flag[(size_t)q]) again.push_back(q);
        }
        if (!again.empty()) {
            if (cudaMemcpyAsync(s->d_list, again.data(), sizeof(int) * again.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                return FASTSASA_CUDA_ERROR;
            }
            exact_warp_kernel<double><<<blocks64, 32 * kWarpsPerBlock, smem64, stream>>>((int)again.size(), s->d_list, s->d_xs,
                                                                                     s->d_Rs, nullptr, gv, s->d_area, s->d_flag);
            if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) return FASTSASA_CUDA_ERROR;
        }
    } else {
        exact_warp_kernel<double><<<blocks64, 32 * kWarpsPerBlock, smem64, stream>>>(count, d_list, s->d_xs, s->d_Rs, nullptr, gv,
                                                                                 s->d_area, s->d_flag);
        if (cudaGetLastError() != cudaSuccess) return FASTSASA_CUDA_ERROR;
    }
    s->area.resize(un);
    if (cudaMemcpyAsync(s->area.data(), s->d_area, sizeof(double) * un, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(s->flag.data(), s->d_flag, un, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return FASTSASA_CUDA_ERROR;
    }
    std::unique_ptr<fastsasa_exact::Work<fastsasa_exact::LargeCaps> > big;
    const fastsasa_exact::GridView hv = f.g.view();
    double est[3 * fastsasa_exact::kEstimatePoints];
    fastsasa_exact::estimate_points(est);
    for (int w = 0; w < count; ++w) {
        const int q = centers != nullptr ? s->list[(size_t)w] : w;
        double a = s->area[(size_t)q];
        if (s->flag[(size_t)q]) {
            if (!big) big.reset(new fastsasa_exact::Work<fastsasa_exact::LargeCaps>());
            int retries = 0, over = 0;
            a = fastsasa_exact::atom_area(q, f.xs.data(), f.Rs.data(), hv, est, *big, &retries, &over);
        }
        if (std::isnan(a)) return FASTSASA_DEGENERATE_GEOMETRY;
        sasa[f.orig[(size_t)q]] = a;
    }
    return FASTSASA_SUCCESS;
}

extern "C" int
fastsasa_exact_cuda_run(struct fastsasa_exact_cuda **state, cudaStream_t stream, int n_atoms, const double *x,
                        const double *y, const double *z, const double *expanded_radii, const int *centers,
                        int n_centers, int mixed, double *sasa)
{
    if (state == nullptr || n_atoms < 0 || sasa == nullptr ||
        (n_atoms > 0 && (x == nullptr || y == nullptr || z == nullptr || expanded_radii == nullptr)) ||
        (centers != nullptr && n_centers < 0)) {
        return FASTSASA_INVALID_ARGUMENT;
    }
    try {
        if (*state == nullptr) *state = new fastsasa_exact_cuda();
        return exact_cuda_run(*state, stream, n_atoms, x, y, z, expanded_radii, centers, n_centers, mixed != 0, sasa);
    } catch (const std::bad_alloc &) {
        return FASTSASA_MEMORY_ERROR;
    } catch (...) {
        return FASTSASA_INVALID_ARGUMENT;
    }
}
