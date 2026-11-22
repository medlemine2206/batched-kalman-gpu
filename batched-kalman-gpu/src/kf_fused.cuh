// kf_fused.cuh — "one thread = one filter", the whole step in a single kernel.
//
// Three memory layouts, same arithmetic. They exist to show that for this problem
// (arithmetic intensity ~5 FLOP/byte) the bottleneck is DRAM traffic, so the data
// layout matters more than the math:
//
//   AoS     P[f][36]  : thread f reads 36 consecutive floats -> neighbouring threads
//                       are 144 bytes apart, each warp load touches 32 cache lines.
//   SoA     P[36][f]  : at every load, 32 threads read 32 consecutive floats ->
//                       one 128-byte transaction per warp (fully coalesced).
//   SoASym  P[21][f]  : SoA + only the upper triangle (P is symmetric) -> 42% fewer bytes.
#pragma once

#include "kf_device.cuh"

namespace kf {

enum class Layout { AoS, SoA, SoASym };

template <Layout L>
__global__ void __launch_bounds__(128)
kf_fused_kernel(float* __restrict__ X, float* __restrict__ P, const float* __restrict__ Z, int filters) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= filters) return;

    float x[N], p[NN], z[M];
    if constexpr (L == Layout::AoS) {
#pragma unroll
        for (int i = 0; i < N; ++i) x[i] = X[(size_t)f * N + i];
#pragma unroll
        for (int i = 0; i < NN; ++i) p[i] = P[(size_t)f * NN + i];
#pragma unroll
        for (int i = 0; i < M; ++i) z[i] = Z[(size_t)f * M + i];
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) x[i] = X[(size_t)i * filters + f];
#pragma unroll
        for (int i = 0; i < M; ++i) z[i] = Z[(size_t)i * filters + f];
        if constexpr (L == Layout::SoA) {
#pragma unroll
            for (int i = 0; i < NN; ++i) p[i] = P[(size_t)i * filters + f];
        } else {
#pragma unroll
            for (int c = 0; c < N; ++c)
#pragma unroll
                for (int r = 0; r <= c; ++r) {
                    const float v = P[(size_t)sym_idx(r, c) * filters + f];
                    p[idx(r, c, N)] = v;
                    p[idx(c, r, N)] = v;
                }
        }
    }

    kf_step_regs(x, p, z);

    if constexpr (L == Layout::AoS) {
#pragma unroll
        for (int i = 0; i < N; ++i) X[(size_t)f * N + i] = x[i];
#pragma unroll
        for (int i = 0; i < NN; ++i) P[(size_t)f * NN + i] = p[i];
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) X[(size_t)i * filters + f] = x[i];
        if constexpr (L == Layout::SoA) {
#pragma unroll
            for (int i = 0; i < NN; ++i) P[(size_t)i * filters + f] = p[i];
        } else {
#pragma unroll
            for (int c = 0; c < N; ++c)
#pragma unroll
                for (int r = 0; r <= c; ++r) P[(size_t)sym_idx(r, c) * filters + f] = p[idx(r, c, N)];
        }
    }
}

template <Layout L>
inline void launch_fused(float* X, float* P, const float* Z, int filters, cudaStream_t st) {
    const int threads = 128;
    const int blocks  = (filters + threads - 1) / threads;
    kf_fused_kernel<L><<<blocks, threads, 0, st>>>(X, P, Z, filters);
}

}  // namespace kf
