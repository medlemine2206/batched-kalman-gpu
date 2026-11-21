// kf_dx.cuh — block-level fused kernel built on cuBLASDx (and optionally cuSolverDx).
//
// One thread block = B filters. Everything for those B filters stays in shared memory
// for the whole step: one global read of (x, P, z), one global write of (x, P).
//
// Key trick — "shared model batching":
//   F is the same for every filter, so the B products F P_i are ONE GEMM:
//        F (6x6)  x  [P_1 | P_2 | ... | P_B]  (6 x 6B)   =  [F P_1 | ... | F P_B]
//   With B = 32 that is a 6 x 192 x 6 GEMM instead of 32 tiny 6x6x6 ones.
//   For F P F^T we use the symmetry of P:  (F P_i)^T = P_i F^T, so after transposing
//   each 6x6 block in shared memory, the SAME GEMM gives  F (P_i F^T) = F P_i F^T.
//   Likewise  H [P_1 | ... | P_B] = [H P_1 | ... | H P_B] is one 3 x 192 x 6 GEMM.
//
//   The per-filter 3x3 solve  S_i Kt_i = H P_i  is done by
//     - cuSolverDx  POSV with BatchesPerBlock<B>   if KF_WITH_CUSOLVERDX is defined
//       (requires the full MathDx package: headers + libcusolverdx LTO library),
//     - a hand-written register Cholesky (one thread per filter) otherwise.
#pragma once

#include <cublasdx.hpp>
#ifdef KF_WITH_CUSOLVERDX
#include <cusolverdx.hpp>
#endif

#include "kf_device.cuh"

#ifndef KF_SM
#define KF_SM 750  // compute capability x 10 (T4 = 750, A100 = 800, L4 = 890, H100 = 900)
#endif

namespace kf {

constexpr int DX_B       = 32;   // filters per block
constexpr int DX_THREADS = 128;  // threads per block

// F (N x N) * [P_1..P_B] (N x N*B)
using DxGemmF = decltype(cublasdx::Size<N, N * DX_B, N>() + cublasdx::Precision<float>() +
                         cublasdx::Type<cublasdx::type::real>() +
                         cublasdx::Function<cublasdx::function::MM>() +
                         cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                         cublasdx::Block() + cublasdx::BlockDim<DX_THREADS>() + cublasdx::SM<KF_SM>());

// H (M x N) * [P_1..P_B] (N x N*B)
using DxGemmH = decltype(cublasdx::Size<M, N * DX_B, N>() + cublasdx::Precision<float>() +
                         cublasdx::Type<cublasdx::type::real>() +
                         cublasdx::Function<cublasdx::function::MM>() +
                         cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>() +
                         cublasdx::Block() + cublasdx::BlockDim<DX_THREADS>() + cublasdx::SM<KF_SM>());

#ifdef KF_WITH_CUSOLVERDX
// S_i (M x M) Kt_i = HP_i (M x N), for B filters at once
using DxPosv = decltype(cusolverdx::Size<M, M, N>() + cusolverdx::Precision<float>() +
                        cusolverdx::Type<cusolverdx::type::real>() +
                        cusolverdx::Function<cusolverdx::function::posv>() +
                        cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
                        cusolverdx::Arrangement<cusolverdx::arrangement::col_major>() +
                        cusolverdx::BatchesPerBlock<DX_B>() + cusolverdx::BlockDim<DX_THREADS>() +
                        cusolverdx::Block() + cusolverdx::SM<KF_SM>());
#endif

// X: [f][6], P: [f][36], Z: [f][3] (AoS: each block reads one contiguous chunk -> coalesced)
__global__ void __launch_bounds__(DX_THREADS)
kf_dx_kernel(float* __restrict__ X, float* __restrict__ P, const float* __restrict__ Z, int filters) {
    __shared__ alignas(16) float sF[NN];
    __shared__ alignas(16) float sH[MN];
    __shared__ alignas(16) float sP[NN * DX_B];   // [P_1 | ... | P_B]
    __shared__ alignas(16) float sT[NN * DX_B];   // GEMM output
    __shared__ alignas(16) float sHP[MN * DX_B];  // [H P_1 | ... ]
    __shared__ alignas(16) float sKt[MN * DX_B];  // right-hand side, then gain^T
    __shared__ alignas(16) float sS[MM * DX_B];   // innovation covariances
    __shared__ float sX[N * DX_B], sXn[N * DX_B], sY[M * DX_B];
#ifdef KF_WITH_CUSOLVERDX
    __shared__ int sInfo[DX_B];
#endif

    const int tid = threadIdx.x;
    const int f0  = blockIdx.x * DX_B;
    const int nb  = min(DX_B, filters - f0);  // valid filters in this block (tail block)

    // ---- 1. load (contiguous, coalesced) -------------------------------------------
    for (int i = tid; i < NN; i += DX_THREADS) sF[i] = cF[i];
    for (int i = tid; i < MN; i += DX_THREADS) sH[i] = cH[i];
    for (int i = tid; i < NN * DX_B; i += DX_THREADS) {
        const int b = i / NN, e = i % NN;
        sP[i] = (b < nb) ? P[(size_t)f0 * NN + i] : ((e % (N + 1)) == 0 ? 1.f : 0.f);  // pad: P = I
    }
    for (int i = tid; i < N * DX_B; i += DX_THREADS) sX[i] = (i / N < nb) ? X[(size_t)f0 * N + i] : 0.f;
    for (int i = tid; i < M * DX_B; i += DX_THREADS) sY[i] = (i / M < nb) ? Z[(size_t)f0 * M + i] : 0.f;
    __syncthreads();

    // ---- 2. predict ------------------------------------------------------------------
    DxGemmF().execute(1.f, sF, N, sP, N, 0.f, sT, N);  // T_i = F P_i
    __syncthreads();
    for (int i = tid; i < NN * DX_B; i += DX_THREADS) {  // P_i <- T_i^T = P_i F^T
        const int b = i / NN, e = i % NN, r = e % N, c = e / N;
        sP[i] = sT[b * NN + idx(c, r, N)];
    }
    for (int i = tid; i < N * DX_B; i += DX_THREADS) {  // x_i <- F x_i
        const int b = i / N, r = i % N;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(sF[idx(r, k, N)], sX[b * N + k], acc);
        sXn[i] = acc;
    }
    __syncthreads();
    DxGemmF().execute(1.f, sF, N, sP, N, 0.f, sT, N);  // T_i = F P_i F^T
    __syncthreads();
    for (int i = tid; i < NN * DX_B; i += DX_THREADS) sP[i] = sT[i] + cQ[i % NN];
    __syncthreads();

    // ---- 3. update -------------------------------------------------------------------
    DxGemmH().execute(1.f, sH, M, sP, N, 0.f, sHP, M);  // HP_i = H P_i
    __syncthreads();
    for (int i = tid; i < MM * DX_B; i += DX_THREADS) {  // S_i = HP_i H^T + R
        const int b = i / MM, e = i % MM, r = e % M, c = e / M;
        float acc = cR[e];
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(sHP[b * MN + idx(r, k, M)], sH[idx(c, k, M)], acc);
        sS[i] = acc;
    }
    for (int i = tid; i < MN * DX_B; i += DX_THREADS) sKt[i] = sHP[i];
    for (int i = tid; i < M * DX_B; i += DX_THREADS) {  // y_i = z_i - H x_i
        const int b = i / M, r = i % M;
        float acc = sY[i];
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(-sH[idx(r, k, M)], sXn[b * N + k], acc);
        sY[i] = acc;
    }
    __syncthreads();

#ifdef KF_WITH_CUSOLVERDX
    DxPosv().execute(sS, M, sKt, M, sInfo);  // S_i Kt_i = HP_i, B systems in one call
#else
    if (tid < DX_B) {  // one thread per filter, 3x3 Cholesky on registers
        float S[MM], B[MN];
#pragma unroll
        for (int e = 0; e < MM; ++e) S[e] = sS[tid * MM + e];
#pragma unroll
        for (int e = 0; e < MN; ++e) B[e] = sKt[tid * MN + e];
        chol_solve<M, N>(S, B);
#pragma unroll
        for (int e = 0; e < MN; ++e) sKt[tid * MN + e] = B[e];
    }
#endif
    __syncthreads();

    // ---- 4. x += Kt^T y,  P -= Kt^T HP, written straight to global memory ------------
    for (int i = tid; i < N * nb; i += DX_THREADS) {
        const int b = i / N, r = i % N;
        float acc = sXn[i];
#pragma unroll
        for (int k = 0; k < M; ++k) acc = fmaf(sKt[b * MN + idx(k, r, M)], sY[b * M + k], acc);
        X[(size_t)f0 * N + i] = acc;
    }
    for (int i = tid; i < NN * nb; i += DX_THREADS) {
        const int b = i / NN, e = i % NN, r = e % N, c = e / N;
        float acc = sP[i];
#pragma unroll
        for (int k = 0; k < M; ++k)
            acc = fmaf(-sKt[b * MN + idx(k, r, M)], sHP[b * MN + idx(k, c, M)], acc);
        P[(size_t)f0 * NN + i] = acc;
    }
}

inline void launch_dx(float* X, float* P, const float* Z, int filters, cudaStream_t st) {
    const int blocks = (filters + DX_B - 1) / DX_B;
    kf_dx_kernel<<<blocks, DX_THREADS, 0, st>>>(X, P, Z, filters);
}

}  // namespace kf
