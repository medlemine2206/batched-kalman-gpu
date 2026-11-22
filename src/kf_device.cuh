// kf_device.cuh — model in constant memory + the Kalman step on registers.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "kf_common.hpp"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace kf {

// The model is identical for every filter: it lives in constant memory, so a warp
// reading the same coefficient gets it in one broadcast.
__constant__ float cF[NN];
__constant__ float cQ[NN];
__constant__ float cH[MN];
__constant__ float cR[MM];

inline void upload_model(const Model& m) {
    float F[NN], Q[NN], H[MN], R[MM];
    for (int i = 0; i < NN; ++i) { F[i] = (float)m.F[i]; Q[i] = (float)m.Q[i]; }
    for (int i = 0; i < MN; ++i) H[i] = (float)m.H[i];
    for (int i = 0; i < MM; ++i) R[i] = (float)m.R[i];
    CUDA_CHECK(cudaMemcpyToSymbol(cF, F, sizeof(F)));
    CUDA_CHECK(cudaMemcpyToSymbol(cQ, Q, sizeof(Q)));
    CUDA_CHECK(cudaMemcpyToSymbol(cH, H, sizeof(H)));
    CUDA_CHECK(cudaMemcpyToSymbol(cR, R, sizeof(R)));
}

// In-place Cholesky of a small SPD matrix (lower) followed by the solve
// L L^T X = B for NRHS right-hand sides. A is M x M, B is M x NRHS, col-major.
template <int DIM, int NRHS>
__device__ __forceinline__ void chol_solve(float* A, float* B) {
#pragma unroll
    for (int j = 0; j < DIM; ++j) {
        float d = A[idx(j, j, DIM)];
#pragma unroll
        for (int k = 0; k < j; ++k) d -= A[idx(j, k, DIM)] * A[idx(j, k, DIM)];
        d = sqrtf(d);
        A[idx(j, j, DIM)] = d;
        const float inv = 1.0f / d;
#pragma unroll
        for (int i = j + 1; i < DIM; ++i) {
            float v = A[idx(i, j, DIM)];
#pragma unroll
            for (int k = 0; k < j; ++k) v -= A[idx(i, k, DIM)] * A[idx(j, k, DIM)];
            A[idx(i, j, DIM)] = v * inv;
        }
    }
#pragma unroll
    for (int c = 0; c < NRHS; ++c) {
        float w[DIM];
#pragma unroll
        for (int i = 0; i < DIM; ++i) {
            float v = B[idx(i, c, DIM)];
#pragma unroll
            for (int k = 0; k < i; ++k) v -= A[idx(i, k, DIM)] * w[k];
            w[i] = v / A[idx(i, i, DIM)];
        }
#pragma unroll
        for (int i = DIM - 1; i >= 0; --i) {
            float v = w[i];
#pragma unroll
            for (int k = i + 1; k < DIM; ++k) v -= A[idx(k, i, DIM)] * B[idx(k, c, DIM)];
            B[idx(i, c, DIM)] = v / A[idx(i, i, DIM)];
        }
    }
}

// One full predict + update on registers. Same algebra as kf_step_ref().
// With everything unrolled, all arrays below are promoted to registers.
__device__ __forceinline__ void kf_step_regs(float (&x)[N], float (&P)[NN], const float (&z)[M]) {
    float T[NN];
    // T = F P
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int r = 0; r < N; ++r) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) acc = fmaf(cF[idx(r, k, N)], P[idx(k, c, N)], acc);
            T[idx(r, c, N)] = acc;
        }
    // P = T F^T + Q
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int r = 0; r < N; ++r) {
            float acc = cQ[idx(r, c, N)];
#pragma unroll
            for (int k = 0; k < N; ++k) acc = fmaf(T[idx(r, k, N)], cF[idx(c, k, N)], acc);
            P[idx(r, c, N)] = acc;
        }
    // x = F x
    float xn[N];
#pragma unroll
    for (int r = 0; r < N; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(cF[idx(r, k, N)], x[k], acc);
        xn[r] = acc;
    }
    // HP = H P,  S = HP H^T + R
    float HP[MN], Kt[MN], S[MM];
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int r = 0; r < M; ++r) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < N; ++k) acc = fmaf(cH[idx(r, k, M)], P[idx(k, c, N)], acc);
            HP[idx(r, c, M)] = acc;
            Kt[idx(r, c, M)] = acc;
        }
#pragma unroll
    for (int c = 0; c < M; ++c)
#pragma unroll
        for (int r = 0; r < M; ++r) {
            float acc = cR[idx(r, c, M)];
#pragma unroll
            for (int k = 0; k < N; ++k) acc = fmaf(HP[idx(r, k, M)], cH[idx(c, k, M)], acc);
            S[idx(r, c, M)] = acc;
        }
    // Kt = S^-1 HP  (Cholesky, no explicit inverse)
    chol_solve<M, N>(S, Kt);
    // y = z - H x ;  x = xn + Kt^T y
    float y[M];
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float acc = z[r];
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(-cH[idx(r, k, M)], xn[k], acc);
        y[r] = acc;
    }
#pragma unroll
    for (int r = 0; r < N; ++r) {
        float acc = xn[r];
#pragma unroll
        for (int k = 0; k < M; ++k) acc = fmaf(Kt[idx(k, r, M)], y[k], acc);
        x[r] = acc;
    }
    // P -= Kt^T HP
#pragma unroll
    for (int c = 0; c < N; ++c)
#pragma unroll
        for (int r = 0; r < N; ++r) {
            float acc = P[idx(r, c, N)];
#pragma unroll
            for (int k = 0; k < M; ++k) acc = fmaf(-Kt[idx(k, r, M)], HP[idx(k, c, M)], acc);
            P[idx(r, c, N)] = acc;
        }
}

}  // namespace kf
