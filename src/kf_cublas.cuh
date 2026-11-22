// kf_cublas.cuh — baseline: the "textbook" way with host-side cuBLAS batched calls.
//
// Each step = 9 kernel launches (6 strided-batched GEMMs, 1 batched 3x3 inverse,
// 2 small element-wise kernels). Every intermediate (F P, H P, S, S^-1, Kt) makes a
// round trip to DRAM. This is exactly the overhead that device-side (Dx) libraries
// and kernel fusion remove.
//
// Note: cuBLAS has no batched Cholesky solve with several right-hand sides, so this
// baseline uses cublasSmatinvBatched (fine for a well-conditioned 3x3 S).
#pragma once

#include <cublas_v2.h>

#include "kf_device.cuh"

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t st__ = (call);                                                      \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                               \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__, __FILE__, __LINE__); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace kf {

// x = F x ; P = Q ; S = R   (P and S are then accumulated into by GEMMs with beta = 1)
__global__ void cublas_aux_predict(float* X, float* P, float* S, int filters) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= filters) return;
    float x[N];
#pragma unroll
    for (int i = 0; i < N; ++i) x[i] = X[(size_t)f * N + i];
#pragma unroll
    for (int r = 0; r < N; ++r) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(cF[idx(r, k, N)], x[k], acc);
        X[(size_t)f * N + r] = acc;
    }
#pragma unroll
    for (int i = 0; i < NN; ++i) P[(size_t)f * NN + i] = cQ[i];
#pragma unroll
    for (int i = 0; i < MM; ++i) S[(size_t)f * MM + i] = cR[i];
}

// y = z - H x ; x += Kt^T y
__global__ void cublas_aux_update(float* X, const float* Kt, const float* Z, int filters) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= filters) return;
    float x[N], y[M];
#pragma unroll
    for (int i = 0; i < N; ++i) x[i] = X[(size_t)f * N + i];
#pragma unroll
    for (int r = 0; r < M; ++r) {
        float acc = Z[(size_t)f * M + r];
#pragma unroll
        for (int k = 0; k < N; ++k) acc = fmaf(-cH[idx(r, k, M)], x[k], acc);
        y[r] = acc;
    }
#pragma unroll
    for (int r = 0; r < N; ++r) {
        float acc = x[r];
#pragma unroll
        for (int k = 0; k < M; ++k) acc = fmaf(Kt[(size_t)f * MN + idx(k, r, M)], y[k], acc);
        X[(size_t)f * N + r] = acc;
    }
}

__global__ void fill_ptrs(float** ptrs, float* base, int stride, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) ptrs[i] = base + (size_t)i * stride;
}

struct CublasKF {
    cublasHandle_t h{};
    int filters = 0;
    float *dF{}, *dH{}, *dT{}, *dHP{}, *dS{}, *dSinv{}, *dKt{};
    float **dSptr{}, **dSinvPtr{};
    int* dInfo{};

    CublasKF(const Model& m, int filters_) : filters(filters_) {
        CUBLAS_CHECK(cublasCreate(&h));
        float F[NN], H[MN];
        for (int i = 0; i < NN; ++i) F[i] = (float)m.F[i];
        for (int i = 0; i < MN; ++i) H[i] = (float)m.H[i];
        CUDA_CHECK(cudaMalloc(&dF, sizeof(F)));
        CUDA_CHECK(cudaMalloc(&dH, sizeof(H)));
        CUDA_CHECK(cudaMemcpy(dF, F, sizeof(F), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dH, H, sizeof(H), cudaMemcpyHostToDevice));
        const size_t f = (size_t)filters;
        CUDA_CHECK(cudaMalloc(&dT, f * NN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dHP, f * MN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dS, f * MM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dSinv, f * MM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dKt, f * MN * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dSptr, f * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&dSinvPtr, f * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&dInfo, f * sizeof(int)));
        const int b = (filters + 255) / 256;
        fill_ptrs<<<b, 256>>>(dSptr, dS, MM, filters);
        fill_ptrs<<<b, 256>>>(dSinvPtr, dSinv, MM, filters);
        CUDA_CHECK(cudaGetLastError());
    }
    ~CublasKF() {
        cublasDestroy(h);
        cudaFree(dF); cudaFree(dH); cudaFree(dT); cudaFree(dHP); cudaFree(dS);
        cudaFree(dSinv); cudaFree(dKt); cudaFree(dSptr); cudaFree(dSinvPtr); cudaFree(dInfo);
    }

    // X: [f][6], P: [f][36], Z: [f][3] (AoS)
    void step(float* X, float* P, const float* Z, cudaStream_t st) {
        CUBLAS_CHECK(cublasSetStream(h, st));
        const float one = 1.f, zero = 0.f, minus1 = -1.f;
        const int threads = 128, blocks = (filters + threads - 1) / threads;
        const auto NT = CUBLAS_OP_N, TT = CUBLAS_OP_T;
        // 1. T = F P
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, NT, NT, N, N, N, &one, dF, N, 0, P, N, NN,
                                               &zero, dT, N, NN, filters));
        // 2. x = F x, P = Q, S = R
        cublas_aux_predict<<<blocks, threads, 0, st>>>(X, P, dS, filters);
        // 3. P += T F^T
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, NT, TT, N, N, N, &one, dT, N, NN, dF, N, 0,
                                               &one, P, N, NN, filters));
        // 4. HP = H P
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, NT, NT, M, N, N, &one, dH, M, 0, P, N, NN,
                                               &zero, dHP, M, MN, filters));
        // 5. S += HP H^T
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, NT, TT, M, M, N, &one, dHP, M, MN, dH, M, 0,
                                               &one, dS, M, MM, filters));
        // 6. S^-1
        CUBLAS_CHECK(cublasSmatinvBatched(h, M, dSptr, M, dSinvPtr, M, dInfo, filters));
        // 7. Kt = S^-1 HP
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, NT, NT, M, N, M, &one, dSinv, M, MM, dHP, M, MN,
                                               &zero, dKt, M, MN, filters));
        // 8. x += Kt^T (z - H x)
        cublas_aux_update<<<blocks, threads, 0, st>>>(X, dKt, Z, filters);
        // 9. P -= Kt^T HP
        CUBLAS_CHECK(cublasSgemmStridedBatched(h, TT, NT, N, N, M, &minus1, dKt, M, MN, dHP, M, MN,
                                               &one, P, N, NN, filters));
    }
};

}  // namespace kf
