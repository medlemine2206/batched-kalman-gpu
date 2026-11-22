// test_layout_logic.cpp — CPU check of the index tricks used by the GPU kernels.
//
//  * dx kernel: "shared-model batching" of F [P_1|...|P_B], block transposes,
//    H [P_1|...|P_B], and the per-filter update, with the exact index arithmetic of
//    src/kf_dx.cuh (the cuBLASDx GEMM is replaced by a plain column-major GEMM).
//  * cuBLAS baseline: same sequence of 9 operations, with an explicit 3x3 inverse.
//  * symmetric packing of P.
// All must match kf_step_ref() to float precision.
#include <cstdio>
#include <vector>

#include "kf_common.hpp"

using namespace kf;

static void gemm(int m, int n, int k, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                 bool ta = false, bool tb = false, float alpha = 1.f, float beta = 0.f) {
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < m; ++r) {
            float acc = 0.f;
            for (int i = 0; i < k; ++i) {
                const float a = ta ? A[idx(i, r, lda)] : A[idx(r, i, lda)];
                const float b = tb ? B[idx(c, i, ldb)] : B[idx(i, c, ldb)];
                acc += a * b;
            }
            C[idx(r, c, ldc)] = alpha * acc + beta * C[idx(r, c, ldc)];
        }
}

static void chol_solve3(float* A, float* B, int nrhs) {
    for (int j = 0; j < M; ++j) {
        float d = A[idx(j, j, M)];
        for (int k = 0; k < j; ++k) d -= A[idx(j, k, M)] * A[idx(j, k, M)];
        d = std::sqrt(d);
        A[idx(j, j, M)] = d;
        for (int i = j + 1; i < M; ++i) {
            float v = A[idx(i, j, M)];
            for (int k = 0; k < j; ++k) v -= A[idx(i, k, M)] * A[idx(j, k, M)];
            A[idx(i, j, M)] = v / d;
        }
    }
    for (int c = 0; c < nrhs; ++c) {
        float w[M];
        for (int i = 0; i < M; ++i) {
            float v = B[idx(i, c, M)];
            for (int k = 0; k < i; ++k) v -= A[idx(i, k, M)] * w[k];
            w[i] = v / A[idx(i, i, M)];
        }
        for (int i = M - 1; i >= 0; --i) {
            float v = w[i];
            for (int k = i + 1; k < M; ++k) v -= A[idx(k, i, M)] * B[idx(k, c, M)];
            B[idx(i, c, M)] = v / A[idx(i, i, M)];
        }
    }
}

// Emulation of kf_dx_kernel for one block of B filters (nb valid).
static void dx_block(const float* F, const float* Q, const float* H, const float* R, float* X, float* P,
                     const float* Z, int nb) {
    constexpr int B = 32;
    std::vector<float> sP(NN * B), sT(NN * B), sHP(MN * B), sKt(MN * B), sS(MM * B), sX(N * B), sXn(N * B), sY(M * B);
    for (int i = 0; i < NN * B; ++i) { const int b = i / NN, e = i % NN; sP[i] = b < nb ? P[i] : (e % (N + 1) == 0 ? 1.f : 0.f); }
    for (int i = 0; i < N * B; ++i) sX[i] = i / N < nb ? X[i] : 0.f;
    for (int i = 0; i < M * B; ++i) sY[i] = i / M < nb ? Z[i] : 0.f;

    gemm(N, N * B, N, F, N, sP.data(), N, sT.data(), N);
    for (int i = 0; i < NN * B; ++i) { const int b = i / NN, e = i % NN, r = e % N, c = e / N; sP[i] = sT[b * NN + idx(c, r, N)]; }
    for (int i = 0; i < N * B; ++i) {
        const int b = i / N, r = i % N; float acc = 0;
        for (int k = 0; k < N; ++k) acc += F[idx(r, k, N)] * sX[b * N + k];
        sXn[i] = acc;
    }
    gemm(N, N * B, N, F, N, sP.data(), N, sT.data(), N);
    for (int i = 0; i < NN * B; ++i) sP[i] = sT[i] + Q[i % NN];
    gemm(M, N * B, N, H, M, sP.data(), N, sHP.data(), M);
    for (int i = 0; i < MM * B; ++i) {
        const int b = i / MM, e = i % MM, r = e % M, c = e / M; float acc = R[e];
        for (int k = 0; k < N; ++k) acc += sHP[b * MN + idx(r, k, M)] * H[idx(c, k, M)];
        sS[i] = acc;
    }
    for (int i = 0; i < MN * B; ++i) sKt[i] = sHP[i];
    for (int i = 0; i < M * B; ++i) {
        const int b = i / M, r = i % M; float acc = sY[i];
        for (int k = 0; k < N; ++k) acc -= H[idx(r, k, M)] * sXn[b * N + k];
        sY[i] = acc;
    }
    for (int t = 0; t < B; ++t) chol_solve3(&sS[t * MM], &sKt[t * MN], N);
    for (int i = 0; i < N * nb; ++i) {
        const int b = i / N, r = i % N; float acc = sXn[i];
        for (int k = 0; k < M; ++k) acc += sKt[b * MN + idx(k, r, M)] * sY[b * M + k];
        X[i] = acc;
    }
    for (int i = 0; i < NN * nb; ++i) {
        const int b = i / NN, e = i % NN, r = e % N, c = e / N; float acc = sP[i];
        for (int k = 0; k < M; ++k) acc -= sKt[b * MN + idx(k, r, M)] * sHP[b * MN + idx(k, c, M)];
        P[i] = acc;
    }
}

// Emulation of CublasKF::step for one filter (same operation order, explicit inverse).
static void cublas_like(const float* F, const float* Q, const float* H, const float* R, float* x, float* P, const float* z) {
    float T[NN], HP[MN], S[MM], Si[MM], Kt[MN], xn[N];
    gemm(N, N, N, F, N, P, N, T, N);
    for (int r = 0; r < N; ++r) { float a = 0; for (int k = 0; k < N; ++k) a += F[idx(r, k, N)] * x[k]; xn[r] = a; }
    for (int i = 0; i < NN; ++i) P[i] = Q[i];
    for (int i = 0; i < MM; ++i) S[i] = R[i];
    gemm(N, N, N, T, N, F, N, P, N, false, true, 1.f, 1.f);
    gemm(M, N, N, H, M, P, N, HP, M);
    gemm(M, M, N, HP, M, H, M, S, M, false, true, 1.f, 1.f);
    const float a = S[0], b = S[3], c = S[6], d = S[1], e = S[4], f = S[7], g = S[2], h = S[5], k = S[8];
    const float det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    // rows/cols of a 3x3 inverse (S col-major: S(r,c) = S[c*3+r])
    const float inv[9] = {(e * k - f * h), -(d * k - f * g), (d * h - e * g),
                          -(b * k - c * h), (a * k - c * g), -(a * h - b * g),
                          (b * f - c * e), -(a * f - c * d), (a * e - b * d)};
    for (int i = 0; i < 9; ++i) Si[i] = inv[i] / det;
    gemm(M, N, M, Si, M, HP, M, Kt, M);
    float y[M];
    for (int r = 0; r < M; ++r) {
        y[r] = z[r];
        for (int q = 0; q < N; ++q) y[r] -= H[idx(r, q, M)] * xn[q];
    }
    for (int q = 0; q < N; ++q) {
        float acc = xn[q];
        for (int r = 0; r < M; ++r) acc += Kt[idx(r, q, M)] * y[r];
        x[q] = acc;
    }
    gemm(N, N, M, Kt, M, HP, M, P, N, true, false, -1.f, 1.f);
}

int main() {
    const Model m = make_model();
    float F[NN], Q[NN], H[MN], R[MM];
    for (int i = 0; i < NN; ++i) { F[i] = (float)m.F[i]; Q[i] = (float)m.Q[i]; }
    for (int i = 0; i < MN; ++i) H[i] = (float)m.H[i];
    for (int i = 0; i < MM; ++i) R[i] = (float)m.R[i];

    const int filters = 100, steps = 50;  // 100 = 3 full blocks + a tail block of 4
    const Scenario s = make_scenario(m, filters, steps, 3);
    std::vector<double> xr, Pr;
    run_reference(m, s, xr, Pr);

    std::vector<float> xd(s.x0), Pd(s.P0), xc(s.x0), Pc(s.P0);
    for (int k = 0; k < steps; ++k) {
        const float* Zk = &s.z[(size_t)k * filters * M];
        for (int f0 = 0; f0 < filters; f0 += 32)
            dx_block(F, Q, H, R, &xd[f0 * N], &Pd[f0 * NN], &Zk[f0 * M], std::min(32, filters - f0));
        for (int f = 0; f < filters; ++f) cublas_like(F, Q, H, R, &xc[f * N], &Pc[f * NN], &Zk[f * M]);
    }
    auto err = [&](const std::vector<float>& x, const std::vector<float>& P) {
        double ex = 0, ep = 0;
        for (int f = 0; f < filters; ++f)
            for (int r = 0; r < N; ++r) {
                const double sr = std::sqrt(Pr[f * NN + idx(r, r, N)]);
                ex = std::fmax(ex, std::fabs(x[f * N + r] - xr[f * N + r]) / sr);
                for (int c = 0; c < N; ++c)
                    ep = std::fmax(ep, std::fabs(P[f * NN + idx(r, c, N)] - Pr[f * NN + idx(r, c, N)]) /
                                           (sr * std::sqrt(Pr[f * NN + idx(c, c, N)])));
            }
        return std::make_pair(ex, ep);
    };
    const auto [dxe, dpe] = err(xd, Pd);
    const auto [cxe, cpe] = err(xc, Pc);

    // symmetric packing round trip
    int sym_bad = 0;
    for (int f = 0; f < filters; ++f) {
        float packed[NSYM], full[NN];
        for (int c = 0; c < N; ++c) for (int r = 0; r <= c; ++r) packed[sym_idx(r, c)] = Pd[f * NN + idx(r, c, N)];
        for (int c = 0; c < N; ++c) for (int r = 0; r < N; ++r) full[idx(r, c, N)] = packed[sym_idx(r, c)];
        for (int c = 0; c < N; ++c) for (int r = 0; r <= c; ++r) sym_bad += full[idx(r, c, N)] != Pd[f * NN + idx(r, c, N)];
    }

    std::printf("dx block emulation   : x err %.2e sigma, P err %.2e\n", dxe, dpe);
    std::printf("cublas emulation     : x err %.2e sigma, P err %.2e\n", cxe, cpe);
    std::printf("symmetric packing    : %d mismatches\n", sym_bad);
    const bool ok = dxe < 1e-2 && dpe < 1e-3 && cxe < 1e-2 && cpe < 1e-3 && sym_bad == 0;
    std::printf(ok ? "ALL LAYOUT TESTS PASSED\n" : "LAYOUT TESTS FAILED\n");
    return ok ? 0 : 1;
}
