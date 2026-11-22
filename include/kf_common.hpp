// kf_common.hpp — model, synthetic data and CPU reference for batched Kalman filtering.
//
// Model: 3D constant-velocity target, state x = [px py pz vx vy vz]^T (N = 6),
// position-only measurements z = [px py pz]^T (M = 3).
//
// All matrices are stored COLUMN-MAJOR: A(r, c) = A[c * rows + r].
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace kf {

constexpr int N    = 6;               // state dimension
constexpr int M    = 3;               // measurement dimension
constexpr int NN   = N * N;           // 36 floats for a full covariance
constexpr int NSYM = N * (N + 1) / 2; // 21 floats for the upper triangle
constexpr int MN   = M * N;           // 18
constexpr int MM   = M * M;           // 9

// Column-major index helpers (usable on host and device).
#if defined(__CUDACC__)
#define KF_HD __host__ __device__ __forceinline__
#else
#define KF_HD inline
#endif

KF_HD constexpr int idx(int r, int c, int ld) { return c * ld + r; }
// Packed upper triangle, column by column: (0,0) (0,1) (1,1) (0,2) (1,2) (2,2) ...
KF_HD constexpr int sym_idx(int r, int c) { return (r <= c) ? c * (c + 1) / 2 + r : r * (r + 1) / 2 + c; }

// ---------------------------------------------------------------------------
// Model matrices (double on the host, converted to float for the GPU)
// ---------------------------------------------------------------------------
struct Model {
    double dt, q, sigma;
    double F[NN];  // state transition
    double Q[NN];  // process noise covariance
    double H[MN];  // measurement matrix (M x N)
    double R[MM];  // measurement noise covariance
};

// dt: time step [s], q: white-acceleration spectral density [m^2/s^3], sigma: sensor noise [m]
inline Model make_model(double dt = 0.1, double q = 1.0, double sigma = 5.0) {
    Model m{};
    m.dt = dt; m.q = q; m.sigma = sigma;
    for (int i = 0; i < NN; ++i) { m.F[i] = 0; m.Q[i] = 0; }
    for (int i = 0; i < MN; ++i) m.H[i] = 0;
    for (int i = 0; i < MM; ++i) m.R[i] = 0;

    const double q11 = q * dt * dt * dt / 3.0, q12 = q * dt * dt / 2.0, q22 = q * dt;
    for (int a = 0; a < 3; ++a) {             // one block per axis
        m.F[idx(a, a, N)]         = 1.0;
        m.F[idx(a, a + 3, N)]     = dt;
        m.F[idx(a + 3, a + 3, N)] = 1.0;
        m.Q[idx(a, a, N)]         = q11;
        m.Q[idx(a, a + 3, N)]     = q12;
        m.Q[idx(a + 3, a, N)]     = q12;
        m.Q[idx(a + 3, a + 3, N)] = q22;
        m.H[idx(a, a, M)]         = 1.0;
        m.R[idx(a, a, M)]         = sigma * sigma;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Synthetic scenario: `filters` independent targets observed for `steps` scans
// ---------------------------------------------------------------------------
struct Scenario {
    int filters = 0, steps = 0;
    std::vector<float> x0;     // [filters][N]       initial estimate
    std::vector<float> P0;     // [filters][NN]      initial covariance (col-major)
    std::vector<float> z;      // [steps][filters][M] measurements (AoS per scan)
    std::vector<float> truth;  // [steps][filters][N] true states (only if keep_truth)
};

inline Scenario make_scenario(const Model& m, int filters, int steps, uint64_t seed = 42,
                              bool keep_truth = false) {
    Scenario s;
    s.filters = filters; s.steps = steps;
    s.x0.resize((size_t)filters * N);
    s.P0.resize((size_t)filters * NN);
    s.z.resize((size_t)steps * filters * M);
    if (keep_truth) s.truth.resize((size_t)steps * filters * N);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> upos(-10000.0, 10000.0), uvel(-300.0, 300.0);

    // Per-axis Cholesky factor of the 2x2 process-noise block [[q11 q12][q12 q22]]
    const double dt = m.dt, q = m.q;
    const double q11 = q * dt * dt * dt / 3.0, q12 = q * dt * dt / 2.0, q22 = q * dt;
    const double l11 = std::sqrt(q11), l21 = q12 / l11, l22 = std::sqrt(q22 - l21 * l21);

    const double pos_std0 = m.sigma, vel_std0 = 30.0;
    std::vector<double> xt(N);
    for (int f = 0; f < filters; ++f) {
        for (int a = 0; a < 3; ++a) { xt[a] = upos(rng); xt[a + 3] = uvel(rng); }
        for (int a = 0; a < 3; ++a) {  // initial estimate = truth + initial uncertainty
            s.x0[(size_t)f * N + a]     = (float)(xt[a] + pos_std0 * g(rng));
            s.x0[(size_t)f * N + a + 3] = (float)(xt[a + 3] + vel_std0 * g(rng));
        }
        float* P = &s.P0[(size_t)f * NN];
        for (int i = 0; i < NN; ++i) P[i] = 0.f;
        for (int a = 0; a < 3; ++a) {
            P[idx(a, a, N)]         = (float)(pos_std0 * pos_std0);
            P[idx(a + 3, a + 3, N)] = (float)(vel_std0 * vel_std0);
        }
        for (int k = 0; k < steps; ++k) {
            for (int a = 0; a < 3; ++a) {  // propagate truth: x <- F x + w
                const double w1 = g(rng), w2 = g(rng);
                const double p = xt[a] + dt * xt[a + 3] + l11 * w1;
                const double v = xt[a + 3] + l21 * w1 + l22 * w2;
                xt[a] = p; xt[a + 3] = v;
            }
            for (int a = 0; a < 3; ++a)
                s.z[((size_t)k * filters + f) * M + a] = (float)(xt[a] + m.sigma * g(rng));
            if (keep_truth)
                for (int i = 0; i < N; ++i) s.truth[((size_t)k * filters + f) * N + i] = (float)xt[i];
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Reference Kalman step for ONE filter. Every GPU variant implements exactly
// this algebra (no symmetrisation, Cholesky solve instead of an inverse):
//
//   predict : x = F x,      P = F P F^T + Q
//   update  : HP = H P (M x N),  S = HP H^T + R,  S = L L^T
//             Kt = S^{-1} HP      (so K = Kt^T, because P is symmetric)
//             y  = z - H x,  x += Kt^T y,  P -= Kt^T HP
// ---------------------------------------------------------------------------
template <class Real>
void kf_step_ref(const Model& m, Real* x, Real* P, const float* z) {
    Real FP[NN], Pn[NN], xn[N], HP[MN], S[MM], Kt[MN], y[M];
    // predict
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            Real acc = 0;
            for (int k = 0; k < N; ++k) acc += (Real)m.F[idx(r, k, N)] * P[idx(k, c, N)];
            FP[idx(r, c, N)] = acc;
        }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            Real acc = (Real)m.Q[idx(r, c, N)];
            for (int k = 0; k < N; ++k) acc += FP[idx(r, k, N)] * (Real)m.F[idx(c, k, N)];
            Pn[idx(r, c, N)] = acc;
        }
    for (int r = 0; r < N; ++r) {
        Real acc = 0;
        for (int k = 0; k < N; ++k) acc += (Real)m.F[idx(r, k, N)] * x[k];
        xn[r] = acc;
    }
    // update
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            Real acc = 0;
            for (int k = 0; k < N; ++k) acc += (Real)m.H[idx(r, k, M)] * Pn[idx(k, c, N)];
            HP[idx(r, c, M)] = acc;
        }
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < M; ++c) {
            Real acc = (Real)m.R[idx(r, c, M)];
            for (int k = 0; k < N; ++k) acc += HP[idx(r, k, M)] * (Real)m.H[idx(c, k, M)];
            S[idx(r, c, M)] = acc;
        }
    // Cholesky S = L L^T (lower, in place)
    for (int j = 0; j < M; ++j) {
        Real d = S[idx(j, j, M)];
        for (int k = 0; k < j; ++k) d -= S[idx(j, k, M)] * S[idx(j, k, M)];
        d = std::sqrt(d);
        S[idx(j, j, M)] = d;
        for (int i = j + 1; i < M; ++i) {
            Real v = S[idx(i, j, M)];
            for (int k = 0; k < j; ++k) v -= S[idx(i, k, M)] * S[idx(j, k, M)];
            S[idx(i, j, M)] = v / d;
        }
    }
    // Solve L L^T Kt = HP, column by column
    for (int c = 0; c < N; ++c) {
        Real w[M];
        for (int i = 0; i < M; ++i) {  // forward: L w = b
            Real v = HP[idx(i, c, M)];
            for (int k = 0; k < i; ++k) v -= S[idx(i, k, M)] * w[k];
            w[i] = v / S[idx(i, i, M)];
        }
        for (int i = M - 1; i >= 0; --i) {  // backward: L^T k = w
            Real v = w[i];
            for (int k = i + 1; k < M; ++k) v -= S[idx(k, i, M)] * Kt[idx(k, c, M)];
            Kt[idx(i, c, M)] = v / S[idx(i, i, M)];
        }
    }
    for (int r = 0; r < M; ++r) {
        Real acc = (Real)z[r];
        for (int k = 0; k < N; ++k) acc -= (Real)m.H[idx(r, k, M)] * xn[k];
        y[r] = acc;
    }
    for (int r = 0; r < N; ++r) {
        Real acc = xn[r];
        for (int k = 0; k < M; ++k) acc += Kt[idx(k, r, M)] * y[k];
        x[r] = acc;
    }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            Real acc = Pn[idx(r, c, N)];
            for (int k = 0; k < M; ++k) acc -= Kt[idx(k, r, M)] * HP[idx(k, c, M)];
            P[idx(r, c, N)] = acc;
        }
}

// Run the reference over a whole scenario (AoS state, in place, double precision).
inline void run_reference(const Model& m, const Scenario& s, std::vector<double>& x,
                          std::vector<double>& P, int first = 0, int count = -1) {
    if (count < 0) count = s.filters - first;
    x.assign((size_t)count * N, 0.0);
    P.assign((size_t)count * NN, 0.0);
    for (int i = 0; i < count; ++i) {
        const int f = first + i;
        for (int r = 0; r < N; ++r) x[(size_t)i * N + r] = s.x0[(size_t)f * N + r];
        for (int r = 0; r < NN; ++r) P[(size_t)i * NN + r] = s.P0[(size_t)f * NN + r];
        for (int k = 0; k < s.steps; ++k)
            kf_step_ref<double>(m, &x[(size_t)i * N], &P[(size_t)i * NN],
                                &s.z[((size_t)k * s.filters + f) * M]);
    }
}

// ---------------------------------------------------------------------------
// Cost model per filter per step (used for GFLOP/s, GB/s and the roofline)
// ---------------------------------------------------------------------------
constexpr double flops_per_filter_step() {
    return 2.0 * N * N * N        // F P
         + 2.0 * N * N * N + NN   // (F P) F^T + Q
         + 2.0 * N * N            // F x
         + 2.0 * M * N * N        // H P
         + 2.0 * M * M * N + MM   // HP H^T + R
         + (M * M * M) / 3.0 + M  // Cholesky (approx.)
         + 2.0 * M * M * N        // two triangular solves, N right-hand sides
         + 2.0 * M * N + M        // innovation
         + 2.0 * M * N            // x += Kt^T y
         + 2.0 * M * N * N;       // P -= Kt^T HP
}

// Minimum DRAM traffic per filter per step: read x, P, z and write x, P.
constexpr double bytes_per_filter_step(int p_floats) {
    return 4.0 * (2.0 * N + 2.0 * p_floats + M);
}

}  // namespace kf
