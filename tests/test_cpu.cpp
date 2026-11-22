// test_cpu.cpp — sanity checks of the reference filter (no GPU needed).
//
//  1. Accuracy  : after convergence, position RMSE must be well below the sensor noise.
//  2. Consistency: average NEES (x-xt)^T P^-1 (x-xt) must be close to N = 6
//                  (chi-square test: the filter's covariance tells the truth).
//  3. Precision : float and double references must stay close (error in units of sigma).
//
// Writes results/cpu_rmse.csv (step, rmse_meas, rmse_kf, nees) for plotting.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "kf_common.hpp"

using namespace kf;

// NEES of one filter: e^T P^-1 e via Cholesky of the 6x6 covariance.
static double nees(const double* x, const double* P, const float* xt) {
    double L[NN];
    std::memcpy(L, P, sizeof(L));
    for (int j = 0; j < N; ++j) {
        double d = L[idx(j, j, N)];
        for (int k = 0; k < j; ++k) d -= L[idx(j, k, N)] * L[idx(j, k, N)];
        if (d <= 0) return 1e30;
        d = std::sqrt(d);
        L[idx(j, j, N)] = d;
        for (int i = j + 1; i < N; ++i) {
            double v = L[idx(i, j, N)];
            for (int k = 0; k < j; ++k) v -= L[idx(i, k, N)] * L[idx(j, k, N)];
            L[idx(i, j, N)] = v / d;
        }
    }
    double w[N], s = 0;
    for (int i = 0; i < N; ++i) {
        double v = x[i] - xt[i];
        for (int k = 0; k < i; ++k) v -= L[idx(i, k, N)] * w[k];
        w[i] = v / L[idx(i, i, N)];
        s += w[i] * w[i];
    }
    return s;
}

int main(int argc, char** argv) {
    const int filters = argc > 1 ? std::atoi(argv[1]) : 2000;
    const int steps   = argc > 2 ? std::atoi(argv[2]) : 200;
    const std::string out = argc > 3 ? argv[3] : "results/cpu_rmse.csv";

    // sym_idx must be a bijection onto [0, 21) on the upper triangle
    {
        bool seen[NSYM] = {};
        for (int c = 0; c < N; ++c)
            for (int r = 0; r <= c; ++r) { assert(!seen[sym_idx(r, c)]); seen[sym_idx(r, c)] = true; }
        for (bool b : seen) assert(b);
        assert(sym_idx(2, 4) == sym_idx(4, 2));
    }

    const Model m = make_model();
    const Scenario s = make_scenario(m, filters, steps, 7, /*keep_truth=*/true);

    std::vector<double> x((size_t)filters * N), P((size_t)filters * NN);
    std::vector<float> xf((size_t)filters * N), Pf((size_t)filters * NN);
    for (size_t i = 0; i < x.size(); ++i) xf[i] = (float)(x[i] = s.x0[i]);
    for (size_t i = 0; i < P.size(); ++i) Pf[i] = (float)(P[i] = s.P0[i]);

    FILE* fo = std::fopen(out.c_str(), "w");
    if (fo) std::fprintf(fo, "step,rmse_meas,rmse_kf,nees\n");

    double rmse_kf_tail = 0, rmse_meas_tail = 0, nees_tail = 0;
    int tail = 0;
    for (int k = 0; k < steps; ++k) {
        double se_kf = 0, se_meas = 0, ne = 0;
        for (int f = 0; f < filters; ++f) {
            const float* z  = &s.z[((size_t)k * filters + f) * M];
            const float* xt = &s.truth[((size_t)k * filters + f) * N];
            kf_step_ref<double>(m, &x[(size_t)f * N], &P[(size_t)f * NN], z);
            kf_step_ref<float>(m, &xf[(size_t)f * N], &Pf[(size_t)f * NN], z);
            for (int a = 0; a < 3; ++a) {
                const double e = x[(size_t)f * N + a] - xt[a], em = z[a] - xt[a];
                se_kf += e * e; se_meas += em * em;
            }
            ne += nees(&x[(size_t)f * N], &P[(size_t)f * NN], xt);
        }
        const double rk = std::sqrt(se_kf / filters), rm = std::sqrt(se_meas / filters), nn = ne / filters;
        if (fo) std::fprintf(fo, "%d,%.4f,%.4f,%.4f\n", k, rm, rk, nn);
        if (k >= steps / 2) { rmse_kf_tail += rk; rmse_meas_tail += rm; nees_tail += nn; ++tail; }
    }
    if (fo) std::fclose(fo);
    rmse_kf_tail /= tail; rmse_meas_tail /= tail; nees_tail /= tail;

    double max_rel = 0;
    for (size_t i = 0; i < x.size(); ++i)
        // error measured in units of the filter's own standard deviation
        max_rel = std::fmax(max_rel, std::fabs(x[i] - xf[i]) / std::sqrt(P[(i / N) * NN + idx(i % N, i % N, N)]));

    std::printf("filters=%d steps=%d\n", filters, steps);
    std::printf("3D position RMSE  measurements: %.3f m | Kalman: %.3f m\n", rmse_meas_tail, rmse_kf_tail);
    std::printf("average NEES (expected ~%d): %.3f\n", N, nees_tail);
    std::printf("float vs double, max state error: %.2e sigma\n", max_rel);

    int fails = 0;
    if (!(rmse_kf_tail < 0.75 * rmse_meas_tail)) { std::printf("FAIL: filter does not beat raw measurements\n"); ++fails; }
    if (!(nees_tail > 0.8 * N && nees_tail < 1.2 * N)) { std::printf("FAIL: NEES not consistent\n"); ++fails; }
    if (!(max_rel < 1e-2)) { std::printf("FAIL: float drifts from double\n"); ++fails; }
    std::printf(fails ? "SOME TESTS FAILED\n" : "ALL CPU TESTS PASSED\n");
    return fails ? 1 : 0;
}
