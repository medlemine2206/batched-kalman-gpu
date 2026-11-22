// main.cu — benchmark driver: runs every variant on the same scenario, checks it
// against the double-precision CPU reference and prints / saves the timings.
//
//   ./kf_bench --filters 65536 --steps 100 --check 4096 --csv results/bench.csv
//   ./kf_bench --sweep --csv results/sweep.csv        (filters = 1K .. 1M)
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "kf_common.hpp"
#include "kf_cublas.cuh"
#include "kf_device.cuh"
#include "kf_dx.cuh"
#include "kf_fused.cuh"

using namespace kf;

enum class Variant { Cublas, FusedAoS, FusedSoA, FusedSoASym, Dx };

struct VariantInfo {
    Variant v;
    const char* name;
    const char* description;
};

static const VariantInfo kVariants[] = {
    {Variant::Cublas, "cublas_batched", "9 launches/step, cuBLAS strided-batched + matinv"},
    {Variant::FusedAoS, "fused_aos", "1 kernel, thread/filter, AoS (uncoalesced)"},
    {Variant::FusedSoA, "fused_soa", "1 kernel, thread/filter, SoA (coalesced)"},
    {Variant::FusedSoASym, "fused_soa_sym", "1 kernel, thread/filter, SoA + symmetric P (21 floats)"},
#ifdef KF_WITH_CUSOLVERDX
    {Variant::Dx, "dx_cublasdx_cusolverdx", "1 kernel, block/32 filters, cuBLASDx GEMMs + cuSolverDx POSV"},
#else
    {Variant::Dx, "dx_cublasdx", "1 kernel, block/32 filters, cuBLASDx GEMMs + register Cholesky"},
#endif
};

static bool is_soa(Variant v) { return v == Variant::FusedSoA || v == Variant::FusedSoASym; }
static int p_floats(Variant v) { return v == Variant::FusedSoASym ? NSYM : NN; }

// ---------------------------------------------------------------------------
// Host <-> device layout conversion
// ---------------------------------------------------------------------------
static void pack(Variant v, const Scenario& s, std::vector<float>& x, std::vector<float>& P,
                 std::vector<float>& z) {
    const int F = s.filters;
    x.assign((size_t)F * N, 0.f);
    P.assign((size_t)F * p_floats(v), 0.f);
    z.assign(s.z.size(), 0.f);
    for (int f = 0; f < F; ++f) {
        for (int r = 0; r < N; ++r)
            x[is_soa(v) ? (size_t)r * F + f : (size_t)f * N + r] = s.x0[(size_t)f * N + r];
        const float* P0 = &s.P0[(size_t)f * NN];
        if (v == Variant::FusedSoASym) {
            for (int c = 0; c < N; ++c)
                for (int r = 0; r <= c; ++r) P[(size_t)sym_idx(r, c) * F + f] = P0[idx(r, c, N)];
        } else {
            for (int e = 0; e < NN; ++e) P[is_soa(v) ? (size_t)e * F + f : (size_t)f * NN + e] = P0[e];
        }
    }
    for (int k = 0; k < s.steps; ++k)
        for (int f = 0; f < F; ++f)
            for (int r = 0; r < M; ++r) {
                const size_t src = ((size_t)k * F + f) * M + r;
                const size_t dst = (size_t)k * F * M + (is_soa(v) ? (size_t)r * F + f : (size_t)f * M + r);
                z[dst] = s.z[src];
            }
}

// back to AoS, full covariance
static void unpack(Variant v, int F, const std::vector<float>& x, const std::vector<float>& P,
                   std::vector<float>& xa, std::vector<float>& Pa) {
    xa.assign((size_t)F * N, 0.f);
    Pa.assign((size_t)F * NN, 0.f);
    for (int f = 0; f < F; ++f) {
        for (int r = 0; r < N; ++r) xa[(size_t)f * N + r] = x[is_soa(v) ? (size_t)r * F + f : (size_t)f * N + r];
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < N; ++r) {
                float val;
                if (v == Variant::FusedSoASym) val = P[(size_t)sym_idx(r, c) * F + f];
                else val = P[is_soa(v) ? (size_t)idx(r, c, N) * F + f : (size_t)f * NN + idx(r, c, N)];
                Pa[(size_t)f * NN + idx(r, c, N)] = val;
            }
    }
}

// ---------------------------------------------------------------------------
// Run one variant
// ---------------------------------------------------------------------------
struct Result {
    double ms_per_step = 0;
    double max_x_err_sigma = 0;  // max |x_gpu - x_ref| / sqrt(P_ii)
    double max_P_err = 0;        // max |P_gpu - P_ref| / sqrt(P_ii P_jj)
};

static Result run_variant(Variant v, const Model& m, const Scenario& s, int repeats, int check,
                          const std::vector<double>& xref, const std::vector<double>& Pref) {
    const int F = s.filters;
    std::vector<float> hx, hP, hz;
    pack(v, s, hx, hP, hz);

    float *dX, *dP, *dZ;
    CUDA_CHECK(cudaMalloc(&dX, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, hP.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dZ, hz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dZ, hz.data(), hz.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));
    CublasKF* cb = (v == Variant::Cublas) ? new CublasKF(m, F) : nullptr;

    auto one_step = [&](int k) {
        const float* Zk = dZ + (size_t)k * F * M;
        switch (v) {
            case Variant::Cublas:      cb->step(dX, dP, Zk, st); break;
            case Variant::FusedAoS:    launch_fused<Layout::AoS>(dX, dP, Zk, F, st); break;
            case Variant::FusedSoA:    launch_fused<Layout::SoA>(dX, dP, Zk, F, st); break;
            case Variant::FusedSoASym: launch_fused<Layout::SoASym>(dX, dP, Zk, F, st); break;
            case Variant::Dx:          launch_dx(dX, dP, Zk, F, st); break;
        }
    };
    auto reset_state = [&]() {
        CUDA_CHECK(cudaMemcpy(dX, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    Result res;
    // ---- correctness (also serves as warm-up) ----
    reset_state();
    for (int k = 0; k < s.steps; ++k) one_step(k);
    CUDA_CHECK(cudaStreamSynchronize(st));
    CUDA_CHECK(cudaGetLastError());
    {
        std::vector<float> gx(hx.size()), gP(hP.size()), xa, Pa;
        CUDA_CHECK(cudaMemcpy(gx.data(), dX, gx.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gP.data(), dP, gP.size() * sizeof(float), cudaMemcpyDeviceToHost));
        unpack(v, F, gx, gP, xa, Pa);
        for (int f = 0; f < check; ++f) {
            const double* Pr = &Pref[(size_t)f * NN];
            for (int r = 0; r < N; ++r) {
                const double sr = std::sqrt(Pr[idx(r, r, N)]);
                res.max_x_err_sigma = std::max(res.max_x_err_sigma,
                    std::fabs(xa[(size_t)f * N + r] - xref[(size_t)f * N + r]) / sr);
                for (int c = 0; c < N; ++c) {
                    const double sc = std::sqrt(Pr[idx(c, c, N)]);
                    res.max_P_err = std::max(res.max_P_err,
                        std::fabs(Pa[(size_t)f * NN + idx(r, c, N)] - Pr[idx(r, c, N)]) / (sr * sc));
                }
            }
        }
    }

    // ---- timing: best of `repeats` full runs (state reset excluded from timing) ----
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double best = 1e30;
    for (int rep = 0; rep < repeats; ++rep) {
        reset_state();
        CUDA_CHECK(cudaEventRecord(e0, st));
        for (int k = 0; k < s.steps; ++k) one_step(k);
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, (double)ms);
    }
    CUDA_CHECK(cudaGetLastError());
    res.ms_per_step = best / s.steps;

    delete cb;
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(st));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dZ));
    return res;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Options {
    int filters = 1 << 16, steps = 100, check = 4096, repeats = 5;
    bool sweep = false;
    std::string csv = "results/bench.csv", only;
};

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--filters") o.filters = std::stoi(next());
        else if (a == "--steps") o.steps = std::stoi(next());
        else if (a == "--check") o.check = std::stoi(next());
        else if (a == "--repeats") o.repeats = std::stoi(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--only") o.only = next();
        else if (a == "--sweep") o.sweep = true;
        else {
            std::printf("usage: kf_bench [--filters N] [--steps S] [--check C] [--repeats R]\n"
                        "                [--only variant_name] [--sweep] [--csv file]\n");
            std::exit(a == "--help" ? 0 : 1);
        }
    }
    return o;
}

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int mem_clock_khz = 0, bus_width = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);
    const double peak_bw = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9;  // GB/s (DDR)
    std::printf("GPU: %s (sm_%d%d, %d SMs), theoretical DRAM bandwidth ~%.0f GB/s\n", prop.name,
                prop.major, prop.minor, prop.multiProcessorCount, peak_bw);
    if (prop.major * 100 + prop.minor * 10 != KF_SM)
        std::printf("note: cuBLASDx was configured for SM %d, this GPU is SM %d%d0 (rebuild with -DKF_ARCH=%d%d)\n",
                    KF_SM, prop.major, prop.minor, prop.major, prop.minor);

    const Model m = make_model();
    upload_model(m);

    std::vector<int> sizes;
    if (o.sweep) for (int f = 1 << 10; f <= (1 << 20); f <<= 1) sizes.push_back(f);
    else sizes.push_back(o.filters);

    FILE* csv = std::fopen(o.csv.c_str(), "w");
    if (csv) std::fprintf(csv, "gpu,variant,filters,steps,ms_per_step,mupdates_per_s,gflops,gbytes_per_s,"
                               "flop_per_byte,max_x_err_sigma,max_P_err\n");

    for (int F : sizes) {
        const int steps = o.sweep ? std::max(20, std::min(o.steps, (int)(4e7 / F))) : o.steps;
        const Scenario s = make_scenario(m, F, steps);
        const int check = std::min(o.check, F);
        std::vector<double> xref, Pref;
        const auto t0 = std::chrono::steady_clock::now();
        run_reference(m, s, xref, Pref, 0, check);
        const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\nfilters=%d steps=%d  (CPU double reference on %d filters: %.1f ms/step, ~%.1f ms/step for all)\n",
                    F, steps, check, cpu_ms / steps, cpu_ms / steps * F / check);
        std::printf("%-24s %10s %12s %9s %9s %11s %11s\n", "variant", "us/step", "Mupdates/s",
                    "GFLOP/s", "GB/s", "x err[sig]", "P err");

        for (const auto& vi : kVariants) {
            if (!o.only.empty() && o.only != vi.name) continue;
            const Result r = run_variant(vi.v, m, s, o.repeats, check, xref, Pref);
            const double ups   = F / (r.ms_per_step * 1e-3);
            const double gflop = ups * flops_per_filter_step() / 1e9;
            const double bytes = bytes_per_filter_step(p_floats(vi.v));
            const double gbs   = ups * bytes / 1e9;
            const bool ok = r.max_x_err_sigma < 1e-2 && r.max_P_err < 1e-3;
            std::printf("%-24s %10.1f %12.1f %9.1f %9.1f %11.2e %11.2e %s\n", vi.name, r.ms_per_step * 1e3,
                        ups / 1e6, gflop, gbs, r.max_x_err_sigma, r.max_P_err, ok ? "OK" : "MISMATCH");
            if (csv)
                std::fprintf(csv, "\"%s\",%s,%d,%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.3e,%.3e\n", prop.name, vi.name, F,
                             steps, r.ms_per_step, ups / 1e6, gflop, gbs, flops_per_filter_step() / bytes,
                             r.max_x_err_sigma, r.max_P_err);
        }
    }
    if (csv) { std::fclose(csv); std::printf("\nresults written to %s\n", o.csv.c_str()); }
    std::printf("\nmodel: %.0f FLOP and %.0f bytes (full P) / %.0f bytes (sym P) per filter per step\n",
                flops_per_filter_step(), bytes_per_filter_step(NN), bytes_per_filter_step(NSYM));
    return 0;
}
