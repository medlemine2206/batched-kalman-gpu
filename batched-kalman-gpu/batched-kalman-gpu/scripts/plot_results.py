"""Plots for the README.

    python scripts/plot_results.py --bench results/bench.csv --sweep results/sweep.csv \
                                   --cpu results/cpu_rmse.csv --out results/

Produces:
    roofline.png    – every variant placed on the GPU roofline
    sweep.png       – throughput (million filter updates / s) vs number of filters
    tracking.png    – CPU reference: measurement error vs Kalman error, NEES
Peak numbers are guessed from the GPU name; override with --peak-gflops / --peak-bw.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# FP32 peak (GFLOP/s, non-tensor) and DRAM bandwidth (GB/s), public spec sheets
KNOWN_GPUS = {
    "T4": (8141, 320),
    "L4": (30290, 300),
    "A10": (31240, 600),
    "A100": (19500, 1555),
    "H100": (51220, 2000),
    "RTX 3090": (35580, 936),
    "RTX 4090": (82580, 1008),
    "RTX 4060": (15110, 272),
    "RTX 3060": (12740, 360),
}

COLORS = {
    "cublas_batched": "#9b2226",
    "fused_aos": "#ee9b00",
    "fused_soa": "#0a9396",
    "fused_soa_sym": "#005f73",
    "dx_cublasdx": "#6a4c93",
    "dx_cublasdx_cusolverdx": "#6a4c93",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def guess_peaks(gpu_name):
    for key, val in KNOWN_GPUS.items():
        if key.lower() in gpu_name.lower():
            return val
    return None


def roofline(rows, peak_gflops, peak_bw, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [10 ** (i / 20) for i in range(-20, 61)]  # 0.1 .. 1000 FLOP/byte
    ax.plot(xs, [min(peak_gflops, x * peak_bw) for x in xs], color="#222", lw=2, label="roofline")
    ridge = peak_gflops / peak_bw
    ax.axvline(ridge, color="#999", ls=":", lw=1)
    ax.text(ridge * 1.08, peak_gflops * 0.04, f"ridge point\n{ridge:.1f} FLOP/B", fontsize=8, color="#555")
    for r in rows:
        name = r["variant"]
        ax.scatter(float(r["flop_per_byte"]), float(r["gflops"]), s=70, zorder=3,
                   color=COLORS.get(name, "#333"), label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte of compulsory DRAM traffic)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Batched Kalman filter on {rows[0]['gpu']} ({rows[0]['filters']} filters)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def sweep(rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append((int(r["filters"]), float(r["mupdates_per_s"])))
    for name, pts in by_variant.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                color=COLORS.get(name, "#333"), label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of filters tracked in parallel")
    ax.set_ylabel("million filter updates / s")
    ax.set_title(f"Throughput vs batch size ({rows[0]['gpu']})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def tracking(rows, out):
    k = [int(r["step"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(k, [float(r["rmse_meas"]) for r in rows], color="#ee9b00", label="raw measurements")
    a1.plot(k, [float(r["rmse_kf"]) for r in rows], color="#005f73", label="Kalman estimate")
    a1.set_xlabel("scan")
    a1.set_ylabel("3D position RMSE [m]")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(k, [float(r["nees"]) for r in rows], color="#6a4c93")
    a2.axhline(6, color="#222", ls="--", lw=1, label="expected (state dim = 6)")
    a2.set_xlabel("scan")
    a2.set_ylabel("average NEES")
    a2.set_ylim(0, 15)
    a2.legend()
    a2.grid(alpha=0.3)
    fig.suptitle("CPU double-precision reference: accuracy and statistical consistency")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench")
    p.add_argument("--sweep")
    p.add_argument("--cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--peak-gflops", type=float)
    p.add_argument("--peak-bw", type=float)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.bench:
        rows = read_csv(a.bench)
        peaks = (a.peak_gflops, a.peak_bw) if a.peak_gflops and a.peak_bw else guess_peaks(rows[0]["gpu"])
        if peaks is None:
            raise SystemExit(f"unknown GPU '{rows[0]['gpu']}': pass --peak-gflops and --peak-bw")
        roofline(rows, peaks[0], peaks[1], os.path.join(a.out, "roofline.png"))
    if a.sweep:
        sweep(read_csv(a.sweep), os.path.join(a.out, "sweep.png"))
    if a.cpu:
        tracking(read_csv(a.cpu), os.path.join(a.out, "tracking.png"))


if __name__ == "__main__":
    main()
