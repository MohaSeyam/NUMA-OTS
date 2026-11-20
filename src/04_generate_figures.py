#!/usr/bin/env python3
"""
04_generate_figures.py — Vector Publication Figure Generator (PDF, embedded fonts)

Generates:
  1. paper/fig1_architecture.pdf : Clean NUMA-OTS architecture & 3-stage evaluation diagram
                                   (explicitly separating live LD_PRELOAD hook vs. C11 DES).
  2. paper/fig2_evaluation.pdf   : (a) Speedup vs. Base across 6 policies with 95% BCa CIs
                                   (auto-scaled y-axis, zero clipping);
                                   (b) Remote DRAM access fraction (1 - alpha_loc).
  3. paper/fig3_sensitivity.pdf  : (a) RQ1 30-trial task-granularity sweep (8N BFS & SpMV);
                                   (b) RQ2 batch-size-calibrated break-even demand R_min(B)
                                   for B in [4..64] across 2N, 4N, and 8N topologies.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def generate_fig1_architecture(out_path):
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_c="#111111"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
                           facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + h - 0.48, body, ha="center", va="top",
                fontsize=7.0, color="#222222", linespacing=1.28)

    # Top row: Live prototype & trace extraction
    box(0.15, 2.35, 3.05, 1.45, "Stage 1a: Live OpenMP + LD_PRELOAD",
        "libnuma_ots_hook.so (GCC 15.2 ABI)\n"
        "Intercepts GOMP_task & GOMP_taskwait\n"
        "depend parser + 64-stripe PHHT + M1 hint\n"
        "Batched move_pages(2) queries & M2 syscalls",
        "#EAF3FC", "#1F5A96")

    box(3.65, 2.35, 3.25, 1.45, "Stage 1b: C11 Min-Heap DES (M1+M2+M3)",
        "Replays deterministic traces (2, 4, 8 nodes)\n"
        "Intel Xeon UPI & AMD EPYC NPS4 latencies\n"
        "M1: depend Eq.(5) | M3: 3-tier SLIT queues\n"
        "M2: Critical-path C_fix + B*C_copy pause",
        "#FDF3E7", "#C26D19")

    box(7.35, 2.35, 3.00, 1.45, "Stage 1c: Statistics & Consistency",
        "30 paired Monte Carlo trials / 13 policies\n"
        "10k paired BCa bootstrap CI + Cohen's d_z\n"
        "Holm-Bonferroni Wilcoxon + 8-way ablation\n"
        "Closed M/M/1 & Roofline cross-check",
        "#EAF7EE", "#247A42")

    # Bottom row: Stage 2, Stage 3, and Hardware/OS scope note
    box(0.15, 0.30, 3.05, 1.50, "Stage 2: QEMU 4-Node Virtual NUMA",
        "Ubuntu 24.04 guest (8 vCPUs, 4 GiB)\n"
        "Explicit ACPI SLIT (10 / 18 / 22)\n"
        "Verifies libnuma, hwloc, move_pages(2)\n"
        "Functional OS/ABI check (no timing claims)",
        "#F5EEFA", "#6C3483")

    box(3.65, 0.30, 3.25, 1.50, "Stage 3: Valgrind Cachegrind Micro-Kernels",
        "Controlled SpMV, BFS, HashJoin, PageRank\n"
        "Identical Ir & Dr+Dw (<0.04% delta)\n"
        "Scaled 16 KiB D1 / 64 KiB LL cache\n"
        "Temporal reuse & model-based energy",
        "#FDECEC", "#A93226")

    box(7.35, 0.30, 3.00, 1.50, "Scope Separation (Live vs. Modeled)",
        "Live LD_PRELOAD: depend parse, PHHT,\n"
        "  M1 priority hint, M2 move_pages(2)\n"
        "C11 DES Only: internal M3 3-tier queues,\n"
        "  SLIT work stealing & multi-socket timing",
        "#F4F6F7", "#566573")

    arrow_kw = dict(arrowstyle="-|>", color="#34495E", lw=1.3, mutation_scale=11)
    ax.annotate("", xy=(3.62, 3.08), xytext=(3.22, 3.08), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.32, 3.08), xytext=(6.92, 3.08), arrowprops=arrow_kw)
    ax.annotate("", xy=(1.68, 1.84), xytext=(1.68, 2.32), arrowprops=arrow_kw)
    ax.annotate("", xy=(5.28, 1.84), xytext=(5.28, 2.32), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.85, 1.84), xytext=(8.85, 2.32), arrowprops=arrow_kw)

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_fig2_evaluation(raw_csv, summary_csv, out_path):
    raw = pd.read_csv(raw_csv)
    summ = pd.read_csv(summary_csv)

    workloads = ["SpMV", "BFS", "HashJoin", "PageRank"]
    nodes_list = [2, 4, 8]
    labels = [f"{w}\n({n}N)" for w in workloads for n in nodes_list]
    x = np.arange(len(labels))

    policies = [
        ("membind0", "Mem0", "#7F8C8D"),
        ("interleave", "Inter", "#5DADE2"),
        ("autonuma", "AutoNUMA", "#AF7AC5"),
        ("numa_ots_m1_m3", "M1+M3", "#F39C12"),
        ("numa_ots_ungated_m2", "Ungated", "#E67E22"),
        ("numa_ots_full", "Full OTS", "#1E8449"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 5.8), sharex=True)
    width = 0.135
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(policies))

    max_spd = 1.0
    for idx_p, (pol, lbl, col) in enumerate(policies):
        spds = []
        errs_lo = []
        errs_hi = []
        rems = []
        for w in workloads:
            for n in nodes_list:
                sub = raw[(raw["workload"] == w) & (raw["nodes"] == n)]
                tb = sub[sub["policy"] == "sim_unguided_base"]["time_ms"].mean()
                tp = sub[sub["policy"] == pol]["time_ms"].mean()
                spd = tb / tp if tp > 0 else 1.0
                spds.append(spd)
                if spd > max_spd:
                    max_spd = spd
                if pol == "numa_ots_full":
                    srow = summ[(summ["workload"] == w) & (summ["nodes"] == n)].iloc[0]
                    errs_lo.append(max(0.0, spd - srow["ci_low"]))
                    errs_hi.append(max(0.0, srow["ci_high"] - spd))
                else:
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                alpha_p = sub[sub["policy"] == pol]["alpha_loc"].mean()
                rems.append(100.0 * (1.0 - alpha_p))

        yerr = [errs_lo, errs_hi] if pol == "numa_ots_full" else None
        bars = ax1.bar(x + offsets[idx_p], spds, width=width, label=lbl, color=col,
                       edgecolor="black", linewidth=0.35, yerr=yerr, capsize=1.8, error_kw={"lw": 0.7})
        if pol == "numa_ots_full":
            for b, val in zip(bars, spds):
                ax1.text(b.get_x() + b.get_width() / 2.0, val + 0.04, f"{val:.2f}x",
                         ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#145A32")

        ax2.bar(x + offsets[idx_p], rems, width=width, label=lbl, color=col,
                edgecolor="black", linewidth=0.35)

    ax1.axhline(1.0, color="#2C3E50", linestyle="--", linewidth=0.8)
    ax1.set_ylim(0.0, max(2.2, max_spd * 1.24))
    ax1.set_ylabel("Speedup vs. Base ($\\times$)")
    ax1.set_title("(a) Simulated Speedup over Default OpenMP Baseline (30 Paired Trials, 95% BCa CI on Full OTS)", pad=6)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=True, framealpha=0.92)

    ax2.set_ylim(0.0, 105.0)
    ax2.set_ylabel("Remote DRAM Misses (%)")
    ax2.set_title("(b) Remote DRAM Access Fraction $1 - \\alpha_{\\mathrm{loc}}$ Across Policies", pad=4)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    plt.tight_layout(h_pad=0.8)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_fig3_sensitivity(gran_csv, out_path):
    gdf = pd.read_csv(gran_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.5))

    # Panel (a): RQ1 30-trial granularity sweep
    tau = gdf["tau_us_est"].to_numpy()
    ax1.plot(tau, gdf["speedup_spmv_full"], "s-", color="#1F618D", lw=1.6, ms=5, label="SpMV (Full OTS)")
    ax1.plot(tau, gdf["speedup_m1_m3"], "^--", color="#D68910", lw=1.6, ms=5, label="BFS (M1+M3 only)")
    ax1.plot(tau, gdf["speedup_full"], "o-", color="#1E8449", lw=1.8, ms=5, label="BFS (Full OTS)")
    ax1.axhline(1.0, color="#566573", linestyle=":", lw=1.0)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Nominal Mean Task Duration $\\tau_{\\mathrm{task}}$ ($\\mu$s, log scale)")
    ax1.set_ylabel("Simulated Speedup vs. Base ($\\times$)")
    ax1.set_title("(a) RQ1: 8-Domain Task Granularity Sweep (30 Trials/Point)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)
    ax1.legend(loc="best", frameon=True)

    # Panel (b): RQ2 Batch-Size-Calibrated Break-Even Demand R_min(B)
    B_vals = np.array([4, 8, 12, 16, 24, 32, 48, 64], dtype=float)
    gamma = 1.5
    c_copy = 1450.0
    configs = [
        ("2N (1-hop, $\\Delta L=65$ ns)", 25600.0, 65.0, "#2E86C1", "o-"),
        ("4N (1-hop, $\\Delta L=65$ ns)", 48000.0, 65.0, "#D68910", "s--"),
        ("8N (1-hop, $\\Delta L=65$ ns)", 92800.0, 65.0, "#C0392B", "^-"),
        ("8N (2-hop, $\\Delta L=102$ ns)", 92800.0, 102.0, "#1E8449", "d-."),
    ]
    for lbl, c_fix, dL, col, fmt in configs:
        c_hat = c_copy + c_fix / B_vals
        r_min = np.floor((gamma * c_hat) / dL) + 1.0
        ax2.plot(B_vals, r_min, fmt, color=col, lw=1.6, ms=4.8, label=lbl)

    ax2.axvline(64.0, color="#34495E", linestyle=":", lw=1.0)
    ax2.text(62.5, 240, "Full batch\n($B=64$)", ha="right", va="center", fontsize=7.5, color="#34495E")
    ax2.set_xlabel("Migration Batch Size $B_{\\mathrm{batch}}$ (pages / move_pages(2) call)")
    ax2.set_ylabel("Break-Even Demand $R_{\\min}(B)$ (misses/epoch)")
    ax2.set_title("(b) RQ2: Batch-Calibrated Break-Even Threshold $R_{\\min}(B)$ ($\\gamma=1.5$)")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout(w_pad=1.6)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    raw_csv     = sys.argv[1] if len(sys.argv) > 1 else "sim_results_raw.csv"
    summary_csv = sys.argv[2] if len(sys.argv) > 2 else "summary_table.csv"
    gran_csv    = sys.argv[3] if len(sys.argv) > 3 else "granularity_sweep.csv"
    out_dir     = sys.argv[4] if len(sys.argv) > 4 else "paper"
    os.makedirs(out_dir, exist_ok=True)

    generate_fig1_architecture(os.path.join(out_dir, "fig1_architecture.pdf"))
    generate_fig2_evaluation(raw_csv, summary_csv, os.path.join(out_dir, "fig2_evaluation.pdf"))
    generate_fig3_sensitivity(gran_csv, os.path.join(out_dir, "fig3_sensitivity.pdf"))
    print(f"[04_generate_figures] Wrote fig1_architecture.pdf, fig2_evaluation.pdf, fig3_sensitivity.pdf to {out_dir}/")


if __name__ == "__main__":
    main()
