#!/usr/bin/env python3
"""Final Figure 1: trajectory-engine speedup relative to single-thread
FreeSASA (=1x), averaged across every benchmarked trajectory dataset.

Raw frames/s is not comparable across systems of different atom counts
(a 160k-atom system and a 35k-atom system will never post the same fps
even on identical hardware/software). Reporting each backend's speedup
relative to that same dataset's own FreeSASA 1-thread baseline is
scale-invariant per dataset, so the ratios can be legitimately averaged
across datasets of very different sizes -- unlike raw fps, which cannot.

Usage:
    python3 tools/manuscript_benchmark_2026_figure1_speedup.py
"""

import csv
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026")

DATASETS = ["cx46", "gabaa_baba"]

FREESASA_COLOR = "#009E73"
FASTSASA_CPU_COLOR = "#CC79A7"
VULKAN_COLOR = "#0072B2"
CUDA_COLOR = "#D55E00"

HATCH_THREADS = "///"
HATCH_FP32 = "xxx"
BAR_ORDER = [
    ("freesasa", 1, "fp64", "FreeSASA\n1 thread\n(=1x)", FREESASA_COLOR, ""),
    ("freesasa", 15, "fp64", "FreeSASA\n15 threads", FREESASA_COLOR, HATCH_THREADS),
    ("fastsasa-cpu", 1, "fp64", "FastSASA CPU\n1 thread", FASTSASA_CPU_COLOR, ""),
    ("fastsasa-cpu", 15, "fp64", "FastSASA CPU\n15 threads", FASTSASA_CPU_COLOR, HATCH_THREADS),
    ("fastsasa-vulkan", 0, "fp64", "Vulkan\nFP64", VULKAN_COLOR, ""),
    ("fastsasa-vulkan", 0, "fp32", "Vulkan\nFP32", VULKAN_COLOR, HATCH_FP32),
    ("fastsasa-cuda", 0, "fp64", "CUDA\nFP64", CUDA_COLOR, ""),
    ("fastsasa-cuda", 0, "fp32", "CUDA\nFP32", CUDA_COLOR, HATCH_FP32),
]


def median_fps(rows, tool, threads, precision, panel):
    values = [float(r["frames_per_second"]) for r in rows
              if r["tool"] == tool and int(r["threads"]) == threads
              and r["precision"] == precision and r["panel"] == panel]
    return stats.median(values) if values else None


def dataset_speedups(dataset, panel):
    rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR, dataset, "figure1_raw_timings.csv"))))
    baseline = median_fps(rows, "freesasa", 1, "fp64", panel)
    out = {}
    for tool, threads, precision, label, _, _ in BAR_ORDER:
        v = median_fps(rows, tool, threads, precision, panel)
        out[(tool, threads, precision)] = v / baseline
    return out


def main():
    panel_titles = {
        "complete": "A. Complete-system SASA -- speedup vs. FreeSASA 1-thread\n(geometric mean across 2 benchmarked systems, protein only)",
        "selected": "B. Selected-subunit SASA -- speedup vs. FreeSASA 1-thread\n(geometric mean across 2 benchmarked systems, full-system occlusion retained)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

    for ax, panel in zip(axes, ("complete", "selected")):
        per_dataset = [dataset_speedups(ds, panel) for ds in DATASETS]
        labels, means, err_low, err_high, colors, hatches = [], [], [], [], [], []
        for tool, threads, precision, label, color, hatch in BAR_ORDER:
            vals = [d[(tool, threads, precision)] for d in per_dataset]
            # Geometric mean, not arithmetic: these are ratios (speedup
            # factors), and geometric mean is the standard way to average
            # ratios -- it's invariant to which system is the "baseline"
            # direction and doesn't let one large ratio dominate the way
            # an arithmetic mean of ratios can.
            m = stats.geometric_mean(vals)
            labels.append(label)
            means.append(m)
            err_low.append(m - min(vals))
            err_high.append(max(vals) - m)
            colors.append(color)
            hatches.append(hatch)

        positions = [0, 1.3, 3.0, 4.5, 6.2, 7.15, 8.65, 9.6]

        # Log scale + bar(bottom=0, the matplotlib default) is a known trap:
        # a rectangle patch with its bottom edge AT zero has no finite
        # log-space y-coordinate (log(0) = -inf), so the exported PDF/EPS
        # path for that edge is an enormous (but on-screen-invisible,
        # clipped) coordinate -- Illustrator doesn't clip it the way
        # matplotlib's own renderer does, so it shows up as a huge stray
        # trace/bounding box. Fix: give bar() an explicit positive `bottom`
        # below the smallest real value (every bar here is >= 1.0x) and set
        # log scale before adding the bars, so no artist ever touches y=0.
        y_bottom = 0.5
        ax.set_yscale("log")
        bars = ax.bar(positions, [m - y_bottom for m in means], bottom=y_bottom,
                       width=0.9, yerr=[err_low, err_high], capsize=3,
                       color=colors, hatch=hatches, edgecolor="black", linewidth=0.7)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.8, positions[-1] + 0.9)
        ax.set_ylim(bottom=y_bottom)
        ax.set_ylabel("Speedup vs. FreeSASA 1-thread (x, log scale)")
        ax.set_title(panel_titles[panel], fontsize=10.5)
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        for rect, m in zip(bars, means):
            ax.annotate(f"{m:.1f}x", (rect.get_x() + rect.get_width() / 2, m),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)

    hatch_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=HATCH_THREADS, label="15 threads (vs. 1 thread)"),
        Patch(facecolor="white", edgecolor="black", hatch=HATCH_FP32, label="FP32 (vs. FP64)"),
    ]
    fig.legend(handles=hatch_legend, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.855))

    fig.suptitle(
        "Figure 1. Trajectory-engine speedup vs. single-thread FreeSASA, averaged across two public systems\n"
        "Shrake-Rupley, 100 points/atom, 1.4 A probe, protein-only systems (no solvent, ions, or membrane).\n"
        "Cx46 hemichannel (34,764 protein atoms) and GABA_A receptor, betaAbetaAgamma stoichiometry (27,324 protein atoms).\n"
        "Error bars = min-max speedup across the 2 datasets, not repeat variance (each per-dataset value is itself a\n"
        "median of 3 timed repeats). Raw per-dataset frames/s and citations are in Table S2.",
        fontsize=9.5, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    out_stem = os.path.join(RESULTS_DIR, "figure1_speedup_averaged")
    for ext in ("png", "pdf", "eps"):
        out_path = f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
