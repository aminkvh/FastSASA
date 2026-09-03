#!/usr/bin/env python3
"""Generate Figure 2 (VMD workflow performance) from
benchmark_corpus/results/manuscript_2026/cx46/figure2_raw_timings.csv.

Reports elapsed seconds (lower is better), per the spec -- this is an
end-to-end VMD analysis-time comparison, not an engine-throughput figure,
so it deliberately does not share Figure 1's frames/s axis or bar order.
"""

import argparse
import csv
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLOR_VMD = "#E69F00"
COLOR_FASTSASA = {
    "fastsasa_cpu_1": "#CC79A7",
    "fastsasa_cpu_15": "#CC79A7",
    "fastsasa_vulkan": "#0072B2",
    "fastsasa_cuda": "#D55E00",
}

# (tool, threads, display label, color, hatch)
BAR_ORDER = [
    ("measure_sasa", "n/a", "VMD\nmeasure sasa", COLOR_VMD, ""),
    ("fastsasa_cpu", "1", "FastSASA VMD\nCPU 1 thread", COLOR_FASTSASA["fastsasa_cpu_1"], ""),
    ("fastsasa_cpu", "15", "FastSASA VMD\nCPU 15 threads", COLOR_FASTSASA["fastsasa_cpu_15"], "///"),
    ("fastsasa_vulkan", "0", "FastSASA VMD\nVulkan FP64", COLOR_FASTSASA["fastsasa_vulkan"], ""),
    ("fastsasa_cuda", "0", "FastSASA VMD\nCUDA FP64", COLOR_FASTSASA["fastsasa_cuda"], ""),
]

PANEL_TITLES = {
    "complete": "A. Complete-system SASA",
    "selected": "B. Selected-atom SASA\n(full-system occlusion)",
}


def load_rows(csv_path):
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows, tool, threads, panel):
    values = [float(r["elapsed_seconds"]) for r in rows
              if r["tool"] == tool and r["threads"] == threads and r["panel"] == panel]
    if not values:
        return None
    return {"median": stats.median(values), "min": min(values), "max": max(values)}


def plot(dataset, rows, n_frames, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))

    for ax, panel in zip(axes, ("complete", "selected")):
        labels, medians, err_low, err_high, colors, hatches = [], [], [], [], [], []
        for tool, threads, label, color, hatch in BAR_ORDER:
            summary = summarize(rows, tool, threads, panel)
            labels.append(label)
            if summary is None:
                medians.append(0.0); err_low.append(0.0); err_high.append(0.0)
            else:
                medians.append(summary["median"])
                err_low.append(summary["median"] - summary["min"])
                err_high.append(summary["max"] - summary["median"])
            colors.append(color); hatches.append(hatch)

        x = np.arange(len(labels))
        bars = ax.bar(x, medians, yerr=[err_low, err_high], capsize=4,
                       color=colors, hatch=hatches, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlim(-0.7, len(labels) - 0.3)
        ax.set_ylabel(f"Elapsed time, {n_frames} frames (s, lower is better)")
        ax.set_title(PANEL_TITLES[panel], fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        for rect, median in zip(bars, medians):
            if median <= 0:
                continue
            ax.annotate(f"{median:.2f}", (rect.get_x() + rect.get_width() / 2, median),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)

    fig.suptitle(f"Figure 2. End-to-end VMD analysis time ({dataset.upper()})\n"
                 f"Post-load frame loop only; {n_frames} frames, 100 samples/atom, 1.4 Å probe. "
                 f"Not an engine-level throughput comparison -- see Figure 1 for that.",
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    for ext in ("png", "pdf", "eps"):
        out_path = f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None)
        print(f"wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cx46")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_ROOT, f"benchmark_corpus/results/manuscript_2026/{args.dataset}")
    csv_path = os.path.join(out_dir, "figure2_raw_timings.csv")
    rows = load_rows(csv_path)
    n_frames = rows[0]["n_frames"]

    out_stem = os.path.join(out_dir, "figure2_vmd_workflow")
    plot(args.dataset, rows, n_frames, out_stem)


if __name__ == "__main__":
    main()
