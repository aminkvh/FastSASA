#!/usr/bin/env python3
"""Combined Figure 1: FP64/FP32 trajectory throughput, complete-system and
selected-subunit SASA. Every tool runs on the identical system, frame
count, and selection -- RustSASA and VMD are excluded because neither can
meet that identical-system bar for the datasets used here (RustSASA cannot
parse free ions at all; VMD's per-frame subprocess cost forces a reduced
frame count) -- see REPORT.md and PRECISION_PANEL_NOTES.md for those
results on their own terms.

Usage:
    python3 tools/manuscript_benchmark_2026_figure1_combined.py --dataset cx46 \\
        --panel-a-title "A. Complete-system SASA (35,172 atoms)" \\
        --panel-b-title "B. Selected-protomer SASA\\n(segid AP1, 1,530 atoms, full-system occlusion)" \\
        --system-label "Cx46 hemichannel" \\
        --system-desc "Identical 35,172-atom system, 250 frames, and segid-AP1 (1,530-atom) selection for every method shown."
"""

import argparse
import csv
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FREESASA_COLOR = "#009E73"
FASTSASA_CPU_COLOR = "#CC79A7"
VULKAN_COLOR = "#0072B2"
CUDA_COLOR = "#D55E00"

# (tool, threads, precision, label, color, hatch)
# Exactly two hatches, each meaning one thing everywhere it appears: "///"
# always means "15 threads" (vs. that same tool's 1-thread bar), "xxx"
# always means "FP32" (vs. that same backend's FP64 bar). No hatch = the
# baseline configuration (1 thread, or FP64 for the GPU backends).
HATCH_THREADS = "///"
HATCH_FP32 = "xxx"
BAR_ORDER = [
    ("freesasa", 1, "fp64", "FreeSASA\n1 thread", FREESASA_COLOR, ""),
    ("freesasa", 15, "fp64", "FreeSASA\n15 threads", FREESASA_COLOR, HATCH_THREADS),
    ("fastsasa-cpu", 1, "fp64", "FastSASA CPU\n1 thread", FASTSASA_CPU_COLOR, ""),
    ("fastsasa-cpu", 15, "fp64", "FastSASA CPU\n15 threads", FASTSASA_CPU_COLOR, HATCH_THREADS),
    ("fastsasa-vulkan", 0, "fp64", "Vulkan\nFP64", VULKAN_COLOR, ""),
    ("fastsasa-vulkan", 0, "fp32", "Vulkan\nFP32", VULKAN_COLOR, HATCH_FP32),
    ("fastsasa-cuda", 0, "fp64", "CUDA\nFP64", CUDA_COLOR, ""),
    ("fastsasa-cuda", 0, "fp32", "CUDA\nFP32", CUDA_COLOR, HATCH_FP32),
]


def load_rows(out_dir):
    with open(os.path.join(out_dir, "figure1_raw_timings.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def summarize(rows, tool, threads, precision, panel):
    values = [float(r["frames_per_second"]) for r in rows
              if r["tool"] == tool and str(r["threads"]) == str(threads)
              and r["precision"] == precision and r["panel"] == panel]
    if not values:
        return None
    return {"median": stats.median(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="cx46",
                         help="subdirectory under benchmark_corpus/results/manuscript_2026/ holding figure1_raw_timings.csv")
    parser.add_argument("--out-dir", default=None, help="override output directory (default: same as --dataset's)")
    parser.add_argument("--panel-a-title", default="A. Complete-system SASA (35,172 atoms)")
    parser.add_argument("--panel-b-title", default="B. Selected-protomer SASA\nsegid AP1, 1,530 atoms, full-system occlusion")
    parser.add_argument("--system-label", default="Cx46 hemichannel")
    parser.add_argument("--system-desc", default="Identical 35,172-atom system, 250 frames, "
                                                   "and segid-AP1 (1,530-atom) selection for every method shown.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(REPO_ROOT, f"benchmark_corpus/results/manuscript_2026/{args.dataset}")
    panel_titles = {"complete": args.panel_a_title, "selected": args.panel_b_title}

    rows = load_rows(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))

    for ax, panel in zip(axes, ("complete", "selected")):
        labels, medians, err_low, err_high, colors, hatches = [], [], [], [], [], []
        for tool, threads, precision, label, color, hatch in BAR_ORDER:
            summary = summarize(rows, tool, threads, precision, panel)
            labels.append(label)
            if summary is None:
                medians.append(0.0); err_low.append(0.0); err_high.append(0.0)
            else:
                medians.append(summary["median"])
                err_low.append(summary["median"] - summary["min"])
                err_high.append(summary["max"] - summary["median"])
            colors.append(color); hatches.append(hatch)

        # A thin gap between the two members of each FP64/FP32 pair (Vulkan,
        # CUDA) and a wider gap before/between groups, so the pairing also
        # reads from bar position, not just the hatch legend below.
        # FastSASA CPU's label text is the widest of the eight, so its pair
        # gets extra room.
        positions = [0, 1.3, 3.0, 4.5, 6.2, 7.15, 8.65, 9.6]

        # Log scale + bar(bottom=0, the matplotlib default) is a known trap:
        # a rectangle patch with its bottom edge AT zero has no finite
        # log-space y-coordinate (log(0) = -inf), so the exported PDF/EPS
        # path for that edge is an enormous (but on-screen-invisible,
        # clipped) coordinate -- Illustrator doesn't clip it the way
        # matplotlib's own renderer does, so it shows up as a huge stray
        # trace/bounding box. Fix: give bar() an explicit positive `bottom`
        # below the smallest real value and set log scale before adding the
        # bars, so no artist ever touches y=0.
        positive_medians = [m for m in medians if m > 0]
        y_bottom = min(positive_medians) * 0.5 if positive_medians else 0.5
        ax.set_yscale("log")
        bars = ax.bar(positions, [max(m, y_bottom) - y_bottom for m in medians], bottom=y_bottom,
                       width=0.9, yerr=[err_low, err_high], capsize=3,
                       color=colors, hatch=hatches, edgecolor="black", linewidth=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.8, positions[-1] + 0.9)
        ax.set_ylim(bottom=y_bottom)
        ax.set_ylabel("Trajectory throughput (frames/s, log scale)")
        ax.set_title(panel_titles[panel], fontsize=10.5)
        ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        for rect, median in zip(bars, medians):
            if median <= 0:
                continue
            ax.annotate(f"{median:.1f}", (rect.get_x() + rect.get_width() / 2, median),
                        textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7.5)

    hatch_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=HATCH_THREADS, label="15 threads (vs. 1 thread)"),
        Patch(facecolor="white", edgecolor="black", hatch=HATCH_FP32, label="FP32 (vs. FP64)"),
    ]
    fig.legend(handles=hatch_legend, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.855))

    fig.suptitle(
        f"Figure 1. Trajectory-engine throughput, complete-system vs. selected-subunit SASA ({args.system_label})\n"
        f"Shrake-Rupley, 100 points/atom, 1.4 Å probe. {args.system_desc}",
        fontsize=10.5, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_stem = os.path.join(out_dir, "figure1_trajectory_throughput")
    for ext in ("png", "pdf", "eps"):
        out_path = f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
