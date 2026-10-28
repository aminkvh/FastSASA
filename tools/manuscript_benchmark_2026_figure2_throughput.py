#!/usr/bin/env python3
"""Figure 2: absolute trajectory throughput per system (frames/s, log scale).

One panel per system and benchmark panel (whole system / one selected
subunit). Each row is a method; dots are the median over the timed repeats
for the heavy-atom (default) and with-hydrogen atom sets, with the
interquartile range as a line. Reads the same raw CSVs as Table S2.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026")
SYSTEMS = [("cx46", "Cx46 channel", "AP1"), ("gabaa_baba", "GABA_A (βαβαγ)", "PROA")]
ATOM_SETS = [("_heavy", "Heavy atoms (default)", "#2a78d6"), ("", "With hydrogens", "#eb6834")]
CONFIGS = [
    ("freesasa", "1", "fp64", "FreeSASA, 1 thread"),
    ("freesasa", "15", "fp64", "FreeSASA, 15 threads"),
    ("fastsasa-cpu", "1", "fp64", "FastSASA CPU, 1 thread"),
    ("fastsasa-cpu", "15", "fp64", "FastSASA CPU, 15 threads"),
    ("fastsasa-vulkan", "0", "fp64", "FastSASA Vulkan FP64"),
    ("fastsasa-vulkan", "0", "fp32", "FastSASA Vulkan FP32"),
    ("fastsasa-cuda", "0", "fp64", "FastSASA CUDA FP64"),
    ("fastsasa-cuda", "0", "fp32", "FastSASA CUDA FP32"),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def load(dataset):
    runs = {}
    with open(os.path.join(RESULTS, dataset, "figure1_raw_timings.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["tool"], row["threads"], row["precision"], row["panel"])
            runs.setdefault(key, []).append(float(row["frames_per_second"]))
    return runs


def atom_counts(dataset):
    prepared = np.load(os.path.join(RESULTS, dataset, "prepared_inputs.npz"))
    return int(prepared["n_atoms"]), int(prepared["n_selected"])


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=True, constrained_layout=True)
    y = np.arange(len(CONFIGS))[::-1]
    for row, (system, title, segment) in enumerate(SYSTEMS):
        counts = {suffix: atom_counts(system + suffix) for suffix, _, _ in ATOM_SETS}
        for col, panel in enumerate(("complete", "selected")):
            ax = axes[row, col]
            for offset, (suffix, label, color) in zip((0.14, -0.14), ATOM_SETS):
                runs = load(system + suffix)
                for yi, (tool, threads, precision, _) in zip(y, CONFIGS):
                    values = runs[(tool, threads, precision, panel)]
                    q1, q3 = np.percentile(values, [25, 75])
                    ax.plot([q1, q3], [yi + offset] * 2, color=color, linewidth=2, solid_capstyle="round")
                    ax.plot(np.median(values), yi + offset, "o", markersize=6, color=color,
                            markeredgecolor="white", markeredgewidth=1.0,
                            label=label if yi == y[0] else None)
            ax.set_xscale("log")
            ax.set_xlim(4, 10000)
            ax.set_yticks(y)
            ax.set_yticklabels([c[3] for c in CONFIGS] if col == 0 else [], fontsize=8, color=INK)
            heavy, with_h = counts["_heavy"], counts[""]
            what = (f"whole system\n{heavy[0]:,} / {with_h[0]:,} atoms" if panel == "complete"
                    else f"one subunit ({segment})\n{heavy[1]:,} / {with_h[1]:,} atoms")
            ax.set_title(f"{'ABCD'[2 * row + col]}  {title}, {what}", loc="left", fontsize=8.5, color=INK)
            ax.grid(axis="x", which="major", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
            if row == 1:
                ax.set_xlabel("Frames per second (log scale)", fontsize=9, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK)
    for ext in ("png", "pdf"):
        path = os.path.join(RESULTS, f"figure2_throughput.{ext}")
        fig.savefig(path, dpi=300)
    print(f"wrote {RESULTS}/figure2_throughput.png and .pdf")


if __name__ == "__main__":
    main()
