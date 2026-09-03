#!/usr/bin/env python3
"""Build Table S2 (public trajectory benchmark corpus) from the raw timing
CSVs produced by tools/manuscript_benchmark_2026.py, in both Markdown and
CSV form.

The table covers the two published trajectory benchmark datasets: Cx46 and
GABA_A betaAbetaAgamma, both protein-only (no solvent/ions/membrane -- see
system_selection in manuscript_benchmark_2026.py's DATASETS dict). Only
datasets actually measured under this methodology get a row; a row with a
real citation next to a wall of "not measured" cells would read as if the
dataset was benchmarked and simply didn't produce a number.

The final column is the maximum per-frame relative difference in total
SASA between the CUDA and Vulkan FP32 results and the FastSASA CPU FP64
reference.
"""

import csv
import json
import os
import statistics as stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOT_MEASURED = "not measured"

DATASET_METADATA = {
    "cx46": {
        "dataset_citation": "Cx46 hemichannel (Yue et al. 2021), Zenodo 10.5281/zenodo.4625961",
        "molecular_system": "Connexin-46 hemichannel, 12 protomers, protein only (no ions)",
        "format": "PSF/DCD",
        "measured": True,
    },
    "gabaa_baba": {
        "dataset_citation": "GABA_A receptor, βαβαγ stoichiometry (Akbari Ahangar & Li 2026), Zenodo 10.5281/zenodo.20705736",
        "molecular_system": "GABA_A receptor pentamer (βαβαγ), protein only (no membrane/water/ions)",
        "format": "PSF/DCD",
        "measured": True,
    },
}

# cmd_numeric() in manuscript_benchmark_2026.py always names its checks the
# same way regardless of dataset -- these keys are not dataset-specific.
FP32_COMPARISON_KEYS = {
    "complete": {"cuda": "cuda_fp32_vs_cpu_fp64", "vulkan": "vulkan_fp32_vs_cpu_fp64"},
    "selected": {"cuda": "selected_cuda_fp32_vs_cpu_fp64", "vulkan": "selected_vulkan_fp32_vs_cpu_fp64"},
}

COLUMNS = [
    "Dataset (citation/DOI)", "Molecular system", "Topology/trajectory format",
    "Total atoms", "Selected atoms", "Frames", "Benchmark type",
    "FreeSASA 1-thread FP64 (frames/s)", "FreeSASA 15-thread FP64 (frames/s)",
    "FastSASA CPU 1-thread FP64 (frames/s)", "FastSASA CPU 15-thread FP64 (frames/s)",
    "FastSASA Vulkan FP64 (frames/s)", "FastSASA CUDA FP64 (frames/s)",
    "FastSASA Vulkan FP32 (frames/s)", "FastSASA CUDA FP32 (frames/s)",
    "Max numerical diff from FP64 CPU ref",
]


def median_fps(rows, tool, threads, precision, panel):
    values = [float(r["frames_per_second"]) for r in rows
              if r["tool"] == tool and int(r["threads"]) == threads
              and r["precision"] == precision and r["panel"] == panel]
    return stats.median(values) if values else None


def fmt(value, digits=1):
    return NOT_MEASURED if value is None else f"{value:.{digits}f}"


def build_dataset_rows(dataset_key):
    out_dir = os.path.join(REPO_ROOT, f"benchmark_corpus/results/manuscript_2026/{dataset_key}")
    csv_path = os.path.join(out_dir, "figure1_raw_timings.csv")
    numeric_path = os.path.join(out_dir, "numerical_checks.json")

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    with open(numeric_path) as fh:
        numeric = json.load(fh)
    diff_by_key = {c["comparison"]: c for c in numeric["checks"]}
    # Selected-atom panel runs through a different code path (the
    # active-center optimization, see manuscript_benchmark_2026.py) than the
    # complete-system panel, and its relative error is measurably larger
    # (smaller per-frame sum -> less cancellation across atoms) -- each
    # panel therefore gets its own max-relative-diff, not a shared value.
    max_rel_diff_by_panel = {
        "complete": max(
            diff_by_key[FP32_COMPARISON_KEYS["complete"]["cuda"]]["max_relative_diff"],
            diff_by_key[FP32_COMPARISON_KEYS["complete"]["vulkan"]]["max_relative_diff"],
        ),
        "selected": max(
            diff_by_key[FP32_COMPARISON_KEYS["selected"]["cuda"]]["max_relative_diff"],
            diff_by_key[FP32_COMPARISON_KEYS["selected"]["vulkan"]]["max_relative_diff"],
        ),
    }

    import numpy as np
    prepared = np.load(os.path.join(out_dir, "prepared_inputs.npz"))
    n_atoms = int(prepared["n_atoms"])
    n_selected = int(prepared["n_selected"])
    n_frames = int(prepared["n_frames"])

    table_rows = []
    for panel, benchmark_type, selected_col in (
        ("complete", "complete-system", NOT_MEASURED),
        ("selected", "selected-atom", str(n_selected)),
    ):
        table_rows.append([
            DATASET_METADATA[dataset_key]["dataset_citation"],
            DATASET_METADATA[dataset_key]["molecular_system"],
            DATASET_METADATA[dataset_key]["format"],
            str(n_atoms),
            selected_col,
            str(n_frames),
            benchmark_type,
            fmt(median_fps(rows, "freesasa", 1, "fp64", panel)),
            fmt(median_fps(rows, "freesasa", 15, "fp64", panel)),
            fmt(median_fps(rows, "fastsasa-cpu", 1, "fp64", panel)),
            fmt(median_fps(rows, "fastsasa-cpu", 15, "fp64", panel)),
            fmt(median_fps(rows, "fastsasa-vulkan", 0, "fp64", panel)),
            fmt(median_fps(rows, "fastsasa-cuda", 0, "fp64", panel)),
            fmt(median_fps(rows, "fastsasa-vulkan", 0, "fp32", panel)),
            fmt(median_fps(rows, "fastsasa-cuda", 0, "fp32", panel)),
            f"{max_rel_diff_by_panel[panel]:.2e} (max relative, FP32 vs FP64 CPU, per-frame)",
        ])
    return table_rows


def main():
    all_rows = []
    skipped = []
    for dataset_key in DATASET_METADATA:
        out_dir = os.path.join(REPO_ROOT, f"benchmark_corpus/results/manuscript_2026/{dataset_key}")
        if not os.path.exists(os.path.join(out_dir, "figure1_raw_timings.csv")):
            skipped.append(dataset_key)
            continue
        all_rows.extend(build_dataset_rows(dataset_key))
    if skipped:
        print(f"skipped (no figure1_raw_timings.csv yet): {', '.join(skipped)}")

    out_dir = os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "table_s2.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(all_rows)
    print(f"wrote {csv_path}")

    md_path = os.path.join(out_dir, "table_s2.md")
    with open(md_path, "w") as fh:
        fh.write("# Table S2. Public trajectory benchmark corpus\n\n")
        fh.write("Shrake-Rupley, 100 points/atom, 1.4 Å probe. Compute-only trajectory-engine "
                 "throughput (frames/s, median of 3 timed repeats after warm-up); this is the "
                 "same compute-only measurement as Figure 1, not an end-to-end/VMD workflow "
                 "number -- see the report for why those are not combined in one column. "
                 "Only benchmarked datasets appear here -- see DATASET_PROVENANCE.md section 3 "
                 "for public trajectories that were identified or partly downloaded but not yet "
                 "run through this methodology (MeTrEx, CBH1); they are intentionally "
                 "not listed as rows with \"not measured\" cells, since that would read as "
                 "measured-but-blank rather than not-yet-run.\n\n")
        fh.write("| " + " | ".join(COLUMNS) + " |\n")
        fh.write("|" + "|".join(["---"] * len(COLUMNS)) + "|\n")
        for row in all_rows:
            fh.write("| " + " | ".join(row) + " |\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
