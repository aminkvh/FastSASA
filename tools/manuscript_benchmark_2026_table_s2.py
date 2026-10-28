#!/usr/bin/env python3
"""Build Table S2 (trajectory benchmark) and Table 1 (speedup decomposition).

Reads the raw timing CSVs written by `manuscript_benchmark_2026.py
orchestrate` for the four datasets (Cx46 and GABA_A, each with all protein
atoms and with protein heavy atoms), the FP32 deviations written by
`manuscript_benchmark_2026_supplement.py fp32`, and the hardware record from
`manuscript_benchmark_2026_supplement.py hardware`. Throughput cells are the
median frames/s over the timed repeats with the interquartile range in
brackets. Writes table_s2.{md,csv} and table_1_decomposition.md next to the
results.
"""

import csv
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026")

DATASETS = {
    "cx46": ("Cx46 gap-junction channel (docked, 12 connexin subunits; Yue et al. 2021, "
             "Zenodo 10.5281/zenodo.4625961)", "all protein atoms, with H", "AP1"),
    "cx46_heavy": ("Cx46 gap-junction channel", "protein heavy atoms (default)", "AP1"),
    "gabaa_baba": ("GABA_A receptor, βαβαγ (Akbari Ahangar & Li 2026, "
                   "Zenodo 10.5281/zenodo.20705736)", "all protein atoms, with H", "PROA"),
    "gabaa_baba_heavy": ("GABA_A receptor, βαβαγ", "protein heavy atoms (default)", "PROA"),
}
CONFIGS = [
    ("freesasa", "1", "fp64", "FreeSASA 1T"),
    ("freesasa", "15", "fp64", "FreeSASA 15T"),
    ("fastsasa-cpu", "1", "fp64", "FastSASA CPU 1T"),
    ("fastsasa-cpu", "15", "fp64", "FastSASA CPU 15T"),
    ("fastsasa-cuda", "0", "fp64", "CUDA FP64"),
    ("fastsasa-cuda", "0", "fp32", "CUDA FP32"),
    ("fastsasa-vulkan", "0", "fp64", "Vulkan FP64"),
    ("fastsasa-vulkan", "0", "fp32", "Vulkan FP32"),
]
PANELS = ("complete", "selected")


def load_runs(dataset):
    runs, totals = {}, {}
    with open(os.path.join(RESULTS, dataset, "figure1_raw_timings.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["tool"], row["threads"], row["precision"], row["panel"])
            runs.setdefault(key, []).append(float(row["frames_per_second"]))
            totals[key] = row["total_sasa_last_repeat"]
    return runs, totals


def fp64_identical(totals, panel):
    """FastSASA CPU, CUDA and Vulkan FP64 give the same total (as printed)."""
    values = {v for (tool, _, precision, p), v in totals.items()
              if p == panel and precision == "fp64" and tool.startswith("fastsasa")}
    return len(values) == 1


def freesasa_relative(totals, panel):
    reference = float(totals[("fastsasa-cpu", "15", "fp64", panel)])
    return max(abs(float(v) - reference) / reference
               for (tool, _, _, p), v in totals.items() if p == panel and tool == "freesasa")


def fp32_max_relative(dataset, panel):
    path = os.path.join(RESULTS, "supplement", f"fp32_{dataset}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = json.load(fh)["rows"]
    return max(r["max_relative"] for r in rows if r["panel"] == panel)


def cell(values):
    median, q1, q3 = np.median(values), np.percentile(values, 25), np.percentile(values, 75)
    return f"{median:,.1f} [{q1:,.1f}–{q3:,.1f}]"


def main():
    with open(os.path.join(RESULTS, "supplement", "hardware.json")) as fh:
        hardware = json.load(fh)
    header = ["System", "Atom set", "Atoms", "Panel", "Frames"] + [c[3] for c in CONFIGS] + [
        "FastSASA FP64 identical (CPU/CUDA/Vulkan)", "FreeSASA vs FastSASA (relative)",
        "FP32 max relative difference"]
    table, decomposition = [], []
    for dataset, (system, atom_set, segment) in DATASETS.items():
        prepared = np.load(os.path.join(RESULTS, dataset, "prepared_inputs.npz"))
        runs, totals = load_runs(dataset)
        median = {k: float(np.median(v)) for k, v in runs.items()}
        for panel in PANELS:
            atoms = (f"{int(prepared['n_atoms']):,}" if panel == "complete"
                     else f"{int(prepared['n_selected']):,} of {int(prepared['n_atoms']):,} ({segment})")
            fp32 = fp32_max_relative(dataset, panel)
            table.append([system, atom_set, atoms, panel, str(int(prepared["n_frames"]))]
                         + [cell(runs[(t, n, p, panel)]) for t, n, p, _ in CONFIGS]
                         + ["yes" if fp64_identical(totals, panel) else "NO",
                            f"{freesasa_relative(totals, panel):.0e}",
                            "not measured" if fp32 is None else f"{fp32:.1e}"])
            f1, f15 = median[("freesasa", "1", "fp64", panel)], median[("freesasa", "15", "fp64", panel)]
            c1, c15 = median[("fastsasa-cpu", "1", "fp64", panel)], median[("fastsasa-cpu", "15", "fp64", panel)]
            gpu = median[("fastsasa-cuda", "0", "fp64", panel)]
            decomposition.append(f"| {system.split(' (')[0]} | {atom_set} | {panel} | {c1 / f1:.1f}× | "
                                 f"{c15 / c1:.1f}× | {gpu / c15:.1f}× | {gpu / f1:.0f}× | "
                                 f"{c15 / f15:.1f}× | {gpu / f15:.0f}× |")
        cpu_gain = median[("fastsasa-cpu", "15", "fp64", "selected")] / median[("fastsasa-cpu", "15", "fp64", "complete")]
        gpu_gain = median[("fastsasa-cuda", "0", "fp64", "selected")] / median[("fastsasa-cuda", "0", "fp64", "complete")]
        decomposition.append(f"| {system.split(' (')[0]} | {atom_set} | reduced work (selected / "
                             f"complete) | CPU 15T {cpu_gain:.1f}× | CUDA {gpu_gain:.1f}× | | | | |")

    note = (f"Shrake–Rupley, 100 points, 1.4 Å probe, ProtOr radii. Compute-only frames/s, "
            f"median [interquartile range] over 5 timed repeats after a warm-up. Hardware: "
            f"{hardware['cpu']} (15 worker threads), {hardware['gpu']}; {hardware['os']}. "
            f"The selected panel computes SASA for the selection only, with every system atom "
            f"occluding; FreeSASA has no such mode and computes the whole system.")
    with open(os.path.join(RESULTS, "table_s2.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(table)
    with open(os.path.join(RESULTS, "table_s2.md"), "w") as fh:
        fh.write("# Table S2. Trajectory benchmark\n\n" + note + "\n\n")
        fh.write("| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n")
        for row in table:
            fh.write("| " + " | ".join(row) + " |\n")
    with open(os.path.join(RESULTS, "table_1_decomposition.md"), "w") as fh:
        fh.write("# Table 1. Speedup decomposition (FP64, medians from Table S2)\n\n")
        fh.write("| System | Atom set | Panel | Algorithm (CPU 1T / FreeSASA 1T) | Threads (CPU 15T / 1T) "
                 "| GPU (CUDA / CPU 15T) | CUDA / FreeSASA 1T | CPU 15T / FreeSASA 15T | CUDA / FreeSASA 15T |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n" + "\n".join(decomposition) + "\n")
    print(f"wrote {RESULTS}/table_s2.md, table_s2.csv, table_1_decomposition.md")


if __name__ == "__main__":
    main()
