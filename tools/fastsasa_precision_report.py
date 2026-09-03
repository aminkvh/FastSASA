#!/usr/bin/env python3
"""Compare backend/precision corpus CSVs against CPU FP64."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


COMPARISON_FIELDS = [
    "pdb_id",
    "category",
    "algorithm",
    "resolution",
    "backend",
    "precision",
    "reference_sasa",
    "total_sasa",
    "absolute_error",
    "relative_error",
    "seconds",
    "reference_seconds",
    "speedup_vs_cpu_fp64",
]

SUMMARY_FIELDS = [
    "backend",
    "precision",
    "cases",
    "failures",
    "max_absolute_error",
    "max_relative_error",
    "median_absolute_error",
    "total_seconds",
    "aggregate_speedup_vs_cpu_fp64",
]


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["pdb_id"], row["algorithm"], row["resolution"], row["iteration"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    reference_rows = read_rows(args.input_dir / "structures_cpu_fp64.csv")
    reference = {key(row): row for row in reference_rows if row["status"] == "pass"}
    comparison_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    modes = (
        ("cpu", "fp64"),
        ("cuda", "fp64"),
        ("cuda", "fp32"),
        ("vulkan", "fp64"),
        ("vulkan", "fp32"),
    )
    for backend, precision in modes:
        rows = read_rows(args.input_dir / f"structures_{backend}_{precision}.csv")
        errors: list[float] = []
        relative_errors: list[float] = []
        total_seconds = 0.0
        reference_seconds = 0.0
        failures = 0
        for row in rows:
            baseline = reference.get(key(row))
            if row["status"] != "pass" or baseline is None:
                failures += 1
                continue
            expected = float(baseline["total_sasa"])
            actual = float(row["total_sasa"])
            absolute_error = abs(actual - expected)
            relative_error = absolute_error / abs(expected) if expected else absolute_error
            seconds = float(row["seconds"])
            baseline_seconds = float(baseline["seconds"])
            errors.append(absolute_error)
            relative_errors.append(relative_error)
            total_seconds += seconds
            reference_seconds += baseline_seconds
            comparison_rows.append({
                "pdb_id": row["pdb_id"],
                "category": row["category"],
                "algorithm": row["algorithm"],
                "resolution": row["resolution"],
                "backend": backend,
                "precision": precision,
                "reference_sasa": f"{expected:.12f}",
                "total_sasa": f"{actual:.12f}",
                "absolute_error": f"{absolute_error:.12g}",
                "relative_error": f"{relative_error:.12g}",
                "seconds": f"{seconds:.6f}",
                "reference_seconds": f"{baseline_seconds:.6f}",
                "speedup_vs_cpu_fp64": f"{baseline_seconds / seconds:.6f}",
            })
        summary_rows.append({
            "backend": backend,
            "precision": precision,
            "cases": len(errors),
            "failures": failures,
            "max_absolute_error": f"{max(errors, default=0.0):.12g}",
            "max_relative_error": f"{max(relative_errors, default=0.0):.12g}",
            "median_absolute_error": f"{statistics.median(errors) if errors else 0.0:.12g}",
            "total_seconds": f"{total_seconds:.6f}",
            "aggregate_speedup_vs_cpu_fp64": (
                f"{reference_seconds / total_seconds:.6f}" if total_seconds else ""
            ),
        })

    write_rows(args.comparison, COMPARISON_FIELDS, comparison_rows)
    write_rows(args.summary, SUMMARY_FIELDS, summary_rows)
    print(args.summary)
    return 0 if all(int(row["failures"]) == 0 for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
