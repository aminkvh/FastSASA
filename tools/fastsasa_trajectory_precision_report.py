#!/usr/bin/env python3
"""Summarize trajectory backend/precision accuracy and throughput."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


DETAIL_FIELDS = [
    "name", "algorithm", "backend", "precision", "frames", "atoms",
    "reference_sasa_sum", "total_sasa_sum", "absolute_error",
    "relative_error", "wall_frames_per_second",
    "reference_frames_per_second", "speedup_vs_cpu_fp64",
]

SUMMARY_FIELDS = [
    "algorithm", "backend", "precision", "cases", "max_absolute_error",
    "max_relative_error", "median_speedup_vs_cpu_fp64",
]


def read(path: Path, algorithm: str | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "pass"]
    return [row for row in rows if algorithm is None or row["algorithm"] == algorithm]


def index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["name"], row["algorithm"]): row for row in rows}


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    sr_reference = index(read(args.input_dir / "trajectories_cpu_fp64.csv", "shrake-rupley"))
    lr_reference = index(read(args.input_dir / "trajectories_cpu_fp64_lr_validation.csv"))
    cases = (
        ("shrake-rupley", "cuda", "fp64", "trajectories_cuda_fp64.csv"),
        ("shrake-rupley", "cuda", "fp32", "trajectories_cuda_fp32.csv"),
        ("shrake-rupley", "vulkan", "fp64", "trajectories_vulkan_fp64_sr.csv"),
        ("shrake-rupley", "vulkan", "fp32", "trajectories_vulkan_fp32_sr.csv"),
        ("lee-richards", "cuda", "fp64", "trajectories_cuda_fp64_lr_validation.csv"),
        ("lee-richards", "cuda", "fp32", "trajectories_cuda_fp32_lr_validation.csv"),
        ("lee-richards", "vulkan", "fp64", "trajectories_vulkan_fp64_lr_validation.csv"),
        ("lee-richards", "vulkan", "fp32", "trajectories_vulkan_fp32_lr_validation.csv"),
    )
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for algorithm, backend, precision, filename in cases:
        reference = sr_reference if algorithm == "shrake-rupley" else lr_reference
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        speedups: list[float] = []
        for row in read(args.input_dir / filename, algorithm):
            baseline = reference[(row["name"], algorithm)]
            expected = float(baseline["total_sasa"])
            actual = float(row["total_sasa"])
            absolute_error = abs(actual - expected)
            relative_error = absolute_error / abs(expected) if expected else absolute_error
            fps = float(row["wall_frames_per_second"])
            baseline_fps = float(baseline["wall_frames_per_second"])
            speedup = fps / baseline_fps
            absolute_errors.append(absolute_error)
            relative_errors.append(relative_error)
            speedups.append(speedup)
            detail_rows.append({
                "name": row["name"],
                "algorithm": algorithm,
                "backend": backend,
                "precision": precision,
                "frames": row["frames"],
                "atoms": row["atoms"],
                "reference_sasa_sum": f"{expected:.12f}",
                "total_sasa_sum": f"{actual:.12f}",
                "absolute_error": f"{absolute_error:.12g}",
                "relative_error": f"{relative_error:.12g}",
                "wall_frames_per_second": f"{fps:.6f}",
                "reference_frames_per_second": f"{baseline_fps:.6f}",
                "speedup_vs_cpu_fp64": f"{speedup:.6f}",
            })
        summary_rows.append({
            "algorithm": algorithm,
            "backend": backend,
            "precision": precision,
            "cases": len(absolute_errors),
            "max_absolute_error": f"{max(absolute_errors):.12g}",
            "max_relative_error": f"{max(relative_errors):.12g}",
            "median_speedup_vs_cpu_fp64": f"{statistics.median(speedups):.6f}",
        })

    args.detail.parent.mkdir(parents=True, exist_ok=True)
    write(args.detail, DETAIL_FIELDS, detail_rows)
    write(args.summary, SUMMARY_FIELDS, summary_rows)
    print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
