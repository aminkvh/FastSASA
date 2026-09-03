#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(fastsasa: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(fastsasa), *args], text=True, capture_output=True, check=False)


def _rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    return list(csv.DictReader(lines))


def _require_success(proc: subprocess.CompletedProcess[str], command: str) -> None:
    if proc.returncode == 0:
        return
    output = proc.stderr + proc.stdout
    if "no CUDA-capable device" in output and os.environ.get("FASTSASA_REQUIRE_GPU_TESTS") != "1":
        print("fastsasa_trajectory_selection_validation,status,skip,reason,no_cuda_device")
        raise SystemExit(0)
    sys.stderr.write(output)
    raise SystemExit(f"command failed: {command}")


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"missing or invalid {key}: {row}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--topology", type=Path, default=Path("benchmark_corpus/trajectories/gabaa_pore_facing/topology.psf"))
    parser.add_argument("--trajectory", type=Path, default=Path("benchmark_corpus/trajectories/gabaa_pore_facing/trajectory.dcd"))
    args = parser.parse_args()

    if not args.topology.exists() or not args.trajectory.exists():
        print("fastsasa_trajectory_selection_validation,status,skip,reason,missing_trajectory_fixture")
        return 0

    base = [
        "trajectory",
        "--topology",
        str(args.topology),
        "--trajectory",
        str(args.trajectory),
        "--batch-size",
        "1",
        "--filter",
        "protein",
    ]

    proc = _run(args.fastsasa, *base, "--frames", ":1", "--summary")
    _require_success(proc, "--frames :1")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or rows[0].get("frames") != "1":
        raise SystemExit(f"first-frame summary validation failed: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0")
    _require_success(proc, "--frames 0")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or rows[0].get("frame") != "0":
        raise SystemExit(f"single-frame per-frame validation failed: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0:2", "--summary")
    _require_success(proc, "--frames 0:2")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or rows[0].get("frames") != "2":
        raise SystemExit(f"range summary validation failed: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--cpu", "--threads", "2", "--frames", "0:2", "--summary")
    _require_success(proc, "--cpu --frames 0:2")
    cpu_rows = _rows(proc.stdout)
    if len(cpu_rows) != 1 or cpu_rows[0].get("frames") != "2":
        raise SystemExit(f"CPU trajectory summary validation failed: {proc.stdout}")
    if abs(_float(cpu_rows[0], "total_sasa_sum") - _float(rows[0], "total_sasa_sum")) > 1.0e-2:
        raise SystemExit(f"CPU trajectory total mismatch: {cpu_rows[0]} vs {rows[0]}")

    proc = _run(args.fastsasa, *base, "--lee-richards", "--frames", "0", "--summary")
    _require_success(proc, "--lee-richards default resolution")
    lr_default_rows = _rows(proc.stdout)
    proc = _run(args.fastsasa, *base, "--lee-richards", "--resolution", "20", "--frames", "0", "--summary")
    _require_success(proc, "--lee-richards --resolution 20")
    lr_explicit_rows = _rows(proc.stdout)
    if len(lr_default_rows) != 1 or len(lr_explicit_rows) != 1:
        raise SystemExit("Lee-Richards trajectory default resolution validation failed")
    lr_default = _float(lr_default_rows[0], "total_sasa_sum")
    lr_explicit = _float(lr_explicit_rows[0], "total_sasa_sum")
    if abs(lr_default - lr_explicit) > 1.0e-9:
        raise SystemExit(f"Lee-Richards trajectory default resolution mismatch: {lr_default} vs {lr_explicit}")

    proc = _run(args.fastsasa, *base, "--frames", "-1", "--summary")
    _require_success(proc, "--frames -1")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or rows[0].get("frames") != "1":
        raise SystemExit(f"negative-frame validation failed: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0", "--summary", "--select", "r10_PROA, segid PROA and resi 10")
    _require_success(proc, "--filter + --select")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or _float(rows[0], "selection_sasa_sum") <= 0.0:
        raise SystemExit(f"filter+select validation failed: {proc.stdout}")
    if "total_sasa_sum" in rows[0]:
        raise SystemExit(f"selection-only summary should not include total_sasa_sum: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0", "--summary", "--classes")
    _require_success(proc, "--classes summary")
    rows = _rows(proc.stdout)
    if len(rows) != 1:
        raise SystemExit(f"class summary validation failed: {proc.stdout}")
    class_sum = _float(rows[0], "polar_sasa_sum") + _float(rows[0], "apolar_sasa_sum") + _float(rows[0], "unknown_sasa_sum")
    if abs(_float(rows[0], "total_sasa_sum") - class_sum) > 1.0e-6:
        raise SystemExit(f"class summary total mismatch: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0", "--summary", "--classes", "--select", "r10_PROA, segid PROA and resi 10")
    _require_success(proc, "--classes selection summary")
    rows = _rows(proc.stdout)
    if len(rows) != 1:
        raise SystemExit(f"class selection summary validation failed: {proc.stdout}")
    selected_class_sum = _float(rows[0], "polar_sasa_sum") + _float(rows[0], "apolar_sasa_sum") + _float(rows[0], "unknown_sasa_sum")
    if abs(_float(rows[0], "total_sasa_sum") - selected_class_sum) > 1.0e-6:
        raise SystemExit(f"class selected total mismatch: {proc.stdout}")

    proc = _run(args.fastsasa, *base, "--frames", "0", "--summary", "--select", "segid PROA and resi 10")
    _require_success(proc, "--filter + unlabeled --select")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or rows[0].get("selection") != "segid_PROA_and_resi_10" or _float(rows[0], "selection_sasa_sum") <= 0.0:
        raise SystemExit(f"unlabeled filter+select validation failed: {proc.stdout}")

    proc = _run(
        args.fastsasa,
        *base,
        "--frames",
        "0",
        "--summary",
        "--select",
        "r10_PROA, segid PROA and resi 10",
        "--resolution",
        "100",
    )
    _require_success(proc, "--filter + --select + --resolution")
    rows = _rows(proc.stdout)
    if len(rows) != 1 or _float(rows[0], "selection_sasa_sum") <= 0.0:
        raise SystemExit(f"filter+select+resolution validation failed: {proc.stdout}")

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "trajectory_selection.csv"
        proc = _run(
            args.fastsasa,
            *base,
            "--frames",
            "0",
            "--summary",
            "--select",
            "r10_PROA, segid PROA and resi 10",
            "--output",
            str(output_path),
        )
        _require_success(proc, "--filter + --select + --output")
        if proc.stdout.strip():
            raise SystemExit(f"trajectory --output should not write CSV to stdout: {proc.stdout}")
        rows = _rows(output_path.read_text())
        if len(rows) != 1 or _float(rows[0], "selection_sasa_sum") <= 0.0:
            raise SystemExit(f"trajectory --output validation failed: {output_path.read_text()}")

    print("fastsasa_trajectory_selection_validation,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
