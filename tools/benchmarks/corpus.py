#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


FIELDNAMES = [
    "pdb_id",
    "category",
    "tier",
    "label",
    "backend",
    "precision",
    "algorithm",
    "resolution",
    "iteration",
    "status",
    "seconds",
    "atoms",
    "residues",
    "total_sasa",
    "structure_path",
    "detail",
]

MAX_DETAIL_CHARS = 2000


def _detail(text: str) -> str:
    cleaned = text.strip().replace("\n", " | ")
    if len(cleaned) <= MAX_DETAIL_CHARS:
        return cleaned
    return cleaned[:MAX_DETAIL_CHARS] + f"...[truncated {len(cleaned) - MAX_DETAIL_CHARS} chars]"


def _split_values(text: str) -> list[str]:
    return [value for value in text.replace(",", " ").split() if value]


def _selected(row: dict[str, str], categories: set[str] | None, ids: set[str] | None, include_nondefault: bool) -> bool:
    pdb_id = row["pdb_id"].upper()
    if ids is not None and pdb_id not in ids:
        return False
    if categories is not None and row["category"].lower() not in categories:
        return False
    if not include_nondefault and row.get("default", "").lower() not in {"true", "yes", "1"}:
        return False
    return True


def _read_manifest(path: Path, categories: set[str] | None, ids: set[str] | None, include_nondefault: bool) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if _selected(row, categories, ids, include_nondefault)]


def _algorithm_flag(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized in {"sr", "shrake-rupley", "shrakerupley"}:
        return "--shrake-rupley"
    if normalized in {"lr", "lee-richards", "leerichards"}:
        return "--lee-richards"
    raise ValueError(f"unknown algorithm: {name}")


def _run_one(fastsasa: Path, row: dict[str, str], structure_path: Path,
             algorithm: str, resolution: str, iteration: int,
             backend: str, precision: str) -> dict[str, str]:
    extra = shlex.split(row.get("cli_extra", ""))
    command = [
        str(fastsasa),
        _algorithm_flag(algorithm),
        "--resolution",
        str(resolution),
        "--precision",
        precision,
        "--format",
        "json",
        *extra,
        str(structure_path),
    ]
    start = time.perf_counter()
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = backend
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    seconds = time.perf_counter() - start

    result = {
        "pdb_id": row["pdb_id"].upper(),
        "category": row["category"],
        "tier": row["tier"],
        "label": row["label"],
        "backend": backend,
        "precision": precision,
        "algorithm": algorithm,
        "resolution": str(resolution),
        "iteration": str(iteration),
        "status": "pass" if completed.returncode == 0 else "fail",
        "seconds": f"{seconds:.6f}",
        "structure_path": str(structure_path),
        "detail": _detail(completed.stderr),
    }
    if completed.returncode != 0:
        if completed.stdout.strip():
            result["detail"] = _detail((result["detail"] + " | " + completed.stdout).strip(" |"))
        return result

    try:
        data = json.loads(completed.stdout)
        result["atoms"] = str(len(data.get("atoms", [])))
        result["residues"] = str(len(data.get("residues", [])))
        result["total_sasa"] = f"{float(data['total_sasa']):.12f}"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        result["status"] = "fail"
        result["detail"] = f"could not parse FastSASA JSON: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a FastSASA-only benchmark over the public structure corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmark_corpus.csv"))
    parser.add_argument("--structure-dir", type=Path, default=Path("benchmark_corpus/structures"))
    parser.add_argument("--fastsasa", type=Path, default=Path("build/fastsasa"))
    parser.add_argument("--output", type=Path, default=Path("profiles/corpus_benchmark.csv"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "vulkan"), default="auto")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--algorithms", default="shrake-rupley lee-richards")
    parser.add_argument("--points", default="100 500", help="Shrake-Rupley point counts")
    parser.add_argument("--slices", default="20", help="Lee-Richards slice counts")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--include-nondefault", action="store_true", help="also run opt-in large/heavy structures")
    parser.add_argument("--categories", help="comma-separated category filter")
    parser.add_argument("--ids", help="comma-separated PDB IDs")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--allow-failures", action="store_true", help="write failed rows but exit 0")
    args = parser.parse_args()

    categories = {value.strip().lower() for value in args.categories.split(",") if value.strip()} if args.categories else None
    ids = {value.strip().upper() for value in args.ids.split(",") if value.strip()} if args.ids else None
    rows = _read_manifest(args.manifest, categories, ids, args.include_nondefault)
    if not rows:
        print("No benchmark corpus entries matched the filters.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[dict[str, str]] = []
    missing = 0
    for row in rows:
        structure_path = args.structure_dir / f"{row['pdb_id'].lower()}.cif"
        if not structure_path.exists():
            missing += 1
            message = f"missing structure file: {structure_path}"
            if args.skip_missing:
                print(message, file=sys.stderr)
                continue
            benchmark_rows.append({
                "pdb_id": row["pdb_id"].upper(),
                "category": row["category"],
                "tier": row["tier"],
                "label": row["label"],
                "status": "fail",
                "structure_path": str(structure_path),
                "detail": message,
            })
            continue
        for algorithm in _split_values(args.algorithms):
            resolutions = _split_values(args.slices if _algorithm_flag(algorithm) == "--lee-richards" else args.points)
            for resolution in resolutions:
                for iteration in range(1, args.iterations + 1):
                    benchmark_rows.append(_run_one(
                        args.fastsasa,
                        row,
                        structure_path,
                        algorithm,
                        resolution,
                        iteration,
                        args.backend,
                        args.precision,
                    ))

    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for benchmark_row in benchmark_rows:
            writer.writerow(benchmark_row)
    print(f"wrote {len(benchmark_rows)} rows to {args.output}", file=sys.stderr)

    failed = [row for row in benchmark_rows if row.get("status") != "pass"]
    if missing and not args.skip_missing:
        print("Run tools/fetch_benchmark_corpus.py before benchmarking, or pass --skip-missing.", file=sys.stderr)
    return 0 if args.allow_failures or not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
