#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


FIELDNAMES = [
    "record_type",
    "name",
    "category",
    "input",
    "topology",
    "algorithm",
    "resolution",
    "backend",
    "precision",
    "batch_size",
    "selection",
    "iteration",
    "status",
    "seconds",
    "atoms",
    "residues",
    "frames",
    "known_frames",
    "total_sasa",
    "gpu_seconds",
    "gpu_frames_per_second",
    "wall_frames_per_second",
    "command",
    "detail",
    "host",
    "os",
    "cpu_model",
    "gpu_model",
    "platform",
    "python",
]

MAX_DETAIL_CHARS = 2000
PROTEIN_SELECTION_COMMAND = "protein"
_CPU_MODEL: str | None = None
_GPU_MODEL: str | None = None


def _split_values(text: str) -> list[str]:
    return [value for value in text.replace(",", " ").split() if value]


def _run(command: list[str], timeout_seconds: int,
         environment: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess[str], float, bool]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
        return completed, time.perf_counter() - start, False
    except subprocess.TimeoutExpired as exc:
        completed = subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=f"timeout after {timeout_seconds} seconds",
        )
        return completed, time.perf_counter() - start, True


def _cpu_model() -> str:
    global _CPU_MODEL

    if _CPU_MODEL is not None:
        return _CPU_MODEL
    try:
        with Path("/proc/cpuinfo").open() as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    _CPU_MODEL = line.split(":", 1)[1].strip()
                    return _CPU_MODEL
    except OSError:
        pass
    _CPU_MODEL = platform.processor() or "unknown"
    return _CPU_MODEL


def _gpu_model() -> str:
    global _GPU_MODEL

    if _GPU_MODEL is not None:
        return _GPU_MODEL
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        _GPU_MODEL = "; ".join(names) if completed.returncode == 0 and names else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        _GPU_MODEL = "unavailable"
    return _GPU_MODEL


def _machine_fields() -> dict[str, str]:
    return {
        "host": platform.node(),
        "os": platform.system(),
        "cpu_model": _cpu_model(),
        "gpu_model": _gpu_model(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _detail(text: str) -> str:
    cleaned = text.strip().replace("\n", " | ")
    if len(cleaned) <= MAX_DETAIL_CHARS:
        return cleaned
    return cleaned[:MAX_DETAIL_CHARS] + f"...[truncated {len(cleaned) - MAX_DETAIL_CHARS} chars]"


def _algorithm_flag(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized in {"sr", "shrake-rupley", "shrakerupley"}:
        return "--shrake-rupley"
    if normalized in {"lr", "lee-richards", "leerichards"}:
        return "--lee-richards"
    raise ValueError(f"unknown algorithm: {name}")


def _read_corpus_entries(path: Path, structure_dir: Path, include_nondefault: bool, include_huge: bool) -> list[dict[str, str]]:
    if not path:
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = []
    for row in rows:
        if not include_nondefault and row.get("default", "").lower() not in {"true", "yes", "1"}:
            continue
        if not include_huge and row.get("tier", "").lower() == "huge":
            continue
        entries.append({
            "name": row["pdb_id"].upper(),
            "category": row.get("category", "corpus"),
            "path": str(structure_dir / f"{row['pdb_id'].lower()}.cif"),
            "extra": row.get("cli_extra", ""),
        })
    return entries


def _structure_entries(paths: list[str]) -> list[dict[str, str]]:
    entries = []
    for path in paths:
        structure = Path(path)
        entries.append({
            "name": structure.stem,
            "category": "user_structure",
            "path": str(structure),
            "extra": "",
        })
    return entries


def _trajectory_entries(items: list[str]) -> list[dict[str, str]]:
    entries = []
    for item in items:
        fields = item.split("|")
        if len(fields) not in {3, 4}:
            raise SystemExit(f"trajectory entries must be name|topology|trajectory[|extra args], got: {item}")
        entries.append({
            "name": fields[0],
            "topology": fields[1],
            "path": fields[2],
            "extra": fields[3] if len(fields) == 4 else "",
        })
    return entries


def _entry_has_trajectory_policy(entry_extra: str) -> bool:
    fields = shlex.split(entry_extra)
    return "--filter" in fields or "--select" in fields


def _trajectory_selection_args(selection: str | None, entry_extra: str) -> list[str]:
    if "--filter" in shlex.split(entry_extra) or "--select" in shlex.split(entry_extra):
        return []
    if selection is None:
        raise ValueError("trajectory benchmarks require --trajectory-selection or per-trajectory --filter/--select extra args")
    normalized = selection.strip()
    if not normalized or normalized.lower() == "all":
        return []
    if normalized.lower() == "protein":
        return ["--filter", PROTEIN_SELECTION_COMMAND]
    return ["--filter", normalized]


def _trajectory_selection_label(selection: str | None, entry_extra: str) -> str:
    if selection is not None:
        return selection
    fields = shlex.split(entry_extra)
    for option in ("--filter", "--select"):
        if option in fields:
            index = fields.index(option)
            if index + 1 < len(fields):
                return fields[index + 1].split(",", 1)[0].strip() or "entry_extra"
    return "entry_extra"


def _structure_row(
    fastsasa: Path,
    entry: dict[str, str],
    algorithm: str,
    resolution: str,
    iteration: int,
    timeout_seconds: int,
    backend: str,
    precision: str,
) -> dict[str, str]:
    command = [
        str(fastsasa),
        _algorithm_flag(algorithm),
        "--resolution",
        str(resolution),
        "--precision",
        precision,
        "--format",
        "json",
        *shlex.split(entry.get("extra", "")),
        entry["path"],
    ]
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = backend
    completed, seconds, timed_out = _run(command, timeout_seconds, environment)
    row = {
        **_machine_fields(),
        "record_type": "structure",
        "name": entry["name"],
        "category": entry["category"],
        "input": entry["path"],
        "algorithm": algorithm,
        "resolution": str(resolution),
        "backend": backend,
        "precision": precision,
        "iteration": str(iteration),
        "status": "pass" if completed.returncode == 0 else "fail",
        "seconds": f"{seconds:.6f}",
        "command": shlex.join(command),
        "detail": _detail(completed.stderr),
    }
    if timed_out:
        row["status"] = "timeout"
    if completed.returncode != 0:
        if completed.stdout.strip():
            row["detail"] = _detail((row["detail"] + " | " + completed.stdout).strip(" |"))
        return row
    try:
        data = json.loads(completed.stdout)
        row["atoms"] = str(len(data.get("atoms", [])))
        row["residues"] = str(len(data.get("residues", [])))
        row["total_sasa"] = f"{float(data['total_sasa']):.12f}"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        row["status"] = "fail"
        row["detail"] = f"could not parse FastSASA JSON: {exc}"
    return row


def _trajectory_row(
    traj_tool: Path,
    entry: dict[str, str],
    algorithm: str,
    frames: int,
    batch_size: str,
    iteration: int,
    timeout_seconds: int,
    trajectory_selection: str | None,
    backend: str,
    precision: str,
) -> dict[str, str]:
    selection_args = _trajectory_selection_args(trajectory_selection, entry.get("extra", ""))
    frame_spec = ":" if frames == 0 else f":{frames}"
    command = [
        str(traj_tool),
        "trajectory",
        "--topology",
        entry["topology"],
        "--trajectory",
        entry["path"],
        "--frames",
        frame_spec,
        "--batch-size",
        str(batch_size),
        "--precision",
        precision,
        "--summary",
        *selection_args,
        *shlex.split(entry.get("extra", "")),
    ]
    if _algorithm_flag(algorithm) == "--lee-richards":
        command.append("--lee-richards")
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = backend
    completed, seconds, timed_out = _run(command, timeout_seconds, environment)
    row = {
        **_machine_fields(),
        "record_type": "trajectory",
        "name": entry["name"],
        "category": "user_trajectory",
        "input": entry["path"],
        "topology": entry["topology"],
        "algorithm": algorithm,
        "resolution": "default",
        "backend": backend,
        "precision": precision,
        "batch_size": str(batch_size),
        "selection": _trajectory_selection_label(trajectory_selection, entry.get("extra", "")),
        "iteration": str(iteration),
        "status": "pass" if completed.returncode == 0 else "fail",
        "seconds": f"{seconds:.6f}",
        "command": shlex.join(command),
        "detail": _detail(completed.stderr),
    }
    if timed_out:
        row["status"] = "timeout"
    if completed.returncode != 0:
        if completed.stdout.strip():
            row["detail"] = _detail((row["detail"] + " | " + completed.stdout).strip(" |"))
        return row
    try:
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        data = next(csv.DictReader(lines))
        row.update({
            "atoms": data.get("atoms", ""),
            "frames": data.get("frames", ""),
            "known_frames": data.get("known_frames", ""),
            "selection": data.get("selection", row["selection"]),
            "total_sasa": data.get("selection_sasa_sum", data.get("total_sasa_sum", "")),
            "gpu_seconds": data.get("gpu_seconds", ""),
            "gpu_frames_per_second": data.get("gpu_frames_per_second", ""),
            "wall_frames_per_second": data.get("wall_frames_per_second", ""),
        })
    except (csv.Error, StopIteration) as exc:
        row["status"] = "fail"
        row["detail"] = f"could not parse trajectory CSV: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a FastSASA-only benchmark suite and write one CSV report.")
    parser.add_argument("--fastsasa", type=Path, default=Path("build/fastsasa"))
    parser.add_argument("--output", type=Path, default=Path("profiles/fastsasa_benchmark_suite.csv"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "vulkan"), default="auto")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--structures", nargs="*", default=[], help="PDB/mmCIF structures to benchmark")
    parser.add_argument("--corpus-manifest", type=Path, help="optional docs/benchmark_corpus.csv manifest")
    parser.add_argument("--corpus-dir", type=Path, default=Path("benchmark_corpus/structures"))
    parser.add_argument("--include-nondefault-corpus", action="store_true")
    parser.add_argument("--include-huge-corpus", action="store_true", help="include tier=huge corpus entries such as capsids")
    parser.add_argument("--trajectories", nargs="*", default=[], help="entries as name|topology|trajectory[|extra args]")
    parser.add_argument("--algorithms", default="shrake-rupley lee-richards")
    parser.add_argument("--sr-points", default="100 500 1000")
    parser.add_argument("--lr-slices", default="10 20")
    parser.add_argument("--trajectory-algorithms", default="shrake-rupley lee-richards")
    parser.add_argument("--trajectory-selection", help="trajectory filter policy: protein, all, or a full FastSASA selection command; required for trajectories unless each trajectory entry provides --filter/--select extra args")
    parser.add_argument("--trajectory-frames", type=int, default=0, help="0 means all frames when supported by the trajectory reader")
    parser.add_argument("--trajectory-batches", default="1 8 32")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300, help="per-run timeout; 0 disables timeouts")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    structures = _structure_entries(args.structures)
    if args.corpus_manifest:
        structures.extend(_read_corpus_entries(
            args.corpus_manifest,
            args.corpus_dir,
            args.include_nondefault_corpus,
            args.include_huge_corpus,
        ))
    trajectories = _trajectory_entries(args.trajectories)
    for entry in trajectories:
        if args.trajectory_selection is None and not _entry_has_trajectory_policy(entry.get("extra", "")):
            raise SystemExit(
                "trajectory benchmarks need an explicit atom policy: add "
                "--trajectory-selection protein, --trajectory-selection all, "
                "or per-trajectory extra args such as |--filter protein"
            )

    rows: list[dict[str, str]] = []
    for entry in structures:
        if not Path(entry["path"]).exists():
            if args.skip_missing:
                continue
            rows.append({
                **_machine_fields(),
                "record_type": "structure",
                "name": entry["name"],
                "category": entry["category"],
                "input": entry["path"],
                "status": "fail",
                "detail": f"missing structure file: {entry['path']}",
            })
            continue
        for algorithm in _split_values(args.algorithms):
            resolutions = _split_values(args.lr_slices if _algorithm_flag(algorithm) == "--lee-richards" else args.sr_points)
            for resolution in resolutions:
                for iteration in range(1, args.iterations + 1):
                    rows.append(_structure_row(
                        args.fastsasa,
                        entry,
                        algorithm,
                        resolution,
                        iteration,
                        args.timeout_seconds,
                        args.backend,
                        args.precision,
                    ))

    for entry in trajectories:
        if not Path(entry["topology"]).exists() or not Path(entry["path"]).exists():
            if args.skip_missing:
                continue
            rows.append({
                **_machine_fields(),
                "record_type": "trajectory",
                "name": entry["name"],
                "input": entry["path"],
                "topology": entry["topology"],
                "status": "fail",
                "detail": "missing topology or trajectory file",
            })
            continue
        for algorithm in _split_values(args.trajectory_algorithms):
            for batch_size in _split_values(args.trajectory_batches):
                for iteration in range(1, args.iterations + 1):
                    rows.append(_trajectory_row(
                        args.fastsasa,
                        entry,
                        algorithm,
                        args.trajectory_frames,
                        batch_size,
                        iteration,
                        args.timeout_seconds,
                        args.trajectory_selection,
                        args.backend,
                        args.precision,
                    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    print(args.output)

    failed = [row for row in rows if row.get("status") != "pass"]
    return 0 if args.allow_failures or not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
