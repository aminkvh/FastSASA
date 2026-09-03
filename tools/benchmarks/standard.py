#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROFILES = {
    "quick": {
        "include_nondefault": False,
        "include_huge": False,
        "sr_points": "100",
        "lr_slices": "20",
    },
    "standard": {
        "include_nondefault": True,
        "include_huge": False,
        "sr_points": "100 500",
        "lr_slices": "10 20",
    },
    "stress": {
        "include_nondefault": True,
        "include_huge": True,
        "sr_points": "100 500",
        "lr_slices": "10 20",
    },
}


def _run(command: list[str], cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return subprocess.CompletedProcess(
            command,
            124,
            stdout,
            stderr + f"\ncommand timed out after {timeout_seconds} seconds",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _tool_output(command: list[str], cwd: Path) -> str:
    try:
        proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return proc.stdout.strip()


def _cpu_model() -> str:
    try:
        with Path("/proc/cpuinfo").open() as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_model(repo_root: Path) -> str:
    output = _tool_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], repo_root)
    names = [line.strip() for line in output.splitlines() if line.strip()]
    return "; ".join(names) if names and "not found" not in output.lower() else "unavailable"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_manifest(source: Path, output_dir: Path) -> Path:
    destination = output_dir / "benchmark_corpus_used.csv"
    shutil.copyfile(source, destination)
    return destination


def _combine_csvs(paths: list[tuple[str, Path]], output_path: Path) -> int:
    fieldnames: list[str] | None = None
    rows = []
    for phase, path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = ["benchmark_phase", *(reader.fieldnames or [])]
            for row in reader:
                rows.append({"benchmark_phase": phase, **row})
    if fieldnames is None:
        fieldnames = ["benchmark_phase"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return len(rows)


def _trajectory_args(items: list[str]) -> list[str]:
    if not items:
        return []
    return ["--trajectories", *items]


def _read_fetched_trajectories(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item["entry"] for item in data.get("trajectories", []) if item.get("entry")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard FastSASA benchmark orchestration and write a result bundle.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fastsasa", type=Path, default=Path("build/fastsasa"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmark_corpus.csv"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("benchmark_corpus/structures"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiles/standard_benchmark"))
    parser.add_argument("--trajectories", nargs="*", default=[], help="optional entries as name|topology|trajectory[|extra args]")
    parser.add_argument("--fetch-standard-trajectories", action="store_true", help="fetch one public trajectory from each standard Zenodo trajectory record")
    parser.add_argument("--trajectory-corpus-dir", type=Path, default=Path("benchmark_corpus/trajectories"))
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--trajectory-frames", type=int, default=0)
    parser.add_argument("--trajectory-selection", help="trajectory filter policy: protein, all, or a full FastSASA selection command; required for trajectories unless each trajectory entry provides --filter/--select extra args")
    parser.add_argument("--trajectory-batches", default="1 8 32")
    parser.add_argument("--skip-lr-trajectory-smoke", action="store_true")
    parser.add_argument("--trajectory-lr-frames", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    profile = PROFILES[args.profile]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_manifest(repo_root / args.manifest, output_dir)

    commands: list[dict[str, object]] = []
    trajectories = list(args.trajectories)
    if args.fetch_standard_trajectories:
        trajectory_manifest = args.trajectory_corpus_dir / "trajectory_benchmark_manifest.json"
        trajectory_fetch_command = [
            sys.executable,
            "tools/fetch_benchmark_trajectories.py",
            "--output-dir",
            str(args.trajectory_corpus_dir),
            "--manifest",
            str(trajectory_manifest),
        ]
        started = time.perf_counter()
        proc = _run(trajectory_fetch_command, repo_root, args.timeout_seconds * 20)
        commands.append({
            "phase": "fetch_trajectories",
            "command": trajectory_fetch_command,
            "returncode": proc.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
        if proc.returncode != 0 and not args.allow_failures:
            _write_text(output_dir / "benchmark_run.json", json.dumps(commands, indent=2))
            return proc.returncode
        if proc.returncode == 0:
            trajectories.extend(_read_fetched_trajectories(repo_root / trajectory_manifest))

    if trajectories and args.trajectory_selection is None:
        for item in trajectories:
            fields = item.split("|")
            extra = fields[3] if len(fields) == 4 else ""
            extra_args = shlex.split(extra)
            if "--filter" not in extra_args and "--select" not in extra_args:
                raise SystemExit(
                    "trajectory benchmarks need an explicit atom policy. Add "
                    "--trajectory-selection protein, --trajectory-selection all, "
                    "or per-trajectory extra args such as "
                    "\"name|topology|trajectory|--filter protein\"."
                )

    if not args.skip_fetch:
        fetch_command = [
            sys.executable,
            "tools/fetch_benchmark_corpus.py",
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.corpus_dir),
        ]
        if profile["include_nondefault"]:
            fetch_command.append("--include-nondefault")
        if not profile["include_huge"]:
            fetch_command.append("--exclude-huge")
        started = time.perf_counter()
        proc = _run(fetch_command, repo_root, args.timeout_seconds)
        commands.append({
            "phase": "fetch",
            "command": fetch_command,
            "returncode": proc.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        })
        if proc.returncode != 0 and not args.allow_failures:
            _write_text(output_dir / "benchmark_run.json", json.dumps(commands, indent=2))
            return proc.returncode

    structure_csv = output_dir / "structures.csv"
    structure_command = [
        sys.executable,
        "tools/benchmarks/suite.py",
        "--fastsasa",
        str(args.fastsasa),
        "--corpus-manifest",
        str(args.manifest),
        "--corpus-dir",
        str(args.corpus_dir),
        "--algorithms",
        "shrake-rupley lee-richards",
        "--sr-points",
        str(profile["sr_points"]),
        "--lr-slices",
        str(profile["lr_slices"]),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output",
        str(structure_csv),
    ]
    if profile["include_nondefault"]:
        structure_command.append("--include-nondefault-corpus")
    if profile["include_huge"]:
        structure_command.append("--include-huge-corpus")
    if args.allow_failures:
        structure_command.append("--allow-failures")
    started = time.perf_counter()
    proc = _run(structure_command, repo_root, args.timeout_seconds * 20)
    commands.append({
        "phase": "structures",
        "command": structure_command,
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    })

    csvs = [("structures", structure_csv)]
    if trajectories:
        trajectory_sr_csv = output_dir / "trajectories_sr.csv"
        trajectory_sr_command = [
            sys.executable,
            "tools/benchmarks/suite.py",
            "--fastsasa",
            str(args.fastsasa),
            "--trajectory-algorithms",
            "shrake-rupley",
            "--trajectory-batches",
            args.trajectory_batches,
            "--trajectory-frames",
            str(args.trajectory_frames),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--output",
            str(trajectory_sr_csv),
            *_trajectory_args(trajectories),
        ]
        if args.trajectory_selection is not None:
            trajectory_sr_command.extend(["--trajectory-selection", args.trajectory_selection])
        if args.allow_failures:
            trajectory_sr_command.append("--allow-failures")
        started = time.perf_counter()
        sr_proc = _run(trajectory_sr_command, repo_root, args.timeout_seconds * 20)
        commands.append({
            "phase": "trajectory_sr",
            "command": trajectory_sr_command,
            "returncode": sr_proc.returncode,
            "seconds": time.perf_counter() - started,
            "stdout": sr_proc.stdout,
            "stderr": sr_proc.stderr,
        })
        csvs.append(("trajectory_sr", trajectory_sr_csv))

        if not args.skip_lr_trajectory_smoke:
            trajectory_lr_csv = output_dir / "trajectories_lr_smoke.csv"
            trajectory_lr_command = [
                sys.executable,
                "tools/benchmarks/suite.py",
                "--fastsasa",
                str(args.fastsasa),
                "--trajectory-algorithms",
                "lee-richards",
                "--trajectory-batches",
                "8",
                "--trajectory-frames",
                str(args.trajectory_lr_frames),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--output",
                str(trajectory_lr_csv),
                *_trajectory_args(trajectories),
            ]
            if args.trajectory_selection is not None:
                trajectory_lr_command.extend(["--trajectory-selection", args.trajectory_selection])
            if args.allow_failures:
                trajectory_lr_command.append("--allow-failures")
            started = time.perf_counter()
            lr_proc = _run(trajectory_lr_command, repo_root, args.timeout_seconds * 20)
            commands.append({
                "phase": "trajectory_lr_smoke",
                "command": trajectory_lr_command,
                "returncode": lr_proc.returncode,
                "seconds": time.perf_counter() - started,
                "stdout": lr_proc.stdout,
                "stderr": lr_proc.stderr,
            })
            csvs.append(("trajectory_lr_smoke", trajectory_lr_csv))

    combined_csv = output_dir / "fastsasa_benchmark_results.csv"
    row_count = _combine_csvs(csvs, combined_csv)
    metadata = {
        "profile": args.profile,
        "row_count": row_count,
        "host": platform.node(),
        "os": platform.system(),
        "cpu_model": _cpu_model(),
        "gpu_model": _gpu_model(repo_root),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvidia_smi": _tool_output(["nvidia-smi"], repo_root),
        "nvcc_version": _tool_output(["nvcc", "--version"], repo_root),
        "commands": commands,
    }
    _write_text(output_dir / "benchmark_run.json", json.dumps(metadata, indent=2))

    failing = [item for item in commands if item["returncode"] != 0]
    print(combined_csv)
    return 0 if args.allow_failures or not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
