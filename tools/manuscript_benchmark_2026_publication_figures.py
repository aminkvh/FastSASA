#!/usr/bin/env python3
"""Focused publication benchmark runner and figure generator for FastSASA.

The output is intentionally figure-oriented: every CSV row records exactly what
was measured, and every figure has a paired pipeline/compute-time panel where
that distinction is meaningful for the tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_log_bar(ax, x, heights, **kwargs):
    """ax.bar() on a log-scale axis, without the invisible-huge-path trap.

    A bar's default bottom=0 has no finite log-space y-coordinate (log(0) =
    -inf); the exported PDF/EPS path for that edge becomes an enormous but
    on-screen-invisible coordinate that matplotlib's own renderer clips but
    Illustrator's importer does not, showing up there as a large stray
    trace/bounding box. Fix: set log scale before drawing, and give bar() an
    explicit positive `bottom` below the smallest real value so no artist
    ever touches y=0. Zero/negative heights (e.g. "not measured" placeholder
    rows) are drawn as zero-height bars sitting at that floor, not at y=0.
    """
    heights = list(heights)
    positive = [h for h in heights if h > 0]
    y_bottom = min(positive) * 0.5 if positive else 0.5
    ax.set_yscale("log")
    bars = ax.bar(x, [max(h, y_bottom) - y_bottom for h in heights], bottom=y_bottom, **kwargs)
    ax.set_ylim(bottom=y_bottom)
    return bars


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "profiles" / ("publication_benchmark_focused_" + time.strftime("%Y%m%d_%H%M%S"))
PYTHON_EXECUTABLE = sys.executable
PYTHON_BENCH_PATH: Path | None = None
FREESASA_CANDIDATES: list[Path] = []
RUSTSASA_CANDIDATES = [
    Path.home() / ".cargo/bin/rust-sasa",
]

TOOL_ORDER = [
    "RustSASA",
    "mdsasa-bolt/RustSASA",
    "VMD",
    "FreeSASA",
    "FastSASA CPU",
    "FastSASA CUDA",
    "FastSASA Vulkan",
]
TOOL_COLORS = {
    "RustSASA": "#0072B2",
    "mdsasa-bolt/RustSASA": "#56B4E9",
    "VMD": "#E69F00",
    "FreeSASA": "#009E73",
    "FastSASA CPU": "#CC79A7",
    "FastSASA CUDA": "#D55E00",
    "FastSASA Vulkan": "#0072B2",
}
HATCHES = {"1": "", "8": "///", "15": "///", "default": "", "auto": ""}
SHORT_TOOL_LABELS = {
    "RustSASA": "Rust",
    "mdsasa-bolt/RustSASA": "Bolt",
    "VMD": "VMD",
    "FreeSASA": "Free",
    "FastSASA CPU": "FastSASA CPU",
    "FastSASA CUDA": "FastSASA CUDA",
    "FastSASA Vulkan": "FastSASA Vulkan",
}

STRUCTURES = [
    {
        "case": "1UBQ",
        "label": "1UBQ small protein (660 atoms)",
        "path": ROOT / "tests/data/1ubq.pdb",
        "selected_fastsasa": "chain A and resi 7",
        "selected_vmd": "protein and chain A and resid 7",
        "selected_rust_residue": 7,
    },
    {
        "case": "2ISK",
        "label": "2ISK large protein (15,127 atoms)",
        "path": ROOT / "tests/data/2isk.pdb",
        "selected_fastsasa": "chain A and resi 10",
        "selected_vmd": "protein and chain A and resid 10",
        "selected_rust_residue": 10,
    },
    {
        "case": "3BKR",
        "label": "3BKR medium protein (1,160 atoms)",
        "path": ROOT / "tests/data/3bkr.pdb",
        "selected_fastsasa": "chain A and resi 10",
        "selected_vmd": "protein and chain A and resid 10",
        "selected_rust_residue": 10,
    },
]

SCALING_STRUCTURES = [
    {"case": "1UBQ", "label": "1UBQ (660 atoms)", "path": ROOT / "tests/data/1ubq.pdb"},
    {"case": "3BKR", "label": "3BKR (1,160 atoms)", "path": ROOT / "tests/data/3bkr.pdb"},
    {"case": "5DX9", "label": "5DX9 (2,569 atoms)", "path": ROOT / "tests/data/5dx9.pdb"},
    {"case": "5HDN", "label": "5HDN (5,033 atoms)", "path": ROOT / "tests/data/5hdn.pdb"},
    {"case": "1SUI", "label": "1SUI (7,446 atoms)", "path": ROOT / "tests/data/1sui.pdb"},
    {"case": "2ISK", "label": "2ISK (15,127 atoms)", "path": ROOT / "tests/data/2isk.pdb"},
]

GABAA_TOP = ROOT / "benchmark_corpus/trajectories/gabaa_pore_facing/topology.psf"
GABAA_TRAJ = ROOT / "benchmark_corpus/trajectories/gabaa_pore_facing/trajectory.dcd"
GABAA_SELECTION_FASTSASA = "segname PROA and resid 155"
GABAA_SELECTION_MDANALYSIS = "segid PROA and resid 155"
GABAA_SELECTION_VMD = "protein and segname PROA and resid 155"

FIELDS = [
    "figure",
    "case",
    "case_label",
    "tool",
    "backend",
    "algorithm",
    "resolution",
    "probe_radius",
    "precision",
    "threads",
    "context",
    "selection_expression",
    "selection_semantics",
    "frames",
    "atoms",
    "compute_seconds",
    "pipeline_seconds",
    "compute_frames_per_second",
    "pipeline_frames_per_second",
    "total_sasa",
    "selection_sasa",
    "absolute_error_vs_fp64",
    "relative_error_vs_fp64",
    "speedup_vs_fp64",
    "status",
    "notes",
    "command",
]


def run_command(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "fastsasa_matplotlib"))
    if env:
        merged.update(env)
    return subprocess.run(command, cwd=ROOT, env=merged, text=True, capture_output=True, timeout=timeout)


def tool_path(path: Path | None, candidates: list[Path], executable: str) -> Path | None:
    if path is not None:
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = shutil.which(executable)
    return Path(found) if found else None


def python_benchmark_env(*, threads: int | None = None) -> dict[str, str]:
    env = {}
    if PYTHON_BENCH_PATH is not None:
        env["PYTHONPATH"] = str(PYTHON_BENCH_PATH)
    if threads is not None:
        env["RAYON_NUM_THREADS"] = str(threads)
        env["OMP_NUM_THREADS"] = str(threads)
    return env


def median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


def count_atoms(path: Path) -> int:
    count = 0
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                count += 1
    return count


def parse_log_total(text: str) -> float:
    match = re.search(r"^Total\s*:\s*([0-9.eE+-]+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError("could not parse total SASA")
    return float(match.group(1))


def parse_log_selection(text: str, name: str = "target") -> float:
    match = re.search(rf"^{re.escape(name)}\s*:\s*([0-9.eE+-]+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"could not parse selection {name}")
    return float(match.group(1))


def parse_traj_summary(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("FastSASA: warning")]
    rows = list(csv.DictReader(lines))
    if len(rows) != 1:
        raise ValueError("could not parse trajectory summary CSV")
    return rows[0]


def parse_rustsasa_protein_json(path: Path) -> float:
    data = json.loads(path.read_text())
    return float(data["Protein"]["global_total"])


def parse_rustsasa_residue_json(path: Path, *, chain_id: str, residue_number: int) -> float:
    data = json.loads(path.read_text())
    for residue in data.get("Residue", []):
        if residue.get("chain_id") == chain_id and int(residue.get("serial_number")) == residue_number:
            return float(residue.get("value"))
    raise ValueError(f"could not find RustSASA residue {chain_id}:{residue_number}")


def base_row(**values: object) -> dict[str, str]:
    row = {field: "" for field in FIELDS}
    row.update({
        "precision": "fp64/double",
        "probe_radius": "1.4",
        "status": "pass",
    })
    row.update({key: "" if value is None else str(value) for key, value in values.items()})
    return row


def timed_repeats(func, repeats: int) -> tuple[float, object]:
    seconds = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        seconds.append(time.perf_counter() - start)
    return median(seconds), result


def add_failure(rows: list[dict[str, str]], *, figure: str, case: str, case_label: str, tool: str, notes: str) -> None:
    rows.append(base_row(figure=figure, case=case, case_label=case_label, tool=tool, status="fail", notes=notes[-800:]))


def measure_fastsasa_static(rows, exe: Path, tool: str, structure: dict, *, algorithm: str, resolution: int, threads: int | None, selected: bool, repeats: int, figure: str | None = None, backend: str | None = None) -> None:
    command = [str(exe)]
    if backend is None:
        # Legacy inference for old two-way callers; explicit backend= is preferred.
        backend = "cuda" if tool.endswith("GPU") else "cpu"
    if backend == "cpu":
        command.append("--cpu")
    else:
        command += ["--backend", backend, "--no-cpu-fallback"]
    if threads is not None:
        command += ["--threads", str(threads)]
    if algorithm == "SR":
        command += ["--shrake-rupley", "--resolution", str(resolution)]
    else:
        command += ["--lee-richards", "--resolution", str(resolution)]
    command += ["--format", "log"]
    selection = structure["selected_fastsasa"] if selected else ""
    if selected:
        command += ["--select", f"target, {selection}"]
    command.append(str(structure["path"]))

    def once():
        proc = run_command(command)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        total = parse_log_total(proc.stdout)
        sel = parse_log_selection(proc.stdout) if selected else ""
        return total, sel

    wall, result = timed_repeats(once, repeats)
    total, sel = result
    rows.append(base_row(
        figure=figure or ("fig1" if not selected else "fig3"),
        case=structure["case"],
        case_label=structure["label"],
        tool=tool,
        backend=backend,
        algorithm=f"{algorithm}{resolution}",
        resolution=resolution,
        threads=threads if threads is not None else "auto",
        context="full structure" if not selected else "full structure context, reported residue",
        selection_expression=selection,
        selection_semantics="FastSASA/FreeSASA-style selected atom SASA" if selected else "full context",
        frames=1,
        atoms=count_atoms(structure["path"]),
        pipeline_seconds=wall,
        pipeline_frames_per_second=1.0 / wall if wall > 0 else "",
        total_sasa=total,
        selection_sasa=sel,
        command=" ".join(command),
    ))


def measure_freesasa_static(rows, freesasa: Path, structure: dict, *, algorithm: str, resolution: int, threads: int, selected: bool, repeats: int, figure: str | None = None) -> None:
    command = [str(freesasa)]
    command += ["--shrake-rupley" if algorithm == "SR" else "--lee-richards"]
    command += ["--resolution", str(resolution), f"--n-threads={threads}", "--format=log", "--no-warnings"]
    selection = structure["selected_fastsasa"] if selected else ""
    if selected:
        command += [f"--select=target, {selection}"]
    command.append(str(structure["path"]))

    def once():
        proc = run_command(command)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        total = parse_log_total(proc.stdout)
        sel = parse_log_selection(proc.stdout) if selected else ""
        return total, sel

    wall, result = timed_repeats(once, repeats)
    total, sel = result
    rows.append(base_row(
        figure=figure or ("fig1" if not selected else "fig3"),
        case=structure["case"],
        case_label=structure["label"],
        tool="FreeSASA",
        backend="cpu",
        algorithm=f"{algorithm}{resolution}",
        resolution=resolution,
        threads=threads,
        context="full structure" if not selected else "full structure context, reported residue",
        selection_expression=selection,
        selection_semantics="FastSASA/FreeSASA-style selected atom SASA" if selected else "full context",
        frames=1,
        atoms=count_atoms(structure["path"]),
        pipeline_seconds=wall,
        pipeline_frames_per_second=1.0 / wall if wall > 0 else "",
        total_sasa=total,
        selection_sasa=sel,
        command=" ".join(command),
    ))


def measure_rust_static(rows, rust_sasa: Path, structure: dict, *, threads: int, selected: bool, repeats: int, figure: str | None = None) -> None:
    residue = structure["selected_rust_residue"]
    depth = "residue" if selected else "protein"
    selection = f"chain A and residue {residue}" if selected else ""

    def once():
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            out_path = Path(handle.name)
        try:
            command = [
                str(rust_sasa), str(structure["path"]), str(out_path),
                "--output-depth", depth,
                "--format", "json",
                "--n-points", "100",
                "--probe-radius", "1.4",
                "--threads", str(threads),
            ]
            proc = run_command(command)
            if proc.returncode:
                raise RuntimeError(proc.stderr + proc.stdout)
            total = "" if selected else parse_rustsasa_protein_json(out_path)
            sel = parse_rustsasa_residue_json(out_path, chain_id="A", residue_number=residue) if selected else ""
            return total, sel, " ".join(command)
        finally:
            out_path.unlink(missing_ok=True)

    wall, result = timed_repeats(once, repeats)
    total, sel, command_text = result
    rows.append(base_row(
        figure=figure or ("fig1" if not selected else "fig3"),
        case=structure["case"],
        case_label=structure["label"],
        tool="RustSASA",
        backend="cpu",
        algorithm="SR100",
        resolution=100,
        threads=threads,
        context="full structure" if not selected else "RustSASA CLI residue output",
        selection_expression=selection,
        selection_semantics="RustSASA CLI per-residue SASA" if selected else "full context",
        frames=1,
        atoms=count_atoms(structure["path"]),
        pipeline_seconds=wall,
        pipeline_frames_per_second=1.0 / wall if wall > 0 else "",
        total_sasa=total,
        selection_sasa=sel,
        command=command_text,
    ))


def measure_vmd_static(rows, structure: dict, *, selected: bool, repeats: int) -> None:
    target = structure["selected_vmd"] if selected else ""
    if selected:
        body = f"""
mol new {{{structure['path']}}} type pdb waitfor all
set global [atomselect top "protein"]
set target [atomselect top "{target}"]
set start [clock clicks -microseconds]
set value [measure sasa 1.4 $global -restrict $target -samples 100]
set stop [clock clicks -microseconds]
puts "FASTSASA_RESULT $value [expr {{($stop - $start)/1000000.0}}]"
quit
"""
    else:
        body = f"""
mol new {{{structure['path']}}} type pdb waitfor all
set global [atomselect top "protein"]
set start [clock clicks -microseconds]
set value [measure sasa 1.4 $global -samples 100]
set stop [clock clicks -microseconds]
puts "FASTSASA_RESULT $value [expr {{($stop - $start)/1000000.0}}]"
quit
"""

    vmd = shutil.which("vmd") or "/usr/local/bin/vmd"

    def once():
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as handle:
            handle.write(body)
            script = Path(handle.name)
        try:
            proc = run_command([vmd, "-dispdev", "text", "-e", str(script)], timeout=900)
        finally:
            script.unlink(missing_ok=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        match = re.search(r"FASTSASA_RESULT\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", proc.stdout + proc.stderr)
        if not match:
            raise RuntimeError("could not parse VMD output")
        return float(match.group(1)), float(match.group(2))

    wall, result = timed_repeats(once, repeats)
    value, compute = result
    rows.append(base_row(
        figure="fig1" if not selected else "fig3",
        case=structure["case"],
        case_label=structure["label"],
        tool="VMD",
        backend="cpu",
        algorithm="SR100",
        resolution=100,
        threads=1,
        context="full protein" if not selected else "VMD protein context, restricted target",
        selection_expression=target,
        selection_semantics="VMD measure sasa -restrict" if selected else "VMD full protein",
        frames=1,
        atoms=count_atoms(structure["path"]),
        compute_seconds=compute,
        pipeline_seconds=wall,
        compute_frames_per_second=1.0 / compute if compute > 0 else "",
        pipeline_frames_per_second=1.0 / wall if wall > 0 else "",
        total_sasa="" if selected else value,
        selection_sasa=value if selected else "",
        command="VMD measure sasa Tcl script",
    ))


def measure_fastsasa_trajectory(rows, exe: Path, tool: str, *, algorithm: str, resolution: int, threads: int | None, context: str, repeats: int, figure: str | None = None, backend: str | None = None) -> None:
    command = [
        str(exe), "trajectory",
        "--topology", str(GABAA_TOP),
        "--trajectory", str(GABAA_TRAJ),
        "--frames", ":",
        "--resolution", str(resolution),
        "--filter", "protein",
        "--summary",
    ]
    if backend is None:
        # Legacy inference for old two-way callers; explicit backend= is preferred.
        backend = "cuda" if tool.endswith("GPU") else "cpu"
    if backend == "cpu":
        command.append("--cpu")
    else:
        command += ["--backend", backend]
    if threads is not None:
        command += ["--threads", str(threads)]
    if algorithm == "LR":
        command.append("--lee-richards")
    selection = ""
    semantics = "protein full context"
    if context == "selected_protein_context":
        selection = GABAA_SELECTION_FASTSASA
        command += ["--select", f"target, {selection}"]
        semantics = "FastSASA/FreeSASA-style selected atom SASA in protein context"
    elif context == "selected_isolated":
        selection = GABAA_SELECTION_FASTSASA
        command += ["--filter", selection]
        semantics = "isolated selected residue calculation"

    def once():
        proc = run_command(command, timeout=900)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        row = parse_traj_summary(proc.stdout)
        return row

    wall, result = timed_repeats(once, repeats)
    frames = int(result.get("frames", "0") or 0)
    compute = float(result.get("gpu_seconds", "nan"))
    rows.append(base_row(
        figure=figure or ("fig4" if context == "full" else "fig5"),
        case="GABAA",
        case_label="GABAA PSF/DCD trajectory",
        tool=tool,
        backend=backend,
        algorithm=f"{algorithm}{resolution}",
        resolution=resolution,
        threads=threads if threads is not None else "auto",
        context=context.replace("_", " "),
        selection_expression=selection,
        selection_semantics=semantics,
        frames=frames,
        atoms=result.get("atoms", ""),
        compute_seconds=compute,
        pipeline_seconds=result.get("wall_seconds", wall),
        compute_frames_per_second=result.get("gpu_frames_per_second", ""),
        pipeline_frames_per_second=result.get("wall_frames_per_second", frames / wall if wall > 0 else ""),
        total_sasa=result.get("total_sasa_sum", ""),
        selection_sasa=result.get("selection_sasa_sum", ""),
        command=" ".join(command),
    ))


def measure_fastsasa_precision(rows, exe: Path, *, repeats: int, tool: str = "FastSASA CUDA", backend: str = "cuda") -> None:
    cases = [
        {
            "case": "2ISK",
            "case_label": "2ISK structure, full context",
            "context": "one-off full structure",
            "algorithm": "SR",
            "resolution": 100,
            "value_key": "total_sasa",
            "commands": {
                "fp32": [str(exe), "--backend", backend, "--precision", "fp32", "--no-cpu-fallback", "--shrake-rupley", "--resolution", "100", "--format", "log", str(ROOT / "tests/data/2isk.pdb")],
                "fp64": [str(exe), "--backend", backend, "--precision", "fp64", "--no-cpu-fallback", "--shrake-rupley", "--resolution", "100", "--format", "log", str(ROOT / "tests/data/2isk.pdb")],
            },
            "env": {
                "fp32": {},
                "fp64": {},
            },
            "parse": lambda proc: (parse_log_total(proc.stdout), ""),
            "atoms": count_atoms(ROOT / "tests/data/2isk.pdb"),
            "frames": 1,
        },
        {
            "case": "2ISK",
            "case_label": "2ISK structure, full context",
            "context": "one-off full structure",
            "algorithm": "LR",
            "resolution": 20,
            "value_key": "total_sasa",
            "commands": {
                "fp32": [str(exe), "--backend", backend, "--precision", "fp32", "--no-cpu-fallback", "--lee-richards", "--resolution", "20", "--format", "log", str(ROOT / "tests/data/2isk.pdb")],
                "fp64": [str(exe), "--backend", backend, "--precision", "fp64", "--no-cpu-fallback", "--lee-richards", "--resolution", "20", "--format", "log", str(ROOT / "tests/data/2isk.pdb")],
            },
            "env": {
                "fp32": {},
                "fp64": {},
            },
            "parse": lambda proc: (parse_log_total(proc.stdout), ""),
            "atoms": count_atoms(ROOT / "tests/data/2isk.pdb"),
            "frames": 1,
        },
    ]
    if GABAA_TOP.exists() and GABAA_TRAJ.exists():
        cases.extend([
            {
                "case": "GABAA",
                "case_label": "GABAA trajectory, full protein",
                "context": "trajectory full protein",
                "algorithm": "SR",
                "resolution": 100,
                "value_key": "total_sasa",
                "commands": {
                    "fp32": [
                        str(exe), "trajectory", "--backend", backend, "--precision", "fp32", "--topology", str(GABAA_TOP), "--trajectory", str(GABAA_TRAJ),
                        "--frames", ":", "--resolution", "100", "--filter", "protein", "--summary",
                    ],
                    "fp64": [
                        str(exe), "trajectory", "--backend", backend, "--precision", "fp64", "--topology", str(GABAA_TOP), "--trajectory", str(GABAA_TRAJ),
                        "--frames", ":", "--resolution", "100", "--filter", "protein", "--summary",
                    ],
                },
                "env": {
                    "fp32": {},
                    "fp64": {},
                },
                "parse": lambda proc: (float(parse_traj_summary(proc.stdout).get("total_sasa_sum", "nan")), parse_traj_summary(proc.stdout)),
                "atoms": "",
                "frames": "",
            },
            {
                "case": "GABAA",
                "case_label": "GABAA trajectory, selected residue",
                "context": "trajectory selected residue in protein context",
                "algorithm": "SR",
                "resolution": 100,
                "value_key": "selection_sasa",
                "commands": {
                    "fp32": [
                        str(exe), "trajectory", "--backend", backend, "--precision", "fp32", "--topology", str(GABAA_TOP), "--trajectory", str(GABAA_TRAJ),
                        "--frames", ":", "--resolution", "100", "--filter", "protein",
                        "--select", f"target, {GABAA_SELECTION_FASTSASA}", "--summary",
                    ],
                    "fp64": [
                        str(exe), "trajectory", "--backend", backend, "--precision", "fp64", "--topology", str(GABAA_TOP), "--trajectory", str(GABAA_TRAJ),
                        "--frames", ":", "--resolution", "100", "--filter", "protein",
                        "--select", f"target, {GABAA_SELECTION_FASTSASA}", "--summary",
                    ],
                },
                "env": {
                    "fp32": {},
                    "fp64": {},
                },
                "parse": lambda proc: (float(parse_traj_summary(proc.stdout).get("selection_sasa_sum", "nan")), parse_traj_summary(proc.stdout)),
                "atoms": "",
                "frames": "",
            },
        ])

    for case in cases:
        measured: dict[str, dict[str, object]] = {}
        for precision_key, label in (("fp32", "FP32"), ("fp64", "FP64")):
            command = case["commands"][precision_key]
            env = case["env"][precision_key]

            def once():
                proc = run_command(command, env=env, timeout=900)
                if proc.returncode:
                    raise RuntimeError(proc.stderr + proc.stdout)
                value, extra = case["parse"](proc)
                return value, extra

            wall, result = timed_repeats(once, repeats)
            value, extra = result
            frames = case["frames"]
            atoms = case["atoms"]
            compute = ""
            fps = ""
            pipeline_fps = ""
            if isinstance(extra, dict):
                frames = extra.get("frames", frames)
                atoms = extra.get("atoms", atoms)
                compute = extra.get("gpu_seconds", "")
                fps = extra.get("gpu_frames_per_second", "")
                pipeline_fps = extra.get("wall_frames_per_second", "")
            elif case["frames"] == 1:
                pipeline_fps = 1.0 / wall if wall > 0 else ""
            measured[precision_key] = {
                "wall": wall,
                "value": float(value),
                "frames": frames,
                "atoms": atoms,
                "compute": compute,
                "fps": fps,
                "pipeline_fps": pipeline_fps,
                "label": label,
                "command": " ".join([f"{key}={val}" for key, val in env.items()] + command),
            }

        fp64_value = measured["fp64"]["value"]
        fp64_wall = measured["fp64"]["wall"]
        for precision_key in ("fp32", "fp64"):
            result = measured[precision_key]
            value = result["value"]
            abs_error = abs(value - fp64_value)
            rel_error = abs_error / abs(fp64_value) if fp64_value else 0.0
            speedup = fp64_wall / result["wall"] if result["wall"] else ""
            rows.append(base_row(
                figure="fig6",
                case=case["case"],
                case_label=case["case_label"],
                tool=tool,
                backend=backend,
                algorithm=f"{case['algorithm']}{case['resolution']}",
                resolution=case["resolution"],
                threads="auto",
                context=case["context"],
                selection_expression=GABAA_SELECTION_FASTSASA if "selected" in case["context"] else "",
                selection_semantics="FastSASA FP32-vs-FP64 precision comparison",
                frames=result["frames"],
                atoms=result["atoms"],
                compute_seconds=result["compute"],
                pipeline_seconds=result["wall"],
                compute_frames_per_second=result["fps"],
                pipeline_frames_per_second=result["pipeline_fps"],
                total_sasa=value if case["value_key"] == "total_sasa" else "",
                selection_sasa=value if case["value_key"] == "selection_sasa" else "",
                absolute_error_vs_fp64=abs_error,
                relative_error_vs_fp64=rel_error,
                speedup_vs_fp64=speedup,
                precision=result["label"],
                command=result["command"],
            ))


def measure_freesasa_trajectory(rows, freesasa: Path, *, algorithm: str, resolution: int, threads: int, selected: bool, repeats: int, figure: str | None = None) -> None:
    env = python_benchmark_env()
    alg_flag = "--shrake-rupley" if algorithm == "SR" else "--lee-richards"
    selection = "chain A and resi 155" if selected else ""

    def once():
        tmp = Path(tempfile.mkdtemp(prefix="fastsasa_pub_fs_frames_"))
        try:
            export_code = f"""
from pathlib import Path
import MDAnalysis as mda
top={str(GABAA_TOP)!r}
traj={str(GABAA_TRAJ)!r}
out=Path({str(tmp)!r})
u=mda.Universe(top, traj)
try:
    u.add_TopologyAttr('chainIDs')
except Exception:
    pass
chain_ids = []
for atom in u.atoms:
    seg = atom.segid
    chain_ids.append(seg[-1] if seg.startswith('PRO') and len(seg) > 3 else 'X')
u.atoms.chainIDs = chain_ids
sel=u.select_atoms('protein')
for i, ts in enumerate(u.trajectory):
    with mda.coordinates.PDB.PDBWriter(str(out / f'frame_{{i:04d}}.pdb'), multiframe=False, bonds=None) as W:
        W.write(sel)
print(len(u.trajectory), len(sel))
"""
            export_start = time.perf_counter()
            proc = run_command([PYTHON_EXECUTABLE, "-c", export_code], env=env, timeout=900)
            export_seconds = time.perf_counter() - export_start
            if proc.returncode:
                raise RuntimeError(proc.stderr + proc.stdout)
            paths = sorted(tmp.glob("frame_*.pdb"))
            total = 0.0
            compute_start = time.perf_counter()
            for path in paths:
                cmd = [str(freesasa), alg_flag, "--resolution", str(resolution), f"--n-threads={threads}", "--format=log", "--no-warnings", str(path)]
                if selected:
                    cmd.insert(-1, f"--select=target, {selection}")
                frame_proc = run_command(cmd, timeout=900)
                if frame_proc.returncode:
                    raise RuntimeError(frame_proc.stderr + frame_proc.stdout)
                total += parse_log_selection(frame_proc.stdout) if selected else parse_log_total(frame_proc.stdout)
            compute_seconds = time.perf_counter() - compute_start
            return len(paths), count_atoms(paths[0]), total, compute_seconds, export_seconds + compute_seconds
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    wall, result = timed_repeats(once, repeats)
    frames, atoms, total, compute, pipeline = result
    rows.append(base_row(
        figure=figure or ("fig5" if selected else "fig4"),
        case="GABAA",
        case_label="GABAA PSF/DCD trajectory",
        tool="FreeSASA",
        backend="cpu",
        algorithm=f"{algorithm}{resolution}",
        resolution=resolution,
        threads=threads,
        context="selected residue in protein context via exported PDB frames" if selected else "full protein via exported PDB frames",
        selection_expression=selection,
        selection_semantics="FreeSASA selected atom SASA in protein context" if selected else "full context",
        frames=frames,
        atoms=atoms,
        compute_seconds=compute,
        pipeline_seconds=pipeline,
        compute_frames_per_second=frames / compute if compute > 0 else "",
        pipeline_frames_per_second=frames / pipeline if pipeline > 0 else "",
        total_sasa="" if selected else total,
        selection_sasa=total if selected else "",
        notes="pipeline includes MDAnalysis PSF/DCD to PDB export plus FreeSASA",
        command="MDAnalysis frame export + FreeSASA per frame",
    ))


def measure_vmd_trajectory(rows, *, selected: bool, repeats: int) -> None:
    target = GABAA_SELECTION_VMD if selected else ""
    if selected:
        body = f"""
mol new {{{GABAA_TOP}}} type psf waitfor all
mol addfile {{{GABAA_TRAJ}}} type dcd waitfor all
set global [atomselect top "protein"]
set target [atomselect top "{target}"]
set n [molinfo top get numframes]
set total 0.0
set start [clock clicks -microseconds]
for {{set i 0}} {{$i < $n}} {{incr i}} {{
  $global frame $i
  $target frame $i
  set value [measure sasa 1.4 $global -restrict $target -samples 100]
  set total [expr {{$total + $value}}]
}}
set stop [clock clicks -microseconds]
puts "FASTSASA_RESULT $n $total [expr {{($stop - $start)/1000000.0}}]"
quit
"""
    else:
        body = f"""
mol new {{{GABAA_TOP}}} type psf waitfor all
mol addfile {{{GABAA_TRAJ}}} type dcd waitfor all
set global [atomselect top "protein"]
set n [molinfo top get numframes]
set total 0.0
set start [clock clicks -microseconds]
for {{set i 0}} {{$i < $n}} {{incr i}} {{
  $global frame $i
  set value [measure sasa 1.4 $global -samples 100]
  set total [expr {{$total + $value}}]
}}
set stop [clock clicks -microseconds]
puts "FASTSASA_RESULT $n $total [expr {{($stop - $start)/1000000.0}}]"
quit
"""

    vmd = shutil.which("vmd") or "/usr/local/bin/vmd"

    def once():
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as handle:
            handle.write(body)
            script = Path(handle.name)
        try:
            proc = run_command([vmd, "-dispdev", "text", "-e", str(script)], timeout=1200)
        finally:
            script.unlink(missing_ok=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        match = re.search(r"FASTSASA_RESULT\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", proc.stdout + proc.stderr)
        if not match:
            raise RuntimeError("could not parse VMD trajectory output")
        return int(match.group(1)), float(match.group(2)), float(match.group(3))

    pipeline, result = timed_repeats(once, repeats)
    frames, value, compute = result
    rows.append(base_row(
        figure="fig5" if selected else "fig4",
        case="GABAA",
        case_label="GABAA PSF/DCD trajectory",
        tool="VMD",
        backend="cpu",
        algorithm="SR100",
        resolution=100,
        threads=1,
        context="VMD protein context, restricted target" if selected else "full protein",
        selection_expression=target,
        selection_semantics="VMD measure sasa -restrict" if selected else "VMD full protein",
        frames=frames,
        compute_seconds=compute,
        pipeline_seconds=pipeline,
        compute_frames_per_second=frames / compute if compute > 0 else "",
        pipeline_frames_per_second=frames / pipeline if pipeline > 0 else "",
        total_sasa="" if selected else value,
        selection_sasa=value if selected else "",
        command="VMD Tcl measure sasa",
    ))


def measure_bolt_trajectory(rows, *, selected: bool, threads: int, repeats: int) -> None:
    selection = "protein"
    code = f"""
import time
import numpy as np
import MDAnalysis as mda
from mdsasa_bolt import SASAAnalysis
top={str(GABAA_TOP)!r}
traj={str(GABAA_TRAJ)!r}
selection={selection!r}
target_selection={GABAA_SELECTION_MDANALYSIS!r}
start=time.perf_counter()
u=mda.Universe(top, traj)
protein = u.select_atoms(selection)
a=SASAAnalysis(protein, select='all', n_points=100, probe_radius=1.4)
compute_start=time.perf_counter()
a.run(start=0, stop=len(u.trajectory), step=1)
compute=time.perf_counter()-compute_start
pipeline=time.perf_counter()-start
arr=np.asarray(a.results.total_area, dtype=float)
value=float(np.nansum(arr))
if {selected!r}:
    target = protein.select_atoms(target_selection)
    target_indices = []
    for target_residue in target.residues:
        for idx, residue in enumerate(protein.residues):
            if residue.segid == target_residue.segid and residue.resid == target_residue.resid:
                target_indices.append(idx)
                break
    residue_area = np.asarray(getattr(a.results, 'residue_area'), dtype=float)
    value = float(np.nansum(residue_area[:, target_indices])) if target_indices else float('nan')
print(len(arr), len(a.atomgroup), value, compute, pipeline)
"""
    command = [PYTHON_EXECUTABLE, "-c", code]
    env = python_benchmark_env(threads=threads)

    def once():
        proc = run_command(command, env=env, timeout=1200)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        parts = proc.stdout.strip().splitlines()[-1].split()
        return int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    pipeline_wall, result = timed_repeats(once, repeats)
    frames, atoms, total, compute, pipeline = result
    rows.append(base_row(
        figure="fig5" if selected else "fig4",
        case="GABAA",
        case_label="GABAA PSF/DCD trajectory",
        tool="mdsasa-bolt/RustSASA",
        backend="cpu",
        algorithm="SR100",
        resolution=100,
        threads=threads,
        context="selected residue in protein context" if selected else "full protein AtomGroup",
        selection_expression=GABAA_SELECTION_MDANALYSIS if selected else selection,
        selection_semantics="mdsasa-bolt per-residue SASA in protein context" if selected else "full protein",
        frames=frames,
        atoms=atoms,
        compute_seconds=compute,
        pipeline_seconds=pipeline,
        compute_frames_per_second=frames / compute if compute > 0 else "",
        pipeline_frames_per_second=frames / pipeline if pipeline > 0 else frames / pipeline_wall if pipeline_wall > 0 else "",
        total_sasa="" if selected else total,
        selection_sasa=total if selected else "",
        command="MDAnalysis + mdsasa-bolt SASAAnalysis",
    ))


def measure_rust_exported_trajectory(rows, rust_sasa: Path, *, threads: int, selected: bool, repeats: int) -> None:
    selection = "chain A and residue 155" if selected else ""
    code = f"""
import time
from pathlib import Path
import tempfile
import shutil
import json
import subprocess
import MDAnalysis as mda

top={str(GABAA_TOP)!r}
traj={str(GABAA_TRAJ)!r}
threads={threads}
selected={selected!r}
rust_sasa={str(rust_sasa)!r}
tmp=Path(tempfile.mkdtemp(prefix='fastsasa_pub_rust_frames_'))
start=time.perf_counter()
try:
    u=mda.Universe(top, traj)
    try:
        u.add_TopologyAttr('chainIDs')
    except Exception:
        pass
    chain_ids = []
    for atom in u.atoms:
        seg = atom.segid
        chain_ids.append(seg[-1] if seg.startswith('PRO') and len(seg) > 3 else 'X')
    u.atoms.chainIDs = chain_ids
    protein=u.select_atoms('protein')
    paths=[]
    for i, ts in enumerate(u.trajectory):
        path=tmp / f'frame_{{i:04d}}.pdb'
        with mda.coordinates.PDB.PDBWriter(str(path), multiframe=False, bonds=None) as writer:
            writer.write(protein)
        paths.append(path)

    compute_start=time.perf_counter()
    total=0.0
    for path in paths:
        out = tmp / (path.stem + '.json')
        depth = 'residue' if selected else 'protein'
        cmd = [rust_sasa, str(path), str(out), '--output-depth', depth, '--format', 'json', '--n-points', '100', '--probe-radius', '1.4', '--threads', str(threads), '--allow-vdw-fallback']
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        data = json.loads(out.read_text())
        if selected:
            for residue in data.get('Residue', []):
                if residue.get('chain_id') == 'A' and int(residue.get('serial_number')) == 155:
                    total += float(residue.get('value'))
                    break
        else:
            total += float(data['Protein']['global_total'])
    compute=time.perf_counter()-compute_start
    pipeline=time.perf_counter()-start
    print(len(paths), len(protein), total, compute, pipeline)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
"""
    command = [PYTHON_EXECUTABLE, "-c", code]
    env = python_benchmark_env(threads=threads)

    def once():
        proc = run_command(command, env=env, timeout=1200)
        if proc.returncode:
            raise RuntimeError(proc.stderr + proc.stdout)
        parts = proc.stdout.strip().splitlines()[-1].split()
        return int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    pipeline_wall, result = timed_repeats(once, repeats)
    frames, atoms, value, compute, pipeline = result
    rows.append(base_row(
        figure="fig5" if selected else "fig4",
        case="GABAA",
        case_label="GABAA PSF/DCD trajectory",
        tool="RustSASA",
        backend="cpu",
        algorithm="SR100",
        resolution=100,
        threads=threads,
        context="selected residue in protein context via exported PDB frames" if selected else "full protein via exported PDB frames",
        selection_expression=selection,
        selection_semantics="RustSASA CLI per-residue SASA in protein context" if selected else "RustSASA CLI full protein",
        frames=frames,
        atoms=atoms,
        compute_seconds=compute,
        pipeline_seconds=pipeline,
        compute_frames_per_second=frames / compute if compute > 0 else "",
        pipeline_frames_per_second=frames / pipeline if pipeline > 0 else frames / pipeline_wall if pipeline_wall > 0 else "",
        total_sasa="" if selected else value,
        selection_sasa=value if selected else "",
        notes="pipeline includes MDAnalysis PSF/DCD to PDB export plus RustSASA CLI",
        command="MDAnalysis frame export + rust-sasa CLI per frame",
    ))


def write_rows(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["case_label"] = row.get("case_label", "").replace("\n", " ")
    return rows


def numeric(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def tool_rank(tool: str) -> int:
    return TOOL_ORDER.index(tool) if tool in TOOL_ORDER else len(TOOL_ORDER)


CASE_LABEL_ORDER = {
    "1UBQ small protein (660 atoms)": 0,
    "3BKR medium protein (1,160 atoms)": 1,
    "2ISK large protein (15,127 atoms)": 2,
}


def algorithm_rank(algorithm: str) -> int:
    if algorithm.startswith("SR"):
        return 0
    if algorithm.startswith("LR"):
        return 1
    return 2


def short_label(row: dict[str, str], *, include_algorithm: bool = False) -> str:
    base = SHORT_TOOL_LABELS.get(row["tool"], row["tool"])
    thread = row.get("threads", "")
    if row["tool"] in ("FastSASA CUDA", "FastSASA Vulkan"):
        label = base
    elif thread not in ("", "auto", "default"):
        label = f"{thread}c {base}"
    else:
        label = base
    if include_algorithm:
        label += f" {row['algorithm']}"
    if row.get("precision") in ("FP32", "FP64"):
        label += f" {row['precision']}"
    return label


def save_all(fig, output: Path) -> None:
    for suffix in ("png", "pdf", "svg", "eps"):
        fig.savefig(output.with_suffix("." + suffix), dpi=220)


def add_broken_axis_marks(upper, lower) -> None:
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(labelbottom=False, bottom=False)
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle="none",
                  color="#222222", mec="#222222", mew=0.8, clip_on=False)
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **kwargs)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **kwargs)


def plot_single_bar_panel(ax, subset: list[dict[str, str]], *, time_key: str, ylabel: str,
                          show_x: bool = True, include_algorithm: bool = False) -> tuple[list[float], list[str]]:
    labels = []
    values = []
    colors = []
    hatches = []
    for row in subset:
        thread = row["threads"]
        labels.append(short_label(row, include_algorithm=include_algorithm))
        values.append(numeric(row, time_key) or 0.0)
        colors.append(TOOL_COLORS.get(row["tool"], "#999999"))
        hatches.append(HATCHES.get(thread, ""))
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.64, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if show_x:
        ax.set_xticklabels(labels, fontsize=7, rotation=90, ha="center")
    else:
        ax.set_xticklabels([])
    ax.grid(axis="y", color="#e4e4e4")
    ax.set_axisbelow(True)
    return values, labels


def plot_oneoff_combined_ms(rows: list[dict[str, str]], out_prefix: Path) -> None:
    panels = [
        ("fig1", "Full context"),
        ("fig3", "Selected residue"),
    ]
    filtered = [
        row for row in rows
        if row["status"] == "pass"
        and row["figure"] in {panel[0] for panel in panels}
        and numeric(row, "pipeline_seconds") is not None
    ]
    if not filtered:
        return
    groups = sorted(
        {row["case_label"] for row in filtered},
        key=lambda label: CASE_LABEL_ORDER.get(label, len(CASE_LABEL_ORDER)),
    )
    fig = plt.figure(figsize=(9.8, max(4.4, 1.8 * len(groups) + 0.9)))
    grid = fig.add_gridspec(len(groups) * 2, 2, height_ratios=[0.55, 1.35] * len(groups),
                            hspace=0.16, wspace=0.18)
    break_ms = 10.0

    for col, (figure, panel_title) in enumerate(panels):
        for group_index, group in enumerate(groups):
            subset = [
                row for row in filtered
                if row["figure"] == figure and row["case_label"] == group
            ]
            if not subset:
                continue
            subset.sort(key=lambda row: (tool_rank(row["tool"]), algorithm_rank(row["algorithm"]), str(row["threads"]), row["context"]))
            upper = fig.add_subplot(grid[group_index * 2, col])
            lower = fig.add_subplot(grid[group_index * 2 + 1, col])
            labels, values, colors, hatches = [], [], [], []
            for row in subset:
                labels.append(short_label(row))
                values.append((numeric(row, "pipeline_seconds") or 0.0) * 1000.0)
                colors.append(TOOL_COLORS.get(row["tool"], "#999999"))
                hatches.append(HATCHES.get(row["threads"], ""))
            x = np.arange(len(values))
            for ax in (upper, lower):
                bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="#222222", linewidth=0.6)
                for bar, hatch in zip(bars, hatches):
                    bar.set_hatch(hatch)
                ax.grid(axis="y", color="#e4e4e4")
                ax.set_axisbelow(True)
                ax.set_xlim(-0.7, len(values) - 0.3)
            lower.set_ylim(0, break_ms)
            high_values = [v for v in values if v > break_ms]
            if high_values:
                upper.set_ylim(break_ms, max(high_values) * 1.12)
                add_broken_axis_marks(upper, lower)
            else:
                upper.set_visible(False)
            show_labels = group_index == len(groups) - 1
            lower.set_xticks(x)
            lower.set_xticklabels(labels if show_labels else [], fontsize=7, rotation=90, ha="center")
            upper.set_xticks(x)
            upper.set_xticklabels([])
            if col == 0:
                lower.set_ylabel("ms")
                upper.text(0.0, 0.96, group, transform=upper.transAxes, ha="left", va="top",
                           fontsize=8.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
            if group_index == 0:
                upper.set_title(panel_title, fontsize=10, pad=2)

    fig.suptitle("One-off structure benchmark", fontsize=13, y=0.988)
    fig.text(0.01, 0.006, "Lower is better. Y-axis break at 10 ms.", fontsize=8)
    fig.subplots_adjust(top=0.93, bottom=0.22, left=0.075, right=0.985, hspace=0.18, wspace=0.18)
    save_all(fig, out_prefix)
    plt.close(fig)


def plot_grouped_time(rows: list[dict[str, str]], *, output: Path, figure: str, title: str, time_key: str, ylabel: str) -> None:
    filtered = [
        row for row in rows
        if row["status"] == "pass" and row["figure"] == figure and numeric(row, time_key) is not None
    ]
    if not filtered:
        return
    groups = sorted(
        {row["case_label"] for row in filtered},
        key=lambda label: CASE_LABEL_ORDER.get(label, len(CASE_LABEL_ORDER)),
    )
    use_broken_axis = figure in ("fig1", "fig3")
    if use_broken_axis:
        height_ratios = []
        for _ in groups:
            height_ratios.extend([0.58, 1.35])
        fig = plt.figure(figsize=(6.8, max(3.4, 2.25 * len(groups))))
        grid = fig.add_gridspec(len(groups) * 2, 1, height_ratios=height_ratios, hspace=0.16)
    else:
        fig, axes = plt.subplots(len(groups), 1, figsize=(6.7, max(2.6, 3.55 * len(groups))), squeeze=False)

    for group_index, group in enumerate(groups):
        subset = [row for row in filtered if row["case_label"] == group]
        subset.sort(key=lambda row: (tool_rank(row["tool"]), algorithm_rank(row["algorithm"]), str(row["threads"]), row["context"]))
        include_algorithm = figure in ("fig6", "fig7")
        if use_broken_axis:
            upper = fig.add_subplot(grid[group_index * 2, 0])
            lower = fig.add_subplot(grid[group_index * 2 + 1, 0])
            show_bottom_labels = group_index == len(groups) - 1
            values, _ = plot_single_bar_panel(lower, subset, time_key=time_key, ylabel=ylabel,
                                              show_x=show_bottom_labels, include_algorithm=include_algorithm)
            plot_single_bar_panel(upper, subset, time_key=time_key, ylabel="",
                                  show_x=False, include_algorithm=include_algorithm)
            upper.set_xlim(lower.get_xlim())
            positive = sorted(v for v in values if v > 0)
            mid = statistics.median(positive) if positive else 0.0
            lower_top = mid * 2.4 if mid else max(values) * 0.35
            upper_values = [v for v in positive if v > lower_top]
            if upper_values:
                lower.set_ylim(0, lower_top)
                upper.set_ylim(min(upper_values) * 0.82, max(upper_values) * 1.12)
                add_broken_axis_marks(upper, lower)
            else:
                upper.set_visible(False)
                lower.set_ylim(0, max(values) * 1.15 if values else 1)
            upper.text(0.0, 0.96, group, transform=upper.transAxes, ha="left", va="top",
                       fontsize=9, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
            lower.tick_params(axis="x", labelsize=7, labelbottom=show_bottom_labels)
        else:
            ax = axes[group_index, 0]
            plot_single_bar_panel(ax, subset, time_key=time_key, ylabel=ylabel,
                                  show_x=True, include_algorithm=include_algorithm)
            ax.set_title(group, loc="left", fontsize=9, pad=2)
            ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(title, fontsize=13, y=0.982)
    fig.text(0.01, 0.006, "Lower is better. Blank methods were not applicable or not available.", fontsize=8)
    if use_broken_axis:
        top_margin = 0.88 if len(groups) < 3 else 0.92
        fig.subplots_adjust(top=top_margin, bottom=0.078, left=0.13, right=0.985, hspace=0.18)
    else:
        fig.tight_layout(rect=(0, 0.018, 1, 0.972))
    save_all(fig, output)
    plt.close(fig)


def plot_cpu_scaling(rows: list[dict[str, str]], out_prefix: Path) -> None:
    filtered = [
        row for row in rows
        if row["status"] == "pass" and row["figure"] == "fig2" and row["tool"] == "FastSASA CPU" and row["algorithm"] == "SR100"
    ]
    cases = sorted({row["case_label"] for row in filtered})
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    thread_values = [1, 2, 4, 8, 15]
    for case in cases:
        by_thread = {
            int(row["threads"]): numeric(row, "pipeline_seconds")
            for row in filtered
            if row["case_label"] == case and row["threads"].isdigit()
        }
        baseline = by_thread.get(1)
        if not baseline:
            continue
        speedups = [baseline / by_thread[t] if by_thread.get(t) else math.nan for t in thread_values]
        ax.plot(thread_values, speedups, marker="o", linewidth=1.8, label=case)
    ax.plot(thread_values, thread_values, color="#222222", linestyle="--", linewidth=1.0, label="ideal linear")
    ax.set_xticks(thread_values)
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Speedup vs 1 thread (higher is better)")
    ax.set_title("Figure 2. FastSASA CPU parallel scaling")
    ax.grid(axis="y", color="#e4e4e4")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save_all(fig, out_prefix)
    plt.close(fig)


def plot_precision_summary(rows: list[dict[str, str]], out_prefix: Path) -> None:
    filtered = [
        row for row in rows
        if row["status"] == "pass" and row["figure"] == "fig6"
    ]
    if not filtered:
        return
    order = [
        ("2ISK structure, full context", "SR100", "2ISK-SR"),
        ("2ISK structure, full context", "LR20", "2ISK-LR"),
        ("GABAA trajectory, full protein", "SR100", "GABAA full-SR"),
        ("GABAA trajectory, selected residue", "SR100", "GABAA selected-SR"),
    ]
    labels = [label for _, _, label in order]
    x = np.arange(len(labels))
    width = 0.34

    by_key = {(row["case_label"], row["algorithm"], row["precision"]): row for row in filtered}
    fp32_times = [numeric(by_key.get((case, alg, "FP32"), {}), "pipeline_seconds") or math.nan for case, alg, _ in order]
    fp64_times = [numeric(by_key.get((case, alg, "FP64"), {}), "pipeline_seconds") or math.nan for case, alg, _ in order]
    fp32_errors = [numeric(by_key.get((case, alg, "FP32"), {}), "relative_error_vs_fp64") or 0.0 for case, alg, _ in order]
    fp32_speedups = [numeric(by_key.get((case, alg, "FP32"), {}), "speedup_vs_fp64") or math.nan for case, alg, _ in order]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.6), sharex=True)
    axes[0].bar(x - width / 2, fp32_times, width, label="FP32", color="#D55E00", edgecolor="#222222", linewidth=0.6)
    axes[0].bar(x + width / 2, fp64_times, width, label="FP64", color="#0072B2", edgecolor="#222222", linewidth=0.6)
    axes[0].set_ylabel("Seconds")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", color="#e4e4e4")

    if fp32_errors and max(fp32_errors) > 0:
        safe_log_bar(axes[1], x, fp32_errors, color="#D55E00", edgecolor="#222222", linewidth=0.6)
    else:
        axes[1].bar(x, fp32_errors, color="#D55E00", edgecolor="#222222", linewidth=0.6)
    axes[1].set_ylabel("Rel. error")
    axes[1].grid(axis="y", color="#e4e4e4")

    axes[2].bar(x, fp32_speedups, color="#D55E00", edgecolor="#222222", linewidth=0.6)
    axes[2].axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Speedup")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(axis="y", color="#e4e4e4")

    fig.suptitle("Figure 6. FastSASA FP32 vs FP64 performance and agreement", fontsize=13, y=0.988)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    save_all(fig, out_prefix)
    plt.close(fig)


def trajectory_values(rows: list[dict[str, str]], figure: str) -> tuple[list[str], list[float], list[str], list[str]]:
    filtered = [
        row for row in rows
        if row["status"] == "pass" and row["figure"] == figure and row["algorithm"] == "SR100"
    ]
    filtered.sort(key=lambda row: (tool_rank(row["tool"]), str(row["threads"]), row["context"]))
    labels = []
    fps_values = []
    colors = []
    hatches = []
    for row in filtered:
        fps = numeric(row, "pipeline_frames_per_second")
        if not fps:
            frames = numeric(row, "frames")
            seconds = numeric(row, "pipeline_seconds")
            fps = frames / seconds if frames and seconds else math.nan
        labels.append(short_label(row))
        fps_values.append(fps)
        colors.append(TOOL_COLORS.get(row["tool"], "#999999"))
        hatches.append(HATCHES.get(row["threads"], ""))
    return labels, fps_values, colors, hatches


def plot_trajectory_combined(rows: list[dict[str, str]], output: Path) -> None:
    panels = [
        ("fig4", "Full context"),
        ("fig5", "Selected residue"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True)
    for ax, (figure, panel_title) in zip(axes, panels):
        labels, fps_values, colors, hatches = trajectory_values(rows, figure)
        if not labels:
            continue
        x = np.arange(len(labels))
        bars = safe_log_bar(ax, x, fps_values, width=0.62, color=colors, edgecolor="#222222", linewidth=0.6)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_ylabel("Frames/s")
        ax.set_title(panel_title, loc="left", fontsize=10, pad=2)
        ax.grid(axis="y", color="#e4e4e4")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)

    fig.suptitle("Trajectory benchmark", fontsize=13, y=0.982)
    fig.text(0.01, 0.006, "Higher is better.", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    save_all(fig, output)
    plt.close(fig)


def plot_trajectory_performance(rows: list[dict[str, str]], *, output: Path, figure: str, title: str) -> None:
    labels, fps_values, colors, hatches = trajectory_values(rows, figure)
    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bars = safe_log_bar(ax, x, fps_values, width=0.68, color=colors, edgecolor="#222222", linewidth=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_ylabel("Frames/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", color="#e4e4e4")
    fig.suptitle(title, fontsize=13, y=0.982)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save_all(fig, output)
    plt.close(fig)


def write_readme(out_dir: Path, rows: list[dict[str, str]]) -> None:
    passed = sum(1 for row in rows if row["status"] == "pass")
    failed = [row for row in rows if row["status"] != "pass"]
    text = [
        "# Focused Publication Benchmark Figures",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Rows passing: {passed}",
        f"Rows failed/skipped: {len(failed)}",
        "",
        "## Figure Questions",
        "",
        "- Figure 1: one-off full-context structure comparison across tools.",
        "- Figure 2: FastSASA CPU one-off parallel speedup from 1 to 15 threads.",
        "- Figure 3: one-off selected-residue comparison. Selection semantics are recorded in `publication_benchmark_measurements.csv`.",
        "- Figure 4: full-context trajectory comparison as FPS and speedup.",
        "- Figure 5: selected-residue trajectory comparison as FPS and speedup.",
        "- Figure 6: FastSASA-only FP32 vs FP64 speed and numerical agreement in one compact figure.",
        "- Figure 7: SR vs LR algorithm comparison for FreeSASA/FastSASA only.",
        "",
        "Figures report end-to-end time, FPS, or speedup. Raw compute-only timings are retained in the CSV only for traceability.",
        "",
        "## Important Interpretation Notes",
        "",
        "- RustSASA static timings use the `rust-sasa` CLI.",
        "- RustSASA trajectory timings export PSF/DCD frames through MDAnalysis, then run the `rust-sasa` CLI per frame.",
        "- VMD selected-residue timings use `measure sasa ... -restrict`; this is not identical to FastSASA/FreeSASA selected atom SASA.",
        "- FastSASA CUDA and FastSASA Vulkan are the same binary, selected per measurement via --backend; both use the default optimized trajectory path for their backend.",
        "- CUDA FP64 totals accumulate per-atom areas via a device-side atomic add across GPU thread blocks (order not guaranteed), so the last bit can differ from the CPU/Vulkan reference on large structures; Vulkan FP64 SR applies the area formula on the host from exact integer counts and is bit-identical to the CPU reference on every case tested.",
        "- FP64/double is the default reported precision in this figure set.",
        "- Figure 6 is the exception: it intentionally compares FastSASA FP32 and FP64.",
        "- SR100 means Shrake-Rupley with 100 points. LR20 means Lee-Richards with 20 slices.",
        "",
    ]
    if failed:
        text.append("## Failed Or Unavailable Rows")
        text.append("")
        for row in failed:
            text.append(f"- {row['tool']} {row['case']} {row['algorithm']}: {row['notes']}")
        text.append("")
    (out_dir / "README.md").write_text("\n".join(text))


def make_figures(rows: list[dict[str, str]], out_dir: Path) -> None:
    plot_oneoff_combined_ms(
        rows,
        out_dir / "fig01_oneoff_full_and_selected_ms_lower_better",
    )
    plot_cpu_scaling(rows, out_dir / "fig02_fastsasa_cpu_thread_scaling_pipeline_seconds_lower_better")
    plot_trajectory_combined(
        rows,
        out_dir / "fig04_fig05_trajectory_full_and_selected_fps_higher_better",
    )
    plot_grouped_time(
        rows,
        output=out_dir / "fig07_sr_vs_lr_seconds_lower_better",
        figure="fig7",
        title="Figure 7. Shrake-Rupley vs Lee-Richards, FreeSASA/FastSASA only",
        time_key="pipeline_seconds",
        ylabel="End-to-end seconds",
    )
    plot_precision_summary(rows, out_dir / "fig06_fastsasa_fp32_fp64_summary")


def run_measurements(args: argparse.Namespace, out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    freesasa = tool_path(args.freesasa, FREESASA_CANDIDATES, "freesasa")
    rust_sasa = tool_path(args.rust_sasa, RUSTSASA_CANDIDATES, "rust-sasa")
    # A single build with both backends compiled in serves both GPU tools;
    # --backend cuda|vulkan picks which one runs per measurement.
    gpu_exe = args.fastsasa_gpu
    gpu_backends = [("FastSASA CUDA", "cuda"), ("FastSASA Vulkan", "vulkan")]
    cpu_exe = args.fastsasa_cpu
    repeats = args.repeats

    def safe(label: str, callback) -> None:
        print(f"RUN {label}", flush=True)
        try:
            callback()
        except Exception as exc:
            print(f"FAIL {label}: {exc}", flush=True)
            tool = label.split(" ", 1)[0]
            if label.startswith("FastSASA CPU"):
                tool = "FastSASA CPU"
            elif label.startswith("FastSASA CUDA"):
                tool = "FastSASA CUDA"
            elif label.startswith("FastSASA Vulkan"):
                tool = "FastSASA Vulkan"
            elif label.startswith("FreeSASA"):
                tool = "FreeSASA"
            elif label.startswith("RustSASA"):
                tool = "RustSASA"
            elif label.startswith("VMD"):
                tool = "VMD"
            elif label.startswith("BOLT"):
                tool = "mdsasa-bolt/RustSASA"
            add_failure(rows, figure="", case="", case_label="", tool=tool, notes=str(exc))

    for structure in STRUCTURES:
        for threads in (1, 15):
            if rust_sasa:
                safe(f"RustSASA {structure['case']} full t{threads}", lambda s=structure, t=threads: measure_rust_static(rows, rust_sasa, s, threads=t, selected=False, repeats=repeats))
            if freesasa:
                safe(f"FreeSASA {structure['case']} SR full t{threads}", lambda s=structure, t=threads: measure_freesasa_static(rows, freesasa, s, algorithm="SR", resolution=100, threads=t, selected=False, repeats=repeats))
            safe(f"FastSASA CPU {structure['case']} SR full t{threads}", lambda s=structure, t=threads: measure_fastsasa_static(rows, cpu_exe, "FastSASA CPU", s, algorithm="SR", resolution=100, threads=t, selected=False, repeats=repeats))
        safe(f"VMD {structure['case']} full", lambda s=structure: measure_vmd_static(rows, s, selected=False, repeats=max(1, min(2, repeats))))
        for tool, backend in gpu_backends:
            safe(f"{tool} {structure['case']} SR full", lambda s=structure, tool=tool, backend=backend: measure_fastsasa_static(rows, gpu_exe, tool, s, algorithm="SR", resolution=100, threads=None, selected=False, repeats=repeats, backend=backend))

    for structure in SCALING_STRUCTURES:
        for threads in (1, 2, 4, 8, 15):
            safe(f"FastSASA CPU scaling {structure['case']} SR full t{threads}", lambda s=structure, t=threads: measure_fastsasa_static(rows, cpu_exe, "FastSASA CPU", s, algorithm="SR", resolution=100, threads=t, selected=False, repeats=repeats, figure="fig2"))

    algorithm_structure = STRUCTURES[1]
    for threads in (15,):
        if freesasa:
            safe("FreeSASA algorithm SR static", lambda t=threads: measure_freesasa_static(rows, freesasa, algorithm_structure, algorithm="SR", resolution=100, threads=t, selected=False, repeats=repeats, figure="fig7"))
            safe("FreeSASA algorithm LR static", lambda t=threads: measure_freesasa_static(rows, freesasa, algorithm_structure, algorithm="LR", resolution=20, threads=t, selected=False, repeats=repeats, figure="fig7"))
        safe("FastSASA CPU algorithm SR static", lambda t=threads: measure_fastsasa_static(rows, cpu_exe, "FastSASA CPU", algorithm_structure, algorithm="SR", resolution=100, threads=t, selected=False, repeats=repeats, figure="fig7"))
        safe("FastSASA CPU algorithm LR static", lambda t=threads: measure_fastsasa_static(rows, cpu_exe, "FastSASA CPU", algorithm_structure, algorithm="LR", resolution=20, threads=t, selected=False, repeats=repeats, figure="fig7"))
    for tool, backend in gpu_backends:
        safe(f"{tool} algorithm SR static", lambda tool=tool, backend=backend: measure_fastsasa_static(rows, gpu_exe, tool, algorithm_structure, algorithm="SR", resolution=100, threads=None, selected=False, repeats=repeats, figure="fig7", backend=backend))
        safe(f"{tool} algorithm LR static", lambda tool=tool, backend=backend: measure_fastsasa_static(rows, gpu_exe, tool, algorithm_structure, algorithm="LR", resolution=20, threads=None, selected=False, repeats=repeats, figure="fig7", backend=backend))

    for structure in STRUCTURES:
        for threads in (1, 15):
            if rust_sasa:
                safe(f"RustSASA {structure['case']} selected t{threads}", lambda s=structure, t=threads: measure_rust_static(rows, rust_sasa, s, threads=t, selected=True, repeats=repeats))
            if freesasa:
                safe(f"FreeSASA {structure['case']} SR selected t{threads}", lambda s=structure, t=threads: measure_freesasa_static(rows, freesasa, s, algorithm="SR", resolution=100, threads=t, selected=True, repeats=repeats))
            safe(f"FastSASA CPU {structure['case']} SR selected t{threads}", lambda s=structure, t=threads: measure_fastsasa_static(rows, cpu_exe, "FastSASA CPU", s, algorithm="SR", resolution=100, threads=t, selected=True, repeats=repeats))
        safe(f"VMD {structure['case']} selected", lambda s=structure: measure_vmd_static(rows, s, selected=True, repeats=max(1, min(2, repeats))))
        for tool, backend in gpu_backends:
            safe(f"{tool} {structure['case']} SR selected", lambda s=structure, tool=tool, backend=backend: measure_fastsasa_static(rows, gpu_exe, tool, s, algorithm="SR", resolution=100, threads=None, selected=True, repeats=repeats, backend=backend))

    if GABAA_TOP.exists() and GABAA_TRAJ.exists():
        for threads in (15,):
            safe(f"FastSASA CPU trajectory SR full t{threads}", lambda t=threads: measure_fastsasa_trajectory(rows, cpu_exe, "FastSASA CPU", algorithm="SR", resolution=100, threads=t, context="full", repeats=repeats))
        for tool, backend in gpu_backends:
            safe(f"{tool} trajectory SR full", lambda tool=tool, backend=backend: measure_fastsasa_trajectory(rows, gpu_exe, tool, algorithm="SR", resolution=100, threads=None, context="full", repeats=repeats, backend=backend))
        if freesasa:
            for threads in (15,):
                safe(f"FreeSASA trajectory SR full t{threads}", lambda t=threads: measure_freesasa_trajectory(rows, freesasa, algorithm="SR", resolution=100, threads=t, selected=False, repeats=1))
        if rust_sasa:
            safe("RustSASA trajectory full t15", lambda: measure_rust_exported_trajectory(rows, rust_sasa, selected=False, threads=15, repeats=1))
        safe("VMD trajectory full", lambda: measure_vmd_trajectory(rows, selected=False, repeats=1))

        for threads in (15,):
            safe(f"FastSASA CPU trajectory selected full-context t{threads}", lambda t=threads: measure_fastsasa_trajectory(rows, cpu_exe, "FastSASA CPU", algorithm="SR", resolution=100, threads=t, context="selected_protein_context", repeats=repeats))
        for tool, backend in gpu_backends:
            safe(f"{tool} trajectory selected full-context", lambda tool=tool, backend=backend: measure_fastsasa_trajectory(rows, gpu_exe, tool, algorithm="SR", resolution=100, threads=None, context="selected_protein_context", repeats=repeats, backend=backend))
        if freesasa:
            safe("FreeSASA trajectory selected t15", lambda: measure_freesasa_trajectory(rows, freesasa, algorithm="SR", resolution=100, threads=15, selected=True, repeats=1))
        if rust_sasa:
            safe("RustSASA trajectory selected t15", lambda: measure_rust_exported_trajectory(rows, rust_sasa, selected=True, threads=15, repeats=1))
        safe("VMD trajectory selected", lambda: measure_vmd_trajectory(rows, selected=True, repeats=1))

        if freesasa:
            safe("FreeSASA algorithm SR trajectory", lambda: measure_freesasa_trajectory(rows, freesasa, algorithm="SR", resolution=100, threads=15, selected=False, repeats=1, figure="fig7"))
            safe("FreeSASA algorithm LR trajectory", lambda: measure_freesasa_trajectory(rows, freesasa, algorithm="LR", resolution=20, threads=15, selected=False, repeats=1, figure="fig7"))
        safe("FastSASA CPU algorithm SR trajectory", lambda: measure_fastsasa_trajectory(rows, cpu_exe, "FastSASA CPU", algorithm="SR", resolution=100, threads=15, context="full", repeats=repeats, figure="fig7"))
        safe("FastSASA CPU algorithm LR trajectory", lambda: measure_fastsasa_trajectory(rows, cpu_exe, "FastSASA CPU", algorithm="LR", resolution=20, threads=15, context="full", repeats=repeats, figure="fig7"))
        for tool, backend in gpu_backends:
            safe(f"{tool} algorithm SR trajectory", lambda tool=tool, backend=backend: measure_fastsasa_trajectory(rows, gpu_exe, tool, algorithm="SR", resolution=100, threads=None, context="full", repeats=repeats, figure="fig7", backend=backend))
            safe(f"{tool} algorithm LR trajectory", lambda tool=tool, backend=backend: measure_fastsasa_trajectory(rows, gpu_exe, tool, algorithm="LR", resolution=20, threads=None, context="full", repeats=repeats, figure="fig7", backend=backend))

    for tool, backend in gpu_backends:
        safe(f"{tool} FP32 vs FP64", lambda tool=tool, backend=backend: measure_fastsasa_precision(rows, gpu_exe, repeats=repeats, tool=tool, backend=backend))

    return rows


def write_metadata(out_dir: Path, args: argparse.Namespace) -> None:
    clean_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "args": clean_args,
        "tool_order": TOOL_ORDER,
        "gpu": "",
        "cuda": "",
    }
    smi = shutil.which("nvidia-smi")
    if smi:
        proc = run_command([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], timeout=30)
        metadata["gpu"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    nvcc = shutil.which("nvcc")
    if nvcc:
        proc = run_command([nvcc, "--version"], timeout=30)
        metadata["cuda"] = proc.stdout.strip() if proc.returncode == 0 else proc.stderr.strip()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> int:
    global PYTHON_EXECUTABLE, PYTHON_BENCH_PATH

    parser = argparse.ArgumentParser(description="Run focused publication benchmarks and generate clear figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fastsasa-cpu", type=Path, default=ROOT / "build-cpu-strict/fastsasa")
    parser.add_argument("--fastsasa-gpu", type=Path, default=ROOT / "build-cuda-vulkan/fastsasa",
                        help="binary with both CUDA and Vulkan compiled in; --backend selects which runs per measurement")
    parser.add_argument("--freesasa", type=Path)
    parser.add_argument("--rust-sasa", type=Path)
    parser.add_argument("--benchmark-python", default=sys.executable)
    parser.add_argument(
        "--benchmark-pythonpath",
        type=Path,
        help="Optional environment containing mdsasa-bolt/MDAnalysis benchmark dependencies",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--plot-only", type=Path, help="Use an existing publication_benchmark_measurements.csv instead of rerunning tools.")
    args = parser.parse_args()

    PYTHON_EXECUTABLE = args.benchmark_python
    PYTHON_BENCH_PATH = args.benchmark_pythonpath

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = read_rows(args.plot_only)
    else:
        rows = run_measurements(args, out_dir)
        write_rows(rows, out_dir / "publication_benchmark_measurements.csv")
    write_metadata(out_dir, args)
    make_figures(rows, out_dir)
    write_readme(out_dir, rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
