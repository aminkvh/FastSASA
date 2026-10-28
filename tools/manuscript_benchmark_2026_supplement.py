#!/usr/bin/env python3
"""Supplementary measurements for the FastSASA manuscript.

Uses the inputs cached by `manuscript_benchmark_2026.py prepare` (datasets
cx46, gabaa_baba and their *_heavy variants). Each mode writes a JSON file
and prints a short table.

Modes:
  hardware   CPU, GPU, driver, compiler and library versions.
  accuracy   Error of SR 100/500/1000 points and LR 20 slices against LR
             1000 slices (total, per-residue RMS and maximum), time per frame.
  fp32       FP32 against FP64: mean and maximum, relative and absolute.
  points     Time per frame across SR point counts (the 255/256 boundary).
  lr         Lee-Richards time per frame on every backend and precision.
  prefilter  Share of point-neighbour tests the FP32 prefilter cannot decide
             (exact NumPy model of the CUDA/Vulkan FP32 test and margin).
  freesasa   Per-atom agreement with FreeSASA's C library (same radii).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import fastsasa_native as fn  # noqa: E402
from manuscript_benchmark_2026 import DATASETS, _fastsasa_library_path, _load_prepared  # noqa: E402

PROBE = 1.4
RESULTS = os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026/supplement")
GPU_CONFIGS = [("cuda", "fp64"), ("cuda", "fp32"), ("vulkan", "fp64"), ("vulkan", "fp32")]


def _engine(backend, precision="fp64"):
    os.environ["FASTSASA_BACKEND"] = backend
    engine = fn.SasaEngine(library_path=_fastsasa_library_path(), precision=precision)
    if engine.backend != backend:
        engine.close()
        raise RuntimeError(f"backend {backend} unavailable (got {engine.backend})")
    return engine


def _prepared(dataset, n_frames=None):
    data = _load_prepared(dataset)
    raw = np.load(os.path.join(DATASETS[dataset]["out_dir"], "prepared_inputs.npz"))
    data["residue_ids"] = raw["residue_ids"]
    if n_frames:
        data["coords"] = data["coords"][:n_frames]
    return data


def _timed(fn_call, repeats=3):
    """Median wall time of fn_call() after one warm-up call, and its result."""
    result = fn_call()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn_call()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def _save(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {path}")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cmd_hardware(_args):
    lscpu = _run(["lscpu"])
    info = {
        "cpu": next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")), ""),
        "logical_cpus": os.cpu_count(),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "vulkan_devices": [l.split("=", 1)[1].strip() for l in _run(["vulkaninfo", "--summary"]).splitlines()
                           if "deviceName" in l or "driverInfo" in l],
        "nvcc": _run(["nvcc", "--version"]).splitlines()[-1:] or [""],
        "gcc": _run(["gcc", "--version"]).splitlines()[:1],
        "kernel": platform.release(),
        "os": _run(["lsb_release", "-ds"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "fastsasa_commit": _run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"]),
        "fastsasa_worktree_modified": bool(_run(["git", "-C", REPO_ROOT, "status", "--porcelain", "--untracked-files=no"])),
    }
    for key, value in info.items():
        print(f"{key:28s} {value}")
    _save("hardware", info)


def _per_residue(atom_values, residue_ids, n_residues):
    return np.stack([np.bincount(residue_ids, frame, n_residues) for frame in atom_values])


def cmd_accuracy(args):
    data = _prepared(args.dataset, args.frames)
    coords, radii, rid = data["coords"], data["radii"], data["residue_ids"]
    n_res = int(rid.max()) + 1
    cpu = _engine("cpu")
    _, ref_atoms = cpu.lee_richards(coords, radii, PROBE, n_slices=args.reference_slices, atom_sasa=True)
    ref_res = _per_residue(np.asarray(ref_atoms), rid, n_res)
    ref_total = ref_res.sum(axis=1)
    settings = [("SR", 100), ("SR", 500), ("SR", 1000), ("LR", 20), ("LR", 100), ("LR", 500)]
    rows = []
    gpu = {cfg: _engine(*cfg) for cfg in [("cuda", "fp64"), ("cuda", "fp32")]}
    for algorithm, resolution in settings:
        for (backend, precision), engine in [(("cpu", "fp64"), cpu)] + list(gpu.items()):
            if algorithm == "SR":
                call = lambda e=engine: e.sasa(coords, radii, PROBE, n_points=resolution, atom_sasa=True)
            else:
                call = lambda e=engine: e.lee_richards(coords, radii, PROBE, n_slices=resolution, atom_sasa=True)
            seconds, (_, atoms) = _timed(call, repeats=1 if backend == "cpu" else 3)
            res = _per_residue(np.asarray(atoms), rid, n_res)
            err = res - ref_res
            rows.append({
                "algorithm": algorithm, "resolution": resolution, "backend": backend, "precision": precision,
                "mean_abs_total_error_pct": float(np.mean(np.abs(res.sum(axis=1) - ref_total) / ref_total) * 100),
                "residue_rms_error_A2": float(np.sqrt(np.mean(err ** 2))),
                "residue_max_error_A2": float(np.max(np.abs(err))),
                "ms_per_frame": seconds / len(coords) * 1000,
            })
            r = rows[-1]
            print(f"{algorithm}{resolution:<5d} {backend:6s} {precision}: total {r['mean_abs_total_error_pct']:.4f}% "
                  f"res RMS {r['residue_rms_error_A2']:.3f} max {r['residue_max_error_A2']:.2f} A2, "
                  f"{r['ms_per_frame']:.2f} ms/frame")
    for engine in [cpu] + list(gpu.values()):
        engine.close()
    _save(f"accuracy_{args.dataset}", {"dataset": args.dataset, "frames": len(coords),
                                       "reference": f"CPU LR {args.reference_slices} slices", "rows": rows})


def cmd_fp32(args):
    data = _prepared(args.dataset)
    coords, radii, mask = data["coords"], data["radii"], data["selection_mask"]
    cpu = _engine("cpu")
    _, ref_atoms = cpu.sasa(coords, radii, PROBE, n_points=100, atom_sasa=True)
    cpu.close()
    ref_atoms = np.asarray(ref_atoms)
    sel = mask.astype(bool)
    out = {"dataset": args.dataset, "frames": len(coords), "rows": []}
    for backend in ("cuda", "vulkan"):
        engine = _engine(backend, "fp32")
        _, atoms = engine.sasa(coords, radii, PROBE, n_points=100, atom_sasa=True)
        engine.close()
        atoms = np.asarray(atoms)
        for panel, columns in (("complete", slice(None)), ("selected", sel)):
            ref = ref_atoms[:, columns].sum(axis=1)
            val = atoms[:, columns].sum(axis=1)
            absolute = np.abs(val - ref)
            relative = absolute / ref
            row = {"backend": backend, "panel": panel,
                   "mean_relative": float(relative.mean()), "max_relative": float(relative.max()),
                   "mean_abs_A2": float(absolute.mean()), "max_abs_A2": float(absolute.max()),
                   "atom_max_abs_A2": float(np.abs(atoms[:, columns] - ref_atoms[:, columns]).max())}
            out["rows"].append(row)
            print(f"{backend:6s} {panel:8s}: relative mean {row['mean_relative']:.2e} max {row['max_relative']:.2e}; "
                  f"absolute mean {row['mean_abs_A2']:.3f} max {row['max_abs_A2']:.3f} A2; "
                  f"per-atom max {row['atom_max_abs_A2']:.3f} A2")
    _save(f"fp32_{args.dataset}", out)


def cmd_points(args):
    data = _prepared(args.dataset, args.frames)
    coords, radii = data["coords"], data["radii"]
    rows = []
    for backend, precision in [("cpu", "fp64")] + GPU_CONFIGS:
        engine = _engine(backend, precision)
        for points in args.points:
            seconds, _ = _timed(lambda: engine.sasa(coords, radii, PROBE, n_points=points))
            rows.append({"backend": backend, "precision": precision, "points": points,
                         "ms_per_frame": seconds / len(coords) * 1000})
            print(f"{backend:6s} {precision} {points:4d} points: {rows[-1]['ms_per_frame']:.3f} ms/frame")
        engine.close()
    _save(f"points_{args.dataset}", {"dataset": args.dataset, "frames": len(coords), "rows": rows})


def cmd_lr(args):
    data = _prepared(args.dataset, args.frames)
    coords, radii = data["coords"], data["radii"]
    rows = []
    reference = None
    for backend, precision in [("cpu", "fp64")] + GPU_CONFIGS:
        engine = _engine(backend, precision)
        seconds, totals = _timed(lambda: engine.lee_richards(coords, radii, PROBE, n_slices=args.slices),
                                 repeats=args.repeats)
        engine.close()
        totals = np.asarray(totals)
        if reference is None:
            reference = totals
        rows.append({"backend": backend, "precision": precision, "slices": args.slices,
                     "ms_per_frame": seconds / len(coords) * 1000,
                     "identical_to_cpu": bool(np.array_equal(totals, reference)),
                     "max_relative_vs_cpu": float(np.max(np.abs(totals - reference) / reference))})
        print(f"{backend:6s} {precision} LR{args.slices}: {rows[-1]['ms_per_frame']:.1f} ms/frame, "
              f"identical to CPU: {rows[-1]['identical_to_cpu']} (max rel {rows[-1]['max_relative_vs_cpu']:.1e})")
    _save(f"lr_{args.dataset}", {"dataset": args.dataset, "frames": len(coords), "atoms": len(radii),
                                 "rows": rows})


def _fibonacci_points(n):
    """Unit test points; any well-spread set gives the same statistics."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def cmd_prefilter(args):
    """Counts point-neighbour tests whose FP32 value lies inside the margin.

    Mirrors shrake_rupley_cell_hybrid_kernel (src/sr_cuda.cu): box-local FP32
    coordinates, diff = dx^2 + dy^2 + dz^2 - R_j^2 in FP32, margin
    2(2 dmax delta + delta^2 + 8 eps dmax^2) with delta = 8 eps extent and
    dmax = 6 cells. All candidates in the 27-cell block are counted (the
    kernel stops early at the first burying neighbour, so this is the share
    of all candidate tests, an upper-bound view of the work).
    """
    data = _prepared(args.dataset, 1)
    xyz = data["coords"][0]
    expanded = data["radii"] + PROBE
    cell = 2.0 * expanded.max()
    lo = xyz.min(axis=0)
    idx = np.floor((xyz - lo) / cell).astype(int)
    dims = idx.max(axis=0) + 1
    eps = 2.0 ** -24
    extent = cell * (dims.max() + 1)
    delta = 8 * eps * extent
    dmax = 6 * cell
    margin = np.float32(2 * (2 * dmax * delta + delta * delta + 8 * eps * dmax * dmax))
    keys = idx[:, 0] + dims[0] * (idx[:, 1] + dims[1] * idx[:, 2])
    order = np.argsort(keys, kind="stable")
    starts = np.searchsorted(keys[order], np.arange(dims.prod() + 1))
    local32 = (xyz - lo).astype(np.float32)
    r2f = (expanded * expanded).astype(np.float32)
    points = _fibonacci_points(args.points)
    total = ambiguous = 0
    for atom in range(len(xyz)):
        neighbours = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    c = idx[atom] + (dx, dy, dz)
                    if np.any(c < 0) or np.any(c >= dims):
                        continue
                    k = c[0] + dims[0] * (c[1] + dims[1] * c[2])
                    neighbours.append(order[starts[k]:starts[k + 1]])
        nb = np.concatenate(neighbours)
        nb = nb[nb != atom]
        p = (xyz[atom] + expanded[atom] * points - lo).astype(np.float32)
        d = p[:, None, :] - local32[nb][None, :, :]
        diff = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]) - r2f[nb][None, :]
        total += diff.size
        ambiguous += int(np.count_nonzero(np.abs(diff) <= margin))
    out = {"dataset": args.dataset, "points": args.points, "margin_A2": float(margin),
           "tests": total, "ambiguous": ambiguous, "ambiguous_fraction": ambiguous / total}
    print(f"{args.dataset}: margin {float(margin):.3e} A^2; {ambiguous:,} of {total:,} tests "
          f"({ambiguous / total:.2e}) go to FP64")
    _save(f"prefilter_{args.dataset}", out)


def cmd_freesasa(args):
    import freesasa

    data = _prepared(args.dataset, args.frames)
    coords, radii = data["coords"], data["radii"]
    freesasa.setVerbosity(freesasa.silent)
    out = {"dataset": args.dataset, "frames": len(coords), "rows": []}
    cpu = _engine("cpu")
    for points in (100, 500):
        params = freesasa.Parameters()
        params.setAlgorithm(freesasa.ShrakeRupley)
        params.setProbeRadius(PROBE)
        params.setNPoints(points)
        _, ours = cpu.sasa(coords, radii, PROBE, n_points=points, atom_sasa=True)
        ours = np.asarray(ours)
        worst_atom = worst_total = 0.0
        for frame in range(len(coords)):
            result = freesasa.calcCoord(coords[frame].reshape(-1).tolist(), radii.tolist(), params)
            theirs = np.array([result.atomArea(i) for i in range(len(radii))])
            worst_atom = max(worst_atom, float(np.max(np.abs(theirs - ours[frame]))))
            worst_total = max(worst_total, abs(result.totalArea() - ours[frame].sum()) / result.totalArea())
        out["rows"].append({"points": points, "max_atom_abs_A2": worst_atom, "max_total_relative": worst_total})
        print(f"SR{points}: max per-atom |diff| {worst_atom:.2e} A2, max total relative {worst_total:.2e}")
    cpu.close()
    _save(f"freesasa_{args.dataset}", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("hardware")
    for name in ("accuracy", "fp32", "points", "lr", "prefilter", "freesasa"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="cx46_heavy", choices=DATASETS.keys())
        p.add_argument("--frames", type=int, default=None)
        if name == "accuracy":
            p.add_argument("--reference-slices", type=int, default=1000)
        if name == "points":
            p.add_argument("--points", type=int, nargs="+", default=[64, 100, 128, 200, 255, 256, 300, 500, 1000])
        if name == "lr":
            p.add_argument("--slices", type=int, default=20)
            p.add_argument("--repeats", type=int, default=3)
        if name == "prefilter":
            p.add_argument("--points", type=int, default=100)
    args = parser.parse_args()
    globals()[f"cmd_{args.mode}"](args)


if __name__ == "__main__":
    main()
