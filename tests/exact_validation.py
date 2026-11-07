#!/usr/bin/env python3
"""Exact (analytic) SASA on every available backend.

- Closed-form cases: an isolated sphere, two overlapping spheres (spherical
  caps), a sphere buried inside a larger one, two coincident spheres.
- 1ubq heavy atoms against per-atom areas from an independent analytic
  implementation (tests/data/1ubq_exact_reference.csv): FP64 within 1e-8 A^2
  per atom, the GPU mixed-precision mode within 1e-4 A^2.
- Degenerate input (a perfect lattice, where cell vertices fall exactly on
  spheres) resolves to finite areas that agree across backends.
- The CPU result is bit-identical for any thread count, and selection sums
  from the selected-atoms path equal those of the full calculation.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"exact_validation: FAIL: {message}")


def engine_for(module, backend: str, library: str, precision: str = "fp64"):
    os.environ["FASTSASA_BACKEND"] = backend
    try:
        engine = module.SasaEngine(library_path=library, precision=precision)
    except RuntimeError:   # backend not built or no device
        return None
    if engine.backend != backend:
        engine.close()
        return None
    return engine


def cap_area(R: float, other: float, d: float) -> float:
    """Exposed area of sphere R next to sphere `other` at distance d (overlapping)."""
    h = (d * d + R * R - other * other) / (2.0 * d)   # centre to the radical plane
    return 4.0 * math.pi * R * R - 2.0 * math.pi * R * (R - h)


def closed_form(engine, label: str) -> None:
    probe = 1.4
    cases = []
    cases.append(("single", np.array([[0.0, 0.0, 0.0]]), np.array([1.6]),
                  [4.0 * math.pi * (1.6 + probe) ** 2]))
    d = 4.1
    R1, R2 = 1.7 + probe, 1.5 + probe
    cases.append(("pair", np.array([[0.0, 0.0, 0.0], [d, 0.3, -0.2]]), np.array([1.7, 1.5]), None))
    cases.append(("buried", np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]), np.array([3.0, 1.0]),
                  [None, 0.0]))
    for name, xyz, radii, expected in cases:
        _, atom = engine.exact(xyz, radii, probe_radius=probe, atom_sasa=True)
        atom = np.asarray(atom).reshape(-1)
        if name == "pair":
            dist = float(np.linalg.norm(xyz[1] - xyz[0]))
            expected = [cap_area(R1, R2, dist), cap_area(R2, R1, dist)]
        for i, want in enumerate(expected):
            if want is not None and abs(atom[i] - want) > 1e-9 * max(1.0, want):
                fail(f"{label} {name} atom {i}: {atom[i]!r} != {want!r}")
    # two coincident spheres: degenerate, but the parts must add up to one sphere
    xyz = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    _, atom = engine.exact(xyz, np.array([1.8, 1.8]), probe_radius=probe, atom_sasa=True)
    total = float(np.sum(atom))
    want = 4.0 * math.pi * (1.8 + probe) ** 2
    if not np.all(np.isfinite(atom)) or abs(total - want) > 1e-2:
        fail(f"{label} coincident spheres: total {total!r} != {want!r}")


def reference_1ubq() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(REPO / "tests" / "data" / "1ubq_exact_reference.csv", delimiter=",", comments="#", skiprows=3)
    return data[:, :3].copy(), data[:, 3].copy(), data[:, 4].copy()


def lattice() -> tuple[np.ndarray, np.ndarray]:
    g = np.stack(np.meshgrid(*[np.arange(7) * 3.0] * 3, indexing="ij"), -1).reshape(-1, 3)
    return g, np.where(np.arange(len(g)) % 2 == 1, 1.2, 1.8)


def cpu_threads_identical(lib_path: str, xyz: np.ndarray, radii: np.ndarray) -> None:
    lib = ctypes.CDLL(lib_path)
    dp = ctypes.POINTER(ctypes.c_double)
    lib.fastsasa_cpu_exact.argtypes = [ctypes.c_int, dp, dp, dp, dp, ctypes.c_int, dp]
    cols = [np.ascontiguousarray(xyz[:, k]) for k in range(3)]
    R = np.ascontiguousarray(radii + 1.4)
    results = []
    for threads in (1, 3, 8):
        out = np.empty(len(R))
        status = lib.fastsasa_cpu_exact(len(R), *(c.ctypes.data_as(dp) for c in cols), R.ctypes.data_as(dp),
                                        threads, out.ctypes.data_as(dp))
        if status != 0:
            fail(f"fastsasa_cpu_exact({threads} threads) returned {status}")
        results.append(out)
    for out in results[1:]:
        if not np.array_equal(out, results[0]):
            fail("CPU exact SASA depends on the thread count")


ATOM_LINE = re.compile(r"(-?\d+\.\d{3})\s*(-?\d+\.\d{3})\s*(-?\d+\.\d{3})\s+(\d+\.\d{2})\s+(-?\d+\.\d{2})")


def cli_check(fastsasa: str, module, library: str) -> None:
    """--exact through the structure CLI agrees with the library on the CLI's own input."""
    export = subprocess.run([fastsasa, "--format", "pdb", "tests/data/1ubq.cif"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    rows = [ATOM_LINE.search(line) for line in export.splitlines() if line.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(m.group(i)) for i in (1, 2, 3)] for m in rows])
    radii = np.array([float(m.group(4)) for m in rows])
    engine = engine_for(module, "cpu", library)
    want = float(np.asarray(engine.exact(xyz, radii, probe_radius=1.4)).reshape(-1)[0])
    engine.close()
    for args in (["--exact"], ["--exact", "--cpu"]):
        proc = subprocess.run([fastsasa, *args, "--format", "json", "tests/data/1ubq.cif"], cwd=REPO,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            fail(f"fastsasa {' '.join(args)}: {proc.stderr}")
        result = json.loads(proc.stdout)
        if result["parameters"]["algorithm"] != "Exact" or "n_points" in result["parameters"]:
            fail(f"unexpected JSON parameters: {result['parameters']}")
        if abs(result["total_sasa"] - want) > 1e-6 * want:
            fail(f"CLI {' '.join(args)} total {result['total_sasa']} != library {want}")


def trajectory_check(fastsasa: str, backends: list[str]) -> None:
    """trajectory --exact: selections from the selected-atoms path equal those of the full
    calculation, and every backend agrees with the CPU."""
    base = ["trajectory", "--topology", "tests/data/1ubq.pdb", "--trajectory", "tests/data/1ubq_5frames.dcd",
            "--exact", "--select", "near, resi 1-20"]

    def run(backend: str, selected_center: str) -> np.ndarray:
        env = dict(os.environ, FASTSASA_SELECTED_CENTER=selected_center)
        env.pop("FASTSASA_BACKEND", None)
        proc = subprocess.run([fastsasa, *base, "--backend", backend], cwd=REPO, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            fail(f"trajectory --exact --backend {backend}: {proc.stderr}")
        rows = [line.split(",") for line in proc.stdout.splitlines()[1:] if line.strip()]
        return np.array([float(row[2]) for row in rows])

    reference = run("cpu", "0")
    if len(reference) != 5:
        fail(f"trajectory --exact: expected 5 frames, got {len(reference)}")
    for backend in backends:
        for selected_center in ("0", "1"):
            values = run(backend, selected_center)
            if backend == "cpu" and not np.array_equal(values, reference):
                fail(f"trajectory --exact selected-atoms path {values} != full {reference}")
            if np.max(np.abs(values - reference)) > 1e-6 * np.max(reference):
                fail(f"trajectory --exact --backend {backend}: {values} != {reference}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--fastsasa", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.python_dir)
    module = importlib.import_module("fastsasa_native")

    xyz, radii, reference = reference_1ubq()
    lat_xyz, lat_radii = lattice()
    rng = np.random.default_rng(7)
    masks = (rng.random(len(radii)) < 0.1).astype(np.uint32) | ((np.arange(len(radii)) < 40) * 2).astype(np.uint32)

    cpu_threads_identical(args.library, xyz, radii)
    lattice_by_backend = {}
    checked = []
    for backend in ("cpu", "cuda", "vulkan"):
        for precision in ("fp64", "fp32"):
            if backend == "cpu" and precision == "fp32":
                continue
            engine = engine_for(module, backend, args.library, precision)
            if engine is None:
                continue
            label = f"{backend} {precision}"
            try:
                tolerance = 1e-8 if precision == "fp64" else 1e-4
                if precision == "fp64":
                    closed_form(engine, label)
                _, atom = engine.exact(xyz, radii, probe_radius=1.4, atom_sasa=True)
                err = float(np.max(np.abs(np.asarray(atom).reshape(-1) - reference)))
                if err > tolerance:
                    fail(f"{label} 1ubq: max per-atom error {err:.3e} > {tolerance:.0e}")
                _, lat_atom = engine.exact(lat_xyz, lat_radii, probe_radius=1.4, atom_sasa=True)
                lat_atom = np.asarray(lat_atom).reshape(-1)
                if not np.all(np.isfinite(lat_atom)):
                    fail(f"{label} lattice: non-finite area")
                lattice_by_backend[label] = (lat_atom, tolerance)
                sel = engine.exact(xyz, radii, probe_radius=1.4, selection_masks=masks, n_selections=2)["selection"]
                atom = np.asarray(atom).reshape(-1)
                for bit in range(2):
                    want = float(np.sum(atom[(masks >> bit) & 1 == 1]))
                    if abs(float(np.asarray(sel).reshape(-1)[bit]) - want) > 1e-9 * want:
                        fail(f"{label} selection {bit}: {sel} vs {want}")
                checked.append(label)
            finally:
                engine.close()
    base, _ = lattice_by_backend["cpu fp64"]
    for label, (values, tolerance) in lattice_by_backend.items():
        diff = float(np.max(np.abs(values - base)))
        if diff > max(tolerance, 1e-6):
            fail(f"{label} lattice differs from CPU by {diff:.3e}")
    cli_check(args.fastsasa, module, args.library)
    trajectory_check(args.fastsasa, sorted({label.split()[0] for label in checked}))
    print("exact_validation,backends," + ";".join(checked) + ",status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
