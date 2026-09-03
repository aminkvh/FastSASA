#!/usr/bin/env python3
"""Bit-identity of the GPU FP64 backends against the CPU reference.

CUDA FP64 and Vulkan FP64 must reproduce the CPU backend's per-atom
Shrake-Rupley and Lee-Richards areas exactly (every double equal).

Structures come from tests/data; coordinates and radii are taken from the
structure CLI's PDB export so every backend sees the CLI's own input.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

STRUCTURES = ["tests/data/1ubq.pdb", "tests/data/3bkr.pdb", "tests/data/2isk.pdb"]
CASES = [("SR", 100), ("SR", 500), ("LR", 10), ("LR", 20)]
ATOM_LINE = re.compile(r"(-?\d+\.\d{3})\s*(-?\d+\.\d{3})\s*(-?\d+\.\d{3})\s+(\d+\.\d{2})\s+(-?\d+\.\d{2})")


def load_structure(fastsasa: Path, path: str) -> tuple[np.ndarray, np.ndarray]:
    text = subprocess.run([str(fastsasa), "--format", "pdb", path],
                          capture_output=True, text=True, check=True).stdout
    xyz, radii = [], []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        match = ATOM_LINE.search(line)
        if match is None:
            raise SystemExit(f"unparseable PDB export line: {line!r}")
        xyz.append(tuple(float(match.group(i)) for i in (1, 2, 3)))
        radii.append(float(match.group(4)))
    return np.array(xyz), np.array(radii)


def per_atom(module, backend: str, algorithm: str, resolution: int,
             xyz: np.ndarray, radii: np.ndarray, library: str) -> tuple[np.ndarray, float]:
    os.environ["FASTSASA_BACKEND"] = backend
    engine = module.SasaEngine(library_path=library, precision="fp64")
    try:
        if engine.backend != backend:
            raise SystemExit(f"requested backend {backend} but got {engine.backend}")
        if algorithm == "SR":
            total, atom = engine.sasa(xyz, radii, probe_radius=1.4, n_points=resolution, atom_sasa=True)
        else:
            total, atom = engine.lee_richards(xyz, radii, probe_radius=1.4, n_slices=resolution, atom_sasa=True)
    finally:
        engine.close()
    return np.asarray(atom).reshape(-1), float(np.asarray(total).reshape(-1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--fastsasa", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, args.python_dir)
    module = importlib.import_module("fastsasa_native")
    os.environ["FASTSASA_BACKEND"] = "cuda"
    probe = module.SasaEngine(library_path=args.library, precision="fp64")
    available = probe.backend
    probe.close()
    if available != "cuda":
        print("fastsasa_backend_bit_identity,status,skip,reason,no CUDA device")
        return 0
    gpu_backends = ["cuda"]
    os.environ["FASTSASA_BACKEND"] = "vulkan"
    try:
        probe = module.SasaEngine(library_path=args.library, precision="fp64")
        if probe.backend == "vulkan":
            gpu_backends.append("vulkan")
        probe.close()
    except Exception:
        pass

    checked = 0
    for structure in STRUCTURES:
        xyz, radii = load_structure(args.fastsasa, structure)
        for algorithm, resolution in CASES:
            reference, _ = per_atom(module, "cpu", algorithm, resolution, xyz, radii, args.library)
            for backend in gpu_backends:
                values, _ = per_atom(module, backend, algorithm, resolution, xyz, radii, args.library)
                differing = int(np.count_nonzero(values != reference))
                if differing:
                    worst = float(np.max(np.abs(values - reference)))
                    raise SystemExit(
                        f"{backend} FP64 {algorithm}{resolution} on {structure}: "
                        f"{differing} of {len(reference)} atoms differ from the CPU reference "
                        f"(max |diff| {worst:.3e})")
                checked += 1

    print(f"fastsasa_backend_bit_identity,status,pass,per_atom_cases,{checked},backends,{'+'.join(gpu_backends)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
