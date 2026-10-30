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
CASES = [("SR", 64), ("SR", 100), ("SR", 255), ("SR", 500), ("LR", 10), ("LR", 20)]
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


def arc_overflow_system() -> tuple[np.ndarray, np.ndarray]:
    """A central atom inside a ring of 100 atoms in one plane: every slice
    through the ring needs more arcs than the GPU slice buffer (64) holds,
    so the GPU flags those atoms and the host recomputes them."""
    angles = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    ring = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles), np.zeros_like(angles)], axis=1)
    xyz = np.vstack([[0.0, 0.0, 0.0], ring, [[0.0, 0.0, 6.0]]])
    radii = np.concatenate([[1.8], np.full(100, 1.0), [1.5]])
    return xyz, radii


def coincident_overflow_system() -> tuple[np.ndarray, np.ndarray]:
    """A centre atom buried by a larger atom at exactly its position, after
    40 partly overlapping neighbours have filled the mask kernels' parking
    list (32 entries). The anchor atom fixes the grid origin at z = 0, so
    with cell = 2 * Rmax = 6.8 A the ring (z = 13) lies in the cell below the
    centre (z = 14) and is walked first. The reference buries every centre
    point; the mask kernels must decide the coincident atom exactly even
    when the list is full."""
    angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    ring = np.stack([3.0 * np.cos(angles), 3.0 * np.sin(angles), np.full(40, 13.0)], axis=1)
    centre = [[0.0, 0.0, 14.0]]
    xyz = np.vstack([centre, ring, centre, [[60.0, 0.0, 0.0]]])
    radii = np.concatenate([[1.5], np.full(40, 1.5), [2.0], [1.0]])
    return xyz, radii


def spread_copies(xyz: np.ndarray, radii: np.ndarray, copies: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Copies of one structure spread along x, so the system extent (and the
    magnitude of the GPU kernels' box-local float coordinates) is large while
    every neighbour distance stays that of the original structure."""
    shifted = [xyz + np.array([k * spacing, 0.0, 0.0]) for k in range(copies)]
    return np.vstack(shifted), np.tile(radii, copies)


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

    def compare(backend: str, label: str, values: np.ndarray, reference: np.ndarray) -> None:
        differing = int(np.count_nonzero(values != reference))
        if differing:
            worst = float(np.max(np.abs(values - reference)))
            raise SystemExit(
                f"{backend} FP64 {label}: {differing} of {len(reference)} atoms differ "
                f"from the CPU reference (max |diff| {worst:.3e})")

    structures = {path: load_structure(args.fastsasa, path) for path in STRUCTURES}
    checked = 0
    # Every SR kernel each backend can choose: the default policy, then the
    # mask kernels forced on for structures the policy would route elsewhere.
    for forced in (False, True):
        for name in ("FASTSASA_CUDA_SR_KERNEL", "FASTSASA_VK_SR_KERNEL"):
            if forced:
                os.environ[name] = "mask"
            else:
                os.environ.pop(name, None)
        for structure, (xyz, radii) in structures.items():
            for algorithm, resolution in CASES:
                if forced and algorithm != "SR":
                    continue
                reference, _ = per_atom(module, "cpu", algorithm, resolution, xyz, radii, args.library)
                for backend in gpu_backends:
                    values, _ = per_atom(module, backend, algorithm, resolution, xyz, radii, args.library)
                    compare(backend, f"{algorithm}{resolution} on {structure}"
                            + (" (mask kernel forced)" if forced else ""), values, reference)
                    checked += 1

    # One context switching between kernels: a small structure (reference
    # kernel), a large one (mask kernel), more points than the mask kernel
    # supports, then the mask kernel again with its resources already bound.
    sequence = [(STRUCTURES[0], 100), (STRUCTURES[2], 100), (STRUCTURES[0], 500), (STRUCTURES[2], 100)]
    for backend in gpu_backends:
        os.environ["FASTSASA_BACKEND"] = backend
        engine = module.SasaEngine(library_path=args.library, precision="fp64")
        try:
            for structure, resolution in sequence:
                xyz, radii = structures[structure]
                reference, _ = per_atom(module, "cpu", "SR", resolution, xyz, radii, args.library)
                _, values = engine.sasa(xyz, radii, probe_radius=1.4, n_points=resolution, atom_sasa=True)
                compare(backend, f"SR{resolution} on {structure} (reused context)",
                        np.asarray(values).reshape(-1), reference)
                checked += 1
        finally:
            engine.close()

    # Mask-kernel edge cases, kernels forced on: a coincident atom after the
    # parking list is full, and a 3,000 A extent (large box-local floats).
    os.environ["FASTSASA_CPU_KERNEL"] = "reference"
    for name in ("FASTSASA_CUDA_SR_KERNEL", "FASTSASA_VK_SR_KERNEL"):
        os.environ[name] = "mask"
    edge_cases = {"coincident atom with a full parking list": coincident_overflow_system(),
                  "four 1UBQ copies over 3,000 A": spread_copies(*structures[STRUCTURES[0]], 4, 1000.0)}
    for label, (xyz, radii) in edge_cases.items():
        reference, _ = per_atom(module, "cpu", "SR", 100, xyz, radii, args.library)
        for backend in gpu_backends:
            values, _ = per_atom(module, backend, "SR", 100, xyz, radii, args.library)
            compare(backend, f"SR100 on {label} (mask kernel forced)", values, reference)
            checked += 1
    for name in ("FASTSASA_CPU_KERNEL", "FASTSASA_CUDA_SR_KERNEL", "FASTSASA_VK_SR_KERNEL"):
        os.environ.pop(name, None)

    # Lee-Richards arc overflow: FP64 stays bit-identical; FP32 recomputes
    # the flagged atoms too and stays within FP32 error.
    xyz, radii = arc_overflow_system()
    for slices in (10, 20):
        reference, reference_total = per_atom(module, "cpu", "LR", slices, xyz, radii, args.library)
        for backend in gpu_backends:
            values, _ = per_atom(module, backend, "LR", slices, xyz, radii, args.library)
            compare(backend, f"LR{slices} on the arc-overflow ring", values, reference)
            os.environ["FASTSASA_BACKEND"] = backend
            engine = module.SasaEngine(library_path=args.library, precision="fp32")
            try:
                total, atom = engine.lee_richards(xyz, radii, probe_radius=1.4, n_slices=slices, atom_sasa=True)
            finally:
                engine.close()
            atom = np.asarray(atom).reshape(-1)
            worst = float(np.max(np.abs(atom - reference)))
            if not np.all(np.isfinite(atom)) or worst > 1e-3 * float(np.max(reference)):
                raise SystemExit(f"{backend} FP32 LR{slices} on the arc-overflow ring: max |diff| {worst:.3e}")
            checked += 2

    print(f"fastsasa_backend_bit_identity,status,pass,per_atom_cases,{checked},backends,{'+'.join(gpu_backends)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
