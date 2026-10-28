#!/usr/bin/env python3
"""Single-structure SASA time against system size (compute only).

Reads PDB/mmCIF files (biological assemblies from RCSB, e.g.
2MS2-assembly1.cif.gz) with gemmi, drops waters and hydrogens, assigns the
built-in ProtOr radii (element radii for atoms not in the table), and times
one SR 100-point calculation per backend (median of repeats after a warm-up),
plus FreeSASA 2.1.3's C library through the persistent benchmark driver
(tools/manuscript_benchmark_2026_freesasa_driver, threaded build). Writes a JSON file.

Usage: python manuscript_benchmark_2026_size_series.py OUT.json FILE [FILE ...]
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import gemmi
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
import fastsasa_native as fn  # noqa: E402
from fastsasa_adapters import load_radius_config  # noqa: E402

LIBRARY = os.path.join(REPO_ROOT, "build-cuda-vulkan/libfastsasa_native.so")
FREESASA_DRIVER = os.path.join(REPO_ROOT, "tools/manuscript_benchmark_2026_freesasa_driver/freesasa_calc_coord_driver")
CONFIGS = [("cpu", "fp64"), ("cuda", "fp64"), ("cuda", "fp32"), ("vulkan", "fp64"), ("vulkan", "fp32")]


def load(path):
    st = gemmi.read_structure(path)
    st.remove_waters()
    st.remove_hydrogens()
    config = load_radius_config()
    xyz, radii = [], []
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
                xyz.append(atom.pos.tolist())
                radii.append(config.radius(residue.name, atom.name, atom.element.name, default=1.8))
    return np.array(xyz), np.array(radii)


def freesasa_driver(xyz, radii, threads, repeats):
    """Median time of FreeSASA 2.1.3 freesasa_calc_coord() (patched, threaded build)."""
    with tempfile.TemporaryDirectory() as tmp:
        files = [os.path.join(tmp, n) for n in ("coords.f64", "radii.f64", "selection.u32")]
        xyz.astype("<f8").tofile(files[0])
        radii.astype("<f8").tofile(files[1])
        np.ones(len(radii), dtype="<u4").tofile(files[2])
        proc = subprocess.run([FREESASA_DRIVER, *files, str(len(radii)), "1", str(threads), "complete",
                               str(repeats)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-300:])
    runs = [dict(kv.split("=") for kv in line.split()) for line in proc.stdout.splitlines() if line.startswith("repeat=")]
    return float(np.median([float(r["elapsed"]) for r in runs])), float(runs[-1]["total_sum"])


def timed(call, repeats):
    call()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = call()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), value


def main():
    out_path, files = sys.argv[1], sys.argv[2:]
    rows = []
    for path in files:
        xyz, radii = load(path)
        name = os.path.basename(path).split(".")[0]
        repeats = 5 if len(radii) < 200_000 else 3
        row = {"structure": name, "atoms": len(radii)}
        reference = None
        for backend, precision in CONFIGS:
            os.environ["FASTSASA_BACKEND"] = backend
            try:
                engine = fn.SasaEngine(library_path=LIBRARY, precision=precision)
                seconds, total = timed(lambda: float(engine.sasa(xyz, radii, 1.4, n_points=100)[0]), repeats)
                engine.close()
            except Exception as error:  # e.g. out of device memory; keep the other backends
                row[f"{backend}_{precision}_error"] = repr(error)
                continue
            reference = total if reference is None else reference
            row[f"{backend}_{precision}_ms"] = seconds * 1000
            row[f"{backend}_{precision}_rel_vs_cpu"] = abs(total - reference) / reference
        for threads in (1, 15):
            try:
                seconds, total = freesasa_driver(xyz, radii, threads, 1 if len(radii) > 200_000 else repeats)
            except (RuntimeError, ValueError) as error:
                row[f"freesasa_{threads}t_error"] = repr(error)
                continue
            row[f"freesasa_{threads}t_ms"] = seconds * 1000
            row[f"freesasa_{threads}t_rel_vs_cpu"] = abs(total - reference) / reference
        rows.append(row)
        with open(out_path, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"{name:16s} {len(radii):9d} atoms: " + ", ".join(
            f"{k[:-3]} {v:.1f}" for k, v in row.items() if k.endswith("_ms")) + " ms")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
