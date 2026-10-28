#!/usr/bin/env python3
"""One-off structure SASA: whole-process and in-process time per tool.

Each structure is computed with Shrake-Rupley 100 points, 1.4 A probe, and
the tools' default atom set (protein heavy atoms; HETATM and waters skipped).
Whole-process time is the median wall time of the command (start-up, file
parsing, device set-up and output included). In-process time is the median
of repeated calls on an already-loaded structure and an open engine.

Usage: python manuscript_benchmark_2026_oneoff.py OUT.json FILE.pdb [...]
FREESASA_CLI, RUST_SASA and VMD may point at the executables.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
import fastsasa_native as fn  # noqa: E402
from fastsasa_adapters import load_radius_config  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manuscript_benchmark_2026_size_series import freesasa_driver  # noqa: E402

FASTSASA = os.path.join(REPO_ROOT, "build-cuda-vulkan/fastsasa")
LIBRARY = os.path.join(REPO_ROOT, "build-cuda-vulkan/libfastsasa_native.so")
FREESASA_CLI = os.environ.get("FREESASA_CLI") or shutil.which("freesasa")
RUST_SASA = os.environ.get("RUST_SASA") or shutil.which("rust-sasa")
VMD = os.environ.get("VMD") or shutil.which("vmd")
REPEATS = 10
THREADS = 15


def median_wall(cmd, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        times.append(time.perf_counter() - t0)
        if proc.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.strip()[-300:]}")
    return float(np.median(times)) * 1000, proc.stdout


def median_call(call, repeats=REPEATS):
    value = call()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000, value


def protein_atoms(path):
    import gemmi

    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()  # first conformer, as the CLIs read it
    config = load_radius_config()
    xyz, radii = [], []
    for chain in st[0]:
        for residue in chain:
            for atom in residue:
                xyz.append(atom.pos.tolist())
                radii.append(config.radius(residue.name, atom.name, atom.element.name, default=1.8))
    return np.array(xyz), np.array(radii)


def whole_process(path, row):
    for backend in ("cpu", "cuda", "vulkan"):
        ms, out = median_wall([FASTSASA, "--backend", backend, "--threads", str(THREADS), path])
        row[f"fastsasa_{backend}_process_ms"] = ms
        row["fastsasa_atoms"] = int(re.search(r"atoms\s*:\s*(\d+)", out).group(1))
        row["fastsasa_total"] = float(re.search(r"Total\s*:\s*([\d.]+)", out).group(1))
    if FREESASA_CLI:
        for threads in (1, THREADS):
            ms, out = median_wall([FREESASA_CLI, "--shrake-rupley", "--resolution=100",
                                   f"--n-threads={threads}", path])
            row[f"freesasa_cli_{threads}t_process_ms"] = ms
            row["freesasa_total"] = float(re.search(r"Total\s*:\s*([\d.]+)", out).group(1))
    if RUST_SASA:
        with tempfile.TemporaryDirectory() as tmp:
            out_json = os.path.join(tmp, "out.json")
            for threads in (1, THREADS):
                ms, _ = median_wall([RUST_SASA, "-o", "protein", "-t", str(threads), path, out_json])
                row[f"rustsasa_cli_{threads}t_process_ms"] = ms
    if VMD:
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "sasa.tcl")
            with open(script, "w") as fh:
                fh.write(f"mol new {path} waitfor all\n"
                         "set sel [atomselect top {protein and noh and (altloc \"\" or altloc A)}]\n"
                         "set t0 [clock microseconds]\n"
                         "set a [measure sasa 1.4 $sel -samples 100]\n"
                         "puts \"VMDRESULT [$sel num] $a [expr {([clock microseconds] - $t0) / 1000.0}]\"\n"
                         "quit\n")
            ms, out = median_wall([VMD, "-dispdev", "text", "-e", script], repeats=5)
            atoms, total, compute_ms = re.search(r"VMDRESULT (\d+) (\S+) (\S+)", out).groups()
            row.update({"vmd_process_ms": ms, "vmd_compute_ms": float(compute_ms),
                        "vmd_atoms": int(atoms), "vmd_total": float(total)})


def in_process(xyz, radii, row):
    for backend, precision in (("cpu", "fp64"), ("cuda", "fp64"), ("cuda", "fp32"),
                               ("vulkan", "fp64"), ("vulkan", "fp32")):
        os.environ["FASTSASA_BACKEND"] = backend
        t0 = time.perf_counter()
        engine = fn.SasaEngine(library_path=LIBRARY, precision=precision)
        engine.sasa(xyz, radii, 1.4, n_points=100)
        row[f"fastsasa_{backend}_{precision}_first_call_ms"] = (time.perf_counter() - t0) * 1000
        ms, _ = median_call(lambda: engine.sasa(xyz, radii, 1.4, n_points=100))
        engine.close()
        row[f"fastsasa_{backend}_{precision}_call_ms"] = ms
    for threads in (1, THREADS):
        seconds, _ = freesasa_driver(xyz, radii, threads, REPEATS)
        row[f"freesasa_lib_{threads}t_call_ms"] = seconds * 1000


def main():
    out_path, files = sys.argv[1], sys.argv[2:]
    rows = []
    for path in files:
        xyz, radii = protein_atoms(path)
        row = {"structure": os.path.basename(path), "protein_heavy_atoms": len(radii)}
        for part in (lambda: whole_process(path, row), lambda: in_process(xyz, radii, row)):
            try:
                part()
            except Exception as error:  # keep what was measured
                row.setdefault("errors", []).append(repr(error))
        rows.append(row)
        print(json.dumps(row, indent=1))
        with open(out_path, "w") as fh:
            json.dump(rows, fh, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
