#!/usr/bin/env python3
"""Trajectory SASA with other tools on the same atoms (end to end per tool).

Writes the protein heavy atoms of a benchmark dataset to PDB, DCD and XTC,
then times each tool reading that trajectory itself: FastSASA CLI (CPU,
CUDA, Vulkan), MDTraj, mdakit-sasa (FreeSASA), mdsasa-bolt (RustSASA),
zsasa, cpptraj (LCPO surf), GROMACS gmx sasa and VMD measure sasa. Tools use
their own radii and algorithms, so mean areas are reported next to times.

Usage: python manuscript_benchmark_2026_tools.py DATASET OUT_DIR [--points 100]
Needs MDAnalysis, mdtraj, mdakit_sasa, mdsasa_bolt and zsasa on this
Python, plus cpptraj, gmx and vmd on PATH (missing tools are skipped).
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from manuscript_benchmark_2026 import DATASETS  # noqa: E402

FASTSASA = os.path.join(REPO_ROOT, "build-cuda-vulkan/fastsasa")
THREADS = 15


def write_inputs(dataset, out, repeat=1):
    import MDAnalysis as mda
    from fastsasa_adapters import _mdanalysis_elements

    ds = DATASETS[dataset]
    u = mda.Universe(ds["topology"], ds["trajectory"])
    atoms = u.select_atoms(ds["system_selection"])
    atoms = atoms[np.array([e != "H" for e in _mdanalysis_elements(atoms)])]
    paths = {k: os.path.join(out, f"system.{k}") for k in ("pdb", "dcd", "xtc")}
    if not all(os.path.exists(p) for p in paths.values()):
        atoms.write(paths["pdb"])
        for key in ("dcd", "xtc"):
            with mda.Writer(paths[key], atoms.n_atoms) as writer:
                for _ in range(repeat):
                    for _ in u.trajectory:
                        writer.write(atoms)
    return paths, atoms.n_atoms, len(u.trajectory) * repeat


def timed_process(cmd, **kwargs):
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return time.perf_counter() - t0, proc


def run_fastsasa(paths, points, rows):
    for backend, precision in (("cpu", "fp64"), ("cpu", "fp32"), ("cuda", "fp64"), ("cuda", "fp32"),
                               ("vulkan", "fp64"), ("vulkan", "fp32")):
        cmd = [FASTSASA, "trajectory", "--topology", paths["pdb"], "--trajectory", paths["dcd"],
               "--resolution", str(points), "--summary", "--backend", backend, "--precision", precision]
        seconds, proc = timed_process(cmd)
        if proc.returncode != 0:
            print(f"fastsasa {backend} {precision} failed: {proc.stderr.strip()[-200:]}")
            continue
        info = list(csv.DictReader(proc.stdout.strip().splitlines()[-2:]))[0]
        rows.append({"tool": f"FastSASA {backend} {precision}", "wall_s": seconds,
                     "compute_s": float(info["gpu_seconds"]),
                     "mean_A2": float(info["total_sasa_sum"]) / int(info["frames"])})


def run_mdtraj(paths, points, rows):
    import mdtraj

    t0 = time.perf_counter()
    traj = mdtraj.load(paths["dcd"], top=paths["pdb"])
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    areas = mdtraj.shrake_rupley(traj, probe_radius=0.14, n_sphere_points=points).sum(axis=1)
    compute = time.perf_counter() - t0
    rows.append({"tool": "MDTraj shrake_rupley", "wall_s": load + compute, "compute_s": compute,
                 "mean_A2": float(areas.mean() * 100)})


def run_mdanalysis_tools(paths, points, rows, names):
    import MDAnalysis as mda
    from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis as FreeSASAAnalysis
    from mdsasa_bolt import SASAAnalysis as BoltAnalysis

    # mdakit-sasa exposes no settings: FreeSASA defaults (Lee-Richards 20 slices, 2 threads).
    tools = (("mdakit-sasa (FreeSASA)", FreeSASAAnalysis, {}, "FreeSASA defaults: LR 20 slices, 2 threads"),
             ("mdsasa-bolt (RustSASA)", BoltAnalysis, {"n_points": points, "probe_radius": 1.4},
              f"SR {points} points"))
    for name, cls, kwargs, note in (tool for tool in tools if tool[0].split()[0] in names):
        t0 = time.perf_counter()
        u = mda.Universe(paths["pdb"], paths["dcd"])
        analysis = cls(u, select="all", **kwargs)
        analysis.run()  # mdsasa-bolt's run() returns None
        seconds = time.perf_counter() - t0
        result = analysis.results
        total = getattr(result, "total_area", None)
        mean = float(np.mean(total)) if total is not None else float("nan")
        rows.append({"tool": name, "wall_s": seconds, "compute_s": None, "mean_A2": mean,
                     "note": note})


def run_zsasa(paths, points, rows, out):
    exe = shutil.which("zsasa") or os.path.join(os.path.dirname(sys.executable), "zsasa")
    if not os.path.exists(exe):
        return
    for precision, extra in (("f64", []), ("f32", []), ("f32", ["--use-bitmask"])):
        result = os.path.join(out, f"zsasa_{precision}{'_bitmask' if extra else ''}.csv")
        cmd = [exe, "traj", paths["dcd"], paths["pdb"], "--classifier=protor", f"--n-points={points}",
               f"--precision={precision}", f"--threads={THREADS}", "-o", result] + extra
        seconds, proc = timed_process(cmd)
        if proc.returncode != 0:
            print(f"zsasa failed: {proc.stderr.strip()[-300:]}")
            continue
        mean = float("nan")
        try:
            data = np.genfromtxt(result, delimiter=",", names=True)
            column = [n for n in data.dtype.names if "total" in n.lower()]
            mean = float(np.mean(data[column[0]])) if column else mean
        except (OSError, ValueError, IndexError):
            pass
        rows.append({"tool": f"zsasa {precision}{' bitmask' if extra else ''}", "wall_s": seconds,
                     "compute_s": None, "mean_A2": mean})


def run_cpptraj(paths, rows, out):
    exe = shutil.which("cpptraj") or os.path.join(os.path.dirname(sys.executable), "cpptraj")
    if not os.path.exists(exe):
        return
    script = os.path.join(out, "surf.in")
    result = os.path.join(out, "cpptraj_surf.dat")
    with open(script, "w") as fh:
        fh.write(f"parm {paths['pdb']}\ntrajin {paths['dcd']}\nsurf S out {result}\nrun\n")
    seconds, proc = timed_process([exe, "-i", script])
    if proc.returncode == 0:
        values = np.loadtxt(result, comments="#")[:, 1]
        rows.append({"tool": "cpptraj surf (LCPO)", "wall_s": seconds, "compute_s": None,
                     "mean_A2": float(values.mean()), "note": "analytic LCPO approximation, 1 thread"})


def run_gmx(paths, points, rows, out):
    exe = shutil.which("gmx")
    if exe is None:
        return
    result = os.path.join(out, "gmx_sasa.xvg")
    cmd = [exe, "sasa", "-f", paths["xtc"], "-s", paths["pdb"], "-o", result, "-ndots", str(points),
           "-probe", "0.14", "-surface", "System", "-quiet"]
    seconds, proc = timed_process(cmd, input="0\n", env=dict(os.environ, OMP_NUM_THREADS=str(THREADS)))
    if proc.returncode == 0 and os.path.exists(result):
        values = np.array([float(l.split()[1]) for l in open(result) if l[:1] not in "#@"])
        rows.append({"tool": "GROMACS gmx sasa", "wall_s": seconds, "compute_s": None,
                     "mean_A2": float(values.mean() * 100), "note": f"-ndots {points}"})
    else:
        print(f"gmx sasa failed: {proc.stderr.strip()[-300:]}")


def run_vmd(paths, points, rows, out):
    exe = shutil.which("vmd")
    if exe is None:
        return
    script = os.path.join(out, "vmd_sasa.tcl")
    with open(script, "w") as fh:
        fh.write(f"""mol new {paths['pdb']} type pdb waitfor all
mol addfile {paths['dcd']} type dcd waitfor all
set sel [atomselect top all]
set n [molinfo top get numframes]
set t0 [clock microseconds]
set sum 0.0
for {{set f 0}} {{$f < $n}} {{incr f}} {{
  $sel frame $f
  set sum [expr {{$sum + [measure sasa 1.4 $sel -samples {points}]}}]
}}
puts "VMDRESULT [expr {{([clock microseconds] - $t0) / 1e6}}] [expr {{$sum / $n}}]"
quit
""")
    seconds, proc = timed_process([exe, "-dispdev", "text", "-e", script])
    match = re.search(r"VMDRESULT (\S+) (\S+)", proc.stdout)
    if match:
        rows.append({"tool": "VMD measure sasa", "wall_s": seconds, "compute_s": float(match.group(1)),
                     "mean_A2": float(match.group(2)), "note": f"-samples {points}, VMD radii, 1 thread"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS.keys())
    parser.add_argument("out")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1,
                        help="write the trajectory this many times over (longer runs for fast tools)")
    parser.add_argument("--only", default="", help="comma list: fastsasa,mdtraj,mdakit,bolt,zsasa,cpptraj,gmx,vmd")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths, n_atoms, n_frames = write_inputs(args.dataset, args.out, args.repeat)
    rows = []
    out_json = os.path.join(args.out, "tools.json")
    steps = {"fastsasa": lambda: run_fastsasa(paths, args.points, rows),
             "mdtraj": lambda: run_mdtraj(paths, args.points, rows),
             "mdakit": lambda: run_mdanalysis_tools(paths, args.points, rows, {"mdakit-sasa"}),
             "bolt": lambda: run_mdanalysis_tools(paths, args.points, rows, {"mdsasa-bolt"}),
             "zsasa": lambda: run_zsasa(paths, args.points, rows, args.out),
             "cpptraj": lambda: run_cpptraj(paths, rows, args.out),
             "gmx": lambda: run_gmx(paths, args.points, rows, args.out),
             "vmd": lambda: run_vmd(paths, args.points, rows, args.out)}
    selected = [name for name in args.only.split(",") if name] or list(steps)
    if args.only:
        out_json = os.path.join(args.out, f"tools_{'_'.join(selected)}.json")
    for step in (steps[name] for name in selected):
        try:
            step()
        except Exception as error:  # keep the other tools' results
            print(f"tool step failed: {error!r}")
        with open(out_json, "w") as fh:
            json.dump({"dataset": args.dataset, "atoms": n_atoms, "frames": n_frames,
                       "points": args.points, "rows": rows}, fh, indent=2)
    for row in rows:
        compute = f"{row['compute_s']:.2f}" if row.get("compute_s") is not None else "-"
        print(f"{row['tool']:28s} wall {row['wall_s']:8.2f} s  compute {compute:>8s} s  "
              f"{n_frames / row['wall_s']:9.1f} frames/s  mean {row['mean_A2']:.0f} A2")


if __name__ == "__main__":
    main()
