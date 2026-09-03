#!/usr/bin/env python3
"""Reproducible trajectory-engine benchmark for the FastSASA manuscript
(Figure 1: FreeSASA vs FastSASA CPU/CUDA/Vulkan compute-only throughput).

Each (tool, thread-count, precision, panel) configuration is measured in its
own subprocess. That is standard benchmarking hygiene on its own, but for
FreeSASA it is also a hard requirement: upstream FreeSASA 2.1.3/2.2.1's
Shrake-Rupley implementation has a NULL-pointer defect in sr_atom_area()
(src/sasa_sr.c) that crashes on atoms with zero SASA-relevant neighbors --
which includes the isolated counter-ions present in the Cx46 system used
here. See tools/manuscript_benchmark_2026_freesasa_driver/ for the
AddressSanitizer trace, the one-line fix, and a persistent C driver built
against a patched libfreesasa. That driver -- not freesasa's Python bindings
-- is what this script calls for the `freesasa` tool: it loads all frames
once per process and times only the freesasa_calc_coord() loop internally,
satisfying the spec's "in-process API or persistent benchmark driver"
requirement without the crash.

Modes:
  prepare      Build canonical radii + decode trajectory once, cache to .npz
               and to flat binaries for the FreeSASA C driver.
  run          Run one configuration (warmup + repeats), append to CSV.
  orchestrate  Invoke `run` as a fresh subprocess for every configuration.
  numeric      Cross-backend numerical checks (bit-identity, FP32 vs FP64).
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

PROBE_RADIUS = 1.4
N_POINTS = 100
ALGORITHM_SR = 0
N_THREADS_LOW = 1
N_THREADS_HIGH = 15
REPEATS = 3

DATASETS = {
    # "system_selection" restricts the whole loaded system (both panels) to
    # protein atoms only -- no solvent, ions, or membrane -- so raw frames/s
    # compares like for like across datasets. MDAnalysis's "protein"
    # selector keyword (standard 20 aa + common CHARMM variants e.g.
    # HSD/HSE) is used, not a hand-rolled segid/resname list, so it's the
    # same selector for every dataset here.
    "cx46": {
        "name": "Cx46 hemichannel (Zenodo 10.5281/zenodo.4625961)",
        "topology": os.path.join(REPO_ROOT, "benchmark_corpus/trajectories/cx46_hemichannel/Cx46_Ace_ProtIon.psf"),
        "trajectory": os.path.join(REPO_ROOT, "benchmark_corpus/trajectories/cx46_hemichannel/Cx46_Ace_ProtIon_01.dcd"),
        "system_selection": "protein",
        "selection": "segid AP1",
        "selection_description": "one connexin protomer (segid AP1)",
        "out_dir": os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026/cx46"),
    },
    # GABA_A receptor, betaAbetaAgamma stoichiometry, Zenodo record
    # 10.5281/zenodo.20705736 (Akbari Ahangar & Li 2026). Fetch with:
    #   python3 tools/fetch_benchmark_trajectories.py --records gabaa
    # This variant has no glycan segments (CARA/CARB/CARC), so
    # "protein only" is unambiguous here, unlike the record's
    # glycosylated variants.
    "gabaa_baba": {
        "name": "GABA-A receptor, betaAbetaAgamma stoichiometry (Zenodo 10.5281/zenodo.20705736)",
        "topology": os.path.join(REPO_ROOT, "benchmark_corpus/trajectories/gabaa_pore_facing/step5_assembly.hmr.psf"),
        "trajectory": os.path.join(REPO_ROOT, "benchmark_corpus/trajectories/gabaa_pore_facing/sim_1.dcd"),
        "system_selection": "protein",
        "selection": "segid PROA",
        "selection_description": "one receptor subunit (segid PROA)",
        "out_dir": os.path.join(REPO_ROOT, "benchmark_corpus/results/manuscript_2026/gabaa_baba"),
    },
}


# --------------------------------------------------------------------------
# ctypes bindings shared by the CPU-direct and prepare paths
# --------------------------------------------------------------------------

class _Topology(ctypes.Structure):
    _fields_ = [
        ("radii", ctypes.POINTER(ctypes.c_double)),
        ("residue_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_atoms", ctypes.c_int),
        ("n_residues", ctypes.c_int),
    ]


class _SoAFrames(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("y", ctypes.POINTER(ctypes.c_double)),
        ("z", ctypes.POINTER(ctypes.c_double)),
        ("n_frames", ctypes.c_int),
    ]


class _Parameters(ctypes.Structure):
    _fields_ = [
        ("probe_radius", ctypes.c_double),
        ("n_points", ctypes.c_int),
        ("algorithm", ctypes.c_int),
    ]


def _dptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _uptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))


def _load_fastsasa_lib(library_path):
    lib = ctypes.CDLL(library_path)
    lib.fastsasa_cpu_calc_trajectory_soa.argtypes = [
        ctypes.POINTER(_Topology), ctypes.POINTER(_SoAFrames), ctypes.POINTER(_Parameters),
        ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.fastsasa_cpu_calc_trajectory_soa.restype = ctypes.c_int
    lib.fastsasa_cpu_calc_trajectory_soa_selection.argtypes = [
        ctypes.POINTER(_Topology), ctypes.POINTER(_SoAFrames), ctypes.POINTER(ctypes.c_uint),
        ctypes.c_int, ctypes.POINTER(_Parameters), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.fastsasa_cpu_calc_trajectory_soa_selection.restype = ctypes.c_int
    lib.fastsasa_cpu_default_threads.restype = ctypes.c_int
    lib.fastsasa_context_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.fastsasa_context_create.restype = ctypes.c_int
    lib.fastsasa_context_free.argtypes = [ctypes.c_void_p]
    lib.fastsasa_context_backend.argtypes = [ctypes.c_void_p]
    lib.fastsasa_context_backend.restype = ctypes.c_char_p
    lib.fastsasa_context_set_precision.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.fastsasa_context_set_precision.restype = ctypes.c_int
    lib.fastsasa_context_calc_trajectory_soa_selection.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Topology), ctypes.POINTER(_SoAFrames),
        ctypes.POINTER(ctypes.c_uint), ctypes.c_int, ctypes.POINTER(_Parameters),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.fastsasa_context_calc_trajectory_soa_selection.restype = ctypes.c_int
    return lib


def _fastsasa_library_path():
    return os.path.join(REPO_ROOT, "build-cuda-vulkan/libfastsasa_native.so")


# --------------------------------------------------------------------------
# prepare: derive canonical radii + decode trajectory once
# --------------------------------------------------------------------------

def cmd_prepare(args):
    import MDAnalysis as mda
    import freesasa

    warnings.filterwarnings("ignore")
    ds = DATASETS[args.dataset]
    os.makedirs(ds["out_dir"], exist_ok=True)

    u = mda.Universe(ds["topology"], ds["trajectory"])
    atoms = u.select_atoms(ds["system_selection"]) if ds.get("system_selection") else u.atoms
    n_atoms = len(atoms)
    n_frames = len(u.trajectory)

    names = [a[:4] for a in atoms.names]
    resnames = [r[:3] for r in atoms.resnames]
    resids = [str(int(r)) for r in atoms.resids]
    segids = list(atoms.segids)

    # Canonical per-atom radii: FreeSASA's own classifier applied to the full
    # atom set. CHARMM PSF atom names (HN, HA1, CAY, ...) mostly aren't in
    # FreeSASA's name+residue table; unmatched atoms fall back to its
    # element-guess table (verified: 0/n_atoms unresolved -> radius 0 for
    # every topology used here). Atoms whose name+residue *does* match a
    # standard PDB pattern despite the CHARMM file (e.g. backbone CA/N/C/O,
    # which CHARMM also names that way) get a real ProtOr classification
    # instead of the generic per-element guess -- confirmed empirically:
    # carbon alone resolves to four different radii here, not one -- so
    # this is not a simple element->radius table, it's whatever FreeSASA's
    # full classifier actually decides per atom. That's harvested once here
    # and applied identically -- via explicit radii-array override, not
    # re-guessed -- to every backend (FreeSASA via Structure.setRadii /
    # calcCoord's radii argument, FastSASA via topology.radii) so "matched
    # radii" holds by construction rather than by hoping two independent
    # classifiers agree on CHARMM names.
    #
    # Uses addAtoms() (bulk) rather than one addAtom() call per atom: the
    # per-atom Python/C boundary crossing does not scale to systems much
    # larger than Cx46's 35k atoms (confirmed: still hadn't finished 20k of
    # 160k atoms after 3+ minutes for the GABA_A system). addAtoms() gives
    # bit-identical radii (verified against Cx46's already-established
    # values) in well under a second regardless of system size, because the
    # classification loop then runs entirely on the C side.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        opts = dict(freesasa.Structure.defaultOptions)
        opts["hetatm"] = True
        opts["hydrogen"] = True
        struct = freesasa.Structure(options=opts)
        struct.addAtoms(
            names, resnames, resids,
            [s[0] if s else "X" for s in segids],
            atoms.positions[:, 0].tolist(),
            atoms.positions[:, 1].tolist(),
            atoms.positions[:, 2].tolist(),
        )
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull)

    assert struct.nAtoms() == n_atoms
    radii = np.array([struct.radius(i) for i in range(n_atoms)], dtype=np.float64)
    n_zero_radius = int(np.sum(radii <= 0.0))

    # sel is intersected with the (possibly system_selection-filtered) atoms
    # group, then mapped from universe-wide indices to positions within the
    # filtered array via np.isin -- atoms.indices are a subset of the full
    # universe's indices in the same relative order, not a fresh 0..n range.
    sel = atoms.select_atoms(ds["selection"])
    selection_mask = np.isin(atoms.indices, sel.indices).astype(np.uint32)

    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    for i, _ts in enumerate(u.trajectory):
        coords[i] = atoms.positions

    cache_path = os.path.join(ds["out_dir"], "prepared_inputs.npz")
    np.savez(
        cache_path,
        coords=coords,
        radii=radii,
        selection_mask=selection_mask,
        n_atoms=n_atoms,
        n_frames=n_frames,
        n_selected=len(sel),
        n_zero_radius=n_zero_radius,
    )

    # Flat binaries for the FreeSASA C driver (tools/manuscript_benchmark_2026_freesasa_driver).
    np.ascontiguousarray(coords, dtype=np.float64).tofile(os.path.join(ds["out_dir"], "coords.f64"))
    np.ascontiguousarray(radii, dtype=np.float64).tofile(os.path.join(ds["out_dir"], "radii.f64"))
    np.ascontiguousarray(selection_mask, dtype=np.uint32).tofile(os.path.join(ds["out_dir"], "selection.u32"))

    print(f"prepared {n_atoms} atoms x {n_frames} frames, "
          f"selection '{ds['selection']}' = {len(sel)} atoms, "
          f"zero-radius atoms = {n_zero_radius}")
    print(f"cache: {cache_path}")


def _load_prepared(dataset):
    ds = DATASETS[dataset]
    cache_path = os.path.join(ds["out_dir"], "prepared_inputs.npz")
    data = np.load(cache_path)
    return {
        "coords": data["coords"],
        "radii": data["radii"],
        "selection_mask": data["selection_mask"],
        "n_atoms": int(data["n_atoms"]),
        "n_frames": int(data["n_frames"]),
        "n_selected": int(data["n_selected"]),
        "n_zero_radius": int(data["n_zero_radius"]),
    }


# --------------------------------------------------------------------------
# run: one (tool, threads, precision, panel) configuration
# --------------------------------------------------------------------------

_FREESASA_DRIVER = os.path.join(
    REPO_ROOT, "tools/manuscript_benchmark_2026_freesasa_driver/freesasa_calc_coord_driver")


def _run_freesasa_driver(dataset, threads, panel, repeats, out_path):
    """Invoke the patched, persistent FreeSASA C driver (see module
    docstring) directly, rather than the generic warmup+frame_loop path used
    for the other tools: the driver already times its own warmup and
    `repeats` passes internally with clock_gettime(CLOCK_MONOTONIC), so this
    just launches it once and parses its per-repeat stdout lines into rows.
    """
    ds = DATASETS[dataset]
    args = [
        _FREESASA_DRIVER,
        os.path.join(ds["out_dir"], "coords.f64"),
        os.path.join(ds["out_dir"], "radii.f64"),
        os.path.join(ds["out_dir"], "selection.u32"),
        str(_load_prepared(dataset)["n_atoms"]),
        str(_load_prepared(dataset)["n_frames"]),
        str(threads),
        panel,
        str(repeats),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    rows = []
    for line in proc.stdout.splitlines():
        fields = dict(item.split("=", 1) for item in line.split())
        n_frames = int(fields["frames"])
        elapsed = float(fields["elapsed"])
        rows.append({
            "dataset": dataset,
            "tool": "freesasa",
            "threads": threads,
            "precision": "fp64",
            "panel": panel,
            "repeat_index": int(fields["repeat"]),
            "n_frames": n_frames,
            "elapsed_seconds": elapsed,
            "frames_per_second": float(fields["frames_per_second"]),
            "total_sasa_last_repeat": float(fields["total_sum"]),
            "warmup_total_sasa": float(fields["total_sum"]),
        })

    write_header = not os.path.exists(out_path)
    import csv
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    median_fps = sorted(r["frames_per_second"] for r in rows)[len(rows) // 2]
    print(f"freesasa threads={threads} precision=fp64 panel={panel}: "
          f"median {median_fps:.3f} frames/s over {repeats} repeats "
          f"(total_sasa={rows[-1]['total_sasa_last_repeat']:.6f})")


def _run_fastsasa_cpu(coords, radii, selection_mask, threads, panel):
    lib = _load_fastsasa_lib(_fastsasa_library_path())
    n_frames, n_atoms, _ = coords.shape
    radii_c = np.ascontiguousarray(radii)
    x = np.ascontiguousarray(coords[:, :, 0])
    y = np.ascontiguousarray(coords[:, :, 1])
    z = np.ascontiguousarray(coords[:, :, 2])
    topo = _Topology(_dptr(radii_c), None, n_atoms, 0)
    frames = _SoAFrames(_dptr(x), _dptr(y), _dptr(z), n_frames)
    params = _Parameters(PROBE_RADIUS, N_POINTS, ALGORITHM_SR)
    # ctypes .ctypes.data_as(...) pointers do not keep the backing numpy
    # array alive -- without this, radii_c/x/y/z are garbage-collected once
    # this function returns (nothing inside frame_loop's body references
    # them by name, only the raw pointers baked into topo/frames), and the
    # next frame_loop() call dereferences freed memory. Tie their lifetime
    # to topo/frames, which frame_loop does hold onto.
    topo._keepalive = radii_c
    frames._keepalive = (x, y, z)

    if panel == "complete":
        def frame_loop():
            total = np.zeros(n_frames, dtype=np.float64)
            status = lib.fastsasa_cpu_calc_trajectory_soa(
                ctypes.byref(topo), ctypes.byref(frames), ctypes.byref(params),
                threads, _dptr(total), None, None,
            )
            assert status == 0, f"fastsasa_cpu_calc_trajectory_soa failed: {status}"
            return float(total.sum())
    else:
        mask_c = np.ascontiguousarray(selection_mask, dtype=np.uint32)

        def frame_loop():
            total = np.zeros(n_frames, dtype=np.float64)
            selection_out = np.zeros(n_frames, dtype=np.float64)
            status = lib.fastsasa_cpu_calc_trajectory_soa_selection(
                ctypes.byref(topo), ctypes.byref(frames), _uptr(mask_c), 1,
                ctypes.byref(params), threads, _dptr(total), _dptr(selection_out),
            )
            assert status == 0, f"fastsasa_cpu_calc_trajectory_soa_selection failed: {status}"
            return float(selection_out.sum())

    return frame_loop, n_frames


def _run_fastsasa_gpu(coords, radii, selection_mask, backend, precision, panel):
    n_frames = coords.shape[0]

    if panel == "complete":
        os.environ["FASTSASA_BACKEND"] = backend
        import fastsasa_native as fn

        engine = fn.SasaEngine(library_path=_fastsasa_library_path(), precision=precision)
        if engine.backend != backend:
            raise RuntimeError(f"requested backend {backend!r} but engine reports {engine.backend!r} "
                                f"(GPU unavailable or precision unsupported)")

        def frame_loop():
            totals = engine.sasa(coords, radii, probe_radius=PROBE_RADIUS, n_points=N_POINTS)
            return float(np.sum(totals))
        return frame_loop, n_frames, engine

    # Selected-atom panel: fastsasa_context_calc_trajectory_soa_selection()
    # (src/fastsasa_trajectory.c) skips Shrake-Rupley sampling entirely for
    # non-selected atoms -- using them only as static occlusion context --
    # but ONLY when the caller passes total_sasa=NULL (selected_center =
    # total_sasa == NULL && SR && use_selected_center_optimization()).
    # SasaEngine.sasa()'s public wrapper always requests the full-system total
    # alongside any selection, which permanently disables this path. Calling
    # the context function directly with total_sasa=NULL is the only way to
    # measure FastSASA's real selected-query throughput; confirmed bit-
    # identical selection sums to the SasaEngine.sasa() path, just faster.
    os.environ["FASTSASA_BACKEND"] = backend
    lib = _load_fastsasa_lib(_fastsasa_library_path())
    x = np.ascontiguousarray(coords[:, :, 0])
    y = np.ascontiguousarray(coords[:, :, 1])
    z = np.ascontiguousarray(coords[:, :, 2])
    radii_c = np.ascontiguousarray(radii)
    mask_c = np.ascontiguousarray(selection_mask, dtype=np.uint32)
    n_atoms = coords.shape[1]
    topo = _Topology(_dptr(radii_c), None, n_atoms, 0)
    frames = _SoAFrames(_dptr(x), _dptr(y), _dptr(z), n_frames)
    topo._keepalive = radii_c
    frames._keepalive = (x, y, z)
    params = _Parameters(PROBE_RADIUS, N_POINTS, ALGORITHM_SR)

    precision_value = {"fp64": 0, "fp32": 1}[precision]
    ctx = ctypes.c_void_p()
    status = lib.fastsasa_context_create(ctypes.byref(ctx))
    if status != 0 or lib.fastsasa_context_backend(ctx).decode() != backend:
        raise RuntimeError(f"could not create a direct {backend!r} context (status={status})")
    prec_status = lib.fastsasa_context_set_precision(ctx, precision_value)
    if prec_status != 0:
        raise RuntimeError(f"backend {backend!r} does not support precision {precision!r} (status={prec_status})")

    def frame_loop():
        selection_out = np.zeros(n_frames, dtype=np.float64)
        status = lib.fastsasa_context_calc_trajectory_soa_selection(
            ctx, ctypes.byref(topo), ctypes.byref(frames), _uptr(mask_c), 1,
            ctypes.byref(params), None, _dptr(selection_out),
        )
        assert status == 0, f"fastsasa_context_calc_trajectory_soa_selection failed: {status}"
        return float(selection_out.sum())

    return frame_loop, n_frames, _CtxCloser(lib, ctx)


class _CtxCloser:
    def __init__(self, lib, ctx):
        self._lib = lib
        self._ctx = ctx

    def close(self):
        if self._ctx:
            self._lib.fastsasa_context_free(self._ctx)
            self._ctx = None


def cmd_run(args):
    if args.tool == "freesasa":
        _run_freesasa_driver(args.dataset, args.threads, args.panel, args.repeats, args.out)
        return

    prepared = _load_prepared(args.dataset)
    coords = prepared["coords"]
    radii = prepared["radii"]
    selection_mask = prepared["selection_mask"]

    engine_to_close = None
    if args.tool == "fastsasa-cpu":
        frame_loop, n_frames = _run_fastsasa_cpu(coords, radii, selection_mask, args.threads, args.panel)
    elif args.tool in ("fastsasa-cuda", "fastsasa-vulkan"):
        backend = "cuda" if args.tool == "fastsasa-cuda" else "vulkan"
        frame_loop, n_frames, engine_to_close = _run_fastsasa_gpu(
            coords, radii, selection_mask, backend, args.precision, args.panel)
    else:
        raise ValueError(f"unknown tool {args.tool!r}")

    try:
        warmup_total = frame_loop()
        timings = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            total = frame_loop()
            t1 = time.perf_counter()
            timings.append((t1 - t0, total))
    finally:
        if engine_to_close is not None:
            engine_to_close.close()

    rows = []
    for repeat_index, (elapsed, total) in enumerate(timings):
        rows.append({
            "dataset": args.dataset,
            "tool": args.tool,
            "threads": args.threads,
            "precision": args.precision,
            "panel": args.panel,
            "repeat_index": repeat_index,
            "n_frames": n_frames,
            "elapsed_seconds": elapsed,
            "frames_per_second": n_frames / elapsed,
            "total_sasa_last_repeat": total,
            "warmup_total_sasa": warmup_total,
        })

    out_path = args.out
    write_header = not os.path.exists(out_path)
    import csv
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    medians = sorted(r["frames_per_second"] for r in rows)[len(rows) // 2]
    print(f"{args.tool} threads={args.threads} precision={args.precision} panel={args.panel}: "
          f"median {medians:.3f} frames/s over {args.repeats} repeats "
          f"(total_sasa={rows[-1]['total_sasa_last_repeat']:.6f})")


# --------------------------------------------------------------------------
# orchestrate: fresh subprocess per configuration
# --------------------------------------------------------------------------

def _configurations():
    configs = []
    for panel in ("complete", "selected"):
        configs.append(dict(tool="freesasa", threads=N_THREADS_LOW, precision="fp64", panel=panel))
        configs.append(dict(tool="freesasa", threads=N_THREADS_HIGH, precision="fp64", panel=panel))
        configs.append(dict(tool="fastsasa-cpu", threads=N_THREADS_LOW, precision="fp64", panel=panel))
        configs.append(dict(tool="fastsasa-cpu", threads=N_THREADS_HIGH, precision="fp64", panel=panel))
        configs.append(dict(tool="fastsasa-cuda", threads=0, precision="fp64", panel=panel))
        configs.append(dict(tool="fastsasa-vulkan", threads=0, precision="fp64", panel=panel))
    return configs


def cmd_orchestrate(args):
    ds = DATASETS[args.dataset]
    out_csv = os.path.join(ds["out_dir"], "figure1_raw_timings.csv")
    if args.fresh and os.path.exists(out_csv):
        os.remove(out_csv)

    for cfg in _configurations():
        cmd = [
            sys.executable, os.path.abspath(__file__), "run",
            "--dataset", args.dataset,
            "--tool", cfg["tool"],
            "--threads", str(cfg["threads"]),
            "--precision", cfg["precision"],
            "--panel", cfg["panel"],
            "--repeats", str(REPEATS),
            "--out", out_csv,
        ]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    print(f"\nall configurations complete: {out_csv}")


# --------------------------------------------------------------------------
# numeric: cross-backend bit-identity + FP32 vs FP64 checks
# --------------------------------------------------------------------------

def cmd_numeric(args):
    prepared = _load_prepared(args.dataset)
    coords = prepared["coords"]
    radii = prepared["radii"]
    selection_mask = prepared["selection_mask"]

    results = {}
    for backend in ("cpu", "cuda", "vulkan"):
        for precision in ("fp64", "fp32"):
            if backend == "cpu" and precision == "fp32":
                continue  # CPU backend is FP64-only by construction
            os.environ["FASTSASA_BACKEND"] = backend
            import importlib
            import fastsasa_native as fn
            importlib.reload(fn)
            engine = fn.SasaEngine(library_path=_fastsasa_library_path(), precision=precision)
            actual_backend = engine.backend
            totals = engine.sasa(coords, radii, probe_radius=PROBE_RADIUS, n_points=N_POINTS)
            engine.close()
            key = f"{backend}_{precision}"
            results[key] = {
                "requested_backend": backend,
                "actual_backend": actual_backend,
                "precision": precision,
                "totals": np.asarray(totals),
            }

    ref = results["cpu_fp64"]["totals"]
    report = {"dataset": args.dataset, "n_frames": len(ref), "checks": []}

    for key in ("cuda_fp64", "vulkan_fp64"):
        entry = results.get(key)
        if entry is None or entry["actual_backend"] != entry["requested_backend"]:
            report["checks"].append({"comparison": f"{key}_vs_cpu_fp64", "status": "unavailable",
                                      "note": "GPU backend not available on this machine"})
            continue
        max_abs_diff = float(np.max(np.abs(entry["totals"] - ref)))
        bit_identical = bool(np.array_equal(entry["totals"], ref))
        report["checks"].append({
            "comparison": f"{key}_vs_cpu_fp64",
            "bit_identical": bit_identical,
            "max_abs_diff": max_abs_diff,
        })

    for key in ("cuda_fp32", "vulkan_fp32"):
        entry = results.get(key)
        if entry is None or entry["actual_backend"] != entry["requested_backend"]:
            report["checks"].append({"comparison": f"{key}_vs_cpu_fp64", "status": "unavailable",
                                      "note": "GPU backend/precision not available on this machine"})
            continue
        rel_diff = np.abs(entry["totals"] - ref) / np.maximum(np.abs(ref), 1e-12)
        report["checks"].append({
            "comparison": f"{key}_vs_cpu_fp64",
            "max_relative_diff": float(np.max(rel_diff)),
            "mean_relative_diff": float(np.mean(rel_diff)),
        })

    # Selected-atom panel uses a different code path (the active-center
    # optimization in fastsasa_context_calc_trajectory_soa_selection, see
    # _run_fastsasa_gpu) than the complete-system panel above, so its
    # numerical agreement isn't guaranteed by the complete-panel checks and
    # is verified separately here.
    lib = _load_fastsasa_lib(_fastsasa_library_path())
    n_frames, n_atoms, _ = coords.shape
    x = np.ascontiguousarray(coords[:, :, 0])
    y = np.ascontiguousarray(coords[:, :, 1])
    z = np.ascontiguousarray(coords[:, :, 2])
    radii_c = np.ascontiguousarray(radii)
    mask_c = np.ascontiguousarray(selection_mask, dtype=np.uint32)
    topo = _Topology(_dptr(radii_c), None, n_atoms, 0)
    frames = _SoAFrames(_dptr(x), _dptr(y), _dptr(z), n_frames)
    topo._keepalive = radii_c
    frames._keepalive = (x, y, z)
    params = _Parameters(PROBE_RADIUS, N_POINTS, ALGORITHM_SR)

    selection_results = {}
    cpu_total = np.zeros(n_frames, dtype=np.float64)
    cpu_selection = np.zeros(n_frames, dtype=np.float64)
    status = lib.fastsasa_cpu_calc_trajectory_soa_selection(
        ctypes.byref(topo), ctypes.byref(frames), _uptr(mask_c), 1,
        ctypes.byref(params), N_THREADS_HIGH, _dptr(cpu_total), _dptr(cpu_selection),
    )
    assert status == 0
    selection_results["cpu_fp64"] = cpu_selection.copy()

    for backend in ("cuda", "vulkan"):
        for precision in ("fp64", "fp32"):
            os.environ["FASTSASA_BACKEND"] = backend
            ctx = ctypes.c_void_p()
            status = lib.fastsasa_context_create(ctypes.byref(ctx))
            if status != 0 or lib.fastsasa_context_backend(ctx).decode() != backend:
                selection_results[f"{backend}_{precision}"] = None
                continue
            prec_status = lib.fastsasa_context_set_precision(ctx, {"fp64": 0, "fp32": 1}[precision])
            if prec_status != 0:
                lib.fastsasa_context_free(ctx)
                selection_results[f"{backend}_{precision}"] = None
                continue
            selection_out = np.zeros(n_frames, dtype=np.float64)
            status = lib.fastsasa_context_calc_trajectory_soa_selection(
                ctx, ctypes.byref(topo), ctypes.byref(frames), _uptr(mask_c), 1,
                ctypes.byref(params), None, _dptr(selection_out),
            )
            lib.fastsasa_context_free(ctx)
            assert status == 0
            selection_results[f"{backend}_{precision}"] = selection_out.copy()

    sel_ref = selection_results["cpu_fp64"]
    for key in ("cuda_fp64", "vulkan_fp64"):
        entry = selection_results.get(key)
        if entry is None:
            report["checks"].append({"comparison": f"selected_{key}_vs_cpu_fp64", "status": "unavailable"})
            continue
        report["checks"].append({
            "comparison": f"selected_{key}_vs_cpu_fp64",
            "bit_identical": bool(np.array_equal(entry, sel_ref)),
            "max_abs_diff": float(np.max(np.abs(entry - sel_ref))),
        })
    for key in ("cuda_fp32", "vulkan_fp32"):
        entry = selection_results.get(key)
        if entry is None:
            report["checks"].append({"comparison": f"selected_{key}_vs_cpu_fp64", "status": "unavailable"})
            continue
        rel_diff = np.abs(entry - sel_ref) / np.maximum(np.abs(sel_ref), 1e-12)
        report["checks"].append({
            "comparison": f"selected_{key}_vs_cpu_fp64",
            "max_relative_diff": float(np.max(rel_diff)),
            "mean_relative_diff": float(np.mean(rel_diff)),
        })

    ds = DATASETS[args.dataset]
    out_path = os.path.join(ds["out_dir"], "numerical_checks.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--dataset", default="cx46", choices=DATASETS.keys())
    p_prepare.set_defaults(func=cmd_prepare)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", default="cx46", choices=DATASETS.keys())
    p_run.add_argument("--tool", required=True,
                        choices=["freesasa", "fastsasa-cpu", "fastsasa-cuda", "fastsasa-vulkan"])
    p_run.add_argument("--threads", type=int, default=0)
    p_run.add_argument("--precision", default="fp64", choices=["fp64", "fp32"])
    p_run.add_argument("--panel", required=True, choices=["complete", "selected"])
    p_run.add_argument("--repeats", type=int, default=REPEATS)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=cmd_run)

    p_orch = sub.add_parser("orchestrate")
    p_orch.add_argument("--dataset", default="cx46", choices=DATASETS.keys())
    p_orch.add_argument("--fresh", action="store_true")
    p_orch.set_defaults(func=cmd_orchestrate)

    p_num = sub.add_parser("numeric")
    p_num.add_argument("--dataset", default="cx46", choices=DATASETS.keys())
    p_num.set_defaults(func=cmd_numeric)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
