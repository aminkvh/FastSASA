#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np


def _cli_rsa_reference(fastsasa: Path, pdb: Path) -> dict[str, dict[str, float]]:
    """Parse `--format rsa` output into {residue_index: {total, side_chain,
    main_chain, rel_total}}, keyed by encounter order (0-based)."""

    proc = subprocess.run(
        [str(fastsasa), "--format", "rsa", str(pdb)],
        text=True, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"--format rsa failed:\n{proc.stderr}")
    rows = {}
    index = 0
    for line in proc.stdout.splitlines():
        if not line.startswith("RES "):
            continue
        fields = line.split()
        # RES <resn> <chain+resnum> total rel_total side rel_side main rel_main apolar rel_apolar polar rel_polar
        total, rel_total, side, _rs, main, _rm = (float(fields[i]) if fields[i] != "N/A" else float("nan")
                                                   for i in (3, 4, 5, 6, 7, 8))
        rows[index] = {"total": total, "rel_total": rel_total, "side_chain": side, "main_chain": main}
        index += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--python-dir", required=True, type=Path)
    parser.add_argument("--fastsasa", type=Path, default=None,
                        help="optional: cross-check per_residue_rsa against `--format rsa`")
    args = parser.parse_args()

    sys.path.insert(0, str(args.python_dir))
    os.environ["FASTSASA_NATIVE_LIBRARY"] = str(args.library)

    from fastsasa import SasaEngine, extract_md_features, flatten_statistics, interface_sasa, summarize_time_series

    stats = summarize_time_series(np.asarray([0.0, 2.0, 0.0]))
    if int(stats["transitions"][0]) != 2:
        raise SystemExit("transition count smoke check failed")

    xyz = np.asarray(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 4.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 4.2, 0.0], [4.0, 4.0, 0.0]],
        ],
        dtype=np.float64,
    )
    radii = np.asarray([1.7, 1.7, 1.55, 1.52], dtype=np.float64)
    residue_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)

    try:
        features = extract_md_features(
            xyz,
            radii,
            residue_ids=residue_ids,
            group_masks={"ligand_sasa": np.asarray([0, 1, 0, 0], dtype=bool)},
            hydrophobic_mask=np.asarray([1, 1, 0, 0], dtype=bool),
            polar_mask=np.asarray([0, 0, 1, 1], dtype=bool),
            interface_a_mask=np.asarray([1, 1, 0, 0], dtype=bool),
            interface_b_mask=np.asarray([0, 0, 1, 1], dtype=bool),
            n_points=32,
        )
    except RuntimeError as exc:
        # A device that lacks shaderFloat64 entirely (MoltenVK/Metal on
        # Apple GPUs, notably - a permanent hardware limit, not a transient
        # "no GPU" case) is a distinct, acceptable skip reason even under
        # FASTSASA_REQUIRE_GPU_TESTS=1: extract_md_features() has no
        # precision= parameter to retry at FP32 with, so there is nothing
        # else this test can validate here.
        if "does not support shaderFloat64" in str(exc):
            print(f"fastsasa_python_feature_validation,status,skip,reason,fp64_unsupported,detail,{exc}")
            return 0
        if os.environ.get("FASTSASA_REQUIRE_GPU_TESTS") == "1":
            raise
        print(f"fastsasa_python_feature_validation,status,skip,reason,gpu_unavailable,detail,{exc}")
        return 0

    names, values = flatten_statistics(features["statistics"])
    required = {
        "total_sasa",
        "per_residue_sasa",
        "ligand_sasa",
        "hydrophobic_exposed_area",
        "polar_exposed_area",
        "interface_a_free_sasa",
        "interface_b_free_sasa",
        "interface_a_bound_sasa",
        "interface_b_bound_sasa",
        "interface_complex_sasa",
        "interface_a_buried_sasa",
        "interface_b_buried_sasa",
        "interface_buried_sasa",
        "interface_buried_sasa_half",
        "interface_a_buried_fraction",
        "interface_b_buried_fraction",
        "per_residue_buried_sasa",
        "per_residue_buried_fraction",
        "interface_polar_buried_sasa",
        "interface_apolar_buried_sasa",
        "interface_apolar_fraction",
    }
    if not required.issubset(features["time_series"]):
        raise SystemExit("missing expected feature time series")
    if features["interface_residue_mask"] is None or not np.all(features["interface_residue_mask"]):
        raise SystemExit("interface_residue_mask should mark both fixture residues (full interface coverage)")
    series = features["time_series"]
    buried_identity = (
        series["interface_a_free_sasa"]
        + series["interface_b_free_sasa"]
        - series["interface_complex_sasa"]
    )
    if not np.allclose(buried_identity, series["interface_buried_sasa"], rtol=1.0e-6, atol=1.0e-6):
        raise SystemExit("interface buried SASA identity failed")
    if not np.allclose(0.5 * series["interface_buried_sasa"], series["interface_buried_sasa_half"], rtol=1.0e-12, atol=1.0e-12):
        raise SystemExit("half buried surface area convention mismatch")
    expected_a_fraction = series["interface_a_buried_sasa"] / series["interface_a_free_sasa"]
    if not np.allclose(expected_a_fraction, series["interface_a_buried_fraction"], rtol=1.0e-9, atol=1.0e-9):
        raise SystemExit("interface_a_buried_fraction does not match buried/free")
    expected_b_fraction = series["interface_b_buried_sasa"] / series["interface_b_free_sasa"]
    if not np.allclose(expected_b_fraction, series["interface_b_buried_fraction"], rtol=1.0e-9, atol=1.0e-9):
        raise SystemExit("interface_b_buried_fraction does not match buried/free")
    per_residue_buried_sum = np.nansum(series["per_residue_buried_sasa"], axis=1)
    if not np.allclose(per_residue_buried_sum, series["interface_buried_sasa"], rtol=1.0e-6, atol=1.0e-6):
        raise SystemExit("per_residue_buried_sasa does not sum to interface_buried_sasa")
    classified_buried = series["interface_polar_buried_sasa"] + series["interface_apolar_buried_sasa"]
    if not np.allclose(classified_buried, series["interface_buried_sasa"], rtol=1.0e-6, atol=1.0e-6):
        raise SystemExit("interface_polar_buried_sasa + interface_apolar_buried_sasa does not equal interface_buried_sasa")
    expected_apolar_fraction = series["interface_apolar_buried_sasa"] / classified_buried
    if not np.allclose(expected_apolar_fraction, series["interface_apolar_fraction"], rtol=1.0e-9, atol=1.0e-9):
        raise SystemExit("interface_apolar_fraction does not match apolar/classified")
    if "metadata" not in features:
        raise SystemExit("extract_md_features did not return a metadata dict")
    if features["metadata"]["units"] != "angstrom^2" or features["metadata"]["algorithm"] != "shrake_rupley":
        raise SystemExit("extract_md_features metadata is missing expected fields")
    direct_interface = interface_sasa(
        xyz,
        radii,
        np.asarray([1, 1, 0, 0], dtype=bool),
        np.asarray([0, 0, 1, 1], dtype=bool),
        n_points=32,
    )
    if not np.allclose(direct_interface["interface_buried_sasa"], series["interface_buried_sasa"], rtol=1.0e-6, atol=1.0e-6):
        raise SystemExit("direct interface_sasa output mismatch")
    if len(names) == 0 or values.ndim != 1 or not np.all(np.isfinite(values)):
        raise SystemExit("feature statistic flattening failed")

    with SasaEngine() as engine:
        batch_atoms = engine.sasa(xyz, radii, n_points=32, atom_sasa=True)
        loop_atoms = [engine.sasa(xyz[frame], radii, n_points=32, atom_sasa=True)
                      for frame in range(xyz.shape[0])]
        if not np.allclose(batch_atoms[0], np.concatenate([value[0] for value in loop_atoms]),
                           rtol=1.0e-7, atol=1.0e-7):
            raise SystemExit("batched Python SR total parity failed")
        if not np.allclose(batch_atoms[1], np.concatenate([value[1] for value in loop_atoms]),
                           rtol=1.0e-7, atol=1.0e-7):
            raise SystemExit("batched Python SR atom parity failed")

        masks = np.asarray([1, 0, 1, 0], dtype=np.uint32)
        batch_selection = engine.sasa(xyz, radii, n_points=32,
                                      selection_masks=masks)
        loop_selection = [engine.sasa(xyz[frame], radii, n_points=32,
                                      selection_masks=masks)
                          for frame in range(xyz.shape[0])]
        if not np.allclose(batch_selection["total"],
                           np.concatenate([value["total"] for value in loop_selection]),
                           rtol=1.0e-7, atol=1.0e-7):
            raise SystemExit("batched Python SR selection total parity failed")
        if not np.allclose(batch_selection["selection"],
                           np.concatenate([value["selection"] for value in loop_selection]),
                           rtol=1.0e-7, atol=1.0e-7):
            raise SystemExit("batched Python SR selection parity failed")

        batch_lr = engine.lee_richards(xyz, radii, n_slices=8)
        loop_lr = np.concatenate([
            engine.lee_richards(xyz[frame], radii, n_slices=8)
            for frame in range(xyz.shape[0])
        ])
        if not np.allclose(batch_lr, loop_lr, rtol=1.0e-7, atol=1.0e-7):
            raise SystemExit("batched Python Lee-Richards parity failed")

        baseline = engine.sasa(xyz[:1], radii, n_points=32)
        translated = engine.sasa(xyz[:1] + 1.0e9, radii, n_points=32)
        if not np.allclose(baseline, translated, rtol=1.0e-6, atol=1.0e-6):
            raise SystemExit("translated FP32 coordinate validation failed")
        lr_baseline = engine.lee_richards(xyz[:1], radii, n_slices=8)
        lr_translated = engine.lee_richards(xyz[:1] + 1.0e9, radii, n_slices=8)
        if not np.allclose(lr_baseline, lr_translated, rtol=1.0e-6, atol=1.0e-6):
            raise SystemExit("translated FP32 Lee-Richards coordinate validation failed")

        empty = engine.sasa(
            np.empty((1, 0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            n_points=32,
            atom_sasa=True,
            selection_masks=np.empty(0, dtype=np.uint32),
        )
        if empty["total"].shape != (1,) or empty["atom"].shape != (1, 0):
            raise SystemExit("empty SASA output shape validation failed")
        if empty["selection"].shape != (1, 1) or np.any(empty["total"] != 0.0):
            raise SystemExit("empty SASA output value validation failed")
        empty_lr = engine.lee_richards(
            np.empty((1, 0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            n_slices=8,
            atom_sasa=True,
        )
        if empty_lr[0].shape != (1,) or empty_lr[1].shape != (1, 0) or np.any(empty_lr[0] != 0.0):
            raise SystemExit("empty Lee-Richards SASA validation failed")

        zero_selection = engine.sasa(
            xyz[:1],
            radii,
            n_points=32,
            selection_masks=np.zeros(xyz.shape[1], dtype=np.uint32),
        )
        if zero_selection["selection"].shape != (1, 1) or np.any(zero_selection["selection"] != 0.0):
            raise SystemExit("empty selection SASA validation failed")

        bad_xyz = xyz[:1].copy()
        bad_xyz[0, 0, 0] = np.nan
        try:
            engine.sasa(bad_xyz, radii, n_points=32)
        except ValueError:
            pass
        else:
            raise SystemExit("non-finite Python coordinate was not rejected")

        sparse_xyz = xyz[:1].copy()
        sparse_xyz[0, 1, 0] = 1.0e9
        if engine.backend == "cpu":
            # The threaded CPU backend handles sparse coordinates without the
            # dense GPU cell grid, so the calculation must simply succeed.
            sparse_total = engine.sasa(sparse_xyz, radii, n_points=32)
            if not np.all(np.isfinite(sparse_total)):
                raise SystemExit("CPU sparse-coordinate SASA was not finite")
        else:
            try:
                engine.sasa(sparse_xyz, radii, n_points=32)
            except RuntimeError as exc:
                message = str(exc)
                if ("unsupported dense cell grid" not in message and
                        "cell grid is too large" not in message):
                    raise
            else:
                raise SystemExit("oversized Python cell grid was not rejected")

    with SasaEngine(precision="fp32") as fp32_engine:
        # The threaded CPU tier supports fp32 too (Shrake-Rupley only); every
        # backend reports the precision the caller asked for.
        if fp32_engine.precision != "fp32":
            raise SystemExit("Python FP32 precision selection failed")
        fp32_total = fp32_engine.sasa(xyz[:1], radii, n_points=32)
        if not np.allclose(fp32_total, baseline, rtol=1.0e-4, atol=1.0e-4):
            raise SystemExit("Python FP32 result exceeds documented tolerance")

    try:
        SasaEngine(precision="invalid")
    except ValueError:
        pass
    else:
        raise SystemExit("invalid Python precision was not rejected")

    if os.environ.get("FASTSASA_BACKEND") == "vulkan":
        # A Vulkan device without shaderFloat64 must fail loudly when Vulkan
        # is requested explicitly and fall back to CPU in auto mode.
        os.environ["FASTSASA_TEST_VULKAN_NO_FP64"] = "1"
        try:
            try:
                SasaEngine()
            except RuntimeError as exc:
                if "shaderFloat64" not in str(exc):
                    raise
            else:
                raise SystemExit(
                    "explicit Vulkan engine without FP64 did not report shaderFloat64")
            os.environ["FASTSASA_BACKEND"] = "auto"
            os.environ["FASTSASA_TEST_DISABLE_CUDA"] = "1"
            with SasaEngine() as fallback_engine:
                if fallback_engine.backend != "cpu":
                    raise SystemExit(
                        "auto engine without FP64 Vulkan did not fall back to CPU")
        finally:
            os.environ.pop("FASTSASA_TEST_VULKAN_NO_FP64", None)
            os.environ.pop("FASTSASA_TEST_DISABLE_CUDA", None)
            os.environ["FASTSASA_BACKEND"] = "vulkan"

    saved_backend = os.environ.get("FASTSASA_BACKEND")
    os.environ["FASTSASA_BACKEND"] = "cpu"
    try:
        with SasaEngine() as cpu_engine:
            if cpu_engine.backend != "cpu":
                raise SystemExit("FASTSASA_BACKEND=cpu did not select the CPU tier")
            cpu_total = cpu_engine.sasa(xyz[:1], radii, n_points=32)
            if not np.allclose(cpu_total, baseline, rtol=1.0e-9, atol=1.0e-9):
                raise SystemExit("Python CPU tier diverges from accelerator result")
            probe_fp64 = cpu_engine.sasa(np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]),
                                         np.array([1.0, 0.5 + 1.0e-9]), probe_radius=0.0, n_points=1)
            if not np.isclose(probe_fp64[0], np.pi, rtol=1.0e-6, atol=0.0):
                raise SystemExit(f"CPU FP64 probe fixture total {probe_fp64[0]}, expected pi")
            cpu_lr = cpu_engine.lee_richards(xyz[:1], radii, n_slices=16)
            if not np.all(np.isfinite(cpu_lr)) or np.any(cpu_lr <= 0.0):
                raise SystemExit("Python CPU Lee-Richards smoke check failed")
        with SasaEngine(precision="fp32") as cpu_fp32_engine:
            if cpu_fp32_engine.backend != "cpu" or cpu_fp32_engine.precision != "fp32":
                raise SystemExit("FASTSASA_BACKEND=cpu did not honor precision='fp32'")
            cpu_fp32_total = cpu_fp32_engine.sasa(xyz[:1], radii, n_points=32)
            if not np.allclose(cpu_fp32_total, baseline, rtol=1.0e-4, atol=1.0e-4):
                raise SystemExit("CPU FP32 result exceeds documented tolerance")
            # Float-boundary probe (see tests/cli_validation.py): A (radius
            # 1.0) at the origin, B (radius 0.5 + 1e-9) at x = 1.5, probe 0,
            # one test point at +x. A's point is buried in double but exposed
            # in float, so the totals are exactly pi (fp64) and 5*pi (fp32).
            probe_xyz = np.array([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
            probe_radii = np.array([1.0, 0.5 + 1.0e-9])
            probe_fp32 = cpu_fp32_engine.sasa(probe_xyz, probe_radii, probe_radius=0.0, n_points=1)
            if not np.isclose(probe_fp32[0], 5.0 * np.pi, rtol=1.0e-6, atol=0.0):
                raise SystemExit(f"CPU FP32 did not run the float kernel: probe total {probe_fp32[0]}, expected 5*pi")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cpu_fp32_engine.lee_richards(xyz[:1], radii, n_slices=16)
            if not any("fp64-only" in str(w.message) for w in caught):
                raise SystemExit("CPU FP32 Lee-Richards did not warn that it fell back to fp64")
    finally:
        if saved_backend is None:
            os.environ.pop("FASTSASA_BACKEND", None)
        else:
            os.environ["FASTSASA_BACKEND"] = saved_backend

    if args.fastsasa is not None:
        from fastsasa_adapters import load_radius_config

        pdb = Path(__file__).resolve().parent / "data" / "1ubq.pdb"
        xs, ys, zs, resn, resi, names, chains = [], [], [], [], [], [], []
        for line in pdb.read_text().splitlines():
            if line.startswith("ATOM"):
                xs.append(float(line[30:38])); ys.append(float(line[38:46])); zs.append(float(line[46:54]))
                names.append(line[12:16].strip())
                resn.append(line[17:20].strip())
                resi.append(int(line[22:26]))
                chains.append(line[21])
        n_atoms_rsa = len(xs)
        rsa_positions = np.array([xs, ys, zs]).T
        radius_config = load_radius_config()
        rsa_radii = np.array([
            radius_config.radius(resn[i], names[i], element=None, default=1.7) for i in range(n_atoms_rsa)
        ])
        rsa_residue_ids = np.empty(n_atoms_rsa, dtype=np.int32)
        rsa_residue_names = []
        key_to_id: dict[tuple[str, int], int] = {}
        for i in range(n_atoms_rsa):
            key = (chains[i], resi[i])
            if key not in key_to_id:
                key_to_id[key] = len(key_to_id)
                rsa_residue_names.append(resn[i])
            rsa_residue_ids[i] = key_to_id[key]
        rsa_features = extract_md_features(
            rsa_positions, rsa_radii,
            residue_ids=rsa_residue_ids, n_residues=len(key_to_id),
            atom_names=names, residue_names=rsa_residue_names,
            probe_radius=1.4, n_points=100,
        )
        py_total = rsa_features["time_series"]["per_residue_sasa"][0]
        py_rsa = rsa_features["time_series"]["per_residue_rsa"][0]
        py_bb = rsa_features["time_series"]["per_residue_backbone_sasa"][0]
        py_sc = rsa_features["time_series"]["per_residue_sidechain_sasa"][0]
        cli_rows = _cli_rsa_reference(args.fastsasa, pdb)
        if len(cli_rows) != len(key_to_id):
            raise SystemExit(f"RSA cross-check residue count mismatch: CLI {len(cli_rows)} vs Python {len(key_to_id)}")
        for index, row in cli_rows.items():
            if not np.isclose(py_total[index], row["total"], atol=1.0e-2):
                raise SystemExit(f"residue {index}: total SASA mismatch {py_total[index]} vs CLI {row['total']}")
            if not np.isclose(py_rsa[index] * 100.0, row["rel_total"], atol=0.15):
                raise SystemExit(f"residue {index}: RSA mismatch {py_rsa[index] * 100.0} vs CLI {row['rel_total']}")
            if not np.isclose(py_bb[index], row["main_chain"], atol=1.0e-2):
                raise SystemExit(f"residue {index}: backbone SASA mismatch {py_bb[index]} vs CLI main_chain {row['main_chain']}")
            if not np.isclose(py_sc[index], row["side_chain"], atol=1.0e-2):
                raise SystemExit(f"residue {index}: sidechain SASA mismatch {py_sc[index]} vs CLI side_chain {row['side_chain']}")

    print("fastsasa_python_feature_validation,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
