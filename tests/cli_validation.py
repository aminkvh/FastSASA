#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


FP32_PROBE_CONFIG = """name: fp32probe

types:
BIG 1.0 apolar
EDGE 0.5000000001 apolar

atoms:
PRB A1 BIG
PRB A2 EDGE
"""

FP32_PROBE_PDB = (
    "ATOM      1  A1  PRB A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
    "ATOM      2  A2  PRB A   1       1.500   0.000   0.000  1.00  0.00           C  \n"
    "END\n"
)


def run_json(fastsasa: Path, data_dir: Path, *args: str) -> dict:
    cmd = [str(fastsasa), *args]
    proc = subprocess.run(cmd, cwd=data_dir.parent.parent, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise SystemExit(f"invalid JSON from {' '.join(cmd)}: {exc}") from exc


def try_json(fastsasa: Path, data_dir: Path, *args: str) -> tuple[dict | None, str]:
    cmd = [str(fastsasa), *args]
    proc = subprocess.run(cmd, cwd=data_dir.parent.parent, text=True, capture_output=True)
    if proc.returncode != 0:
        return None, proc.stderr + proc.stdout
    try:
        return json.loads(proc.stdout), ""
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from {' '.join(cmd)}: {exc}\n{proc.stderr}\n{proc.stdout}"


def run_text(fastsasa: Path, data_dir: Path, *args: str) -> str:
    cmd = [str(fastsasa), *args]
    proc = subprocess.run(cmd, cwd=data_dir.parent.parent, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.stderr.write(proc.stdout)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return proc.stdout


def assert_close(name: str, left: float, right: float, tolerance: float) -> None:
    diff = abs(left - right)
    if diff > tolerance:
        raise SystemExit(f"{name}: {left} vs {right}, diff {diff} > {tolerance}")


def total(result: dict) -> float:
    return float(result["total_sasa"])


def assert_atoms_close(name: str, left: dict, right: dict,
                       tolerance: float) -> None:
    left_atoms = left.get("atoms", [])
    right_atoms = right.get("atoms", [])
    if len(left_atoms) != len(right_atoms):
        raise SystemExit(f"{name}: atom count mismatch")
    for index, (left_atom, right_atom) in enumerate(zip(left_atoms, right_atoms)):
        assert_close(f"{name} atom {index}", float(left_atom["sasa"]),
                     float(right_atom["sasa"]), tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()

    pdb = "tests/data/1ubq.pdb"
    cif = "tests/data/2isk.cif"
    cif_peer = "tests/data/2isk.pdb"

    help_proc = subprocess.run([str(args.fastsasa), "--help"], cwd=args.data_dir.parent.parent, text=True, capture_output=True)
    if help_proc.returncode != 0 or "usage:" not in help_proc.stdout:
        raise SystemExit("CLI --help validation failed")
    version_proc = subprocess.run([str(args.fastsasa), "--version"], cwd=args.data_dir.parent.parent, text=True, capture_output=True)
    if version_proc.returncode != 0 or not version_proc.stdout.startswith("FastSASA "):
        raise SystemExit("CLI --version validation failed")
    invalid_resolution = subprocess.run(
        [str(args.fastsasa), "--cpu", "--resolution", "8x", "--format", "json", pdb],
        cwd=args.data_dir.parent.parent,
        text=True,
        capture_output=True,
    )
    if invalid_resolution.returncode == 0 or "invalid --resolution value" not in invalid_resolution.stderr:
        raise SystemExit("strict --resolution validation failed")

    for removed_flag in ("--lee-richards-cpu", "--compat-output", "--cif-namespace"):
        removed_proc = subprocess.run(
            [str(args.fastsasa), removed_flag, pdb],
            cwd=args.data_dir.parent.parent,
            text=True,
            capture_output=True,
        )
        if removed_proc.returncode == 0:
            raise SystemExit(f"removed option was still accepted: {removed_flag}")

    default_rsa = run_text(args.fastsasa, args.data_dir, "--format", "rsa", pdb)
    if "REM  Atomic radii and reference values for relative SASA: ProtOr" not in default_rsa:
        raise SystemExit("--format rsa with the bundled default config did not report ProtOr references")
    if re.search(r"^RES \S+ +\S+ +[\d.]+ +N/A", default_rsa, re.MULTILINE):
        raise SystemExit("--format rsa with the bundled default config reported N/A relative SASA")

    # References follow the table's declared name, not how it was loaded:
    # the bundled ProtOr file passed explicitly still reports references,
    # the glycan extension (ProtOr-Glycans) and any custom table do not.
    explicit_rsa = run_text(
        args.fastsasa, args.data_dir,
        "--format", "rsa", "--config-file", "share/protor.config", pdb,
    )
    if "REM  Atomic radii and reference values for relative SASA: ProtOr" not in explicit_rsa:
        raise SystemExit("--format rsa with --config-file share/protor.config did not report ProtOr references")
    if explicit_rsa != default_rsa:
        raise SystemExit("--format rsa differs between the auto-discovered and the explicit bundled config")
    glycan_rsa = run_text(
        args.fastsasa, args.data_dir,
        "--format", "rsa", "--config-file", "share/protor_glycans.config", pdb,
    )
    if "REM  No reference values available for custom radii" not in glycan_rsa:
        raise SystemExit("--format rsa with the glycan extension did not report N/A references")
    if not re.search(r"^RES \S+ +\S+ +[\d.]+ +N/A", glycan_rsa, re.MULTILINE):
        raise SystemExit("--format rsa with the glycan extension printed relative values")

    # MD residue variants (HIE, CYX, ...) use the standard residue's radii,
    # classes, and RSA references: renaming every HIS to HIE must change
    # nothing except the printed residue name, with no element-guess warning.
    with tempfile.TemporaryDirectory() as alias_dir:
        alias_pdb = Path(alias_dir) / "1ubq_hie.pdb"
        alias_pdb.write_text((args.data_dir / "1ubq.pdb").read_text().replace(" HIS ", " HIE "))
        alias_json = run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--format", "json", str(alias_pdb))
        alias_proc = subprocess.run(
            [str(args.fastsasa), "--backend", "cpu", "--format", "rsa", str(alias_pdb)],
            cwd=args.data_dir.parent.parent, text=True, capture_output=True,
        )
    if total(alias_json) != total(cpu_fp64_reference := run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--format", "json", pdb)):
        raise SystemExit(f"renaming HIS to HIE changed the total SASA: {total(alias_json)} vs {total(cpu_fp64_reference)}")
    if "guessing element" in alias_proc.stderr:
        raise SystemExit(f"HIE atoms fell back to element radii:\n{alias_proc.stderr}")
    if alias_proc.returncode != 0:
        raise SystemExit(f"--format rsa on the HIE structure failed:\n{alias_proc.stderr}")
    if "RES HIE" not in alias_proc.stdout or re.search(r"^RES HIE .*N/A", alias_proc.stdout, re.MULTILINE):
        raise SystemExit("--format rsa printed N/A for HIE residues")
    if default_rsa.replace("HIS", "HIE") != alias_proc.stdout.replace(str(alias_pdb), "tests/data/1ubq.pdb"):
        raise SystemExit("--format rsa for the HIE structure differs from the HIS structure beyond the residue name")

    cpu_fp64 = run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--precision", "fp64", "--format", "json", pdb)
    cpu_fp32 = run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--precision", "fp32", "--format", "json", pdb)
    if not math.isclose(total(cpu_fp32), total(cpu_fp64), rel_tol=1.0e-3):
        raise SystemExit(f"--backend cpu --precision fp32 diverged too far from fp64: {total(cpu_fp32)} vs {total(cpu_fp64)}")

    # The FP32 kernel scales the exact exposed-point count in FP64, so on a
    # small structure it is usually bit-identical to FP64 (no point test
    # flips). Prove the point tests really run in float with a fixture built
    # on the float boundary instead: atom A (radius 1.0) at the origin, atom
    # B (radius 0.5 + 1e-9) at x = 1.5, probe 0, and --resolution 1 so the
    # only test point is A + (1, 0, 0). That point is 0.5 from B: buried in
    # double (0.25 < 0.25 + 1e-9) but exposed in float, where B's radius
    # rounds to exactly 0.5. Every value is exactly representable, so the
    # outcome does not depend on the compiler or on FMA contraction.
    with tempfile.TemporaryDirectory() as probe_dir:
        probe_config = Path(probe_dir) / "fp32_probe.config"
        probe_pdb = Path(probe_dir) / "fp32_probe.pdb"
        probe_config.write_text(FP32_PROBE_CONFIG)
        probe_pdb.write_text(FP32_PROBE_PDB)
        probe_args = ("--resolution", "1", "--probe-radius", "0",
                      "--config-file", str(probe_config), "--format", "json", str(probe_pdb))
        probe_fp64 = run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--precision", "fp64", *probe_args)
        probe_fp32 = run_json(args.fastsasa, args.data_dir, "--backend", "cpu", "--precision", "fp32", *probe_args)
    probe_fp64_first = float(probe_fp64["atoms"][0]["sasa"])
    probe_fp32_first = float(probe_fp32["atoms"][0]["sasa"])
    if probe_fp64_first != 0.0:
        raise SystemExit(f"fp32 probe fixture: fp64 should bury atom A, got {probe_fp64_first}")
    if not math.isclose(probe_fp32_first, 4.0 * math.pi, rel_tol=1.0e-9):
        raise SystemExit(f"--backend cpu --precision fp32 did not run the float kernel: atom A {probe_fp32_first}, expected 4*pi")

    cpu_fp32_lr = subprocess.run(
        [str(args.fastsasa), "--backend", "cpu", "--precision", "fp32", "--lee-richards", pdb],
        cwd=args.data_dir.parent.parent,
        text=True,
        capture_output=True,
    )
    if cpu_fp32_lr.returncode != 0:
        raise SystemExit("--backend cpu --precision fp32 --lee-richards failed")
    if "CPU Lee-Richards is FP64-only" not in cpu_fp32_lr.stderr:
        raise SystemExit("--backend cpu --precision fp32 --lee-richards did not warn about the fp64-only fallback")

    multiple_inputs = subprocess.run(
        [str(args.fastsasa), "--cpu", pdb, pdb],
        cwd=args.data_dir.parent.parent,
        text=True,
        capture_output=True,
    )
    if multiple_inputs.returncode == 0 or "multiple structure inputs" not in multiple_inputs.stderr:
        raise SystemExit("multiple structure input validation failed")

    excessive_selections = [str(args.fastsasa), "--cpu"]
    for index in range(32):
        excessive_selections.extend(["--select", f"s{index}, name ca"])
    excessive_selections.append(pdb)
    selection_limit = subprocess.run(
        excessive_selections,
        cwd=args.data_dir.parent.parent,
        text=True,
        capture_output=True,
    )
    if selection_limit.returncode == 0 or "at most 31 selections" not in selection_limit.stderr:
        raise SystemExit("selection count limit validation failed")

    sr_cpu = run_json(args.fastsasa, args.data_dir, "--cpu", "--shrake-rupley", "--format", "json", pdb)
    sr_cpu_parallel = run_json(args.fastsasa, args.data_dir, "--cpu", "--shrake-rupley", "--threads", "4", "--format", "json", pdb)
    assert_close("SR CPU serial vs threaded", total(sr_cpu), total(sr_cpu_parallel), 1e-9)
    # Independent regression values from the established 1UBQ fixture with
    # ProtOr radii and the matching default resolutions.
    assert_close("SR external 1UBQ reference", total(sr_cpu), 4834.716265, 5e-6)

    lr_cpu = run_json(args.fastsasa, args.data_dir, "--cpu", "--lee-richards", "--resolution", "20", "--format", "json", pdb)
    lr_cpu_default = run_json(args.fastsasa, args.data_dir, "--cpu", "--lee-richards", "--format", "json", pdb)
    assert_close("LR CPU default resolution", total(lr_cpu_default), total(lr_cpu), 1e-12)
    assert_close("LR external 1UBQ reference", total(lr_cpu), 4804.055641, 5e-6)
    lr_cpu_parallel = run_json(args.fastsasa, args.data_dir, "--cpu", "--lee-richards", "--resolution", "20", "--threads", "4", "--format", "json", pdb)
    assert_close("LR CPU serial vs threaded", total(lr_cpu), total(lr_cpu_parallel), 1e-9)

    with tempfile.TemporaryDirectory() as tmp_dir:
        invalid_pdb = Path(tmp_dir) / "invalid_nan.pdb"
        lines = (args.data_dir / "1ubq.pdb").read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith("ATOM"):
                lines[index] = f"{line[:30]}     nan{line[38:]}"
                break
        invalid_pdb.write_text("\n".join(lines) + "\n")
        invalid_proc = subprocess.run(
            [str(args.fastsasa), "--cpu", "--format", "json", str(invalid_pdb)],
            cwd=args.data_dir.parent.parent,
            text=True,
            capture_output=True,
        )
        if invalid_proc.returncode == 0:
            raise SystemExit("CPU non-finite coordinate validation failed")

        short_pdb = Path(tmp_dir) / "short.pdb"
        short_pdb.write_text("ATOM      1  CA  ALA A   1\n")
        short_proc = subprocess.run(
            [str(args.fastsasa), "--cpu", "--format", "json", str(short_pdb)],
            cwd=args.data_dir.parent.parent,
            text=True,
            capture_output=True,
        )
        if short_proc.returncode == 0 or "invalid PDB coordinate record" not in short_proc.stderr:
            raise SystemExit("short PDB coordinate validation failed")

    sr_gpu, sr_gpu_error = try_json(args.fastsasa, args.data_dir, "--shrake-rupley", "--no-cpu-fallback", "--format", "json", pdb)
    lr_gpu, lr_gpu_error = try_json(args.fastsasa, args.data_dir, "--lee-richards", "--resolution", "20", "--no-cpu-fallback", "--format", "json", pdb)
    gpu_available = sr_gpu is not None and lr_gpu is not None
    if gpu_available:
        assert_close("SR FP64 GPU vs CPU", total(sr_gpu), total(sr_cpu), 1e-9)
        assert_close("LR FP64 GPU vs CPU", total(lr_gpu), total(lr_cpu), 1e-9)
        assert_atoms_close("SR FP64 GPU vs CPU", sr_gpu, sr_cpu, 1e-9)
        assert_atoms_close("LR FP64 GPU vs CPU", lr_gpu, lr_cpu, 1e-9)
        sr_gpu_fp32 = run_json(
            args.fastsasa, args.data_dir, "--shrake-rupley", "--precision", "fp32",
            "--no-cpu-fallback", "--format", "json", pdb,
        )
        lr_gpu_fp32 = run_json(
            args.fastsasa, args.data_dir, "--lee-richards", "--resolution", "20",
            "--precision", "fp32", "--no-cpu-fallback", "--format", "json", pdb,
        )
        assert_close("SR FP32 GPU vs CPU", total(sr_gpu_fp32), total(sr_cpu), 1e-2)
        assert_close(
            "LR FP32 GPU vs CPU", total(lr_gpu_fp32), total(lr_cpu),
            max(2e-2, abs(total(lr_cpu)) * 1e-4),
        )
    elif os.environ.get("FASTSASA_REQUIRE_GPU_TESTS") == "1":
        raise SystemExit(
            "GPU CLI validation required but unavailable:\n"
            f"SR: {sr_gpu_error}\nLR: {lr_gpu_error}"
        )
    else:
        sr_fallback = run_json(args.fastsasa, args.data_dir, "--shrake-rupley", "--format", "json", pdb)
        lr_fallback = run_json(args.fastsasa, args.data_dir, "--lee-richards", "--resolution", "20", "--format", "json", pdb)
        assert_close("SR default CPU fallback", total(sr_fallback), total(sr_cpu), 1e-9)
        assert_close("LR default CPU fallback", total(lr_fallback), total(lr_cpu), 1e-9)

    sr_mode_args = [] if gpu_available else ["--cpu"]
    pdb_total = total(run_json(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "json", cif_peer))
    cif_total = total(run_json(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "json", cif))
    assert_close("PDB vs mmCIF SR", pdb_total, cif_total, 1e-6)

    with tempfile.TemporaryDirectory() as tmp_dir:
        forced_cif = Path(tmp_dir) / "2isk.no_suffix"
        forced_cif.write_text((args.data_dir / "2isk.cif").read_text())
        forced_cif_total = total(run_json(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "json", "--cif", str(forced_cif)))
        assert_close("--cif forced mmCIF reader", forced_cif_total, cif_total, 1e-6)

    selection_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "log", "--select", "ca, name ca", pdb)
    match = re.search(r"^ca\s*:\s*([0-9.]+)", selection_output, flags=re.MULTILINE)
    if match is None or float(match.group(1)) <= 0.0:
        raise SystemExit("selection validation failed")

    class_output = run_json(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "json", "--classes", "--select", "ca, name ca", pdb)
    classes = class_output.get("classes", {})
    class_sum = float(classes.get("polar_sasa", 0.0)) + float(classes.get("apolar_sasa", 0.0)) + float(classes.get("unknown_sasa", 0.0))
    assert_close("class total", total(class_output), class_sum, 1e-6)
    class_selection = class_output["selections"][0]
    selected_class_sum = float(class_selection["polar_sasa"]) + float(class_selection["apolar_sasa"]) + float(class_selection["unknown_sasa"])
    assert_close("selection class total", float(class_selection["sasa"]), selected_class_sum, 1e-6)

    unlabeled_selection_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "log", "--select", "name ca", pdb)
    match = re.search(r"^name_ca\s*:\s*([0-9.]+)", unlabeled_selection_output, flags=re.MULTILINE)
    if match is None or float(match.group(1)) <= 0.0:
        raise SystemExit("unlabeled selection validation failed")

    xml_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "xml", "--select", "ca, name ca", pdb)
    xml_root = ET.fromstring(xml_output)
    if xml_root.tag != "FastSASA" or xml_root.find("selections/selection") is None:
        raise SystemExit("XML exporter validation failed")

    rsa_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "rsa", pdb)
    rsa_residue = re.search(r"^RES\s+MET\s+A\s*1\s+(.+)$", rsa_output, flags=re.MULTILINE)
    if ("REM RES _ NUM" not in rsa_output or rsa_residue is None or
            "END  Absolute sums over single chains surface" not in rsa_output or
            not re.search(r"^TOTAL\s+", rsa_output, flags=re.MULTILINE)):
        raise SystemExit("RSA exporter validation failed")
    if len(rsa_residue.group(1).split()) != 10:
        raise SystemExit("RSA exporter did not emit five ABS/REL field pairs")

    pdb_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "pdb", pdb)
    if "ATOM" not in pdb_output or "REMARK" not in pdb_output:
        raise SystemExit("PDB exporter validation failed")

    cif_output = run_text(args.fastsasa, args.data_dir, *sr_mode_args, "--shrake-rupley", "--format", "cif", cif)
    if "_FastSASA_results.total_sasa" not in cif_output:
        raise SystemExit("FastSASA CIF namespace validation failed")

    # Surface-point export: exposed counts must reproduce the per-atom SASA
    # exactly (same point test as the calculation).
    with tempfile.TemporaryDirectory() as tmp_dir:
        surface_path = Path(tmp_dir) / "surface.txt"
        sasa_pdb = run_text(args.fastsasa, args.data_dir, "--shrake-rupley",
                            "--format", "pdb", "--surface-points",
                            str(surface_path), "tests/data/1ubq.pdb")
        counts = collections.Counter()
        for line in surface_path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 4:
                raise SystemExit("surface-points line is not 'x y z atom_index'")
            counts[int(fields[3])] += 1
        atom_rows = [line for line in sasa_pdb.splitlines()
                     if line.startswith(("ATOM", "HETATM"))]
        if not atom_rows:
            raise SystemExit("surface-points validation found no atoms")
        for index, line in enumerate(atom_rows):
            radius = float(line[54:60])
            sasa = float(line[60:66])
            derived = (4.0 * math.pi * (radius + 1.4) ** 2 *
                       counts.get(index, 0) / 100.0)
            if abs(derived - sasa) > 0.006:
                raise SystemExit(
                    f"surface-point count disagrees with SASA at atom {index}: "
                    f"{derived:.4f} vs {sasa:.4f}")

    print(f"fastsasa_cli_validation,status,pass,gpu_available,{int(gpu_available)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
