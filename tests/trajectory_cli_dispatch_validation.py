#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path


def _run(fastsasa: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(fastsasa), *args], text=True, capture_output=True, check=False)


def _run_cpu_backend(fastsasa: Path, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = "cpu"
    return subprocess.run(
        [str(fastsasa), *args],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_failure_with(proc: subprocess.CompletedProcess[str], text: str) -> None:
    output = proc.stderr + proc.stdout
    if proc.returncode == 0 or text not in output:
        raise SystemExit(f"expected failed command containing {text!r}, got:\n{output}")


def _write_record(handle, payload: bytes) -> None:
    handle.write(struct.pack("<i", len(payload)))
    handle.write(payload)
    handle.write(struct.pack("<i", len(payload)))


def _write_dcd(path: Path, *frames: list[tuple[float, float, float]]) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<i", header, 4, len(frames))
    title = struct.pack("<i", 1) + b"FastSASA trajectory CLI validation".ljust(80)
    n_atoms = len(frames[0])
    with path.open("wb") as handle:
        _write_record(handle, bytes(header))
        _write_record(handle, title)
        _write_record(handle, struct.pack("<i", n_atoms))
        for coordinates in frames:
            for axis in range(3):
                _write_record(handle, struct.pack(f"<{n_atoms}f", *(xyz[axis] for xyz in coordinates)))


def _pdb_coordinates(path: Path) -> list[tuple[float, float, float]]:
    return [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def _frame_rows(proc: subprocess.CompletedProcess[str]) -> list[dict[str, str]]:
    if proc.returncode != 0:
        raise SystemExit(f"trajectory surface-point command failed:\n{proc.stderr}{proc.stdout}")
    return list(csv.DictReader(line for line in proc.stdout.splitlines() if line.strip()))


def _parse_xyz_frames(text: str) -> list[list[str]]:
    lines = text.splitlines()
    frames: list[list[str]] = []
    cursor = 0
    while cursor < len(lines):
        slots = int(lines[cursor])
        if not lines[cursor + 1].startswith("FastSASA accessible surface frame"):
            raise SystemExit("surface XYZ comment line is malformed")
        block = lines[cursor + 2 : cursor + 2 + slots]
        if len(block) != slots or any(not line.startswith("X ") for line in block):
            raise SystemExit("surface XYZ frame block is malformed")
        frames.append([line[2:] for line in block])
        cursor += 2 + slots
    return frames


def _parse_dcd_frames(data: bytes) -> list[list[str]]:
    # CHARMM DCD: 84-byte control block, title block, atom count, then
    # X/Y/Z float planes per frame, each wrapped in 4-byte record markers.
    def i32(offset: int) -> int:
        return struct.unpack_from("<i", data, offset)[0]

    if i32(0) != 84 or data[4:8] != b"CORD" or i32(88) != 84:
        raise SystemExit("surface DCD control block is malformed")
    n_frames = i32(8)
    title_bytes = i32(92)
    cursor = 96 + title_bytes + 4
    if i32(cursor) != 4:
        raise SystemExit("surface DCD atom-count record is malformed")
    slots = i32(cursor + 4)
    cursor += 12
    frames: list[list[str]] = []
    for _ in range(n_frames):
        planes = []
        for _axis in range(3):
            if i32(cursor) != 4 * slots or i32(cursor + 4 + 4 * slots) != 4 * slots:
                raise SystemExit("surface DCD coordinate record is malformed")
            planes.append(struct.unpack_from(f"<{slots}f", data, cursor + 4))
            cursor += 8 + 4 * slots
        frames.append([f"{x:.3f} {y:.3f} {z:.3f}" for x, y, z in zip(*planes)])
    if cursor != len(data):
        raise SystemExit("surface DCD has trailing bytes")
    return frames


def _validate_surface_points(fastsasa: Path, tmp_path: Path) -> None:
    # Per-frame surface export must reproduce each frame's reported total
    # exactly (same point test as the calculation), and the .xyz and .dcd
    # forms must hold the same points per frame, padded to a common slot
    # count.
    topology = Path("tests/data/1ubq.pdb")
    trajectory = tmp_path / "surface.dcd"
    frame0 = _pdb_coordinates(topology)
    frame1 = [(x + 1.0, y, z) for x, y, z in frame0]
    _write_dcd(trajectory, frame0, frame1)
    base = (
        "trajectory",
        "--topology",
        str(topology),
        "--trajectory",
        str(trajectory),
        "--cpu",
        "--threads",
        "1",
        "--frames",
        ":",
        "--shrake-rupley",
        "--surface-points",
    )
    surface_txt = tmp_path / "surface.txt"
    surface_xyz = tmp_path / "surface.xyz"
    surface_dcd = tmp_path / "surface_points.dcd"
    rows = _frame_rows(_run(fastsasa, *base, str(surface_txt)))
    _frame_rows(_run(fastsasa, *base, str(surface_xyz)))
    _frame_rows(_run(fastsasa, *base, str(surface_dcd)))
    if len(rows) != 2:
        raise SystemExit(f"expected two trajectory frames, got {len(rows)}")

    radii_pdb = _run(fastsasa, "--shrake-rupley", "--format", "pdb", str(topology))
    if radii_pdb.returncode != 0:
        raise SystemExit(f"structure radius export failed:\n{radii_pdb.stderr}")
    radii = [float(line[54:60]) for line in radii_pdb.stdout.splitlines()
             if line.startswith(("ATOM", "HETATM"))]

    counts: list[collections.Counter[int]] = []
    frame_points: list[set[str]] = []
    for line in surface_txt.read_text().splitlines():
        fields = line.split()
        if line.startswith("# frame"):
            if int(fields[2]) != len(counts):
                raise SystemExit(f"unexpected surface frame header: {line}")
            counts.append(collections.Counter())
            frame_points.append(set())
            continue
        if len(fields) != 4:
            raise SystemExit("trajectory surface-points line is not 'x y z atom_index'")
        counts[-1][int(fields[3])] += 1
        frame_points[-1].add(" ".join(fields[:3]))
    if len(counts) != 2:
        raise SystemExit(f"expected two surface frame blocks, got {len(counts)}")
    for frame, (row, frame_counts) in enumerate(zip(rows, counts)):
        if len(radii) != len(frame_counts) and sum(frame_counts.values()) == 0:
            raise SystemExit("surface export produced no points")
        derived = sum(4.0 * math.pi * (radius + 1.4) ** 2 * frame_counts.get(index, 0) / 100.0
                      for index, radius in enumerate(radii))
        if abs(derived - float(row["total_sasa"])) > 1.0e-3:
            raise SystemExit(
                f"surface-point count disagrees with frame {frame} total: "
                f"{derived:.4f} vs {row['total_sasa']}")

    xyz_frames = _parse_xyz_frames(surface_xyz.read_text())
    if len(xyz_frames) != 2:
        raise SystemExit(f"expected two XYZ frames, got {len(xyz_frames)}")
    slots = max(sum(frame_counts.values()) for frame_counts in counts)
    for frame, (block, expected) in enumerate(zip(xyz_frames, frame_points)):
        if len(block) != slots:
            raise SystemExit(f"XYZ frame {frame} has {len(block)} slots, expected {slots}")
        if set(block) != expected:
            raise SystemExit(f"XYZ frame {frame} points differ from the text export")
    dcd_frames = _parse_dcd_frames(surface_dcd.read_bytes())
    if len(dcd_frames) != 2:
        raise SystemExit(f"expected two DCD frames, got {len(dcd_frames)}")
    for frame, (block, xyz_block) in enumerate(zip(dcd_frames, xyz_frames)):
        if block != xyz_block:
            raise SystemExit(f"DCD frame {frame} points differ from the XYZ export")
    shifted = {" ".join(f"{float(value) + offset:.3f}" for value, offset in zip(point.split(), (1.0, 0.0, 0.0)))
               for point in frame_points[0]}
    if shifted != frame_points[1]:
        raise SystemExit("frame 1 surface points are not the translated frame 0 points")


def _validate_surface_resolution(fastsasa: Path, tmp_path: Path) -> None:
    # --surface-resolution must decouple the exported point density from
    # --resolution: the reported total still reflects --resolution, and the
    # surface point count per atom is bounded by --surface-resolution, not
    # --resolution.
    topology = Path("tests/data/1ubq.pdb")
    trajectory = tmp_path / "surface_resolution.dcd"
    frame0 = _pdb_coordinates(topology)
    _write_dcd(trajectory, frame0)
    surface_txt = tmp_path / "surface_resolution.txt"
    high_res_rows = _frame_rows(_run(
        fastsasa, "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
        "--cpu", "--threads", "1", "--frames", ":", "--shrake-rupley", "--resolution", "260",
    ))
    coupled_rows = _frame_rows(_run(
        fastsasa, "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
        "--cpu", "--threads", "1", "--frames", ":", "--shrake-rupley", "--resolution", "260",
        "--surface-points", str(surface_txt), "--surface-resolution", "8",
    ))
    if abs(float(high_res_rows[0]["total_sasa"]) - float(coupled_rows[0]["total_sasa"])) > 1.0e-6:
        raise SystemExit("--surface-resolution changed the reported --resolution total")
    counts: collections.Counter[int] = collections.Counter()
    for line in surface_txt.read_text().splitlines():
        fields = line.split()
        if len(fields) == 4:
            counts[int(fields[3])] += 1
    if not counts or max(counts.values()) > 8:
        raise SystemExit(f"--surface-resolution 8 produced more than 8 points for an atom: {counts}")


def _validate_cpu_precision(fastsasa: Path, tmp_path: Path) -> None:
    # --precision fp32 on --backend cpu must actually change the Shrake-Rupley
    # result (not be silently ignored), stay close to the fp64 reference, and
    # must warn - not silently ignore - when combined with --lee-richards,
    # which stays fp64-only on CPU.
    topology = Path("tests/data/1ubq.pdb")
    trajectory = tmp_path / "cpu_precision.dcd"
    frame0 = _pdb_coordinates(topology)
    _write_dcd(trajectory, frame0)
    base = (
        "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
        "--cpu", "--threads", "1", "--frames", ":", "--shrake-rupley", "--resolution", "260",
    )
    fp64_rows = _frame_rows(_run(fastsasa, *base, "--precision", "fp64"))
    fp32_rows = _frame_rows(_run(fastsasa, *base, "--precision", "fp32"))
    fp64_total = float(fp64_rows[0]["total_sasa"])
    fp32_total = float(fp32_rows[0]["total_sasa"])
    if abs(fp32_total - fp64_total) / fp64_total > 1.0e-3:
        raise SystemExit(f"trajectory --cpu --precision fp32 diverged too far from fp64: {fp32_total} vs {fp64_total}")

    # The FP32 kernel scales exact exposed counts in FP64, so a real structure
    # is often bit-identical to FP64. Prove the float point tests run with a
    # float-boundary fixture: A (radius 1.0) at the origin, B (radius
    # 0.5 + 1e-9) at x = 1.5, probe 0, --resolution 1 (single test point at
    # +x). A's point is buried in double, exposed in float (B's radius rounds
    # to exactly 0.5): totals are pi (fp64) versus 5*pi (fp32), exactly.
    probe_topology = tmp_path / "fp32_probe.pdb"
    probe_topology.write_text(
        "ATOM      1  A1  PRB A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      2  A2  PRB A   1       1.500   0.000   0.000  1.00  0.00           C  \n"
        "END\n")
    probe_config = tmp_path / "fp32_probe.config"
    probe_config.write_text("name: fp32probe\n\ntypes:\nBIG 1.0 apolar\nEDGE 0.5000000001 apolar\n\n"
                            "atoms:\nPRB A1 BIG\nPRB A2 EDGE\n")
    probe_trajectory = tmp_path / "fp32_probe.dcd"
    _write_dcd(probe_trajectory, [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    probe_base = (
        "trajectory", "--topology", str(probe_topology), "--trajectory", str(probe_trajectory),
        "--cpu", "--threads", "1", "--frames", ":", "--shrake-rupley", "--resolution", "1",
        "--probe-radius", "0", "--config-file", str(probe_config),
    )
    probe_fp64 = float(_frame_rows(_run(fastsasa, *probe_base, "--precision", "fp64"))[0]["total_sasa"])
    probe_fp32 = float(_frame_rows(_run(fastsasa, *probe_base, "--precision", "fp32"))[0]["total_sasa"])
    if abs(probe_fp64 - math.pi) > 1.0e-6:
        raise SystemExit(f"fp32 probe fixture: trajectory fp64 total should be pi, got {probe_fp64}")
    if abs(probe_fp32 - 5.0 * math.pi) > 1.0e-6:
        raise SystemExit(f"trajectory --cpu --precision fp32 did not run the float kernel: total {probe_fp32}, expected 5*pi")

    lr_proc = _run(
        fastsasa, "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
        "--cpu", "--threads", "1", "--frames", ":", "--lee-richards", "--precision", "fp32",
    )
    if lr_proc.returncode != 0:
        raise SystemExit(f"trajectory --cpu --lee-richards --precision fp32 failed:\n{lr_proc.stderr}")
    if "CPU Lee-Richards is FP64-only" not in lr_proc.stderr:
        raise SystemExit("trajectory --cpu --lee-richards --precision fp32 did not warn about the fp64-only fallback")


def _validate_residue_aliases(fastsasa: Path, tmp_path: Path) -> None:
    # MD residue variants must resolve to the standard residue's radii in the
    # trajectory CLI's own config parser: HIS -> HIE changes nothing.
    topology = Path("tests/data/1ubq.pdb")
    alias_topology = tmp_path / "1ubq_hie.pdb"
    alias_topology.write_text(topology.read_text().replace(" HIS ", " HIE "))
    trajectory = tmp_path / "alias.dcd"
    _write_dcd(trajectory, _pdb_coordinates(topology))
    results = []
    for top in (topology, alias_topology):
        proc = _run(fastsasa, "trajectory", "--topology", str(top), "--trajectory", str(trajectory),
                    "--cpu", "--threads", "1", "--frames", ":", "--classes")
        if proc.returncode != 0:
            raise SystemExit(f"trajectory run on {top.name} failed:\n{proc.stderr}")
        if "guessing element" in proc.stderr:
            raise SystemExit(f"{top.name}: atoms fell back to element radii:\n{proc.stderr}")
        results.append(_frame_rows(proc)[0])
    for column in ("total_sasa", "polar_sasa", "apolar_sasa"):
        if results[0][column] != results[1][column]:
            raise SystemExit(f"HIS -> HIE changed {column}: {results[0][column]} vs {results[1][column]}")


# A CHARMM-named peptide: ACE cap (CAY/CY/OY), HSD, ILE CD, C-terminal
# OT1/OT2, one hydrogen with a repartitioned mass, and force-field type names
# that are not elements (CT3, HA3, NR1, ...). Fields: segment, resid,
# residue, atom, PSF type, mass, element.
_CHARMM_PEPTIDE = (
    ("PROA", 1, "GLY", "CAY", "CT3", 12.011, "C"), ("PROA", 1, "GLY", "HY1", "HA3", 1.008, "H"),
    ("PROA", 1, "GLY", "HY2", "HA3", 1.008, "H"), ("PROA", 1, "GLY", "CY", "C", 12.011, "C"),
    ("PROA", 1, "GLY", "OY", "O", 15.999, "O"), ("PROA", 1, "GLY", "N", "NH1", 14.007, "N"),
    ("PROA", 1, "GLY", "HN", "H", 1.008, "H"), ("PROA", 1, "GLY", "CA", "CT2", 12.011, "C"),
    ("PROA", 1, "GLY", "HA1", "HB2", 1.008, "H"), ("PROA", 1, "GLY", "C", "C", 12.011, "C"),
    ("PROA", 1, "GLY", "O", "O", 15.999, "O"), ("PROA", 2, "HSD", "N", "NH1", 14.007, "N"),
    ("PROA", 2, "HSD", "CA", "CT1", 12.011, "C"), ("PROA", 2, "HSD", "CB", "CT2", 12.011, "C"),
    ("PROA", 2, "HSD", "ND1", "NR1", 14.007, "N"), ("PROA", 2, "HSD", "HD1", "H", 1.008, "H"),
    ("PROA", 2, "HSD", "CG", "CPH1", 12.011, "C"), ("PROA", 2, "HSD", "CE1", "CPH2", 12.011, "C"),
    ("PROA", 2, "HSD", "NE2", "NR2", 14.007, "N"), ("PROA", 2, "HSD", "CD2", "CPH1", 12.011, "C"),
    ("PROA", 2, "HSD", "C", "C", 12.011, "C"), ("PROA", 2, "HSD", "O", "O", 15.999, "O"),
    ("PROA", 3, "ILE", "N", "NH1", 14.007, "N"), ("PROA", 3, "ILE", "CA", "CT1", 12.011, "C"),
    ("PROA", 3, "ILE", "CB", "CT1", 12.011, "C"), ("PROA", 3, "ILE", "CG2", "CT3", 12.011, "C"),
    ("PROA", 3, "ILE", "CG1", "CT2", 12.011, "C"), ("PROA", 3, "ILE", "CD", "CT3", 12.011, "C"),
    ("PROA", 3, "ILE", "HD1", "HA3", 3.024, "H"), ("PROA", 3, "ILE", "C", "C", 12.011, "C"),
    ("PROA", 3, "ILE", "O", "O", 15.999, "O"), ("PROA", 4, "LEU", "N", "NH1", 14.007, "N"),
    ("PROA", 4, "LEU", "CA", "CT1", 12.011, "C"), ("PROA", 4, "LEU", "CB", "CT2", 12.011, "C"),
    ("PROA", 4, "LEU", "CG", "CT1", 12.011, "C"), ("PROA", 4, "LEU", "CD1", "CT3", 12.011, "C"),
    ("PROA", 4, "LEU", "CD2", "CT3", 12.011, "C"), ("PROA", 4, "LEU", "C", "CC", 12.011, "C"),
    ("PROA", 4, "LEU", "OT1", "OC", 15.9994, "O"), ("PROA", 4, "LEU", "OT2", "OC", 15.9994, "O"),
)
# Atoms the ProtOr table does not cover: ions and an unknown ligand atom.
_CHARMM_EXTRAS = (
    ("ION", 5, "SOD", "SOD", "SOD", 22.98977, "NA"), ("ION", 6, "CLA", "CLA", "CLA", 35.45, "CL"),
    ("LIG", 7, "LIG", "XX1", "CG2R61", 12.011, "C"),
)


def _charmm_coordinates(n_atoms: int) -> list[tuple[float, float, float]]:
    return [(1.9 * (i % 4), 1.9 * ((i // 4) % 4), 1.9 * (i // 16)) for i in range(n_atoms)]


def _write_charmm_inputs(tmp_path: Path, stem: str, atoms) -> tuple[Path, Path, Path]:
    coordinates = _charmm_coordinates(len(atoms))
    psf_lines = ["PSF", "", "       1 !NTITLE", " REMARKS FastSASA CHARMM validation", "",
                 f"{len(atoms):8d} !NATOM"]
    pdb_lines = []
    for index, ((segment, resid, residue, name, atom_type, mass, element), xyz) in enumerate(
            zip(atoms, coordinates), start=1):
        psf_lines.append(f"{index:8d} {segment:<4s} {resid:<4d} {residue:<4s} {name:<4s} {atom_type:<6s}"
                         f"  0.000000 {mass:13.4f}           0")
        pdb_name = name if len(name) == 4 else f" {name:<3s}"
        pdb_lines.append(f"ATOM  {index:5d} {pdb_name} {residue:<4s}X{resid:4d}    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00      {segment:<4s}{element:>2s}")
    psf = tmp_path / f"{stem}.psf"
    pdb = tmp_path / f"{stem}.pdb"
    dcd = tmp_path / f"{stem}.dcd"
    psf.write_text("\n".join(psf_lines + ["", "       0 !NBOND", ""]) + "\n")
    pdb.write_text("\n".join(pdb_lines + ["END"]) + "\n")
    _write_dcd(dcd, coordinates)
    return psf, pdb, dcd


def _validate_charmm_psf(fastsasa: Path, tmp_path: Path) -> None:
    # PSF type columns hold force-field types, not elements: elements must come
    # from the mass, CHARMM atom names must resolve to the ProtOr names, and
    # the PSF and PDB readers must then agree atom for atom.
    def summary(topology: Path, trajectory: Path, *extra: str) -> tuple[dict[str, str], str]:
        proc = _run(fastsasa, "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
                    "--cpu", "--threads", "1", "--frames", "0", "--summary", "--classes", *extra)
        return _summary_row(proc), proc.stderr

    psf, pdb, dcd = _write_charmm_inputs(tmp_path, "charmm_peptide", _CHARMM_PEPTIDE)
    n_heavy = sum(1 for atom in _CHARMM_PEPTIDE if atom[6] != "H")
    for extra, expected_atoms in (((), n_heavy), (("--hydrogen",), len(_CHARMM_PEPTIDE))):
        psf_row, psf_err = summary(psf, dcd, *extra)
        pdb_row, pdb_err = summary(pdb, dcd, *extra)
        if int(psf_row["atoms"]) != expected_atoms or int(pdb_row["atoms"]) != expected_atoms:
            raise SystemExit(f"CHARMM {extra}: expected {expected_atoms} atoms, PSF {psf_row['atoms']}, "
                             f"PDB {pdb_row['atoms']}")
        for column in ("total_sasa_sum", "polar_sasa_sum", "apolar_sasa_sum"):
            if psf_row[column] != pdb_row[column]:
                raise SystemExit(f"CHARMM {extra}: PSF and PDB disagree on {column}: "
                                 f"{psf_row[column]} vs {pdb_row[column]}")
        if not extra:
            if float(psf_row["unknown_sasa_sum"]) != 0.0 or "radius table" in psf_err + pdb_err:
                raise SystemExit("CHARMM peptide heavy atoms were not all found in the ProtOr table:\n"
                                 f"{psf_row}\n{psf_err}{pdb_err}")

    psf, pdb, dcd = _write_charmm_inputs(tmp_path, "charmm_extras", _CHARMM_PEPTIDE + _CHARMM_EXTRAS)
    psf_row, psf_err = summary(psf, dcd, "--hetatm")
    pdb_row, pdb_err = summary(pdb, dcd, "--hetatm")
    if psf_row["total_sasa_sum"] != pdb_row["total_sasa_sum"]:
        raise SystemExit(f"CHARMM extras: PSF {psf_row['total_sasa_sum']} vs PDB {pdb_row['total_sasa_sum']}")
    for stderr in (psf_err, pdb_err):
        if "radius table" not in stderr or "LIG XX1" not in stderr or "SOD SOD" not in stderr:
            raise SystemExit(f"atoms outside the radius table were not reported:\n{stderr}")


def _validate_cpu_selected_centers(fastsasa: Path, tmp_path: Path) -> None:
    # Selection runs compute surface points only for the selected atoms; the
    # CPU result must be identical to computing every atom.
    topology = Path("tests/data/1ubq.pdb")
    trajectory = tmp_path / "selected_centers.dcd"
    coordinates = _pdb_coordinates(topology)
    _write_dcd(trajectory, coordinates, [(x + 0.3, y, z) if i % 5 == 0 else (x, y, z)
                                         for i, (x, y, z) in enumerate(coordinates)])
    outputs = []
    for enabled in ("0", "1"):
        environment = dict(os.environ, FASTSASA_SELECTED_CENTER=enabled)
        proc = subprocess.run([str(fastsasa), "trajectory", "--topology", str(topology), "--trajectory",
                               str(trajectory), "--cpu", "--frames", ":", "--select", "first, resi 1-10",
                               "--select", "core, resn LEU+ILE+VAL"],
                              env=environment, text=True, capture_output=True, check=False)
        rows = _frame_rows(proc)
        outputs.append([(row["frame"], row["selection"], row["selection_sasa"]) for row in rows])
    if outputs[0] != outputs[1] or len(outputs[0]) != 4:
        raise SystemExit(f"CPU selected-centre results differ from the full calculation:\n{outputs}")
    # A selection matching no atom reports zero, as the full calculation does.
    proc = _run(fastsasa, "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
                "--cpu", "--frames", "0", "--select", "none, segid NOPE")
    rows = _frame_rows(proc)
    if len(rows) != 1 or float(rows[0]["selection_sasa"]) != 0.0:
        raise SystemExit(f"empty selection did not report zero:\n{proc.stdout}{proc.stderr}")


def _validate_automatic_backend(fastsasa: Path, tmp_path: Path) -> None:
    # Automatic mode runs single structures on the CPU (a GPU context costs
    # more than the whole calculation) and never picks a software Vulkan
    # device; explicit requests are unchanged.
    import json

    environment = {key: value for key, value in os.environ.items() if key != "FASTSASA_BACKEND"}
    proc = subprocess.run([str(fastsasa), "--format", "json", "tests/data/1ubq.pdb"], env=environment,
                          text=True, capture_output=True, check=False)
    if proc.returncode != 0 or json.loads(proc.stdout)["parameters"]["backend"] != "cpu":
        raise SystemExit(f"automatic backend for a single structure is not the CPU:\n{proc.stdout}{proc.stderr}")

    icds = sorted(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json")) if Path("/usr/share/vulkan/icd.d").is_dir() else []
    if not icds:
        return
    topology = Path("tests/data/1ubq.pdb")
    trajectory = tmp_path / "software_vulkan.dcd"
    _write_dcd(trajectory, _pdb_coordinates(topology))
    environment.update(VK_DRIVER_FILES=str(icds[0]), FASTSASA_TEST_DISABLE_CUDA="1")
    proc = subprocess.run([str(fastsasa), "trajectory", "--topology", str(topology), "--trajectory", str(trajectory),
                           "--summary"], env=environment, text=True, capture_output=True, check=False)
    if proc.returncode != 0 or "using threaded CPU" not in proc.stderr:
        raise SystemExit(f"automatic mode used the software Vulkan device:\n{proc.stderr}")


def _validate_native_xtc(fastsasa: Path, tmp_path: Path) -> None:
    # The built-in XTC reader (vendored xdrfile) must give the same per-frame
    # SASA as a DCD holding the decoded coordinates (written by MDAnalysis),
    # for the compressed (> 9 atoms) and raw (<= 9 atoms) frame layouts, and
    # with frame selection.
    for stem, topology in (("1ubq_5frames", "1ubq.pdb"), ("1ubq_4atoms", "1ubq_4atoms.pdb")):
        for frames in (":", "-1", "1:4:2"):
            rows = []
            for suffix in ("xtc", "dcd"):
                proc = _run(fastsasa, "trajectory", "--topology", f"tests/data/{topology}",
                            "--trajectory", f"tests/data/{stem}.{suffix}", "--cpu", "--frames", frames)
                rows.append([(row["frame"], row["total_sasa"]) for row in _frame_rows(proc)])
            if not rows[0] or rows[0] != rows[1]:
                raise SystemExit(f"{stem} --frames {frames}: XTC {rows[0]} differs from DCD {rows[1]}")

    # A file cut off inside its last frame lists only the complete frames, and
    # asking for a missing frame is an error, not an empty result.
    source = Path("tests/data/1ubq_5frames.xtc").read_bytes()
    truncated = tmp_path / "truncated.xtc"
    truncated.write_bytes(source[:-100])
    proc = _run(fastsasa, "trajectory", "--topology", "tests/data/1ubq.pdb", "--trajectory", str(truncated),
                "--cpu", "--frames", ":")
    if [row["frame"] for row in _frame_rows(proc)] != ["0", "1", "2", "3"]:
        raise SystemExit(f"truncated XTC frames:\n{proc.stdout}{proc.stderr}")
    proc = subprocess.run([str(fastsasa), "trajectory", "--topology", "tests/data/1ubq.pdb", "--trajectory",
                           str(truncated), "--cpu", "--frames", "4"], text=True, capture_output=True, check=False)
    if proc.returncode == 0:
        raise SystemExit(f"missing XTC frame 4 was accepted:\n{proc.stdout}{proc.stderr}")

    # A compressed block larger than the decoder's buffer (a corrupt or
    # crafted file) is rejected before any decoding.
    header = bytearray(source[:92])
    header[88:92] = (4_000_000).to_bytes(4, "big")
    corrupt = tmp_path / "corrupt.xtc"
    corrupt.write_bytes(bytes(header) + bytes(4_000_000))
    proc = subprocess.run([str(fastsasa), "trajectory", "--topology", "tests/data/1ubq.pdb", "--trajectory",
                           str(corrupt), "--cpu", "--frames", ":"], text=True, capture_output=True, check=False)
    if proc.returncode == 0 or proc.returncode < 0 or proc.returncode >= 128:
        raise SystemExit(f"corrupt XTC: exit {proc.returncode}\n{proc.stderr}")

    # An empty coordinate range (maxint < minint) must be an error, not a
    # division by zero inside the decoder.
    empty_range = bytearray(source)
    min_z = int.from_bytes(empty_range[68:72], "big", signed=True)
    empty_range[80:84] = (min_z - 1).to_bytes(4, "big", signed=True)
    corrupt.write_bytes(bytes(empty_range))
    proc = subprocess.run([str(fastsasa), "trajectory", "--topology", "tests/data/1ubq.pdb", "--trajectory",
                           str(corrupt), "--cpu", "--frames", "0"], text=True, capture_output=True, check=False)
    if proc.returncode == 0 or proc.returncode < 0 or proc.returncode >= 128:
        raise SystemExit(f"XTC with an empty coordinate range: exit {proc.returncode}\n{proc.stderr}")


def _validate_radii_without_source_tree(fastsasa: Path, tmp_path: Path) -> None:
    # The default ProtOr table must not depend on the working directory.
    structure = Path("tests/data/1ubq.pdb").resolve()
    fastsasa = fastsasa.resolve()
    in_tree = subprocess.run([str(fastsasa), str(structure)], text=True, capture_output=True, check=False)
    elsewhere = subprocess.run([str(fastsasa), str(structure)], text=True, capture_output=True, check=False,
                               cwd=tmp_path)
    if in_tree.returncode != 0 or elsewhere.returncode != 0:
        raise SystemExit(f"1ubq run failed:\n{in_tree.stderr}{elsewhere.stderr}")
    if in_tree.stdout != elsewhere.stdout or "guessing element" in elsewhere.stderr:
        raise SystemExit("radii changed with the working directory:\n"
                         f"{in_tree.stdout}\n---\n{elsewhere.stdout}\n{elsewhere.stderr}")


def _write_mmcif(path: Path) -> None:
    path.write_text(
        "data_policy\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "_atom_site.pdbx_formal_charge\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.0 0.0 ? 1 ALA A CA 1\n"
        "ATOM 2 H H  . ALA A 1 1 ? 10.0 0.0 0.0 1.0 0.0 ? 1 ALA A H 1\n"
        "HETATM 3 O O . HOH W 2 2 ? 20.0 0.0 0.0 1.0 0.0 ? 2 HOH W O 1\n"
        "#\n"
    )


def _summary_row(proc: subprocess.CompletedProcess[str]) -> dict[str, str]:
    if proc.returncode != 0:
        raise SystemExit(f"trajectory policy command failed:\n{proc.stderr}{proc.stdout}")
    rows = list(csv.DictReader(line for line in proc.stdout.splitlines() if line.strip()))
    if len(rows) != 1:
        raise SystemExit(f"expected one trajectory summary row, got:\n{proc.stdout}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    args = parser.parse_args()

    missing_topology = "tests/data/does-not-exist.pdb"
    missing_trajectory = "tests/data/does-not-exist.xtc"

    proc = _run(args.fastsasa, "trajectory", "--help")
    if proc.returncode != 0 or "usage:" not in proc.stdout:
        raise SystemExit("trajectory --help validation failed")

    proc = _run(args.fastsasa, "trajectory", missing_topology, missing_trajectory, "--help")
    if proc.returncode != 0 or "usage:" not in proc.stdout:
        raise SystemExit("position-independent trajectory --help validation failed")

    proc = _run(args.fastsasa, "trajectory", missing_topology, missing_trajectory, "--version")
    if proc.returncode != 0 or not proc.stdout.startswith("FastSASA "):
        raise SystemExit("position-independent trajectory --version validation failed")

    proc = _run(args.fastsasa, "trajectory", "--trajectory", missing_trajectory)
    _require_failure_with(proc, "trajectory mode requires --topology FILE and --trajectory FILE")

    proc = _run(args.fastsasa, "trajectory", "--topology")
    _require_failure_with(proc, "missing --topology value")

    proc = _run(args.fastsasa, "trajectory", "--trajectory")
    _require_failure_with(proc, "missing --trajectory value")

    proc = _run(args.fastsasa, "trajectory", "--topology", missing_topology, "--trajectory")
    _require_failure_with(proc, "missing --trajectory value")

    proc = _run(
        args.fastsasa,
        "trajectory",
        "--topology",
        missing_topology,
        "--trajectory",
        missing_trajectory,
        "--output",
    )
    _require_failure_with(proc, "missing --output value")

    proc = _run(
        args.fastsasa,
        "trajectory",
        missing_topology,
        missing_trajectory,
        "--probe-radius",
        "invalid",
    )
    _require_failure_with(proc, "invalid --probe-radius value")

    proc = _run(
        args.fastsasa,
        "trajectory",
        missing_topology,
        missing_trajectory,
        "--summary",
        "--residue",
    )
    _require_failure_with(proc, "trajectory mode accepts only one of --summary and --residue")

    proc = _run(
        args.fastsasa,
        "trajectory",
        "--topology",
        missing_topology,
        "--trajectory",
        missing_trajectory,
        "--summary",
    )
    _require_failure_with(proc, "failed to read topology")
    if "trajectory input has no --filter" not in proc.stderr:
        raise SystemExit("missing warning for unfiltered trajectory mode")

    proc = _run(
        args.fastsasa,
        "trajectory",
        missing_topology,
        missing_trajectory,
        "--typoed-option",
    )
    _require_failure_with(proc, "unknown trajectory option: --typoed-option")

    proc = _run(
        args.fastsasa,
        "--topology",
        missing_topology,
        "--trajectory",
        missing_trajectory,
        "--summary",
        "--filter",
        "protein, protein",
    )
    _require_failure_with(proc, "failed to read topology")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        topology = tmp_path / "policy.pdb"
        mmcif_topology = tmp_path / "policy.cif"
        trajectory = tmp_path / "policy.dcd"
        topology.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      2  H   ALA A   1      10.000   0.000   0.000  1.00  0.00           H  \n"
            "HETATM    3  O   HOH W   2      20.000   0.000   0.000  1.00  0.00           O  \n"
            "END\n"
        )
        _write_mmcif(mmcif_topology)
        _write_dcd(trajectory, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
        base = (
            "trajectory",
            "--topology",
            str(topology),
            "--trajectory",
            str(trajectory),
            "--cpu",
            "--threads",
            "1",
            "--frames",
            "0",
            "--summary",
            "--shrake-rupley",
        )

        default_row = _summary_row(_run(args.fastsasa, *base))
        backend_cpu_base = tuple(value for value in base if value != "--cpu")
        backend_cpu_row = _summary_row(_run_cpu_backend(args.fastsasa, *backend_cpu_base))
        hydrogen_row = _summary_row(_run(args.fastsasa, *base, "--hydrogen"))
        hetatm_row = _summary_row(_run(args.fastsasa, *base, "--hetatm"))
        all_row = _summary_row(_run(args.fastsasa, *base, "-Y", "-H"))
        zero_probe_row = _summary_row(_run(args.fastsasa, *base, "-p", "0"))

        atom_counts = tuple(int(row["atoms"]) for row in (default_row, hydrogen_row, hetatm_row, all_row))
        if atom_counts != (1, 2, 2, 3):
            raise SystemExit(f"trajectory hydrogen/HETATM policy mismatch: {atom_counts}")
        if backend_cpu_row["total_sasa_sum"] != default_row["total_sasa_sum"]:
            raise SystemExit("FASTSASA_BACKEND=cpu trajectory result differs from --cpu")
        if float(zero_probe_row["total_sasa_sum"]) >= float(default_row["total_sasa_sum"]):
            raise SystemExit("trajectory --probe-radius did not change SASA as expected")

        mmcif_base = list(base)
        mmcif_base[2] = str(mmcif_topology)
        mmcif_default = _summary_row(_run(args.fastsasa, *mmcif_base))
        mmcif_all = _summary_row(_run(args.fastsasa, *mmcif_base, "--hydrogen", "--hetatm"))
        if (int(mmcif_default["atoms"]), int(mmcif_all["atoms"])) != (1, 3):
            raise SystemExit(
                "trajectory mmCIF atom-order policy mismatch: "
                f"{mmcif_default['atoms']}, {mmcif_all['atoms']}"
            )
        _validate_surface_points(args.fastsasa, tmp_path)
        _validate_surface_resolution(args.fastsasa, tmp_path)
        _validate_cpu_precision(args.fastsasa, tmp_path)
        _validate_residue_aliases(args.fastsasa, tmp_path)
        _validate_charmm_psf(args.fastsasa, tmp_path)
        _validate_cpu_selected_centers(args.fastsasa, tmp_path)
        _validate_automatic_backend(args.fastsasa, tmp_path)
        _validate_native_xtc(args.fastsasa, tmp_path)
        _validate_radii_without_source_tree(args.fastsasa, tmp_path)

    print("fastsasa_trajectory_cli_dispatch_validation,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
