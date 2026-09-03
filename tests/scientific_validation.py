#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


STRUCTURES = [
    ("large_protein", "tests/data/2isk.pdb", [], 13928, 1752, 0, 61277.824015612583),
    ("nucleic_acid_or_cofactor_heavy", "tests/data/1sui.pdb", ["--hetatm"], 7446, 983, 0, 43579.761754526357),
    ("glycan_or_sugar", "tests/data/5dx9.pdb", ["--hetatm", "--select", "glc, resn GLC"], 2540, 407, 1, 14264.580728034009),
    ("ligand_heavy", "tests/data/3bkr.pdb", ["--hetatm", "--select", "plm, resn PLM"], 1135, 288, 1, 7265.254396354858),
    ("alternate_locations", "tests/external_mmcif/1EN2.cif", [], 614, 85, 0, 4644.496800662938),
    ("multi_model_mmcif", "tests/external_mmcif/2K39.cif", ["--join-models"], 69832, 8816, 0, 7031.280501001203),
    ("unusual_mmcif_metadata", "tests/data/7cma-assembly1.cif", ["--hetatm"], 2584, 330, 0, 17862.263997408743),
]

def _run(command, cwd):
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _structure_rows(fastsasa, repo_root):
    rows = []
    for label, path, extra, expected_atoms, expected_residues, expected_selections, expected_total in STRUCTURES:
        if not (repo_root / path).exists():
            rows.append({
                "kind": "structure",
                "label": label,
                "path": path,
                "status": "skip",
                "detail": "optional fixture is not included in this source package",
            })
            continue
        command = [str(fastsasa), "--shrake-rupley", "--format", "json", *extra, path]
        proc = _run(command, repo_root)
        row = {"kind": "structure", "label": label, "path": path, "status": "pass" if proc.returncode == 0 else "fail"}
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            row["atoms"] = str(len(data.get("atoms", [])))
            row["residues"] = str(len(data.get("residues", [])))
            row["total_sasa"] = f"{float(data['total_sasa']):.12f}"
            row["selections"] = str(len(data.get("selections", [])))
            actual_total = float(data["total_sasa"])
            actual_shape = (int(row["atoms"]), int(row["residues"]), int(row["selections"]))
            expected_shape = (expected_atoms, expected_residues, expected_selections)
            if actual_shape != expected_shape:
                row["status"] = "fail"
                row["detail"] = f"shape {actual_shape} != expected {expected_shape}"
            elif not math.isclose(actual_total, expected_total, rel_tol=2e-5, abs_tol=0.02):
                row["status"] = "fail"
                row["detail"] = f"total {actual_total:.12f} != expected {expected_total:.12f}"
        else:
            row["detail"] = (proc.stderr + proc.stdout).strip().replace("\n", " | ")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run broader FastSASA scientific validation fixtures.")
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    rows = _structure_rows(args.fastsasa, args.repo_root)

    fieldnames = ["kind", "label", "path", "status", "atoms", "residues", "frames", "known_frames", "total_sasa", "selections", "gpu_frames_per_second", "detail"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    failed = [row for row in rows if row["status"] == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
