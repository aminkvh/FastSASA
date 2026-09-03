#!/usr/bin/env python3
from __future__ import annotations

import argparse

from rdkit import Chem

from fastsasa_adapters import sasa_rdkit_mol


def read_molecule(path):
    if path.lower().endswith(".sdf"):
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        mol = next((item for item in supplier if item is not None), None)
    else:
        mol = Chem.MolFromMolFile(path, removeHs=False)
    if mol is None:
        raise SystemExit(f"failed to read molecule: {path}")
    return mol


def main():
    parser = argparse.ArgumentParser(description="Calculate SASA for an RDKit molecule with FastSASA.")
    parser.add_argument("molecule", help="MOL or SDF file with 3D coordinates")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--smarts", action="append", help="optional SMARTS selection to reduce on the GPU")
    args = parser.parse_args()

    result = sasa_rdkit_mol(
        read_molecule(args.molecule),
        n_points=args.points,
        probe_radius=args.probe_radius,
        smarts=args.smarts,
    )
    if isinstance(result, dict):
        print("selection,total_sasa,selection_sasa")
        for name, value in zip(result["selection_names"], result["selection"][0]):
            print(f"{name},{result['total'][0]:.12f},{value:.12f}")
    else:
        print("total_sasa")
        print(f"{result[0]:.12f}")


if __name__ == "__main__":
    main()
