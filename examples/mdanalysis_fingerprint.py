#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import MDAnalysis as mda

from fastsasa import embed_sasa_fingerprints, extract_md_features, sasa_fingerprint_matrix
from fastsasa_adapters import load_radius_config, mdanalysis_residue_ids, mdanalysis_selection_arrays


def main():
    parser = argparse.ArgumentParser(description="Create SASA fingerprints and PCA-style embeddings from an MD trajectory.")
    parser.add_argument("topology")
    parser.add_argument("trajectory")
    parser.add_argument("--selection", default="protein")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--normalize", choices=("none", "zscore", "minmax"), default="zscore")
    parser.add_argument("--config-file")
    args = parser.parse_args()

    universe = mda.Universe(args.topology, args.trajectory)
    atoms = universe.select_atoms(args.selection)
    radius_config = load_radius_config(args.config_file) if args.config_file else None
    _, radii = mdanalysis_selection_arrays(atoms, radius_config=radius_config)

    frames = []
    for _ in universe.trajectory[args.start:args.stop:args.step]:
        frames.append(np.asarray(atoms.positions, dtype=np.float64).copy())
    if not frames:
        raise ValueError("the requested trajectory slice contains no frames")

    residue_ids, n_residues = mdanalysis_residue_ids(atoms)
    features = extract_md_features(
        np.asarray(frames, dtype=np.float64),
        radii,
        residue_ids=residue_ids,
        n_residues=n_residues,
        probe_radius=args.probe_radius,
        n_points=args.points,
    )
    fingerprints = sasa_fingerprint_matrix(features, normalize=args.normalize)
    embedding = embed_sasa_fingerprints(fingerprints["matrix"], n_components=args.components)

    print("frame," + ",".join(f"sasa_pc{i + 1}" for i in range(embedding["embedding"].shape[1])))
    for frame, row in enumerate(embedding["embedding"]):
        print(f"{frame}," + ",".join(f"{value:.12g}" for value in row))


if __name__ == "__main__":
    main()
