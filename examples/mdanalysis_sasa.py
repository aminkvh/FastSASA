#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import MDAnalysis as mda

from fastsasa_adapters import SasaEngine, load_radius_config, mdanalysis_selection_arrays


def main():
    parser = argparse.ArgumentParser(description="Stream an MDAnalysis trajectory into FastSASA.")
    parser.add_argument("topology")
    parser.add_argument("trajectory")
    parser.add_argument("--selection", default="all")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--config-file", help="optional classifier radii config file")
    args = parser.parse_args()

    universe = mda.Universe(args.topology, args.trajectory)
    atoms = universe.select_atoms(args.selection)
    radius_config = load_radius_config(args.config_file) if args.config_file else None

    # Optional: apply any MDAnalysis transformations your workflow wants before
    # this loop. FastSASA consumes the coordinates as provided and does not alter
    # imaging.

    print("frame,total_sasa")
    with SasaEngine() as engine:
        for ts in universe.trajectory[args.start:args.stop:args.step]:
            positions, radii = mdanalysis_selection_arrays(atoms, radius_config=radius_config)
            total = engine.sasa(
                np.expand_dims(positions, axis=0),
                radii,
                probe_radius=args.probe_radius,
                n_points=args.points,
            )[0]
            print(f"{ts.frame},{total:.12f}")


if __name__ == "__main__":
    main()
