#!/usr/bin/env python3
"""Compute SASA for a PyMOL selection using FastSASA's array API."""
from __future__ import annotations

import argparse

import numpy as np
import pymol2

from fastsasa_adapters import SasaEngine, load_radius_config


def state_arrays(cmd, selection, state, radius_config):
    model = cmd.get_model(selection, state=state)
    positions = np.empty((len(model.atom), 3), dtype=np.float64)
    radii = np.empty(len(model.atom), dtype=np.float64)
    for i, atom in enumerate(model.atom):
        positions[i] = atom.coord
        radii[i] = radius_config.radius(atom.resn, atom.name, element=atom.symbol)
    return positions, radii


def main():
    parser = argparse.ArgumentParser(description="Compute SASA for a PyMOL selection.")
    parser.add_argument("structure")
    parser.add_argument("--selection", default="polymer")
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--config-file", help="optional classifier radii config file")
    args = parser.parse_args()

    radius_config = load_radius_config(args.config_file)

    with pymol2.PyMOL() as session:
        cmd = session.cmd
        cmd.load(args.structure, "fastsasa_target")
        selection = f"fastsasa_target and ({args.selection})"
        n_states = cmd.count_states("fastsasa_target")

        print("state,total_sasa")
        with SasaEngine() as engine:
            for state in range(1, n_states + 1):
                positions, radii = state_arrays(cmd, selection, state, radius_config)
                total = engine.sasa(
                    np.expand_dims(positions, axis=0),
                    radii,
                    probe_radius=args.probe_radius,
                    n_points=args.points,
                )[0]
                print(f"{state},{total:.12f}")


if __name__ == "__main__":
    main()
