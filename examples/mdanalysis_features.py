#!/usr/bin/env python3
"""Extract ML-ready SASA features from an MDAnalysis trajectory."""
from __future__ import annotations

import argparse
import re

import numpy as np
import MDAnalysis as mda

from fastsasa import extract_md_features, interface_sasa, flatten_statistics
from fastsasa_adapters import load_radius_config, mdanalysis_residue_ids, mdanalysis_selection_arrays


def _mask(atoms, selection):
    if not selection:
        return None
    selected = set(atoms.universe.select_atoms(selection).indices.tolist())
    return np.asarray([atom.index in selected for atom in atoms], dtype=bool)


def _named_selection(spec):
    """Parse ``expression`` or ``name, expression`` like the native CLI."""
    if "," in spec:
        name, expression = (part.strip() for part in spec.split(",", 1))
        if not name or not expression:
            raise ValueError(f"invalid --group-selection {spec!r}")
        return name, expression
    expression = spec.strip()
    if not expression:
        raise ValueError("--group-selection must not be empty")
    name = re.sub(r"[^A-Za-z0-9]+", "_", expression).strip("_") or "selection"
    return name, expression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topology")
    parser.add_argument("trajectory")
    parser.add_argument("--selection", default="all")
    parser.add_argument("--group-selection", action="append", default=[], metavar="[NAME,] EXPR",
                         help="exposure group; optional name before a comma; repeatable")
    parser.add_argument("--hydrophobic-selection", help="optional hydrophobic atom group")
    parser.add_argument("--polar-selection", help="optional polar atom group")
    parser.add_argument("--interface-a-selection", help="first interface group")
    parser.add_argument("--interface-b-selection", help="second, non-overlapping interface group")
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
    _, radii = mdanalysis_selection_arrays(atoms, radius_config=radius_config)

    frames = []
    for _ in universe.trajectory[args.start:args.stop:args.step]:
        frames.append(np.asarray(atoms.positions, dtype=np.float64).copy())
    if not frames:
        raise ValueError("the requested trajectory slice contains no frames")
    positions = np.asarray(frames, dtype=np.float64)

    group_masks = {}
    for spec in args.group_selection:
        name, expr = _named_selection(spec)
        group_masks[name] = _mask(atoms, expr)

    residue_ids, n_residues = mdanalysis_residue_ids(atoms)
    features = extract_md_features(
        positions,
        radii,
        residue_ids=residue_ids,
        n_residues=n_residues,
        group_masks=group_masks or None,
        hydrophobic_mask=_mask(atoms, args.hydrophobic_selection),
        polar_mask=_mask(atoms, args.polar_selection),
        probe_radius=args.probe_radius,
        n_points=args.points,
    )

    print("feature,value")
    names, values = flatten_statistics(features["statistics"], prefix="sasa")
    for name, value in zip(names, values):
        print(f"{name},{value:.12g}")

    if bool(args.interface_a_selection) != bool(args.interface_b_selection):
        raise ValueError("--interface-a-selection and --interface-b-selection must be used together")
    if args.interface_a_selection:
        interface = interface_sasa(
            positions,
            radii,
            group_a_mask=_mask(atoms, args.interface_a_selection),
            group_b_mask=_mask(atoms, args.interface_b_selection),
            probe_radius=args.probe_radius,
            n_points=args.points,
        )
        for key, series in interface.items():
            print(f"interface.{key}.mean,{np.mean(series):.12g}")


if __name__ == "__main__":
    main()
