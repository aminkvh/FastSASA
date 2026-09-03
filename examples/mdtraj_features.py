#!/usr/bin/env python3
"""Extract ML-ready SASA features from an MDTraj trajectory."""
from __future__ import annotations

import argparse
import re

import numpy as np
import mdtraj as md

from fastsasa import extract_md_features, interface_sasa, flatten_statistics
from fastsasa_adapters import element_radii


def _residue_ids(topology):
    residues = list(topology.residues)
    residue_lookup = {residue: index for index, residue in enumerate(residues)}
    return np.asarray([residue_lookup[atom.residue] for atom in topology.atoms], dtype=np.int32), len(residues)


def _mask(topology, selection):
    if not selection:
        return None
    indices = set(topology.select(selection).tolist())
    return np.asarray([atom.index in indices for atom in topology.atoms], dtype=bool)


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
    parser.add_argument("trajectory")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--group-selection", action="append", default=[], metavar="[NAME,] EXPR",
                         help="exposure group; optional name before a comma; repeatable")
    parser.add_argument("--hydrophobic-selection", help="optional hydrophobic atom group")
    parser.add_argument("--polar-selection", help="optional polar atom group")
    parser.add_argument("--interface-a-selection", help="first interface group")
    parser.add_argument("--interface-b-selection", help="second, non-overlapping interface group")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    args = parser.parse_args()

    traj = md.load(args.trajectory, top=args.topology, stride=args.stride)
    positions = np.ascontiguousarray(traj.xyz, dtype=np.float64) * 10.0
    radii = element_radii([atom.element.symbol for atom in traj.topology.atoms])
    residue_ids, n_residues = _residue_ids(traj.topology)

    group_masks = {}
    for spec in args.group_selection:
        name, expr = _named_selection(spec)
        group_masks[name] = _mask(traj.topology, expr)

    features = extract_md_features(
        positions,
        radii,
        residue_ids=residue_ids,
        n_residues=n_residues,
        group_masks=group_masks or None,
        hydrophobic_mask=_mask(traj.topology, args.hydrophobic_selection),
        polar_mask=_mask(traj.topology, args.polar_selection),
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
            group_a_mask=_mask(traj.topology, args.interface_a_selection),
            group_b_mask=_mask(traj.topology, args.interface_b_selection),
            probe_radius=args.probe_radius,
            n_points=args.points,
        )
        for key, series in interface.items():
            print(f"interface.{key}.mean,{np.mean(series):.12g}")


if __name__ == "__main__":
    main()
