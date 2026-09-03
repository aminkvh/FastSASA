#!/usr/bin/env python3
from __future__ import annotations

import argparse

import mdtraj as md

from fastsasa_adapters import SasaEngine, mdtraj_frame_arrays


def main():
    parser = argparse.ArgumentParser(description="Calculate per-frame SASA from MDTraj with FastSASA.")
    parser.add_argument("trajectory")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    args = parser.parse_args()

    traj = md.load(args.trajectory, top=args.topology, stride=args.stride)

    print("frame,total_sasa")
    with SasaEngine() as engine:
        for frame_index in range(traj.n_frames):
            positions, radii = mdtraj_frame_arrays(traj, frame_index)
            total = engine.sasa(
                positions,
                radii,
                probe_radius=args.probe_radius,
                n_points=args.points,
            )[0]
            print(f"{frame_index},{total:.12f}")


if __name__ == "__main__":
    main()
