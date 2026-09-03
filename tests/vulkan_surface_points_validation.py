#!/usr/bin/env python3
"""Validate the Vulkan surface-point export kernel against the CPU reference.

fastsasa_cpu_exposed_points() and fastsasa_context_shrake_rupley_exposed_points_cell_list()
must produce the exact same per-point exposed/buried mask, not just the same
count - two different point sets can share a count. This compares the full
XYZ point set frame by frame, not the reported totals.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
from pathlib import Path


def _write_record(handle, payload: bytes) -> None:
    handle.write(struct.pack("<i", len(payload)))
    handle.write(payload)
    handle.write(struct.pack("<i", len(payload)))


def _write_dcd(path: Path, *frames: list[tuple[float, float, float]]) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into("<i", header, 4, len(frames))
    title = struct.pack("<i", 1) + b"FastSASA Vulkan surface-point validation".ljust(80)
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


def _run(fastsasa: Path, backend: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = backend
    return subprocess.run(
        [str(fastsasa), *args], env=environment, text=True, capture_output=True, check=False,
    )


def _parse_xyz_frames(text: str) -> list[set[str]]:
    # Standard XYZ layout: a point count, a "FastSASA ... frame N" comment,
    # then exactly that many "X x y z" lines (element symbol X, not an atom
    # index - unlike the plain-text --surface-points form).
    lines = [line for line in text.splitlines() if line.strip() != ""]
    frames: list[set[str]] = []
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor].strip())
        cursor += 1  # comment line
        cursor += 1
        points: set[str] = set()
        for _ in range(count):
            fields = lines[cursor].split()
            points.add(" ".join(fields[1:4]))
            cursor += 1
        frames.append(points)
    return frames


def _surface_points(fastsasa: Path, backend: str, topology: Path, trajectory: Path,
                     out_path: Path) -> list[set[str]]:
    proc = _run(
        fastsasa, backend, "trajectory",
        "--topology", str(topology), "--trajectory", str(trajectory),
        "--frames", ":", "--shrake-rupley", "--resolution", "260",
        "--surface-points", str(out_path),
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return _parse_xyz_frames(out_path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--allow-missing-device", action="store_true")
    args = parser.parse_args()

    topology = args.repo_root / "tests/data/1ubq.pdb"
    base_frame = _pdb_coordinates(topology)
    frames = [
        base_frame,
        [(x + 1.0, y, z) for x, y, z in base_frame],
        [(x, y + 1.0, z) for x, y, z in base_frame],
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        trajectory = tmp_path / "surface.dcd"
        _write_dcd(trajectory, *frames)

        cpu_frames = _surface_points(args.fastsasa, "cpu", topology, trajectory,
                                     tmp_path / "cpu.xyz")

        try:
            vk_frames = _surface_points(args.fastsasa, "vulkan", topology, trajectory,
                                        tmp_path / "vulkan.xyz")
        except RuntimeError as error:
            if args.allow_missing_device:
                print(f"fastsasa_vulkan_surface_points_validation,status,skip,detail,{error}")
                return 77
            raise

        if len(vk_frames) != len(cpu_frames):
            raise RuntimeError(
                f"Vulkan surface export produced {len(vk_frames)} frames, expected {len(cpu_frames)}"
            )
        for index, (cpu_points, vk_points) in enumerate(zip(cpu_frames, vk_frames)):
            if not cpu_points:
                raise RuntimeError(f"frame {index}: CPU surface export produced no points")
            if cpu_points != vk_points:
                only_cpu = sorted(cpu_points - vk_points)[:5]
                only_vk = sorted(vk_points - cpu_points)[:5]
                raise RuntimeError(
                    f"frame {index}: Vulkan and CPU surface-point sets differ "
                    f"({len(cpu_points)} vs {len(vk_points)} points); "
                    f"CPU-only sample {only_cpu}, Vulkan-only sample {only_vk}"
                )

        # A CUDA-backed context has no exposed-points kernel this round and
        # must fall back to the CPU path inside the CLI, not fail or silently
        # diverge. Tolerate a missing CUDA device the same way as Vulkan.
        try:
            cuda_frames = _surface_points(args.fastsasa, "cuda", topology, trajectory,
                                          tmp_path / "cuda.xyz")
        except RuntimeError as error:
            if args.allow_missing_device:
                cuda_frames = None
            else:
                raise
        if cuda_frames is not None and cuda_frames != cpu_frames:
            raise RuntimeError("CUDA-backed surface export (CPU fallback) diverged from the CPU backend")

    print("fastsasa_vulkan_surface_points_validation,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
