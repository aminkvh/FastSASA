#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


class _FakeTrajectory:
    def __init__(self, universe, n_frames):
        self.universe = universe
        self.n_frames = n_frames
        self.frame = 0

    def __iter__(self):
        return iter(self[slice(None, None, None)])

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = range(self.n_frames)[item]
            return (_FakeTimestep(self, index) for index in indices)
        index = int(item)
        self.frame = index
        return _FakeTimestep(self, index)


class _FakeTimestep:
    def __init__(self, trajectory, frame):
        trajectory.frame = frame
        trajectory.universe._frame = frame
        self.frame = frame


class _FakeSelection:
    def __init__(self, universe, indices):
        self.universe = universe
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return int(self.indices.size)

    def __iter__(self):
        for index in self.indices:
            yield _FakeAtom(self, int(index))

    def select_atoms(self, select):
        selected = self.universe._select_indices(select)
        selected = [index for index in selected if index in set(self.indices.tolist())]
        return _FakeSelection(self.universe, selected)

    @property
    def positions(self):
        return self.universe._coords[self.universe._frame, self.indices, :]

    @property
    def radii(self):
        return self.universe._radii[self.indices]

    @property
    def resindices(self):
        return self.universe._resindices[self.indices]

    @property
    def elements(self):
        return self.universe._elements[self.indices]

    @property
    def resnames(self):
        return self.universe._resnames[self.indices]


class _FakeAtom:
    def __init__(self, selection, index):
        self.index = index
        self.name = f"C{index}"
        self.resname = selection.universe._resnames[index]


class _FakeUniverse:
    def __init__(self):
        n_frames = 3
        n_atoms = 125
        coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
        for frame in range(n_frames):
            coords[frame, :, 0] = np.arange(n_atoms, dtype=np.float64)
            coords[frame, :, 1] = frame
        self._coords = coords
        self._radii = np.full(n_atoms, 1.7, dtype=np.float64)
        self._resindices = np.repeat(np.arange(25, dtype=np.int64), 5)
        self._resnames = np.asarray(["ALA"] * n_atoms)
        self._elements = np.asarray(["C"] * n_atoms)
        self._frame = 0
        self.trajectory = _FakeTrajectory(self, n_frames)

    def select_atoms(self, select):
        return _FakeSelection(self, self._select_indices(select))

    def _select_indices(self, select):
        if select == "all":
            return list(range(125))
        if select == "index 0:9":
            return list(range(10))
        if select == "segindex 3:4":
            return list(range(75, 125))
        raise ValueError(f"unsupported fake selection: {select}")


class _FakeEngine:
    close_calls = 0

    def sasa(self, positions, radii, residue_ids=None, n_residues=None, **kwargs):
        # Mirror the real SasaEngine contract: (atoms, 3) or (frames, atoms, 3)
        # input, per-frame output rows.
        coords = np.asarray(positions, dtype=np.float64)
        if coords.ndim == 2:
            coords = coords[None, :, :]
        n_frames = coords.shape[0]
        n_atoms = int(coords.shape[1])
        residue = np.zeros((n_frames, int(n_residues)), dtype=np.float64)
        for residue_id in residue_ids:
            residue[:, int(residue_id)] += 1.0
        return {"total": np.full(n_frames, float(n_atoms), dtype=np.float64),
                "residue": residue}

    def close(self):
        type(self).close_calls += 1


class _FailingEngine(_FakeEngine):
    def __init__(self):
        self.calls = 0

    def sasa(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic frame failure")
        return super().sasa(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-dir", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.python_dir))

    from fastsasa_adapters import (
        RadiusConfig,
        SASAAnalysis,
        default_radius_config_path,
        load_radius_config,
        mdanalysis_residue_ids,
        mdanalysis_selection_arrays,
    )
    from fastsasa_native import selection_masks_from_metadata

    universe = _FakeUniverse()
    all_atoms = universe.select_atoms("all")
    positions, radii = mdanalysis_selection_arrays(all_atoms)
    if positions.shape != (125, 3) or radii.shape != (125,):
        raise SystemExit("array conversion shape mismatch")

    residue_ids, n_residues = mdanalysis_residue_ids(all_atoms)
    if residue_ids.shape != (125,) or n_residues != 25:
        raise SystemExit("residue id conversion mismatch")

    if len(SASAAnalysis(universe, select="all", engine_factory=_FakeEngine).atomgroup) != 125:
        raise SystemExit("default all selection size mismatch")
    if len(SASAAnalysis(universe, select="index 0:9", engine_factory=_FakeEngine).atomgroup) != 10:
        raise SystemExit("index selection size mismatch")
    if len(SASAAnalysis(universe, select="segindex 3:4", engine_factory=_FakeEngine).atomgroup) != 50:
        raise SystemExit("segment selection size mismatch")

    analysis = SASAAnalysis(universe, select="all", engine_factory=_FakeEngine).run(stop=3)
    if analysis.n_frames != 3:
        raise SystemExit("frame count mismatch")
    if analysis.results.total_area.dtype != np.float64 or analysis.results.total_area.shape != (3,):
        raise SystemExit("total_area shape or dtype mismatch")
    if analysis.results.residue_area.dtype != np.float64 or analysis.results.residue_area.shape != (3, 25):
        raise SystemExit("residue_area shape or dtype mismatch")
    if np.any(analysis.results.total_area < 0.0) or np.any(analysis.results.residue_area < 0.0):
        raise SystemExit("negative SASA value returned")

    unbatched = SASAAnalysis(
        universe, select="all", engine_factory=_FakeEngine, batch_frames=1
    ).run(stop=3)
    if not np.array_equal(unbatched.results.total_area, analysis.results.total_area):
        raise SystemExit("batched and unbatched total_area differ")
    if not np.array_equal(unbatched.results.residue_area, analysis.results.residue_area):
        raise SystemExit("batched and unbatched residue_area differ")

    before_close = _FailingEngine.close_calls
    try:
        SASAAnalysis(universe, select="all", engine_factory=_FailingEngine,
                     batch_frames=1).run(stop=3)
    except RuntimeError as exc:
        if "synthetic frame failure" not in str(exc):
            raise
    else:
        raise SystemExit("synthetic frame failure was not propagated")
    if _FailingEngine.close_calls != before_close + 1:
        raise SystemExit("analysis engine was not closed after a frame failure")

    config = RadiusConfig({"KNOWN": 1.7}, {("ALA", "CA"): "KNOWN"})
    if config.radius("ALA", "CA", "C") != 1.7:
        raise SystemExit("known configured radius mismatch")
    try:
        config.radius("UNK", "XX", "XX")
    except ValueError as exc:
        if "no radius" not in str(exc):
            raise
    else:
        raise SystemExit("unknown configured atom silently received a radius")

    default_path = default_radius_config_path()
    if default_path.name != "protor.config":
        raise SystemExit("default radius config lookup returned the wrong file")
    if not load_radius_config().type_radii:
        raise SystemExit("default radius config loaded no radius types")

    masks, names, warnings = selection_masks_from_metadata(
        ["icode, resi 1A", "segid AP and name CA", "warn, name ABCDE+CA+ZZ"],
        atom_names=["CA", "CB", "CA"],
        residue_names=["ALA", "ALA", "GLY"],
        residue_numbers=[1, 1, 2],
        residue_number_strings=["1", "1A", "2"],
        chain_ids=["A", "A", "B"],
        segment_ids=["AP", "AP", "BP"],
        elements=["C", "C", "C"],
    )
    if names != ["icode", "segid_AP_and_name_CA", "warn"]:
        raise SystemExit("selection name parsing mismatch")
    if masks.tolist() != [6, 1, 4]:
        raise SystemExit(f"selection mask mismatch: {masks.tolist()}")
    if len(warnings) != 2 or not all(warning.startswith("FastSASA: warning: selection:") for warning in warnings):
        raise SystemExit("selection warning count mismatch")

    print("fastsasa_python_mdanalysis_adapter_unit,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
