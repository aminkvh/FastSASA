"""Optional adapters from common chemistry/MD Python objects to FastSASA arrays.

These helpers do not make RDKit, MDAnalysis, or MDTraj mandatory dependencies.
They convert external objects to ``positions`` and ``radii`` arrays that can be
passed to ``fastsasa.sasa()`` or ``fastsasa.SasaEngine``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .fastsasa_native import SasaEngine, default_engine, sasa
except ImportError:
    from fastsasa_native import SasaEngine, default_engine, sasa


_VDW_RADII = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}


def _optional_topology_attribute(value, name):
    try:
        return getattr(value, name)
    except AttributeError:
        return None
    except Exception as exc:
        if exc.__class__.__name__ == "NoDataError":
            return None
        raise


# Common MD force-field residue names (protonation states, tautomers,
# disulfide-bonded cysteine) mapped to the standard residue they
# parameterize. Mirrors fastsasa_canonical_residue() in src/fastsasa_radius.c;
# tests/python_feature_unit.py checks the two tables agree.
RESIDUE_NAME_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG",
}


def canonical_residue_name(residue_name):
    """Return the standard residue name for an MD variant (``HIE`` ->
    ``HIS``), or the upper-cased input when it has no alias."""

    key = str(residue_name).strip().upper()
    return RESIDUE_NAME_ALIASES.get(key, key)


class RadiusConfig:
    """Simple residue/atom-name radius table loaded from a classifier config.

    Lookups try the exact residue/atom pair first, then the atom under the
    canonical residue name for MD variants (see ``RESIDUE_NAME_ALIASES``).
    """

    def __init__(self, type_radii, atom_types):
        self.type_radii = dict(type_radii)
        self.atom_types = dict(atom_types)

    def radius(self, residue_name, atom_name, element=None, default=None):
        key = (str(residue_name).strip().upper(), str(atom_name).strip().upper())
        atom_type = self.atom_types.get(key)
        if atom_type is None:
            canonical = canonical_residue_name(key[0])
            if canonical != key[0]:
                atom_type = self.atom_types.get((canonical, key[1]))

        if atom_type is not None and atom_type in self.type_radii:
            return self.type_radii[atom_type]
        if element is not None:
            value = element_radii([element], default=np.nan)[0]
            if np.isfinite(value):
                return float(value)
        if default is None:
            raise ValueError(
                f"no radius for residue {key[0]!r}, atom {key[1]!r}, "
                f"element {str(element).strip().upper()!r}"
            )
        return float(default)

    def radii(self, residue_names, atom_names, elements=None, default=None):
        if elements is None:
            elements = [None] * len(atom_names)
        values = [
            self.radius(residue_name, atom_name, element, default=default)
            for residue_name, atom_name, element in zip(residue_names, atom_names, elements)
        ]
        return np.asarray(values, dtype=np.float64)


def default_radius_config_path():
    """Return the installed or source-tree path to ``protor.config``."""

    candidates = []
    env_path = os.environ.get("FASTSASA_DEFAULT_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    module_path = Path(__file__).resolve()
    candidates.extend([
        module_path.parent / "share" / "fastsasa" / "protor.config",
        module_path.parents[1] / "share" / "protor.config",
        Path(sys.prefix) / "share" / "fastsasa" / "protor.config",
        Path("/usr/local/share/fastsasa/protor.config"),
        Path("/usr/share/fastsasa/protor.config"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not find the FastSASA radius configuration; pass a path to "
        "load_radius_config() or set FASTSASA_DEFAULT_CONFIG"
    )


def load_radius_config(path=None):
    """Load a ``types:``/``atoms:`` radius configuration file.

    With no argument, use the table installed with FastSASA or the table in a
    source checkout.
    """

    if path is None:
        path = default_radius_config_path()

    type_radii = {}
    atom_types = {}
    section = None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            lower = line.lower()
            if lower == "types:":
                section = "types"
                continue
            if lower == "atoms:":
                section = "atoms"
                continue
            if line.endswith(":"):
                section = None
                continue

            fields = line.split()
            if section == "types" and len(fields) >= 2:
                try:
                    type_radii[fields[0].upper()] = float(fields[1])
                except ValueError:
                    continue
            elif section == "atoms" and len(fields) >= 3:
                residue_name, atom_name, atom_type = fields[:3]
                atom_types[(residue_name.upper(), atom_name.upper())] = atom_type.upper()

    return RadiusConfig(type_radii, atom_types)


def element_radii(symbols, default=1.70):
    """Return fallback van der Waals radii for element symbols."""

    values = []
    for symbol in symbols:
        key = str(symbol).strip().upper()
        values.append(_VDW_RADII.get(key, default))
    return np.asarray(values, dtype=np.float64)


def _validated_radii(radii, n_atoms):
    values = np.asarray(radii, dtype=np.float64)

    if values.shape != (n_atoms,):
        raise ValueError(f"expected {n_atoms} radii, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("radii must be finite")
    if np.any(values <= 0.0):
        raise ValueError("radii must be positive")
    return np.ascontiguousarray(values)


def rdkit_conformer_arrays(mol, conf_id=-1, radii=None):
    """Return ``(positions, radii)`` arrays for one RDKit conformer."""

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit_conformer_arrays requires RDKit") from exc

    if mol is None:
        raise ValueError("mol must not be None")
    if mol.GetNumConformers() == 0:
        raise ValueError("mol has no conformer coordinates")

    conformer = mol.GetConformer(conf_id)
    positions = np.empty((mol.GetNumAtoms(), 3), dtype=np.float64)
    for atom_index in range(mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        positions[atom_index, 0] = point.x
        positions[atom_index, 1] = point.y
        positions[atom_index, 2] = point.z

    if radii is None:
        table = Chem.GetPeriodicTable()
        radii = [table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    return np.ascontiguousarray(positions), _validated_radii(radii, mol.GetNumAtoms())


def rdkit_smarts_masks(mol, smarts, start_bit=0):
    """Build FastSASA selection bit masks from RDKit SMARTS patterns."""

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("rdkit_smarts_masks requires RDKit") from exc

    if isinstance(smarts, str):
        smarts_items = [("selection", smarts)]
    else:
        smarts_items = []
        for index, item in enumerate(smarts):
            if isinstance(item, tuple):
                name, pattern = item
            else:
                name, pattern = f"selection_{index}", item
            smarts_items.append((str(name), str(pattern)))

    if len(smarts_items) > 31:
        raise ValueError("at most 31 SMARTS selections are supported")
    if start_bit < 0 or start_bit + len(smarts_items) > 31:
        raise ValueError("SMARTS selection bits must fit in 31 bits")

    masks = np.zeros(mol.GetNumAtoms(), dtype=np.uint32)
    names = []
    for offset, (name, pattern) in enumerate(smarts_items):
        query = Chem.MolFromSmarts(pattern)
        bit = np.uint32(1 << (start_bit + offset))

        if query is None:
            raise ValueError(f"invalid SMARTS selection: {pattern}")
        names.append(name)
        for match in mol.GetSubstructMatches(query):
            for atom_index in match:
                masks[atom_index] |= bit
    return np.ascontiguousarray(masks), names


def rdkit_residue_ids(mol):
    """Return residue ids from RDKit PDB residue metadata when available."""

    residue_ids = np.empty(mol.GetNumAtoms(), dtype=np.int32)
    residue_lookup = {}

    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            return None, 0
        key = (
            info.GetChainId().strip(),
            info.GetResidueNumber(),
            info.GetInsertionCode().strip(),
            info.GetResidueName().strip(),
        )
        if key not in residue_lookup:
            residue_lookup[key] = len(residue_lookup)
        residue_ids[atom.GetIdx()] = residue_lookup[key]
    return np.ascontiguousarray(residue_ids), len(residue_lookup)


def sasa_rdkit_mol(mol, conf_id=-1, radii=None, smarts=None, residue_sasa=False, **kwargs):
    """Compute FastSASA directly from an RDKit molecule/conformer."""

    positions, radii_array = rdkit_conformer_arrays(mol, conf_id=conf_id, radii=radii)
    selection_names = None

    if smarts is not None:
        if "selection_masks" in kwargs or "n_selections" in kwargs:
            raise ValueError("pass either smarts or explicit selection_masks, not both")
        selection_masks, selection_names = rdkit_smarts_masks(mol, smarts)
        kwargs["selection_masks"] = selection_masks
        kwargs["n_selections"] = len(selection_names)

    if residue_sasa:
        if "residue_ids" in kwargs or "n_residues" in kwargs:
            raise ValueError("pass either residue_sasa=True or explicit residue_ids, not both")
        residue_ids, n_residues = rdkit_residue_ids(mol)
        if residue_ids is None:
            raise ValueError("RDKit molecule does not contain PDB residue metadata")
        kwargs["residue_ids"] = residue_ids
        kwargs["n_residues"] = n_residues

    result = sasa(positions, radii_array, **kwargs)
    if selection_names is not None and isinstance(result, dict):
        result["selection_names"] = selection_names
    elif selection_names is not None and hasattr(result, "selection_names"):
        result.selection_names = selection_names
    return result


def mdanalysis_selection_arrays(selection, radius_config=None):
    """Return ``(positions, radii)`` arrays from an MDAnalysis AtomGroup."""

    positions = np.asarray(selection.positions, dtype=np.float64)
    radii = _optional_topology_attribute(selection, "radii")
    if radius_config is not None:
        residue_names = _optional_topology_attribute(selection, "resnames")
        if residue_names is None:
            residue_names = [atom.resname for atom in selection]
        atom_names = [atom.name for atom in selection]
        elements = _optional_topology_attribute(selection, "elements")
        if elements is None:
            elements = atom_names
        radii = radius_config.radii(residue_names, atom_names, elements)
    elif radii is None or len(radii) != len(selection):
        symbols = _optional_topology_attribute(selection, "elements")
        if symbols is None:
            symbols = [atom.name for atom in selection]
        radii = element_radii(symbols)
    return np.ascontiguousarray(positions), np.ascontiguousarray(radii, dtype=np.float64)


def mdanalysis_residue_ids(selection):
    """Return contiguous residue ids for an MDAnalysis AtomGroup.

    The ids are local to the selected atoms, so a filtered protein/domain can be
    aggregated without carrying unused residues from the full system.
    """

    residue_lookup = {}
    residue_ids = np.empty(len(selection), dtype=np.int32)
    source_ids = _optional_topology_attribute(selection, "resindices")
    if source_ids is None:
        source_ids = [atom.residue.ix for atom in selection]

    for atom_index, residue_index in enumerate(source_ids):
        residue_index = int(residue_index)
        if residue_index not in residue_lookup:
            residue_lookup[residue_index] = len(residue_lookup)
        residue_ids[atom_index] = residue_lookup[residue_index]

    return np.ascontiguousarray(residue_ids), len(residue_lookup)


def sasa_mdanalysis(
    universe_or_atomgroup,
    select="all",
    radius_config=None,
    engine=None,
    residue_sasa=False,
    **kwargs,
):
    """Compute FastSASA for the current MDAnalysis frame.

    ``select`` is evaluated by MDAnalysis before FastSASA sees the coordinates.
    This is the same scientific behavior as filtering atoms before a SASA
    calculation, not summing a post-hoc selection from a larger calculation.
    """

    atomgroup = universe_or_atomgroup.select_atoms(select)
    positions, radii = mdanalysis_selection_arrays(atomgroup, radius_config=radius_config)
    call_engine = engine if engine is not None else default_engine()

    if residue_sasa:
        residue_ids, n_residues = mdanalysis_residue_ids(atomgroup)
        kwargs.setdefault("residue_ids", residue_ids)
        kwargs.setdefault("n_residues", n_residues)

    return call_engine.sasa(positions, radii, **kwargs)


class SASAAnalysis:
    """MDAnalysis-compatible trajectory analysis object backed by FastSASA.

    The constructor follows the common MDAnalysis pattern used by existing
    SASA wrappers: pass a Universe or AtomGroup and a MDAnalysis selection
    string. Results are stored as ``results.total_area`` and
    ``results.residue_area`` for workflow compatibility.
    """

    def __init__(
        self,
        universe_or_atomgroup,
        select="all",
        radius_config=None,
        probe_radius=1.4,
        n_points=100,
        engine_factory=SasaEngine,
        batch_frames=16,
        **kwargs,
    ):
        self.atomgroup = universe_or_atomgroup.select_atoms(select)
        self.select = select
        self.radius_config = radius_config
        self.probe_radius = float(probe_radius)
        self.n_points = int(n_points)
        self.engine_factory = engine_factory
        if int(batch_frames) < 1:
            raise ValueError("batch_frames must be at least 1")
        self.batch_frames = int(batch_frames)
        self.kwargs = dict(kwargs)
        self.results = SimpleNamespace()
        self.n_frames = 0

    def run(self, start=None, stop=None, step=None, frames=None):
        """Run the analysis over the selected trajectory frames."""

        if frames is not None and any(value is not None for value in (start, stop, step)):
            raise ValueError("pass either frames or start/stop/step, not both")

        trajectory = self.atomgroup.universe.trajectory
        if frames is None:
            frame_iterable = trajectory[slice(start, stop, step)]
        else:
            frame_iterable = (trajectory[int(frame)] for frame in frames)

        frame_indices = []
        self._prepare()
        try:
            total_values = []
            residue_values = []
            # Frames are gathered into fixed-size coordinate batches so one
            # native call covers batch_frames frames; this removes per-frame
            # call and synchronization overhead without changing results.
            n_atoms = len(self.atomgroup)
            chunk = np.empty((self.batch_frames, n_atoms, 3), dtype=np.float64)
            filled = 0

            def flush():
                nonlocal filled
                if filled == 0:
                    return
                result = self._engine.sasa(
                    chunk[:filled],
                    self._radii,
                    probe_radius=self.probe_radius,
                    n_points=self.n_points,
                    residue_ids=self._residue_ids,
                    n_residues=self._n_residues,
                    **self.kwargs,
                )
                total_values.extend(float(v) for v in result["total"])
                residue_values.extend(
                    np.asarray(row, dtype=np.float64) for row in result["residue"]
                )
                filled = 0

            for frame_index, _ts in enumerate(frame_iterable):
                self._frame_index = frame_index
                chunk[filled] = self.atomgroup.positions
                filled += 1
                frame_indices.append(int(trajectory.frame))
                if filled == self.batch_frames:
                    flush()
            flush()

            self.n_frames = len(total_values)
            self.frames = np.asarray(frame_indices, dtype=np.int64)
            self.results.total_area = np.asarray(total_values, dtype=np.float64)
            if residue_values:
                self.results.residue_area = np.asarray(residue_values, dtype=np.float64)
            else:
                self.results.residue_area = np.zeros((0, self._n_residues), dtype=np.float64)
            self._conclude()
            return self
        finally:
            if hasattr(self._engine, "close"):
                self._engine.close()

    def _prepare(self):
        _, self._radii = mdanalysis_selection_arrays(self.atomgroup, radius_config=self.radius_config)
        self._residue_ids, self._n_residues = mdanalysis_residue_ids(self.atomgroup)
        self._engine = self.engine_factory()

    def _single_frame(self):
        positions = np.ascontiguousarray(np.asarray(self.atomgroup.positions, dtype=np.float64))
        result = self._engine.sasa(
            positions,
            self._radii,
            probe_radius=self.probe_radius,
            n_points=self.n_points,
            residue_ids=self._residue_ids,
            n_residues=self._n_residues,
            **self.kwargs,
        )
        return float(result["total"][0]), np.asarray(result["residue"][0], dtype=np.float64)

    def _conclude(self):
        self.results.mean_total_area = float(np.mean(self.results.total_area))
        self.results.total_sasa = self.results.total_area
        self.results.residue_sasa = self.results.residue_area


def mdtraj_frame_arrays(trajectory, frame_index=0):
    """Return ``(positions, radii)`` arrays for one MDTraj frame.

    MDTraj stores coordinates in nanometers; returned positions are converted to
    Angstrom.
    """

    frame = trajectory[frame_index]
    positions = np.asarray(frame.xyz[0], dtype=np.float64) * 10.0
    symbols = [atom.element.symbol for atom in frame.topology.atoms]
    return np.ascontiguousarray(positions), element_radii(symbols)
