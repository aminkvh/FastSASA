"""ctypes bindings for the native FastSASA accelerator library.

This module is intentionally format-neutral: callers pass NumPy-compatible
coordinate and radius arrays, and optional residue/selection masks. Use
``fastsasa_adapters`` for MDAnalysis, MDTraj, or RDKit convenience conversion.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np


FASTSASA_SUCCESS = 0
_FASTSASA_ABI_VERSION = 1


class SasaResult:
    """Container with stable FastSASA result shapes.

    Attributes:
        total: Total SASA per frame, shape ``(frames,)``.
        atom: Optional atom SASA, shape ``(frames, atoms)``.
        residue: Optional residue SASA, shape ``(frames, residues)``.
        selection: Optional selection SASA, shape ``(frames, selections)``.
        selection_names: Optional names corresponding to ``selection`` columns.
    """

    def __init__(self, total, atom=None, residue=None, selection=None, selection_names=None):
        self.total = np.asarray(total, dtype=np.float64)
        self.atom = None if atom is None else np.asarray(atom, dtype=np.float64)
        self.residue = None if residue is None else np.asarray(residue, dtype=np.float64)
        self.selection = None if selection is None else np.asarray(selection, dtype=np.float64)
        self.selection_names = list(selection_names) if selection_names is not None else None

    def totalArea(self, frame=0):
        return float(self.total[frame])

    def atomArea(self, atom, frame=0):
        if self.atom is None:
            raise ValueError("atom SASA was not requested")
        return float(self.atom[frame, atom])

    def atomAreas(self, frame=0):
        if self.atom is None:
            raise ValueError("atom SASA was not requested")
        return self.atom[frame].copy()

    def residueAreas(self, frame=0):
        if self.residue is None:
            raise ValueError("residue SASA was not requested")
        return self.residue[frame].copy()

    def selectionAreas(self, frame=0):
        if self.selection is None:
            raise ValueError("selection SASA was not requested")
        return self.selection[frame].copy()


class _SrInput(ctypes.Structure):
    _fields_ = [
        ("n_atoms", ctypes.c_int),
        ("n_points", ctypes.c_int),
        ("xyz", ctypes.POINTER(ctypes.c_double)),
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("y", ctypes.POINTER(ctypes.c_double)),
        ("z", ctypes.POINTER(ctypes.c_double)),
        ("radii", ctypes.POINTER(ctypes.c_double)),
        ("test_points", ctypes.POINTER(ctypes.c_double)),
        ("neighbor_offsets", ctypes.POINTER(ctypes.c_int)),
        ("neighbor_indices", ctypes.POINTER(ctypes.c_int)),
        ("n_neighbor_indices", ctypes.c_int),
        ("reuse_test_points", ctypes.c_int),
        ("residue_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_residues", ctypes.c_int),
        ("residue_sasa", ctypes.POINTER(ctypes.c_double)),
        ("selection_masks", ctypes.POINTER(ctypes.c_uint)),
        ("n_selections", ctypes.c_int),
        ("selection_sasa", ctypes.POINTER(ctypes.c_double)),
        ("active_center_mask", ctypes.c_uint),
        ("active_center_indices", ctypes.POINTER(ctypes.c_int)),
        ("n_active_centers", ctypes.c_int),
        ("force_double_precision", ctypes.c_int),
    ]


class _TrajectoryTopology(ctypes.Structure):
    _fields_ = [
        ("radii", ctypes.POINTER(ctypes.c_double)),
        ("residue_ids", ctypes.POINTER(ctypes.c_int)),
        ("n_atoms", ctypes.c_int),
        ("n_residues", ctypes.c_int),
    ]


class _SoAFrames(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("y", ctypes.POINTER(ctypes.c_double)),
        ("z", ctypes.POINTER(ctypes.c_double)),
        ("n_frames", ctypes.c_int),
    ]


class _TrajectoryParameters(ctypes.Structure):
    _fields_ = [
        ("probe_radius", ctypes.c_double),
        ("n_points", ctypes.c_int),
        ("algorithm", ctypes.c_int),
        ("precision", ctypes.c_int),
    ]


class _OwnedTopology(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.POINTER(ctypes.c_double)),
        ("y", ctypes.POINTER(ctypes.c_double)),
        ("z", ctypes.POINTER(ctypes.c_double)),
        ("radii", ctypes.POINTER(ctypes.c_double)),
        ("residue_ids", ctypes.POINTER(ctypes.c_int)),
        ("residue_numbers", ctypes.POINTER(ctypes.c_int)),
        ("residue_number_strings", ctypes.POINTER(ctypes.c_char_p)),
        ("atom_names", ctypes.POINTER(ctypes.c_char_p)),
        ("residue_names", ctypes.POINTER(ctypes.c_char_p)),
        ("chain_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("segment_ids", ctypes.POINTER(ctypes.c_char_p)),
        ("elements", ctypes.POINTER(ctypes.c_char_p)),
        ("n_atoms", ctypes.c_int),
        ("n_residues", ctypes.c_int),
        ("atom_flags", ctypes.POINTER(ctypes.c_ubyte)),
    ]


_SelectionWarningCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def _null_ptr(ctype):
    return ctypes.POINTER(ctype)()


def _as_double_array(values, shape=None):
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array)


def _as_double_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _as_int_array(values, shape=None):
    array = np.asarray(values, dtype=np.int32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array)


def _as_uint_array(values, shape=None):
    array = np.asarray(values, dtype=np.uint32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array)


def _as_int_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _as_uint_ptr(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))


def _as_char_pointer_array(values, n_atoms, name):
    if values is None:
        values = [""] * n_atoms
    if len(values) != n_atoms:
        raise ValueError(f"{name} must contain {n_atoms} values")
    encoded = [str(value).encode("utf-8") for value in values]
    return encoded, (ctypes.c_char_p * n_atoms)(*encoded)


def _library_candidates():
    env_path = os.environ.get("FASTSASA_NATIVE_LIBRARY")
    if env_path:
        yield Path(env_path)

    here = Path(__file__).resolve()
    roots = [
        here.parent,
        here.parent / "lib",
        here.parents[1] / "build",
        here.parents[1] / "build" / "lib",
        here.parents[1] / "build-cpu",
        here.parents[1] / "build-cpu" / "lib",
        here.parents[1] / "lib",
        Path.cwd() / "build",
        Path.cwd() / "build-cpu",
        Path.cwd(),
    ]
    if len(here.parents) > 3:
        roots.append(here.parents[3] / "lib")
    if sys.platform == "darwin":
        library_name = "libfastsasa_native.dylib"
    elif sys.platform == "win32":
        library_name = "fastsasa_native.dll"
    else:
        library_name = "libfastsasa_native.so"
    for root in roots:
        yield root / library_name


@lru_cache(maxsize=1)
def _load_library():
    errors = []
    for path in _library_candidates():
        try:
            library = ctypes.CDLL(str(path))
            _verify_native_abi(library)
            return library
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError(
        "could not load the FastSASA native library; build FastSASA or set "
        "FASTSASA_NATIVE_LIBRARY to its full path\n" + "\n".join(errors)
    )


def _verify_native_abi(library):
    checks = [
        ("fastsasa_sizeof_sr_input", ctypes.sizeof(_SrInput)),
        ("fastsasa_offsetof_sr_input_active_center_mask", _SrInput.active_center_mask.offset),
        ("fastsasa_offsetof_sr_input_active_center_indices", _SrInput.active_center_indices.offset),
        ("fastsasa_offsetof_sr_input_n_active_centers", _SrInput.n_active_centers.offset),
        ("fastsasa_offsetof_sr_input_force_double_precision", _SrInput.force_double_precision.offset),
        ("fastsasa_sizeof_owned_topology", ctypes.sizeof(_OwnedTopology)),
        ("fastsasa_offsetof_owned_topology_atom_flags", _OwnedTopology.atom_flags.offset),
    ]

    try:
        library.fastsasa_abi_version.argtypes = []
        library.fastsasa_abi_version.restype = ctypes.c_uint
        version = int(library.fastsasa_abi_version())
        if version != _FASTSASA_ABI_VERSION:
            raise RuntimeError(
                f"FastSASA native ABI version {version} does not match Python ABI "
                f"version {_FASTSASA_ABI_VERSION}"
            )
        for function_name, expected in checks:
            function = getattr(library, function_name)
            function.argtypes = []
            function.restype = ctypes.c_size_t
            actual = int(function())
            if actual != expected:
                raise RuntimeError(
                    f"FastSASA native ABI layout mismatch for {function_name}: "
                    f"native={actual}, python={expected}"
                )
    except AttributeError as exc:
        raise RuntimeError(
            "libfastsasa_native.so is incompatible with this Python package; "
            "rebuild or reinstall FastSASA so the Python and native versions match"
        ) from exc


@lru_cache(maxsize=32)
def _cached_fibonacci_sphere_points(n_points):
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    dz = 2.0 / n_points
    increment = np.pi * (3.0 - np.sqrt(5.0))
    index = np.arange(n_points, dtype=np.float64)
    z = 1.0 - dz / 2.0 - index * dz
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    longitude = index * increment
    points = np.column_stack(
        (np.cos(longitude) * radius, np.sin(longitude) * radius, z)
    )
    flat_points = np.ascontiguousarray(points.reshape(-1))
    flat_points.setflags(write=False)
    return flat_points


def fibonacci_sphere_points(n_points):
    """Return deterministic Shrake-Rupley unit-sphere points as a flat array.

    The returned array has shape ``(3 * n_points,)`` and dtype ``float64``.
    """

    return _cached_fibonacci_sphere_points(n_points).copy()


def selection_masks_from_metadata(
    commands,
    atom_names,
    residue_names,
    residue_numbers,
    chain_ids,
    elements,
    residue_number_strings=None,
    segment_ids=None,
):
    """Build FastSASA selection bit masks from selector-expression commands.

    Args:
        commands: One command or a sequence of commands in ``expression`` or
            ``name, expression`` form. Expression-only commands get a generated
            name by replacing non-alphanumeric runs with ``_``. At most 31
            commands are supported.
        atom_names, residue_names, residue_numbers, chain_ids, elements:
            Per-atom metadata arrays.
        residue_number_strings: Optional per-atom residue labels, including
            insertion codes such as ``"125A"``. Defaults to ``residue_numbers``.
        segment_ids: Optional PSF-style segment ids for FastSASA ``segid`` and
            ``segname`` selections.

    Returns:
        ``(masks, names, warnings)`` where ``masks`` has dtype ``uint32`` and
        shape ``(atoms,)``. Bit ``1 << i`` marks command ``i``.
    """

    if isinstance(commands, str):
        command_list = [commands]
    else:
        command_list = [str(command) for command in commands]
    if not command_list:
        raise ValueError("at least one selection command is required")
    if len(command_list) > 31:
        raise ValueError("at most 31 selection commands are supported")

    n_atoms = len(atom_names)
    if n_atoms == 0:
        raise ValueError("metadata must contain at least one atom")
    if len(residue_names) != n_atoms or len(residue_numbers) != n_atoms or \
            len(chain_ids) != n_atoms or len(elements) != n_atoms:
        raise ValueError("metadata arrays must have the same length")
    if residue_number_strings is None:
        residue_number_strings = [str(value) for value in residue_numbers]

    dummy = np.zeros(n_atoms, dtype=np.float64)
    residue_ids = np.arange(n_atoms, dtype=np.int32)
    residue_numbers_array = _as_int_array(residue_numbers, (n_atoms,))
    masks = np.zeros(n_atoms, dtype=np.uint32)

    atom_bytes, atom_array = _as_char_pointer_array(atom_names, n_atoms, "atom_names")
    residue_bytes, residue_array = _as_char_pointer_array(residue_names, n_atoms, "residue_names")
    residue_number_bytes, residue_number_array = _as_char_pointer_array(
        residue_number_strings, n_atoms, "residue_number_strings"
    )
    chain_bytes, chain_array = _as_char_pointer_array(chain_ids, n_atoms, "chain_ids")
    segment_bytes, segment_array = _as_char_pointer_array(segment_ids, n_atoms, "segment_ids")
    element_bytes, element_array = _as_char_pointer_array(elements, n_atoms, "elements")

    topology = _OwnedTopology()
    topology.x = _as_double_ptr(dummy)
    topology.y = _as_double_ptr(dummy)
    topology.z = _as_double_ptr(dummy)
    topology.radii = _as_double_ptr(dummy)
    topology.residue_ids = _as_int_ptr(residue_ids)
    topology.residue_numbers = _as_int_ptr(residue_numbers_array)
    topology.residue_number_strings = residue_number_array
    topology.atom_names = atom_array
    topology.residue_names = residue_array
    topology.chain_ids = chain_array
    topology.segment_ids = segment_array
    topology.elements = element_array
    topology.n_atoms = n_atoms
    topology.n_residues = n_atoms

    lib = _load_library()
    lib.fastsasa_topology_selection_mask_ex.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_OwnedTopology),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_char_p,
        ctypes.c_size_t,
        _SelectionWarningCallback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.fastsasa_topology_selection_mask_ex.restype = ctypes.c_int

    warnings = []

    @_SelectionWarningCallback
    def collect_warning(message, userdata):
        del userdata
        warnings.append(message.decode("utf-8"))

    names = []
    for index, command in enumerate(command_list):
        name_buffer = ctypes.create_string_buffer(51)
        warning_count = ctypes.c_int(0)
        ok = lib.fastsasa_topology_selection_mask_ex(
            command.encode("utf-8"),
            ctypes.byref(topology),
            ctypes.c_uint(1 << index),
            _as_uint_ptr(masks),
            name_buffer,
            ctypes.sizeof(name_buffer),
            collect_warning,
            None,
            ctypes.byref(warning_count),
        )
        if not ok:
            raise ValueError(f"invalid selection command: {command}")
        names.append(name_buffer.value.decode("utf-8"))

    return np.ascontiguousarray(masks), names, warnings


def _format_result(totals, atom_values, residue_values, selection_values, as_result):
    if as_result:
        return SasaResult(totals, atom_values, residue_values, selection_values)

    if residue_values is not None or selection_values is not None:
        result = {"total": totals}
        if atom_values is not None:
            result["atom"] = atom_values
        if residue_values is not None:
            result["residue"] = residue_values
        if selection_values is not None:
            result["selection"] = selection_values
        return result

    return (totals, atom_values) if atom_values is not None else totals


def _empty_result(n_frames, n_atoms, atom_sasa, residue_ids, n_residues,
                  selection_masks, n_selections, as_result):
    totals = np.zeros(n_frames, dtype=np.float64)
    atom_values = np.zeros((n_frames, n_atoms), dtype=np.float64) if atom_sasa else None
    residue_values = None
    selection_values = None

    if residue_ids is not None or n_residues is not None:
        n_residues = 0 if n_residues is None else int(n_residues)
        if n_residues < 0:
            raise ValueError("n_residues must be non-negative")
        residue_values = np.zeros((n_frames, n_residues), dtype=np.float64)
    if selection_masks is not None or n_selections is not None:
        n_selections = 1 if n_selections is None else int(n_selections)
        if n_selections < 0 or n_selections > 31:
            raise ValueError("n_selections must be between 0 and 31")
        selection_values = np.zeros((n_frames, n_selections), dtype=np.float64)
    return _format_result(totals, atom_values, residue_values, selection_values, as_result)


def _precision_value(precision):
    value = str(precision).lower()
    if value in ("fp64", "double"):
        return 0, "fp64"
    if value in ("fp32", "float"):
        return 1, "fp32"
    raise ValueError("precision must be 'fp64' or 'fp32'")


class SasaEngine:
    """Reusable FastSASA accelerator context.

    Create one ``SasaEngine`` per worker and reuse it for many structures or
    trajectory batches. The context keeps CUDA or Vulkan buffers and cached SR
    test points alive between calls. For long file-backed DCD/XTC trajectories,
    the native ``fastsasa trajectory`` CLI avoids Python-loop overhead and is
    usually faster. ``precision`` is ``"fp64"`` by default; use ``"fp32"``
    for faster reduced-precision kernels.

    Backend selection follows the native Vulkan -> CUDA -> threaded CPU
    order and honors ``FASTSASA_BACKEND=auto|vulkan|cuda|cpu``. When no usable
    GPU exists (or a Vulkan device lacks FP64 for the default precision), the
    engine transparently uses the native threaded CPU backend, honoring the
    requested precision, and reports ``engine.backend == "cpu"``. CPU
    Lee-Richards is fp64-only regardless of ``precision``; requesting fp32
    with ``lee_richards()`` on the CPU backend emits a ``RuntimeWarning`` and
    computes at fp64.
    """

    def __init__(self, library_path=None, precision="fp64"):
        precision_value, self.precision = _precision_value(precision)
        self._lib = ctypes.CDLL(str(library_path)) if library_path else _load_library()
        self._ctx = ctypes.c_void_p()
        self._closed = False
        self._configure_abi()
        request = os.environ.get("FASTSASA_BACKEND", "").lower() or "auto"
        if request == "cpu":
            self._ctx = None
            self.backend = "cpu"
            # self.precision keeps the caller's requested value: the CPU
            # backend supports both fp64 and fp32 for Shrake-Rupley
            # (Lee-Richards on CPU is fp64-only regardless; see _cpu_run).
            return
        status = self._lib.fastsasa_context_create(ctypes.byref(self._ctx))
        if status != FASTSASA_SUCCESS:
            if request != "auto":
                self._ctx = None
                self._check(status)
            # No usable GPU: complete the Vulkan -> CUDA -> CPU chain with the
            # native threaded CPU backend, matching the CLI. Keep the
            # requested precision; the CPU backend honors it.
            self._ctx = None
            self.backend = "cpu"
            return
        self.backend = self._lib.fastsasa_context_backend(self._ctx).decode("utf-8")
        precision_status = self._lib.fastsasa_context_set_precision(
            self._ctx, precision_value
        )
        if precision_status != FASTSASA_SUCCESS:
            # Typically a Vulkan device without shaderFloat64 asked for the
            # FP64 default. Only an explicit backend request keeps the error.
            if request == "auto":
                self._lib.fastsasa_context_free(self._ctx)
                self._ctx = None
                self.backend = "cpu"
                return
            self._check(precision_status)

    def close(self):
        if getattr(self, "_ctx", None):
            self._lib.fastsasa_context_free(self._ctx)
            self._ctx = None
        self._closed = True

    def _require_open(self):
        if getattr(self, "_closed", False):
            raise RuntimeError("FastSASA context is closed")
        if self.backend != "cpu" and not self._ctx:
            raise RuntimeError("FastSASA context is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()

    def sasa(
        self,
        positions,
        radii,
        probe_radius=1.4,
        n_points=100,
        atom_sasa=False,
        residue_ids=None,
        n_residues=None,
        selection_masks=None,
        n_selections=None,
        as_result=False,
    ):
        """Compute Shrake-Rupley SASA on the selected accelerator.

        Args:
            positions: Coordinates with shape ``(atoms, 3)`` or
                ``(frames, atoms, 3)``.
            radii: Atom radii with shape ``(atoms,)``. Do not include the probe.
            probe_radius: Probe radius added to every atom radius.
            n_points: Shrake-Rupley surface points per atom.
            atom_sasa: Return per-atom SASA when true.
            residue_ids: Optional atom-to-residue ids, shape ``(atoms,)``.
            n_residues: Optional residue count. Inferred from ``residue_ids``.
            selection_masks: Optional bit mask per atom, shape ``(atoms,)``.
            n_selections: Optional number of selection bits, maximum 31.
            as_result: Return ``SasaResult`` instead of arrays/dicts.

        Returns:
            ``total`` with shape ``(frames,)``; ``(total, atom)`` when
            ``atom_sasa`` is true and no grouped outputs are requested; a dict
            with keys ``total``, optional ``atom``, ``residue``, and
            ``selection`` when grouped outputs are requested; or ``SasaResult``
            when ``as_result`` is true.
        """

        self._require_open()
        coords = _as_double_array(positions)
        if coords.ndim == 2:
            coords = coords.reshape((1,) + coords.shape)
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError("positions must have shape (atoms, 3) or (frames, atoms, 3)")
        if n_points <= 0:
            raise ValueError("n_points must be positive")

        n_frames, n_atoms, _ = coords.shape
        base_radii = _as_double_array(radii, (n_atoms,))
        radii_array = base_radii + probe_radius
        if not np.all(np.isfinite(coords)):
            raise ValueError("positions must contain only finite values")
        if not np.all(np.isfinite(radii_array)) or np.any(radii_array <= 0.0):
            raise ValueError("radii plus probe_radius must contain only positive finite values")
        if n_atoms == 0:
            return _empty_result(n_frames, n_atoms, atom_sasa, residue_ids, n_residues,
                                 selection_masks, n_selections, as_result)
        test_points = _cached_fibonacci_sphere_points(n_points)
        totals = np.empty(n_frames, dtype=np.float64)
        atom_values = np.empty((n_frames, n_atoms), dtype=np.float64) if atom_sasa else None
        residue_array = None
        residue_values = None
        selection_array = None
        selection_values = None

        if residue_ids is not None:
            residue_array = _as_int_array(residue_ids, (n_atoms,))
            if n_residues is None:
                valid_residues = residue_array[residue_array >= 0]
                if valid_residues.size == 0:
                    raise ValueError("residue_ids does not contain any non-negative residues")
                n_residues = int(valid_residues.max()) + 1
            if n_residues <= 0:
                raise ValueError("n_residues must be positive")
            residue_values = np.empty((n_frames, n_residues), dtype=np.float64)

        if selection_masks is not None:
            selection_array = _as_uint_array(selection_masks, (n_atoms,))
            if n_selections is None:
                max_mask = int(selection_array.max(initial=0))
                n_selections = max(1, max_mask.bit_length())
            if n_selections <= 0 or n_selections > 31:
                raise ValueError("n_selections must be between 1 and 31")
            selection_values = np.empty((n_frames, n_selections), dtype=np.float64)

        if self.backend == "cpu":
            return self._cpu_run(
                coords, radii_array, n_points, test_points, 0,
                totals, atom_values, residue_array, residue_values,
                selection_array, n_selections, selection_values, as_result,
            )

        input_data = _SrInput()
        input_data.n_atoms = n_atoms
        input_data.n_points = n_points
        input_data.x = _null_ptr(ctypes.c_double)
        input_data.y = _null_ptr(ctypes.c_double)
        input_data.z = _null_ptr(ctypes.c_double)
        input_data.radii = _as_double_ptr(radii_array)
        input_data.test_points = _as_double_ptr(test_points)
        input_data.neighbor_offsets = _null_ptr(ctypes.c_int)
        input_data.neighbor_indices = _null_ptr(ctypes.c_int)
        input_data.n_neighbor_indices = 0
        input_data.reuse_test_points = 1
        input_data.residue_ids = _as_int_ptr(residue_array) if residue_array is not None else _null_ptr(ctypes.c_int)
        input_data.n_residues = int(n_residues) if residue_array is not None else 0
        input_data.residue_sasa = _null_ptr(ctypes.c_double)
        input_data.selection_masks = _as_uint_ptr(selection_array) if selection_array is not None else _null_ptr(ctypes.c_uint)
        input_data.n_selections = int(n_selections) if selection_array is not None else 0
        input_data.selection_sasa = _null_ptr(ctypes.c_double)
        input_data.active_center_mask = 0
        input_data.active_center_indices = _null_ptr(ctypes.c_int)
        input_data.n_active_centers = 0
        input_data.force_double_precision = 0

        if self._run_vulkan_batch(
            coords, base_radii, probe_radius, n_points, 0,
            totals, atom_values, residue_array, residue_values,
            selection_array, n_selections, selection_values,
        ):
            return _format_result(
                totals, atom_values, residue_values, selection_values, as_result
            )

        for frame in range(n_frames):
            # coords, radii_array, and test_points stay alive until each
            # queued transfer completes at the synchronization below.
            input_data.xyz = _as_double_ptr(coords[frame])
            input_data.residue_sasa = (
                _as_double_ptr(residue_values[frame])
                if residue_values is not None
                else _null_ptr(ctypes.c_double)
            )
            input_data.selection_sasa = (
                _as_double_ptr(selection_values[frame])
                if selection_values is not None
                else _null_ptr(ctypes.c_double)
            )
            if atom_sasa:
                status = self._lib.fastsasa_context_shrake_rupley_cell_list_async(
                    self._ctx,
                    ctypes.byref(input_data),
                    _as_double_ptr(atom_values[frame]),
                )
                self._check(status)
            else:
                status = self._lib.fastsasa_context_shrake_rupley_cell_list_total_async(
                    self._ctx,
                    ctypes.byref(input_data),
                    totals[frame:].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
                self._check(status)
        self._check(self._lib.fastsasa_context_synchronize(self._ctx))
        if atom_sasa:
            for frame in range(n_frames):
                self._aggregate(atom_values[frame], totals[frame:frame + 1], None, None, None, 0, None)

        return _format_result(totals, atom_values, residue_values, selection_values, as_result)

    def lee_richards(
        self,
        positions,
        radii,
        probe_radius=1.4,
        n_slices=20,
        atom_sasa=False,
        residue_ids=None,
        n_residues=None,
        selection_masks=None,
        n_selections=None,
        as_result=False,
    ):
        """Compute Lee-Richards SASA on the selected accelerator.

        Input and return shapes match ``sasa()``. ``n_slices`` controls the LR
        slicing resolution instead of SR point count.
        """

        self._require_open()
        coords = _as_double_array(positions)
        if coords.ndim == 2:
            coords = coords.reshape((1,) + coords.shape)
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError("positions must have shape (atoms, 3) or (frames, atoms, 3)")
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")

        n_frames, n_atoms, _ = coords.shape
        base_radii = _as_double_array(radii, (n_atoms,))
        radii_array = base_radii + probe_radius
        if not np.all(np.isfinite(coords)):
            raise ValueError("positions must contain only finite values")
        if not np.all(np.isfinite(radii_array)) or np.any(radii_array <= 0.0):
            raise ValueError("radii plus probe_radius must contain only positive finite values")
        if n_atoms == 0:
            return _empty_result(n_frames, n_atoms, atom_sasa, residue_ids, n_residues,
                                 selection_masks, n_selections, as_result)
        totals = np.empty(n_frames, dtype=np.float64)
        atom_values = np.empty((n_frames, n_atoms), dtype=np.float64)
        residue_array = None
        residue_values = None
        selection_array = None
        selection_values = None

        if residue_ids is not None:
            residue_array = _as_int_array(residue_ids, (n_atoms,))
            if n_residues is None:
                valid_residues = residue_array[residue_array >= 0]
                if valid_residues.size == 0:
                    raise ValueError("residue_ids does not contain any non-negative residues")
                n_residues = int(valid_residues.max()) + 1
            if n_residues <= 0:
                raise ValueError("n_residues must be positive")
            residue_values = np.zeros((n_frames, n_residues), dtype=np.float64)

        if selection_masks is not None:
            selection_array = _as_uint_array(selection_masks, (n_atoms,))
            if n_selections is None:
                max_mask = int(selection_array.max(initial=0))
                n_selections = max(1, max_mask.bit_length())
            if n_selections <= 0 or n_selections > 31:
                raise ValueError("n_selections must be between 1 and 31")
            selection_values = np.zeros((n_frames, n_selections), dtype=np.float64)

        if self.backend == "cpu":
            result = self._cpu_run(
                coords, radii_array, n_slices, None, 1,
                totals, atom_values if atom_sasa else None,
                residue_array, residue_values,
                selection_array, n_selections, selection_values, as_result,
            )
            return result

        input_data = _SrInput()
        input_data.n_atoms = n_atoms
        input_data.n_points = int(n_slices)
        input_data.x = _null_ptr(ctypes.c_double)
        input_data.y = _null_ptr(ctypes.c_double)
        input_data.z = _null_ptr(ctypes.c_double)
        input_data.radii = _as_double_ptr(radii_array)
        input_data.test_points = _null_ptr(ctypes.c_double)
        input_data.neighbor_offsets = _null_ptr(ctypes.c_int)
        input_data.neighbor_indices = _null_ptr(ctypes.c_int)
        input_data.n_neighbor_indices = 0
        input_data.reuse_test_points = 0
        input_data.residue_ids = _null_ptr(ctypes.c_int)
        input_data.n_residues = 0
        input_data.residue_sasa = _null_ptr(ctypes.c_double)
        input_data.selection_masks = _null_ptr(ctypes.c_uint)
        input_data.n_selections = 0
        input_data.selection_sasa = _null_ptr(ctypes.c_double)
        input_data.active_center_mask = 0
        input_data.active_center_indices = _null_ptr(ctypes.c_int)
        input_data.n_active_centers = 0
        input_data.force_double_precision = 0

        batch_atom_values = atom_values if atom_sasa else None
        if self._run_vulkan_batch(
            coords, base_radii, probe_radius, n_slices, 1,
            totals, batch_atom_values, residue_array, residue_values,
            selection_array, n_selections, selection_values,
        ):
            returned_atom_values = atom_values if atom_sasa else None
            return _format_result(
                totals, returned_atom_values, residue_values, selection_values,
                as_result,
            )

        for frame in range(n_frames):
            # coords and radii_array stay alive until each synchronous ctypes
            # call returns.
            input_data.xyz = _as_double_ptr(coords[frame])
            status = self._lib.fastsasa_context_lee_richards(
                self._ctx,
                ctypes.byref(input_data),
                _as_double_ptr(atom_values[frame]),
            )
            self._check(status)
            self._aggregate(
                atom_values[frame], totals[frame:frame + 1],
                residue_array, residue_values[frame] if residue_values is not None else None,
                selection_array, n_selections,
                selection_values[frame] if selection_values is not None else None)

        returned_atom_values = atom_values if atom_sasa else None
        return _format_result(totals, returned_atom_values, residue_values, selection_values, as_result)

    def _configure_abi(self):
        _verify_native_abi(self._lib)
        self._lib.fastsasa_context_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.fastsasa_context_create.restype = ctypes.c_int
        self._lib.fastsasa_context_free.argtypes = [ctypes.c_void_p]
        self._lib.fastsasa_context_free.restype = None
        self._lib.fastsasa_context_backend.argtypes = [ctypes.c_void_p]
        self._lib.fastsasa_context_backend.restype = ctypes.c_char_p
        self._lib.fastsasa_context_set_precision.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.fastsasa_context_set_precision.restype = ctypes.c_int
        self._lib.fastsasa_context_precision.argtypes = [ctypes.c_void_p]
        self._lib.fastsasa_context_precision.restype = ctypes.c_int
        self._lib.fastsasa_context_synchronize.argtypes = [ctypes.c_void_p]
        self._lib.fastsasa_sum_atoms.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self._lib.fastsasa_sum_atoms.restype = ctypes.c_int
        self._lib.fastsasa_sum_residues.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self._lib.fastsasa_sum_residues.restype = ctypes.c_int
        self._lib.fastsasa_sum_selections.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self._lib.fastsasa_sum_selections.restype = ctypes.c_int
        self._lib.fastsasa_context_synchronize.restype = ctypes.c_int
        self._lib.fastsasa_context_shrake_rupley_cell_list.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SrInput),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_shrake_rupley_cell_list.restype = ctypes.c_int
        self._lib.fastsasa_context_shrake_rupley_cell_list_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SrInput),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_shrake_rupley_cell_list_async.restype = ctypes.c_int
        self._lib.fastsasa_context_shrake_rupley_cell_list_total.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SrInput),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_shrake_rupley_cell_list_total.restype = ctypes.c_int
        self._lib.fastsasa_context_shrake_rupley_cell_list_total_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SrInput),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_shrake_rupley_cell_list_total_async.restype = ctypes.c_int
        self._lib.fastsasa_context_lee_richards.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SrInput),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_lee_richards.restype = ctypes.c_int
        self._lib.fastsasa_context_calc_trajectory_soa.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_TrajectoryTopology),
            ctypes.POINTER(_SoAFrames),
            ctypes.POINTER(_TrajectoryParameters),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_calc_trajectory_soa.restype = ctypes.c_int
        self._lib.fastsasa_context_calc_trajectory_soa_selection.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_TrajectoryTopology),
            ctypes.POINTER(_SoAFrames),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_int,
            ctypes.POINTER(_TrajectoryParameters),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.fastsasa_context_calc_trajectory_soa_selection.restype = ctypes.c_int
        self._lib.fastsasa_status_string.argtypes = [ctypes.c_int]
        self._lib.fastsasa_status_string.restype = ctypes.c_char_p
        self._lib.fastsasa_last_error.argtypes = []
        self._lib.fastsasa_last_error.restype = ctypes.c_char_p
        double_ptr = ctypes.POINTER(ctypes.c_double)
        self._lib.fastsasa_cpu_shrake_rupley.argtypes = [
            ctypes.c_int, ctypes.c_int, double_ptr, double_ptr, double_ptr,
            double_ptr, double_ptr, ctypes.c_int, double_ptr,
        ]
        self._lib.fastsasa_cpu_shrake_rupley.restype = ctypes.c_int
        self._lib.fastsasa_cpu_shrake_rupley_precision.argtypes = [
            ctypes.c_int, ctypes.c_int, double_ptr, double_ptr, double_ptr,
            double_ptr, double_ptr, ctypes.c_int, ctypes.c_int, double_ptr,
        ]
        self._lib.fastsasa_cpu_shrake_rupley_precision.restype = ctypes.c_int
        self._lib.fastsasa_cpu_lee_richards.argtypes = [
            ctypes.c_int, ctypes.c_int, double_ptr, double_ptr, double_ptr,
            double_ptr, ctypes.c_int, double_ptr,
        ]
        self._lib.fastsasa_cpu_lee_richards.restype = ctypes.c_int

    def _check(self, status):
        if status == FASTSASA_SUCCESS:
            return
        status_text = self._lib.fastsasa_status_string(status).decode("utf-8")
        backend_text = self._lib.fastsasa_last_error().decode("utf-8")
        raise RuntimeError(f"FastSASA failed: {status_text} ({backend_text})")

    def _aggregate(self, atom_values, total_out, residue_array, residue_out,
                   selection_array, n_selections, selection_out):
        """Totals/residue/selection sums through the library's fixed-order
        Kahan helpers, so they match the C and CLI results bit for bit on
        every backend."""
        values = np.ascontiguousarray(atom_values, dtype=np.float64)
        n_atoms = values.shape[0]
        total = ctypes.c_double()
        self._check(self._lib.fastsasa_sum_atoms(_as_double_ptr(values), n_atoms, ctypes.byref(total)))
        total_out[...] = total.value
        if residue_out is not None:
            ids = np.ascontiguousarray(residue_array, dtype=np.int32)
            self._check(self._lib.fastsasa_sum_residues(
                _as_double_ptr(values), _as_int_ptr(ids), n_atoms,
                int(residue_out.shape[0]), _as_double_ptr(residue_out)))
        if selection_out is not None:
            masks = np.ascontiguousarray(selection_array, dtype=np.uint32)
            self._check(self._lib.fastsasa_sum_selections(
                _as_double_ptr(values), _as_uint_ptr(masks), n_atoms,
                int(n_selections), _as_double_ptr(selection_out)))

    def _cpu_run(
        self,
        coords,
        expanded_radii,
        resolution,
        test_points,
        algorithm,
        totals,
        atom_values,
        residue_array,
        residue_values,
        selection_array,
        n_selections,
        selection_values,
        as_result,
    ):
        """Native threaded CPU execution shared by sasa() and lee_richards()."""

        n_frames, n_atoms, _ = coords.shape
        expanded = np.ascontiguousarray(expanded_radii, dtype=np.float64)
        precision_value, _ = _precision_value(self.precision)
        if algorithm != 0 and precision_value != 0:
            warnings.warn(
                "CPU Lee-Richards is fp64-only; precision='fp32' is ignored for this algorithm",
                RuntimeWarning,
                stacklevel=3,
            )
        frame_scratch = (
            np.empty(n_atoms, dtype=np.float64) if atom_values is None else None
        )
        for frame in range(n_frames):
            x = np.ascontiguousarray(coords[frame, :, 0])
            y = np.ascontiguousarray(coords[frame, :, 1])
            z = np.ascontiguousarray(coords[frame, :, 2])
            target = atom_values[frame] if atom_values is not None else frame_scratch
            if algorithm == 0:
                status = self._lib.fastsasa_cpu_shrake_rupley_precision(
                    n_atoms, int(resolution), _as_double_ptr(x),
                    _as_double_ptr(y), _as_double_ptr(z),
                    _as_double_ptr(expanded), _as_double_ptr(test_points), 0,
                    precision_value, _as_double_ptr(target),
                )
            else:
                status = self._lib.fastsasa_cpu_lee_richards(
                    n_atoms, int(resolution), _as_double_ptr(x),
                    _as_double_ptr(y), _as_double_ptr(z),
                    _as_double_ptr(expanded), 0, _as_double_ptr(target),
                )
            self._check(status)
            self._aggregate(
                target, totals[frame:frame + 1],
                residue_array, residue_values[frame] if residue_values is not None else None,
                selection_array, n_selections,
                selection_values[frame] if selection_values is not None else None)
        return _format_result(
            totals, atom_values, residue_values, selection_values, as_result
        )

    def _run_vulkan_batch(
        self,
        coords,
        radii,
        probe_radius,
        resolution,
        algorithm,
        totals,
        atom_values,
        residue_array,
        residue_values,
        selection_array,
        n_selections,
        selection_values,
    ):
        if self.backend != "vulkan" or coords.shape[0] <= 1:
            return False
        if atom_values is not None and (
            residue_values is not None or selection_values is not None
        ):
            return False
        if residue_values is not None and selection_values is not None:
            return False

        x = np.ascontiguousarray(coords[:, :, 0])
        y = np.ascontiguousarray(coords[:, :, 1])
        z = np.ascontiguousarray(coords[:, :, 2])
        radii = np.ascontiguousarray(radii, dtype=np.float64)
        topology = _TrajectoryTopology(
            _as_double_ptr(radii),
            _as_int_ptr(residue_array)
            if residue_array is not None
            else _null_ptr(ctypes.c_int),
            coords.shape[1],
            residue_values.shape[1] if residue_values is not None else 0,
        )
        frames = _SoAFrames(
            _as_double_ptr(x), _as_double_ptr(y), _as_double_ptr(z), coords.shape[0]
        )
        precision_value, _ = _precision_value(self.precision)
        parameters = _TrajectoryParameters(probe_radius, resolution, algorithm, precision_value)

        if selection_values is not None:
            status = self._lib.fastsasa_context_calc_trajectory_soa_selection(
                self._ctx,
                ctypes.byref(topology),
                ctypes.byref(frames),
                _as_uint_ptr(selection_array),
                int(n_selections),
                ctypes.byref(parameters),
                _as_double_ptr(totals),
                _as_double_ptr(selection_values),
            )
        else:
            status = self._lib.fastsasa_context_calc_trajectory_soa(
                self._ctx,
                ctypes.byref(topology),
                ctypes.byref(frames),
                ctypes.byref(parameters),
                _as_double_ptr(totals),
                _as_double_ptr(atom_values)
                if atom_values is not None
                else _null_ptr(ctypes.c_double),
                _as_double_ptr(residue_values)
                if residue_values is not None
                else _null_ptr(ctypes.c_double),
            )
        self._check(status)
        return True


_DEFAULT_ENGINES: dict = {}
_DEFAULT_ENGINE_LOCK = threading.Lock()


def default_engine(precision="fp64"):
    """Return a shared, lazily created ``SasaEngine`` for one-shot calls.

    Creating a GPU context costs orders of magnitude more than a small SASA
    calculation, so the module-level ``sasa()`` and ``lee_richards()``
    helpers reuse one cached engine per precision instead of creating and
    destroying a context per call. The cached engines live for the process;
    call ``close_default_engines()`` to release them early. Like ``SasaEngine``
    itself, a cached engine must not be used from multiple threads at once.
    """

    _, key = _precision_value(precision)
    with _DEFAULT_ENGINE_LOCK:
        engine = _DEFAULT_ENGINES.get(key)
        if engine is None or getattr(engine, "_closed", False):
            engine = SasaEngine(precision=key)
            _DEFAULT_ENGINES[key] = engine
        return engine


def close_default_engines():
    """Release the cached one-shot engines and their device resources."""

    with _DEFAULT_ENGINE_LOCK:
        for engine in _DEFAULT_ENGINES.values():
            engine.close()
        _DEFAULT_ENGINES.clear()


def sasa(
    positions,
    radii,
    probe_radius=1.4,
    n_points=100,
    atom_sasa=False,
    residue_ids=None,
    n_residues=None,
    selection_masks=None,
    n_selections=None,
    as_result=False,
    precision="fp64",
):
    """One-shot Shrake-Rupley SASA using the shared default engine.

    The call reuses a cached per-precision ``SasaEngine`` (see
    ``default_engine``), so repeated one-shot calls do not pay context
    creation. For explicit lifetime control or multi-threaded use, create
    your own ``SasaEngine``. ``precision`` accepts ``"fp64"`` (default) or
    ``"fp32"``.
    """

    engine = default_engine(precision)
    return engine.sasa(
        positions,
        radii,
        probe_radius,
        n_points,
        atom_sasa,
        residue_ids,
        n_residues,
        selection_masks,
        n_selections,
        as_result,
    )


def lee_richards(
    positions,
    radii,
    probe_radius=1.4,
    n_slices=20,
    atom_sasa=False,
    residue_ids=None,
    n_residues=None,
    selection_masks=None,
    n_selections=None,
    as_result=False,
    precision="fp64",
):
    """One-shot Lee-Richards SASA using the shared default engine.

    ``precision`` accepts ``"fp64"`` (default) or ``"fp32"``.
    """

    engine = default_engine(precision)
    return engine.lee_richards(
        positions,
        radii,
        probe_radius,
        n_slices,
        atom_sasa,
        residue_ids,
        n_residues,
        selection_masks,
        n_selections,
        as_result,
    )

