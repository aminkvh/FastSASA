"""SASA feature extraction utilities for MD and ML workflows.

All feature functions accept NumPy-compatible arrays. Coordinates have shape
``(atoms, 3)`` or ``(frames, atoms, 3)``. Per-frame time series are returned as
``(frames,)`` or ``(frames, features)`` arrays, and summary statistics keep the
feature axis intact.
"""

from __future__ import annotations

import warnings

import numpy as np

from fastsasa_adapters import canonical_residue_name

try:
    from .fastsasa_native import SasaEngine
except ImportError:
    from fastsasa_native import SasaEngine

# ProtOr theoretical maximum-ASA reference values (total, main_chain,
# side_chain, apolar, polar), used for relative SASA (RSA). These are
# FreeSASA's own ProtOr reference values (Tsai, Taylor, Chothia & Gerstein,
# J Mol Biol 1999) - the same table `tools/fastsasa_cli.cc` uses for
# `--format rsa`, copied here so Python callers do not need a subprocess.
# tests/python_feature_unit.py checks every field of this table against the
# C source table, and tests/python_feature_validation.py cross-checks
# `per_residue_rsa` against a real `--format rsa` run.
_PROTOR_MAX_ASA = {
    "ALA": {"total": 108.76, "main_chain": 43.96, "side_chain": 64.80, "polar": 37.75, "apolar": 71.01},
    "ARG": {"total": 238.17, "main_chain": 42.00, "side_chain": 196.17, "polar": 165.00, "apolar": 73.17},
    "ASN": {"total": 145.01, "main_chain": 41.53, "side_chain": 103.48, "polar": 103.46, "apolar": 41.55},
    "ASP": {"total": 142.76, "main_chain": 42.29, "side_chain": 100.47, "polar": 100.27, "apolar": 42.49},
    "CYS": {"total": 132.20, "main_chain": 42.55, "side_chain": 89.66, "polar": 92.74, "apolar": 39.47},
    "GLN": {"total": 178.83, "main_chain": 42.00, "side_chain": 136.83, "polar": 131.85, "apolar": 46.98},
    "GLU": {"total": 174.18, "main_chain": 42.00, "side_chain": 132.18, "polar": 122.48, "apolar": 51.70},
    "GLY": {"total": 81.09, "main_chain": 81.09, "side_chain": 0.00, "polar": 44.65, "apolar": 36.44},
    "HIS": {"total": 182.97, "main_chain": 39.09, "side_chain": 143.87, "polar": 85.94, "apolar": 97.03},
    "ILE": {"total": 175.73, "main_chain": 41.49, "side_chain": 134.23, "polar": 36.85, "apolar": 138.87},
    "LEU": {"total": 179.56, "main_chain": 39.78, "side_chain": 139.78, "polar": 37.16, "apolar": 142.39},
    "LYS": {"total": 204.98, "main_chain": 42.00, "side_chain": 162.98, "polar": 93.88, "apolar": 111.10},
    "MET": {"total": 193.10, "main_chain": 42.00, "side_chain": 151.10, "polar": 75.48, "apolar": 117.62},
    "PHE": {"total": 199.88, "main_chain": 38.43, "side_chain": 161.45, "polar": 34.94, "apolar": 164.94},
    "PRO": {"total": 137.21, "main_chain": 27.51, "side_chain": 109.70, "polar": 16.09, "apolar": 121.12},
    "SER": {"total": 118.34, "main_chain": 43.41, "side_chain": 74.93, "polar": 71.38, "apolar": 46.96},
    "THR": {"total": 140.60, "main_chain": 41.96, "side_chain": 98.64, "polar": 66.15, "apolar": 74.45},
    "TRP": {"total": 249.19, "main_chain": 42.59, "side_chain": 206.60, "polar": 61.64, "apolar": 187.55},
    "TYR": {"total": 214.19, "main_chain": 38.43, "side_chain": 175.76, "polar": 81.12, "apolar": 133.07},
    "VAL": {"total": 151.97, "main_chain": 41.50, "side_chain": 110.46, "polar": 36.87, "apolar": 115.09},
}

# Protein and nucleic-acid backbone atom names, matching is_backbone_atom()
# in tools/fastsasa_cli.cc (used there for --format rsa's main/side-chain
# split).
_BACKBONE_ATOM_NAMES = frozenset({
    "CA", "N", "O", "C", "OXT",
    "P", "OP1", "OP2", "O5'", "C5'", "C4'",
    "O4'", "C3'", "O3'", "C2'", "C1'",
})


def _backbone_mask(atom_names, n_atoms):
    names = np.asarray([str(name).strip().upper() for name in atom_names])
    if names.shape != (n_atoms,):
        raise ValueError(f"atom_names must have shape ({n_atoms},)")
    return np.isin(names, list(_BACKBONE_ATOM_NAMES))


_WARNED_UNKNOWN_RESIDUES = set()


def _residue_max_asa(residue_names, n_residues, field="total"):
    """Max-ASA reference per residue; NaN (with a one-time warning per name)
    for residues outside the 20 standard amino acids. MD variants such as
    HIE, HID, HIP, CYX, ASH, GLH, or LYN resolve to their standard residue."""

    names = np.asarray([canonical_residue_name(name) for name in residue_names])
    if names.shape != (n_residues,):
        raise ValueError(f"residue_names must have shape ({n_residues},)")
    max_asa = np.full(n_residues, np.nan, dtype=np.float64)
    unknown = set()
    for index, name in enumerate(names):
        entry = _PROTOR_MAX_ASA.get(name)
        if entry is not None:
            max_asa[index] = entry[field]
        else:
            unknown.add(str(name))
    new_unknown = sorted(unknown - _WARNED_UNKNOWN_RESIDUES)
    if new_unknown:
        _WARNED_UNKNOWN_RESIDUES.update(new_unknown)
        warnings.warn(
            "no max-ASA reference for residue name(s) "
            + ", ".join(new_unknown)
            + "; their relative SASA is NaN (only the 20 standard amino acids "
            "and their common MD variants have references)",
            RuntimeWarning,
            stacklevel=3,
        )
    return max_asa


def _per_residue_masked_sum(atom_sasa, residue_ids, n_residues, atom_mask):
    """Sum atom_sasa (frames, atoms) into (frames, residues), restricted to
    atoms where atom_mask is True."""

    n_frames = atom_sasa.shape[0]
    valid = (residue_ids >= 0) & (residue_ids < n_residues) & atom_mask
    if not np.any(valid):
        return np.zeros((n_frames, n_residues), dtype=np.float64)
    residue_valid = residue_ids[valid]
    values = atom_sasa[:, valid]
    flat_index = (np.arange(n_frames)[:, None] * n_residues + residue_valid[None, :]).reshape(-1)
    summed = np.bincount(flat_index, weights=values.reshape(-1), minlength=n_frames * n_residues)
    return summed.reshape(n_frames, n_residues)


def _coords_array(positions):
    coords = np.asarray(positions, dtype=np.float64)
    if coords.ndim == 2:
        coords = coords.reshape((1,) + coords.shape)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("positions must have shape (atoms, 3) or (frames, atoms, 3)")
    return np.ascontiguousarray(coords)


def _radii_array(radii, n_atoms):
    values = np.asarray(radii, dtype=np.float64)
    if values.shape != (n_atoms,):
        raise ValueError(f"expected {n_atoms} radii, got shape {values.shape}")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("radii must be finite and non-negative")
    return np.ascontiguousarray(values)


def _bool_mask(mask, n_atoms, name):
    values = np.asarray(mask, dtype=bool)
    if values.shape != (n_atoms,):
        raise ValueError(f"{name} must have shape ({n_atoms},)")
    return values


def _subset_sasa(engine, coords, radii, mask, probe_radius, n_points, atom_sasa=False):
    indices = np.flatnonzero(mask)
    n_frames = coords.shape[0]
    if indices.size == 0:
        total = np.zeros(n_frames, dtype=np.float64)
        if atom_sasa:
            return total, np.zeros((n_frames, 0), dtype=np.float64), indices
        return total
    if atom_sasa:
        total, atom = engine.sasa(
            coords[:, indices, :],
            radii[indices],
            probe_radius=probe_radius,
            n_points=n_points,
            atom_sasa=True,
        )
        return total, atom, indices
    return engine.sasa(
        coords[:, indices, :],
        radii[indices],
        probe_radius=probe_radius,
        n_points=n_points,
        atom_sasa=False,
    )


def _atom_sasa_from_result(result):
    if isinstance(result, dict):
        return result["atom"]
    return result[1]


def _interface_masks(group_a_mask, group_b_mask, n_atoms):
    a_mask = _bool_mask(group_a_mask, n_atoms, "group_a_mask")
    b_mask = _bool_mask(group_b_mask, n_atoms, "group_b_mask")
    if not np.any(a_mask):
        raise ValueError("group_a_mask must select at least one atom")
    if not np.any(b_mask):
        raise ValueError("group_b_mask must select at least one atom")
    if np.any(a_mask & b_mask):
        raise ValueError("group_a_mask and group_b_mask must not overlap")
    return a_mask, b_mask


def _interface_sasa_with_engine(engine, coords, radii, a_mask, b_mask, probe_radius, n_points,
                                residue_ids=None, n_residues=None,
                                polar_mask=None, apolar_mask=None):
    need_atom_detail = residue_ids is not None and n_residues is not None
    if need_atom_detail:
        a_free, a_free_atom, a_indices = _subset_sasa(
            engine, coords, radii, a_mask, probe_radius, n_points, atom_sasa=True)
        b_free, b_free_atom, b_indices = _subset_sasa(
            engine, coords, radii, b_mask, probe_radius, n_points, atom_sasa=True)
    else:
        a_free = _subset_sasa(engine, coords, radii, a_mask, probe_radius, n_points)
        b_free = _subset_sasa(engine, coords, radii, b_mask, probe_radius, n_points)
    interface_mask = a_mask | b_mask
    interface_indices = np.flatnonzero(interface_mask)
    interface_result = engine.sasa(
        coords[:, interface_indices, :],
        radii[interface_indices],
        probe_radius=probe_radius,
        n_points=n_points,
        atom_sasa=True,
    )
    interface_atom_sasa = _atom_sasa_from_result(interface_result)
    a_in_interface = a_mask[interface_mask]
    b_in_interface = b_mask[interface_mask]
    a_bound = interface_atom_sasa[:, a_in_interface].sum(axis=1)
    b_bound = interface_atom_sasa[:, b_in_interface].sum(axis=1)
    a_buried = a_free - a_bound
    b_buried = b_free - b_bound
    buried = a_buried + b_buried
    result = {
        "interface_a_free_sasa": a_free,
        "interface_b_free_sasa": b_free,
        "interface_a_bound_sasa": a_bound,
        "interface_b_bound_sasa": b_bound,
        "interface_complex_sasa": a_bound + b_bound,
        "interface_a_buried_sasa": a_buried,
        "interface_b_buried_sasa": b_buried,
        "interface_buried_sasa": buried,
        "interface_buried_sasa_half": 0.5 * buried,
        # Fraction of each side's isolated SASA that is lost on binding.
        # Raw BSA scales with molecule size, so this is what is comparable
        # across ligands/partners of different sizes.
        "interface_a_buried_fraction": a_buried / np.maximum(a_free, 1e-12),
        "interface_b_buried_fraction": b_buried / np.maximum(b_free, 1e-12),
    }

    if need_atom_detail:
        n_frames, n_atoms = coords.shape[0], coords.shape[1]
        isolated_full = np.zeros((n_frames, n_atoms), dtype=np.float64)
        isolated_full[:, a_indices] = a_free_atom
        isolated_full[:, b_indices] = b_free_atom
        bound_full = np.zeros((n_frames, n_atoms), dtype=np.float64)
        bound_full[:, interface_indices] = interface_atom_sasa
        # Only meaningful where interface_mask is True: elsewhere isolated
        # and bound are both left at the placeholder 0.
        atom_buried = isolated_full - bound_full

        residue_ids_arr = np.asarray(residue_ids)
        per_residue_buried = _per_residue_masked_sum(atom_buried, residue_ids_arr, n_residues, interface_mask)
        per_residue_isolated = _per_residue_masked_sum(isolated_full, residue_ids_arr, n_residues, interface_mask)
        valid_residue = (residue_ids_arr >= 0) & (residue_ids_arr < n_residues) & interface_mask
        interface_residue_mask = np.zeros(n_residues, dtype=bool)
        interface_residue_mask[residue_ids_arr[valid_residue]] = True
        with np.errstate(invalid="ignore", divide="ignore"):
            per_residue_fraction = per_residue_buried / np.maximum(per_residue_isolated, 1e-12)
        result["per_residue_buried_sasa"] = np.where(interface_residue_mask[None, :], per_residue_buried, np.nan)
        result["per_residue_buried_fraction"] = np.where(interface_residue_mask[None, :], per_residue_fraction, np.nan)
        result["interface_residue_mask"] = interface_residue_mask

        if polar_mask is not None:
            polar_interface = np.asarray(polar_mask, dtype=bool) & interface_mask
            result["interface_polar_buried_sasa"] = (
                atom_buried[:, polar_interface].sum(axis=1) if np.any(polar_interface)
                else np.zeros(n_frames, dtype=np.float64)
            )
        if apolar_mask is not None:
            apolar_interface = np.asarray(apolar_mask, dtype=bool) & interface_mask
            result["interface_apolar_buried_sasa"] = (
                atom_buried[:, apolar_interface].sum(axis=1) if np.any(apolar_interface)
                else np.zeros(n_frames, dtype=np.float64)
            )
        if polar_mask is not None and apolar_mask is not None:
            polar_buried = result["interface_polar_buried_sasa"]
            apolar_buried = result["interface_apolar_buried_sasa"]
            classified_buried = polar_buried + apolar_buried
            result["interface_apolar_fraction"] = apolar_buried / np.maximum(classified_buried, 1e-12)

    return result


def interface_sasa(positions, radii, group_a_mask, group_b_mask, probe_radius=1.4, n_points=100):
    """Compute interface/buried SASA for two non-overlapping atom groups.

    The interface is defined as ``group_a_mask | group_b_mask``. Returned
    values are per-frame time series:

    ``interface_buried_sasa = A_free + B_free - SASA(A+B)``

    ``interface_a_buried_sasa`` and ``interface_b_buried_sasa`` report the
    per-side loss of SASA after A and B are calculated together.
    ``interface_buried_sasa_half`` is half of the combined loss, a convention
    used by some interface-analysis literature.
    """

    coords = _coords_array(positions)
    radii = _radii_array(radii, coords.shape[1])
    a_mask, b_mask = _interface_masks(group_a_mask, group_b_mask, coords.shape[1])
    with SasaEngine() as engine:
        return _interface_sasa_with_engine(engine, coords, radii, a_mask, b_mask, probe_radius, n_points)


def aggregate_atom_sasa(atom_sasa, masks):
    """Sum atom SASA into named masks.

    Args:
        atom_sasa: Per-atom SASA with shape ``(atoms,)`` or
            ``(frames, atoms)``.
        masks: Mapping of name to boolean mask, each shape ``(atoms,)``.

    Returns:
        Dict mapping each name to a time series with shape ``(frames,)``.
    """

    values = np.asarray(atom_sasa, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((1, -1))
    if values.ndim != 2:
        raise ValueError("atom_sasa must have shape (frames, atoms) or (atoms,)")

    n_atoms = values.shape[1]
    result = {}
    for name, mask in masks.items():
        selected = _bool_mask(mask, n_atoms, name)
        result[name] = values[:, selected].sum(axis=1)
    return result


# The statistics summarize_time_series() reports for every feature, in the
# sorted order flatten_statistics() emits them. Fingerprint vectors are only
# comparable when they were built from the same layout: pass
# ``feature_names`` to SasaFingerprintEmbedder.fit()/transform() to have
# that checked. Vectors built before v0.1.0-rc19 carried only the first
# seven of these and must be regenerated.
SUMMARY_STATISTIC_NAMES = (
    "exposure_frequency", "iqr", "mad", "max", "mean", "median", "min",
    "q05", "q25", "q75", "q95", "std", "transitions", "variance",
)


def summarize_time_series(series, threshold=0.0):
    """Compute distribution and exposure statistics for a feature time series.

    ``series`` may be ``(frames,)`` or ``(frames, features)``. Returned arrays
    have shape ``(features,)``.

    ``variance``/``std`` use the population convention (``ddof=0``).
    ``exposure_frequency`` is the fraction of frames with ``value > threshold``
    (strict, not ``>=``). ``transitions`` counts changes in that thresholded
    state between consecutive frames, with no hysteresis. ``mad`` is the
    unscaled median absolute deviation, ``median(|x - median(x)|)`` (not
    multiplied by the 1.4826 normal-consistency constant).
    """

    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if values.ndim != 2:
        raise ValueError("series must be one- or two-dimensional")

    exposed = values > threshold
    transitions = np.count_nonzero(exposed[1:] != exposed[:-1], axis=0) if values.shape[0] > 1 else np.zeros(values.shape[1], dtype=np.int64)
    median = np.median(values, axis=0)
    q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95], axis=0)
    return {
        "mean": values.mean(axis=0),
        "variance": values.var(axis=0),
        "std": values.std(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "median": median,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "iqr": q75 - q25,
        "mad": np.median(np.abs(values - median), axis=0),
        "exposure_frequency": exposed.mean(axis=0),
        "transitions": transitions,
    }


def _exposure_state_trace(column, threshold, hysteresis):
    n = column.shape[0]
    states = np.empty(n, dtype=bool)
    state = bool(column[0] > threshold)
    states[0] = state
    transitions = 0
    upper = threshold + hysteresis
    lower = threshold - hysteresis
    for i in range(1, n):
        value = column[i]
        if state and value <= lower:
            state = False
            transitions += 1
        elif not state and value > upper:
            state = True
            transitions += 1
        states[i] = state
    return states, transitions


def _interior_run_lengths(states):
    n = states.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=bool)
    change_points = np.flatnonzero(states[1:] != states[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [n]))
    run_lengths = np.diff(boundaries).astype(np.float64)
    run_states = states[boundaries[:-1]]
    if run_lengths.shape[0] <= 2:
        return np.array([], dtype=np.float64), np.array([], dtype=bool)
    return run_lengths[1:-1], run_states[1:-1]


def exposure_kinetics(series, threshold=0.0, hysteresis=0.0, frame_interval=1.0):
    """Dwell-time and transition-rate kinetics for a thresholded time series.

    ``series`` may be ``(frames,)`` or ``(frames, features)``. This uses a
    Schmitt-trigger state machine, not the simple per-frame diff
    ``summarize_time_series()`` uses for ``transitions``: once a feature is
    "exposed" (state ``True``), it only flips back to "buried" when the value
    drops to ``threshold - hysteresis`` or below, and only flips back to
    "exposed" when the value rises strictly above ``threshold + hysteresis``.
    The initial state is ``series[0] > threshold`` (not offset by
    hysteresis), matching ``summarize_time_series()``'s ``exposed = value >
    threshold`` convention. With ``hysteresis=0.0`` this reproduces
    ``summarize_time_series()``'s ``transitions`` count and per-frame state
    exactly.

    Dwell times exclude the first and last run of each feature: both are
    truncated by the trajectory boundary (the true run may have started or
    ended outside the analyzed window), and including them biases the mean
    downward. A feature with no complete interior run of a given state gets
    ``NaN`` for that state's dwell mean.

    ``frame_interval`` scales frame counts into whatever time unit the
    caller's frames represent (default ``1.0``: dwell time and rate are in
    units of frames). ``exposure_transition_rate`` is
    ``transitions / ((frames - 1) * frame_interval)`` — transitions per unit
    time over the analyzed span; ``NaN`` for a single-frame series.
    """
    if not np.isfinite(hysteresis) or hysteresis < 0.0:
        raise ValueError("hysteresis must be a finite non-negative number")
    if not np.isfinite(frame_interval) or frame_interval <= 0.0:
        raise ValueError("frame_interval must be a finite positive number")

    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if values.ndim != 2:
        raise ValueError("series must be one- or two-dimensional")

    n_frames, n_features = values.shape
    transitions = np.zeros(n_features, dtype=np.int64)
    exposed_dwell_mean = np.full(n_features, np.nan)
    buried_dwell_mean = np.full(n_features, np.nan)

    for feature in range(n_features):
        states, count = _exposure_state_trace(values[:, feature], threshold, hysteresis)
        transitions[feature] = count
        run_lengths, run_states = _interior_run_lengths(states)
        if run_lengths.size and np.any(run_states):
            exposed_dwell_mean[feature] = run_lengths[run_states].mean() * frame_interval
        if run_lengths.size and np.any(~run_states):
            buried_dwell_mean[feature] = run_lengths[~run_states].mean() * frame_interval

    if n_frames > 1:
        transition_rate = transitions / ((n_frames - 1) * frame_interval)
    else:
        transition_rate = np.full(n_features, np.nan)

    return {
        "transitions": transitions,
        "exposed_dwell_time_mean": exposed_dwell_mean,
        "buried_dwell_time_mean": buried_dwell_mean,
        "exposure_transition_rate": transition_rate,
    }


_MIN_FRAMES_FOR_AUTOCORRELATION = 8
_MAX_AUTOCORRELATION_LAG = 1000


def _integrated_autocorrelation_time(column):
    n = column.shape[0]
    centered = column - column.mean()
    variance = centered.var()
    if variance <= 0.0:
        return 1.0

    tau = 1.0
    max_lag = min(n - 1, _MAX_AUTOCORRELATION_LAG)
    for lag in range(1, max_lag + 1):
        rho = np.dot(centered[: n - lag], centered[lag:]) / n / variance
        if rho < 0.0:
            break
        tau += 2.0 * rho
    return tau


def _block_standard_error(values, block_size):
    n_frames, n_features = values.shape
    if block_size is None:
        block_size = max(1, int(round(np.sqrt(n_frames))))
    block_size = max(1, int(block_size))
    n_blocks = n_frames // block_size
    if n_blocks < 2:
        return np.full(n_features, np.nan)
    trimmed = values[: n_blocks * block_size]
    block_means = trimmed.reshape(n_blocks, block_size, n_features).mean(axis=1)
    return block_means.std(axis=0, ddof=1) / np.sqrt(n_blocks)


def time_series_uncertainty(series, block_size=None):
    """Autocorrelation-aware uncertainty estimates for a feature time series.

    ``series`` may be ``(frames,)`` or ``(frames, features)``. Consecutive MD
    frames are correlated, so the naive standard error (``std / sqrt(n)``)
    understates the true uncertainty of the mean; these estimators correct
    for it.

    ``autocorrelation_time`` is the integrated autocorrelation time,
    ``tau = 1 + 2 * sum(rho_k)``, summed over lag ``k`` until the normalized
    autocovariance ``rho_k`` first goes negative (the "initial positive
    sequence" estimator, Geyer 1992) or ``k`` reaches 1000, whichever comes
    first — a bound on cost for long trajectories, not a claim about the true
    correlation length. A constant feature (zero variance) gets ``tau = 1.0``.
    ``effective_sample_size`` is ``n_frames / tau``. ``standard_error`` is
    ``std(series, ddof=1) * sqrt(tau / n_frames)``.

    ``block_standard_error`` is an independent, model-free cross-check:
    split the series into non-overlapping blocks (default size
    ``round(sqrt(n_frames))``, at least 1 frame; the remainder is dropped)
    and report ``std(block_means, ddof=1) / sqrt(n_blocks)``.

    Both estimators need enough frames to be meaningful and return ``NaN``
    otherwise: ``autocorrelation_time``/``effective_sample_size``/
    ``standard_error`` below 8 frames, ``block_standard_error`` below 2
    complete blocks.
    """

    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if values.ndim != 2:
        raise ValueError("series must be one- or two-dimensional")

    n_frames, n_features = values.shape
    tau = np.full(n_features, np.nan)
    n_eff = np.full(n_features, np.nan)
    standard_error = np.full(n_features, np.nan)

    if n_frames >= _MIN_FRAMES_FOR_AUTOCORRELATION:
        for feature in range(n_features):
            tau[feature] = _integrated_autocorrelation_time(values[:, feature])
        n_eff = n_frames / tau
        standard_error = values.std(axis=0, ddof=1) * np.sqrt(tau / n_frames)

    return {
        "autocorrelation_time": tau,
        "effective_sample_size": n_eff,
        "standard_error": standard_error,
        "block_standard_error": _block_standard_error(values, block_size),
    }


def residue_sasa(positions, radii, residue_ids, n_residues=None, probe_radius=1.4, n_points=100):
    """Compute per-residue SR SASA over one structure or a trajectory batch.

    Returns an array with shape ``(frames, residues)``.
    """

    coords = _coords_array(positions)
    radii = _radii_array(radii, coords.shape[1])
    residue_ids = np.asarray(residue_ids, dtype=np.int32)
    if residue_ids.shape != (coords.shape[1],):
        raise ValueError(f"residue_ids must have shape ({coords.shape[1]},)")
    if n_residues is None:
        valid = residue_ids[residue_ids >= 0]
        if valid.size == 0:
            raise ValueError("residue_ids must contain at least one non-negative residue")
        n_residues = int(valid.max()) + 1

    with SasaEngine() as engine:
        result = engine.sasa(
            coords,
            radii,
            probe_radius=probe_radius,
            n_points=n_points,
            atom_sasa=True,
            residue_ids=residue_ids,
            n_residues=n_residues,
        )
    return result["residue"]


def extract_md_features(
    positions,
    radii,
    residue_ids=None,
    n_residues=None,
    group_masks=None,
    hydrophobic_mask=None,
    polar_mask=None,
    interface_a_mask=None,
    interface_b_mask=None,
    glycan_mask=None,
    glycan_target_mask=None,
    atom_names=None,
    residue_names=None,
    rsa_exposed_threshold=0.25,
    probe_radius=1.4,
    n_points=100,
    exposure_threshold=0.0,
):
    """Extract ML-ready SASA descriptors from an MD coordinate batch.

    Returns a dict with ``time_series``, ``statistics``, and ``metadata``.
    Common outputs include ``total_sasa``, optional ``per_residue_sasa``,
    named group exposure (``group_masks``, e.g. a ligand or a binding pocket
    under whatever key you choose), hydrophobic/polar exposed area, interface
    SASA, buried surface area, and glycan shielding metrics.

    ``atom_names`` (shape ``(atoms,)``) additionally enables ``backbone_sasa``/
    ``sidechain_sasa`` (and, with ``residue_ids``, their per-residue forms),
    using the same backbone atom-name list as the CLI's ``--format rsa``.

    ``residue_names`` (one three-letter code per residue, shape
    ``(residues,)``, requires ``residue_ids``) enables ``per_residue_rsa``,
    ``mean_rsa``, ``exposed_residue_fraction``, and
    ``buried_residue_fraction`` using the bundled ProtOr maximum-ASA
    reference table. This assumes ProtOr-compatible radii, same as the CLI's
    ``--format rsa``; RSA is not meaningful for a materially different radius
    model. Residues outside the standard 20 amino acids have no reference
    value and get ``NaN`` RSA (excluded from ``mean_rsa``/the exposed and
    buried fractions, not treated as zero). ``rsa_exposed_threshold``
    (default ``0.25``, i.e. 25% relative accessibility) sets the
    exposed/buried cutoff.

    With ``residue_ids`` and both ``interface_a_mask``/``interface_b_mask``,
    the interface descriptors additionally include ``per_residue_buried_sasa``,
    ``per_residue_buried_fraction`` (both ``NaN`` for residues with no atom in
    the interface - see ``interface_residue_mask``), and, if ``polar_mask``/
    ``hydrophobic_mask`` are also given, ``interface_polar_buried_sasa``,
    ``interface_apolar_buried_sasa``, and ``interface_apolar_fraction``
    (buried area, not exposed area - the isolated-vs-bound loss restricted to
    each chemical class, at the interface only).
    """

    coords = _coords_array(positions)
    n_frames, n_atoms, _ = coords.shape
    radii = _radii_array(radii, n_atoms)

    residue_array = None
    interface_residue_mask = None
    if residue_ids is not None:
        residue_array = np.asarray(residue_ids, dtype=np.int32)
        if residue_array.shape != (n_atoms,):
            raise ValueError(f"residue_ids must have shape ({n_atoms},)")
        if n_residues is None:
            valid = residue_array[residue_array >= 0]
            if valid.size == 0:
                raise ValueError("residue_ids must contain at least one non-negative residue")
            n_residues = int(valid.max()) + 1

    masks = {}
    for name, mask in (group_masks or {}).items():
        masks[str(name)] = _bool_mask(mask, n_atoms, str(name))
    optional_masks = {
        "hydrophobic_exposed_area": hydrophobic_mask,
        "polar_exposed_area": polar_mask,
    }
    for name, mask in optional_masks.items():
        if mask is not None:
            masks[name] = _bool_mask(mask, n_atoms, name)

    backbone_mask = None
    if atom_names is not None:
        backbone_mask = _backbone_mask(atom_names, n_atoms)
        masks["backbone_sasa"] = backbone_mask
        masks["sidechain_sasa"] = ~backbone_mask

    if residue_names is not None and residue_array is None:
        raise ValueError("residue_names requires residue_ids")

    glycan = None
    target = None
    if glycan_mask is not None and glycan_target_mask is not None:
        glycan = _bool_mask(glycan_mask, n_atoms, "glycan_mask")
        target = _bool_mask(glycan_target_mask, n_atoms, "glycan_target_mask")
        if np.any(glycan & target):
            raise ValueError("glycan_mask and glycan_target_mask must not overlap")

    with SasaEngine() as engine:
        result = engine.sasa(
            coords,
            radii,
            probe_radius=probe_radius,
            n_points=n_points,
            atom_sasa=True,
            residue_ids=residue_array,
            n_residues=n_residues,
        )
        if isinstance(result, dict):
            atom_sasa = result["atom"]
            total_sasa = result["total"]
            residue_values = result.get("residue")
        else:
            total_sasa, atom_sasa = result
            residue_values = None

        time_series = {"total_sasa": total_sasa}
        if residue_values is not None:
            time_series["per_residue_sasa"] = residue_values
        time_series.update(aggregate_atom_sasa(atom_sasa, masks))

        if residue_array is not None and backbone_mask is not None:
            time_series["per_residue_backbone_sasa"] = _per_residue_masked_sum(
                atom_sasa, residue_array, n_residues, backbone_mask)
            time_series["per_residue_sidechain_sasa"] = _per_residue_masked_sum(
                atom_sasa, residue_array, n_residues, ~backbone_mask)

        if residue_array is not None and residue_names is not None and residue_values is not None:
            max_asa = _residue_max_asa(residue_names, n_residues)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_residue_rsa = residue_values / max_asa[None, :]
            time_series["per_residue_rsa"] = per_residue_rsa
            time_series["mean_rsa"] = np.nanmean(per_residue_rsa, axis=1)
            has_reference = np.isfinite(max_asa)
            if np.any(has_reference):
                exposed = per_residue_rsa[:, has_reference] > rsa_exposed_threshold
                time_series["exposed_residue_fraction"] = exposed.mean(axis=1)
                time_series["buried_residue_fraction"] = 1.0 - exposed.mean(axis=1)
            else:
                time_series["exposed_residue_fraction"] = np.full(n_frames, np.nan)
                time_series["buried_residue_fraction"] = np.full(n_frames, np.nan)

        if interface_a_mask is not None and interface_b_mask is not None:
            a_mask, b_mask = _interface_masks(interface_a_mask, interface_b_mask, n_atoms)
            interface_values = _interface_sasa_with_engine(
                engine,
                coords,
                radii,
                a_mask,
                b_mask,
                probe_radius,
                n_points,
                residue_ids=residue_array,
                n_residues=n_residues,
                polar_mask=masks.get("polar_exposed_area"),
                apolar_mask=masks.get("hydrophobic_exposed_area"),
            )
            # interface_residue_mask is per-residue, not per-frame: it does
            # not belong in time_series (summarize_time_series() would treat
            # its residue axis as a frame axis).
            interface_residue_mask = interface_values.pop("interface_residue_mask", None)
            time_series.update(interface_values)
            time_series["interface_a_sasa"] = interface_values["interface_a_bound_sasa"]
            time_series["interface_b_sasa"] = interface_values["interface_b_bound_sasa"]

        if glycan is not None and target is not None:
            keep = ~glycan
            target_in_keep = target[keep]
            target_with_glycan = atom_sasa[:, target].sum(axis=1)
            without_glycan = engine.sasa(
                coords[:, keep, :],
                radii[keep],
                probe_radius=probe_radius,
                n_points=n_points,
                atom_sasa=True,
            )[1]
            target_without_glycan = without_glycan[:, target_in_keep].sum(axis=1)
            shielding = target_without_glycan - target_with_glycan
            time_series["glycan_target_sasa"] = target_with_glycan
            time_series["glycan_shielding"] = shielding
            time_series["glycan_shielding_fraction"] = np.divide(
                shielding,
                np.maximum(target_without_glycan, 1e-12),
            )

    statistics = {
        name: summarize_time_series(values, threshold=exposure_threshold)
        for name, values in time_series.items()
    }
    return {
        "time_series": time_series,
        "statistics": statistics,
        "n_frames": n_frames,
        "n_atoms": n_atoms,
        "interface_residue_mask": interface_residue_mask,
        "metadata": {
            "units": "angstrom^2",
            "coordinate_units": "angstrom",
            "algorithm": "shrake_rupley",
            "probe_radius": probe_radius,
            "n_points": n_points,
            "exposure_threshold": exposure_threshold,
            # extract_md_features() takes a caller-supplied radii array with
            # no way to know its provenance (a config file, MDAnalysis, a
            # manual lookup, ...) - "caller_provided" is a fact, not a claim
            # that it is any particular radius set.
            "radii_set": "caller_provided",
            "rsa_reference": "protor_tsai_1999" if residue_names is not None else None,
            "rsa_exposed_threshold": rsa_exposed_threshold if residue_names is not None else None,
        },
    }


def sasa_fingerprint_matrix(features_or_series, keys=("per_residue_sasa",), normalize="none", reference=None):
    """Build a frame-by-feature SASA fingerprint matrix.

    Args:
        features_or_series: Output from ``extract_md_features()``, a time-series
            dict, or a raw array.
        keys: Time-series keys to concatenate.
        normalize: ``"none"``, ``"relative"``, ``"zscore"``, or ``"minmax"``.
        reference: Required for ``"relative"`` normalization.

    Returns:
        Dict with ``matrix`` shape ``(frames, features)``, ``raw_matrix``, and
        feature ``names``.
    """

    if isinstance(features_or_series, dict) and "time_series" in features_or_series:
        series = features_or_series["time_series"]
    elif isinstance(features_or_series, dict):
        series = features_or_series
    else:
        series = {"fingerprint": features_or_series}

    blocks = []
    names = []
    for key in keys:
        if key not in series:
            raise ValueError(f"missing SASA time series: {key}")
        values = np.asarray(series[key], dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape((-1, 1))
        if values.ndim != 2:
            raise ValueError(f"{key} must be one- or two-dimensional")
        blocks.append(values)
        names.extend([f"{key}.{index}" for index in range(values.shape[1])])

    matrix = np.concatenate(blocks, axis=1)
    mode = str(normalize).lower()
    if mode in ("none", "raw", ""):
        normalized = matrix.copy()
    elif mode == "relative":
        if reference is None:
            raise ValueError("relative normalization requires reference values")
        ref = np.asarray(reference, dtype=np.float64).reshape(1, -1)
        if ref.shape[1] != matrix.shape[1]:
            raise ValueError(f"reference must contain {matrix.shape[1]} values")
        normalized = matrix / np.maximum(ref, 1e-12)
    elif mode == "zscore":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        normalized = (matrix - mean) / np.maximum(std, 1e-12)
    elif mode == "minmax":
        lower = matrix.min(axis=0, keepdims=True)
        upper = matrix.max(axis=0, keepdims=True)
        normalized = (matrix - lower) / np.maximum(upper - lower, 1e-12)
    else:
        raise ValueError("normalize must be one of none, relative, zscore, or minmax")

    return {
        "matrix": np.ascontiguousarray(normalized),
        "raw_matrix": np.ascontiguousarray(matrix),
        "names": names,
        "normalize": mode,
    }


def embed_sasa_fingerprints(fingerprints, n_components=2, center=True, scale=False):
    """Project SASA fingerprints with an SVD/PCA-style embedding.

    ``fingerprints`` must have shape ``(frames, features)``. The returned
    ``embedding`` has shape ``(frames, min(n_components, rank))``.
    """

    matrix = np.asarray(fingerprints, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("fingerprints must have shape (frames, features)")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("fingerprints must not be empty")

    working = matrix.copy()
    mean = working.mean(axis=0) if center else np.zeros(matrix.shape[1], dtype=np.float64)
    if center:
        working -= mean
    scale_values = working.std(axis=0) if scale else np.ones(matrix.shape[1], dtype=np.float64)
    if scale:
        working /= np.maximum(scale_values, 1e-12)

    u, singular_values, vt = np.linalg.svd(working, full_matrices=False)
    components = min(n_components, vt.shape[0])
    embedding = u[:, :components] * singular_values[:components]
    variance = (singular_values * singular_values) / max(1, matrix.shape[0] - 1)
    total_variance = variance.sum()
    explained = variance[:components] / total_variance if total_variance > 0.0 else np.zeros(components, dtype=np.float64)

    return {
        "embedding": np.ascontiguousarray(embedding),
        "components": np.ascontiguousarray(vt[:components]),
        "mean": mean,
        "scale": scale_values,
        "explained_variance_ratio": explained,
        "singular_values": singular_values[:components],
    }


class SasaFingerprintEmbedder:
    """A persisted SVD/PCA embedding, fit once and applied to new data.

    ``embed_sasa_fingerprints()`` recomputes its projection from scratch on
    every call, so there is no way to apply the same axes to a held-out
    trajectory. This class fits the projection (mean, scale, components)
    once and reuses it:

    ```python
    embedder = SasaFingerprintEmbedder(n_components=2)
    training_embedding = embedder.fit_transform(training_matrix)
    held_out_embedding = embedder.transform(held_out_matrix)
    ```

    ``transform()`` applies ``(x - mean) / scale @ components.T`` — the
    standard out-of-sample PCA projection, algebraically identical to the
    ``u[:, :k] * s[:k]`` embedding ``embed_sasa_fingerprints()`` returns for
    the data it was fit on (verified: ``fit_transform(x)`` and
    ``fit(x).transform(x)`` agree to floating-point precision).
    """

    def __init__(self, n_components=2, center=True, scale=False):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.center = center
        self.scale = scale
        self.mean_ = None
        self.scale_ = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.feature_names_ = None

    def fit(self, fingerprints, feature_names=None):
        """Fit the projection to ``fingerprints`` (shape ``(frames, features)``).

        ``feature_names`` (the names ``flatten_statistics()`` or
        ``sasa_fingerprint_matrix()`` return alongside the values) is stored
        as ``feature_names_`` and checked by ``transform()`` so a matrix built
        with a different statistic layout is rejected instead of silently
        projected onto the wrong axes.
        """

        self.fit_transform(fingerprints, feature_names=feature_names)
        return self

    def fit_transform(self, fingerprints, feature_names=None):
        """Fit the projection and return its embedding of ``fingerprints``."""

        matrix = np.asarray(fingerprints, dtype=np.float64)
        self.feature_names_ = self._checked_feature_names(feature_names, matrix)
        result = embed_sasa_fingerprints(
            fingerprints, n_components=self.n_components, center=self.center, scale=self.scale
        )
        self.mean_ = result["mean"]
        self.scale_ = result["scale"]
        self.components_ = result["components"]
        self.singular_values_ = result["singular_values"]
        self.explained_variance_ratio_ = result["explained_variance_ratio"]
        self.n_components_ = self.components_.shape[0]
        return result["embedding"]

    @staticmethod
    def _checked_feature_names(feature_names, matrix):
        if feature_names is None:
            return None
        names = tuple(str(name) for name in feature_names)
        if matrix.ndim == 2 and len(names) != matrix.shape[1]:
            raise ValueError(
                f"feature_names has {len(names)} entries but fingerprints have {matrix.shape[1]} features")
        return names

    def transform(self, fingerprints, feature_names=None):
        """Project ``fingerprints`` using the axes learned by ``fit()``.

        When both ``fit()`` and this call received ``feature_names``, they
        must match exactly (same statistics, same order).
        """

        if self.components_ is None:
            raise RuntimeError("SasaFingerprintEmbedder.transform() called before fit()")

        matrix = np.asarray(fingerprints, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("fingerprints must have shape (frames, features)")
        if matrix.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"fingerprints must have {self.mean_.shape[0]} features, got {matrix.shape[1]}")
        names = self._checked_feature_names(feature_names, matrix)
        if names is not None and self.feature_names_ is not None and names != self.feature_names_:
            mismatch = next(
                (index for index, (a, b) in enumerate(zip(names, self.feature_names_)) if a != b),
                min(len(names), len(self.feature_names_)))
            raise ValueError(
                "fingerprint feature layout differs from the one this embedder was fit on "
                f"(first difference at column {mismatch}: {names[mismatch] if mismatch < len(names) else '<missing>'!r} "
                f"vs {self.feature_names_[mismatch] if mismatch < len(self.feature_names_) else '<missing>'!r}); "
                "regenerate the fingerprints with the current summarize_time_series() layout")

        working = matrix - self.mean_ if self.center else matrix
        if self.scale:
            working = working / np.maximum(self.scale_, 1e-12)
        return np.ascontiguousarray(working @ self.components_.T)


def flatten_statistics(statistics, prefix=""):
    """Flatten nested feature statistics into parallel names and values."""

    names = []
    values = []
    base = f"{prefix}." if prefix else ""
    for feature_name in sorted(statistics):
        for stat_name in sorted(statistics[feature_name]):
            stat_values = np.asarray(statistics[feature_name][stat_name], dtype=np.float64).reshape(-1)
            for index, value in enumerate(stat_values):
                suffix = f".{index}" if stat_values.size > 1 else ""
                names.append(f"{base}{feature_name}.{stat_name}{suffix}")
                values.append(float(value))
    return names, np.asarray(values, dtype=np.float64)
