"""Public Python API for FastSASA.

The functions exported here operate on NumPy-compatible arrays. Coordinates
must have shape ``(atoms, 3)`` for one structure or ``(frames, atoms, 3)`` for a
trajectory batch. Radii must have shape ``(atoms,)`` and should not include the
probe radius; FastSASA adds ``probe_radius`` internally.
"""

from fastsasa_native import (
    SasaEngine,
    SasaResult,
    fibonacci_sphere_points,
    lee_richards,
    sasa,
    selection_masks_from_metadata,
)
from fastsasa_features import (
    SUMMARY_STATISTIC_NAMES,
    SasaFingerprintEmbedder,
    aggregate_atom_sasa,
    embed_sasa_fingerprints,
    exposure_kinetics,
    extract_md_features,
    flatten_statistics,
    interface_sasa,
    residue_sasa,
    sasa_fingerprint_matrix,
    summarize_time_series,
    time_series_uncertainty,
)
from fastsasa_adapters import (
    RESIDUE_NAME_ALIASES,
    SASAAnalysis,
    canonical_residue_name,
    default_radius_config_path,
    load_radius_config,
    mdanalysis_residue_ids,
    mdanalysis_selection_arrays,
    mdtraj_frame_arrays,
    rdkit_conformer_arrays,
    rdkit_residue_ids,
    rdkit_smarts_masks,
    sasa_mdanalysis,
    sasa_rdkit_mol,
)

__version__ = "0.1.0rc19"

__all__ = [
    "SasaEngine",
    "SasaResult",
    "SASAAnalysis",
    "SasaFingerprintEmbedder",
    "RESIDUE_NAME_ALIASES",
    "SUMMARY_STATISTIC_NAMES",
    "__version__",
    "aggregate_atom_sasa",
    "canonical_residue_name",
    "embed_sasa_fingerprints",
    "default_radius_config_path",
    "exposure_kinetics",
    "extract_md_features",
    "fibonacci_sphere_points",
    "flatten_statistics",
    "interface_sasa",
    "lee_richards",
    "load_radius_config",
    "mdanalysis_residue_ids",
    "mdanalysis_selection_arrays",
    "mdtraj_frame_arrays",
    "rdkit_conformer_arrays",
    "rdkit_residue_ids",
    "rdkit_smarts_masks",
    "residue_sasa",
    "sasa",
    "sasa_mdanalysis",
    "sasa_rdkit_mol",
    "sasa_fingerprint_matrix",
    "selection_masks_from_metadata",
    "summarize_time_series",
    "time_series_uncertainty",
]
