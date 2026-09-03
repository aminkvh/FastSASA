#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _assert_close(actual, expected, name):
    if not np.allclose(actual, expected):
        raise SystemExit(f"{name} mismatch: {actual!r} != {expected!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-dir", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.python_dir))

    from fastsasa_features import (
        SasaFingerprintEmbedder,
        aggregate_atom_sasa,
        embed_sasa_fingerprints,
        exposure_kinetics,
        extract_md_features,
        flatten_statistics,
        _interface_masks,
        sasa_fingerprint_matrix,
        summarize_time_series,
        time_series_uncertainty,
    )

    atom_sasa = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 1.0, 5.0],
            [0.5, 1.5, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    masks = {
        "ligand": np.asarray([False, True, False, True]),
        "polar": np.asarray([False, False, True, True]),
    }
    grouped = aggregate_atom_sasa(atom_sasa, masks)
    _assert_close(grouped["ligand"], [6.0, 5.0, 3.5], "ligand aggregate")
    _assert_close(grouped["polar"], [7.0, 6.0, 2.0], "polar aggregate")

    one_frame = aggregate_atom_sasa(atom_sasa[0], {"first_two": [True, True, False, False]})
    _assert_close(one_frame["first_two"], [3.0], "one-frame aggregate")

    sample = np.asarray([0.0, 2.0, 1.0, 3.0])
    stats = summarize_time_series(sample, threshold=1.5)
    _assert_close(stats["mean"], [1.5], "mean")
    _assert_close(stats["variance"], [1.25], "variance")
    _assert_close(stats["exposure_frequency"], [0.5], "exposure frequency")
    if int(stats["transitions"][0]) != 3:
        raise SystemExit("transition count mismatch")
    _assert_close(stats["median"], [np.median(sample)], "median")
    q05, q25, q75, q95 = np.percentile(sample, [5, 25, 75, 95])
    _assert_close(stats["q05"], [q05], "q05")
    _assert_close(stats["q25"], [q25], "q25")
    _assert_close(stats["q75"], [q75], "q75")
    _assert_close(stats["q95"], [q95], "q95")
    _assert_close(stats["iqr"], [q75 - q25], "iqr")
    _assert_close(stats["mad"], [np.median(np.abs(sample - np.median(sample)))], "mad")

    matrix_stats = summarize_time_series(
        np.asarray([[0.0, 2.0], [2.0, 2.0], [2.0, 0.0]], dtype=np.float64),
        threshold=1.0,
    )
    _assert_close(matrix_stats["transitions"], [1, 1], "matrix transitions")

    a_mask, b_mask = _interface_masks(
        [True, True, False, False],
        [False, False, True, True],
        4,
    )
    if not np.array_equal(a_mask | b_mask, [True, True, True, True]):
        raise SystemExit("interface mask union mismatch")
    try:
        _interface_masks([True, False, False, False], [True, False, False, False], 4)
    except ValueError as exc:
        if "must not overlap" not in str(exc):
            raise
    else:
        raise SystemExit("overlapping interface masks were accepted")
    try:
        _interface_masks([False, False, False, False], [False, False, True, True], 4)
    except ValueError as exc:
        if "group_a_mask must select" not in str(exc):
            raise
    else:
        raise SystemExit("empty interface group was accepted")

    try:
        extract_md_features(
            np.zeros((1, 4, 3), dtype=np.float64),
            np.full(4, 1.7, dtype=np.float64),
            glycan_mask=[True, False, False, False],
            glycan_target_mask=[True, False, False, False],
        )
    except ValueError as exc:
        if "must not overlap" not in str(exc):
            raise
    else:
        raise SystemExit("overlapping glycan masks reached the compute engine")

    try:
        aggregate_atom_sasa(atom_sasa, {"bad": [True, False]})
    except ValueError as exc:
        if "bad must have shape" not in str(exc):
            raise
    else:
        raise SystemExit("invalid mask shape was accepted")

    names, values = flatten_statistics({"ligand": stats}, prefix="md")
    if names != [
        "md.ligand.exposure_frequency",
        "md.ligand.iqr",
        "md.ligand.mad",
        "md.ligand.max",
        "md.ligand.mean",
        "md.ligand.median",
        "md.ligand.min",
        "md.ligand.q05",
        "md.ligand.q25",
        "md.ligand.q75",
        "md.ligand.q95",
        "md.ligand.std",
        "md.ligand.transitions",
        "md.ligand.variance",
    ]:
        raise SystemExit(f"unexpected flattened names: {names!r}")
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise SystemExit("flattened values are invalid")

    features = {
        "time_series": {
            "per_residue_sasa": np.asarray(
                [
                    [10.0, 2.0, 0.0],
                    [8.0, 4.0, 1.0],
                    [4.0, 8.0, 2.0],
                ],
                dtype=np.float64,
            ),
            "ligand_sasa": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        }
    }
    fingerprint = sasa_fingerprint_matrix(features, keys=("per_residue_sasa", "ligand_sasa"), normalize="zscore")
    if fingerprint["matrix"].shape != (3, 4) or len(fingerprint["names"]) != 4:
        raise SystemExit("fingerprint matrix shape mismatch")
    _assert_close(fingerprint["matrix"].mean(axis=0), np.zeros(4), "fingerprint zscore mean")

    relative = sasa_fingerprint_matrix(
        features,
        keys=("per_residue_sasa",),
        normalize="relative",
        reference=np.asarray([10.0, 8.0, 2.0]),
    )
    _assert_close(relative["matrix"][0], [1.0, 0.25, 0.0], "relative fingerprint")

    embedding = embed_sasa_fingerprints(fingerprint["matrix"], n_components=2)
    if embedding["embedding"].shape != (3, 2) or embedding["components"].shape != (2, 4):
        raise SystemExit("SASA embedding shape mismatch")
    if not np.all(np.isfinite(embedding["explained_variance_ratio"])):
        raise SystemExit("SASA embedding variance is invalid")

    # exposure_kinetics: hysteresis=0.0 must reproduce summarize_time_series()'s
    # transitions exactly, including on the same fixture used above.
    kinetics_compat = exposure_kinetics(sample, threshold=1.5, hysteresis=0.0)
    if int(kinetics_compat["transitions"][0]) != int(stats["transitions"][0]):
        raise SystemExit("exposure_kinetics hysteresis=0 transition count diverged from summarize_time_series")

    rng = np.random.default_rng(0)
    for _ in range(200):
        trial = rng.normal(size=int(rng.integers(2, 50)))
        threshold = float(rng.normal())
        expected = summarize_time_series(trial, threshold=threshold)["transitions"][0]
        actual = exposure_kinetics(trial, threshold=threshold, hysteresis=0.0)["transitions"][0]
        if expected != actual:
            raise SystemExit(f"exposure_kinetics/summarize_time_series transition mismatch: {actual} != {expected}")

    # Dwell time excludes boundary-truncated runs: exposed(3, boundary) buried(4)
    # exposed(5) buried(2, boundary) -> only the interior buried(4)/exposed(5) count.
    dwell_series = np.asarray([2.0] * 3 + [-1.0] * 4 + [2.0] * 5 + [-1.0] * 2)
    dwell = exposure_kinetics(dwell_series, threshold=0.0)
    _assert_close(dwell["buried_dwell_time_mean"], [4.0], "buried dwell time")
    _assert_close(dwell["exposed_dwell_time_mean"], [5.0], "exposed dwell time")

    no_interior_run = exposure_kinetics(np.asarray([2.0, 2.0, -1.0, -1.0]), threshold=0.0)
    if not np.isnan(no_interior_run["exposed_dwell_time_mean"][0]) or not np.isnan(no_interior_run["buried_dwell_time_mean"][0]):
        raise SystemExit("exposure_kinetics should return NaN dwell times with no complete interior run")

    # time_series_uncertainty: i.i.d. noise -> tau close to 1; AR(1, phi=0.8) ->
    # tau close to the closed-form (1+phi)/(1-phi) = 9.
    iid = rng.normal(size=5000)
    iid_uncertainty = time_series_uncertainty(iid)
    if not (0.5 < iid_uncertainty["autocorrelation_time"][0] < 2.0):
        raise SystemExit(f"i.i.d. autocorrelation time out of range: {iid_uncertainty['autocorrelation_time'][0]}")

    phi = 0.8
    n_ar = 200000
    ar1 = np.empty(n_ar)
    ar1[0] = rng.normal()
    innovations = rng.normal(size=n_ar) * np.sqrt(1.0 - phi * phi)
    for i in range(1, n_ar):
        ar1[i] = phi * ar1[i - 1] + innovations[i]
    ar1_uncertainty = time_series_uncertainty(ar1)
    theoretical_tau = (1.0 + phi) / (1.0 - phi)
    if abs(ar1_uncertainty["autocorrelation_time"][0] - theoretical_tau) > 1.5:
        raise SystemExit(
            f"AR(1) autocorrelation time diverged from theory: "
            f"{ar1_uncertainty['autocorrelation_time'][0]} vs {theoretical_tau}"
        )

    constant_uncertainty = time_series_uncertainty(np.full(50, 3.0))
    if constant_uncertainty["autocorrelation_time"][0] != 1.0 or constant_uncertainty["standard_error"][0] != 0.0:
        raise SystemExit("constant-series uncertainty should give tau=1.0 and standard_error=0.0")

    short_uncertainty = time_series_uncertainty(np.asarray([1.0, 2.0, 3.0]))
    if not np.isnan(short_uncertainty["autocorrelation_time"][0]) or not np.isnan(short_uncertainty["block_standard_error"][0]):
        raise SystemExit("short-series uncertainty should return NaN")

    # SasaFingerprintEmbedder: fit_transform must match embed_sasa_fingerprints()
    # exactly, and fit().transform(same data) must match fit_transform().
    embed_matrix = rng.normal(size=(40, 12))
    direct_embedding = embed_sasa_fingerprints(embed_matrix, n_components=3)["embedding"]
    embedder = SasaFingerprintEmbedder(n_components=3)
    class_embedding = embedder.fit_transform(embed_matrix)
    _assert_close(class_embedding, direct_embedding, "SasaFingerprintEmbedder.fit_transform")

    embedder_two_step = SasaFingerprintEmbedder(n_components=3)
    embedder_two_step.fit(embed_matrix)
    _assert_close(embedder_two_step.transform(embed_matrix), direct_embedding, "SasaFingerprintEmbedder.fit().transform()")

    held_out = rng.normal(size=(10, 12))
    if not np.array_equal(embedder_two_step.transform(held_out), embedder_two_step.transform(held_out)):
        raise SystemExit("SasaFingerprintEmbedder.transform() is not deterministic")

    try:
        SasaFingerprintEmbedder().transform(embed_matrix)
    except RuntimeError as exc:
        if "before fit" not in str(exc):
            raise
    else:
        raise SystemExit("transform() before fit() should raise")

    try:
        embedder_two_step.transform(rng.normal(size=(5, 99)))
    except ValueError as exc:
        if "must have" not in str(exc):
            raise
    else:
        raise SystemExit("transform() with mismatched feature width should raise")

    _check_protor_table_matches_cli(args.python_dir)
    _check_residue_aliases(args.python_dir)
    _check_summary_layout_guard()

    print("fastsasa_python_feature_unit,status,pass")
    return 0


def _check_residue_aliases(python_dir: Path) -> None:
    """MD residue variants resolve to standard residues for radii and RSA
    references, the Python and C alias tables agree, and unknown residues
    warn once instead of silently producing NaN."""
    import re
    import warnings

    from fastsasa_adapters import RESIDUE_NAME_ALIASES, RadiusConfig, canonical_residue_name
    from fastsasa_features import _PROTOR_MAX_ASA, _residue_max_asa

    source = (python_dir.resolve().parent / "src" / "fastsasa_radius.c").read_text()
    c_aliases = dict(re.findall(r'\{"([A-Z]{3})", "([A-Z]{3})"\}', source))
    if not c_aliases:
        raise SystemExit("no residue alias table found in src/fastsasa_radius.c")
    if c_aliases != RESIDUE_NAME_ALIASES:
        raise SystemExit(f"RESIDUE_NAME_ALIASES {RESIDUE_NAME_ALIASES} differs from the C table {c_aliases}")
    if canonical_residue_name(" hie ") != "HIS" or canonical_residue_name("ala") != "ALA":
        raise SystemExit("canonical_residue_name() did not normalise names as expected")

    config = RadiusConfig({"HISCA": 1.88, "CYSSG": 1.77}, {("HIS", "CA"): "HISCA", ("CYS", "SG"): "CYSSG"})
    if config.radius("HIE", "CA") != 1.88 or config.radius("CYX", "SG") != 1.77:
        raise SystemExit("RadiusConfig.radius() did not resolve MD residue variants")
    if config.radius("HIS", "CA") != 1.88:
        raise SystemExit("RadiusConfig.radius() exact lookup regressed")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = _residue_max_asa(["HIE", "CYX", "ASH", "ZZZ"], 4)
    expected = [_PROTOR_MAX_ASA["HIS"]["total"], _PROTOR_MAX_ASA["CYS"]["total"], _PROTOR_MAX_ASA["ASP"]["total"]]
    if list(values[:3]) != expected or not np.isnan(values[3]):
        raise SystemExit(f"_residue_max_asa() alias resolution failed: {values}")
    if not any("ZZZ" in str(w.message) for w in caught):
        raise SystemExit("_residue_max_asa() did not warn about an unknown residue name")


def _check_summary_layout_guard() -> None:
    """SUMMARY_STATISTIC_NAMES matches summarize_time_series(), and the
    embedder rejects a matrix whose feature layout differs from the fit."""
    from fastsasa_features import (
        SUMMARY_STATISTIC_NAMES,
        SasaFingerprintEmbedder,
        flatten_statistics,
        summarize_time_series,
    )

    statistics = {"f": summarize_time_series(np.arange(12.0).reshape(12, 1))}
    names, values = flatten_statistics(statistics)
    if tuple(name.split(".", 1)[1] for name in names) != SUMMARY_STATISTIC_NAMES:
        raise SystemExit(f"SUMMARY_STATISTIC_NAMES is stale: flatten_statistics() emits {names}")

    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(8, 4))
    feature_names = ["a.mean", "a.std", "b.mean", "b.std"]
    embedder = SasaFingerprintEmbedder(n_components=2).fit(matrix, feature_names=feature_names)
    if embedder.feature_names_ != tuple(feature_names):
        raise SystemExit("SasaFingerprintEmbedder did not record feature_names_")
    embedder.transform(matrix, feature_names=feature_names)
    embedder.transform(matrix)
    try:
        embedder.transform(matrix, feature_names=["a.mean", "a.median", "b.mean", "b.std"])
    except ValueError as exc:
        if "a.median" not in str(exc):
            raise
    else:
        raise SystemExit("SasaFingerprintEmbedder.transform() accepted a different feature layout")
    try:
        SasaFingerprintEmbedder(n_components=2).fit(matrix, feature_names=feature_names[:3])
    except ValueError:
        pass
    else:
        raise SystemExit("SasaFingerprintEmbedder.fit() accepted mismatched feature_names length")


def _check_protor_table_matches_cli(python_dir: Path) -> None:
    """Every field of the Python max-ASA table must equal the C table.

    The C table in tools/fastsasa_cli.cc is the one --format rsa uses and the
    one validated byte-for-byte against FreeSASA; the Python copy exists so
    callers need no subprocess. Field order in the C initializer is the
    rsa_area struct order: total, main_chain, side_chain, polar, apolar.
    """
    import re

    from fastsasa_features import _PROTOR_MAX_ASA

    source = (python_dir.resolve().parent / "tools" / "fastsasa_cli.cc").read_text()
    rows = re.findall(
        r'\{"([A-Z]{3})", \{([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+), 0\.0\}\}', source)
    if len(rows) != 20:
        raise SystemExit(f"expected 20 ProtOr reference rows in fastsasa_cli.cc, found {len(rows)}")
    fields = ("total", "main_chain", "side_chain", "polar", "apolar")
    for residue, *values in rows:
        python_row = _PROTOR_MAX_ASA.get(residue)
        if python_row is None:
            raise SystemExit(f"_PROTOR_MAX_ASA is missing {residue}")
        for field, value in zip(fields, values):
            if python_row[field] != float(value):
                raise SystemExit(
                    f"_PROTOR_MAX_ASA[{residue}][{field}] = {python_row[field]} but the C table has {value}")
    if len(_PROTOR_MAX_ASA) != 20:
        raise SystemExit(f"_PROTOR_MAX_ASA has {len(_PROTOR_MAX_ASA)} residues, expected 20")


if __name__ == "__main__":
    raise SystemExit(main())
