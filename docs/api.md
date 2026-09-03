# API Reference

FastSASA accepts NumPy-compatible coordinate and radius arrays. It does not
require a particular structure or trajectory reader. The simplest entry points
are `sasa()` for Shrake-Rupley and `lee_richards()` for Lee-Richards; both use
the available Vulkan, CUDA, or threaded CPU backend automatically.

## CLI Or Python?

Use the CLI when:

- your data is already in PDB, mmCIF, PSF+DCD, or PDB/XTC form;
- you want CSV, JSON, PDB, CIF, XML, or RSA-style files;
- you want to run the standard benchmark package.

Use Python when:

- your trajectory is already loaded in MDAnalysis, MDTraj, or custom arrays;
- you need complex selections or coordinate transformations before SASA;
- you want SASA features for plotting, clustering, or ML workflows.

## Coordinate Contract

Coordinates:

- one structure: `(atoms, 3)`
- trajectory batch: `(frames, atoms, 3)`
- dtype: any NumPy-compatible numeric type, converted to `float64`
- units: Angstrom
- values: finite coordinates only

Radii:

- shape: `(atoms,)`
- units: Angstrom
- pass bare atom radii (no probe radius added in); FastSASA adds `probe_radius`
  itself for every atom
- most users don't build this array by hand: `mdanalysis_selection_arrays()`
  and `mdtraj_frame_arrays()` ([Adapter API](#adapter-api)) return
  `(positions, radii)` directly, applying the bundled ProtOr classifier. Build
  one manually (e.g. from `load_radius_config()` or your own lookup) only for
  a custom radius model.

Precision:

- `SasaEngine()` and one-shot functions use `precision="fp64"` by default
- use `precision="fp32"` for faster reduced-precision kernels, on GPU or CPU
- CPU Shrake-Rupley supports fp32; CPU Lee-Richards is fp64-only — requesting
  fp32 with `lee_richards()` on the CPU backend emits a `RuntimeWarning` and
  computes at fp64 (the CLI prints an equivalent warning to stderr)

Native C contexts are available for Vulkan and CUDA and default to
`FASTSASA_PRECISION_FP64`. The CPU backend uses the separate
`fastsasa_cpu_*` functions. Call
`fastsasa_context_set_precision(context, FASTSASA_PRECISION_FP32)` after
`fastsasa_context_create()` and before any calculation call to select the fast
mode.

Empty Python atom subsets return zero totals and correctly shaped empty
per-atom/group arrays. Frames with coordinate bounds that would require an
impractically large dense cell grid are rejected; prepare wrapped, reimaged, or
filtered coordinates before calling FastSASA.

## `sasa()`

```python
from fastsasa import sasa

total = sasa(
    positions,
    radii,
    probe_radius=1.4,
    n_points=100,
)
```

Computes Shrake-Rupley SASA on the selected accelerator backend.

Common arguments:

| Argument | Meaning |
| --- | --- |
| `positions` | `(atoms, 3)` or `(frames, atoms, 3)` coordinates |
| `radii` | atom radii without probe radius |
| `probe_radius` | solvent probe radius, default `1.4` |
| `n_points` | Shrake-Rupley points per atom |
| `atom_sasa` | return per-atom SASA when true |
| `residue_ids` | atom-to-residue ids, shape `(atoms,)` |
| `selection_masks` | per-atom bit masks for selected-area sums |
| `as_result` | return `SasaResult` instead of arrays/dicts |
| `precision` | `"fp64"` (default) or `"fp32"` |

Return shapes:

| Request | Return |
| --- | --- |
| total only | array `(frames,)` |
| `atom_sasa=True` | `(total, atom)` with atom shape `(frames, atoms)` |
| residue/selection requested | dict with `total`, optional `atom`, `residue`, `selection` |
| `as_result=True` | `SasaResult` object |

## `lee_richards()`

```python
from fastsasa import lee_richards

total = lee_richards(
    positions,
    radii,
    probe_radius=1.4,
    n_slices=20,
)
```

Computes Lee-Richards SASA on the selected accelerator backend. Input and return shapes match `sasa()`.
Use `n_slices` instead of `n_points`.

Shrake-Rupley remains the recommended high-throughput trajectory algorithm.
Lee-Richards is useful when an LR-style reference is required.

## Advanced: `SasaEngine`

```python
from fastsasa import SasaEngine

with SasaEngine(precision="fp64") as engine:
    print(engine.backend)  # "vulkan", "cuda", or "cpu"
    total_1 = engine.sasa(batch_1, radii)
    total_2 = engine.sasa(batch_2, radii)
```

**Most code should use `sasa()` or `lee_richards()` instead.** Those functions
share a cached engine, so calling them repeatedly does not recreate the backend
context each time.

Create `SasaEngine` directly when you need deterministic resource release or
one independent engine per Python worker thread. An engine is not thread-safe
to share:

```python
from fastsasa import SasaEngine
import threading

def worker(batch, radii, results, index):
    with SasaEngine(precision="fp64") as engine:
        results[index] = engine.sasa(batch, radii)

threads = [threading.Thread(target=worker, args=(b, radii, results, i))
           for i, b in enumerate(batches)]
```

It also gives explicit control over when GPU resources are released
(`engine.close()`, or the `with` block above) rather than waiting on
`close_default_engines()` or process exit — useful in a long-running
service or notebook where you don't want to hold VRAM.

For long file-backed DCD/XTC runs, the native `fastsasa trajectory` command
also avoids a Python frame loop.

## `SasaResult`

```python
result = sasa(
    positions,
    radii,
    atom_sasa=True,
    residue_ids=residue_ids,
    as_result=True,
)

result.totalArea(frame=0)
result.atomArea(10, frame=0)
result.atomAreas(frame=0)
result.residueAreas(frame=0)
```

Attributes:

| Attribute | Shape |
| --- | --- |
| `total` | `(frames,)` |
| `atom` | `(frames, atoms)` or `None` |
| `residue` | `(frames, residues)` or `None` |
| `selection` | `(frames, selections)` or `None` |

## Feature API

```python
from fastsasa import extract_md_features, interface_sasa, sasa_fingerprint_matrix, embed_sasa_fingerprints

features = extract_md_features(
    positions,
    radii,
    residue_ids=residue_ids,
    group_masks={"ligand_sasa": ligand_mask},
)

fingerprints = sasa_fingerprint_matrix(features, normalize="zscore")
embedding = embed_sasa_fingerprints(fingerprints["matrix"], n_components=2)
```

Standalone helpers cover kinetics, uncertainty, and reusable fingerprint
embeddings:

```python
from fastsasa import exposure_kinetics, time_series_uncertainty, SasaFingerprintEmbedder

kinetics = exposure_kinetics(features["time_series"]["total_sasa"], threshold=1500.0, hysteresis=50.0)
uncertainty = time_series_uncertainty(features["time_series"]["total_sasa"])

embedder = SasaFingerprintEmbedder(n_components=2)
train_embedding = embedder.fit_transform(train_matrix)
held_out_embedding = embedder.transform(held_out_matrix)
```

`exposure_kinetics()` adds hysteresis-aware transitions and dwell times.
`time_series_uncertainty()` estimates autocorrelation-aware uncertainty and a
block-averaged cross-check. `SasaFingerprintEmbedder` fits one projection and
reuses it for held-out trajectories. Definitions and interpretation are in
the [Feature Extraction Tutorial](tutorial_feature_extraction.md).

Feature outputs are dictionaries containing:

- `time_series`: per-frame feature arrays
- `statistics`: mean, variance, standard deviation, min, max, median,
  quantiles (q05/q25/q75/q95), iqr, mad, exposure frequency, and transition
  counts
- `n_frames`, `n_atoms`
- `metadata`: units, algorithm, probe radius, `n_points`, and (when
  requested) the RSA reference table and threshold used
- `interface_residue_mask`: `None`, or a per-residue (not per-frame) bool
  array when `interface_a_mask`/`interface_b_mask` and `residue_ids` are
  given

Passing `atom_names` and/or `residue_ids` + `residue_names` additionally
enables backbone/sidechain SASA, relative SASA (RSA), and — combined with
`interface_a_mask`/`interface_b_mask` — per-residue and polar/apolar
interface burial. See the
[Feature Extraction Tutorial](tutorial_feature_extraction.md) for the full
set of keys and their definitions.

`interface_sasa()` computes two-group buried/interface SASA directly:

```python
interface = interface_sasa(
    positions,
    radii,
    group_a_mask=protein_mask,
    group_b_mask=ligand_mask,
)
```

Returned time-series keys include:

| Key | Meaning |
| --- | --- |
| `interface_a_free_sasa` | group A SASA in isolation, using its coordinates from the supplied frame |
| `interface_b_free_sasa` | group B SASA in isolation, using its coordinates from the supplied frame |
| `interface_a_bound_sasa` | group A SASA calculated with group B present |
| `interface_b_bound_sasa` | group B SASA calculated with group A present |
| `interface_complex_sasa` | SASA of group A plus group B together |
| `interface_buried_sasa` | `A_free + B_free - SASA(A+B)` |
| `interface_buried_sasa_half` | half of the combined buried SASA |
| `interface_a_buried_fraction` | `interface_a_buried_sasa / interface_a_free_sasa` |
| `interface_b_buried_fraction` | `interface_b_buried_sasa / interface_b_free_sasa` |

`interface_buried_sasa` is the total ΔSASA lost from both sides combined.
`interface_buried_sasa_half` reports the half-ΔSASA convention used by some
interface studies. The per-side values remain available separately because
the two sides need not lose exactly the same sampled area.

Group A and group B define the complete calculation universe for this
function. They should normally be the two complete binding partners. Atoms
outside both masks do not occlude either group.

## Adapter API

Optional helper functions live in `fastsasa_adapters`:

```python
from fastsasa_adapters import (
    SASAAnalysis,
    load_radius_config,
    mdanalysis_residue_ids,
    mdanalysis_selection_arrays,
    mdtraj_frame_arrays,
    sasa_mdanalysis,
    sasa_rdkit_mol,
)
```

These helpers convert external objects into FastSASA arrays. They do not make
MDAnalysis, MDTraj, or RDKit required dependencies.

| Helper | Purpose |
| --- | --- |
| `sasa_rdkit_mol()` | calculate an RDKit conformer, with optional SMARTS or residue output |
| `rdkit_conformer_arrays()` | extract coordinates and van der Waals radii from an RDKit conformer |
| `rdkit_smarts_masks()` | convert SMARTS matches into selection masks |
| `rdkit_residue_ids()` | read residue groups from RDKit PDB metadata |
| `canonical_residue_name()` | map common MD residue variants such as HIE to their standard name |
| `default_radius_config_path()` | locate the installed default radius table |

For PyMOL, there is no dedicated adapter — `cmd.get_model(selection,
state=...)` already returns per-atom coordinates and elements directly, so
`load_radius_config()` is enough to build a radii array. See
[`examples/pymol_sasa.py`](https://github.com/aminkvh/fastsasa/blob/main/examples/pymol_sasa.py).

## MDAnalysis Analysis Object

`SASAAnalysis(universe, select=..., radius_config=..., n_points=...)` is a
standard MDAnalysis analysis class; after `.run()`, results are in
`analysis.results.total_area` (shape `(frames,)`) and
`analysis.results.residue_area` (shape `(frames, selected_residues)`).
It applies the MDAnalysis selection before calculation — the same semantics
as trajectory `--filter`: atoms outside the selected AtomGroup do not bury
the selected atoms. See the
[MDAnalysis tutorial](tutorial_mdanalysis.md) for a worked example.

## Native CPU Fallback API

`fastsasa_cpu.h` exports:

- `fastsasa_cpu_default_threads()`
- `fastsasa_cpu_shrake_rupley(...)` — always fp64
- `fastsasa_cpu_shrake_rupley_precision(...)` — same, with an added
  `precision` argument (`FASTSASA_PRECISION_FP64` or `FASTSASA_PRECISION_FP32`)
- `fastsasa_cpu_lee_richards(...)` — fp64 only, no fp32 variant

The CPU functions use FastSASA-owned cell-list implementations. Passing
`n_threads=0` selects `max(1, detected CPU threads - 1)`. The CLI uses that
automatic worker count unless `--threads N` is supplied. Input and output
arrays stay `double` regardless of precision; fp32 only changes the internal
arithmetic.

## Selection Masks From Metadata

Python users can build FastSASA selection masks from structure metadata without
using MDAnalysis:

```python
from fastsasa import selection_masks_from_metadata

masks, names, warnings = selection_masks_from_metadata(
    ["protein and segid AP and resi 677"],
    atom_names=atom_names,
    residue_names=residue_names,
    residue_numbers=residue_numbers,
    residue_number_strings=residue_labels,
    chain_ids=chain_ids,
    segment_ids=segment_ids,
    elements=elements,
)

result = engine.sasa(xyz, radii, selection_masks=masks, n_selections=len(names))
```

This uses the same native parser as the CLI: the compact selector syntax
described in [Selection Syntax](selection.md), including `protein` and `segid`.
Selection names are optional; expression-only commands get generated names such
as `protein_and_segid_AP_and_resi_677`.
Warnings are returned as FastSASA-branded strings such as
`FastSASA: warning: selection: ...`.

## Small Utilities

- `fibonacci_sphere_points(n_points)` returns the deterministic unit-sphere
  points used by Shrake-Rupley.
- `aggregate_atom_sasa(atom_sasa, masks)` sums per-atom values into named
  groups.
- `summarize_time_series()` and `flatten_statistics()` summarize feature
  arrays and prepare flat names/values for tabular output.
- `RESIDUE_NAME_ALIASES` and `SUMMARY_STATISTIC_NAMES` expose the mappings and
  output order used by those helpers.
- `fastsasa.__version__` reports the installed Python package version.
