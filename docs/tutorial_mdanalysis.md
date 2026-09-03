# MDAnalysis Tutorial

This tutorial shows the recommended pattern for explicit-solvent MD:

1. Use MDAnalysis to read the system.
2. Apply any wrapping, unwrapping, centering, or alignment transformations.
3. Select the solute atoms.
4. Pass final coordinates and radii to FastSASA.

FastSASA does not perform PBC reconstruction. It computes SASA on the coordinates
you pass in.

## Install Extras

```sh
python3 -m pip install ".[analysis]"
```

## Per-Frame Protein SASA

For drop-in MDAnalysis-style use, FastSASA provides `SASAAnalysis`. The
`select` argument is a normal MDAnalysis selection. Atoms outside that
selection are removed before SASA is calculated.

```python
import MDAnalysis as mda
from fastsasa import SASAAnalysis, load_radius_config

u = mda.Universe("topology.psf", "trajectory.dcd")
radius_config = load_radius_config()

analysis = SASAAnalysis(
    u,
    select="protein",
    radius_config=radius_config,
    n_points=100,
).run()

print(analysis.results.total_area.shape)
print(analysis.results.residue_area.shape)
```

This is the easiest path for existing MDAnalysis workflows. The lower-level
array path below is useful when you want manual batching, custom masks, or
feature extraction.

Compatibility note: wrappers such as MDAKit SASA use MDAnalysis selections to
filter atoms before calling a SASA backend. `SASAAnalysis(..., select=...)`
matches that selection model, while using FastSASA as the backend.

```python
import numpy as np
import MDAnalysis as mda
from fastsasa import SasaEngine
from fastsasa_adapters import load_radius_config, mdanalysis_selection_arrays

u = mda.Universe("topology.psf", "trajectory.dcd")
protein = u.select_atoms("protein")
radius_config = load_radius_config()

print("frame,total_sasa")
with SasaEngine() as engine:
    for ts in u.trajectory:
        positions, radii = mdanalysis_selection_arrays(
            protein,
            radius_config=radius_config,
        )
        total = engine.sasa(
            np.expand_dims(positions, axis=0),
            radii,
            probe_radius=1.4,
            n_points=100,
        )[0]
        print(f"{ts.frame},{total:.6f}")
```

Output shape:

```text
frame,total_sasa
0,....
1,....
2,....
```

## Chain Or Domain Selection

Use MDAnalysis for complex selections:

```python
domain = u.select_atoms("segid PROA and resid 125:300")
positions, radii = mdanalysis_selection_arrays(domain, radius_config=radius_config)
total = SasaEngine().sasa(positions, radii)[0]
```

The equivalent analysis-object form is:

```python
analysis = SASAAnalysis(u, select="segid PROA and resid 125:300").run()
domain_sasa = analysis.results.total_area
```

## Ligand Burial Time Series

```python
import numpy as np
import MDAnalysis as mda
from fastsasa import extract_md_features
from fastsasa_adapters import load_radius_config, mdanalysis_selection_arrays

u = mda.Universe("topology.psf", "trajectory.dcd")
system = u.select_atoms("protein or resname ABU")
ligand = u.select_atoms("resname ABU")
radius_config = load_radius_config()

frames = []
for _ in u.trajectory:
    positions, radii = mdanalysis_selection_arrays(system, radius_config=radius_config)
    frames.append(positions.copy())

system_indices = np.asarray([atom.index for atom in system])
ligand_indices = set(ligand.indices.tolist())
ligand_mask = np.asarray([index in ligand_indices for index in system_indices], dtype=bool)

features = extract_md_features(
    np.asarray(frames, dtype=np.float64),
    radii,
    group_masks={"ligand_sasa": ligand_mask},
    probe_radius=1.4,
    n_points=100,
)

ligand_sasa = features["time_series"]["ligand_sasa"]
print(ligand_sasa.shape)
```

Expected output shape:

```text
(n_frames,)
```

## Command-Line Example Script

The repository includes:

```sh
python examples/mdanalysis_sasa.py topology.psf trajectory.dcd --selection protein
python examples/mdanalysis_features.py topology.psf trajectory.dcd \
  --selection "protein or resname ABU" \
  --interface-a-selection "protein" \
  --interface-b-selection "resname ABU"
python examples/mdanalysis_fingerprint.py topology.psf trajectory.dcd --selection protein
```
