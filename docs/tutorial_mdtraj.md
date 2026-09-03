# MDTraj Tutorial

MDTraj can read many trajectory formats and returns coordinates in nanometers.
FastSASA expects Angstrom, so convert MDTraj coordinates by multiplying by `10.0`.

## Install Extras

```sh
python3 -m pip install ".[analysis]"
```

## Per-Frame SASA

```python
import numpy as np
import mdtraj as md
from fastsasa import SasaEngine
from fastsasa_adapters import element_radii

traj = md.load("trajectory.xtc", top="topology.pdb")
positions = np.ascontiguousarray(traj.xyz, dtype=np.float64) * 10.0
radii = element_radii([atom.element.symbol for atom in traj.topology.atoms])

with SasaEngine() as engine:
    total = engine.sasa(
        positions,
        radii,
        probe_radius=1.4,
        n_points=100,
    )

print(total.shape)
```

Expected output shape:

```text
(n_frames,)
```

## Per-Residue SASA

```python
import numpy as np
import mdtraj as md
from fastsasa import sasa
from fastsasa_adapters import element_radii

traj = md.load("trajectory.xtc", top="topology.pdb")
positions = np.ascontiguousarray(traj.xyz, dtype=np.float64) * 10.0
radii = element_radii([atom.element.symbol for atom in traj.topology.atoms])

residues = list(traj.topology.residues)
residue_lookup = {residue: index for index, residue in enumerate(residues)}
residue_ids = np.asarray(
    [residue_lookup[atom.residue] for atom in traj.topology.atoms],
    dtype=np.int32,
)

result = sasa(
    positions,
    radii,
    residue_ids=residue_ids,
    n_residues=len(residues),
)

print(result["residue"].shape)
```

Expected output shape:

```text
(n_frames, n_residues)
```

## Command-Line Example Scripts

```sh
python examples/mdtraj_sasa.py trajectory.xtc --topology topology.pdb
python examples/mdtraj_features.py trajectory.xtc --topology topology.pdb \
  --interface-a-selection "protein" --interface-b-selection "resname ABU"
```

## Selection Policy

Use MDTraj selections to create coordinate subsets or boolean masks. FastSASA does
not attempt to reimplement MDTraj's selection language.
