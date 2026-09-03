# FastSASA

<p align="center">
  <img src="docs/images/logo.png" alt="FastSASA" width="760">
</p>

FastSASA calculates solvent-accessible surface area (SASA) for molecular
structures and molecular dynamics trajectories. It provides Shrake-Rupley and
Lee-Richards calculations through a command-line tool, a Python array API, and
VMD integration.

FastSASA is designed for repeated calculations. Trajectory coordinates are
streamed in batches while reusable data stays on the selected backend. Vulkan,
CUDA, and threaded CPU implementations are included; FastSASA chooses an
available backend automatically.

[Documentation](docs/index.md) ·
[CLI reference](docs/cli.md) ·
[Python API](docs/api.md) ·
[Trajectory guide](docs/trajectory.md) ·
[VMD integration](docs/vmd_integration.md)

## Build

Requirements: CMake 3.18 or newer and a C/C++ compiler. GPU support is optional.
Vulkan builds also need Vulkan headers and `glslc`; CUDA builds need the CUDA
toolkit.

```sh
git clone https://github.com/aminkvh/fastsasa.git
cd fastsasa
cmake -S . -B build
cmake --build build -j4
```

Install the Python package from the same checkout:

```sh
python3 -m pip install .
```

Platform-specific build options and test commands are in
[Getting started](docs/index.md).

## Calculate A Structure

```sh
./build/fastsasa --format log structure.pdb
./build/fastsasa --format json --output result.json structure.cif
```

Shrake-Rupley with 100 points and a 1.4 Å probe is the default. Use
`--lee-richards` for Lee-Richards or `--backend cpu|vulkan|cuda` to choose a
backend explicitly.

## Calculate A Trajectory

```sh
./build/fastsasa trajectory \
  --topology topology.psf \
  --trajectory trajectory.dcd \
  --frames : \
  --filter protein \
  --output protein_sasa.csv
```

The default trajectory output is one CSV row per frame. `--filter protein`
defines the atoms included in the calculation; use `--select` when only a
chain, residue, or other subset should be reported in that protein context.
See [Trajectory analysis](docs/trajectory.md) and
[Selection syntax](docs/selection.md).

## Use Python

```python
import numpy as np
from fastsasa import sasa

positions = np.asarray(coordinates, dtype=np.float64)  # (atoms, 3) or (frames, atoms, 3)
radii = np.asarray(atom_radii, dtype=np.float64)       # (atoms,), without probe radius

total_sasa = sasa(positions, radii)
```

The array API does not require a particular trajectory reader. MDAnalysis,
MDTraj, PyMOL, RDKit, feature-extraction, and interface-SASA examples are
linked from the [Python API](docs/api.md).

FastSASA can also turn trajectories into per-residue or named-group exposure
time series, interface burial, glycan shielding, summary statistics, and
PCA/SVD fingerprints. These outputs support plotting, clustering, rare-state
detection, and frame selection. See the
[Feature Extraction Tutorial](docs/tutorial_feature_extraction.md) for the
definitions and worked examples.

## License

FastSASA is MIT-licensed. It includes small third-party or attributed data
components under their respective licenses; see [NOTICE](NOTICE) and
[`licenses/`](licenses/).
