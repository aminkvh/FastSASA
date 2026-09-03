# Prerequisites And Compatibility

## Toolchain

- CMake 3.18+
- A C11/C++17 compiler (GCC, Clang, or MSVC)
- Python 3.9+ with NumPy for the Python API; Python is not required for the
  native CLI
- Optional: CUDA 12.x (NVIDIA GPU backend), Vulkan loader + headers + `glslc`
  (Vulkan GPU backend). Neither is required — FastSASA falls back to the
  threaded CPU backend when no GPU backend is available.

## OS / Backend Support Matrix

This matrix reflects what CI actually builds and runs, not a general claim.

| OS | CPU | Vulkan | CUDA |
| --- | --- | --- | --- |
| Ubuntu | tested | tested in CI and on a GPU | tested with CUDA 12.0, 12.2, and 12.4; GPU tests use a self-hosted runner |
| Windows | tested | compile-tested | compile-tested |
| macOS | tested | tested through MoltenVK | not supported |
| Rocky Linux 9 | tested | tested with Mesa | not covered in CI |

Software Vulkan devices in CI verify correctness, not GPU throughput. CUDA
execution requires an NVIDIA GPU and a compatible driver.

## Python Extras

```sh
pip install ".[analysis]"   # MDAnalysis + mdtraj, for the trajectory adapters
```

PyMOL and RDKit are not declared as extras — install them from their own
distribution (`conda install -c conda-forge pymol-open-source`, `pip install
rdkit`) if you use the corresponding example script.

## Where To Look Next

- [CLI Reference](cli.md) for building and running the command-line tool.
- [API Reference](api.md) for the Python array API.
- [Radius Configuration](classifier_config.md) for the bundled ProtOr table
  and the element-radius fallback.
