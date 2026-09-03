# Contributing

Thanks for helping improve FastSASA.

## Scope

FastSASA is focused on GPU-accelerated solvent accessible surface area
calculations for structures, MD trajectories, and feature extraction, across
its Vulkan, CUDA, and CPU backends. Changes should keep the package usable
as an independent tool with optional compatibility modes.

## Before Opening A Pull Request

1. Open an issue for large changes or algorithmic changes.
2. Keep changes narrowly scoped.
3. Include a short validation note: what you ran, what passed, and whether a
   GPU was available.
4. Do not commit local runner state, build directories, trajectories, Nsight
   reports, or private datasets.

## Build And Test

Build FastSASA (see [docs/index.md](docs/index.md#build-from-source)) and
run the test suite (see [docs/testing.md](docs/testing.md)). In addition to
that suite, two developer-only CMake targets exist:

GPU-required validation:

```sh
FASTSASA_REQUIRE_GPU_TESTS=1 \
cmake --build build --target fastsasa_cli_validate -j2
```

CPU-only feature-statistics validation:

```sh
cmake --build build --target fastsasa_python_feature_unit -j2
```

## Coding Guidelines

- Prefer existing code style and local APIs.
- Keep CUDA changes measurable: include validation and, when performance is the
  motivation, a benchmark command.
- Keep default behavior conservative and broadly compatible.
- Do not add new SASA algorithms without a validation plan and a clear reason
  they belong in the core package.

## Documentation

User-facing docs live in `README.md` and `docs/`. Update them when changing
CLI behavior, Python return shapes, install steps, or supported workflows.

## Licensing

Keep `LICENSE`, `NOTICE`, and files under `licenses/` intact when redistributing
source or binary packages. New dependencies must have compatible redistribution
terms and visible notices.
