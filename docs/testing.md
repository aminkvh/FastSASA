# Verifying A Build

After building FastSASA (see [Getting Started](index.md)), run the test
suite for the backends you built.

## Native Test Suite

```sh
ctest --test-dir build --output-on-failure -L release
```

For a Vulkan build, the `vulkan` label runs the Vulkan-specific tests; they
are skipped automatically when no Vulkan compute device is available:

```sh
ctest --test-dir build --output-on-failure -L vulkan
```

Set `FASTSASA_BACKEND=cuda` to exercise CUDA explicitly on a CUDA+Vulkan
build; normal execution prefers CUDA, then Vulkan.

The exact analytic method has its own test, `fastsasa_exact`. It checks
closed-form cases (an isolated sphere, overlapping and buried pairs,
coincident spheres) and per-atom 1UBQ areas against an independent analytic
implementation (`tests/data/1ubq_exact_reference.csv`). It also checks a
degenerate lattice, thread-count invariance, selection sums, and the
structure and trajectory CLIs, on every backend that is available:

```sh
ctest --test-dir build --output-on-failure -R fastsasa_exact
```

`fastsasa_trajectory_lanes` (and `fastsasa_trajectory_lanes_vulkan`) check
that GPU selection trajectories, run with several frames in flight and
point counts up to 1,024, are bit-identical to one frame per call and to the
CPU.

## Python Install Smoke Test

```sh
python3 -m pip install .
python3 -c "import fastsasa, fastsasa_native; print('FastSASA import ok')"
```

## Sanitizers

CPU-side sanitizers:

```sh
cmake -S . -B build-asan \
  -DFASTSASA_ENABLE_CUDA=OFF \
  -DFASTSASA_ENABLE_SANITIZERS=ON \
  -DFASTSASA_BUILD_NATIVE_TESTS=ON
cmake --build build-asan -j4
ctest --test-dir build-asan --output-on-failure -L release
```

CUDA memory checking, on a machine with a visible GPU:

```sh
compute-sanitizer ./build/fastsasa --format log tests/data/1ubq.pdb
```

## Benchmarking

See [Benchmark Corpus](benchmark_corpus.md) for running and reproducing
FastSASA's benchmarks.
