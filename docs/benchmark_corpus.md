# Benchmarking FastSASA

FastSASA provides a reproducible benchmark command for structures and
trajectories. It records timing, SASA results, input size, backend, precision,
CPU, GPU, operating system, and the exact command in CSV form.

The repository contains a small [structure manifest](benchmark_corpus.csv),
not the structure files themselves. Public structures are downloaded from
RCSB when requested. Downloaded inputs and benchmark results are ignored by
git.

## Build

```sh
cmake -S . -B build -DFASTSASA_BUILD_NATIVE_TESTS=ON
cmake --build build -j4
```

## Standard Benchmark

Fetch the standard structures and run the same matrix used by other testers:

```sh
python3 tools/fetch_benchmark_corpus.py \
  --output-dir benchmark_corpus/structures

python3 tools/fastsasa_benchmark.py standard \
  --fastsasa build/fastsasa \
  --profile standard \
  --output-dir profiles/standard_benchmark
```

The main result is:

```text
profiles/standard_benchmark/fastsasa_benchmark_results.csv
```

The output directory also contains the input manifest and `benchmark_run.json`
with system and command metadata.

## Include Public Trajectories

The standard structure run does not download trajectories. Add the trajectory
option when you want end-to-end frame throughput:

```sh
python3 tools/fastsasa_benchmark.py standard \
  --fastsasa build/fastsasa \
  --profile standard \
  --fetch-standard-trajectories \
  --trajectory-frames 100 \
  --trajectory-batches '1 8 32' \
  --output-dir profiles/standard_with_trajectory
```

This fetches one representative trajectory from each configured public
record, not every file in each archive. The standard subset covers both
PDB/XTC and PSF/DCD workflows.

## Benchmark Your Own Trajectory

Use the `suite` mode with `name|topology|trajectory`:

```sh
python3 tools/fastsasa_benchmark.py suite \
  --fastsasa build/fastsasa \
  --backend auto \
  --precision fp64 \
  --trajectory-selection protein \
  --trajectory-batches '1 8 32' \
  --trajectory-frames 100 \
  --trajectories 'run1|topology.psf|trajectory.dcd' \
  --output profiles/run1.csv
```

Trajectory benchmarks require an atom policy. Use `protein` for a normal
protein-only benchmark, `all` for every topology atom, or a FastSASA selection
expression for another system definition.

## Structure Matrix

Use `corpus` mode to choose algorithms, resolutions, backend, and precision:

```sh
python3 tools/fastsasa_benchmark.py corpus \
  --fastsasa build/fastsasa \
  --structure-dir benchmark_corpus/structures \
  --backend vulkan \
  --precision fp32 \
  --points '100 500' \
  --slices '10 20' \
  --output profiles/structures_vulkan_fp32.csv
```

Use `--include-nondefault` for the larger optional structures. Run
`python3 tools/fastsasa_benchmark.py MODE --help` for the complete options for
`standard`, `suite`, or `corpus`.

## Precision Reports

After collecting several structure or trajectory CSVs:

```sh
python3 tools/fastsasa_precision_report.py \
  --input-dir profiles --summary profiles/structure_precision.csv

python3 tools/fastsasa_trajectory_precision_report.py \
  --input-dir profiles \
  --detail profiles/trajectory_precision_detail.csv \
  --summary profiles/trajectory_precision.csv
```

FastSASA benchmark tools measure FastSASA itself. Comparisons with other tools
belong in a separate validation environment so those programs and their data
do not become FastSASA package dependencies.
