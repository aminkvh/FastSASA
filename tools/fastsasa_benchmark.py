#!/usr/bin/env python3
"""Single entry point for FastSASA's benchmark tooling.

Three modes, in order of how much control they give you:

  corpus    Bare FastSASA-only benchmark over the public structure corpus,
            one backend/precision per run. The primitive the other two
            modes are built on. See `fastsasa_benchmark.py corpus --help`.

  standard  A fixed, reproducible structure(+trajectory) profile: the same
            public structures, algorithms, and settings every time, with
            an environment capture (CUDA toolchain, nvidia-smi) alongside
            the results. Use this for consistent, cross-machine numbers.
            See `fastsasa_benchmark.py standard --help`.

  suite     The flexible option: custom structures and trajectories,
            multiple SR/LR resolutions, multiple trajectory batch sizes,
            in one sweep. `standard` calls this internally for its
            trajectory pass. See `fastsasa_benchmark.py suite --help`.

Each mode's flags are unchanged from what were previously three separate
standalone scripts, now tools/benchmarks/{corpus,standard,suite}.py; this
wrapper gives them one shared command name. See docs/benchmark_corpus.md
for full usage examples.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODES = ("corpus", "standard", "suite")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(__doc__)
        print(f"usage: {Path(__file__).name} {{{','.join(MODES)}}} [args...]", file=sys.stderr)
        return 1

    mode = sys.argv[1]
    sys.argv = [f"{Path(__file__).name} {mode}", *sys.argv[2:]]

    if mode == "corpus":
        from benchmarks import corpus as module
    elif mode == "standard":
        from benchmarks import standard as module
    else:
        from benchmarks import suite as module

    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
