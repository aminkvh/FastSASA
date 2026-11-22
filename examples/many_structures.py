#!/usr/bin/env python3
"""SASA for many structure files: one single-threaded FastSASA process per file.

For a batch of small structures, running several files at once with one CPU
thread each is faster than running them one after another with all threads,
because each small calculation has little work to split across threads and
process start-up and parsing are paid once per file either way.

    python examples/many_structures.py structures/*.pdb --jobs 8 > sasa.csv

Writes CSV: structure,atoms,total_sasa,seconds. Extra FastSASA options can be
passed after "--", e.g. "-- --resolution 500 --classes".
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def run_one(exe: str, path: str, extra: list[str]) -> tuple[str, int, float, float]:
    command = [exe, "--backend", "cpu", "--threads", "1", "--format", "json", *extra, path]
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{path}: {proc.stderr.strip()}")
    result = json.loads(proc.stdout)
    return path, len(result["atoms"]), float(result["total_sasa"]), seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("structures", nargs="+", help="PDB or mmCIF files")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="files processed at once (default: CPU threads - 1)")
    parser.add_argument("--exe", default=os.environ.get("FASTSASA_EXE") or shutil.which("fastsasa"),
                        help="fastsasa executable (default: FASTSASA_EXE or PATH)")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not args.exe:
        parser.error("fastsasa not found; pass --exe or set FASTSASA_EXE")

    print("structure,atoms,total_sasa,seconds")
    failed = 0
    # Threads only wait on the child processes, so a thread pool is enough.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, args.exe, path, extra) for path in args.structures]
        for future in futures:
            try:
                path, atoms, total, seconds = future.result()
                print(f"{path},{atoms},{total:.6f},{seconds:.4f}")
            except RuntimeError as error:
                print(error, file=sys.stderr)
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
