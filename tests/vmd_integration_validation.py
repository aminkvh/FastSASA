#!/usr/bin/env python3
"""Headless VMD checks for integration/vmd/fastsasa/fastsasa.tcl.

Covers: fp64 default precision (fp32 still selectable), and
"-restrict SEL2 -frames all" matching "-restrict SEL2" per frame.
Skips (exit 0) when VMD is not installed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TCL = r"""
lappend auto_path {integration}
package require fastsasa
# a user ~/.vmdrc may set FASTSASA_EXE; force the logging shim
set ::env(FASTSASA_EXE) {shim}
mol new {pdb}
for {{set i 1}} {{$i < 4}} {{incr i}} {{
    mol addfile {pdb} waitfor all
    set moved [atomselect top "resid < [expr {{10 + $i * 10}}]" frame $i]
    $moved moveby [list [expr {{$i * 1.5}}] [expr {{-$i * 1.0}}] [expr {{$i * 2.0}}]]
    $moved delete
}}
set sel [atomselect top protein]
set part [atomselect top "resid 20 to 40"]
set outside [atomselect top "water or resid 1000"]
set all_frames [fastsasa -sel $sel -restrict $part -frames all]
puts "RESTRICT_ALL $all_frames"
for {{set i 0}} {{$i < 4}} {{incr i}} {{
    $sel frame $i
    $part frame $i
    puts "RESTRICT_NOW $i [fastsasa -sel $sel -restrict $part]"
}}
puts "PART_COUNT [$part num]"
$sel frame 0
puts "UNRESTRICTED [fastsasa -sel $sel -frames all]"
puts "SAME_AS_SEL [fastsasa -sel $sel -restrict $sel -frames all]"
puts "EMPTY_RESTRICT [fastsasa -sel $sel -restrict $outside -frames all]"
puts "FP32 [fastsasa -sel $sel -precision fp32]"
if {{[catch {{fastsasa -sel $sel -peratom pa -frames all}} msg]}} {{puts "PERATOM_ERR $msg"}}
if {{[catch {{fastsasa -sel $sel -points pp -frames all}} msg]}} {{puts "POINTS_ERR $msg"}}
quit
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--vmd", default=os.environ.get("VMD", "vmd"))
    args = parser.parse_args()
    vmd = shutil.which(args.vmd)
    if vmd is None:
        print("fastsasa_vmd_integration_validation,status,skip,reason,no_vmd")
        return 0
    fastsasa = args.fastsasa.resolve()

    with tempfile.TemporaryDirectory(prefix="fastsasa_vmd_test_") as tmp_name:
        tmp = Path(tmp_name)
        log = tmp / "args.log"
        shim = tmp / "fastsasa_shim"
        shim.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\nexec "{fastsasa}" "$@"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
        script = tmp / "test.tcl"
        script.write_text(TCL.format(
            shim=shim,
            integration=REPO / "integration" / "vmd",
            pdb=REPO / "tests" / "data" / "1ubq.pdb"))
        env = dict(os.environ, FASTSASA_EXE=str(shim), TMPDIR=str(tmp),
                   FASTSASA_BACKEND=os.environ.get("FASTSASA_BACKEND", "cpu"))
        proc = subprocess.run([vmd, "-dispdev", "text", "-eofexit", "-e", str(script)],
                              text=True, capture_output=True, env=env, check=False)
        out = {}
        for line in proc.stdout.splitlines():
            key, _, rest = line.partition(" ")
            out.setdefault(key, []).append(rest)
        invocations = log.read_text().splitlines() if log.exists() else []

    def fail(message: str) -> int:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.stderr.write(f"FAIL: {message}\n")
        return 1

    for key in ("RESTRICT_ALL", "UNRESTRICTED", "SAME_AS_SEL", "EMPTY_RESTRICT",
                "PART_COUNT", "FP32"):
        if key not in out:
            return fail(f"missing output {key}")
    all_frames = [float(v) for v in out["RESTRICT_ALL"][0].split()]
    now = [float(row.split()[1]) for row in out.get("RESTRICT_NOW", [])]
    if len(all_frames) != 4 or len(now) != 4:
        return fail(f"expected 4 frames, got {len(all_frames)} and {len(now)}")
    # -frames now round-trips per-atom SASA through the PDB B-factor column
    # (0.01 A^2 resolution), so allow half a unit per restricted atom.
    tolerance = 0.005 * int(out["PART_COUNT"][0])
    for frame, (a, n) in enumerate(zip(all_frames, now)):
        if abs(a - n) > tolerance:
            return fail(f"frame {frame}: -frames all {a!r} != -frames now {n!r}")
    if len(set(all_frames)) < 4:
        return fail("restricted SASA identical across moved frames; frames not distinct")
    unrestricted = [float(v) for v in out["UNRESTRICTED"][0].split()]
    same = [float(v) for v in out["SAME_AS_SEL"][0].split()]
    if any(abs(u - s) > 1e-6 for u, s in zip(unrestricted, same)) or len(same) != 4:
        return fail("-restrict equal to -sel differs from unrestricted totals")
    if any(a >= u for a, u in zip(all_frames, unrestricted)):
        return fail("restricted total is not smaller than the full total")
    if [float(v) for v in out["EMPTY_RESTRICT"][0].split()] != [0.0] * 4:
        return fail("restrict outside -sel should give zeros per frame")
    trajectory_calls = [c for c in invocations if " trajectory " in f" {c} "]
    if not trajectory_calls or not all("--precision fp64" in c for c in trajectory_calls):
        return fail(f"default precision is not fp64: {invocations}")
    structure_calls = [c for c in invocations if " trajectory " not in f" {c} "]
    if not any("--precision fp32" in c for c in structure_calls):
        return fail("explicit -precision fp32 was not passed through")
    # one CLI process for the restricted -frames all call
    if sum(1 for c in trajectory_calls if "--select" in c) != 2:
        return fail("expected exactly one --select trajectory run per restricted -frames all call (two calls made)")
    for key in ("PERATOM_ERR", "POINTS_ERR"):
        message = out.get(key, [""])[0]
        if "-restrict" in message or "support -frames now only" not in message:
            return fail(f"{key} message unexpected: {message!r}")
    print("fastsasa_vmd_integration_validation,status,pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
