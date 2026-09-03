#!/usr/bin/env python3
"""Validate the integrated Vulkan fallback against the threaded CPU path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def invoke(fastsasa: Path, root: Path, backend: str, *arguments: str,
           extra_environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["FASTSASA_BACKEND"] = backend
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [str(fastsasa), *arguments],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )


def run(fastsasa: Path, root: Path, backend: str, *arguments: str) -> dict:
    process = invoke(fastsasa, root, backend, *arguments)
    if process.returncode != 0:
        raise RuntimeError(process.stderr or process.stdout)
    return json.loads(process.stdout)


def close(
    label: str,
    expected: float,
    actual: float,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> None:
    difference = abs(expected - actual)
    tolerance = max(absolute_tolerance, abs(expected) * relative_tolerance)
    if difference > tolerance:
        raise RuntimeError(
            f"{label}: {actual:.12f} vs {expected:.12f}; "
            f"difference {difference:.6g} exceeds {tolerance:.6g}"
        )


def close_atoms(label: str, expected: dict, actual: dict,
                absolute_tolerance: float, relative_tolerance: float = 0.0) -> None:
    expected_atoms = expected.get("atoms", [])
    actual_atoms = actual.get("atoms", [])
    if len(expected_atoms) != len(actual_atoms):
        raise RuntimeError(f"{label}: atom count mismatch")
    for index, (expected_atom, actual_atom) in enumerate(zip(expected_atoms, actual_atoms)):
        close(f"{label} atom {index}", float(expected_atom["sasa"]),
              float(actual_atom["sasa"]), absolute_tolerance,
              relative_tolerance)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastsasa", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--allow-missing-device", action="store_true")
    args = parser.parse_args()
    structure = "tests/data/1ubq.pdb"

    common = ("--format", "json", "--select", "ca, name ca", structure)
    cpu_sr = run(args.fastsasa, args.repo_root, "cpu", "--shrake-rupley",
                 "--resolution", "100", *common)
    try:
        vk_sr = run(args.fastsasa, args.repo_root, "vulkan", "--shrake-rupley",
                    "--resolution", "100", "--precision", "fp64",
                    "--no-cpu-fallback", *common)
    except RuntimeError as error:
        if args.allow_missing_device:
            print(f"fastsasa_vulkan_backend_validation,status,skip,detail,{error}")
            return 77
        raise
    close("SR FP64 total", cpu_sr["total_sasa"], vk_sr["total_sasa"], 1.0e-9)
    close("SR selection", cpu_sr["selections"][0]["sasa"],
          vk_sr["selections"][0]["sasa"], 1.0e-9)
    close_atoms("SR FP64", cpu_sr, vk_sr, 1.0e-9)
    vk_sr_default = run(args.fastsasa, args.repo_root, "vulkan", "--shrake-rupley",
                        "--resolution", "100", "--no-cpu-fallback", *common)
    close("SR default is FP64", vk_sr["total_sasa"],
          vk_sr_default["total_sasa"], 0.0)

    vk_sr_fp32 = run(args.fastsasa, args.repo_root, "vulkan", "--shrake-rupley",
                     "--resolution", "100", "--precision", "fp32",
                     "--no-cpu-fallback", *common)
    close("SR FP32 total", cpu_sr["total_sasa"], vk_sr_fp32["total_sasa"],
          1.0e-2, 1.0e-4)

    cpu_lr = run(args.fastsasa, args.repo_root, "cpu", "--lee-richards",
                 "--resolution", "20", *common)
    vk_lr = run(args.fastsasa, args.repo_root, "vulkan", "--lee-richards",
                "--resolution", "20", "--precision", "fp64",
                "--no-cpu-fallback", *common)
    close("LR FP64 total", cpu_lr["total_sasa"], vk_lr["total_sasa"],
          1.0e-8, 1.0e-12)
    close("LR FP64 selection", cpu_lr["selections"][0]["sasa"],
          vk_lr["selections"][0]["sasa"], 1.0e-9, 1.0e-12)
    close_atoms("LR FP64", cpu_lr, vk_lr, 1.0e-9, 1.0e-12)
    vk_lr_default = run(args.fastsasa, args.repo_root, "vulkan", "--lee-richards",
                        "--resolution", "20", "--no-cpu-fallback", *common)
    close("LR default is FP64", vk_lr["total_sasa"],
          vk_lr_default["total_sasa"], 0.0)

    vk_lr_fp32 = run(args.fastsasa, args.repo_root, "vulkan", "--lee-richards",
                     "--resolution", "20", "--precision", "fp32",
                     "--no-cpu-fallback", *common)
    close("LR FP32 total", cpu_lr["total_sasa"], vk_lr_fp32["total_sasa"],
          2.0e-2, 1.0e-4)

    no_fp64 = {"FASTSASA_TEST_VULKAN_NO_FP64": "1"}
    explicit_fp64 = invoke(
        args.fastsasa, args.repo_root, "vulkan", "--shrake-rupley",
        "--precision", "fp64", "--no-cpu-fallback", *common,
        extra_environment=no_fp64,
    )
    if explicit_fp64.returncode == 0 or "shaderFloat64" not in explicit_fp64.stderr:
        raise RuntimeError("explicit Vulkan FP64 did not report missing shaderFloat64")

    explicit_fp32 = invoke(
        args.fastsasa, args.repo_root, "vulkan", "--shrake-rupley",
        "--precision", "fp32", "--no-cpu-fallback", *common,
        extra_environment=no_fp64,
    )
    if explicit_fp32.returncode != 0:
        raise RuntimeError(explicit_fp32.stderr or explicit_fp32.stdout)

    automatic_fallback = invoke(
        args.fastsasa, args.repo_root, "auto", "--shrake-rupley",
        "--precision", "fp64", *common,
        extra_environment={
            "FASTSASA_TEST_DISABLE_CUDA": "1",
            "FASTSASA_TEST_VULKAN_NO_FP64": "1",
        },
    )
    if automatic_fallback.returncode != 0:
        raise RuntimeError(automatic_fallback.stderr or automatic_fallback.stdout)
    fallback_result = json.loads(automatic_fallback.stdout)
    close("automatic CPU fallback", cpu_sr["total_sasa"],
          fallback_result["total_sasa"], 1.0e-9)

    bad_precision = invoke(args.fastsasa, args.repo_root, "vulkan",
                           "--precision", "fp16", *common)
    if bad_precision.returncode == 0 or "--precision" not in bad_precision.stderr:
        raise RuntimeError("invalid --precision value was not rejected")

    bad_backend = invoke(args.fastsasa, args.repo_root, "auto",
                         "--backend", "metal", *common)
    if bad_backend.returncode == 0 or "--backend" not in bad_backend.stderr:
        raise RuntimeError("invalid --backend value was not rejected")

    bad_backend_env = invoke(args.fastsasa, args.repo_root, "opencl",
                             "--no-cpu-fallback", *common)
    if bad_backend_env.returncode == 0 or "FASTSASA_BACKEND" not in bad_backend_env.stderr:
        raise RuntimeError("invalid FASTSASA_BACKEND value was not rejected")

    flag_vulkan = run(args.fastsasa, args.repo_root, "auto", "--backend", "vulkan",
                      "--shrake-rupley", "--resolution", "100",
                      "--no-cpu-fallback", *common)
    close("--backend vulkan matches env selection", vk_sr["total_sasa"],
          flag_vulkan["total_sasa"], 0.0)

    print(
        "fastsasa_vulkan_backend_validation,status,pass,"
        f"sr_total,{vk_sr['total_sasa']:.9f},"
        f"lr_total,{vk_lr['total_sasa']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
