from pathlib import Path
from shutil import copy2, rmtree, which
import os
import subprocess
import sys

from setuptools import setup
from setuptools import Distribution
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildPyWithNativeLibrary(build_py):
    def run(self):
        # setuptools may reuse build/lib from an older checkout or package
        # name. Start this package's Python output clean so stale modules and
        # shared libraries cannot leak into the wheel.
        rmtree(self.build_lib, ignore_errors=True)
        cmake_build = ROOT / "build" / "wheel_native"
        if sys.platform == "darwin":
            native_name = "libfastsasa_native.dylib"
        elif sys.platform == "win32":
            native_name = "fastsasa_native.dll"
        else:
            native_name = "libfastsasa_native.so"
        cuda_enabled = os.environ.get("FASTSASA_ENABLE_CUDA")
        vulkan_enabled = os.environ.get("FASTSASA_ENABLE_VULKAN", "ON")
        if cuda_enabled is None:
            cuda_enabled = "ON" if which("nvcc") else "OFF"
        subprocess.check_call([
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(cmake_build),
            "-DFASTSASA_BUILD_STATIC=OFF",
            "-DFASTSASA_BUILD_CLI=OFF",
            "-DFASTSASA_BUILD_NATIVE_TESTS=OFF",
            f"-DFASTSASA_ENABLE_CUDA={cuda_enabled}",
            f"-DFASTSASA_ENABLE_VULKAN={vulkan_enabled}",
            "-DCMAKE_BUILD_TYPE=Release",
        ])
        subprocess.check_call([
            "cmake",
            "--build",
            str(cmake_build),
            "--config",
            "Release",
            "--target",
            "fastsasa_native",
            "-j4",
        ])
        super().run()
        # Multi-config generators (Visual Studio) place outputs in a
        # per-configuration subdirectory.
        candidates = [cmake_build / native_name, cmake_build / "Release" / native_name]
        native_lib = next((path for path in candidates if path.exists()), None)
        if native_lib is not None:
            copy2(native_lib, Path(self.build_lib) / native_name)
        else:
            raise RuntimeError(f"native library was not built: {candidates[0]}")


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPyWithNativeLibrary},
    data_files=[
        ("share/fastsasa", ["share/protor.config", "share/protor_glycans.config"]),
        ("share/fastsasa/examples", [str(path.relative_to(ROOT)) for path in sorted((ROOT / "examples").glob("*.py"))]),
        ("share/fastsasa/licenses",
         ["LICENSE", "NOTICE"] +
         [str(path.relative_to(ROOT)) for path in sorted((ROOT / "licenses").iterdir())
          if path.suffix in {".md", ".txt"}]),
    ],
)
