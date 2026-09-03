#!/usr/bin/env bash
set -eu

: "${NVCC:=nvcc}"
: "${FASTSASA_MIN_CUDA_MAJOR:=12}"

cuda_version="$($NVCC --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' | head -n 1)"
host_cxx="${CUDAHOSTCXX:-${CXX:-c++}}"
host_version="$($host_cxx -dumpfullversion -dumpversion 2>/dev/null || true)"
host_name="$($host_cxx --version 2>/dev/null | head -n 1 || true)"

if [ -z "$cuda_version" ]; then
    echo "could not parse CUDA version from $NVCC --version" >&2
    exit 1
fi

cuda_major="${cuda_version%%.*}"
cuda_minor="${cuda_version#*.}"
host_major="${host_version%%.*}"
host_is_gcc=1
case "$host_name" in
    *clang*|*Clang*) host_is_gcc=0 ;;
esac

echo "cuda,$cuda_version"
echo "host_compiler,$host_cxx"
echo "host_version,$host_version"
echo "host_name,$host_name"

if [ "$cuda_major" -lt "$FASTSASA_MIN_CUDA_MAJOR" ]; then
    echo "CUDA $cuda_version is older than the FastSASA release minimum ${FASTSASA_MIN_CUDA_MAJOR}.x" >&2
    exit 1
fi

if [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -eq 0 ] &&
   [ "$host_is_gcc" -eq 1 ] && [ -n "$host_major" ] && [ "$host_major" -ge 13 ]; then
    echo "CUDA 12.0 rejects GCC/G++ $host_version as an nvcc host compiler." >&2
    echo "Set CUDAHOSTCXX to clang-14, clang-15, gcc-11, or gcc-12 before configuring FastSASA." >&2
    exit 1
fi
