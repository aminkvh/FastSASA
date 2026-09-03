#!/usr/bin/env bash
# Builds a persistent, in-process FreeSASA trajectory driver for the
# manuscript benchmark. FreeSASA has no CLI/Python path that reuses one
# process across many frames without hitting the NULL-pointer defect
# documented in freesasa_isolated_atom_nullptr_fix.patch (isolated ions in
# the Cx46 system have zero SASA neighbors, which crashes upstream
# FreeSASA's Shrake-Rupley neighbor walk). This script builds a patched
# libfreesasa from source and links freesasa_calc_coord_driver.c against
# it -- a single process that loads all frames once and times only the
# freesasa_calc_coord() loop, matching the trajectory-engine benchmark's
# "persistent benchmark driver" allowance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${1:-$SCRIPT_DIR/build}"
FREESASA_REF="${FREESASA_GIT_REF:-2.1.3}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d freesasa ]; then
    git clone --branch "$FREESASA_REF" --depth 1 https://github.com/mittinatten/freesasa.git
fi

cd freesasa
git submodule update --init --recursive 2>/dev/null || true
patch -p1 --forward -N < "$SCRIPT_DIR/freesasa_isolated_atom_nullptr_fix.patch" || \
    echo "patch already applied (or applied manually) -- continuing"

autoreconf -i 2>/dev/null || true
CFLAGS="-O2 -g" ./configure --disable-json --disable-xml
make -j"$(nproc)"

gcc -O2 -g -o "$SCRIPT_DIR/freesasa_calc_coord_driver" \
    "$SCRIPT_DIR/freesasa_calc_coord_driver.c" \
    -I src -I . \
    src/libfreesasa.a -lpthread -lm

echo "built: $SCRIPT_DIR/freesasa_calc_coord_driver"
