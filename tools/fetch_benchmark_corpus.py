#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import urllib.error
import urllib.request
from pathlib import Path


RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def _selected(
    row: dict[str, str],
    categories: set[str] | None,
    ids: set[str] | None,
    include_nondefault: bool,
    exclude_huge: bool,
) -> bool:
    pdb_id = row["pdb_id"].upper()
    if ids is not None and pdb_id not in ids:
        return False
    if categories is not None and row["category"].lower() not in categories:
        return False
    if not include_nondefault and row.get("default", "").lower() not in {"true", "yes", "1"}:
        return False
    if exclude_huge and row.get("tier", "").lower() == "huge":
        return False
    return True


def _read_manifest(
    path: Path,
    categories: set[str] | None,
    ids: set[str] | None,
    include_nondefault: bool,
    exclude_huge: bool,
) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if _selected(row, categories, ids, include_nondefault, exclude_huge)]


def _download(pdb_id: str, output_path: Path, force: bool) -> str:
    if output_path.exists() and not force:
        return "exists"
    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {pdb_id} from {url}: {exc}") from exc
    if not data.startswith(b"data_"):
        raise RuntimeError(f"downloaded {pdb_id} is not a valid mmCIF data block")
    temporary.write_bytes(data)
    temporary.replace(output_path)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch public mmCIF files for the FastSASA benchmark corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("docs/benchmark_corpus.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_corpus/structures"))
    parser.add_argument("--include-nondefault", action="store_true", help="also fetch opt-in large/heavy structures")
    parser.add_argument("--exclude-huge", action="store_true", help="skip tier=huge entries even with --include-nondefault")
    parser.add_argument("--categories", help="comma-separated category filter, for example dna,rna,glycoprotein")
    parser.add_argument("--ids", help="comma-separated PDB IDs to fetch")
    parser.add_argument("--force", action="store_true", help="redownload files that already exist")
    args = parser.parse_args()

    categories = {value.strip().lower() for value in args.categories.split(",") if value.strip()} if args.categories else None
    ids = {value.strip().upper() for value in args.ids.split(",") if value.strip()} if args.ids else None
    rows = _read_manifest(args.manifest, categories, ids, args.include_nondefault, args.exclude_huge)
    if not rows:
        print("No benchmark corpus entries matched the filters.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for row in rows:
        pdb_id = row["pdb_id"].upper()
        output_path = args.output_dir / f"{pdb_id.lower()}.cif"
        try:
            status = _download(pdb_id, output_path, args.force)
            print(f"{status}: {pdb_id} -> {output_path}", file=sys.stderr)
        except RuntimeError as exc:
            failures += 1
            print(str(exc), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
