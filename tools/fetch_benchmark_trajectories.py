#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import json
import shutil
import struct
import sys
import urllib.request
import zlib
from pathlib import Path


ZENODO_API = "https://zenodo.org/api/records"
DEFAULT_TIMEOUT_SECONDS = 120


def _record(record_id: str) -> dict:
    with urllib.request.urlopen(f"{ZENODO_API}/{record_id}", timeout=DEFAULT_TIMEOUT_SECONDS) as handle:
        return json.load(handle)


def _file(record: dict, key: str) -> dict:
    for item in record.get("files", []):
        if item.get("key") == key:
            return item
    raise RuntimeError(f"record {record.get('id')} does not contain {key}")


def _download_file(url: str, destination: Path, expected_size: int | None = None) -> None:
    if destination.exists() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {destination.name} has unexpected size")
    temporary.replace(destination)


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as handle:
        return handle.read()


def _zip_entries(url: str, size: int) -> list[dict]:
    tail_start = max(0, size - 131072)
    tail = _fetch_range(url, tail_start, size - 1)
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise RuntimeError("could not find ZIP end-of-central-directory record")
    eocd = tail[eocd_offset:eocd_offset + 22]
    _, _, _, _, total_entries, central_size, central_offset, _ = struct.unpack("<IHHHHIIH", eocd)
    central = _fetch_range(url, central_offset, central_offset + central_size - 1)
    entries = []
    offset = 0
    for _ in range(total_entries):
        if central[offset:offset + 4] != b"PK\x01\x02":
            raise RuntimeError("invalid ZIP central-directory entry")
        fields = struct.unpack("<IHHHHHHIIIHHHHHII", central[offset:offset + 46])
        (
            _,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            comment_size,
            _,
            _,
            _,
            local_offset,
        ) = fields
        name = central[offset + 46:offset + 46 + name_size].decode("utf-8")
        entries.append({
            "name": name,
            "flags": flags,
            "method": method,
            "crc": crc,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        })
        offset += 46 + name_size + extra_size + comment_size
    return entries


def _extract_zip_member(url: str, entry: dict, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == entry["uncompressed_size"]:
        return
    local = _fetch_range(url, entry["local_offset"], entry["local_offset"] + 30 - 1)
    if local[:4] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local ZIP header for {entry['name']}")
    _, _, _, method, _, _, _, _, _, name_size, extra_size = struct.unpack("<IHHHHHIIIHH", local)
    if method != entry["method"]:
        raise RuntimeError(f"ZIP method mismatch for {entry['name']}")
    data_start = entry["local_offset"] + 30 + name_size + extra_size
    compressed = _fetch_range(url, data_start, data_start + entry["compressed_size"] - 1)
    if entry["method"] == 0:
        payload = compressed
    elif entry["method"] == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {entry['method']} for {entry['name']}")
    if len(payload) != entry["uncompressed_size"]:
        raise RuntimeError(f"unexpected extracted size for {entry['name']}")
    if binascii.crc32(payload) & 0xffffffff != entry["crc"]:
        raise RuntimeError(f"CRC mismatch for {entry['name']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def _fetch_metrex(output_dir: Path) -> dict:
    record = _record("14512968")
    target = output_dir / "metrex_6pmb"
    topology = _file(record, "example_topology_with_6PMB.pdb")
    trajectory = _file(record, "example_simulation_6PMB_500Frames.xtc")
    topology_path = target / topology["key"]
    trajectory_path = target / trajectory["key"]
    extra = "--filter 'system, resn DPE+LP0+DDM+LH0+DPP+PMB'"
    _download_file(topology["links"]["self"], topology_path, topology.get("size"))
    _download_file(trajectory["links"]["self"], trajectory_path, trajectory.get("size"))
    return {
        "name": "zenodo_14512968_metrex_6pmb_500frames",
        "record": "14512968",
        "doi": "10.5281/zenodo.14512968",
        "topology": str(topology_path),
        "trajectory": str(trajectory_path),
        "selection": "system",
        "extra": extra,
        "entry": f"zenodo_14512968_metrex_6pmb_500frames|{topology_path}|{trajectory_path}|{extra}",
    }


def _fetch_gabaa(output_dir: Path) -> dict:
    # Zenodo record 20705736 (Akbari Ahangar & Li 2026) ships one zip with
    # eight subunit-stoichiometry variants; extract exactly the two members
    # used by the published trajectory benchmark (the βαβαγ variant's
    # topology and first replica) instead of downloading the 1.4 GB archive.
    record = _record("20705736")
    archive = _file(record, "pore_facing_simulation_data.zip")
    archive_url = archive["links"]["self"]
    entries = {entry["name"]: entry for entry in _zip_entries(archive_url, archive["size"])}
    member_topology = "pore_facing_simulation_data/βαβαγ/step5_assembly.hmr.psf"
    member_trajectory = "pore_facing_simulation_data/βαβαγ/sim_1.dcd"
    for member in (member_topology, member_trajectory):
        if member not in entries:
            raise RuntimeError(f"GABAA archive does not contain expected member {member}")
    target = output_dir / "gabaa_pore_facing"
    topology_path = target / "step5_assembly.hmr.psf"
    trajectory_path = target / "sim_1.dcd"
    extra = "--filter protein"
    _extract_zip_member(archive_url, entries[member_topology], topology_path)
    _extract_zip_member(archive_url, entries[member_trajectory], trajectory_path)
    return {
        "name": "zenodo_20705736_gabaa_babag_replica1",
        "record": "20705736",
        "doi": "10.5281/zenodo.20705736",
        "archive_member_topology": member_topology,
        "archive_member_trajectory": member_trajectory,
        "topology": str(topology_path),
        "trajectory": str(trajectory_path),
        "selection": "protein",
        "extra": extra,
        "entry": f"zenodo_20705736_gabaa_babag_replica1|{topology_path}|{trajectory_path}|{extra}",
    }


def _fetch_cx46(output_dir: Path) -> dict:
    record = _record("4625961")
    target = output_dir / "cx46_hemichannel"
    topology = _file(record, "Cx46_Ace_ProtIon.psf")
    trajectory = _file(record, "Cx46_Ace_ProtIon_01.dcd")
    topology_path = target / topology["key"]
    trajectory_path = target / trajectory["key"]
    extra = "--filter protein"
    _download_file(topology["links"]["self"], topology_path, topology.get("size"))
    _download_file(trajectory["links"]["self"], trajectory_path, trajectory.get("size"))
    return {
        "name": "zenodo_4625961_cx46_protion_replica1",
        "record": "4625961",
        "doi": "10.5281/zenodo.4625961",
        "topology": str(topology_path),
        "trajectory": str(trajectory_path),
        "selection": "protein",
        "extra": extra,
        "entry": f"zenodo_4625961_cx46_protion_replica1|{topology_path}|{trajectory_path}|{extra}",
    }


def _fetch_cbh1(output_dir: Path) -> dict:
    record = _record("2537734")
    target = output_dir / "cbh1_cellulase"
    topology = _file(record, "cbh1test.pdb")
    trajectory = _file(record, "cbh1test.dcd")
    topology_path = target / topology["key"]
    trajectory_path = target / trajectory["key"]
    extra = "--filter protein"
    _download_file(topology["links"]["self"], topology_path, topology.get("size"))
    _download_file(trajectory["links"]["self"], trajectory_path, trajectory.get("size"))
    return {
        "name": "zenodo_2537734_cbh1_cellulase",
        "record": "2537734",
        "doi": "10.5281/zenodo.2537734",
        "topology": str(topology_path),
        "trajectory": str(trajectory_path),
        "selection": "protein",
        "extra": extra,
        "entry": f"zenodo_2537734_cbh1_cellulase|{topology_path}|{trajectory_path}|{extra}",
    }


FETCHERS = {
    "metrex": _fetch_metrex,
    "gabaa": _fetch_gabaa,
    "cx46": _fetch_cx46,
    "cbh1": _fetch_cbh1,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch one public trajectory benchmark from each configured Zenodo record.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_corpus/trajectories"))
    parser.add_argument("--manifest", type=Path, default=Path("benchmark_corpus/trajectories/trajectory_benchmark_manifest.json"))
    parser.add_argument("--records", default="metrex,gabaa",
                        help="comma-separated subset: metrex,gabaa,cx46,cbh1")
    args = parser.parse_args()

    selected = {item.strip().lower() for item in args.records.replace(",", " ").split() if item.strip()}
    unknown = selected - set(FETCHERS)
    if unknown:
        parser.error("unknown --records value(s): " + ", ".join(sorted(unknown)))
    if not selected:
        parser.error("--records must select at least one of: " + ", ".join(sorted(FETCHERS)))
    results = []
    for name in ("metrex", "gabaa", "cx46", "cbh1"):
        if name in selected:
            results.append(FETCHERS[name](args.output_dir))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"trajectories": results}, indent=2), encoding="utf-8")
    for result in results:
        print(result["entry"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
