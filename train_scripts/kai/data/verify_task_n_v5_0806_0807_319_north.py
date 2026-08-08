#!/usr/bin/env python3
"""Verify a North copy of the frozen Task_N 319 dataset and mark it usable."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


TRAIN = "nail_v5_0806_0807_319_joint14_train"
VAL = "nail_v5_0806_0807_319_joint14_val"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--sync-started-at", required=True)
    args = parser.parse_args()

    report = json.loads(args.freeze_report.read_text())
    records = report["derived"]["file_hashes"]
    expected_paths = {record["path"] for record in records}
    if len(expected_paths) != len(records):
        raise ValueError("duplicate paths in freeze report")

    actual_paths = {
        str(path.relative_to(args.parent))
        for dataset in (TRAIN, VAL)
        for path in (args.parent / dataset).rglob("*")
        if path.is_file() and path.name != "NORTH_SYNC_OK.json"
    }
    missing = expected_paths - actual_paths
    extra = actual_paths - expected_paths
    if missing or extra:
        raise ValueError(f"North file-set mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")

    total_bytes = 0
    for index, record in enumerate(records, 1):
        path = args.parent / record["path"]
        stat = path.stat()
        if stat.st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"North content mismatch: {record['path']}")
        total_bytes += stat.st_size
        if index % 100 == 0 or index == len(records):
            print(f"verified {index}/{len(records)}", flush=True)

    marker = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sync_started_at": args.sync_started_at,
        "freeze_report_sha256": sha256(args.freeze_report),
        "files": len(records),
        "bytes": total_bytes,
        "train_episodes": report["split"]["train_episodes"],
        "train_frames": report["split"]["train_frames"],
        "val_episodes": report["split"]["val_episodes"],
        "val_frames": report["split"]["val_frames"],
    }
    marker_path = args.parent / TRAIN / "NORTH_SYNC_OK.json"
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker_path)
    print(f"NORTH_SYNC_VERIFY_OK marker={marker_path}")


if __name__ == "__main__":
    main()
