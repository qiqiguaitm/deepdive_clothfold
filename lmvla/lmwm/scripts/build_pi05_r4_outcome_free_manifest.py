#!/usr/bin/env python3
"""Project the accepted R4 query manifest onto an outcome-free label input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SOURCE_PROTOCOL = "pi05_r4_policy_query_observations_combined_v1"
OUTPUT_PROTOCOL = "pi05_r4_outcome_free_query_inputs_v1"
RECORD_KEYS = (
    "behavior_policy_sha256",
    "episode_id",
    "eval_seed",
    "query_observations",
    "query_observations_sha256",
    "scene_seed",
    "source_manifest_index",
    "split",
    "task",
)
FORBIDDEN_KEYS = {"success", "reward", "return", "terminal_success"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def project(source_path: Path) -> dict:
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"unexpected query protocol: {source.get('protocol')!r}")
    records = source.get("records", [])
    if not records:
        raise ValueError("query manifest has no records")

    projected = []
    identities = set()
    for source_record in records:
        missing = set(RECORD_KEYS) - set(source_record)
        if missing:
            raise ValueError(f"query record is missing fields: {sorted(missing)}")
        if source_record.get("split") != "train":
            raise ValueError("outcome-free labels may only consume the train split")
        query_path = resolve_artifact(source_path, str(source_record["query_observations"]))
        if not query_path.is_file() or sha256(query_path) != source_record["query_observations_sha256"]:
            raise ValueError(f"query artifact hash mismatch: {query_path}")
        record = {key: source_record[key] for key in RECORD_KEYS}
        if FORBIDDEN_KEYS & set(record):
            raise AssertionError("outcome field leaked into projected record")
        identity = (str(record["task"]), int(record["scene_seed"]))
        if identity in identities:
            raise ValueError(f"duplicate task/scene identity: {identity}")
        identities.add(identity)
        projected.append(record)

    projected.sort(key=lambda row: (str(row["task"]), int(row["scene_seed"])))
    return {
        "schema_version": 1,
        "protocol": OUTPUT_PROTOCOL,
        "interpretation": (
            "This projection contains no terminal outcome, reward, return, trajectory, or video field. "
            "It is the only authorized input for outcome-free CRAVE label generation."
        ),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "behavior_policy_sha256": str(source["behavior_policy_sha256"]),
        "record_count": len(projected),
        "omitted_record_fields": sorted(set().union(*(set(row) for row in records)) - set(RECORD_KEYS)),
        "records": projected,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = project(args.query_manifest)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"records": payload["record_count"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
