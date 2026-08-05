#!/usr/bin/env python3
"""Merge R4 outcome manifests without leaking or cherry-picking scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_artifact(record: dict, field: str, source: Path, output: Path) -> str:
    artifact = Path(record[field])
    if not artifact.is_absolute():
        artifact = source.parent / artifact
    return os.path.relpath(artifact.resolve(), output.parent.resolve())


def merge(manifest_paths: list[Path], output_path: Path) -> dict:
    if len(manifest_paths) < 2:
        raise ValueError("at least two manifests are required")
    policies: set[str] = set()
    seen_scenes: set[tuple[str, int]] = set()
    counters: dict[tuple[str, str], int] = defaultdict(int)
    records: list[dict] = []
    sources = []
    for source_index, path in enumerate(manifest_paths):
        payload = json.loads(path.read_text())
        policy = str(payload["behavior_policy_sha256"])
        policies.add(policy)
        source_records = payload.get("records", [])
        if not source_records:
            raise ValueError(f"manifest has no records: {path}")
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "record_count": len(source_records),
            }
        )
        for record in source_records:
            task = str(record["task"])
            split = str(record["split"])
            scene_seed = int(record["scene_seed"])
            scene = (task, scene_seed)
            if scene in seen_scenes:
                raise ValueError(f"duplicate task/scene across manifests: {scene}")
            seen_scenes.add(scene)
            key = (split, task)
            merged = dict(record)
            merged["episode_id"] = counters[key]
            counters[key] += 1
            merged["trajectory"] = relative_artifact(record, "trajectory", path, output_path)
            merged["video"] = relative_artifact(record, "video", path, output_path)
            merged["source_manifest_index"] = source_index
            records.append(merged)
    if len(policies) != 1:
        raise ValueError(f"behavior policy mismatch: {sorted(policies)}")
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_action_bearing_outcomes_combined_v1",
        "behavior_policy_sha256": policies.pop(),
        "source_manifests": sources,
        "records": records,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge([path.resolve() for path in args.manifest], args.output.resolve())
    atomic_json(args.output, result)
    print(json.dumps({"records": len(result["records"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
