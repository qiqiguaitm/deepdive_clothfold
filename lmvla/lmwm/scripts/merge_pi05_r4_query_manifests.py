#!/usr/bin/env python3
"""Merge predeclared R4 query manifests while retaining every scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARTIFACT_FIELDS = ("trajectory", "video", "query_observations")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebase(value: str, source: Path, output: Path) -> str:
    artifact = Path(value)
    if not artifact.is_absolute():
        artifact = source.parent / artifact
    return os.path.relpath(artifact.resolve(), output.parent.resolve())


def merge(manifests: list[Path], output: Path) -> dict:
    if len(manifests) < 2:
        raise ValueError("at least two query manifests are required")
    policies = set()
    scenes = set()
    records = []
    sources = []
    for source_index, source in enumerate(manifests):
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("protocol") != "pi05_r4_policy_query_observations_v1":
            raise ValueError(f"unexpected query manifest protocol: {source}")
        policies.add(str(payload["behavior_policy_sha256"]))
        source_records = payload.get("records", [])
        if not source_records:
            raise ValueError(f"query manifest has no records: {source}")
        sources.append(
            {"path": str(source.resolve()), "sha256": sha256(source), "record_count": len(source_records)}
        )
        for record in source_records:
            scene = (str(record["task"]), int(record["scene_seed"]))
            if scene in scenes:
                raise ValueError(f"duplicate task/scene across query manifests: {scene}")
            scenes.add(scene)
            merged = dict(record)
            for field in ARTIFACT_FIELDS:
                merged[field] = rebase(str(record[field]), source, output)
            merged["source_manifest_index"] = source_index
            records.append(merged)
    if len(policies) != 1:
        raise ValueError(f"behavior policy mismatch: {sorted(policies)}")
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_policy_query_observations_combined_v1",
        "behavior_policy_sha256": policies.pop(),
        "source_manifests": sources,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge([path.resolve() for path in args.manifest], args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"records": len(result["records"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
