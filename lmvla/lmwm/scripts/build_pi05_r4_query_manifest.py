#!/usr/bin/env python3
"""Attach policy-query observations to an R4 outcome manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(outcome_manifest: Path, output: Path) -> dict:
    source = json.loads(outcome_manifest.read_text(encoding="utf-8"))
    records = []
    for source_record in source.get("records", []):
        record = dict(source_record)
        trajectory = Path(str(record["trajectory"]))
        if not trajectory.is_absolute():
            trajectory = outcome_manifest.parent / trajectory
        episode_stem = trajectory.stem
        if not episode_stem.startswith("episode"):
            raise ValueError(f"unexpected trajectory name: {trajectory}")
        query = trajectory.with_name(f"query_{episode_stem}.npz")
        if not query.is_file():
            raise FileNotFoundError(query)
        record["trajectory"] = os.path.relpath(trajectory.resolve(), output.parent.resolve())
        video = Path(str(record["video"]))
        if not video.is_absolute():
            video = outcome_manifest.parent / video
        record["video"] = os.path.relpath(video.resolve(), output.parent.resolve())
        record["query_observations"] = os.path.relpath(
            query.resolve(), output.parent.resolve()
        )
        record["query_observations_sha256"] = sha256(query)
        records.append(record)
    if not records:
        raise ValueError("outcome manifest has no records")
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_policy_query_observations_v1",
        "behavior_policy_sha256": source["behavior_policy_sha256"],
        "source_outcome_manifest": str(outcome_manifest.resolve()),
        "source_outcome_manifest_sha256": sha256(outcome_manifest),
        "records": records,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.outcome_manifest.resolve(), args.output.resolve())
    atomic_json(args.output, result)
    print(json.dumps({"records": len(result["records"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
