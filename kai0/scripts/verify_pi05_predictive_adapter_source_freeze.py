#!/usr/bin/env python3
"""Verify the immutable source and artifact identity for P1/P2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_PATHS = {
    "config.py": "kai0/src/openpi/training/config.py",
    "pi0.py": "kai0/src/openpi/models/pi0.py",
    "weight_loaders.py": "kai0/src/openpi/training/weight_loaders.py",
    "train_pi05_robotwin_confirmatory.py": "kai0/scripts/train_pi05_robotwin_confirmatory.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, audit: dict) -> dict:
    expected_sources = audit["source_identity"]["current"]
    source_checks = {}
    for name, relative_path in SOURCE_PATHS.items():
        path = repo / relative_path
        actual = sha256(path)
        expected = expected_sources.get(name)
        source_checks[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }

    base = repo / "kai0/checkpoints/pi05_base/params/_METADATA"
    norm = repo / (
        "kai0/assets/pi05_robotwin_a0_public_exact_bj/"
        "robotwin2.0_absolute_meanstd/norm_stats.json"
    )
    artifact_checks = {
        "base_metadata": {
            "path": str(base),
            "expected_sha256": audit["base_identity"]["current_metadata_sha256"],
            "actual_sha256": sha256(base),
        },
        "norm_stats": {
            "path": str(norm),
            "expected_sha256": audit["normalization_identity"]["current_sha256"],
            "actual_sha256": sha256(norm),
        },
    }
    for row in artifact_checks.values():
        row["match"] = row["actual_sha256"] == row["expected_sha256"]

    protocol_checks = {
        "decision_requires_current_source_a0": audit.get("decision")
        == "retrain_current_source_a0",
        "matched_recipe_passed": bool(audit.get("matched_recipe", {}).get("passed")),
        "all_sources_match": all(row["match"] for row in source_checks.values()),
        "all_artifacts_match": all(row["match"] for row in artifact_checks.values()),
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_source_freeze_v1",
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "protocol_checks": protocol_checks,
        "passed": all(protocol_checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(args.repo.resolve(), json.loads(args.audit.read_text()))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.output)
    print(payload, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
