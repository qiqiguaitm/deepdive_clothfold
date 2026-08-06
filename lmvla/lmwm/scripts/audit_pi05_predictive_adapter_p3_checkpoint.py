#!/usr/bin/env python3
"""Audit an independently trained P3 A0 checkpoint before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_pi05_predictive_adapter_p2_checkpoint import (  # noqa: E402
    EXPECTED_HANDLERS,
    atomic_write,
    nonempty,
    payload_files,
    read_json,
    sha256,
    tree_keys,
)


REQUIRED = (
    "_CHECKPOINT_METADATA",
    "params/_METADATA",
    "params/_sharding",
    "params/manifest.ocdbt",
    "train_state/_METADATA",
    "train_state/_sharding",
    "train_state/manifest.ocdbt",
    "assets/robotwin2.0_absolute_meanstd/norm_stats.json",
)


def audit(
    *, checkpoint: Path, reference: Path, source_preflight: Path, dataset: Path, seed: int
) -> dict:
    if seed not in (1001, 1002):
        raise ValueError("P3 only permits seeds 1001 and 1002")
    files = {relative: nonempty(checkpoint / relative) for relative in REQUIRED}
    root = read_json(checkpoint / "_CHECKPOINT_METADATA") if files[REQUIRED[0]] else {}
    params = tree_keys(read_json(checkpoint / "params/_METADATA")) if files[REQUIRED[1]] else set()
    state = tree_keys(read_json(checkpoint / "train_state/_METADATA")) if files[REQUIRED[4]] else set()
    ref_params = tree_keys(read_json(reference / "params/_METADATA"))
    ref_state = tree_keys(read_json(reference / "train_state/_METADATA"))
    source = read_json(source_preflight) if nonempty(source_preflight) else {}
    source_rows = [
        *source.get("source_checks", {}).values(),
        *source.get("artifact_checks", {}).values(),
    ]
    payload = {
        item: sum(path.stat().st_size for path in payload_files(checkpoint / item))
        for item in ("params", "train_state")
    }
    norm = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    ref_norm = reference / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    checks = {
        "final_step": checkpoint.name == "49999",
        "required_files": all(files.values()),
        "atomic_commit": root.get("item_handlers") == EXPECTED_HANDLERS
        and int(root.get("commit_timestamp_nsecs", 0))
        >= int(root.get("init_timestamp_nsecs", 0))
        > 0,
        "parameter_tree_matches_seed1000_a0": bool(params) and params == ref_params,
        "optimizer_tree_matches_seed1000_a0": bool(state) and state == ref_state,
        "optimizer_moments_present": any("'mu'" in key for key in state)
        and any("'nu'" in key for key in state)
        and "('step',)" in state,
        "payload_nonempty": all(value > 0 for value in payload.values()),
        "normalization_exact": nonempty(norm) and sha256(norm) == sha256(ref_norm),
        "source_freeze": source.get("passed") is True
        and bool(source_rows)
        and all(row.get("match") is True for row in source_rows),
        "dataset_identity": nonempty(dataset)
        and sha256(dataset)
        == "9d05f20e76cee73acc7240a5f2ed97881bb4ea621daaf64f70a5c3767d47ef8d",
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_p3_checkpoint_audit_v1",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "reference_checkpoint": str(reference),
        "source_preflight": str(source_preflight),
        "dataset": str(dataset),
        "files": files,
        "payload_bytes": payload,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        checkpoint=args.checkpoint,
        reference=args.reference,
        source_preflight=args.source_preflight,
        dataset=args.dataset,
        seed=args.seed,
    )
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        args.marker.unlink(missing_ok=True)
        raise SystemExit(1)
    atomic_write(args.marker, f"checkpoint={args.checkpoint}\nreport={args.output}\n")


if __name__ == "__main__":
    main()
