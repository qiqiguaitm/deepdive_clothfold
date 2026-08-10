#!/usr/bin/env python3
"""Validate TG2R sidecars and recover the known missing arm metadata.

The frozen TG2R launcher exported TG2R_ARM/TG2R_TRAIN_SEED while the inherited
audit writer read TG2_ARM/TG2_TRAIN_SEED. Consequently only the redundant arm
field is null. This verifier accepts null only after all independent route,
seed, rank, world-size, count, and digest checks pass. Raw files are never
modified; normalized copies are optional integrity-overlay inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
SEEDS = (1000, 1001, 1002)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _check_arm(value: object, expected: str, label: str) -> bool:
    if value not in (None, expected):
        raise ValueError(f"TG2R {label} arm mismatch: expected null/{expected}, got {value}")
    return value is None


def audit_sidecars(
    initialization_path: Path,
    data_order_dir: Path,
    arm: str,
    seed: int,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unsupported TG2R arm: {arm}")
    if seed not in SEEDS:
        raise ValueError(f"Unsupported TG2R seed: {seed}")
    if not initialization_path.is_file():
        raise FileNotFoundError(initialization_path)
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "protocol": "lawam_matched_initialization_v1",
        "training_seed": seed,
        "optimizer_state_entries_before_training": 0,
    }
    observed = {key: initialization.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"TG2R initialization mismatch: expected {expected}, got {observed}"
        )
    initialization_arm_recovered = _check_arm(
        initialization.get("arm"), arm, "initialization"
    )
    route = initialization.get("route") or {}
    expected_future_off = arm == "future_off"
    expected_milestone = arm == "raw_milestone"
    if route.get("lawam_future_off") is not expected_future_off:
        raise ValueError(f"TG2R future-off route mismatch for {arm} seed {seed}")
    if bool(route.get("milestone_target")) is not expected_milestone:
        raise ValueError(f"TG2R milestone route mismatch for {arm} seed {seed}")
    if bool(route.get("require_full_target_coverage")) is not expected_milestone:
        raise ValueError(f"TG2R target-coverage route mismatch for {arm} seed {seed}")
    if route.get("dual_route") is not False:
        raise ValueError(f"TG2R dual route must be disabled for {arm} seed {seed}")

    order_paths = sorted(data_order_dir.glob("rank*.json"))
    if len(order_paths) != 4:
        raise ValueError(f"Expected four TG2R data-order sidecars, found {len(order_paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in order_paths]
    recovered_ranks = []
    for rank, record in enumerate(records):
        expected_rank = {
            "schema_version": 1,
            "protocol": "lawam_exact_data_order_v1",
            "training_seed": seed,
            "rank": rank,
            "world_size": 4,
        }
        observed_rank = {key: record.get(key) for key in expected_rank}
        if observed_rank != expected_rank:
            raise ValueError(
                f"TG2R data-order rank {rank} mismatch: "
                f"expected {expected_rank}, got {observed_rank}"
            )
        if _check_arm(record.get("arm"), arm, f"data-order rank {rank}"):
            recovered_ranks.append(rank)
        if int(record.get("microbatches") or 0) <= 0:
            raise ValueError(f"TG2R rank {rank} has no audited microbatches")
        if int(record.get("samples") or 0) <= 0:
            raise ValueError(f"TG2R rank {rank} has no audited samples")
        if not SHA256_PATTERN.fullmatch(str(record.get("sha256") or "")):
            raise ValueError(f"TG2R rank {rank} has an invalid data-order digest")

    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg2r_sidecar_metadata_recovery_v1",
        "complete": True,
        "arm": arm,
        "training_seed": seed,
        "initialization_arm_recovered": initialization_arm_recovered,
        "data_order_arm_recovered_ranks": recovered_ranks,
        "raw_initialization_sha256": sha256(initialization_path),
        "raw_data_order_file_sha256_by_rank": [sha256(path) for path in order_paths],
        "data_order_sha256_by_rank": [record["sha256"] for record in records],
    }


def normalize_sidecars(
    initialization_path: Path,
    data_order_dir: Path,
    destination_initialization: Path,
    destination_order_dir: Path,
    arm: str,
    seed: int,
) -> dict:
    result = audit_sidecars(initialization_path, data_order_dir, arm, seed)
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    initialization["arm"] = arm
    initialization["metadata_recovery"] = {
        "protocol": result["protocol"],
        "source_sha256": result["raw_initialization_sha256"],
        "recovered_field": "arm",
    }
    atomic_json(destination_initialization, initialization)

    destination_order_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination_order_dir.name}.incoming.",
        dir=destination_order_dir.parent,
    ) as temporary:
        incoming = Path(temporary) / destination_order_dir.name
        incoming.mkdir()
        for source in sorted(data_order_dir.glob("rank*.json")):
            record = json.loads(source.read_text(encoding="utf-8"))
            record["arm"] = arm
            record["metadata_recovery"] = {
                "protocol": result["protocol"],
                "source_sha256": sha256(source),
                "recovered_field": "arm",
            }
            atomic_json(incoming / source.name, record)
        if destination_order_dir.exists():
            shutil.rmtree(destination_order_dir)
        os.replace(incoming, destination_order_dir)
    result["normalized_initialization_sha256"] = sha256(destination_initialization)
    result["normalized_data_order_file_sha256_by_rank"] = [
        sha256(path) for path in sorted(destination_order_dir.glob("rank*.json"))
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--data-order-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--normalized-initialization", type=Path)
    parser.add_argument("--normalized-data-order-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.normalized_initialization) != bool(args.normalized_data_order_dir):
        parser.error("both normalized output paths must be supplied together")
    if args.normalized_initialization:
        result = normalize_sidecars(
            args.initialization.resolve(),
            args.data_order_dir.resolve(),
            args.normalized_initialization.resolve(),
            args.normalized_data_order_dir.resolve(),
            args.arm,
            args.seed,
        )
    else:
        result = audit_sidecars(
            args.initialization.resolve(),
            args.data_order_dir.resolve(),
            args.arm,
            args.seed,
        )
    if args.output:
        atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
