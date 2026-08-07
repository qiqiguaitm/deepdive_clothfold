#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
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


def resolve_sidecars(repo: Path, run_id: str) -> tuple[Path, Path]:
    staged = (
        repo
        / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
        / run_id
    )
    staged_initialization = staged / "initialization.json"
    staged_order = staged / "data_order"
    if staged.exists():
        if not staged_initialization.is_file() or not staged_order.is_dir():
            raise ValueError(f"Incomplete staged TG2 sidecars for {run_id}: {staged}")
        return staged_initialization, staged_order
    canonical = repo / "logs/temporal_grounding/tg2"
    return (
        canonical / "initialization" / f"{run_id}.json",
        canonical / "data_order" / run_id,
    )


def audit_sidecars(
    initialization_path: Path,
    data_order_dir: Path,
    arm: str,
    seed: int,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unsupported TG2 arm: {arm}")
    if seed not in SEEDS:
        raise ValueError(f"Unsupported TG2 seed: {seed}")
    if not initialization_path.is_file():
        raise FileNotFoundError(initialization_path)
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    expected_initialization = {
        "schema_version": 1,
        "protocol": "lawam_matched_initialization_v1",
        "arm": arm,
        "training_seed": seed,
        "optimizer_state_entries_before_training": 0,
    }
    observed_initialization = {
        key: initialization.get(key) for key in expected_initialization
    }
    if observed_initialization != expected_initialization:
        raise ValueError(
            "TG2 initialization sidecar mismatch: "
            f"expected {expected_initialization}, got {observed_initialization}"
        )
    route = initialization.get("route") or {}
    expected_future_off = arm == "future_off"
    expected_milestone = arm == "raw_milestone"
    if route.get("lawam_future_off") is not expected_future_off:
        raise ValueError(f"TG2 future-off route mismatch for {arm} seed {seed}")
    if bool(route.get("milestone_target")) is not expected_milestone:
        raise ValueError(f"TG2 milestone route mismatch for {arm} seed {seed}")
    if bool(route.get("require_full_target_coverage")) is not expected_milestone:
        raise ValueError(f"TG2 target-coverage route mismatch for {arm} seed {seed}")
    if route.get("dual_route") is not False:
        raise ValueError(f"TG2 dual route must be disabled for {arm} seed {seed}")

    order_paths = sorted(data_order_dir.glob("rank*.json"))
    if len(order_paths) != 4:
        raise ValueError(
            f"Expected four TG2 data-order sidecars, found {len(order_paths)}"
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in order_paths]
    for rank, record in enumerate(records):
        expected = {
            "schema_version": 1,
            "protocol": "lawam_exact_data_order_v1",
            "arm": arm,
            "training_seed": seed,
            "rank": rank,
            "world_size": 4,
        }
        observed = {key: record.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"TG2 data-order rank {rank} mismatch: expected {expected}, got {observed}"
            )
        if int(record.get("microbatches") or 0) <= 0:
            raise ValueError(f"TG2 rank {rank} has no audited microbatches")
        if int(record.get("samples") or 0) <= 0:
            raise ValueError(f"TG2 rank {rank} has no audited samples")
        if not SHA256_PATTERN.fullmatch(str(record.get("sha256") or "")):
            raise ValueError(f"TG2 rank {rank} has an invalid data-order digest")

    return {
        "schema_version": 1,
        "complete": True,
        "arm": arm,
        "training_seed": seed,
        "initialization_sha256": sha256(initialization_path),
        "data_order_sha256_by_rank": [sha256(path) for path in order_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--data-order-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    result = audit_sidecars(
        args.initialization.resolve(),
        args.data_order_dir.resolve(),
        args.arm,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
