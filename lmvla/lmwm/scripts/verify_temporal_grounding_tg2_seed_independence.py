#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .verify_temporal_grounding_tg2_sidecars import resolve_sidecars
except ImportError:
    from verify_temporal_grounding_tg2_sidecars import resolve_sidecars


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
SEEDS = (1000, 1001, 1002)


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def audit(repo: Path) -> dict:
    order_by_seed: dict[int, tuple[str, ...]] = {}
    records: dict[str, dict] = {}
    for seed in SEEDS:
        arm_orders: dict[str, tuple[str, ...]] = {}
        for arm in ARMS:
            run_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
            _, order_dir = resolve_sidecars(repo, run_id)
            paths = sorted(order_dir.glob("rank*.json"))
            if len(paths) != 4:
                raise ValueError(f"Expected four rank audits for {run_id}, found {len(paths)}")
            rows = [json.loads(path.read_text()) for path in paths]
            if [row.get("rank") for row in rows] != list(range(4)):
                raise ValueError(f"Noncanonical rank records for {run_id}")
            if any(row.get("world_size") != 4 for row in rows):
                raise ValueError(f"Wrong world size for {run_id}")
            if any(row.get("training_seed") != seed for row in rows):
                raise ValueError(f"Training-seed audit mismatch for {run_id}")
            if any(row.get("arm") != arm for row in rows):
                raise ValueError(f"Arm audit mismatch for {run_id}")
            order = tuple(str(row["sha256"]) for row in rows)
            arm_orders[arm] = order
            records[run_id] = {
                "data_order_sha256_by_rank": list(order),
                "microbatches_by_rank": [row.get("microbatches") for row in rows],
                "samples_by_rank": [row.get("samples") for row in rows],
            }
        unique_within_seed = set(arm_orders.values())
        if len(unique_within_seed) != 1:
            raise ValueError(f"Dataset order differs across TG2 arms for seed {seed}")
        order_by_seed[seed] = unique_within_seed.pop()

    if len(set(order_by_seed.values())) != len(SEEDS):
        raise ValueError("TG2 training seeds do not induce distinct rank data orders")
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg2_seed_independence_v1",
        "complete": True,
        "checks": {
            "dataset_order_equal_within_seed": True,
            "dataset_order_distinct_across_seeds": True,
        },
        "data_order_sha256_by_seed": {
            str(seed): list(order) for seed, order in order_by_seed.items()
        },
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    atomic_write(args.output, result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
