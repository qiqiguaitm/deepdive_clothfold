#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARMS = (
    "clean_base",
    "future_off",
    "auxiliary_only",
    "conditioning_only",
    "parameter_matched_null",
    "full",
)
PRETRAINED_ARMS = tuple(arm for arm in ARMS if arm != "clean_base")
SEEDS = (1100, 1101, 1102)
EXPECTED_ROUTE = {
    "clean_base": (False, False, False),
    "future_off": (False, False, False),
    "auxiliary_only": (False, False, True),
    "conditioning_only": (False, True, False),
    "parameter_matched_null": (True, False, False),
    "full": (False, False, False),
}
ARM_CONFIG_PATHS = (
    ("framework", "action_model", "future_prediction"),
    ("framework", "action_model", "enable_loss_distill"),
    ("trainer", "pretrained_checkpoint"),
    ("trainer", "load_pretrained_policy_flow"),
    ("trainer", "ddp_find_unused_parameters"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o664)
    os.replace(temporary, path)


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(paths)}: {paths}")
    return paths[0]


def resolve_sidecars(repo: Path, run_id: str) -> tuple[Path, Path]:
    staged = (
        repo
        / "logs/resource_scheduler_local/temporal_grounding_tg4_sidecars"
        / run_id
    )
    if staged.exists():
        initialization = staged / "initialization.json"
        order = staged / "data_order"
        if not initialization.is_file() or not order.is_dir():
            raise ValueError(f"Incomplete staged TG4 sidecars: {staged}")
        return initialization, order
    canonical = repo / "logs/temporal_grounding/tg4"
    return canonical / "initialization" / f"{run_id}.json", canonical / "data_order" / run_id


def nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        value = value[key]
    return value


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for key in ("seed", "run_id", "output_dir"):
        result.pop(key, None)
    for path in ARM_CONFIG_PATHS:
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    return result


def expected_arm_config(arm: str) -> dict[tuple[str, ...], Any]:
    future_enabled = arm not in {"clean_base", "future_off"}
    # parameter_matched_null preserves the full serialized config and parameter
    # surface; LAWAM_FUTURE_OFF disables its effective future route at runtime.
    auxiliary_enabled = arm not in {"clean_base", "future_off"}
    return {
        ("framework", "action_model", "future_prediction"): future_enabled,
        ("framework", "action_model", "enable_loss_distill"): auxiliary_enabled,
        ("trainer", "pretrained_checkpoint"): (
            None
            if arm == "clean_base"
            else "results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"
        ),
        ("trainer", "load_pretrained_policy_flow"): arm != "clean_base",
        ("trainer", "ddp_find_unused_parameters"): arm in {
            "clean_base",
            "future_off",
            "conditioning_only",
        },
    }


def verify(
    repo: Path,
    *,
    min_checkpoint_bytes: int = 1_000_000_000,
    min_optimizer_bytes: int = 1_000_000_000,
) -> dict[str, Any]:
    checkpoint_root = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    runs: dict[str, Any] = {}
    parameter_trees: set[str] = set()
    trainable_trees: set[str] = set()
    optimizer_trees: set[str] = set()
    statistics_hashes: set[str] = set()
    normalized_configs: set[str] = set()
    orders_by_seed: dict[int, tuple[str, ...]] = {}

    for seed in SEEDS:
        pretrained_payloads: set[str] = set()
        clean_payload: str | None = None
        arm_orders: dict[str, tuple[str, ...]] = {}
        for arm in ARMS:
            run_id = f"temporal_grounding_tg4_{arm}_seed{seed}"
            run = one(sorted(checkpoint_root.glob(f"*+{run_id}")), f"run {run_id}")
            config_path = run / "config.json"
            stats_path = run / "dataset_statistics.json"
            final_path = run / "final_model/pytorch_model.pt"
            state = run / "checkpoints/steps_20000_state"
            optimizer_path = state / "optimizer.bin"
            trainer_state_path = state / "trainer_state.json"
            for path in (config_path, stats_path, final_path, optimizer_path, trainer_state_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if final_path.stat().st_size < min_checkpoint_bytes:
                raise ValueError(f"Truncated final checkpoint for {run_id}")
            if optimizer_path.stat().st_size < min_optimizer_bytes:
                raise ValueError(f"Truncated optimizer state for {run_id}")
            if json.loads(trainer_state_path.read_text())["steps"] != 20000:
                raise ValueError(f"Wrong final optimizer step for {run_id}")

            config = json.loads(config_path.read_text())
            if config.get("seed") != seed:
                raise ValueError(f"Training seed mismatch for {run_id}")
            for path, expected in expected_arm_config(arm).items():
                observed = nested(config, path)
                if observed != expected:
                    raise ValueError(
                        f"TG4 arm config mismatch for {run_id} at {'.'.join(path)}: "
                        f"expected {expected!r}, got {observed!r}"
                    )
            data = config["datasets"]["vla_data"]
            trainer = config["trainer"]
            expected_common = {
                "data_mix": "robotwin2_lmwm_all6_v2",
                "per_device_batch_size": 16,
                "num_workers": 8,
                "in_order": True,
                "gradient_accumulation_steps": 2,
                "max_train_steps": 20000,
                "save_interval": 20000,
                "optimizer_fused": False,
            }
            observed_common = {
                "data_mix": data["data_mix"],
                "per_device_batch_size": data["per_device_batch_size"],
                "num_workers": data["num_workers"],
                "in_order": data["in_order"],
                "gradient_accumulation_steps": trainer["gradient_accumulation_steps"],
                "max_train_steps": trainer["max_train_steps"],
                "save_interval": trainer["save_interval"],
                "optimizer_fused": trainer["optimizer"]["fused"],
            }
            if observed_common != expected_common:
                raise ValueError(f"TG4 common config drift for {run_id}: {observed_common}")
            normalized_configs.add(
                json.dumps(normalized_config(config), sort_keys=True, separators=(",", ":"))
            )

            initialization_path, order_dir = resolve_sidecars(repo, run_id)
            initialization = json.loads(initialization_path.read_text())
            expected_identity = {
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": arm,
                "training_seed": seed,
                "optimizer_state_entries_before_training": 0,
            }
            observed_identity = {key: initialization.get(key) for key in expected_identity}
            if observed_identity != expected_identity:
                raise ValueError(
                    f"TG4 initialization identity mismatch for {run_id}: {observed_identity}"
                )
            route = initialization["route"]
            observed_route = (
                route.get("lawam_future_off"),
                route.get("lawam_auxiliary_off"),
                route.get("lawam_conditioning_off"),
            )
            if observed_route != EXPECTED_ROUTE[arm]:
                raise ValueError(f"TG4 route mismatch for {run_id}: {observed_route}")
            if route.get("milestone_target") or route.get("dual_route"):
                raise ValueError(f"Legacy TG4 route enabled for {run_id}")
            parameter_trees.add(initialization["parameter_tree_sha256"])
            trainable_trees.add(initialization["trainable_tree_sha256"])
            optimizer_trees.add(initialization["optimizer_tree_sha256"])
            payload = initialization["initialization_payload_sha256"]
            if arm == "clean_base":
                clean_payload = payload
            else:
                pretrained_payloads.add(payload)

            order_paths = sorted(order_dir.glob("rank*.json"))
            if len(order_paths) != 4:
                raise ValueError(f"Expected four data-order records for {run_id}")
            order_records = [json.loads(path.read_text()) for path in order_paths]
            for rank, record in enumerate(order_records):
                expected_order = {
                    "arm": arm,
                    "training_seed": seed,
                    "rank": rank,
                    "world_size": 4,
                    "microbatches": 40000,
                    "samples": 640000,
                }
                observed_order = {key: record.get(key) for key in expected_order}
                if observed_order != expected_order:
                    raise ValueError(
                        f"TG4 data-order mismatch for {run_id} rank {rank}: {observed_order}"
                    )
            order = tuple(str(record["sha256"]) for record in order_records)
            arm_orders[arm] = order
            statistics_hashes.add(sha256(stats_path))
            runs[run_id] = {
                "run": str(run.relative_to(repo)),
                "final_checkpoint_bytes": final_path.stat().st_size,
                "optimizer_state_bytes": optimizer_path.stat().st_size,
                "normalization_sha256": sha256(stats_path),
                "initialization_audit": str(initialization_path),
                "data_order_sha256_by_rank": list(order),
            }

        if len(pretrained_payloads) != 1:
            raise ValueError(f"Pretrained initialization differs within seed {seed}")
        if clean_payload in pretrained_payloads:
            raise ValueError(f"Clean-base initialization equals pretrained initialization for seed {seed}")
        if len(set(arm_orders.values())) != 1:
            raise ValueError(f"Dataset order differs across TG4 arms for seed {seed}")
        orders_by_seed[seed] = next(iter(arm_orders.values()))

    checks = {
        "all_18_final_checkpoints_present": len(runs) == 18,
        "optimizer_state_at_step_20000_present": True,
        "parameter_tree_equal": len(parameter_trees) == 1,
        "trainable_tree_equal": len(trainable_trees) == 1,
        "optimizer_tree_equal": len(optimizer_trees) == 1,
        "pretrained_initialization_equal_within_seed": True,
        "clean_initialization_distinct": True,
        "dataset_order_equal_within_seed": True,
        "dataset_order_distinct_across_seeds": len(set(orders_by_seed.values())) == len(SEEDS),
        "normalization_equal": len(statistics_hashes) == 1,
        "non_arm_config_equal": len(normalized_configs) == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"TG4 integrity checks failed: {failed}")
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg4_training_integrity_v1",
        "complete": True,
        "checks": checks,
        "data_order_sha256_by_seed": {
            str(seed): list(order) for seed, order in orders_by_seed.items()
        },
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-checkpoint-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--min-optimizer-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    result = verify(
        args.repo.resolve(),
        min_checkpoint_bytes=args.min_checkpoint_bytes,
        min_optimizer_bytes=args.min_optimizer_bytes,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
