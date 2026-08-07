#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
SEEDS = (1000, 1001, 1002)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(paths)}: {paths}")
    return paths[0]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-state-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    checkpoint_root = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    init_root = repo / "logs/temporal_grounding/tg2/initialization"
    order_root = repo / "logs/temporal_grounding/tg2/data_order"

    runs = {}
    normalization_hashes = set()
    parameter_tree_hashes = set()
    trainable_tree_hashes = set()
    optimizer_tree_hashes = set()
    for seed in SEEDS:
        seed_payload_hashes = set()
        seed_order_by_arm = {}
        for arm in ARMS:
            run_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
            run = one(sorted(checkpoint_root.glob(f"*+{run_id}")), f"run {run_id}")
            config_path = run / "config.yaml"
            stats_path = run / "dataset_statistics.json"
            final_path = run / "final_model/pytorch_model.pt"
            state_dir = run / "checkpoints/steps_20000_state"
            optimizer_path = state_dir / "optimizer.bin"
            trainer_state_path = state_dir / "trainer_state.json"
            for path in (config_path, stats_path, final_path, optimizer_path, trainer_state_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if final_path.stat().st_size < args.min_state_bytes or optimizer_path.stat().st_size < args.min_state_bytes:
                raise ValueError(f"Truncated final checkpoint or optimizer state for {run_id}")
            if json.loads(trainer_state_path.read_text())["steps"] != 20000:
                raise ValueError(f"Wrong final optimizer step for {run_id}")

            config = yaml.safe_load(config_path.read_text())
            action = config["framework"]["action_model"]
            data = config["datasets"]["vla_data"]
            trainer = config["trainer"]
            expected_config = {
                "future_prediction": action["future_prediction"],
                "enable_loss_distill": action["enable_loss_distill"],
                "future_action_window_size": action["future_action_window_size"],
                "action_horizon": action["action_horizon"],
                "horizon_sec": action["flow_cfg"]["horizon_sec"],
                "data_mix": data["data_mix"],
                "sec_chunk": data["sec_chunk"],
                "batch": data["per_device_batch_size"],
                "gradient_accumulation_steps": trainer["gradient_accumulation_steps"],
                "max_train_steps": trainer["max_train_steps"],
            }
            if expected_config != {
                "future_prediction": True,
                "enable_loss_distill": True,
                "future_action_window_size": 49,
                "action_horizon": 50,
                "horizon_sec": 1.0,
                "data_mix": "robotwin2_lmwm_all6_v2",
                "sec_chunk": 1.0,
                "batch": 16,
                "gradient_accumulation_steps": 2,
                "max_train_steps": 20000,
            }:
                raise ValueError(f"TG2 config drift in {run_id}: {expected_config}")

            initialization = json.loads((init_root / f"{run_id}.json").read_text())
            expected_route = {
                "future_off": arm == "future_off",
                "raw": arm == "raw_milestone",
            }
            route = initialization["route"]
            if route["lawam_future_off"] != expected_route["future_off"]:
                raise ValueError(f"Future-off route mismatch for {run_id}")
            if bool(route["milestone_target"]) != expected_route["raw"]:
                raise ValueError(f"Raw milestone route mismatch for {run_id}")
            if route["dual_route"]:
                raise ValueError(f"TG2 must not use the historical dual route: {run_id}")
            if expected_route["raw"] and not route["require_full_target_coverage"]:
                raise ValueError(f"Raw milestone full-coverage guard absent: {run_id}")
            if initialization["optimizer_state_entries_before_training"] != 0:
                raise ValueError(f"Nonempty initial optimizer state for {run_id}")
            parameter_tree_hashes.add(initialization["parameter_tree_sha256"])
            trainable_tree_hashes.add(initialization["trainable_tree_sha256"])
            optimizer_tree_hashes.add(initialization["optimizer_tree_sha256"])
            seed_payload_hashes.add(initialization["initialization_payload_sha256"])

            order_files = sorted((order_root / run_id).glob("rank*.json"))
            if len(order_files) != 4:
                raise ValueError(f"Expected four data-order rank records for {run_id}")
            orders = [json.loads(path.read_text()) for path in order_files]
            if [row["rank"] for row in orders] != list(range(4)) or any(
                row["world_size"] != 4 for row in orders
            ):
                raise ValueError(f"Wrong TG2 world size/ranks for {run_id}")
            seed_order_by_arm[arm] = [row["sha256"] for row in orders]
            normalization_hashes.add(sha256(stats_path))
            runs[run_id] = {
                "run": str(run.relative_to(repo)),
                "final_checkpoint_bytes": final_path.stat().st_size,
                "optimizer_state_bytes": optimizer_path.stat().st_size,
                "normalization_sha256": sha256(stats_path),
                "initialization_audit": str((init_root / f"{run_id}.json").relative_to(repo)),
                "data_order_sha256_by_rank": seed_order_by_arm[arm],
            }
        if len(seed_payload_hashes) != 1:
            raise ValueError(f"Initialization payload differs across TG2 arms for seed {seed}")
        if len({tuple(value) for value in seed_order_by_arm.values()}) != 1:
            raise ValueError(f"Dataset order differs across TG2 arms for seed {seed}")

    if len(parameter_tree_hashes) != 1 or len(trainable_tree_hashes) != 1:
        raise ValueError("TG2 parameter or trainable trees differ across arms/seeds")
    if len(optimizer_tree_hashes) != 1:
        raise ValueError("TG2 optimizer parameter trees differ across arms/seeds")
    if len(normalization_hashes) != 1:
        raise ValueError("TG2 normalization payloads differ")
    result = {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg2_training_integrity_v1",
        "complete": True,
        "runs": runs,
        "checks": {
            "parameter_tree_equal": True,
            "trainable_tree_equal": True,
            "initialization_payload_equal_within_seed": True,
            "dataset_order_equal_within_seed": True,
            "normalization_equal": True,
            "optimizer_tree_equal": True,
            "optimizer_state_at_step_20000_present": True,
            "raw_target_coverage_guard_enabled": True,
            "fixed_final_checkpoint_present": True,
        },
    }
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
