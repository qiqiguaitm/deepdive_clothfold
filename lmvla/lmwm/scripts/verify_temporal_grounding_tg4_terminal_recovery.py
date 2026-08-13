#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CELLS = (("auxiliary_only", 1100), ("auxiliary_only", 1101))
ALLOWED_CELLS = {
    *DEFAULT_CELLS,
    ("clean_base", 1100),
    ("clean_base", 1101),
    ("clean_base", 1102),
    ("auxiliary_only", 1102),
    ("full", 1100),
    ("full", 1101),
    ("full", 1102),
    ("parameter_matched_null", 1101),
    ("parameter_matched_null", 1102),
}
EXPECTED_ROUTE = {
    "clean_base": (False, False, False),
    "auxiliary_only": (False, False, True),
    "full": (False, False, False),
    "parameter_matched_null": (True, False, False),
}
EXPECTED_ERROR = "line 118: el.future_action_window_size=49: command not found"


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


def verify_run(repo: Path, arm: str, seed: int, resource: str) -> dict[str, Any]:
    if (arm, seed) not in ALLOWED_CELLS:
        raise ValueError(f"TG4 terminal recovery is not allowlisted for {arm}:{seed}")
    run_id = f"temporal_grounding_tg4_{arm}_seed{seed}"
    task_id = f"{run_id}_train"
    checkpoint_root = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    run = one(sorted(checkpoint_root.glob(f"*+{run_id}")), f"run {run_id}")
    state = run / "checkpoints/steps_20000_state"
    paths = {
        "config": run / "config.json",
        "statistics": run / "dataset_statistics.json",
        "final": run / "final_model/pytorch_model.pt",
        "optimizer": state / "optimizer.bin",
        "trainer_state": state / "trainer_state.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if paths["final"].stat().st_size < 1_000_000_000:
        raise ValueError(f"Truncated final checkpoint for {run_id}")
    if paths["optimizer"].stat().st_size < 1_000_000_000:
        raise ValueError(f"Truncated optimizer state for {run_id}")
    if json.loads(paths["trainer_state"].read_text())["steps"] != 20000:
        raise ValueError(f"Wrong final optimizer step for {run_id}")

    config = json.loads(paths["config"].read_text())
    expected = {
        "run_id": run_id,
        "seed": seed,
        "data_mix": "robotwin2_lmwm_all6_v2",
        "batch": 16,
        "workers": 8,
        "in_order": True,
        "grad_accum": 2,
        "steps": 20000,
        "save_interval": 20000,
        "future_prediction": arm != "clean_base",
        "auxiliary_loss": arm in {
            "auxiliary_only",
            "conditioning_only",
            "parameter_matched_null",
            "full",
        },
        "ddp_find_unused_parameters": arm == "clean_base",
        "pretrained_checkpoint": (
            None
            if arm == "clean_base"
            else "results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt"
        ),
        "load_pretrained_policy_flow": arm != "clean_base",
    }
    action = config["framework"]["action_model"]
    data = config["datasets"]["vla_data"]
    trainer = config["trainer"]
    observed = {
        "run_id": config["run_id"],
        "seed": config["seed"],
        "data_mix": data["data_mix"],
        "batch": data["per_device_batch_size"],
        "workers": data["num_workers"],
        "in_order": data["in_order"],
        "grad_accum": trainer["gradient_accumulation_steps"],
        "steps": trainer["max_train_steps"],
        "save_interval": trainer["save_interval"],
        "future_prediction": action["future_prediction"],
        "auxiliary_loss": action["enable_loss_distill"],
        "ddp_find_unused_parameters": trainer["ddp_find_unused_parameters"],
        "pretrained_checkpoint": trainer["pretrained_checkpoint"],
        "load_pretrained_policy_flow": trainer["load_pretrained_policy_flow"],
    }
    if observed != expected:
        raise ValueError(f"Frozen TG4 config mismatch for {run_id}: {observed}")

    initialization_path = (
        repo / "logs/temporal_grounding/tg4/initialization" / f"{run_id}.json"
    )
    initialization = json.loads(initialization_path.read_text())
    identity = {
        "schema_version": initialization.get("schema_version"),
        "protocol": initialization.get("protocol"),
        "arm": initialization.get("arm"),
        "training_seed": initialization.get("training_seed"),
        "optimizer_state_entries_before_training": initialization.get(
            "optimizer_state_entries_before_training"
        ),
    }
    expected_identity = {
        "schema_version": 1,
        "protocol": "lawam_matched_initialization_v1",
        "arm": arm,
        "training_seed": seed,
        "optimizer_state_entries_before_training": 0,
    }
    if identity != expected_identity:
        raise ValueError(f"Initialization identity mismatch for {run_id}: {identity}")
    route = initialization["route"]
    observed_route = (
        route.get("lawam_future_off"),
        route.get("lawam_auxiliary_off"),
        route.get("lawam_conditioning_off"),
    )
    if observed_route != EXPECTED_ROUTE[arm] or route.get(
        "milestone_target"
    ) or route.get("dual_route"):
        raise ValueError(f"Route mismatch for {run_id}: {route}")

    order_dir = repo / "logs/temporal_grounding/tg4/data_order" / run_id
    order_paths = sorted(order_dir.glob("rank*.json"))
    if len(order_paths) != 4:
        raise ValueError(f"Expected four data-order records for {run_id}")
    order_hashes: list[str] = []
    for rank, path in enumerate(order_paths):
        record = json.loads(path.read_text())
        order_identity = {
            "arm": record.get("arm"),
            "training_seed": record.get("training_seed"),
            "rank": record.get("rank"),
            "world_size": record.get("world_size"),
            "microbatches": record.get("microbatches"),
            "samples": record.get("samples"),
        }
        expected_order = {
            "arm": arm,
            "training_seed": seed,
            "rank": rank,
            "world_size": 4,
            "microbatches": 40000,
            "samples": 640000,
        }
        if order_identity != expected_order:
            raise ValueError(f"Data-order mismatch for {run_id}: {order_identity}")
        order_hashes.append(record["sha256"])

    log = one(
        sorted(
            (repo / "logs/temporal_grounding/tg4/entrypoint").glob(
                f"{arm}_s{seed}_{resource}_*.log"
            )
        ),
        f"entrypoint log {run_id}",
    )
    log_text = log.read_text(errors="replace")
    required_log_evidence = (
        f"{run_id}: 100%",
        "and that's all",
        EXPECTED_ERROR,
    )
    missing = [text for text in required_log_evidence if text not in log_text]
    if missing:
        raise ValueError(f"Missing terminal log evidence for {run_id}: {missing}")
    if log_text.count(EXPECTED_ERROR) != 1:
        raise ValueError(f"Unexpected post-training error count for {run_id}")

    return {
        "task_id": task_id,
        "run": str(run.relative_to(repo)),
        "entrypoint_log": str(log.relative_to(repo)),
        "final_checkpoint_bytes": paths["final"].stat().st_size,
        "final_checkpoint_sha256": sha256(paths["final"]),
        "optimizer_state_bytes": paths["optimizer"].stat().st_size,
        "trainer_steps": 20000,
        "data_order_sha256_by_rank": order_hashes,
        "recovered_terminal_reason": "runner mutated only after successful training child exit",
    }


def verify(
    repo: Path, cells: list[tuple[str, int]], resource: str
) -> dict[str, Any]:
    runs = [verify_run(repo, arm, seed, resource) for arm, seed in cells]
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg4_validated_terminal_recovery_v1",
        "complete": True,
        "accepted_task_ids": [run["task_id"] for run in runs],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell",
        action="append",
        help="Allowlisted ARM:SEED cell; defaults to the two East auxiliary recoveries",
    )
    parser.add_argument("--resource", choices=("east", "north"), default="east")
    args = parser.parse_args()
    cells = list(DEFAULT_CELLS)
    if args.cell:
        cells = []
        for value in args.cell:
            arm, separator, seed = value.rpartition(":")
            if not separator:
                raise ValueError(f"Invalid --cell value: {value}")
            cells.append((arm, int(seed)))
    result = verify(args.repo.resolve(), cells, args.resource)
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
