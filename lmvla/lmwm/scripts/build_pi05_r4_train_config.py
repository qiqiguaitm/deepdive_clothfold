#!/usr/bin/env python3
"""Build one matched R4 pi0.5 fine-tuning config from the public recipe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ARMS = ("ordinary", "terminal_outcome", "outcome_free_crave")
OFFICIAL_GLOBAL_BATCH = 16
OFFICIAL_SEED = 1000
OFFICIAL_LR = 2.5e-5


def _assert_public_recipe(config: dict) -> None:
    policy = config["policy"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    expected = {
        "steps": (config["steps"], 50_000),
        "batch_size": (config["batch_size"], OFFICIAL_GLOBAL_BATCH),
        "seed": (config["seed"], OFFICIAL_SEED),
        "learning_rate": (optimizer["lr"], OFFICIAL_LR),
        "weight_decay": (optimizer["weight_decay"], 0.01),
        "warmup_steps": (scheduler["num_warmup_steps"], 1_000),
        "decay_steps": (scheduler["num_decay_steps"], 30_000),
        "decay_lr": (scheduler["decay_lr"], 2.5e-6),
        "chunk_size": (policy["chunk_size"], 50),
        "vision_encoder_frozen": (policy["freeze_vision_encoder"], False),
        "gradient_checkpointing": (policy["gradient_checkpointing"], True),
        "compile_model": (policy["compile_model"], True),
        "compile_mode": (policy["compile_mode"], "max-autotune"),
    }
    mismatches = [f"{name}={actual!r}, expected {wanted!r}" for name, (actual, wanted) in expected.items() if actual != wanted]
    if optimizer["betas"] != [0.9, 0.95] or optimizer["eps"] != 1e-8:
        mismatches.append("AdamW beta/epsilon settings differ from the public recipe")
    if policy["output_features"]["action"]["shape"] != [14]:
        mismatches.append("public action dimension is not 14")
    if mismatches:
        raise ValueError("public pi0.5 recipe drifted: " + "; ".join(mismatches))


def build_config(
    public_config: Path,
    arm: str,
    world_size: int,
    steps: int,
    output_dir: Path,
    dataset_root: Path,
    model_path: Path,
    sidecar: Path,
    *,
    smoke: bool = False,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown R4 arm: {arm}")
    if world_size <= 0 or OFFICIAL_GLOBAL_BATCH % world_size:
        raise ValueError(
            f"world size must divide the frozen global batch {OFFICIAL_GLOBAL_BATCH}: {world_size}"
        )
    if steps <= 0:
        raise ValueError("training steps must be positive")
    config = json.loads(public_config.read_text(encoding="utf-8"))
    _assert_public_recipe(config)

    config["dataset"].update(
        {
            "repo_id": "local/pi05-r4-query-train-v1",
            "root": str(dataset_root.resolve()),
            "video_backend": "pyav",
            "eval_split": 0.0,
        }
    )
    config["policy"].update(
        {
            "pretrained_path": str(model_path.resolve()),
            "push_to_hub": False,
            "repo_id": None,
        }
    )
    config.update(
        {
            "output_dir": str(output_dir.resolve()),
            "job_name": f"pi05-r4-{arm}-seed{OFFICIAL_SEED}",
            "seed": OFFICIAL_SEED,
            "batch_size": OFFICIAL_GLOBAL_BATCH // world_size,
            "steps": steps,
            "num_workers": 4,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "dataloader_multiprocessing_context": "spawn",
            "save_checkpoint": not smoke,
            "save_freq": steps,
            "log_freq": 1 if smoke else 50,
            "env_eval_freq": 0,
            "eval_steps": 0,
            "use_policy_training_preset": False,
            "save_checkpoint_to_hub": False,
        }
    )
    config["wandb"].update({"enable": False, "run_id": None, "mode": "disabled"})
    if arm == "ordinary":
        config["sample_weighting"] = None
    elif arm == "terminal_outcome":
        config["sample_weighting"] = {
            "type": "batch_field",
            "extra_params": {"field": "sample_weight"},
        }
    else:
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        config["sample_weighting"] = {
            "type": "sidecar_index",
            "extra_params": {"path": str(sidecar.resolve()), "field": "weight"},
        }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = build_config(
        args.public_config,
        args.arm,
        args.world_size,
        args.steps,
        args.output_dir,
        args.dataset_root,
        args.model_path,
        args.sidecar,
        smoke=args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "world_size": args.world_size,
                "per_process_batch": config["batch_size"],
                "effective_batch": config["batch_size"] * args.world_size,
                "steps": config["steps"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
