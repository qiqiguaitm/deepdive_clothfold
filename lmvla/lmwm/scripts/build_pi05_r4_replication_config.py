#!/usr/bin/env python3
"""Build a seed-replicated R4 config without changing the frozen recipe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pi05_r4_train_config import ARMS, build_config  # noqa: E402


REPLICATION_SEEDS = (1001, 1002)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=REPLICATION_SEEDS, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    )
    config["seed"] = args.seed
    config["job_name"] = f"pi05-r4-{args.arm}-seed{args.seed}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "seed": args.seed,
                "world_size": args.world_size,
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
