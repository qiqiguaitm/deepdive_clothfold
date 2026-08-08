#!/usr/bin/env python3
"""Exercise the Task_N 319 training dataloader with production worker settings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("TASK_N_319_KAI0_ROOT", str(REPO_ROOT / "kai0"))
os.environ.setdefault("TASK_N_319_INIT", str(REPO_ROOT / "kai0" / "checkpoints" / "pi05_base" / "params"))
sys.path.insert(0, str(REPO_ROOT / "kai0" / "src"))

from openpi.training.config import get_config  # noqa: E402
from openpi.training.data_loader import create_data_loader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()

    config = get_config("pi05_task_n_v5_0806_0807_319_sft")
    print(
        "config",
        config.name,
        config.batch_size,
        config.fsdp_devices,
        config.num_workers,
        config.num_train_steps,
        flush=True,
    )
    loader = create_data_loader(
        config,
        shuffle=True,
        num_batches=args.batches,
        framework="pytorch",
    )
    started = time.time()
    for index, (_observation, actions) in enumerate(loader, 1):
        print("batch", index, "actions", tuple(actions.shape), flush=True)
    print(
        "DATALOADER_OK",
        "samples",
        args.batches * config.batch_size,
        "seconds",
        round(time.time() - started, 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
