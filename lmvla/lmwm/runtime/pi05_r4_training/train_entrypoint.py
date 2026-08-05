#!/usr/bin/env python3
"""Bind the opt-in R4 runtime patches before entering LeRobot training."""

from __future__ import annotations

import sys

from lerobot.policies import factory as policy_factory
from lerobot.scripts import lerobot_train


def bind_training_runtime() -> None:
    patched = policy_factory.make_pre_post_processors
    if not getattr(patched, "_pi05_r4_runtime", False):
        raise RuntimeError("R4 processor runtime patch is not installed")
    lerobot_train.make_pre_post_processors = patched
    if lerobot_train.make_pre_post_processors is not patched:
        raise RuntimeError("failed to bind R4 processor runtime into lerobot_train")


def main() -> None:
    bind_training_runtime()
    if sys.argv[1:] == ["--check-binding"]:
        print("R4_TRAIN_ENTRYPOINT_BINDING_OK", flush=True)
        return
    lerobot_train.main()


if __name__ == "__main__":
    main()
