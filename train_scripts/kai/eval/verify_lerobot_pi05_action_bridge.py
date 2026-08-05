#!/usr/bin/env python3
"""Verify PI0.5 padded actions against a checkpoint's real postprocessor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from lerobot_pi05_action_bridge import action_feature_dim
from lerobot_pi05_action_bridge import trim_action_for_postprocessor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--native-action-dim", type=int, default=32)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = checkpoint / "config.json"
    stats_path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not config_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint postprocessor assets: {checkpoint}")

    config = PI05Config.from_pretrained(str(checkpoint))
    configured_dim = action_feature_dim(config)
    if args.native_action_dim < configured_dim:
        raise ValueError(
            f"Native action dimension {args.native_action_dim} is smaller than {configured_dim}"
        )
    _, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
    )
    native_action = torch.zeros((1, args.native_action_dim), dtype=torch.float32)
    trimmed = trim_action_for_postprocessor(native_action, configured_dim)
    restored = postprocessor(trimmed)
    if restored.shape != (1, configured_dim):
        raise ValueError(f"Unexpected postprocessed action shape: {tuple(restored.shape)}")
    if not torch.isfinite(restored).all():
        raise ValueError("Postprocessed action contains non-finite values")

    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "config_sha256": _sha256(config_path),
        "postprocessor_stats_sha256": _sha256(stats_path),
        "native_action_dim": args.native_action_dim,
        "configured_action_dim": configured_dim,
        "trimmed_shape": list(trimmed.shape),
        "postprocessed_shape": list(restored.shape),
        "finite": True,
        "status": "passed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.marker.write_text(
        f"status=passed\nreport={args.report.resolve()}\ncheckpoint={checkpoint}\n"
    )


if __name__ == "__main__":
    main()
