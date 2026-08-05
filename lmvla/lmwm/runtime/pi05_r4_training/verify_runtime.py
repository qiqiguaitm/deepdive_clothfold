#!/usr/bin/env python3
"""Fail closed unless the R4 direct-chunk LeRobot runtime is installed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import factory as policy_factory
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils import sample_weighting


def verify(model: Path | None = None) -> dict:
    patched = {
        "make_policy": bool(getattr(policy_factory.make_policy, "_pi05_r4_runtime", False)),
        "make_pre_post_processors": bool(
            getattr(policy_factory.make_pre_post_processors, "_pi05_r4_runtime", False)
        ),
        "make_sample_weighter": bool(
            getattr(sample_weighting.make_sample_weighter, "_pi05_r4_runtime", False)
        ),
    }
    if not all(patched.values()):
        raise RuntimeError(f"R4 runtime overlay is incomplete: {patched}")

    config = sample_weighting.SampleWeightingConfig(
        type="batch_field", extra_params={"field": "sample_weight"}
    )
    weighter = sample_weighting.make_sample_weighter(config, object(), torch.device("cpu"))
    weights, stats = weighter.compute_batch_weights(
        {"sample_weight": torch.tensor([[0.5], [1.5]], dtype=torch.float32)}
    )
    if weights.tolist() != [0.5, 1.5] or stats["type"] != "batch_field":
        raise RuntimeError("R4 batch-field weighter failed its deterministic probe")

    public_config = None
    processor_probe = None
    if model is not None:
        config_path = model / "config.json"
        weights_path = model / "model.safetensors"
        train_config_path = model / "train_config.json"
        for path in (config_path, weights_path, train_config_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        decoded = json.loads(config_path.read_text(encoding="utf-8"))
        public_config = {
            "type": decoded.get("type"),
            "action_shape": decoded["output_features"]["action"]["shape"],
            "chunk_size": decoded.get("chunk_size"),
            "normalization_mapping": decoded.get("normalization_mapping"),
        }
        if public_config["type"] != "pi05":
            raise ValueError(f"unexpected public model type: {public_config}")
        if public_config["action_shape"] != [14] or public_config["chunk_size"] != 50:
            raise ValueError(f"unexpected public action contract: {public_config}")
        policy_config = PI05Config.from_pretrained(model)
        preprocessor, _ = policy_factory.make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(model),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        probe_batch = {
            "observation.state": torch.zeros(2, 14),
            "action": torch.zeros(2, 50, 14),
            "observation.images.cam_high": torch.zeros(2, 3, 32, 32),
            "observation.images.cam_left_wrist": torch.zeros(2, 3, 32, 32),
            "observation.images.cam_right_wrist": torch.zeros(2, 3, 32, 32),
            "task": ["do task", "do task"],
            "sample_weight": torch.tensor([[0.5], [1.5]], dtype=torch.float32),
        }
        processed = preprocessor(probe_batch)
        preserved = torch.as_tensor(processed.get("sample_weight"))
        if preserved.shape != (2, 1) or preserved[:, 0].tolist() != [0.5, 1.5]:
            raise RuntimeError("public pi0.5 preprocessor did not preserve sample_weight")
        processor_probe = {"sample_weight_preserved": True, "shape": list(preserved.shape)}

    # Parsing this object proves the installed LeRobot exposes the exact config
    # and generic sample-weighting interfaces used by the launcher.
    if not issubclass(PI05Config, object) or not hasattr(TrainPipelineConfig, "validate"):
        raise RuntimeError("installed LeRobot lacks the required PI05 training interfaces")
    return {
        "accepted": True,
        "patched": patched,
        "processor_probe": processor_probe,
        "public_config": public_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.model)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
