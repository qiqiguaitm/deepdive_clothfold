#!/usr/bin/env python3
"""Serve a LeRobot PI0.5 checkpoint through the OpenPI websocket protocol."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

from lerobot_pi05_action_bridge import action_feature_dim
from lerobot_pi05_action_bridge import trim_action_for_postprocessor


def _image_tensor(image: Any) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        chw = array
    else:
        chw = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
    return tensor.float().div_(255.0)


class LeRobotPi05Policy:
    def __init__(
        self, checkpoint: Path, tokenizer_path: Path, device: str, action_steps: int
    ) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.action_steps = action_steps
        self.infer_count = 0
        self.policy = PI05Policy.from_pretrained(str(checkpoint))
        self.policy.to(self.device)
        self.policy.eval()
        self.action_dim = action_feature_dim(self.policy.config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
            },
        )

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self.infer_count += 1
        request_id = self.infer_count
        started = time.perf_counter()
        images = observation["images"]
        batch: dict[str, Any] = {
            "observation.images.cam_high": _image_tensor(images["cam_high"]),
            "observation.images.cam_left_wrist": _image_tensor(images["cam_left_wrist"]),
            "observation.images.cam_right_wrist": _image_tensor(images["cam_right_wrist"]),
            "observation.state": torch.from_numpy(
                np.asarray(observation["state"], dtype=np.float32)
            ).unsqueeze(0),
            "task": [str(observation.get("prompt", ""))],
        }
        batch = self.preprocessor(batch)
        logging.info("infer[%d] preprocess_s=%.3f", request_id, time.perf_counter() - started)
        self.policy.reset()
        actions = []
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for step in range(self.action_steps):
                action = self.policy.select_action(batch)
                action = trim_action_for_postprocessor(action, self.action_dim)
                actions.append(self.postprocessor(action)[0].detach().float().cpu().numpy())
                if request_id == 1 and step + 1 in {1, 10, 25, self.action_steps}:
                    logging.info(
                        "infer[%d] actions=%d/%d elapsed_s=%.3f",
                        request_id,
                        step + 1,
                        self.action_steps,
                        time.perf_counter() - started,
                    )
        logging.info(
            "infer[%d] complete actions=%d elapsed_s=%.3f",
            request_id,
            len(actions),
            time.perf_counter() - started,
        )
        return {"actions": np.stack(actions, axis=0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy.dir", dest="policy_dir")
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get(
            "PALIGEMMA_TOKENIZER_PATH",
            "/vePFS/tim/hf_models/paligemma_tokenizer",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=50)
    args, _ = parser.parse_known_args()

    checkpoint = Path(args.checkpoint or args.policy_dir or "").expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"LeRobot checkpoint is incomplete: {checkpoint}")

    logging.basicConfig(level=logging.INFO)
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    if not (tokenizer_path / "tokenizer.model").is_file():
        raise FileNotFoundError(f"PaliGemma tokenizer is incomplete: {tokenizer_path}")

    policy = LeRobotPi05Policy(checkpoint, tokenizer_path, args.device, args.action_steps)
    server = WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"checkpoint": str(checkpoint), "framework": "lerobot-pytorch"},
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
