#!/usr/bin/env python3
"""Serve public pi0.5 with an isolated R3 semantic prompt intervention."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from pi05_r3_semantic_prompt import SemanticPromptFormatter
from serve_lerobot_pi05 import LeRobotPi05Policy, _image_tensor


class R3LeRobotPi05Policy(LeRobotPi05Policy):
    def __init__(
        self,
        checkpoint: Path,
        tokenizer_path: Path,
        device: str,
        action_steps: int,
        formatter: SemanticPromptFormatter,
    ) -> None:
        super().__init__(checkpoint, tokenizer_path, device, action_steps)
        self.formatter = formatter

    def infer(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self.infer_count += 1
        request_id = self.infer_count
        started = time.perf_counter()
        images = observation["images"]
        base_prompt = str(observation.get("prompt", ""))
        prompt = self.formatter.format(base_prompt, observation)
        batch: dict[str, Any] = {
            "observation.images.cam_high": _image_tensor(images["cam_high"]),
            "observation.images.cam_left_wrist": _image_tensor(images["cam_left_wrist"]),
            "observation.images.cam_right_wrist": _image_tensor(images["cam_right_wrist"]),
            "observation.state": torch.from_numpy(
                np.asarray(observation["state"], dtype=np.float32)
            ).unsqueeze(0),
            "task": [prompt],
        }
        batch = self.preprocessor(batch)
        logging.info("infer[%d] mode=%s preprocess_s=%.3f", request_id, self.formatter.mode, time.perf_counter() - started)
        self.policy.reset()
        actions = []
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            for _ in range(self.action_steps):
                action = self.policy.select_action(batch)
                actions.append(self.postprocessor(action)[0].detach().float().cpu().numpy())
        logging.info("infer[%d] complete actions=%d elapsed_s=%.3f", request_id, len(actions), time.perf_counter() - started)
        return {"actions": np.stack(actions, axis=0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy.dir", dest="policy_dir")
    parser.add_argument("--tokenizer-path", default=os.environ.get("PALIGEMMA_TOKENIZER_PATH", "/vePFS/tim/hf_models/paligemma_tokenizer"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-steps", type=int, default=50)
    parser.add_argument("--semantic-vocabulary", default=os.environ.get("R3_SEMANTIC_VOCABULARY"))
    parser.add_argument("--semantic-task-map", default=os.environ.get("R3_SEMANTIC_TASK_MAP"))
    parser.add_argument("--semantic-mode", default=os.environ.get("R3_SEMANTIC_MODE", "none"))
    args, _ = parser.parse_known_args()

    checkpoint = Path(args.checkpoint or args.policy_dir or "").expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    vocabulary_path = Path(args.semantic_vocabulary or "").expanduser().resolve()
    task_map_path = Path(args.semantic_task_map or "").expanduser().resolve()
    for required in (checkpoint / "config.json", tokenizer_path / "tokenizer.model", vocabulary_path, task_map_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    logging.basicConfig(level=logging.INFO)
    formatter = SemanticPromptFormatter(vocabulary_path, task_map_path, args.semantic_mode)
    policy = R3LeRobotPi05Policy(checkpoint, tokenizer_path, args.device, args.action_steps, formatter)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={"checkpoint": str(checkpoint), "framework": "lerobot-pytorch", "r3_mode": args.semantic_mode},
    ).serve_forever()


if __name__ == "__main__":
    main()
