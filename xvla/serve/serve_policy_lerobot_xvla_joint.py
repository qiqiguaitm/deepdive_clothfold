"""Serve a LeRobot-format XVLA checkpoint as a kai0 joint-action policy.

This is for checkpoints saved by `lerobot_train` as:
  pretrained_model/{config.json, model.safetensors, policy_*processor*.safetensors}

It emits the existing kai0 WebSocket protocol with:
  action_kind="joint", action_dim=14, action_horizon=30
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "kai0" / "packages" / "openpi-client" / "src"))

from openpi_client import base_policy as _base_policy  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402

logger = logging.getLogger("lerobot_xvla_joint")

_IMG_KEYS = [
    "observation.images.image",
    "observation.images.image2",
    "observation.images.image3",
]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_hwc_uint8(arr: Any, slot: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.transpose(a, (1, 2, 0))
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"image '{slot}' must be HWC/CHW 3ch; got {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out


def _load_sidecar(root: Path) -> dict[str, Any]:
    p = root / "sidecar.json"
    return json.load(open(p)) if p.is_file() else {}


class LeRobotXVLAJointPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        ckpt_dir: Path,
        tokenizer_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        default_prompt: str,
        default_domain_id: int,
        image_slots: list[str],
        image_sizes: list[int],
        imagenet_norm: bool,
        fixed_seed: int | None,
        clip_to_train_range: bool,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._default_prompt = default_prompt
        self._domain_id = int(default_domain_id)
        self._image_slots = image_slots
        self._image_sizes = image_sizes
        self._imagenet_norm = bool(imagenet_norm)
        self._fixed_seed = fixed_seed

        logger.info("loading LeRobot XVLA policy from %s", ckpt_dir)
        self._policy = XVLAPolicy.from_pretrained(ckpt_dir, local_files_only=True, strict=False)
        self._policy.config.device = str(device)
        self._policy.config.dtype = "bfloat16" if dtype == torch.bfloat16 else "float32"
        self._policy = self._policy.to(device, dtype=dtype).eval()

        logger.info("loading tokenizer from %s", tokenizer_dir)
        self._tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self._tok_cache: dict[str, torch.Tensor] = {}

        stats_path = ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        stats = load_file(str(stats_path))
        self._a_mean = stats["action.mean"].to(device=device, dtype=torch.float32)
        self._a_std = stats["action.std"].to(device=device, dtype=torch.float32)
        self._a_min = stats["action.min"].to(device=device, dtype=torch.float32)
        self._a_max = stats["action.max"].to(device=device, dtype=torch.float32)
        self._clip = bool(clip_to_train_range)

        self._in_mean = _IMAGENET_MEAN.to(device=device, dtype=dtype)
        self._in_std = _IMAGENET_STD.to(device=device, dtype=dtype)
        self._chunk = int(getattr(self._policy.config, "chunk_size", 30))
        self._action_dim = int(self._a_mean.numel())
        self._metadata = {
            "action_kind": "joint",
            "action_dim": self._action_dim,
            "action_horizon": self._chunk,
            "model_name": f"lerobot_xvla_joint::{ckpt_dir.parent.name}",
            "prompt": default_prompt,
            "domain_id": self._domain_id,
            "image_slots": self._image_slots,
            "image_norm": "imagenet" if self._imagenet_norm else "none",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _tokens(self, prompt: str) -> torch.Tensor:
        if prompt not in self._tok_cache:
            ids = self._tok(
                [prompt],
                padding="max_length",
                max_length=50,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            self._tok_cache[prompt] = ids.to(self._device)
        return self._tok_cache[prompt]

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        images = obs.get("images") or obs.get("image") or {}
        if not isinstance(images, dict):
            raise ValueError(f"obs['images'] must be a dict, got {type(images).__name__}")

        batch: dict[str, torch.Tensor] = {}
        for model_key, slot, size in zip(_IMG_KEYS, self._image_slots, self._image_sizes):
            if slot not in images:
                raise ValueError(f"obs['images'] missing '{slot}'; available={list(images.keys())}")
            hwc = _resize_pad(_to_hwc_uint8(images[slot], slot), size)
            t = torch.from_numpy(hwc).permute(2, 0, 1).float().div_(255.0)
            t = t.unsqueeze(0).to(self._device, dtype=self._dtype)
            if self._imagenet_norm:
                t = (t - self._in_mean) / self._in_std
            batch[model_key] = t

        state = obs.get("state")
        if state is None:
            raise ValueError("obs['state'] is required for this 14D joint-space XVLA checkpoint")
        state_np = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        if state_np.shape[0] < 14:
            raise ValueError(f"obs['state'] must have at least 14 dims; got {state_np.shape[0]}")
        batch["observation.state"] = torch.from_numpy(state_np[:14]).unsqueeze(0).to(
            self._device, dtype=self._dtype
        )

        prompt = obs.get("prompt") or self._default_prompt
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        batch["observation.language.tokens"] = self._tokens(str(prompt))
        batch["domain_id"] = torch.tensor([self._domain_id], dtype=torch.long, device=self._device)

        if self._fixed_seed is not None:
            torch.manual_seed(self._fixed_seed)
            if self._device.type == "cuda":
                torch.cuda.manual_seed_all(self._fixed_seed)

        infer_t0 = time.monotonic()
        with torch.inference_mode():
            raw = self._policy.predict_action_chunk(batch)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.monotonic() - infer_t0) * 1000.0

        actions = raw.float() * self._a_std.view(1, 1, -1) + self._a_mean.view(1, 1, -1)
        if self._clip:
            actions = torch.maximum(torch.minimum(actions, self._a_max.view(1, 1, -1)), self._a_min.view(1, 1, -1))
        actions_np = actions.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return {
            "actions": actions_np,
            "action_kind": "joint",
            "server_backend": "lerobot_xvla_joint",
            "policy_timing": {
                "infer_ms": float(infer_ms),
                "total_ms": float((time.monotonic() - t0) * 1000.0),
            },
        }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve LeRobot XVLA checkpoint as kai0 joint policy")
    p.add_argument("--ckpt_dir", type=Path, required=True, help="Path to pretrained_model dir")
    p.add_argument("--tokenizer", type=Path, default=_REPO_ROOT / "xvla" / "assets" / "bart-large-tokenizer")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--default-prompt", default=None)
    p.add_argument("--default-domain-id", type=int, default=None)
    p.add_argument(
        "--image-slots",
        nargs=3,
        default=["top_head", "hand_left", "hand_right"],
        help="obs['images'] keys mapped to model image,image2,image3. foldDATA training used high,left,right.",
    )
    p.add_argument("--image-sizes", nargs=3, type=int, default=[256, 256, 224])
    p.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--seed", type=int, default=42, help="Fixed flow-matching seed; -1 for random each infer")
    p.add_argument("--clip-to-train-range", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup-iters", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sidecar = _load_sidecar(args.ckpt_dir.parent)
    default_prompt = args.default_prompt or sidecar.get("deploy_prompt") or "Flatten and fold the cloth."
    default_domain_id = (
        args.default_domain_id if args.default_domain_id is not None else int(sidecar.get("deploy_domain_id", 0))
    )
    if args.imagenet_norm is None:
        imagenet_norm = str(sidecar.get("image_norm", "imagenet")).lower() != "none"
    else:
        imagenet_norm = bool(args.imagenet_norm)
    seed = None if args.seed < 0 else int(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)

    policy = LeRobotXVLAJointPolicy(
        ckpt_dir=args.ckpt_dir,
        tokenizer_dir=args.tokenizer,
        device=device,
        dtype=dtype,
        default_prompt=default_prompt,
        default_domain_id=default_domain_id,
        image_slots=list(args.image_slots),
        image_sizes=list(args.image_sizes),
        imagenet_norm=imagenet_norm,
        fixed_seed=seed,
        clip_to_train_range=args.clip_to_train_range,
    )

    if args.warmup_iters > 0:
        dummy = {
            "images": {
                slot: np.zeros((480, 640, 3), dtype=np.uint8)
                for slot in args.image_slots
            },
            "state": np.zeros(14, dtype=np.float32),
            "prompt": default_prompt,
        }
        for i in range(args.warmup_iters):
            out = policy.infer(dummy)
            logger.info(
                "warmup %d/%d: actions=%s total=%.1fms",
                i + 1,
                args.warmup_iters,
                out["actions"].shape,
                out["policy_timing"]["total_ms"],
            )

    logger.info(
        "Serving LeRobot XVLA joint policy on ws://%s:%d (action_kind=joint, dim=%d, H=%d, host=%s)",
        args.host,
        args.port,
        policy.metadata["action_dim"],
        policy.metadata["action_horizon"],
        socket.gethostname(),
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
