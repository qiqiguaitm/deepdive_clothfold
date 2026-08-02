#!/usr/bin/env python3
"""Encode one reference frame in the frozen A3 checkpoint vision space."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
from PIL import Image


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_frame(path: pathlib.Path, frame: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        key = str(frame)
        if key not in archive:
            raise KeyError(f"frame {frame} is absent from {path}")
        encoded = np.asarray(archive[key], dtype=np.uint8)
    image = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert("RGB"), dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"decoded image has invalid shape {image.shape}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--config", default="pi05_robotwin_a3_live_residual_prefix_official_eval")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--frame-cache", type=pathlib.Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--pair-artifact", type=pathlib.Path)
    parser.add_argument("--pair-task", type=int)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pair_provenance = None
    pair_args = (args.pair_artifact, args.pair_task, args.episode)
    if any(value is not None for value in pair_args):
        if any(value is None for value in pair_args):
            parser.error("--pair-artifact, --pair-task, and --episode must be provided together")
        with np.load(args.pair_artifact, allow_pickle=False) as pairs:
            required = {"cur_ep", "cur_fi", "tgt_fi", "pair_task"}
            missing = required.difference(pairs.files)
            if missing:
                raise KeyError(f"pair artifact is missing arrays: {sorted(missing)}")
            selected = (
                (pairs["pair_task"] == args.pair_task)
                & (pairs["cur_ep"] == args.episode)
                & (pairs["cur_fi"] == 0)
            )
            target_frames = np.unique(pairs["tgt_fi"][selected])
        if target_frames.shape != (1,) or int(target_frames[0]) != args.frame:
            raise ValueError(
                "frame is not the unique milestone+1 target at cur_fi=0: "
                f"task={args.pair_task} episode={args.episode} "
                f"expected={target_frames.tolist()} requested={args.frame}"
            )
        pair_provenance = {
            "artifact": str(args.pair_artifact),
            "pair_task": args.pair_task,
            "episode": args.episode,
            "cur_fi": 0,
            "tgt_fi": args.frame,
            "sha256": sha256(args.pair_artifact),
        }

    sys.path.insert(0, str(args.repo / "kai0/src"))
    from openpi.models import model as model_lib
    from openpi.shared import image_tools
    from openpi.training import config as training_config

    image = decode_frame(args.frame_cache, args.frame)
    resized = image_tools.resize_with_pad(jnp.asarray(image), 224, 224)
    normalized = resized.astype(jnp.float32) / 255.0 * 2.0 - 1.0

    config = training_config.get_config(args.config)
    model = config.model.load(
        model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    )
    image_tokens, _ = model.PaliGemma.img(normalized[None], train=False)
    feature = np.asarray(jnp.mean(image_tokens.astype(jnp.float32), axis=1)[0])
    if feature.ndim != 1 or not np.all(np.isfinite(feature)):
        raise ValueError(f"invalid extracted feature shape={feature.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, feature.astype(np.float32))
    metadata_path = args.checkpoint / "params/_METADATA"
    manifest = {
        "config": args.config,
        "checkpoint": str(args.checkpoint),
        "frame_cache": str(args.frame_cache),
        "frame": args.frame,
        "decoded_image_shape": list(image.shape),
        "feature_shape": list(feature.shape),
        "feature_norm": float(np.linalg.norm(feature)),
        "pair_provenance": pair_provenance,
        "sha256": {
            "checkpoint_metadata": sha256(metadata_path),
            "frame_cache": sha256(args.frame_cache),
            "feature": sha256(args.output),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
