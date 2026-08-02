#!/usr/bin/env python3
"""Encode audited same-task future frames in the online A2 So400m space."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_frame(path: pathlib.Path, frame: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        encoded = np.asarray(archive[str(frame)], dtype=np.uint8)
    image = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert("RGB"), dtype=np.uint8)
    if image.shape != (256, 256, 3):
        raise ValueError(f"unexpected cached frame shape {image.shape} in {path}")
    return image


def parse_spec(value: str) -> tuple[str, int, int, int]:
    task, pair_task, episode, frame = value.split(":")
    return task, int(pair_task), int(episode), int(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--pair-artifact", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--spec", action="append", required=True, type=parse_spec)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.pair_artifact, allow_pickle=False) as pairs:
        required = {"cur_ep", "cur_fi", "tgt_fi", "pair_task"}
        missing = required.difference(pairs.files)
        if missing:
            raise KeyError(f"pair artifact is missing arrays: {sorted(missing)}")
        for task, pair_task, episode, frame in args.spec:
            selected = (
                (pairs["pair_task"] == pair_task)
                & (pairs["cur_ep"] == episode)
                & (pairs["cur_fi"] == 0)
            )
            target_frames = np.unique(pairs["tgt_fi"][selected])
            if target_frames.shape != (1,) or int(target_frames[0]) != frame:
                raise ValueError(
                    f"invalid milestone target for {task}: "
                    f"expected={target_frames.tolist()} requested={frame}"
                )

    images = []
    frame_paths = []
    for _, _, episode, frame in args.spec:
        chunk = episode // 1000
        frame_path = (
            args.repo
            / "lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
            / f"chunk-{chunk:03d}/observation.images.cam_high/episode_{episode:06d}.npz"
        )
        frame_paths.append(frame_path)
        images.append(decode_frame(frame_path, frame))

    # Online A2 pads requests below four to reproduce the offline-cache bf16 kernel.
    padded = list(images)
    while len(padded) < 4:
        padded.append(padded[-1])
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(args.device).eval()
    pixel_values = processor(
        images=[Image.fromarray(image) for image in padded], return_tensors="pt"
    )["pixel_values"].to(args.device, torch.bfloat16)
    with torch.no_grad():
        tokens = model.vision_model(pixel_values=pixel_values).last_hidden_state[: len(images)]
    features = tokens.float().mean(dim=1).cpu().numpy().astype(np.float32)
    if features.shape != (len(args.spec), 1152) or not np.all(np.isfinite(features)):
        raise ValueError(f"invalid So400m features shape={features.shape}")

    pair_hash = sha256(args.pair_artifact)
    model_hash = sha256(args.model_dir / "model.safetensors")
    config_hash = sha256(args.model_dir / "config.json")
    for spec, frame_path, feature in zip(args.spec, frame_paths, features, strict=True):
        task, pair_task, episode, frame = spec
        output_dir = args.output_root / task
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "feature.npy"
        np.save(output, feature)
        manifest = {
            "control": "within-task-different-episode-milestone+1",
            "task": task,
            "encoder": "So400m/14@224",
            "feature_shape": list(feature.shape),
            "feature_norm": float(np.linalg.norm(feature)),
            "pair_provenance": {
                "artifact": str(args.pair_artifact),
                "pair_task": pair_task,
                "episode": episode,
                "cur_fi": 0,
                "tgt_fi": frame,
            },
            "frame_cache": str(frame_path),
            "sha256": {
                "pair_artifact": pair_hash,
                "frame_cache": sha256(frame_path),
                "model_safetensors": model_hash,
                "model_config": config_hash,
                "feature": sha256(output),
            },
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

