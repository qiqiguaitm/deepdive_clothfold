#!/usr/bin/env python3
"""Extract frozen DINOv3-base features from outcome-labeled RoboTwin videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO = Path(os.environ.get("RT_REPO", "/vePFS/tim/workspace/deepdive_kai0"))
CRAVE_SRC = REPO / "lmvla/crave/src"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_episodes(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        task = str(payload["task_name"])
        seed_dir = next((part for part in summary_path.parts if part.startswith("seed")), None)
        if seed_dir is None:
            raise ValueError(f"cannot recover simulator seed from {summary_path}")
        simulator_seed = int(seed_dir.removeprefix("seed"))
        for episode in payload["episodes"]:
            episode_id = int(episode["episode_id"])
            video = summary_path.parent / f"episode{episode_id}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
            rows.append(
                {
                    "task": task,
                    "simulator_seed": simulator_seed,
                    "episode_id": episode_id,
                    "scene_seed": int(episode["seed"]),
                    "success": bool(episode["success"]),
                    "steps": int(episode["steps"]),
                    "summary": str(summary_path),
                    "video": str(video),
                }
            )
    identity = {(row["task"], row["simulator_seed"], row["episode_id"]) for row in rows}
    if len(identity) != len(rows):
        raise ValueError("duplicate task/simulator-seed/episode identity")
    return rows


def decode_video(path: Path, stride: int) -> tuple[list[np.ndarray], np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0
    last_rgb: np.ndarray | None = None
    last_index = -1
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        last_rgb = rgb
        last_index = frame_index
        if frame_index % stride == 0:
            frames.append(rgb)
            indices.append(frame_index)
        frame_index += 1
    capture.release()
    if last_rgb is None:
        raise ValueError(f"video has no decodable frames: {path}")
    if not indices or indices[-1] != last_index:
        frames.append(last_rgb)
        indices.append(last_index)
    return frames, np.asarray(indices, dtype=np.int32), frame_index


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard < args.num_shards:
        raise ValueError("invalid shard")
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    all_rows = discover_episodes(args.rollout_root)
    rows = all_rows[args.shard :: args.num_shards]
    if not rows:
        raise ValueError(f"empty shard {args.shard}/{args.num_shards}")

    sys.path.insert(0, str(CRAVE_SRC))
    import torch
    from crave.encoders import load_encoder

    encoder = load_encoder("dinov3-base", dtype="bf16")
    records = []
    for row in rows:
        output = (
            args.output
            / f"seed{row['simulator_seed']}"
            / row["task"]
            / f"episode{row['episode_id']}.npz"
        )
        frames, frame_indices, decoded_frames = decode_video(Path(row["video"]), args.stride)
        pooled_parts = []
        with torch.inference_mode():
            for start in range(0, len(frames), args.batch_size):
                grid = encoder.encode_grid(frames[start : start + args.batch_size])
                if hasattr(grid, "detach"):
                    pooled = grid.detach().float().mean(dim=(2, 3)).cpu().numpy()
                else:
                    pooled = np.asarray(grid).mean(axis=(2, 3))
                pooled_parts.append(pooled)
        pooled = np.concatenate(pooled_parts).astype(np.float16)
        if pooled.shape != (len(frame_indices), 768) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid pooled features for {row['video']}: {pooled.shape}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, pooled=pooled, frame_index=frame_indices)
        temporary.replace(output)
        records.append(
            {
                **row,
                "decoded_frames": decoded_frames,
                "encoded_frames": len(frame_indices),
                "feature": str(output),
                "video_sha256": sha256(Path(row["video"])),
                "feature_sha256": sha256(output),
            }
        )
        print(
            f"FEATURE {row['task']} seed{row['simulator_seed']} ep{row['episode_id']} "
            f"frames={decoded_frames}->{len(frame_indices)}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "protocol": "pi05_crave_r0_rollout_features_v1",
        "encoder": "dinov3-base pooled patch grid",
        "future_observation_used": False,
        "stride": args.stride,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "all_episode_count": len(all_rows),
        "episode_count": len(records),
        "records": records,
    }
    atomic_json(args.output / f"shard{args.shard}.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
