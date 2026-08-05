#!/usr/bin/env python3
"""Audit frozen MT3 tracker rows and short-history media availability."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
HISTORY_OFFSETS = (-15, -7, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_frames(frame: int, offsets: tuple[int, ...] = HISTORY_OFFSETS) -> tuple[int, ...]:
    return tuple(max(0, frame + offset) for offset in offsets)


def episode_path(root: Path, category: str, episode: int, suffix: str, camera: str | None = None) -> Path:
    chunk = episode // 1000
    parts = [category, f"chunk-{chunk:03d}"]
    if camera is not None:
        parts.append(camera)
    return root.joinpath(*parts, f"episode_{episode:06d}.{suffix}")


def load_lengths(meta: Path) -> dict[int, int]:
    result = {}
    with meta.open() as stream:
        for line in stream:
            row = json.loads(line)
            result[int(row["episode_index"])] = int(row["length"])
    return result


def cached_frames(path: Path) -> set[int]:
    with zipfile.ZipFile(path) as archive:
        return {
            int(Path(name).stem)
            for name in archive.namelist()
            if name.endswith(".npy") and Path(name).stem.isdigit()
        }


def audit(pairs_path: Path, split_path: Path, data_root: Path, cache_root: Path) -> dict:
    pairs = np.load(pairs_path)
    split = json.loads(split_path.read_text())
    train = {int(value) for value in split["train_episodes"]}
    validation = {int(value) for value in split["val_episodes"]}
    selected_episodes = train | validation
    lengths = load_lengths(data_root / "meta/episodes.jsonl")

    episode = np.asarray(pairs["cur_ep"], dtype=np.int64)
    frame = np.asarray(pairs["cur_fi"], dtype=np.int64)
    stage = np.asarray(pairs["cur_ms"], dtype=np.int64)
    task = np.asarray(pairs["pair_task"], dtype=np.int64)
    selected = np.isin(episode, list(selected_episodes))
    if set(np.unique(episode[selected])) != selected_episodes:
        raise ValueError("one or more frozen split episodes have no transition rows")
    keys = list(zip(episode[selected].tolist(), frame[selected].tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate episode/frame transition rows")

    missing = []
    history_rows = 0
    for episode_id in sorted(selected_episodes):
        if episode_id not in lengths:
            missing.append(f"episode metadata:{episode_id}")
            continue
        parquet = episode_path(data_root, "data", episode_id, "parquet")
        if not parquet.is_file():
            missing.append(str(parquet))
        for camera in CAMERAS:
            video = episode_path(data_root, "videos", episode_id, "mp4", camera)
            if not video.is_file():
                missing.append(str(video))
        cache = episode_path(cache_root, "", episode_id, "npz", "observation.images.cam_high")
        if not cache.is_file():
            missing.append(str(cache))
            continue
        available = cached_frames(cache)
        episode_frames = frame[selected & (episode == episode_id)]
        if np.any(episode_frames < 0) or np.any(episode_frames >= lengths[episode_id]):
            raise ValueError(f"transition frame outside episode length: episode {episode_id}")
        required = {
            history
            for current in episode_frames.tolist()
            for history in history_frames(int(current))
        }
        absent = required.difference(available)
        if absent:
            missing.append(f"{cache}:missing_frames={len(absent)}")
        history_rows += len(episode_frames) * len(HISTORY_OFFSETS)
    if missing:
        raise FileNotFoundError(f"MT3 tracker media audit found {len(missing)} missing artifacts; first={missing[:5]}")

    train_rows = selected & np.isin(episode, list(train))
    val_rows = selected & np.isin(episode, list(validation))
    return {
        "version": "robotwin-mt3-data-audit-v1",
        "pairs": str(pairs_path),
        "pairs_sha256": sha256(pairs_path),
        "split": str(split_path),
        "split_sha256": sha256(split_path),
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "history_offsets_at_50hz": list(HISTORY_OFFSETS),
        "episodes": {
            "train": len(train),
            "validation": len(validation),
            "total": len(selected_episodes),
        },
        "rows": {
            "train": int(np.sum(train_rows)),
            "validation": int(np.sum(val_rows)),
            "total": int(np.sum(selected)),
            "history_frame_references": history_rows,
        },
        "task_row_counts": {
            str(key): value for key, value in sorted(Counter(task[selected].tolist()).items())
        },
        "stage_row_counts": {
            str(key): value for key, value in sorted(Counter(stage[selected].tolist()).items())
        },
        "checks": {
            "episode_frame_unique": True,
            "frame_within_episode_length": True,
            "parquet_complete": True,
            "three_view_video_complete": True,
            "base_history_cache_complete": True,
            "episode_split_leakage": bool(train & validation),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pairs, args.split, args.data_root, args.cache_root)
    if result["checks"]["episode_split_leakage"]:
        raise ValueError("episode leakage in frozen split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
