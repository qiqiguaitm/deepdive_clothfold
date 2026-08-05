#!/usr/bin/env python3
"""Build the frozen fixed-physical-horizon target index for pi0.5 MT5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pairs(episodes_path: Path, fps: int, horizon_seconds: float) -> dict[str, np.ndarray]:
    horizon_frames = round(fps * horizon_seconds)
    if horizon_frames <= 0 or not np.isclose(horizon_frames / fps, horizon_seconds):
        raise ValueError("horizon must map exactly to a positive integer frame offset")
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    cur_ep: list[np.ndarray] = []
    cur_fi: list[np.ndarray] = []
    tgt_fi: list[np.ndarray] = []
    for record in episodes:
        episode = int(record["episode_index"])
        length = int(record["length"])
        valid = max(length - horizon_frames, 0)
        frames = np.arange(valid, dtype=np.int32)
        cur_ep.append(np.full(valid, episode, dtype=np.int32))
        cur_fi.append(frames)
        tgt_fi.append(frames + horizon_frames)
    return {
        "cur_ep": np.concatenate(cur_ep),
        "cur_fi": np.concatenate(cur_fi),
        "tgt_fi": np.concatenate(tgt_fi),
        "horizon_frames": np.asarray(horizon_frames, dtype=np.int32),
        "fps": np.asarray(fps, dtype=np.int32),
        "horizon_seconds": np.asarray(horizon_seconds, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    info_path = args.dataset / "meta/info.json"
    episodes_path = args.dataset / "meta/episodes.jsonl"
    info = json.loads(info_path.read_text())
    arrays = build_pairs(episodes_path, int(info["fps"]), args.horizon_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(args.output)

    episode_count = sum(1 for line in episodes_path.read_text().splitlines() if line.strip())
    audit = {
        "protocol": "robotwin_pi05_mt5_fixed_physical_horizon_v1",
        "dataset": str(args.dataset),
        "dataset_info_sha256": sha256(info_path),
        "episodes_sha256": sha256(episodes_path),
        "episodes": episode_count,
        "dataset_frames": int(info["total_frames"]),
        "fps": int(arrays["fps"]),
        "horizon_seconds": float(arrays["horizon_seconds"]),
        "horizon_frames": int(arrays["horizon_frames"]),
        "valid_pairs": int(arrays["cur_ep"].size),
        "tail_policy": "retain policy sample; auxiliary target mask is false",
        "pairs": str(args.output),
        "pairs_sha256": sha256(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit_tmp = args.audit.with_suffix(args.audit.suffix + ".tmp")
    audit_tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    audit_tmp.replace(args.audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
