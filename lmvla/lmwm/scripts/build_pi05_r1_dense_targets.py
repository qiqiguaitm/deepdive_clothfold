#!/usr/bin/env python3
"""Build dense fixed-horizon R1 targets from frozen CRAVE trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REQUIRED = {
    "episode",
    "frame",
    "physical_task",
    "recurrence_density",
    "progress",
    "phase_boundary",
    "phase",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(reference_path: Path, output: Path, manifest: Path, horizon: int) -> dict:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    source = np.load(reference_path)
    missing = REQUIRED.difference(source.files)
    if missing:
        raise ValueError(f"reference trajectories missing keys: {sorted(missing)}")
    lengths = {len(source[key]) for key in REQUIRED}
    if len(lengths) != 1:
        raise ValueError("reference trajectory arrays are not aligned")
    if not np.isfinite(source["progress"]).all() or not np.isfinite(
        source["recurrence_density"]
    ).all():
        raise ValueError("reference trajectory fields must be finite")

    episodes = source["episode"].astype(np.int64)
    frames = source["frame"].astype(np.int64)
    tasks = source["physical_task"].astype(np.int64)
    keys = list(zip(episodes.tolist(), frames.tolist(), strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError("reference trajectories contain duplicate episode/frame rows")

    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "cur_ep",
            "cur_fi",
            "tgt_fi",
            "physical_task",
            "progress_change",
            "target_recurrence_density",
            "phase_boundary_crossing",
        )
    }
    episode_lengths: list[int] = []
    order = np.lexsort((frames, episodes))
    sorted_episodes = episodes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_episodes)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for lower, upper in zip(starts, ends, strict=True):
        rows = order[lower:upper]
        episode = int(episodes[rows[0]])
        episode_frames = frames[rows]
        if not np.array_equal(episode_frames, np.arange(len(rows))):
            raise ValueError(f"episode {episode} frames are not contiguous from zero")
        if len(np.unique(tasks[rows])) != 1:
            raise ValueError(f"episode {episode} crosses physical tasks")
        episode_lengths.append(len(rows))
        valid = len(rows) - horizon
        if valid <= 0:
            continue
        current_rows = rows[:valid]
        target_rows = rows[horizon:]
        current_phase = source["phase"][current_rows].astype(np.int64)
        target_phase = source["phase"][target_rows].astype(np.int64)
        parts["cur_ep"].append(np.full(valid, episode, dtype=np.int32))
        parts["cur_fi"].append(np.arange(valid, dtype=np.int32))
        parts["tgt_fi"].append(np.arange(horizon, len(rows), dtype=np.int32))
        parts["physical_task"].append(
            np.full(valid, tasks[rows[0]], dtype=np.int16)
        )
        parts["progress_change"].append(
            (
                source["progress"][target_rows]
                - source["progress"][current_rows]
            ).astype(np.float32)
        )
        parts["target_recurrence_density"].append(
            source["recurrence_density"][target_rows].astype(np.float32)
        )
        parts["phase_boundary_crossing"].append(target_phase > current_phase)

    if not parts["cur_ep"]:
        raise ValueError("no episode is longer than the fixed horizon")
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    if np.any(arrays["progress_change"] < -1.0) or np.any(
        arrays["progress_change"] > 1.0
    ):
        raise ValueError("dense progress-change targets leave [-1, 1]")
    if np.any(arrays["target_recurrence_density"] < 0.0) or np.any(
        arrays["target_recurrence_density"] > 1.0
    ):
        raise ValueError("dense recurrence-density targets leave [0, 1]")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    result = {
        "schema_version": 1,
        "protocol": "pi05_r1_dense_crave_targets_v1",
        "reference_trajectories": str(reference_path.resolve()),
        "reference_trajectories_sha256": sha256(reference_path),
        "dense_targets": str(output.resolve()),
        "dense_targets_sha256": sha256(output),
        "horizon_frames": horizon,
        "episode_count": len(episode_lengths),
        "source_rows": len(episodes),
        "target_rows": len(arrays["cur_ep"]),
        "eligible_row_fraction": len(arrays["cur_ep"]) / len(episodes),
        "physical_task_count": len(np.unique(arrays["physical_task"])),
        "boundary_positive_rate": float(
            np.mean(arrays["phase_boundary_crossing"])
        ),
    }
    atomic_json(manifest, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()
    result = build(args.reference, args.output, args.manifest, args.horizon)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
