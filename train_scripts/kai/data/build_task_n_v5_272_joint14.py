#!/usr/bin/env python3
"""Build the Task_N v5 cleaned 272-episode joint-14 train/val datasets.

The v5 source is a multi-station snapshot and is intentionally not directly
LeRobot-compatible: episode ids repeat across dates/chunks and one station lacks
the mid-head camera.  This builder creates globally reindexed datasets using
the three RGB cameras common to both stations and crops state/action from 32 to
the 14 joint+gripper dimensions executed by the Agilex policy.

Plan:
  docs/training/future_plans/plans/pi05_task_n_v5_272_base_sft_plan.md

Run on North-E after syncing ``kai0/data/Task_N/base/v5``::

  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 \
    .venv/bin/python ../train_scripts/kai/data/build_task_n_v5_272_joint14.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import _select_job, per_episode_stats  # noqa: E402


KAI0_ROOT = Path(os.environ.get("KAI0_ROOT", Path(__file__).resolve().parents[3] / "kai0"))
REPO_ROOT = KAI0_ROOT.parent
SOURCE_ROOT = KAI0_ROOT / "data" / "Task_N" / "base" / "v5"
OUTPUT_ROOT = KAI0_ROOT / "data" / "Task_N" / "self_built"
TRAIN_ROOT = OUTPUT_ROOT / "nail_v5_272_joint14_train"
VAL_ROOT = OUTPUT_ROOT / "nail_v5_272_joint14_val"
REPORT_PATH = REPO_ROOT / "docs" / "training" / "analysis" / "task_n_v5_272_preflight.json"

CAMERAS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
KEEP_COLS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
FPS = 30
VAL_EPISODES = 32
EXPECTED_EPISODES = 272
EXPECTED_FRAMES = 249_460
EXPECTED_TRAIN_EPISODES = 240


@dataclass(frozen=True)
class SourceEpisode:
    date: str
    station: str
    chunk: str
    source_episode_id: int
    created_at: float | None
    source_root: Path
    parquet: Path
    videos: tuple[Path, ...]
    source_meta: dict

    @property
    def identity(self) -> str:
        return f"{self.date}/{self.station}/{self.chunk}/episode_{self.source_episode_id:06d}"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _video_path(root: Path, chunk: str, camera: str, episode_id: int) -> Path:
    candidates = (
        root / "videos" / chunk / camera / f"episode_{episode_id:06d}.mp4",
        root / "videos" / chunk / camera.removeprefix("observation.images.") / f"episode_{episode_id:06d}.mp4",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing video for {root.name}/{chunk}/{camera}/episode_{episode_id:06d}")


def discover() -> list[SourceEpisode]:
    episodes: list[SourceEpisode] = []
    for date_root in sorted(SOURCE_ROOT.glob("*-v5")):
        manifest = date_root / "meta" / "multistation_episodes.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing multi-station manifest: {manifest}")
        for meta in _read_jsonl(manifest):
            chunk = str(meta["source_chunk"])
            episode_id = int(meta["source_episode_id"])
            station = str(meta["station_id"])
            parquet = date_root / "data" / chunk / f"episode_{episode_id:06d}.parquet"
            if not parquet.is_file():
                raise FileNotFoundError(f"missing parquet: {parquet}")
            videos = tuple(_video_path(date_root, chunk, camera, episode_id) for camera in CAMERAS)
            episodes.append(
                SourceEpisode(
                    date=date_root.name,
                    station=station,
                    chunk=chunk,
                    source_episode_id=episode_id,
                    created_at=float(meta["created_at"]) if meta.get("created_at") is not None else None,
                    source_root=date_root,
                    parquet=parquet,
                    videos=videos,
                    source_meta=meta,
                )
            )
    identities = [episode.identity for episode in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source identities discovered")
    return episodes


def allocate_val(episodes: list[SourceEpisode]) -> tuple[list[SourceEpisode], list[SourceEpisode], dict[str, int]]:
    """Tail holdout within every (date, station), with proportional allocation."""
    groups: dict[tuple[str, str], list[SourceEpisode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.date, episode.station)].append(episode)
    if VAL_EPISODES < len(groups):
        raise ValueError("val size is smaller than the number of strata")

    allocation = {key: 1 for key in groups}
    remaining = VAL_EPISODES - len(groups)
    total = sum(len(value) for value in groups.values())
    exact = {key: remaining * len(value) / total for key, value in groups.items()}
    for key, value in exact.items():
        allocation[key] += math.floor(value)
    left = VAL_EPISODES - sum(allocation.values())
    for key in sorted(groups, key=lambda item: (-(exact[item] - math.floor(exact[item])), item))[:left]:
        allocation[key] += 1

    train: list[SourceEpisode] = []
    val: list[SourceEpisode] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda episode: (
                episode.created_at if episode.created_at is not None else float(episode.source_episode_id),
                episode.source_episode_id,
            ),
        )
        n_val = allocation[key]
        if n_val >= len(ordered):
            raise ValueError(f"holdout consumes stratum {key}: {n_val}/{len(ordered)}")
        train.extend(ordered[:-n_val])
        val.extend(ordered[-n_val:])
    return train, val, {f"{key[0]}::{key[1]}": allocation[key] for key in sorted(allocation)}


def validate_episode(episode: SourceEpisode, seek_video: bool) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(episode.parquet)
    missing = [column for column in KEEP_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"{episode.identity}: missing columns {missing}")
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    if state.shape != (len(df), 32) or action.shape != (len(df), 32):
        raise ValueError(f"{episode.identity}: state/action shapes {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{episode.identity}: NaN/Inf in state/action")

    frame_index = df["frame_index"].to_numpy()
    timestamp = df["timestamp"].to_numpy(dtype=np.float64)
    if not np.array_equal(frame_index, np.arange(len(df))):
        raise ValueError(f"{episode.identity}: non-contiguous frame_index")
    dt = np.diff(timestamp)
    if len(dt) and (np.any(dt <= 0) or not 20 <= 1.0 / np.median(dt) <= 40):
        raise ValueError(f"{episode.identity}: invalid timestamps, median dt={np.median(dt)}")

    if seek_video:
        import av

        for video in episode.videos:
            container = av.open(str(video))
            stream = container.streams.video[0]
            if len(df) > 2:
                target_s = float(timestamp[len(df) // 2])
                container.seek(int(target_s / stream.time_base), stream=stream, any_frame=False, backward=True)
                try:
                    next(container.decode(video=0))
                except StopIteration:
                    # Some otherwise-valid H.264 files have an incomplete MP4
                    # seek index.  Reopen and decode through the midpoint so the
                    # gate still proves that the requested frame is readable.
                    container.close()
                    container = av.open(str(video))
                    decoded = False
                    for frame_number, _frame in enumerate(container.decode(video=0)):
                        if frame_number >= len(df) // 2:
                            decoded = True
                            break
                    if not decoded:
                        raise ValueError(f"{episode.identity}: cannot decode midpoint of {video}")
            container.close()

    return df, {
        "identity": episode.identity,
        "frames": len(df),
        "state_min": state[:, :14].min(axis=0).tolist(),
        "state_max": state[:, :14].max(axis=0).tolist(),
        "action_min": action[:, :14].min(axis=0).tolist(),
        "action_max": action[:, :14].max(axis=0).tolist(),
    }


def transform(df: pd.DataFrame, episode_index: int, global_start: int) -> pd.DataFrame:
    out = df[list(KEEP_COLS)].copy()
    out["observation.state"] = [np.asarray(value, dtype=np.float32)[:14] for value in out["observation.state"]]
    out["action"] = [np.asarray(value, dtype=np.float32)[:14] for value in out["action"]]
    out["timestamp"] = np.arange(len(out), dtype=np.float32) / FPS
    out["frame_index"] = np.arange(len(out), dtype=np.int64)
    out["episode_index"] = np.int64(episode_index)
    out["index"] = np.arange(global_start, global_start + len(out), dtype=np.int64)
    out["task_index"] = np.int64(0)
    return out


def build_split(
    root: Path,
    episodes: list[SourceEpisode],
    validated: dict[str, pd.DataFrame],
    *,
    symlink_videos: bool,
    nproc: int,
) -> tuple[int, list[dict]]:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    metadata: list[dict] = []
    stats: list[dict] = []
    video_jobs: list[tuple[str, str, np.ndarray, int]] = []
    total_frames = 0

    for new_episode_id, episode in enumerate(episodes):
        df = transform(validated[episode.identity], new_episode_id, total_frames)
        destination = root / "data" / "chunk-000" / f"episode_{new_episode_id:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), destination)
        for camera, source_video in zip(CAMERAS, episode.videos, strict=True):
            target = root / "videos" / "chunk-000" / camera / f"episode_{new_episode_id:06d}.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            if symlink_videos:
                os.symlink(str(source_video.resolve()), target)
            else:
                video_jobs.append((str(source_video.resolve()), str(target), np.arange(len(df)), len(df)))

        metadata.append(
            {
                "episode_index": new_episode_id,
                "tasks": ["nail painting"],
                "length": len(df),
                "source_date": episode.date,
                "station_id": episode.station,
                "source_chunk": episode.chunk,
                "source_episode_id": episode.source_episode_id,
                "source_identity": episode.identity,
            }
        )
        stats.append({"episode_index": new_episode_id, "stats": per_episode_stats(df)})
        total_frames += len(df)

    if video_jobs:
        print(f"re-encoding {len(video_jobs)} videos with nproc={nproc}", flush=True)
        with Pool(nproc) as pool:
            for index, _ in enumerate(pool.imap_unordered(_select_job, video_jobs, chunksize=2), 1):
                if index % 25 == 0 or index == len(video_jobs):
                    print(f"encoded {index}/{len(video_jobs)}", flush=True)

    source_info = json.loads((episodes[0].source_root / "meta" / "info.json").read_text())
    features = source_info["features"]
    for feature in list(features):
        if feature.startswith("observation.images.") and feature not in CAMERAS:
            features.pop(feature)
        if feature.startswith("observation.depth."):
            features.pop(feature)
    features["observation.state"]["shape"] = [14]
    features["action"]["shape"] = [14]
    info = source_info
    info.update(
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=1,
        total_videos=len(episodes) * len(CAMERAS),
        total_chunks=1,
        chunks_size=max(1000, len(episodes)),
        splits={"train": f"0:{len(episodes)}"},
        features=features,
    )
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    with (root / "meta" / "episodes.jsonl").open("w") as handle:
        for item in metadata:
            handle.write(json.dumps(item) + "\n")
    with (root / "meta" / "episodes_stats.jsonl").open("w") as handle:
        for item in stats:
            handle.write(json.dumps(item) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "nail painting"}) + "\n")
    return total_frames, metadata


def _replace_outputs(temp_train: Path, temp_val: Path, overwrite: bool) -> None:
    for target in (TRAIN_ROOT, VAL_ROOT):
        if target.exists() and not overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {target}")
    for target in (TRAIN_ROOT, VAL_ROOT):
        if target.exists():
            shutil.rmtree(target)
    temp_train.rename(TRAIN_ROOT)
    temp_val.rename(VAL_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-source-seek", action="store_true")
    parser.add_argument("--symlink-videos", action="store_true")
    parser.add_argument("--nproc", type=int, default=max(1, len(os.sched_getaffinity(0)) // 2))
    parser.add_argument("--no-norm", action="store_true")
    args = parser.parse_args()

    episodes = discover()
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} source episodes, discovered {len(episodes)}")
    train, val, allocation = allocate_val(episodes)
    if (len(train), len(val)) != (EXPECTED_TRAIN_EPISODES, VAL_EPISODES):
        raise ValueError(f"bad split: train={len(train)}, val={len(val)}")

    validated: dict[str, pd.DataFrame] = {}
    details: list[dict] = []
    for index, episode in enumerate(episodes, 1):
        df, detail = validate_episode(episode, seek_video=args.check_source_seek)
        validated[episode.identity] = df
        details.append(detail)
        if index % 25 == 0 or index == len(episodes):
            print(f"validated {index}/{len(episodes)}", flush=True)
    total_source_frames = sum(len(value) for value in validated.values())
    if total_source_frames != EXPECTED_FRAMES:
        raise ValueError(f"expected {EXPECTED_FRAMES} frames, found {total_source_frames}")

    report = {
        "source_root": str(SOURCE_ROOT),
        "source_episodes": len(episodes),
        "source_frames": total_source_frames,
        "required_rgb_videos": len(episodes) * len(CAMERAS),
        "train_episodes": len(train),
        "val_episodes": len(val),
        "val_allocation": allocation,
        "source_video_seek_checked": args.check_source_seek,
        "output_video_mode": "symlink" if args.symlink_videos else "libx264 re-encode",
        "episodes": details,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))
    if args.dry_run:
        print("DRY_RUN_OK")
        return

    temp_train = OUTPUT_ROOT / f".{TRAIN_ROOT.name}.building"
    temp_val = OUTPUT_ROOT / f".{VAL_ROOT.name}.building"
    for path in (temp_train, temp_val):
        if path.exists():
            shutil.rmtree(path)
    train_frames, train_meta = build_split(
        temp_train, train, validated, symlink_videos=args.symlink_videos, nproc=args.nproc
    )
    val_frames, val_meta = build_split(
        temp_val, val, validated, symlink_videos=args.symlink_videos, nproc=args.nproc
    )
    split_manifest = {
        "strategy": "tail holdout within (date, station)",
        "val_allocation": allocation,
        "train": train_meta,
        "val": val_meta,
        "train_frames": train_frames,
        "val_frames": val_frames,
    }
    for path in (temp_train, temp_val):
        (path / "meta" / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")

    if not args.no_norm:
        from norm_stats_from_dataset import compute_norm_stats

        compute_norm_stats(str(temp_train), action_dim=32)
    _replace_outputs(temp_train, temp_val, args.overwrite)
    print(f"BUILD_DONE train={TRAIN_ROOT} ({len(train)} ep/{train_frames} frames)")
    print(f"BUILD_DONE val={VAL_ROOT} ({len(val)} ep/{val_frames} frames)")


if __name__ == "__main__":
    main()
