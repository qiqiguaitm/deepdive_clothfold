"""Normalize one or more KAI0/AgileX leaves into a contiguous LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from .lerobot import (
    DEFAULT_CAMERAS,
    DatasetLayout,
    VideoMode,
    discover_episodes,
    iter_episode_artifacts,
    latest_good_file,
    parse_episode_file,
    place_file,
    read_jsonl,
    rewrite_episode_table,
    strip_depth_features,
    write_jsonl,
)

CHUNK_SIZE = 1000


def _meta_by_episode(root: Path) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for row in read_jsonl(root / "meta" / "episodes.jsonl"):
        raw = row.get("episode_index", row.get("episode_id"))
        if raw is not None:
            result[int(raw)] = row
    return result


def _stats(values) -> dict:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(len(array))],
    }


def _episode_stats(table, features: dict) -> dict:
    stats: dict = {}
    for key, spec in features.items():
        if spec.get("dtype") in {"image", "video"}:
            stats[key] = {"min": [0.0], "max": [1.0], "mean": [0.5], "std": [0.5], "count": [len(table)]}
        elif key in table.column_names:
            stats[key] = _stats(table[key].to_pylist())
    return stats


def normalize(
    sources: Iterable[Path],
    destination: Path,
    *,
    task: str,
    fps: int = 30,
    cameras: Iterable[str] = DEFAULT_CAMERAS,
    video_mode: VideoMode = "hardlink",
    use_latest_good: bool = False,
) -> dict:
    """Merge/reindex sources without modifying them; destination must not exist."""
    import pyarrow.parquet as pq

    roots = [path.resolve() for path in sources]
    cameras = tuple(cameras)
    if not roots:
        raise ValueError("at least one source is required")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    for root in roots:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(root / "meta" / "info.json")

    first_info = json.loads((roots[0] / "meta" / "info.json").read_text(encoding="utf-8"))
    features = strip_depth_features(first_info.get("features", {}))
    selected: list[tuple[Path, int, dict]] = []
    for root in roots:
        ids = discover_episodes(root)
        if use_latest_good:
            good = latest_good_file(root)
            if good is None:
                raise FileNotFoundError(f"no *.good_episodes.txt below {root / 'meta/quality'}")
            ids = parse_episode_file(good)
        metadata = _meta_by_episode(root)
        selected.extend((root, ep, metadata.get(ep, {})) for ep in ids)
    if not selected:
        raise RuntimeError("no episodes selected")

    destination.mkdir(parents=True)
    output = DatasetLayout(destination)
    episode_rows: list[dict] = []
    stats_rows: list[dict] = []
    manifest_rows: list[dict] = []
    global_offset = 0
    for new_ep, (root, old_ep, old_meta) in enumerate(selected):
        artifacts = list(iter_episode_artifacts(root, [old_ep], cameras))
        _, source_parquet, source_videos = artifacts[0]
        table = rewrite_episode_table(pq.read_table(source_parquet), new_ep, global_offset, fps)
        chunk = new_ep // CHUNK_SIZE
        target_parquet = output.parquet(new_ep, chunk)
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target_parquet, compression="zstd")
        for camera, source_video in source_videos.items():
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            place_file(source_video, output.video(camera, new_ep, chunk), video_mode)
        length = len(table)
        episode_rows.append({
            "episode_index": new_ep, "tasks": [task], "length": length,
            "duration_s": round(length / float(fps), 6),
            "operator": old_meta.get("operator"), "success": old_meta.get("success", True),
            "source_root": str(root), "source_episode_id": old_ep,
        })
        stats_rows.append({"episode_index": new_ep, "stats": _episode_stats(table, features)})
        manifest_rows.append({"episode_index": new_ep, "source_root": str(root), "source_episode_id": old_ep})
        global_offset += length

    info = dict(first_info)
    info.update({
        "codebase_version": "v2.1", "fps": fps, "chunks_size": CHUNK_SIZE,
        "total_episodes": len(episode_rows), "total_frames": global_offset,
        "total_tasks": 1, "total_videos": len(episode_rows) * len(cameras),
        "total_chunks": (len(episode_rows) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    })
    info.pop("depth_path", None)
    output.meta.mkdir(parents=True, exist_ok=True)
    (output.meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(output.meta / "tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(output.meta / "episodes.jsonl", episode_rows)
    write_jsonl(output.meta / "episodes_stats.jsonl", stats_rows)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task,
        "fps": fps, "video_mode": video_mode, "sources": [str(root) for root in roots],
        "episodes": manifest_rows,
    }
    (destination / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"episodes": len(episode_rows), "frames": global_offset, "destination": str(destination)}
