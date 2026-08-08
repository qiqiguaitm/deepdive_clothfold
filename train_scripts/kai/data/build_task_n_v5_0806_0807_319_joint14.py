#!/usr/bin/env python3
"""Build the metadata-frozen Task_N 08-06/08-07 319-episode dataset.

Only ipc01/chunk-000 episodes referenced by each date's authoritative
``meta/by_station/ipc01/episodes.jsonl`` are included.  This intentionally
excludes the two unreferenced parquet/video episode groups on disk.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import build_task_n_v5_272_joint14 as base


DATES = ("2026-08-06-v5", "2026-08-07-v5")
EXPECTED_VAL_IDS = {
    "2026-08-06-v5": {142, 143, 145, 147, 149, 150, 151, 152, 153},
    "2026-08-07-v5": {
        269, 270, 272, 273, 274, 275, 277, 278, 279, 280, 281, 282,
        283, 284, 285, 290, 291, 293, 294, 295, 296, 297, 299,
    },
}
EXPECTED_FRAMES_BY_DATE = {
    "2026-08-06-v5": 66_025,
    "2026-08-07-v5": 148_904,
}


def discover() -> list[base.SourceEpisode]:
    episodes: list[base.SourceEpisode] = []
    for date in DATES:
        date_root = base.SOURCE_ROOT / date
        manifest = date_root / "meta" / "by_station" / "ipc01" / "episodes.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing authoritative manifest: {manifest}")
        for meta in base._read_jsonl(manifest):
            chunk = str(meta.get("chunk", f"chunk-{int(meta['episode_chunk']):03d}"))
            if chunk != "chunk-000" or int(meta.get("episode_chunk", 0)) != 0:
                raise ValueError(f"non chunk-000 metadata entry: {date} {meta}")
            episode_id = int(meta["episode_id"])
            parquet = date_root / "data" / chunk / f"episode_{episode_id:06d}.parquet"
            if not parquet.is_file():
                raise FileNotFoundError(f"missing parquet: {parquet}")
            videos = tuple(base._video_path(date_root, chunk, camera, episode_id) for camera in base.CAMERAS)
            episodes.append(
                base.SourceEpisode(
                    date=date,
                    station="ipc01",
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
    counts = Counter(episode.date for episode in episodes)
    if counts != Counter({"2026-08-06-v5": 87, "2026-08-07-v5": 232}):
        raise ValueError(f"unexpected metadata counts: {dict(counts)}")
    return episodes


def allocate_val(
    episodes: list[base.SourceEpisode],
) -> tuple[list[base.SourceEpisode], list[base.SourceEpisode], dict[str, int]]:
    val = [episode for episode in episodes if episode.source_episode_id in EXPECTED_VAL_IDS[episode.date]]
    val_identities = {episode.identity for episode in val}
    train = [episode for episode in episodes if episode.identity not in val_identities]
    actual = {
        date: {episode.source_episode_id for episode in val if episode.date == date}
        for date in DATES
    }
    if actual != EXPECTED_VAL_IDS:
        raise ValueError(f"fixed validation identities mismatch: {actual}")
    return train, val, {"2026-08-06-v5::ipc01": 9, "2026-08-07-v5::ipc01": 23}


def main() -> None:
    base.TRAIN_ROOT = base.OUTPUT_ROOT / "nail_v5_0806_0807_319_joint14_train"
    base.VAL_ROOT = base.OUTPUT_ROOT / "nail_v5_0806_0807_319_joint14_val"
    base.REPORT_PATH = (
        base.REPO_ROOT / "docs" / "training" / "analysis" / "task_n_v5_0806_0807_319_preflight.json"
    )
    base.EXPECTED_EPISODES = 319
    base.EXPECTED_FRAMES = 214_929
    base.EXPECTED_TRAIN_EPISODES = 287
    base.VAL_EPISODES = 32
    base.discover = discover
    base.allocate_val = allocate_val
    base.main()

    report = json.loads(base.REPORT_PATH.read_text())
    frames_by_date = Counter()
    for item in report["episodes"]:
        frames_by_date[item["identity"].split("/", 1)[0]] += int(item["frames"])
    if dict(frames_by_date) != EXPECTED_FRAMES_BY_DATE:
        raise ValueError(f"unexpected frames by date: {dict(frames_by_date)}")


if __name__ == "__main__":
    main()
