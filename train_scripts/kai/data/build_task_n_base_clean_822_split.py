#!/usr/bin/env python3
"""Build the fixed 790/32 Task_N base_clean split used by the 16-H20 run.

The 32 validation trajectories are the exact local-v5 source identities used by
``nail_v5_343_joint14_val``.  ``base_clean`` is already joint-14, so this builder
only validates/reindexes parquet rows, creates relocatable video symlinks, and
computes train-only normalization statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot_build import CanonicalBuildSpec, build_canonical_split, read_jsonl  # noqa: E402


KAI0_ROOT = Path(os.environ.get("KAI0_ROOT", Path(__file__).resolve().parents[3] / "kai0"))
SOURCE = KAI0_ROOT / "data" / "Task_N" / "base_clean"
OUTPUT = KAI0_ROOT / "data" / "Task_N" / "self_built"
OLD_VAL = OUTPUT / "nail_v5_343_joint14_val" / "meta" / "split_manifest.json"
TRAIN = OUTPUT / "nail_base_clean_822_joint14_train"
VAL = OUTPUT / "nail_base_clean_822_joint14_val"
REPORT = KAI0_ROOT.parent / "docs" / "training" / "analysis" / "task_n_base_clean_822_preflight.json"

CAMERAS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")
EXPECTED = {
    "source_episodes": 822,
    "source_frames": 649_636,
    "source_videos": 2_466,
    "train_episodes": 790,
    "train_frames": 628_135,
    "train_videos": 2_370,
    "val_episodes": 32,
    "val_frames": 21_501,
    "val_videos": 96,
}

def source_identity(row: dict) -> str | None:
    if row.get("source_kind") != "local_v5":
        return None
    station = row.get("source_station") or "ipc01"
    return (
        f"{row['source_leaf']}/{station}/{row['source_chunk']}/"
        f"episode_{int(row['source_episode_id']):06d}"
    )


def validate_source(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(rows) != EXPECTED["source_episodes"]:
        raise ValueError(f"expected 822 source episodes, found {len(rows)}")
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("source episode_index is not contiguous 0..821")
    if sum(int(row["length"]) for row in rows) != EXPECTED["source_frames"]:
        raise ValueError("source frame total mismatch")

    old = json.loads(OLD_VAL.read_text())
    frozen_val = {item["source_identity"] for item in old["val"]}
    if len(frozen_val) != EXPECTED["val_episodes"]:
        raise ValueError(f"old frozen val must contain 32 unique identities, got {len(frozen_val)}")

    matched: dict[str, dict] = {}
    for row in rows:
        identity = source_identity(row)
        if identity in frozen_val:
            if identity in matched:
                raise ValueError(f"duplicate frozen val identity in base_clean: {identity}")
            matched[identity] = row
    missing = sorted(frozen_val - set(matched))
    if missing:
        raise ValueError(f"base_clean is missing frozen val identities: {missing}")

    val_indices = {int(row["episode_index"]) for row in matched.values()}
    train = [row for row in rows if int(row["episode_index"]) not in val_indices]
    val = [row for row in rows if int(row["episode_index"]) in val_indices]
    if len(train) != EXPECTED["train_episodes"] or len(val) != EXPECTED["val_episodes"]:
        raise ValueError(f"bad split train={len(train)} val={len(val)}")
    if sum(int(row["length"]) for row in train) != EXPECTED["train_frames"]:
        raise ValueError("train frame total mismatch")
    if sum(int(row["length"]) for row in val) != EXPECTED["val_frames"]:
        raise ValueError("val frame total mismatch")
    return train, val


def build_split(root: Path, rows: list[dict]):
    def metadata(source_row: dict, old_episode: int, _new_episode: int) -> dict:
        item = dict(source_row)
        item["source_base_clean_episode_index"] = old_episode
        item["source_identity"] = source_identity(source_row)
        return item

    return build_canonical_split(
        CanonicalBuildSpec(
            source_root=SOURCE,
            output_root=root,
            cameras=CAMERAS,
            action_width=14,
            video_mode="relative_symlink",
        ),
        rows,
        metadata_transform=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-norm", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(SOURCE / "meta" / "episodes.jsonl")
    train_rows, val_rows = validate_source(rows)
    source_video_count = len(list((SOURCE / "videos").rglob("*.mp4")))
    if source_video_count != EXPECTED["source_videos"]:
        raise ValueError(f"expected 2466 source videos, found {source_video_count}")
    print(f"SPLIT_OK train={len(train_rows)}/{sum(r['length'] for r in train_rows)} "
          f"val={len(val_rows)}/{sum(r['length'] for r in val_rows)}", flush=True)
    if args.dry_run:
        return

    temp_train = OUTPUT / f".{TRAIN.name}.building"
    temp_val = OUTPUT / f".{VAL.name}.building"
    for path in (temp_train, temp_val):
        if path.exists():
            shutil.rmtree(path)
    for target in (TRAIN, VAL):
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"output exists (use --overwrite): {target}")

    train_result = build_split(temp_train, train_rows)
    val_result = build_split(temp_val, val_rows)
    manifest = {
        "strategy": "fixed source identities from nail_v5_343_joint14_val",
        "train": list(train_result.metadata),
        "val": list(val_result.metadata),
        "train_frames": train_result.frames,
        "val_frames": val_result.frames,
    }
    for path in (temp_train, temp_val):
        (path / "meta" / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.no_norm:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from norm_stats_from_dataset import compute_norm_stats

        compute_norm_stats(str(temp_train), action_dim=32)

    for target in (TRAIN, VAL):
        if target.exists():
            shutil.rmtree(target)
    temp_train.rename(TRAIN)
    temp_val.rename(VAL)

    actual = {
        "source_episodes": len(rows),
        "source_frames": sum(int(row["length"]) for row in rows),
        "source_videos": source_video_count,
        "train_episodes": train_result.episodes,
        "train_frames": train_result.frames,
        "train_videos": len(list((TRAIN / "videos").rglob("*.mp4"))),
        "val_episodes": val_result.episodes,
        "val_frames": val_result.frames,
        "val_videos": len(list((VAL / "videos").rglob("*.mp4"))),
    }
    if actual != EXPECTED:
        raise ValueError(f"final output mismatch: actual={actual} expected={EXPECTED}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"checks": actual, "train_root": str(TRAIN), "val_root": str(VAL)}, indent=2) + "\n")
    print(f"BUILD_DONE {json.dumps(actual, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
