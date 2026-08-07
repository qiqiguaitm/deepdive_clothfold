#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.feature_root.resolve()
    scene_manifest = args.scene_manifest.resolve()
    scenes = json.loads(scene_manifest.read_text(encoding="utf-8"))
    expected: set[tuple[str, int, int]] = set()
    for eval_seed, tasks in scenes["eval_seeds"].items():
        for task, seeds in tasks.items():
            expected.update((str(task), int(eval_seed), int(seed)) for seed in seeds)

    observed: set[tuple[str, int, int]] = set()
    file_count = 0
    total_bytes = 0
    shapes: set[tuple[int, ...]] = set()
    tree_digest = hashlib.sha256()
    per_scene_query_counts: list[int] = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for eval_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
            if not eval_dir.name.startswith("eval_seed_"):
                raise ValueError(f"Unexpected capture directory: {eval_dir}")
            eval_seed = int(eval_dir.name.removeprefix("eval_seed_"))
            for scene_dir in sorted(path for path in eval_dir.iterdir() if path.is_dir()):
                if not scene_dir.name.startswith("scene_seed_"):
                    raise ValueError(f"Unexpected capture directory: {scene_dir}")
                scene_seed = int(scene_dir.name.removeprefix("scene_seed_"))
                key = (task_dir.name, eval_seed, scene_seed)
                if key in observed:
                    raise ValueError(f"Duplicate captured scene: {key}")
                observed.add(key)
                files = sorted(scene_dir.glob("query_*.npy"))
                indices = [int(path.stem.removeprefix("query_")) for path in files]
                if indices != list(range(len(indices))) or not indices:
                    raise ValueError(f"Non-contiguous or empty query capture for {key}: {indices[:10]}")
                per_scene_query_counts.append(len(files))
                for path in files:
                    value = np.load(path, mmap_mode="r", allow_pickle=False)
                    if value.ndim < 2 or not np.isfinite(value).all():
                        raise ValueError(f"Invalid captured feature: {path}")
                    shapes.add(tuple(map(int, value.shape)))
                    relative = path.relative_to(root).as_posix()
                    digest = sha256(path)
                    tree_digest.update(relative.encode("utf-8"))
                    tree_digest.update(b"\0")
                    tree_digest.update(digest.encode("ascii"))
                    tree_digest.update(b"\n")
                    file_count += 1
                    total_bytes += path.stat().st_size

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"Capture scene mismatch: missing={missing[:5]} extra={extra[:5]}")
    if len(shapes) != 1:
        raise ValueError(f"Captured feature shapes differ: {sorted(shapes)}")

    result = {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg1a_normal_capture_v1",
        "complete": True,
        "feature_root": str(args.feature_root),
        "scene_manifest": str(args.scene_manifest),
        "scene_manifest_sha256": sha256(scene_manifest),
        "scenes": len(observed),
        "query_files": file_count,
        "total_bytes": total_bytes,
        "feature_shape": list(next(iter(shapes))),
        "queries_per_scene": {
            "min": min(per_scene_query_counts),
            "max": max(per_scene_query_counts),
            "mean": float(np.mean(per_scene_query_counts)),
        },
        "tree_sha256": tree_digest.hexdigest(),
        "checks": {
            "exact_scene_set": True,
            "nonempty_contiguous_queries": True,
            "single_feature_shape": True,
            "all_finite": True,
        },
    }
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
