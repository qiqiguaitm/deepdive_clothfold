#!/usr/bin/env python3
"""Extract frozen pi0.5 features for the two preregistered MT3 trackers."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np


CURRENT_VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
HISTORY_OFFSETS = (-15, -7, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(
    episode: np.ndarray,
    split: dict,
    *,
    shard_index: int,
    num_shards: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard index/count")
    train = {int(value) for value in split["train_episodes"]}
    validation = {int(value) for value in split["val_episodes"]}
    if train & validation:
        raise ValueError("frozen tracker split contains episode leakage")
    all_rows = np.flatnonzero(np.isin(episode, list(train | validation)))
    selected_episode = episode[all_rows].astype(np.int64)
    episode_ids, counts = np.unique(selected_episode, return_counts=True)
    loads = [0] * num_shards
    assignment: dict[int, int] = {}
    for episode_id, count in sorted(
        zip(episode_ids.tolist(), counts.tolist(), strict=True),
        key=lambda value: (-value[1], value[0]),
    ):
        destination = min(range(num_shards), key=lambda value: (loads[value], value))
        assignment[int(episode_id)] = destination
        loads[destination] += int(count)
    rows = all_rows[
        np.fromiter(
            (assignment[int(value)] == shard_index for value in selected_episode),
            dtype=bool,
            count=len(selected_episode),
        )
    ]
    split_id = np.fromiter(
        (0 if int(episode[row]) in train else 1 for row in rows),
        dtype=np.uint8,
        count=len(rows),
    )
    return rows, split_id


def history_sample_ids(sample_ids: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (episode, max(0, frame + offset))
        for episode, frame in sample_ids
        for offset in HISTORY_OFFSETS
    ]


def unpack_pooled(pooled: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    expected = batch_size * (len(CURRENT_VIEWS) + len(HISTORY_OFFSETS))
    if pooled.ndim != 2 or pooled.shape[0] != expected:
        raise ValueError(f"unexpected pooled feature shape: {pooled.shape}, expected rows={expected}")
    current = np.stack(
        [pooled[index * batch_size : (index + 1) * batch_size] for index in range(3)],
        axis=1,
    )
    history = pooled[3 * batch_size :].reshape(batch_size, len(HISTORY_OFFSETS), -1)
    return current, history


def unwrap_lerobot(dataset):
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset
    if not hasattr(current, "hf_dataset"):
        raise TypeError("could not locate underlying LeRobotDataset")
    return current


class HistoryTensorCache:
    def __init__(self, dataset, index: dict[tuple[int, int], int], capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("history cache capacity must be positive")
        self.dataset = dataset
        self.index = index
        self.capacity = capacity
        self.values: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def tensors(item) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(item["image"]["base_0_rgb"])
        state = np.asarray(item["state"], dtype=np.float32)[:14]
        return image, state

    def put(self, sample_id: tuple[int, int], item) -> None:
        self.values[sample_id] = self.tensors(item)
        self.values.move_to_end(sample_id)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)

    def get(self, sample_id: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if sample_id in self.values:
            self.hits += 1
        else:
            self.misses += 1
            self.put(sample_id, self.dataset[self.index[sample_id]])
        self.values.move_to_end(sample_id)
        return self.values[sample_id]


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rows-per-file", type=int, default=4096)
    parser.add_argument("--history-cache-frames", type=int, default=4096)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.rows_per_file <= 0:
        raise ValueError("batch size and rows per file must be positive")
    repo = args.repo.resolve()
    metadata = args.checkpoint / "params" / "_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    split = json.loads(args.split.read_text())
    pairs = np.load(args.pairs)
    rows, split_id = select_rows(
        np.asarray(pairs["cur_ep"]),
        split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
        split_id = split_id[: args.max_rows]
    if not len(rows):
        raise ValueError("feature shard has no rows")

    output = args.output.resolve() / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    provenance = {
        "checkpoint_metadata_sha256": sha256(metadata),
        "pairs_sha256": sha256(args.pairs),
        "split_sha256": sha256(args.split),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("provenance") != provenance or existing.get("rows") != len(rows):
            raise ValueError("existing feature shard manifest has different provenance or row count")
        if all((output / item["file"]).is_file() for item in existing["chunks"]):
            print(f"complete={manifest_path}")
            return

    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib
    from openpi.training import data_loader

    episodes = sorted({int(value) for value in pairs["cur_ep"][rows]})
    base = config_lib.get_config("pi05_robotwin_a0_public_exact_bj")
    base_data = base.data.base_config or config_lib.DataConfig(prompt_from_task=True)
    config = dataclasses.replace(
        base,
        assets_base_dir=str(repo / "kai0/assets"),
        data=dataclasses.replace(
            base.data,
            repo_id=str(args.data_repo.resolve()),
            base_config=dataclasses.replace(base_data, episodes=episodes),
        ),
        batch_size=args.batch_size,
        num_workers=0,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        episodes=episodes,
    )
    dataset = data_loader.transform_dataset(raw_dataset, data_config)
    base_dataset = unwrap_lerobot(dataset)
    episode_ids = np.asarray(base_dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(base_dataset.hf_dataset["frame_index"], dtype=np.int64)
    index = {
        (int(episode), int(frame)): position
        for position, (episode, frame) in enumerate(zip(episode_ids, frame_ids, strict=True))
    }
    history_cache = HistoryTensorCache(
        dataset, index, capacity=args.history_cache_frames
    )

    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    encode = nnx_utils.module_jit(model.encode_base_image)
    chunks = []
    chunk_arrays: dict[str, list[np.ndarray]] = {}
    chunk_rows = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal chunk_arrays, chunk_rows, chunk_index
        if not chunk_rows:
            return
        arrays = {name: np.concatenate(values, axis=0) for name, values in chunk_arrays.items()}
        filename = f"features-{chunk_index:05d}.npz"
        path = output / filename
        atomic_npz(path, **arrays)
        chunks.append({"file": filename, "rows": chunk_rows, "sha256": sha256(path)})
        chunk_arrays = {}
        chunk_rows = 0
        chunk_index += 1

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_split = split_id[start : start + args.batch_size]
        sample_ids = [
            (int(pairs["cur_ep"][row]), int(pairs["cur_fi"][row])) for row in batch_rows
        ]
        current_items = [dataset[index[value]] for value in sample_ids]
        for sample_id, item in zip(sample_ids, current_items, strict=True):
            history_cache.put(sample_id, item)
        current_batch = data_loader._collate_fn(current_items)
        history_ids = history_sample_ids(sample_ids)
        history_tensors = [history_cache.get(value) for value in history_ids]
        history_images = np.stack([value[0] for value in history_tensors])
        history_states = np.stack([value[1] for value in history_tensors])
        observation = model_lib.Observation.from_dict(current_batch)
        if history_images.dtype == np.uint8:
            history_images = history_images.astype(np.float32) / 255.0 * 2.0 - 1.0
        images = jnp.concatenate(
            [observation.images[name] for name in CURRENT_VIEWS]
            + [jnp.asarray(history_images)],
            axis=0,
        )
        tokens = encode(images)
        pooled = np.asarray(jnp.mean(tokens.astype(jnp.float32), axis=1), dtype=np.float16)
        current_features, history_features = unpack_pooled(pooled, len(batch_rows))
        proprio = history_states.reshape(
            len(batch_rows), len(HISTORY_OFFSETS), 14
        )
        values = {
            "episode": np.asarray(pairs["cur_ep"][batch_rows], dtype=np.int32),
            "frame": np.asarray(pairs["cur_fi"][batch_rows], dtype=np.int32),
            "task": np.asarray(pairs["pair_task"][batch_rows], dtype=np.int16),
            "current_target": np.asarray(pairs["cur_ms"][batch_rows], dtype=np.int16),
            "next_target": np.asarray(pairs["cur_ms"][batch_rows] + 1, dtype=np.int16),
            "split": batch_split,
            "current_view_features": current_features,
            "history_base_features": history_features,
            "history_proprio": proprio,
        }
        if not all(np.all(np.isfinite(value)) for value in values.values()):
            raise FloatingPointError(f"non-finite feature batch at row {start}")
        for name, value in values.items():
            chunk_arrays.setdefault(name, []).append(value)
        chunk_rows += len(batch_rows)
        if chunk_rows >= args.rows_per_file:
            flush()
        if start % (args.batch_size * 25) == 0:
            print(f"shard={args.shard_index}/{args.num_shards} rows={start + len(batch_rows)}/{len(rows)}", flush=True)
    flush()
    manifest = {
        "version": "pi05-mt3-frozen-features-v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "episodes": len({int(value) for value in pairs["cur_ep"][rows]}),
        "current_views": list(CURRENT_VIEWS),
        "history_offsets_at_50hz": list(HISTORY_OFFSETS),
        "feature_width": 2048,
        "proprio_width": 14,
        "history_cache": {
            "capacity_frames": args.history_cache_frames,
            "hits": history_cache.hits,
            "misses": history_cache.misses,
        },
        "provenance": provenance,
        "chunks": chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"complete={manifest_path}")


if __name__ == "__main__":
    main()
