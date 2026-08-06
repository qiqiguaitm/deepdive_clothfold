#!/usr/bin/env python3
"""Smoke-test frozen pi0.5 visual features against MT stage labels."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import data_loader


CURRENT_VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
HISTORY_OFFSETS = (-15, -7, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unwrap_lerobot(dataset):
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset
    if not hasattr(current, "hf_dataset"):
        raise TypeError("could not locate underlying LeRobotDataset")
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = args.checkpoint / "params" / "_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    split = json.loads(args.split.read_text())
    validation = {int(value) for value in split["val_episodes"]}
    pairs = np.load(args.pairs)
    selected_rows = np.flatnonzero(
        np.isin(pairs["cur_ep"], list(validation)) & (pairs["cur_fi"] >= 15)
    )[: args.batch_size]
    if len(selected_rows) != args.batch_size:
        raise ValueError("not enough covered validation frames for the requested batch")
    sample_ids = [
        (int(pairs["cur_ep"][row]), int(pairs["cur_fi"][row]))
        for row in selected_rows
    ]
    episodes = sorted({episode for episode, _ in sample_ids})

    base = config_lib.get_config("pi05_robotwin_a0_public_exact_bj")
    base_data = base.data.base_config or config_lib.DataConfig(prompt_from_task=True)
    config = dataclasses.replace(
        base,
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
    items = [dataset[index[sample_id]] for sample_id in sample_ids]
    batch = data_loader._collate_fn(items)
    observation = model_lib.Observation.from_dict(batch)
    history_ids = [
        (episode, max(0, frame + offset))
        for episode, frame in sample_ids
        for offset in HISTORY_OFFSETS
    ]
    history_batch = data_loader._collate_fn([dataset[index[sample_id]] for sample_id in history_ids])
    history_observation = model_lib.Observation.from_dict(history_batch)

    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    encode = nnx_utils.module_jit(model.encode_base_image)
    current_tokens = {name: encode(observation.images[name]) for name in CURRENT_VIEWS}
    current_pooled = {
        name: np.asarray(jnp.mean(tokens.astype(jnp.float32), axis=1))
        for name, tokens in current_tokens.items()
    }
    history_tokens = encode(history_observation.images["base_0_rgb"])
    history_pooled = np.asarray(jnp.mean(history_tokens.astype(jnp.float32), axis=1)).reshape(
        args.batch_size, len(HISTORY_OFFSETS), -1
    )
    proprio = np.asarray(history_observation.state)[:, :14].reshape(
        args.batch_size, len(HISTORY_OFFSETS), 14
    )
    pooled_finite = all(np.all(np.isfinite(value)) for value in current_pooled.values())
    pooled_finite = bool(pooled_finite and np.all(np.isfinite(history_pooled)))
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": sha256(metadata),
        "pairs_sha256": sha256(args.pairs),
        "split_sha256": sha256(args.split),
        "encoder": "pi0.5 PaliGemma.img (SigLIP, frozen accepted A0 checkpoint)",
        "current_views": {
            name: {
                "image_tokens_shape": list(current_tokens[name].shape),
                "pooled_shape": list(current_pooled[name].shape),
                "pooled_l2": np.linalg.norm(current_pooled[name], axis=-1).tolist(),
            }
            for name in CURRENT_VIEWS
        },
        "history": {
            "image_key": "base_0_rgb",
            "frame_offsets_at_50hz": list(HISTORY_OFFSETS),
            "sample_ids": [[episode, frame] for episode, frame in history_ids],
            "image_tokens_shape": list(history_tokens.shape),
            "pooled_shape": list(history_pooled.shape),
            "proprio_shape": list(proprio.shape),
        },
        "pooled_finite": pooled_finite,
        "samples": [
            {
                "episode": episode,
                "frame": frame,
                "task": int(pairs["pair_task"][row]),
                "current_stage": int(pairs["cur_ms"][row]),
                "next_stage": int(pairs["cur_ms"][row] + 1),
            }
            for row, (episode, frame) in zip(selected_rows, sample_ids, strict=True)
        ],
    }
    if not result["pooled_finite"] or not np.all(np.isfinite(proprio)):
        raise FloatingPointError("non-finite pi0.5 visual features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
