"""Export frozen current-token and predictive-adapter features for CRAVE R0."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models import pi0 as pi0_lib
from openpi.shared import nnx_utils
from openpi.training import data_loader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pi05_robotwin_confirmatory as confirmatory  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unwrap_lerobot(dataset):
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset
    if not hasattr(current, "hf_dataset"):
        raise TypeError("could not locate the underlying LeRobotDataset")
    return current


def build_config(args: argparse.Namespace, split_name: str):
    train_args = argparse.Namespace(
        arm="p0_predictive",
        config_name="pi05_predictive_adapter_p0",
        exp_name="crave_r0_feature_export",
        seed=1000,
        data_repo=str(args.data_repo),
        init_params=str(args.checkpoint / "params"),
        asset_id="robotwin2.0_absolute_meanstd",
        norm_assets_dir=str(args.norm_assets_dir),
        hint_path=None,
        target_pairs=str(args.pairs),
        frame_cache_root=str(args.frame_cache_root),
        transition_pairs=None,
        tracker_checkpoint=None,
        tracker_candidate=None,
        episodes_json=str(args.episode_split),
        episode_split=split_name,
        num_train_steps=1,
        save_interval=1,
        skip_final_checkpoint=False,
        log_interval=1,
        num_workers=0,
        fsdp_devices=1,
        dry_run=False,
        resume=False,
    )
    return dataclasses.replace(confirmatory.build_config(train_args), batch_size=args.batch_size)


def make_dataset(config, episodes: list[int]):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model, episodes=episodes
    )
    dataset = data_loader.transform_dataset(dataset, data_config)
    base = unwrap_lerobot(dataset)
    episode_ids = np.asarray(base.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(base.hf_dataset["frame_index"], dtype=np.int64)
    lookup = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episode_ids, frame_ids, strict=True))
    }
    return dataset, lookup


def summarize_grid(grid: jax.Array) -> jax.Array:
    grid = grid.astype(jnp.float32)
    mean = jnp.mean(grid, axis=1)
    std = jnp.sqrt(jnp.mean(jnp.square(grid - mean[:, None]), axis=1) + 1e-8)
    return jnp.concatenate([mean, std], axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--episode-split", type=Path, required=True)
    parser.add_argument("--frame-cache-root", type=Path, required=True)
    parser.add_argument("--norm-assets-dir", type=Path, required=True)
    parser.add_argument("--probe-train", type=Path, required=True)
    parser.add_argument("--probe-eval", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    metadata = args.checkpoint / "params" / "_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    norm_stats = (
        args.norm_assets_dir
        / "robotwin2.0_absolute_meanstd"
        / "norm_stats.json"
    )
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)

    split = json.loads(args.episode_split.read_text())
    sources = {
        "train": (args.probe_train, "train", set(map(int, split["train_episodes"]))),
        "eval": (args.probe_eval, "heldout", set(map(int, split["heldout_episodes"]))),
    }
    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = build_config(args, "heldout").model.load(params)
    model.eval()
    encode = nnx_utils.module_jit(model.encode_base_image)
    predict = nnx_utils.module_jit(
        model.predictive_action_adapter.__call__, static_argnames=("intervention",)
    )
    summarize = jax.jit(summarize_grid)
    root_key = jax.random.key(20260804)
    output_arrays = {}
    split_reports = {}

    for split_offset, (name, (source_path, split_name, allowed_episodes)) in enumerate(sources.items()):
        with np.load(source_path) as source_file:
            source = {key: np.asarray(source_file[key]) for key in source_file.files}
        episodes = sorted(set(map(int, source["cur_ep"])))
        if set(episodes) - allowed_episodes:
            raise ValueError(f"{name}: probe rows cross the frozen episode split")
        total = len(source["cur_ep"])
        start = total * args.shard_index // args.shard_count
        stop = total * (args.shard_index + 1) // args.shard_count
        rows = np.arange(start, stop, dtype=np.int64)
        config = build_config(args, split_name)
        dataset, lookup = make_dataset(config, episodes)
        ordered_indices = [
            lookup[(int(source["cur_ep"][row]), int(source["cur_fi"][row]))]
            for row in rows
        ]
        features = {key: [] for key in ("current", "normal", "shuffled", "masked")}
        for batch_start in range(0, len(rows), args.batch_size):
            batch_indices = ordered_indices[batch_start : batch_start + args.batch_size]
            items = [dataset[index] for index in batch_indices]
            batch = data_loader._collate_fn(items)
            observation = model_lib.Observation.from_dict(batch)
            observation = model_lib.preprocess_observation(
                jax.random.fold_in(root_key, split_offset * 1_000_000 + batch_start),
                observation,
                train=False,
                augment_level=model.augment_level,
            )
            current_tokens = jax.lax.stop_gradient(
                encode(observation.images["base_0_rgb"])
            )
            current_grid = pi0_lib._spatial_pool_tokens(
                current_tokens.astype(jnp.float32), model.predictive_action_adapter.grid_size
            )
            features["current"].append(
                np.asarray(summarize(current_grid), dtype=np.float16)
            )
            actions = jnp.asarray(batch["actions"])
            for intervention in ("normal", "shuffled", "masked"):
                prediction, _ = predict(
                    current_tokens, actions, intervention=intervention
                )
                features[intervention].append(
                    np.asarray(summarize(prediction), dtype=np.float16)
                )
            print(f"{name} {min(batch_start + len(batch_indices), len(rows))}/{len(rows)}", flush=True)
        output_arrays[f"{name}_row"] = rows
        for key, parts in features.items():
            output_arrays[f"{name}_{key}"] = np.concatenate(parts) if parts else np.empty((0, 0), np.float16)
        split_reports[name] = {
            "source": str(source_path.resolve()),
            "source_sha256": sha256(source_path),
            "full_rows": total,
            "selected_episodes": len(episodes),
            "shard_start": start,
            "shard_stop": stop,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.npz")
    np.savez_compressed(temporary, **output_arrays)
    temporary.replace(args.output)
    report = {
        "schema_version": 1,
        "protocol": "pi05_crave_r0_frozen_probe_features_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata_sha256": sha256(metadata),
        "norm_stats_sha256": sha256(norm_stats),
        "extractor_sha256": sha256(Path(__file__).resolve()),
        "model_source_sha256": sha256(Path(pi0_lib.__file__).resolve()),
        "feature": "mean and standard deviation over the 4x4 current/predicted SigLIP token grid",
        "future_image_used": False,
        "interventions": ["normal", "shuffled", "masked", "current"],
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "splits": split_reports,
        "output_sha256": sha256(args.output),
    }
    report_path = args.output.with_suffix(".json")
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
