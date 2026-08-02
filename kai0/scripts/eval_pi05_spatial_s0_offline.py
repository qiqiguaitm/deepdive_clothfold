"""Evaluate matched spatial S0 arms on an exact frozen held-out frame manifest."""

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
from openpi.shared import nnx_utils
from openpi.training import data_loader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pi05_robotwin_confirmatory as confirmatory  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_config(args: argparse.Namespace, manifest: dict) -> object:
    train_args = argparse.Namespace(
        arm=args.arm,
        config_name=f"pi05_spatial_{args.arm}",
        exp_name="offline_eval",
        seed=1000,
        data_repo=args.data_repo,
        init_params=str(args.checkpoint / "params"),
        asset_id="robotwin2.0_absolute_meanstd",
        hint_path=None,
        target_pairs=args.target_pairs if args.arm == "s0_privileged" else None,
        frame_cache_root=args.frame_cache_root if args.arm == "s0_privileged" else None,
        episodes_json=str(args.split),
        episode_split="heldout",
        num_train_steps=1,
        save_interval=1,
        log_interval=1,
        num_workers=0,
        dry_run=False,
        resume=False,
    )
    config = confirmatory.build_config(train_args)
    return dataclasses.replace(config, batch_size=int(manifest["batch_size"]), num_workers=0)


def transformed_dataset(config: object, episodes: list[int]):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        episodes=episodes,
    )
    return data_loader.transform_dataset(dataset, data_config), data_config


def unwrap_lerobot(dataset):
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset
    if not hasattr(current, "hf_dataset"):
        raise TypeError("could not locate the underlying LeRobotDataset")
    return current


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "flow_loss",
        "action_cosine",
        "endpoint_l2",
        "action_sensitivity_l1",
        "gate_mean",
        "token_norm_mean",
        "target_available",
    )
    return {
        key: None if not rows else float(np.mean([float(row[key]) for row in rows]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("s0_no_goal", "s0_current", "s0_privileged"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--target-pairs", type=Path, required=True)
    parser.add_argument("--frame-cache-root", type=Path, required=True)
    parser.add_argument("--num-diffusion-steps", type=int, default=10)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--smoke-one-batch-per-task", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.checkpoint = args.checkpoint.resolve()
    args.manifest = args.manifest.resolve()
    args.split = args.split.resolve()
    args.data_repo = str(args.data_repo.resolve())
    args.target_pairs = str(args.target_pairs.resolve())
    args.frame_cache_root = str(args.frame_cache_root.resolve())
    metadata = args.checkpoint / "params" / "_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    manifest = json.loads(args.manifest.read_text())
    split = json.loads(args.split.read_text())
    if manifest["split_sha256"] != sha256(args.split):
        raise ValueError("evaluation manifest does not match the frozen episode split")

    config = build_config(args, manifest)
    dataset, data_config = transformed_dataset(config, split["heldout_episodes"])
    base_dataset = unwrap_lerobot(dataset)
    episode_ids = np.asarray(base_dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(base_dataset.hf_dataset["frame_index"], dtype=np.int64)
    index_by_id = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episode_ids, frame_ids, strict=True))
    }
    ordered_indices = [
        index_by_id[(int(sample["episode_index"]), int(sample["frame_index"]))]
        for sample in manifest["samples"]
    ]

    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    compute_loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))
    sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    embed_prefix = nnx_utils.module_jit(model._embed_prefix_impl, static_argnames=("want_lmwm_aux",))

    action_stats = data_config.norm_stats["actions"]
    mean = np.asarray(action_stats.mean, dtype=np.float32)[:14]
    std = np.asarray(action_stats.std, dtype=np.float32)[:14]
    batch_size = int(manifest["batch_size"])
    if len(ordered_indices) % batch_size:
        raise ValueError("frozen sample count must be divisible by batch size")

    rows: list[dict] = []
    root_key = jax.random.key(int(manifest["eval_seed"]))
    offsets = list(range(0, len(ordered_indices), batch_size))
    if args.smoke_one_batch_per_task:
        offsets = []
        for task in sorted(manifest["sample_count_by_task"]):
            first = next(index for index, sample in enumerate(manifest["samples"]) if sample["task"] == task)
            offset = first - first % batch_size
            batch_tasks = {sample["task"] for sample in manifest["samples"][offset : offset + batch_size]}
            if batch_tasks != {task}:
                raise ValueError(f"smoke batch at offset {offset} is not task-pure: {batch_tasks}")
            offsets.append(offset)
    elif args.max_batches is not None:
        offsets = offsets[: args.max_batches]
    for batch_index, offset in enumerate(offsets):
        sample_defs = manifest["samples"][offset : offset + batch_size]
        items = [dataset[index] for index in ordered_indices[offset : offset + batch_size]]
        batch = data_loader._collate_fn(items)
        observation = model_lib.Observation.from_dict(batch)
        actions = jnp.asarray(batch["actions"])

        loss_key = jax.random.fold_in(root_key, batch_index * 3)
        action_key = jax.random.fold_in(root_key, batch_index * 3 + 1)
        noise_key = jax.random.fold_in(root_key, batch_index * 3 + 2)
        loss = compute_loss(loss_key, observation, actions, train=False)
        main_loss = loss["main_loss"] if isinstance(loss, dict) else loss
        flow_loss = np.asarray(jnp.mean(main_loss, axis=-1), dtype=np.float32)

        noise = jax.random.normal(noise_key, actions.shape)
        predicted = sample_actions(
            action_key,
            observation,
            num_steps=args.num_diffusion_steps,
            noise=noise,
        )
        _, _, _, aux = embed_prefix(observation, want_lmwm_aux=False)

        predicted_raw = np.asarray(predicted[..., :14], dtype=np.float32) * std + mean
        actions_raw = np.asarray(actions[..., :14], dtype=np.float32) * std + mean
        pred_flat = predicted_raw.reshape(batch_size, -1)
        action_flat = actions_raw.reshape(batch_size, -1)
        cosine = np.sum(pred_flat * action_flat, axis=-1) / np.maximum(
            np.linalg.norm(pred_flat, axis=-1) * np.linalg.norm(action_flat, axis=-1),
            1e-8,
        )
        endpoint_l2 = np.linalg.norm(predicted_raw[:, -1] - actions_raw[:, -1], axis=-1)

        if args.arm == "s0_privileged":
            shuffled_observation = dataclasses.replace(
                observation,
                lmwm_target_image=jnp.roll(observation.lmwm_target_image, 1, axis=0),
                lmwm_target_mask=jnp.roll(observation.lmwm_target_mask, 1, axis=0),
            )
            shuffled = sample_actions(
                action_key,
                shuffled_observation,
                num_steps=args.num_diffusion_steps,
                noise=noise,
            )
            shuffled_raw = np.asarray(shuffled[..., :14], dtype=np.float32) * std + mean
            sensitivity = np.mean(np.abs(predicted_raw - shuffled_raw), axis=(1, 2))
        else:
            sensitivity = np.zeros((batch_size,), dtype=np.float32)

        availability = np.asarray(aux["lmwm_spatial_availability"], dtype=np.float32)
        gate = np.asarray(aux["lmwm_spatial_gate"], dtype=np.float32).mean(axis=-1)
        token_norm = np.asarray(aux["lmwm_spatial_token_norm"], dtype=np.float32).mean(axis=-1)
        for index, sample in enumerate(sample_defs):
            rows.append(
                {
                    **sample,
                    "flow_loss": float(flow_loss[index]),
                    "action_cosine": float(cosine[index]),
                    "endpoint_l2": float(endpoint_l2[index]),
                    "action_sensitivity_l1": float(sensitivity[index]),
                    "gate_mean": float(gate[index]),
                    "token_norm_mean": float(token_norm[index]),
                    "target_available": float(availability[index]),
                }
            )
        print(f"batch {batch_index + 1}/{len(offsets)}", flush=True)

    result = {
        "schema_version": 2,
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata_sha256": sha256(metadata),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "action_noise_protocol": "jax_normal_fold_in(eval_seed,batch_index*3+2)_manifest_order",
        "eval_seed": int(manifest["eval_seed"]),
        "batch_size": batch_size,
        "num_diffusion_steps": args.num_diffusion_steps,
        "sample_count": len(rows),
        "expected_sample_count": int(manifest["sample_count"]),
        "complete": len(rows) == int(manifest["sample_count"]),
        "aggregate": aggregate(rows),
        "by_task": {
            task: aggregate([row for row in rows if row["task"] == task])
            for task in sorted(manifest["sample_count_by_task"])
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"aggregate": result["aggregate"], "by_task": result["by_task"]}, indent=2))


if __name__ == "__main__":
    main()
