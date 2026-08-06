"""Evaluate the frozen P0 predictive-adapter panel and apply its gate."""

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


def resolve_norm_assets_dir(checkpoint: Path, override: Path | None) -> Path:
    root = override.resolve() if override is not None else (checkpoint / "assets").resolve()
    norm_stats = root / "robotwin2.0_absolute_meanstd/norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(f"normalization statistics missing: {norm_stats}")
    return root


def paired_episode_bootstrap(
    episodes: np.ndarray,
    normal: np.ndarray,
    control: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(episodes, return_inverse=True)
    counts = np.bincount(inverse)
    differences = normal - control
    episode_means = np.bincount(inverse, weights=differences) / counts
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        sample = rng.integers(0, unique.size, size=(stop - start, unique.size))
        estimates[start:stop] = np.mean(episode_means[sample], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "episode_count": int(unique.size),
        "mean_difference": float(np.mean(episode_means)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def build_config(args: argparse.Namespace, episodes_json: Path):
    norm_assets_dir = resolve_norm_assets_dir(args.checkpoint, args.norm_assets_dir)
    train_args = argparse.Namespace(
        arm="p0_predictive",
        config_name="pi05_predictive_adapter_p0",
        exp_name="offline_eval",
        seed=1000,
        data_repo=str(args.data_repo),
        init_params=str(args.checkpoint / "params"),
        asset_id="robotwin2.0_absolute_meanstd",
        norm_assets_dir=str(norm_assets_dir),
        hint_path=None,
        target_pairs=str(args.pairs),
        frame_cache_root=str(args.frame_cache_root),
        transition_pairs=None,
        tracker_checkpoint=None,
        tracker_candidate=None,
        episodes_json=str(episodes_json),
        episode_split="heldout",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-repo", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--eval-panel", type=Path, required=True)
    parser.add_argument("--episode-split", type=Path, required=True)
    parser.add_argument("--frame-cache-root", type=Path, required=True)
    parser.add_argument(
        "--norm-assets-dir",
        type=Path,
        help="assets root; defaults to CHECKPOINT/assets",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in (
        "checkpoint",
        "data_repo",
        "pairs",
        "eval_panel",
        "episode_split",
        "frame_cache_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.norm_assets_dir is not None:
        args.norm_assets_dir = args.norm_assets_dir.resolve()
    metadata = args.checkpoint / "params" / "_METADATA"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")

    with np.load(args.eval_panel) as panel_file:
        panel = {name: np.asarray(panel_file[name]) for name in panel_file.files}
    sample_count = int(panel["row_index"].size)
    if sample_count % args.batch_size:
        raise ValueError("frozen panel size must be divisible by batch size")
    if not np.all(panel["heldout"]):
        raise ValueError("evaluation panel contains training rows")

    split = json.loads(args.episode_split.read_text())
    heldout_episodes = [int(value) for value in split["heldout_episodes"]]
    if set(map(int, panel["cur_ep"])) - set(heldout_episodes):
        raise ValueError("evaluation panel and episode split disagree")

    config = build_config(args, args.episode_split)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(
        data_config,
        config.model.action_horizon,
        config.model,
        episodes=heldout_episodes,
    )
    dataset = data_loader.transform_dataset(dataset, data_config)
    base_dataset = unwrap_lerobot(dataset)
    episode_ids = np.asarray(base_dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(base_dataset.hf_dataset["frame_index"], dtype=np.int64)
    index_by_id = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episode_ids, frame_ids, strict=True))
    }
    ordered_indices = [
        index_by_id[(int(episode), int(frame))] for episode, frame in zip(panel["cur_ep"], panel["cur_fi"], strict=True)
    ]

    params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    evaluate = nnx_utils.module_jit(model.compute_predictive_control_cosines)

    outputs = {name: [] for name in ("normal", "shuffled", "masked")}
    full_batch_count = sample_count // args.batch_size
    first_batch = full_batch_count * args.shard_index // args.shard_count
    last_batch = full_batch_count * (args.shard_index + 1) // args.shard_count
    total_batches = last_batch - first_batch
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    root_key = jax.random.key(args.bootstrap_seed)
    for local_batch_index in range(total_batches):
        batch_index = first_batch + local_batch_index
        start = batch_index * args.batch_size
        indices = ordered_indices[start : start + args.batch_size]
        items = [dataset[index] for index in indices]
        batch = data_loader._collate_fn(items)
        observation = model_lib.Observation.from_dict(batch)
        actions = jnp.asarray(batch["actions"])
        scores = evaluate(jax.random.fold_in(root_key, batch_index), observation, actions)
        for name in outputs:
            outputs[name].append(np.asarray(scores[name], dtype=np.float32))
        print(f"batch {local_batch_index + 1}/{total_batches}", flush=True)

    completed_count = total_batches * args.batch_size
    panel_start = first_batch * args.batch_size
    panel_stop = panel_start + completed_count
    scores = {name: np.concatenate(values) for name, values in outputs.items()}
    episodes = panel["cur_ep"][panel_start:panel_stop]
    comparisons = {
        control: paired_episode_bootstrap(
            episodes,
            scores["normal"],
            scores[control],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed + offset,
        )
        for offset, control in enumerate(("shuffled", "masked"), start=1)
    }
    shard_expected_count = (last_batch - first_batch) * args.batch_size
    shard_complete = completed_count == shard_expected_count
    complete = args.shard_count == 1 and shard_complete
    accepted = complete and all(value["ci95_low"] > 0.0 for value in comparisons.values())
    result = {
        "schema_version": 1,
        "protocol": "pi05_predictive_action_adapter_p0_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata_sha256": sha256(metadata),
        "pairs_sha256": sha256(args.pairs),
        "eval_panel": str(args.eval_panel),
        "eval_panel_sha256": sha256(args.eval_panel),
        "episode_split_sha256": sha256(args.episode_split),
        "norm_stats_sha256": sha256(
            (args.norm_assets_dir or args.checkpoint / "assets") / "robotwin2.0_absolute_meanstd/norm_stats.json"
        ),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "model_source_sha256": sha256(Path(__file__).resolve().parents[1] / "src/openpi/models/pi0.py"),
        "interventions": {
            "normal": "unaltered action sequence",
            "shuffled": "deterministic reversal along the action-horizon axis",
            "masked": "all-zero action sequence",
        },
        "sample_count": completed_count,
        "expected_sample_count": shard_expected_count,
        "full_panel_sample_count": sample_count,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "shard_complete": shard_complete,
        "panel_start": panel_start,
        "panel_stop": panel_stop,
        "complete": complete,
        "batch_size": args.batch_size,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "aggregate": {name: float(np.mean(value)) for name, value in scores.items()},
        "comparisons": comparisons,
        "accepted": accepted,
        "samples": {
            "panel_row": list(range(panel_start, panel_stop)),
            "episode_index": episodes.tolist(),
            "frame_index": panel["cur_fi"][panel_start:panel_stop].tolist(),
            **{name: value.tolist() for name, value in scores.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("aggregate", "comparisons", "accepted")}, indent=2))


if __name__ == "__main__":
    main()
