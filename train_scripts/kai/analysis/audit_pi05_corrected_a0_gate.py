#!/usr/bin/env python3
"""Audit the corrected pi0.5 A0 protocol and emit its confirmatory gate."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_counts(data: dict) -> tuple[int, int]:
    """Read both current list-based and legacy scalar RoboTwin summaries."""
    episode_records = data.get("episodes", [])
    episodes = int(
        data.get(
            "n_episodes",
            len(episode_records) if isinstance(episode_records, list) else episode_records,
        )
    )
    successes = int(
        data.get(
            "successes",
            sum(bool(item.get("success")) for item in episode_records)
            if isinstance(episode_records, list)
            else round(float(data["success_rate"]) * episodes),
        )
    )
    return successes, episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--eval-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--norm-stats", type=pathlib.Path, required=True)
    parser.add_argument("--launch-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--launch-config-snapshot", type=pathlib.Path)
    parser.add_argument("--config-name", default="pi05_robotwin_a0_public_recipe_bj")
    parser.add_argument("--expected-job-name", default="pi05-a0-public-recipe-s1000-bj4g")
    parser.add_argument("--expected-exp-name", default="pi05_robotwin_a0_public_recipe_seed1000")
    parser.add_argument("--expected-seed", type=int, default=1000)
    parser.add_argument("--dataset-manifest", type=pathlib.Path)
    parser.add_argument("--require-no-augmentation", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--marker", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-macro", type=float, default=0.70)
    args = parser.parse_args()
    launch_config_snapshot = args.launch_config_snapshot or (
        args.repo
        / "lmvla/paper_iclr_lmvla/manifests/pi05_a0_seed1000_config_at_launch.py"
    )

    sys.path.insert(0, str(args.repo / "kai0/src"))
    from openpi.training import config as training_config

    failures: list[str] = []
    launch_manifest = json.loads(args.launch_manifest.read_text())
    expected_env = {
        "PI05_ABSOLUTE_ACTIONS": "1",
        "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
        "PI05_CONFIG_NAME": args.config_name,
        "PI05_EXP_NAME": args.expected_exp_name,
        "PI05_TRAIN_SEED": str(args.expected_seed),
    }
    manifest_protocol = launch_manifest.get("protocol", {})
    launch_id = str(launch_manifest.get("job_id", ""))
    manifest_checks = {
        "launch_run_id": launch_id.startswith("t-") or launch_id.startswith("gf1-"),
        "launch_job_name": launch_manifest.get("job_name") == args.expected_job_name,
        "launch_env": launch_manifest.get("runtime_env") == expected_env,
        "launch_gpus_4": launch_manifest.get("resource", {}).get("gpus") == 4,
        "launch_batch_size_16": manifest_protocol.get("batch_size") == 16,
        "launch_steps_50000": manifest_protocol.get("num_train_steps") == 50000,
        "launch_absolute_actions": manifest_protocol.get("actions") == "absolute",
        "launch_mean_std_norm": manifest_protocol.get("normalization") == "mean_std",
        "launch_seed": manifest_protocol.get("training_seed") == args.expected_seed,
        "launch_runtime_command_captured": bool(
            launch_manifest.get("sha256", {}).get("runtime_command_from_api")
            or launch_manifest.get("sha256", {}).get("submitted_yaml")
            or launch_manifest.get("sha256", {}).get("launch_command")
        ),
    }
    failures.extend(name for name, passed in manifest_checks.items() if not passed)

    config = training_config.get_config(args.config_name)
    checks = {
        "batch_size_16": config.batch_size == 16,
        "steps_50000": config.num_train_steps == 50000,
        "absolute_actions": config.data.use_delta_joint_actions is False,
        "mean_std_norm": config.data.use_quantile_norm_override is False,
        "absolute_meanstd_asset": config.data.assets.asset_id == "robotwin2.0_absolute_meanstd",
        "no_hint": config.model.lmwm_hint_dim == 0 and config.model.lmwm_live_hint is False,
        "action_horizon_50": config.model.action_horizon == 50,
        "pi05": config.model.pi05 is True,
        "pi05_base_init": str(config.weight_loader.params_path).endswith(
            "/base_init_ckpts/extracted/pi05_base/params"
        ),
        "optimizer_accumulation_1": config.optimizer.grad_accumulation_steps == 1,
        "warmup_1000": config.lr_schedule.warmup_steps == 1000,
        "peak_lr_2p5e5": config.lr_schedule.peak_lr == 2.5e-5,
    }
    if args.require_no_augmentation:
        checks["augmentation_disabled"] = config.model.augment_level == "none"
    if args.dataset_manifest:
        expected_dataset = args.dataset_manifest.parent.parent
        checks["official_prompt_dataset"] = (
            pathlib.Path(str(config.data.repo_id)).resolve() == expected_dataset.resolve()
        )
    failures.extend(name for name, passed in checks.items() if not passed)

    config_source = args.repo / "kai0/src/openpi/training/config.py"
    launched_config_hash = launch_manifest.get("sha256", {}).get(
        "north_config_source_at_launch"
    )
    if not config_source.is_file():
        failures.append(f"missing_training_config_source={config_source}")
    if not launch_config_snapshot.is_file():
        failures.append(f"missing_launch_config_snapshot={launch_config_snapshot}")
    elif sha256(launch_config_snapshot) != launched_config_hash:
        failures.append("launch_config_snapshot_hash_mismatch")

    summaries = []
    for path_text in glob.glob(str(args.eval_root / "**/summary.json"), recursive=True):
        data = json.loads(pathlib.Path(path_text).read_text())
        successes, episodes = outcome_counts(data)
        summaries.append((path_text, successes, episodes))
    if len(summaries) != 24:
        failures.append(f"summary_count={len(summaries)} expected=24")
    total_episodes = sum(item[2] for item in summaries)
    total_successes = sum(item[1] for item in summaries)
    macro = (
        sum(item[1] / item[2] for item in summaries if item[2]) / len(summaries)
        if summaries
        else 0.0
    )
    if macro < args.minimum_macro:
        failures.append(f"macro={macro:.6f} below={args.minimum_macro:.6f}")

    required_files = {
        "checkpoint_metadata": args.checkpoint / "params/_METADATA",
        "norm_stats": args.norm_stats,
        "training_config_source": config_source,
        "launch_config_snapshot": launch_config_snapshot,
        "launch_manifest": args.launch_manifest,
    }
    if args.dataset_manifest:
        required_files["dataset_manifest"] = args.dataset_manifest
    for name, path in required_files.items():
        if not path.is_file():
            failures.append(f"missing_{name}={path}")

    dataset_audit = None
    if args.dataset_manifest and args.dataset_manifest.is_file():
        dataset_audit = json.loads(args.dataset_manifest.read_text())
        invariants = dataset_audit.get("invariants", {})
        dataset_checks = {
            "dataset_27500_episodes": dataset_audit.get("episodes") == 27_500,
            "dataset_6075103_frames": dataset_audit.get("frames") == 6_075_103,
            "dataset_23559_tasks": dataset_audit.get("official_tasks") == 23_559,
            "one_task_per_episode": invariants.get("one_official_task_per_episode") is True,
            "source_timestamps_preserved": invariants.get("source_timestamps_preserved") is True,
            "source_videos_symlinked": invariants.get(
                "source_videos_symlinked_without_reencoding"
            )
            is True,
        }
        failures.extend(name for name, passed in dataset_checks.items() if not passed)
    else:
        dataset_checks = {}

    result = {
        "accepted": not failures,
        "minimum_macro": args.minimum_macro,
        "macro_success_rate": macro,
        "summary_count": len(summaries),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "protocol_checks": checks,
        "launch_manifest_checks": manifest_checks,
        "dataset_checks": dataset_checks,
        "source_provenance": {
            "launch_snapshot_matches_recorded_hash": launch_config_snapshot.is_file()
            and sha256(launch_config_snapshot) == launched_config_hash,
            "current_source_matches_launch": config_source.is_file()
            and sha256(config_source) == launched_config_hash,
        },
        "launch_job_id": launch_manifest.get("job_id"),
        "failures": failures,
        "hashes_sha256": {
            name: sha256(path) for name, path in required_files.items() if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.unlink(missing_ok=True)
    if result["accepted"]:
        args.marker.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
