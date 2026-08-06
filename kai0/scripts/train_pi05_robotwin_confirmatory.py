"""Launch matched pi0.5 RoboTwin confirmatory arms without editing config.py."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import flax.nnx as nnx
import train as train_script

from openpi import transforms as transforms_lib
from openpi.shared import nnx_utils
from openpi.training import config as config_lib
from openpi.training import optimizer as optimizer_lib
from openpi.training import weight_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=(
            "a0",
            "a2_abs",
            "a3_live",
            "s0_no_goal",
            "s0_current",
            "s0_privileged",
            "mt1_oracle",
            "mt2_null",
            "mt3_learned",
            "mt5_local",
            "mt5_combined",
            "p0_predictive",
            "p1_predictive",
            "r1_crave",
            "r1_combined",
        ),
        required=True,
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-repo", required=True)
    parser.add_argument("--init-params", required=True)
    parser.add_argument("--asset-id", default="robotwin2.0_absolute_meanstd")
    parser.add_argument(
        "--norm-assets-dir",
        help="Explicit parent directory containing asset-id/norm_stats.json",
    )
    parser.add_argument("--hint-path")
    parser.add_argument("--target-pairs")
    parser.add_argument("--frame-cache-root")
    parser.add_argument("--transition-pairs")
    parser.add_argument("--tracker-checkpoint")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--crave-targets")
    parser.add_argument(
        "--tracker-candidate", choices=("current_frame", "history_proprio")
    )
    parser.add_argument("--episodes-json")
    parser.add_argument("--episode-split", choices=("train", "heldout"), default="train")
    parser.add_argument("--num-train-steps", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def build_config(args: argparse.Namespace) -> config_lib.TrainConfig:
    if args.fsdp_devices < 1:
        raise ValueError("--fsdp-devices must be positive")
    if args.arm == "a0":
        base_name = "pi05_robotwin_a0_public_recipe_bj"
    elif args.arm == "a2_abs":
        base_name = "pi05_robotwin_a2_prefix_official_bj"
        if not args.hint_path:
            raise ValueError("--hint-path is required for a2_abs")
        require_file(args.hint_path, "A2 absolute hint")
    elif args.arm == "a3_live":
        base_name = "pi05_robotwin_a3_live_residual_prefix_official_bj"
        if not args.target_pairs or not args.frame_cache_root:
            raise ValueError("--target-pairs and --frame-cache-root are required for a3_live")
        require_file(args.target_pairs, "A3 target pairs")
        if not Path(args.frame_cache_root).is_dir():
            raise FileNotFoundError(f"A3 frame cache missing: {args.frame_cache_root}")
    else:
        base_name = "pi05_robotwin_a0_public_exact_bj"
        if args.arm == "s0_privileged":
            if not args.target_pairs or not args.frame_cache_root:
                raise ValueError("S0 privileged requires --target-pairs and --frame-cache-root")
            require_file(args.target_pairs, "S0 target pairs")
            if not Path(args.frame_cache_root).is_dir():
                raise FileNotFoundError(f"S0 frame cache missing: {args.frame_cache_root}")
        if args.arm in {"mt1_oracle", "mt2_null", "mt3_learned", "mt5_combined"}:
            if not args.transition_pairs:
                raise ValueError(f"{args.arm} requires --transition-pairs")
            require_file(args.transition_pairs, "milestone-transition pairs")
        if args.arm in {"mt3_learned", "mt5_combined"}:
            if not args.tracker_checkpoint or not args.tracker_candidate:
                raise ValueError(
                    f"{args.arm} requires --tracker-checkpoint and --tracker-candidate"
                )
            require_file(args.tracker_checkpoint, "selected MT3 tracker checkpoint")
        if args.arm in {
            "mt5_local",
            "mt5_combined",
            "p0_predictive",
            "p1_predictive",
            "r1_combined",
        }:
            if not args.target_pairs or not args.frame_cache_root:
                raise ValueError(f"{args.arm} requires --target-pairs and --frame-cache-root")
            require_file(args.target_pairs, "fixed-horizon target pairs")
            if not Path(args.frame_cache_root).is_dir():
                raise FileNotFoundError(f"fixed-horizon frame cache missing: {args.frame_cache_root}")
        if args.arm in {"r1_crave", "r1_combined"}:
            if not args.crave_targets:
                raise ValueError(f"{args.arm} requires --crave-targets")
            require_file(args.crave_targets, "frozen CRAVE training targets")

    if not Path(args.data_repo, "meta").is_dir():
        raise FileNotFoundError(f"RoboTwin data repo missing meta/: {args.data_repo}")
    require_file(str(Path(args.init_params) / "_METADATA"), "pi0.5 initialization")

    base = config_lib.get_config(base_name)
    norm_assets_dir = Path(args.norm_assets_dir).resolve() if args.norm_assets_dir else base.assets_dirs
    norm_stats_path = norm_assets_dir / args.asset_id / "norm_stats.json"
    require_file(str(norm_stats_path), "normalization statistics")
    episodes = None
    if args.episodes_json:
        split = json.loads(Path(args.episodes_json).read_text())
        episodes = [int(value) for value in split[f"{args.episode_split}_episodes"]]
        if not episodes:
            raise ValueError(f"empty {args.episode_split} episode split: {args.episodes_json}")
    base_data_config = base.data.base_config or config_lib.DataConfig(prompt_from_task=True)
    data_changes = {
        "repo_id": args.data_repo,
        "assets": config_lib.AssetsConfig(
            assets_dir=str(norm_assets_dir), asset_id=args.asset_id
        ),
        "use_delta_joint_actions": False,
        "use_quantile_norm_override": False,
        "base_config": dataclasses.replace(base_data_config, episodes=episodes),
    }
    if args.arm == "a2_abs":
        data_changes["lmwm_hint_path"] = args.hint_path
    elif args.arm == "a3_live":
        data_changes["lmwm_target_pairs_path"] = args.target_pairs
        data_changes["lmwm_target_frame_cache_root"] = args.frame_cache_root
    elif args.arm == "s0_privileged":
        data_changes["lmwm_target_pairs_path"] = args.target_pairs
        data_changes["lmwm_target_frame_cache_root"] = args.frame_cache_root
        original_repack = base.data.repack_transforms.inputs[0]
        spatial_structure = dict(original_repack.structure)
        spatial_structure["lmwm_target_image"] = "lmwm_target_image"
        spatial_structure["lmwm_target_mask"] = "lmwm_target_mask"
        data_changes["repack_transforms"] = transforms_lib.Group(
            inputs=[dataclasses.replace(original_repack, structure=spatial_structure)]
        )
    elif args.arm in {"mt1_oracle", "mt2_null", "mt3_learned"}:
        data_changes["lmwm_transition_pairs_path"] = args.transition_pairs
        if args.arm == "mt3_learned" and args.tracker_candidate == "history_proprio":
            data_changes["lmwm_transition_history_offsets"] = (-15, -7, 0)
        original_repack = base.data.repack_transforms.inputs[0]
        transition_keys = (
            "lmwm_transition_task",
            "lmwm_transition_current",
            "lmwm_transition_next",
            "lmwm_transition_mask",
        )
        if args.arm == "mt3_learned" and args.tracker_candidate == "history_proprio":
            transition_keys += (
                "lmwm_transition_history_images",
                "lmwm_transition_history_state",
            )
        transition_structure = {
            **original_repack.structure,
            **{key: key for key in transition_keys},
        }
        data_changes["repack_transforms"] = transforms_lib.Group(
            inputs=[dataclasses.replace(original_repack, structure=transition_structure)]
        )
    elif args.arm in {
        "mt5_local",
        "mt5_combined",
        "p0_predictive",
        "p1_predictive",
        "r1_combined",
    }:
        data_changes["lmwm_target_pairs_path"] = args.target_pairs
        data_changes["lmwm_target_frame_cache_root"] = args.frame_cache_root
        original_repack = base.data.repack_transforms.inputs[0]
        structure = {
            **original_repack.structure,
            "lmwm_target_image": "lmwm_target_image",
            "lmwm_target_mask": "lmwm_target_mask",
        }
        if args.arm == "p0_predictive":
            # Offline P0 never consumes wrist tokens. Let AlohaInputs create
            # masked placeholders instead of decoding two unused videos.
            structure["images"] = {
                "cam_high": original_repack.structure["images"]["cam_high"]
            }
        if args.arm == "mt5_combined":
            data_changes["lmwm_transition_pairs_path"] = args.transition_pairs
            if args.tracker_candidate == "history_proprio":
                data_changes["lmwm_transition_history_offsets"] = (-15, -7, 0)
            transition_keys = (
                "lmwm_transition_task",
                "lmwm_transition_current",
                "lmwm_transition_next",
                "lmwm_transition_mask",
            )
            if args.tracker_candidate == "history_proprio":
                transition_keys += (
                    "lmwm_transition_history_images",
                    "lmwm_transition_history_state",
                )
            structure.update({key: key for key in transition_keys})
        if args.arm == "r1_combined":
            data_changes["crave_targets_path"] = args.crave_targets
            structure.update(
                {
                    key: key
                    for key in (
                        "crave_progress_change",
                        "crave_target_density",
                        "crave_boundary_crossing",
                        "crave_target_mask",
                    )
                }
            )
        data_changes["repack_transforms"] = transforms_lib.Group(
            inputs=[dataclasses.replace(original_repack, structure=structure)]
        )
    elif args.arm == "r1_crave":
        data_changes["crave_targets_path"] = args.crave_targets
        original_repack = base.data.repack_transforms.inputs[0]
        structure = {
            **original_repack.structure,
            **{
                key: key
                for key in (
                    "crave_progress_change",
                    "crave_target_density",
                    "crave_boundary_crossing",
                    "crave_target_mask",
                )
            },
        }
        data_changes["repack_transforms"] = transforms_lib.Group(
            inputs=[dataclasses.replace(original_repack, structure=structure)]
        )

    model_changes = {"augment_level": "none"}
    if args.arm.startswith("s0_"):
        model_changes["lmwm_spatial_condition"] = args.arm.removeprefix("s0_")
    elif args.arm == "mt1_oracle":
        model_changes["lmwm_transition_condition"] = "oracle"
    elif args.arm == "mt2_null":
        model_changes["lmwm_transition_condition"] = "null"
    elif args.arm == "mt3_learned":
        model_changes["lmwm_transition_condition"] = "learned"
        model_changes["lmwm_transition_tracker"] = args.tracker_candidate
    elif args.arm in {"mt5_local", "mt5_combined"}:
        model_changes["lmwm_local_dynamics"] = True
        if args.arm == "mt5_combined":
            model_changes["lmwm_transition_condition"] = "learned"
            model_changes["lmwm_transition_tracker"] = args.tracker_candidate
    elif args.arm in {"p0_predictive", "p1_predictive", "r1_combined"}:
        model_changes["predictive_adapter_mode"] = (
            "offline" if args.arm == "p0_predictive" else "joint"
        )
        model_changes["predictive_adapter_intervention"] = "normal"
    if args.arm in {"r1_crave", "r1_combined"}:
        model_changes["recurrence_adapter_mode"] = "joint"
        model_changes["recurrence_adapter_intervention"] = "normal"

    weight_loader = weight_loaders.CheckpointWeightLoader(args.init_params)
    if args.arm in {"mt3_learned", "mt5_combined"}:
        weight_loader = weight_loaders.CheckpointWithMT3TrackerWeightLoader(
            args.init_params,
            args.tracker_checkpoint,
            args.tracker_candidate,
        )
    elif args.arm in {"p1_predictive", "r1_combined"}:
        if not args.adapter_checkpoint:
            raise ValueError(f"{args.arm} requires --adapter-checkpoint")
        adapter_params = Path(args.adapter_checkpoint) / "params"
        require_file(str(adapter_params / "_METADATA"), "accepted P0 adapter checkpoint")
        weight_loader = weight_loaders.CheckpointWithPredictiveAdapterWeightLoader(
            args.init_params,
            str(adapter_params),
        )

    freeze_filter = base.freeze_filter
    if args.arm == "p0_predictive":
        freeze_filter = nnx.Not(nnx_utils.PathRegex(".*predictive_action_adapter.*"))

    return dataclasses.replace(
        base,
        name=args.config_name,
        exp_name=args.exp_name,
        seed=args.seed,
        model=dataclasses.replace(base.model, **model_changes),
        data=dataclasses.replace(base.data, **data_changes),
        weight_loader=weight_loader,
        freeze_filter=freeze_filter,
        lr_schedule=optimizer_lib.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=optimizer_lib.AdamW(weight_decay=0.01),
        ema_decay=None,
        num_train_steps=args.num_train_steps,
        save_interval=args.save_interval,
        save_final_checkpoint=not args.skip_final_checkpoint,
        keep_period=5_000,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        batch_size=16,
        fsdp_devices=args.fsdp_devices,
        overwrite=not args.resume,
        resume=args.resume,
        wandb_enabled=False,
    )


def audit_view(config: config_lib.TrainConfig) -> dict[str, object]:
    data = config.data
    model = config.model
    episodes = None if data.base_config is None else data.base_config.episodes
    episode_audit = None
    if episodes is not None:
        encoded = json.dumps(episodes, separators=(",", ":")).encode()
        episode_audit = {
            "count": len(episodes),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    norm_stats_path = Path(data.assets.assets_dir) / data.assets.asset_id / "norm_stats.json"
    return {
        "name": config.name,
        "exp_name": config.exp_name,
        "seed": config.seed,
        "model": {
            "action_horizon": model.action_horizon,
            "augment_level": model.augment_level,
            "lmwm_hint_dim": model.lmwm_hint_dim,
            "lmwm_live_hint": model.lmwm_live_hint,
            "lmwm_live_residual": model.lmwm_live_residual,
            "lmwm_spatial_condition": model.lmwm_spatial_condition,
            "lmwm_spatial_grid_size": model.lmwm_spatial_grid_size,
            "lmwm_transition_condition": model.lmwm_transition_condition,
            "lmwm_transition_num_tasks": model.lmwm_transition_num_tasks,
            "lmwm_transition_num_stages": model.lmwm_transition_num_stages,
            "lmwm_transition_tracker": model.lmwm_transition_tracker,
            "lmwm_transition_dropout_probability": model.lmwm_transition_dropout_probability,
            "lmwm_transition_auxiliary_loss_weight": model.lmwm_transition_auxiliary_loss_weight,
            "lmwm_local_dynamics": model.lmwm_local_dynamics,
            "lmwm_local_hidden_dim": model.lmwm_local_hidden_dim,
            "lmwm_local_loss_weight": model.lmwm_local_loss_weight,
            "predictive_adapter_mode": model.predictive_adapter_mode,
            "predictive_adapter_grid_size": model.predictive_adapter_grid_size,
            "predictive_adapter_hidden_dim": model.predictive_adapter_hidden_dim,
            "predictive_adapter_loss_weight": model.predictive_adapter_loss_weight,
            "predictive_adapter_intervention": model.predictive_adapter_intervention,
            "recurrence_adapter_mode": model.recurrence_adapter_mode,
            "recurrence_adapter_bins": model.recurrence_adapter_bins,
            "recurrence_adapter_hidden_dim": model.recurrence_adapter_hidden_dim,
            "recurrence_adapter_loss_weight": model.recurrence_adapter_loss_weight,
            "recurrence_adapter_intervention": model.recurrence_adapter_intervention,
        },
        "data": {
            "repo_id": data.repo_id,
            "asset_id": data.assets.asset_id,
            "assets_dir": data.assets.assets_dir,
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_sha256": hashlib.sha256(norm_stats_path.read_bytes()).hexdigest(),
            "adapt_to_pi": data.adapt_to_pi,
            "use_delta_joint_actions": data.use_delta_joint_actions,
            "use_quantile_norm_override": data.use_quantile_norm_override,
            "lmwm_hint_path": data.lmwm_hint_path,
            "lmwm_target_pairs_path": data.lmwm_target_pairs_path,
            "lmwm_target_frame_cache_root": data.lmwm_target_frame_cache_root,
            "lmwm_transition_pairs_path": data.lmwm_transition_pairs_path,
            "lmwm_transition_history_offsets": data.lmwm_transition_history_offsets,
            "crave_targets_path": data.crave_targets_path,
            "tracker_checkpoint": getattr(config.weight_loader, "tracker_path", None),
            "tracker_candidate": getattr(config.weight_loader, "candidate", None),
            "episodes": episode_audit,
            "repack_outputs": sorted(data.repack_transforms.inputs[0].structure),
        },
        "training": {
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "save_interval": config.save_interval,
            "save_final_checkpoint": config.save_final_checkpoint,
            "keep_period": config.keep_period,
            "fsdp_devices": config.fsdp_devices,
            "ema_decay": config.ema_decay,
            "lr_schedule": dataclasses.asdict(config.lr_schedule),
            "optimizer": dataclasses.asdict(config.optimizer),
        },
        "initialization": {
            "loader": type(config.weight_loader).__name__,
            "base_params": getattr(config.weight_loader, "params_path", None),
            "adapter_params": getattr(config.weight_loader, "adapter_params_path", None),
        },
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.dry_run:
        print(json.dumps(audit_view(config), indent=2, sort_keys=True))
        return
    train_script.main(config)


if __name__ == "__main__":
    main()
