"""Launch matched pi0.5 RoboTwin confirmatory arms without editing config.py."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from openpi.training import config as config_lib
from openpi.training import optimizer as optimizer_lib
from openpi.training import weight_loaders
from openpi import transforms as transforms_lib

import train as train_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("a0", "a2_abs", "a3_live", "s0_no_goal", "s0_current", "s0_privileged"),
        required=True,
    )
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-repo", required=True)
    parser.add_argument("--init-params", required=True)
    parser.add_argument("--asset-id", default="robotwin2.0_absolute_meanstd")
    parser.add_argument("--hint-path")
    parser.add_argument("--target-pairs")
    parser.add_argument("--frame-cache-root")
    parser.add_argument("--episodes-json")
    parser.add_argument("--episode-split", choices=("train", "heldout"), default="train")
    parser.add_argument("--num-train-steps", type=int, default=50_000)
    parser.add_argument("--save-interval", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def require_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def build_config(args: argparse.Namespace) -> config_lib.TrainConfig:
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

    if not Path(args.data_repo, "meta").is_dir():
        raise FileNotFoundError(f"RoboTwin data repo missing meta/: {args.data_repo}")
    require_file(str(Path(args.init_params) / "_METADATA"), "pi0.5 initialization")

    base = config_lib.get_config(base_name)
    episodes = None
    if args.episodes_json:
        split = json.loads(Path(args.episodes_json).read_text())
        episodes = [int(value) for value in split[f"{args.episode_split}_episodes"]]
        if not episodes:
            raise ValueError(f"empty {args.episode_split} episode split: {args.episodes_json}")
    base_data_config = base.data.base_config or config_lib.DataConfig(prompt_from_task=True)
    data_changes = {
        "repo_id": args.data_repo,
        "assets": config_lib.AssetsConfig(asset_id=args.asset_id),
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

    model_changes = {"augment_level": "none"}
    if args.arm.startswith("s0_"):
        model_changes["lmwm_spatial_condition"] = args.arm.removeprefix("s0_")

    return dataclasses.replace(
        base,
        name=args.config_name,
        exp_name=args.exp_name,
        seed=args.seed,
        model=dataclasses.replace(base.model, **model_changes),
        data=dataclasses.replace(base.data, **data_changes),
        weight_loader=weight_loaders.CheckpointWeightLoader(args.init_params),
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
        keep_period=5_000,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        batch_size=16,
        fsdp_devices=1,
        overwrite=not args.resume,
        resume=args.resume,
        wandb_enabled=False,
    )


def audit_view(config: config_lib.TrainConfig) -> dict[str, object]:
    data = config.data
    model = config.model
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
        },
        "data": {
            "repo_id": data.repo_id,
            "asset_id": data.assets.asset_id,
            "adapt_to_pi": data.adapt_to_pi,
            "use_delta_joint_actions": data.use_delta_joint_actions,
            "use_quantile_norm_override": data.use_quantile_norm_override,
            "lmwm_hint_path": data.lmwm_hint_path,
            "lmwm_target_pairs_path": data.lmwm_target_pairs_path,
            "lmwm_target_frame_cache_root": data.lmwm_target_frame_cache_root,
            "episodes": None if data.base_config is None else data.base_config.episodes,
            "repack_outputs": sorted(data.repack_transforms.inputs[0].structure),
        },
        "training": {
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "save_interval": config.save_interval,
            "keep_period": config.keep_period,
            "fsdp_devices": config.fsdp_devices,
            "ema_decay": config.ema_decay,
            "lr_schedule": dataclasses.asdict(config.lr_schedule),
            "optimizer": dataclasses.asdict(config.optimizer),
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
