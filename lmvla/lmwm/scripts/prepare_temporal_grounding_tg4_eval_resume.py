#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
]


def expected_run_dir(
    result_root: Path,
    *,
    arm: str,
    train_seed: int,
    condition: str,
    eval_seed: int,
) -> Path:
    alias = f"tg4_{arm}_s{train_seed}"
    run_group = f"{alias}__demo_clean"
    run_tag = f"tg4-{arm}-s{train_seed}-{condition}-e{eval_seed}"
    return result_root / f"seed{eval_seed}" / run_group / run_tag


def validate_resume_root(
    result_root: Path,
    checkpoint: Path,
    *,
    arm: str,
    train_seed: int,
    condition: str,
    eval_seed: int,
) -> Path | None:
    if not result_root.exists():
        return None
    if not result_root.is_dir():
        raise ValueError(f"TG4 result root is not a directory: {result_root}")

    expected_dirs = {
        seed: expected_run_dir(
            result_root,
            arm=arm,
            train_seed=train_seed,
            condition=condition,
            eval_seed=seed,
        )
        for seed in range(4)
    }
    allowed_seed_names = {f"seed{seed}" for seed in range(4)}
    for entry in result_root.iterdir():
        if not entry.is_dir() or entry.name not in allowed_seed_names:
            raise ValueError(f"unexpected entry in TG4 result root: {entry}")

    for seed, run_dir in expected_dirs.items():
        seed_root = result_root / f"seed{seed}"
        if not seed_root.exists():
            continue
        run_group = run_dir.parent
        unexpected_groups = [entry for entry in seed_root.iterdir() if entry != run_group]
        if unexpected_groups:
            raise ValueError(f"unexpected TG4 run group under {seed_root}: {unexpected_groups}")
        if not run_group.exists():
            continue
        unexpected_runs = [entry for entry in run_group.iterdir() if entry != run_dir]
        if unexpected_runs:
            raise ValueError(f"unexpected TG4 run tag under {run_group}: {unexpected_runs}")
        if not run_dir.exists():
            continue
        meta_path = run_dir / "run_meta.json"
        if not meta_path.is_file():
            raise ValueError(f"cannot resume TG4 run without run_meta.json: {run_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {
            "run_group": run_group.name,
            "run_tag": run_dir.name,
            "checkpoint_alias": f"tg4_{arm}_s{train_seed}",
            "task_config": "demo_clean",
            "requested_tasks": TASKS,
            "expected_test_num": 50,
        }
        mismatches = {
            key: {"expected": value, "actual": meta.get(key)}
            for key, value in expected.items()
            if meta.get(key) != value
        }
        meta_checkpoint = Path(str(meta.get("checkpoint_path", ""))).expanduser()
        if not meta_checkpoint.is_file() or meta_checkpoint.resolve() != checkpoint.resolve():
            mismatches["checkpoint_path"] = {
                "expected": str(checkpoint.resolve()),
                "actual": str(meta_checkpoint),
            }
        if mismatches:
            raise ValueError(f"TG4 resume metadata mismatch in {meta_path}: {mismatches}")

    selected = expected_dirs[eval_seed]
    return selected if selected.is_dir() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--eval-seed", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    selected = validate_resume_root(
        args.result_root.resolve(),
        args.checkpoint.resolve(),
        arm=args.arm,
        train_seed=args.train_seed,
        condition=args.condition,
        eval_seed=args.eval_seed,
    )
    if selected is not None:
        print(selected)


if __name__ == "__main__":
    main()
