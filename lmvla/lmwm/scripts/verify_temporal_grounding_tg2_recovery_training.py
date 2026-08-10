#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
SEEDS = (1000, 1001, 1002)


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {label}, found {len(paths)}: {paths}")
    return paths[0]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_overlay(repo: Path, overlay: Path) -> dict[str, str]:
    source_root = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    checkpoint_root = overlay / "lmvla/lawam/results/Checkpoints/robotwin"
    sidecar_root = (
        overlay / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
    )
    checkpoint_root.mkdir(parents=True)
    sidecar_root.mkdir(parents=True)
    mapping = {}
    for seed in SEEDS:
        for arm in ARMS:
            old_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
            recovery_id = f"temporal_grounding_tg2r_{arm}_seed{seed}"
            run = one(sorted(source_root.glob(f"*+{recovery_id}")), recovery_id)
            sidecars = (
                repo
                / "logs/resource_scheduler_local/temporal_grounding_tg2r_sidecars"
                / recovery_id
            )
            if not sidecars.is_dir():
                raise FileNotFoundError(sidecars)
            (checkpoint_root / f"recovery+{old_id}").symlink_to(
                run, target_is_directory=True
            )
            (sidecar_root / old_id).symlink_to(sidecars, target_is_directory=True)
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            data = config["datasets"]["vla_data"]
            if data.get("in_order") is not True:
                raise ValueError(f"TG2R in_order is not true for {recovery_id}")
            if data.get("num_workers") != 8:
                raise ValueError(f"TG2R worker drift for {recovery_id}")
            mapping[old_id] = str(run)
    return mapping


def verify(repo: Path, output: Path, seed_output: Path, min_state_bytes: int) -> dict:
    script_dir = Path(__file__).resolve().parent
    training_v2 = script_dir / "verify_temporal_grounding_tg2_training_v2.py"
    seed_verifier = script_dir / "verify_temporal_grounding_tg2_seed_independence.py"
    with tempfile.TemporaryDirectory(prefix="tg2r-integrity-") as temporary:
        overlay = Path(temporary)
        mapping = build_overlay(repo, overlay)
        temporary_training = overlay / "training.json"
        temporary_seed = overlay / "seed.json"
        subprocess.run(
            [
                sys.executable,
                str(training_v2),
                "--repo",
                str(overlay),
                "--output",
                str(temporary_training),
                "--min-state-bytes",
                str(min_state_bytes),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(seed_verifier),
                "--repo",
                str(overlay),
                "--output",
                str(temporary_seed),
            ],
            check=True,
        )
        result = json.loads(temporary_training.read_text(encoding="utf-8"))
        seed_result = json.loads(temporary_seed.read_text(encoding="utf-8"))
    result["protocol"] = "temporal_grounding_tg2_recovery_training_integrity_v1"
    result["checks"]["in_order_true_all_runs"] = True
    result["checks"]["num_workers_eight_all_runs"] = True
    result["recovery_run_mapping"] = mapping
    seed_result["protocol"] = "temporal_grounding_tg2_recovery_seed_independence_v1"
    atomic_json(output, result)
    atomic_json(seed_output, seed_result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--min-state-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    result = verify(
        args.repo.resolve(),
        args.output.resolve(),
        args.seed_output.resolve(),
        args.min_state_bytes,
    )
    print(json.dumps(result["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
