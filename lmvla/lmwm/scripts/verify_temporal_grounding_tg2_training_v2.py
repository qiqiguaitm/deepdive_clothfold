#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from .verify_temporal_grounding_tg2_sidecars import (
        ARMS,
        SEEDS,
        audit_sidecars,
        resolve_sidecars,
    )
except ImportError:
    from verify_temporal_grounding_tg2_sidecars import (
        ARMS,
        SEEDS,
        audit_sidecars,
        resolve_sidecars,
    )


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-state-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args()
    repo = args.repo.resolve()
    base_verifier = Path(__file__).with_name(
        "verify_temporal_grounding_tg2_training.py"
    )
    sidecar_audits = {}

    with tempfile.TemporaryDirectory(prefix="tg2-integrity-v2-") as temporary:
        overlay = Path(temporary)
        checkpoint_link = overlay / "lmvla/lawam/results/Checkpoints/robotwin"
        checkpoint_link.parent.mkdir(parents=True)
        checkpoint_link.symlink_to(
            repo / "lmvla/lawam/results/Checkpoints/robotwin", target_is_directory=True
        )
        init_root = overlay / "logs/temporal_grounding/tg2/initialization"
        order_root = overlay / "logs/temporal_grounding/tg2/data_order"
        init_root.mkdir(parents=True)
        order_root.mkdir(parents=True)

        for seed in SEEDS:
            for arm in ARMS:
                run_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
                initialization, order = resolve_sidecars(repo, run_id)
                sidecar_audits[run_id] = audit_sidecars(
                    initialization, order, arm, seed
                )
                (init_root / f"{run_id}.json").symlink_to(initialization)
                (order_root / run_id).symlink_to(order, target_is_directory=True)

        base_output = overlay / "training_integrity_v1.json"
        subprocess.run(
            [
                sys.executable,
                str(base_verifier),
                "--repo",
                str(overlay),
                "--output",
                str(base_output),
                "--min-state-bytes",
                str(args.min_state_bytes),
            ],
            check=True,
        )
        result = json.loads(base_output.read_text(encoding="utf-8"))

    result["protocol"] = "temporal_grounding_tg2_training_integrity_v2"
    result["checks"]["sidecar_identity_and_completeness"] = True
    result["sidecar_audits"] = sidecar_audits
    atomic_write(args.output.resolve(), result)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
