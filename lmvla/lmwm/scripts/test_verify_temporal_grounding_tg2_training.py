from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
SEEDS = (1000, 1001, 1002)


def test_tg2_training_integrity_verifier(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "lmvla/lawam/results/Checkpoints/robotwin"
    init_root = tmp_path / "logs/temporal_grounding/tg2/initialization"
    order_root = tmp_path / "logs/temporal_grounding/tg2/data_order"
    init_root.mkdir(parents=True)
    for seed in SEEDS:
        for arm in ARMS:
            run_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
            run = checkpoint_root / f"stamp+{run_id}"
            state = run / "checkpoints/steps_20000_state"
            (run / "final_model").mkdir(parents=True)
            state.mkdir(parents=True)
            config = {
                "framework": {"action_model": {
                    "future_prediction": True,
                    "enable_loss_distill": True,
                    "future_action_window_size": 49,
                    "action_horizon": 50,
                    "flow_cfg": {"horizon_sec": 1.0},
                }},
                "datasets": {"vla_data": {
                    "data_mix": "robotwin2_lmwm_all6_v2",
                    "sec_chunk": 1.0,
                    "per_device_batch_size": 16,
                }},
                "trainer": {"gradient_accumulation_steps": 2, "max_train_steps": 20000},
            }
            (run / "config.yaml").write_text(yaml.safe_dump(config))
            (run / "dataset_statistics.json").write_text('{"same":true}\n')
            (run / "final_model/pytorch_model.pt").write_bytes(b"checkpoint")
            (state / "optimizer.bin").write_bytes(b"optimizer")
            (state / "trainer_state.json").write_text('{"steps":20000}\n')
            route = {
                "lawam_future_off": arm == "future_off",
                "milestone_target": "pairs.npz" if arm == "raw_milestone" else None,
                "milestone_target_compact": "target.npz" if arm == "raw_milestone" else None,
                "require_full_target_coverage": arm == "raw_milestone",
                "dual_route": False,
            }
            initialization = {
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": arm,
                "training_seed": seed,
                "parameter_tree_sha256": "parameter-tree",
                "trainable_tree_sha256": "trainable-tree",
                "initialization_payload_sha256": f"payload-seed-{seed}",
                "optimizer_tree_sha256": "optimizer-tree",
                "optimizer_state_entries_before_training": 0,
                "route": route,
            }
            (init_root / f"{run_id}.json").write_text(json.dumps(initialization))
            audit_dir = order_root / run_id
            audit_dir.mkdir(parents=True)
            for rank in range(4):
                (audit_dir / f"rank{rank}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "protocol": "lawam_exact_data_order_v1",
                            "arm": arm,
                            "training_seed": seed,
                            "rank": rank,
                            "world_size": 4,
                            "microbatches": 20000,
                            "samples": 320000,
                            "sha256": f"{seed * 10 + rank:064x}",
                        }
                    )
                )

    staged_run = "temporal_grounding_tg2_fixed_endpoint_seed1000"
    staged = (
        tmp_path
        / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
        / staged_run
    )
    staged.mkdir(parents=True)
    shutil.move(init_root / f"{staged_run}.json", staged / "initialization.json")
    shutil.move(order_root / staged_run, staged / "data_order")

    output = tmp_path / "integrity.json"
    script = Path(__file__).with_name("verify_temporal_grounding_tg2_training_v2.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo",
            str(tmp_path),
            "--output",
            str(output),
            "--min-state-bytes",
            "1",
        ],
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["complete"] is True
    assert all(result["checks"].values())
    assert result["protocol"] == "temporal_grounding_tg2_training_integrity_v2"
    assert staged_run in result["sidecar_audits"]
