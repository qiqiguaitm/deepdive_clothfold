from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import verify_temporal_grounding_tg4_training as verifier


def make_repo(tmp_path: Path, *, collapse_seeds: bool = False) -> Path:
    repo = tmp_path / "repo"
    checkpoint_root = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    init_root = repo / "logs/temporal_grounding/tg4/initialization"
    order_root = repo / "logs/temporal_grounding/tg4/data_order"
    base_config = {
        "framework": {"action_model": {
            "future_prediction": True,
            "enable_loss_distill": True,
            "action_horizon": 50,
        }},
        "datasets": {"vla_data": {
            "data_mix": "robotwin2_lmwm_all6_v2",
            "per_device_batch_size": 16,
            "num_workers": 8,
            "in_order": True,
        }},
        "trainer": {
            "pretrained_checkpoint": "results/Checkpoints/pretrain/lawam_pretrain/final_model/pytorch_model.pt",
            "load_pretrained_policy_flow": True,
            "ddp_find_unused_parameters": False,
            "gradient_accumulation_steps": 2,
            "max_train_steps": 20000,
            "save_interval": 20000,
            "optimizer": {"fused": False},
        },
    }
    for seed in verifier.SEEDS:
        effective_seed = verifier.SEEDS[0] if collapse_seeds else seed
        for arm in verifier.ARMS:
            run_id = f"temporal_grounding_tg4_{arm}_seed{seed}"
            run = checkpoint_root / f"stamp+{run_id}"
            state = run / "checkpoints/steps_20000_state"
            (run / "final_model").mkdir(parents=True)
            state.mkdir(parents=True)
            config = copy.deepcopy(base_config)
            config.update({"seed": seed, "run_id": run_id, "output_dir": str(run)})
            for path, value in verifier.expected_arm_config(arm).items():
                parent = config
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
            (run / "config.json").write_text(json.dumps(config))
            (run / "dataset_statistics.json").write_text('{"same":true}\n')
            (run / "final_model/pytorch_model.pt").write_bytes(b"checkpoint")
            (state / "optimizer.bin").write_bytes(b"optimizer")
            (state / "trainer_state.json").write_text('{"steps":20000}\n')

            init_root.mkdir(parents=True, exist_ok=True)
            route = verifier.EXPECTED_ROUTE[arm]
            payload = f"pretrained-{seed}" if arm != "clean_base" else f"clean-{seed}"
            (init_root / f"{run_id}.json").write_text(json.dumps({
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": arm,
                "training_seed": seed,
                "optimizer_state_entries_before_training": 0,
                "parameter_tree_sha256": "parameter-tree",
                "trainable_tree_sha256": "trainable-tree",
                "optimizer_tree_sha256": "optimizer-tree",
                "initialization_payload_sha256": payload,
                "route": {
                    "lawam_future_off": route[0],
                    "lawam_auxiliary_off": route[1],
                    "lawam_conditioning_off": route[2],
                    "milestone_target": None,
                    "dual_route": False,
                },
            }))
            directory = order_root / run_id
            directory.mkdir(parents=True)
            for rank in range(4):
                (directory / f"rank{rank}.json").write_text(json.dumps({
                    "arm": arm,
                    "training_seed": seed,
                    "rank": rank,
                    "world_size": 4,
                    "microbatches": 40000,
                    "samples": 640000,
                    "sha256": f"seed-{effective_seed}-rank-{rank}",
                }))
    return repo


def test_accepts_complete_matched_matrix(tmp_path: Path) -> None:
    result = verifier.verify(
        make_repo(tmp_path), min_checkpoint_bytes=1, min_optimizer_bytes=1
    )
    assert result["complete"]
    assert len(result["runs"]) == 18
    assert all(result["checks"].values())


def test_conditioning_only_enables_unused_parameter_discovery() -> None:
    expected = verifier.expected_arm_config("conditioning_only")
    assert expected[("trainer", "ddp_find_unused_parameters")] is True
    assert expected[("framework", "action_model", "enable_loss_distill")] is True
    assert verifier.EXPECTED_ROUTE["conditioning_only"] == (False, True, False)
    assert verifier.expected_arm_config("full")[
        ("trainer", "ddp_find_unused_parameters")
    ] is False


def test_parameter_matched_null_preserves_full_config_surface() -> None:
    expected = verifier.expected_arm_config("parameter_matched_null")
    assert expected[("framework", "action_model", "future_prediction")] is True
    assert expected[("framework", "action_model", "enable_loss_distill")] is True
    assert verifier.EXPECTED_ROUTE["parameter_matched_null"] == (True, False, False)


def test_rejects_seed_order_collapse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_order_distinct_across_seeds"):
        verifier.verify(
            make_repo(tmp_path, collapse_seeds=True),
            min_checkpoint_bytes=1,
            min_optimizer_bytes=1,
        )


def test_rejects_route_mismatch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    path = (
        repo / "logs/temporal_grounding/tg4/initialization/"
        "temporal_grounding_tg4_full_seed1100.json"
    )
    payload = json.loads(path.read_text())
    payload["route"]["lawam_conditioning_off"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="route mismatch"):
        verifier.verify(repo, min_checkpoint_bytes=1, min_optimizer_bytes=1)
