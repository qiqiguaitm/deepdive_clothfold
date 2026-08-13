import json
from pathlib import Path

import pytest

from lmvla.lmwm.scripts import verify_temporal_grounding_tg4_terminal_recovery as verifier


ARM = "parameter_matched_null"
SEED = 1101
RUN_ID = f"temporal_grounding_tg4_{ARM}_seed{SEED}"


def sparse_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def prepare_run(repo: Path) -> tuple[Path, Path]:
    run = repo / "lmvla/lawam/results/Checkpoints/robotwin" / f"run+{RUN_ID}"
    state = run / "checkpoints/steps_20000_state"
    sparse_file(run / "final_model/pytorch_model.pt", 1_000_000_001)
    sparse_file(state / "optimizer.bin", 1_000_000_001)
    (state / "trainer_state.json").write_text('{"steps": 20000}\n')
    (run / "dataset_statistics.json").write_text("{}\n")
    config = {
        "run_id": RUN_ID,
        "seed": SEED,
        "framework": {
            "action_model": {
                "future_prediction": True,
                "enable_loss_distill": True,
            }
        },
        "datasets": {
            "vla_data": {
                "data_mix": "robotwin2_lmwm_all6_v2",
                "per_device_batch_size": 16,
                "num_workers": 8,
                "in_order": True,
            }
        },
        "trainer": {
            "gradient_accumulation_steps": 2,
            "max_train_steps": 20000,
            "save_interval": 20000,
            "ddp_find_unused_parameters": False,
            "pretrained_checkpoint": (
                "results/Checkpoints/pretrain/lawam_pretrain/"
                "final_model/pytorch_model.pt"
            ),
            "load_pretrained_policy_flow": True,
        },
    }
    (run / "config.json").write_text(json.dumps(config))

    initialization = repo / "logs/temporal_grounding/tg4/initialization" / f"{RUN_ID}.json"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": ARM,
                "training_seed": SEED,
                "optimizer_state_entries_before_training": 0,
                "route": {
                    "lawam_future_off": True,
                    "lawam_auxiliary_off": False,
                    "lawam_conditioning_off": False,
                    "milestone_target": False,
                    "dual_route": False,
                },
            }
        )
    )
    order_dir = repo / "logs/temporal_grounding/tg4/data_order" / RUN_ID
    order_dir.mkdir(parents=True)
    for rank in range(4):
        (order_dir / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "arm": ARM,
                    "training_seed": SEED,
                    "rank": rank,
                    "world_size": 4,
                    "microbatches": 40000,
                    "samples": 640000,
                    "sha256": f"rank-{rank}",
                }
            )
        )
    log = (
        repo
        / "logs/temporal_grounding/tg4/entrypoint"
        / f"{ARM}_s{SEED}_east_test.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text(f"{RUN_ID}: 100%\nand that's all\n")
    marker = repo / "ready.ok"
    marker.write_text(
        "ready=2026-08-13T13:00:00Z\n"
        f"run_id={RUN_ID}\n"
        f"entrypoint_log={log}\n"
        "terminal_mode=clean_platform_completion\n"
        "platform_job_id=t-test\n"
    )
    return log, marker


def test_verifier_accepts_bound_clean_platform_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, marker = prepare_run(tmp_path)
    monkeypatch.setattr(verifier, "sha256", lambda _path: "model-hash")

    result = verifier.verify_run(tmp_path, ARM, SEED, "east", marker)

    assert result["recovered_terminal_reason"] == (
        "clean successful East platform terminal state"
    )


def test_verifier_rejects_marker_bound_to_another_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, marker = prepare_run(tmp_path)
    marker.write_text(marker.read_text().replace("platform_job_id=t-test", "platform_job_id="))
    monkeypatch.setattr(verifier, "sha256", lambda _path: "model-hash")

    with pytest.raises(ValueError, match="Incomplete platform completion marker"):
        verifier.verify_run(tmp_path, ARM, SEED, "east", marker)
