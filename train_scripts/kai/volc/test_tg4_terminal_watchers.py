import os
from pathlib import Path
import subprocess
import json


SCRIPT = Path(__file__).parents[1] / "recover_temporal_grounding_tg4_east_terminal.sh"
RUN_ID = "temporal_grounding_tg4_parameter_matched_null_seed1101"
EXPECTED_ERROR = "line 118: el.future_action_window_size=49: command not found"


def prepare_artifacts(repo: Path, log_text: str) -> Path:
    run = (
        repo
        / "lmvla/lawam/results/Checkpoints/robotwin"
        / f"run+{RUN_ID}"
    )
    checkpoint = run / "final_model/pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    state = run / "checkpoints/steps_20000_state"
    state.mkdir(parents=True)
    (state / "optimizer.bin").write_bytes(b"optimizer")
    (state / "trainer_state.json").write_text('{"steps": 20000}\n')
    log = (
        repo
        / "logs/temporal_grounding/tg4/entrypoint"
        / "parameter_matched_null_s1101_east_test.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text(log_text)
    return repo / "ready.ok"


def write_scheduler_state(repo: Path, state: str) -> None:
    path = repo / "logs/resource_scheduler_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    f"{RUN_ID}_train": {
                        "status": "completed" if state == "Completed" else "running",
                        "attempts": [
                            {
                                "kind": "platform",
                                "resource": "Robot-East-H20",
                                "last_state": state,
                                "job_id": "t-test",
                            }
                        ],
                    }
                }
            }
        )
    )


def run_watcher(repo: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REPO": str(repo),
            "TG4_ARM": "parameter_matched_null",
            "TG4_TRAIN_SEED": "1101",
            "TG4_READY_OUTPUT": str(output),
            "TG4_RECOVERY_TIMEOUT_SECONDS": "0",
            "TG4_RECOVERY_POLL_SECONDS": "0",
        },
    )


def test_east_terminal_watcher_rejects_incomplete_evidence_at_timeout(
    tmp_path: Path,
) -> None:
    output = prepare_artifacts(tmp_path, f"{RUN_ID}: 100%\nand that's all\n")

    completed = run_watcher(tmp_path, output)

    assert completed.returncode != 0
    assert "exact complete terminal evidence" in completed.stderr
    assert not output.exists()


def test_east_terminal_watcher_publishes_only_exact_complete_evidence(
    tmp_path: Path,
) -> None:
    output = prepare_artifacts(
        tmp_path,
        f"{RUN_ID}: 100%\nand that's all\n{EXPECTED_ERROR}\n",
    )

    completed = run_watcher(tmp_path, output)

    assert completed.returncode == 0, completed.stderr
    marker = output.read_text()
    assert f"run_id={RUN_ID}" in marker
    assert "parameter_matched_null_s1101_east_test.log" in marker
    assert "terminal_mode=validated_post_training_shell_error" in marker


def test_east_terminal_watcher_accepts_clean_platform_completion(
    tmp_path: Path,
) -> None:
    output = prepare_artifacts(tmp_path, f"{RUN_ID}: 100%\nand that's all\n")
    write_scheduler_state(tmp_path, "Completed")

    completed = run_watcher(tmp_path, output)

    assert completed.returncode == 0, completed.stderr
    marker = output.read_text()
    assert "terminal_mode=clean_platform_completion" in marker
    assert "platform_job_id=t-test" in marker


def test_east_terminal_watcher_rejects_clean_log_while_platform_running(
    tmp_path: Path,
) -> None:
    output = prepare_artifacts(tmp_path, f"{RUN_ID}: 100%\nand that's all\n")
    write_scheduler_state(tmp_path, "Running")

    completed = run_watcher(tmp_path, output)

    assert completed.returncode != 0
    assert not output.exists()
