import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "recover_temporal_grounding_tg4_east_terminal.sh"
RUN_ID = "temporal_grounding_tg4_parameter_matched_null_seed1101"
EXPECTED_ERROR = "line 118: el.future_action_window_size=49: command not found"


def prepare_artifacts(repo: Path, log_text: str) -> Path:
    checkpoint = (
        repo
        / "lmvla/lawam/results/Checkpoints/robotwin"
        / f"run+{RUN_ID}"
        / "final_model/pytorch_model.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    log = (
        repo
        / "logs/temporal_grounding/tg4/entrypoint"
        / "parameter_matched_null_s1101_east_test.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text(log_text)
    return repo / "ready.ok"


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
