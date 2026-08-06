from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "train_scripts/kai/lib/checkpoint_resume.sh"


def _resume_args(checkpoint_root: Path) -> tuple[str, str]:
    script = f"""
set -euo pipefail
source {HELPER}
args=()
checkpoint_resume_args {checkpoint_root} args
printf '%s' "${{args[*]}}"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr


def test_fresh_directory_does_not_resume(tmp_path: Path) -> None:
    stdout, stderr = _resume_args(tmp_path / "fresh")
    assert stdout == ""
    assert stderr == ""


def test_latest_complete_numeric_checkpoint_enables_resume(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for step in (5000, 10000):
        checkpoint = root / str(step)
        checkpoint.mkdir(parents=True)
        (checkpoint / "_CHECKPOINT_METADATA").write_text("complete\n")
    incomplete = root / "15000"
    incomplete.mkdir()

    stdout, stderr = _resume_args(root)

    assert stdout == "--resume"
    assert "checkpoint step 10000" in stderr


def test_non_numeric_directory_is_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "latest"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("complete\n")

    stdout, stderr = _resume_args(checkpoint.parent)

    assert stdout == ""
    assert stderr == ""


def test_formal_launchers_use_resume_helper() -> None:
    for relative in (
        "train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh",
        "train_scripts/kai/run_pi05_r1_train.sh",
    ):
        text = (REPO / relative).read_text()
        assert 'source "$REPO/train_scripts/kai/lib/checkpoint_resume.sh"' in text
        assert 'checkpoint_resume_args "$CHECKPOINT_ROOT" RESUME_ARGS' in text
        assert '"${RESUME_ARGS[@]}"' in text


def test_p1_launcher_allows_audited_external_runtime_and_dataset() -> None:
    text = (
        REPO / "train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
    ).read_text()

    assert "DATA_REPO=${DATA_REPO:-" in text
    assert "PYTHON_BIN=${PYTHON_BIN:-" in text
    assert (
        'exec "$PYTHON_BIN" -u "$TRAIN_SOURCE_REPO/kai0/scripts/'
        'train_pi05_robotwin_confirmatory.py"' in text
    )
    assert '--data-repo "$DATA_REPO"' in text
