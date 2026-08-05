from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path(__file__).resolve().parent
TRAINING_PYTHON = Path("/vePFS/tim/workspace/lerobot-main/.venv/bin/python")
MODEL = Path("/vePFS/tim/hf_models/SidneyXie_pi05_robotwin")


def test_training_runtime_overlay_is_fail_closed_and_reads_batch_weights(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PI05_R4_TRAINING_RUNTIME"] = "1"
    env["PYTHONPATH"] = f"{RUNTIME}:{env.get('PYTHONPATH', '')}"
    output = tmp_path / "runtime.json"
    sidecar = tmp_path / "weights.npz"
    np.savez(sidecar, weight=np.asarray([0.5, 1.5], dtype=np.float32))
    result = subprocess.run(
        [
            str(TRAINING_PYTHON),
            str(RUNTIME / "verify_runtime.py"),
            "--model",
            str(MODEL),
            "--output",
            str(output),
            "--sidecar",
            str(sidecar),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"accepted": true' in output.read_text(encoding="utf-8")
    assert '"type": "sidecar_index"' not in output.read_text(encoding="utf-8")
    assert '"sidecar_probe"' in output.read_text(encoding="utf-8")


def test_runtime_is_inert_without_explicit_opt_in() -> None:
    env = os.environ.copy()
    env.pop("PI05_R4_TRAINING_RUNTIME", None)
    env["PYTHONPATH"] = f"{RUNTIME}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            str(TRAINING_PYTHON),
            "-c",
            (
                "from lerobot.utils import sample_weighting as s; "
                "assert not getattr(s, '_pi05_r4_runtime_installed', False)"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
