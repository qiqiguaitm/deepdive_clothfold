from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify_robotwin_fixed_seed_eval.py")


def write_fixture(
    tmp_path: Path,
    *,
    expected_seeds: list[int],
    actual_seeds: list[int],
) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "eval_seeds": {
                    "0": {"test_task": expected_seeds},
                }
            }
        )
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    root = tmp_path / "results" / "seed0" / "run" / "tasks" / "test_task"
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "task_name": "test_task",
                "fixed_seed_manifest": {"sha256": manifest_sha},
                "episodes": [{"seed": seed} for seed in actual_seeds],
            }
        )
    )
    return manifest, tmp_path / "results"


def run_verifier(manifest: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--root",
            str(root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_parallel_completion_order(tmp_path: Path) -> None:
    manifest, root = write_fixture(
        tmp_path,
        expected_seeds=[10, 11, 12, 13],
        actual_seeds=[10, 12, 11, 13],
    )

    result = run_verifier(manifest, root)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["complete"] is True


def test_rejects_duplicate_scene_seed(tmp_path: Path) -> None:
    manifest, root = write_fixture(
        tmp_path,
        expected_seeds=[10, 11, 12, 13],
        actual_seeds=[10, 11, 11, 13],
    )

    result = run_verifier(manifest, root)

    assert result.returncode != 0
    assert "duplicate scene seeds" in result.stderr


def test_rejects_wrong_scene_seed_set(tmp_path: Path) -> None:
    manifest, root = write_fixture(
        tmp_path,
        expected_seeds=[10, 11, 12, 13],
        actual_seeds=[10, 11, 12, 99],
    )

    result = run_verifier(manifest, root)

    assert result.returncode != 0
    assert "scene seeds differ" in result.stderr
