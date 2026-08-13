from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify_temporal_grounding_tg4_eval_results.py")


def fixture(tmp_path: Path, *, eval_seed: int = 0, count: int = 2, successes: list[object] | None = None) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"eval_seeds": {"0": {"task": [10, 11]}}}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if successes is None:
        successes = [True, False]
    summary = tmp_path / "results/seed0/run/tasks/task/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "task_name": "task",
        "fixed_seed_manifest": {"sha256": digest, "eval_seed": eval_seed},
        "n_episodes": count,
        "episodes": [
            {"seed": seed, "success": success}
            for seed, success in zip([10, 11], successes, strict=True)
        ],
    }))
    return manifest, tmp_path / "results"


def run(manifest: Path, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--root", str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_accepts_complete_typed_summary(tmp_path: Path) -> None:
    manifest, root = fixture(tmp_path)
    result = run(manifest, root)
    assert result.returncode == 0, result.stderr


def test_rejects_wrong_eval_seed(tmp_path: Path) -> None:
    manifest, root = fixture(tmp_path, eval_seed=1)
    result = run(manifest, root)
    assert result.returncode != 0
    assert "eval_seed mismatch" in result.stderr


def test_rejects_wrong_episode_count(tmp_path: Path) -> None:
    manifest, root = fixture(tmp_path, count=1)
    result = run(manifest, root)
    assert result.returncode != 0
    assert "n_episodes does not match" in result.stderr


def test_rejects_non_boolean_success(tmp_path: Path) -> None:
    manifest, root = fixture(tmp_path, successes=[True, 1])
    result = run(manifest, root)
    assert result.returncode != 0
    assert "invalid success" in result.stderr
