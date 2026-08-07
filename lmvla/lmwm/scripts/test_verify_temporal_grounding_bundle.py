import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from lmvla.lmwm.scripts.verify_temporal_grounding_bundle import verify


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    inner = repo / "lmvla/lawam"
    inner.mkdir(parents=True)
    _run(repo, "init")
    _run(repo, "config", "user.email", "test@example.com")
    _run(repo, "config", "user.name", "Test")
    (repo / "source.txt").write_text("fixed\n", encoding="utf-8")
    _run(repo, "add", "source.txt")
    _run(repo, "commit", "-m", "outer")
    outer_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    _run(inner, "init")
    _run(inner, "config", "user.email", "test@example.com")
    _run(inner, "config", "user.name", "Test")
    (inner / "inner.txt").write_text("inner\n", encoding="utf-8")
    _run(inner, "add", "inner.txt")
    _run(inner, "commit", "-m", "inner")
    inner_commit = subprocess.check_output(
        ["git", "-C", str(inner), "rev-parse", "HEAD"], text=True
    ).strip()

    manifest = repo / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "bundle": "TG1A",
                "frozen": True,
                "manual_execution_authorized": False,
                "source": {
                    "outer_implementation_commit": outer_commit,
                    "lawam_commit": inner_commit,
                },
                "file_sha256": {
                    "source.txt": hashlib.sha256(b"fixed\n").hexdigest(),
                    "lmvla/lawam/inner.txt": hashlib.sha256(b"inner\n").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    return repo, manifest


def test_verify_accepts_frozen_tree(tmp_path: Path) -> None:
    repo, manifest = _fixture(tmp_path)
    result = verify(repo, manifest, "TG1A")
    assert result["verified_files"] == 2


def test_verify_rejects_mutated_file(tmp_path: Path) -> None:
    repo, manifest = _fixture(tmp_path)
    (repo / "source.txt").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(repo, manifest, "TG1A")
