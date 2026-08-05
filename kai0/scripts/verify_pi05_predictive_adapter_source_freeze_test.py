from __future__ import annotations

import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_pi05_predictive_adapter_source_freeze import SOURCE_PATHS, verify


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_repo(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    sources = {}
    for name, relative in SOURCE_PATHS.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        sources[name] = digest(path)
    base = repo / "kai0/checkpoints/pi05_base/params/_METADATA"
    norm = repo / (
        "kai0/assets/pi05_robotwin_a0_public_exact_bj/"
        "robotwin2.0_absolute_meanstd/norm_stats.json"
    )
    base.parent.mkdir(parents=True, exist_ok=True)
    norm.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("base")
    norm.write_text("norm")
    audit = {
        "decision": "retrain_current_source_a0",
        "matched_recipe": {"passed": True},
        "source_identity": {"current": sources},
        "base_identity": {"current_metadata_sha256": digest(base)},
        "normalization_identity": {"current_sha256": digest(norm)},
    }
    return repo, audit


def test_accepts_exact_freeze(tmp_path: Path) -> None:
    repo, audit = make_repo(tmp_path)
    assert verify(repo, audit)["passed"]


def test_rejects_source_drift(tmp_path: Path) -> None:
    repo, audit = make_repo(tmp_path)
    (repo / SOURCE_PATHS["pi0.py"]).write_text("changed")
    result = verify(repo, audit)
    assert not result["passed"]
    assert not result["source_checks"]["pi0.py"]["match"]
