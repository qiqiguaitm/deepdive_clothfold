from __future__ import annotations

import hashlib
from pathlib import Path

from audit_pi05_p1_north_failover_stage import audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> dict:
    artifact = root / "artifact"
    inventory = hashlib.sha256(b"value.bin\t4\n").hexdigest()
    return {
        "protocol": "test",
        "promotion_requirements": ["never launch from a stage audit"],
        "artifacts": [
            {
                "path": "artifact",
                "file_count": 1,
                "bytes": 4,
                "inventory_sha256": inventory,
                "key_checksums": {"value.bin": _sha(artifact / "value.bin")},
            }
        ],
        "control_files": {
            "control.txt": {
                "bytes": 7,
                "sha256": _sha(root / "control.txt"),
            }
        },
    }


def test_complete_stage_passes_without_authorizing_launch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "value.bin").write_bytes(b"data")
    (tmp_path / "control.txt").write_text("control")

    report = audit(_manifest(tmp_path), tmp_path)

    assert report["stage_verified"] is True
    assert report["launch_authorized"] is False


def test_mutated_stage_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "value.bin").write_bytes(b"data")
    (tmp_path / "control.txt").write_text("control")
    manifest = _manifest(tmp_path)
    (artifact / "value.bin").write_bytes(b"changed")

    report = audit(manifest, tmp_path)

    assert report["stage_verified"] is False
    assert report["launch_authorized"] is False


def test_source_tree_uses_explicit_source_path(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    source = tmp_path / "source"
    target = tmp_path / "target"
    artifact.mkdir()
    source.mkdir()
    target.mkdir()
    (artifact / "value.bin").write_bytes(b"data")
    (source / "value.bin").write_bytes(b"data")
    (target / "value.bin").write_bytes(b"wrong")
    (tmp_path / "control.txt").write_text("control")
    manifest = _manifest(tmp_path)
    manifest["artifacts"][0]["path"] = "target"
    manifest["artifacts"][0]["source_path"] = "source"

    source_report = audit(manifest, tmp_path, use_source_paths=True)
    stage_report = audit(manifest, tmp_path)

    assert source_report["stage_verified"] is True
    assert source_report["path_mode"] == "source"
    assert source_report["artifact_checks"][0]["observed_path"] == "source"
    assert stage_report["stage_verified"] is False
    assert stage_report["path_mode"] == "stage"
