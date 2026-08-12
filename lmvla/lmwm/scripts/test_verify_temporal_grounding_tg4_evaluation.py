import hashlib
import json
from pathlib import Path

import pytest

from verify_temporal_grounding_tg4_evaluation import verify


def make_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "temporal_grounding_tg4_evaluation_v1",
                "frozen": True,
                "panels": {
                    "normal": {"training_seeds": [1100, 1101, 1102]},
                    "shuffled": {"arms": ["full"]},
                },
                "runtime": {"evaluation_seeds": [0, 1, 2, 3]},
                "analysis": {"bootstrap_samples": 20000},
                "sha256": {"source.txt": hashlib.sha256(b"frozen").hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_accepts_frozen_evaluation_bundle(tmp_path: Path) -> None:
    verify(tmp_path, make_manifest(tmp_path))


def test_rejects_non_full_shuffled_panel(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["panels"]["shuffled"]["arms"] = ["full", "future_off"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="full-only"):
        verify(tmp_path, manifest)
