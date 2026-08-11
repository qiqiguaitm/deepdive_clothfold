from __future__ import annotations

import json
from pathlib import Path

import pytest

from activate_temporal_grounding_tg1_retry500 import (
    ARCHIVE_SUFFIX,
    FEATURE_ROOT,
    LEGACY_FEATURE_ROOT,
    PROTOCOL,
    ROOT_SPECS,
    activate,
)
from verify_temporal_grounding_tg1_retry500 import verify


CONDITIONS = {
    "TG1A": ["normal", "null", "persistence", "shuffled"],
    "TG1B": [
        "future_off_e36",
        "future_off_e50",
        "local_wm_e36",
        "local_wm_e50",
    ],
}


def prepare_repo(repo: Path) -> Path:
    for relative, count in ROOT_SPECS.items():
        root = repo / relative
        if count:
            for index in range(count):
                summary = root / f"seed{index % 4}/run/tasks/task{index}/summary.json"
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text("{}\n", encoding="utf-8")
    feature = repo / LEGACY_FEATURE_ROOT / "seed0/task0/feature.pt"
    feature.parent.mkdir(parents=True, exist_ok=True)
    feature.write_bytes(b"feature")
    manifest = repo / "amendment.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "frozen": True,
                "operator_authorized": True,
                "experiment_protocol_changed": True,
                "manual_execution_authorized": False,
                "retry_cap": {"new": 500},
                "conditions": CONDITIONS,
                "activation": {"marker": "logs/tg1/activation.json"},
                "file_sha256": {},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_activation_archives_every_partial_root_and_binds_marker(tmp_path: Path) -> None:
    manifest = prepare_repo(tmp_path)

    preflight = activate(tmp_path, manifest, check_only=True)
    assert preflight["activated"] is False
    assert all((tmp_path / relative).exists() for relative, n in ROOT_SPECS.items() if n)

    report = activate(tmp_path, manifest, check_only=False)
    assert report["canonical_roots_empty_after_archive"] is True
    assert all(not (tmp_path / relative).exists() for relative in ROOT_SPECS)
    assert all(
        (tmp_path / relative).with_name(Path(relative).name + ARCHIVE_SUFFIX).exists()
        for relative, count in ROOT_SPECS.items()
        if count
    )
    assert (tmp_path / LEGACY_FEATURE_ROOT).exists()
    assert not (tmp_path / FEATURE_ROOT).exists()
    assert verify(tmp_path, manifest, "TG1A")["retry_cap"] == 500
    assert verify(tmp_path, manifest, "TG1B")["retry_cap"] == 500


def test_activation_rejects_partial_count_drift_without_moving(tmp_path: Path) -> None:
    manifest = prepare_repo(tmp_path)
    missing = next((tmp_path / next(iter(ROOT_SPECS))).glob("seed*/**/summary.json"))
    missing.unlink()

    with pytest.raises(RuntimeError, match="partial-root drift"):
        activate(tmp_path, manifest, check_only=False)

    assert (tmp_path / next(iter(ROOT_SPECS))).exists()
    assert not (tmp_path / "logs/tg1/activation.json").exists()
