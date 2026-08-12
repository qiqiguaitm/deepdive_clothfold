import json
from pathlib import Path

import pytest

from verify_temporal_grounding_tg4_bundle import verify


def test_verify_accepts_frozen_six_arm_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "temporal_grounding_tg4_source_decomposition_v1",
                "frozen": True,
                "training": {"seeds": [1100, 1101, 1102]},
                "arms": {arm: {} for arm in (
                    "clean_base",
                    "future_off",
                    "auxiliary_only",
                    "conditioning_only",
                    "parameter_matched_null",
                    "full",
                )},
                "sha256": {
                    "source.txt": "ffb304816a1090313e833215c08dae3d209cfad1ffd1f674f0909a2ae99e1394"
                },
            }
        ),
        encoding="utf-8",
    )
    verify(tmp_path, manifest)


def test_verify_rejects_incomplete_arm_set(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "temporal_grounding_tg4_source_decomposition_v1",
                "frozen": True,
                "training": {"seeds": [1100, 1101, 1102]},
                "arms": {"full": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="arm set"):
        verify(tmp_path, manifest)
