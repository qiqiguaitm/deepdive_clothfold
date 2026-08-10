from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_temporal_grounding_tg2_data_order import compare


def write_probe(repo: Path, label: str, rank: int, digest: str) -> Path:
    path = (
        repo
        / "logs/temporal_grounding/tg2/data_order_recovery_probe_v1"
        / label
        / f"rank{rank}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "temporal_grounding_tg2_data_order_recovery_probe_v1",
                "label": label,
                "training_seed": 1000,
                "rank": rank,
                "world_size": 4,
                "microbatches": 256,
                "samples": 4096,
                "num_workers": 8,
                "in_order": True,
                "sha256": digest,
            }
        )
    )
    return path


def test_compare_requires_exact_rank_reproduction(tmp_path: Path) -> None:
    paths = {}
    for label in ("a", "b"):
        for rank in range(4):
            paths[label, rank] = write_probe(tmp_path, label, rank, f"digest-{rank}")

    compare(tmp_path, 256)
    result = json.loads(
        (
            tmp_path
            / "logs/temporal_grounding/tg2/data_order_recovery_probe_v1/matched.json"
        ).read_text()
    )
    assert result["complete"] is True

    payload = json.loads(paths["b", 2].read_text())
    payload["sha256"] = "different"
    paths["b", 2].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="did not reproduce exact rank order"):
        compare(tmp_path, 256)
