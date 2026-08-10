import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_temporal_grounding_tg2r_sidecars import (  # noqa: E402
    audit_sidecars,
    normalize_sidecars,
)


def fixture(tmp_path: Path, arm: str = "fixed_endpoint", seed: int = 1000):
    initialization = tmp_path / "initialization.json"
    initialization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": None,
                "training_seed": seed,
                "optimizer_state_entries_before_training": 0,
                "route": {
                    "lawam_future_off": arm == "future_off",
                    "milestone_target": "pairs.npz" if arm == "raw_milestone" else None,
                    "require_full_target_coverage": arm == "raw_milestone",
                    "dual_route": False,
                },
            }
        ),
        encoding="utf-8",
    )
    order = tmp_path / "order"
    order.mkdir()
    for rank in range(4):
        (order / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "lawam_exact_data_order_v1",
                    "arm": None,
                    "training_seed": seed,
                    "rank": rank,
                    "world_size": 4,
                    "microbatches": 40000,
                    "samples": 640000,
                    "sha256": f"{rank + 1:064x}",
                }
            ),
            encoding="utf-8",
        )
    return initialization, order


@pytest.mark.parametrize("arm", ["future_off", "fixed_endpoint", "raw_milestone"])
def test_accepts_null_arm_only_with_matching_independent_route(
    tmp_path: Path, arm: str
) -> None:
    initialization, order = fixture(tmp_path, arm=arm)
    result = audit_sidecars(initialization, order, arm, 1000)
    assert result["initialization_arm_recovered"] is True
    assert result["data_order_arm_recovered_ranks"] == [0, 1, 2, 3]


def test_rejects_nonnull_wrong_arm(tmp_path: Path) -> None:
    initialization, order = fixture(tmp_path)
    payload = json.loads(initialization.read_text())
    payload["arm"] = "future_off"
    initialization.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="arm mismatch"):
        audit_sidecars(initialization, order, "fixed_endpoint", 1000)


def test_rejects_route_mismatch(tmp_path: Path) -> None:
    initialization, order = fixture(tmp_path)
    payload = json.loads(initialization.read_text())
    payload["route"]["lawam_future_off"] = True
    initialization.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="future-off route mismatch"):
        audit_sidecars(initialization, order, "fixed_endpoint", 1000)


def test_normalization_preserves_raw_files_and_records_source_hashes(tmp_path: Path) -> None:
    initialization, order = fixture(tmp_path)
    raw_initialization = initialization.read_bytes()
    raw_order = {path.name: path.read_bytes() for path in order.glob("*.json")}
    normalized_initialization = tmp_path / "normalized" / "initialization.json"
    normalized_order = tmp_path / "normalized" / "data_order"
    result = normalize_sidecars(
        initialization,
        order,
        normalized_initialization,
        normalized_order,
        "fixed_endpoint",
        1000,
    )
    assert initialization.read_bytes() == raw_initialization
    assert {path.name: path.read_bytes() for path in order.glob("*.json")} == raw_order
    assert json.loads(normalized_initialization.read_text())["arm"] == "fixed_endpoint"
    assert all(
        json.loads(path.read_text())["arm"] == "fixed_endpoint"
        for path in normalized_order.glob("rank*.json")
    )
    assert result["normalized_initialization_sha256"]
