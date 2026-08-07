import json
from pathlib import Path

import pytest

from lmvla.lmwm.scripts.verify_temporal_grounding_tg2_sidecars import (
    audit_sidecars,
    resolve_sidecars,
)


def _fixture(tmp_path: Path, arm: str = "raw_milestone", seed: int = 1001):
    initialization = tmp_path / "initialization.json"
    initialization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "lawam_matched_initialization_v1",
                "arm": arm,
                "training_seed": seed,
                "optimizer_state_entries_before_training": 0,
                "route": {
                    "lawam_future_off": arm == "future_off",
                    "milestone_target": "pairs.npz" if arm == "raw_milestone" else None,
                    "require_full_target_coverage": arm == "raw_milestone",
                    "dual_route": False,
                },
            }
        )
        + "\n",
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
                    "arm": arm,
                    "training_seed": seed,
                    "rank": rank,
                    "world_size": 4,
                    "microbatches": 20000,
                    "samples": 320000,
                    "sha256": f"{rank + 1:064x}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return initialization, order


@pytest.mark.parametrize("arm", ["future_off", "fixed_endpoint", "raw_milestone"])
def test_audit_accepts_complete_matched_sidecars(tmp_path: Path, arm: str) -> None:
    initialization, order = _fixture(tmp_path, arm=arm)

    result = audit_sidecars(initialization, order, arm, 1001)

    assert result["complete"] is True
    assert len(result["data_order_sha256_by_rank"]) == 4


def test_audit_rejects_missing_rank(tmp_path: Path) -> None:
    initialization, order = _fixture(tmp_path)
    (order / "rank3.json").unlink()

    with pytest.raises(ValueError, match="Expected four"):
        audit_sidecars(initialization, order, "raw_milestone", 1001)


def test_audit_rejects_route_mismatch(tmp_path: Path) -> None:
    initialization, order = _fixture(tmp_path)
    payload = json.loads(initialization.read_text(encoding="utf-8"))
    payload["route"]["dual_route"] = True
    initialization.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dual route"):
        audit_sidecars(initialization, order, "raw_milestone", 1001)


def test_audit_rejects_rank_identity_mismatch(tmp_path: Path) -> None:
    initialization, order = _fixture(tmp_path)
    path = order / "rank2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training_seed"] = 1002
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rank 2 mismatch"):
        audit_sidecars(initialization, order, "raw_milestone", 1001)


def test_resolve_sidecars_prefers_complete_materialized_copy(tmp_path: Path) -> None:
    run_id = "temporal_grounding_tg2_future_off_seed1000"
    canonical_init = tmp_path / "logs/temporal_grounding/tg2/initialization" / f"{run_id}.json"
    canonical_init.parent.mkdir(parents=True)
    canonical_init.write_text("stale", encoding="utf-8")
    staged = (
        tmp_path
        / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
        / run_id
    )
    (staged / "data_order").mkdir(parents=True)
    (staged / "initialization.json").write_text("materialized", encoding="utf-8")

    initialization, order = resolve_sidecars(tmp_path, run_id)

    assert initialization == staged / "initialization.json"
    assert order == staged / "data_order"


def test_resolve_sidecars_rejects_partial_materialized_copy(tmp_path: Path) -> None:
    run_id = "temporal_grounding_tg2_future_off_seed1000"
    staged = (
        tmp_path
        / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
        / run_id
    )
    staged.mkdir(parents=True)

    with pytest.raises(ValueError, match="Incomplete staged"):
        resolve_sidecars(tmp_path, run_id)
