import json
from pathlib import Path

import pytest

import verify_temporal_grounding_tg2_recovery_training_v2 as verifier


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    checkpoints = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    staged_root = repo / "logs/resource_scheduler_local/temporal_grounding_tg2r_sidecars"
    canonical_root = repo / "logs/temporal_grounding/tg2r"
    for seed in verifier.SEEDS:
        for arm in verifier.ARMS:
            run_id = f"temporal_grounding_tg2r_{arm}_seed{seed}"
            run = checkpoints / f"stamp+{run_id}"
            run.mkdir(parents=True)
            (run / "config.json").write_text(
                json.dumps({"datasets": {"vla_data": {"in_order": True, "num_workers": 8}}}),
                encoding="utf-8",
            )
            if arm == "raw_milestone" and seed in {1000, 1001}:
                initialization = canonical_root / "initialization" / f"{run_id}.json"
                order = canonical_root / "data_order" / run_id
            else:
                staged = staged_root / run_id
                initialization = staged / "initialization.json"
                order = staged / "data_order"
            initialization.parent.mkdir(parents=True, exist_ok=True)
            initialization.write_text("{}\n", encoding="utf-8")
            order.mkdir(parents=True)
    return repo


def test_build_overlay_accepts_mixed_staged_and_canonical_sidecars(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)

    mapping = verifier.build_overlay(repo, tmp_path / "overlay")

    assert len(mapping["runs"]) == 9
    for seed in (1000, 1001):
        key = f"temporal_grounding_tg2_raw_milestone_seed{seed}"
        assert "/logs/temporal_grounding/tg2r/" in mapping["sidecars"][key]["initialization"]


def test_resolver_rejects_partial_staged_copy_without_fallback(tmp_path: Path) -> None:
    run_id = "temporal_grounding_tg2r_raw_milestone_seed1000"
    canonical = tmp_path / "logs/temporal_grounding/tg2r"
    (canonical / "initialization").mkdir(parents=True)
    (canonical / "initialization" / f"{run_id}.json").write_text("{}\n")
    (canonical / "data_order" / run_id).mkdir(parents=True)
    staged = tmp_path / "logs/resource_scheduler_local/temporal_grounding_tg2r_sidecars" / run_id
    staged.mkdir(parents=True)

    with pytest.raises(ValueError, match="Incomplete staged"):
        verifier.resolve_recovery_sidecars(tmp_path, run_id)
