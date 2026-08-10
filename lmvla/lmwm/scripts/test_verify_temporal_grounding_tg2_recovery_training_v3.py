import json
from pathlib import Path

import verify_temporal_grounding_tg2_recovery_training_v3 as verifier


def write_sidecars(initialization: Path, order: Path, arm: str, seed: int) -> None:
    initialization.parent.mkdir(parents=True, exist_ok=True)
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
    order.mkdir(parents=True)
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
                    "sha256": f"{seed + rank:064x}",
                }
            ),
            encoding="utf-8",
        )


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
                json.dumps(
                    {"datasets": {"vla_data": {"in_order": True, "num_workers": 8}}}
                ),
                encoding="utf-8",
            )
            if arm == "raw_milestone" and seed in {1000, 1001}:
                initialization = canonical_root / "initialization" / f"{run_id}.json"
                order = canonical_root / "data_order" / run_id
            else:
                root = staged_root / run_id
                initialization = root / "initialization.json"
                order = root / "data_order"
            write_sidecars(initialization, order, arm, seed)
    return repo


def test_build_overlay_normalizes_null_arm_without_mutating_sources(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    source = (
        repo
        / "logs/temporal_grounding/tg2r/initialization"
        / "temporal_grounding_tg2r_raw_milestone_seed1000.json"
    )
    before = source.read_bytes()
    overlay = tmp_path / "overlay"
    result = verifier.build_overlay(repo, overlay)
    normalized = (
        overlay
        / "logs/resource_scheduler_local/temporal_grounding_tg2_sidecars"
        / "temporal_grounding_tg2_raw_milestone_seed1000/initialization.json"
    )
    assert source.read_bytes() == before
    assert json.loads(normalized.read_text())["arm"] == "raw_milestone"
    assert len(result["metadata_recovery"]) == 9
    assert all(
        row["initialization_arm_recovered"]
        for row in result["metadata_recovery"].values()
    )
