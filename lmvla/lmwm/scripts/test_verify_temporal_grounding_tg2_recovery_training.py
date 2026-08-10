import json
from pathlib import Path

import pytest

import verify_temporal_grounding_tg2_recovery_training as verifier


def make_repo(tmp_path: Path, *, in_order: bool = True) -> Path:
    repo = tmp_path / "repo"
    checkpoints = repo / "lmvla/lawam/results/Checkpoints/robotwin"
    sidecars = repo / "logs/resource_scheduler_local/temporal_grounding_tg2r_sidecars"
    for seed in verifier.SEEDS:
        for arm in verifier.ARMS:
            run_id = f"temporal_grounding_tg2r_{arm}_seed{seed}"
            run = checkpoints / f"stamp+{run_id}"
            run.mkdir(parents=True)
            (run / "config.json").write_text(
                json.dumps(
                    {
                        "datasets": {
                            "vla_data": {
                                "in_order": in_order,
                                "num_workers": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (sidecars / run_id).mkdir(parents=True)
    return repo


def test_build_overlay_maps_all_recovery_runs(tmp_path: Path):
    repo = make_repo(tmp_path)
    overlay = tmp_path / "overlay"
    mapping = verifier.build_overlay(repo, overlay)
    assert len(mapping) == 9
    assert all(Path(path).name.startswith("stamp+temporal_grounding_tg2r_") for path in mapping.values())


def test_build_overlay_rejects_non_deterministic_loader(tmp_path: Path):
    repo = make_repo(tmp_path, in_order=False)
    with pytest.raises(ValueError, match="in_order"):
        verifier.build_overlay(repo, tmp_path / "overlay")
