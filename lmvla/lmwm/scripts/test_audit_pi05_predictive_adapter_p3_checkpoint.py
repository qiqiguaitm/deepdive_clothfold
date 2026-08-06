from __future__ import annotations

from pathlib import Path

from audit_pi05_predictive_adapter_p3_checkpoint import REQUIRED


def test_contract_covers_params_optimizer_assets_and_atomic_metadata() -> None:
    assert Path("_CHECKPOINT_METADATA").as_posix() in REQUIRED
    assert "params/_METADATA" in REQUIRED
    assert "train_state/_METADATA" in REQUIRED
    assert "assets/robotwin2.0_absolute_meanstd/norm_stats.json" in REQUIRED
