from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_pi05_r4_train_config import OFFICIAL_GLOBAL_BATCH, build_config


def public_recipe(tmp_path: Path) -> Path:
    source = Path("/vePFS/tim/hf_models/SidneyXie_pi05_robotwin/train_config.json")
    path = tmp_path / "train_config.json"
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("arm", "weight_type"),
    [("ordinary", None), ("terminal_outcome", "batch_field"), ("outcome_free_crave", "sidecar_index")],
)
def test_builds_matched_arm_configs(tmp_path: Path, arm: str, weight_type: str | None) -> None:
    sidecar = tmp_path / "weights.npz"
    sidecar.write_bytes(b"test")
    config = build_config(
        public_recipe(tmp_path),
        arm,
        world_size=4,
        steps=5_000,
        output_dir=tmp_path / arm,
        dataset_root=tmp_path / "dataset",
        model_path=tmp_path / "model",
        sidecar=sidecar,
    )
    assert config["batch_size"] * 4 == OFFICIAL_GLOBAL_BATCH
    assert config["steps"] == 5_000
    assert config["seed"] == 1_000
    assert config["optimizer"]["lr"] == 2.5e-5
    assert config["policy"]["freeze_vision_encoder"] is False
    assert config["policy"]["compile_model"] is True
    assert config["policy"]["push_to_hub"] is False
    assert config["wandb"]["enable"] is False
    assert (config["sample_weighting"] or {}).get("type") == weight_type


def test_rejects_world_size_that_changes_global_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must divide"):
        build_config(
            public_recipe(tmp_path),
            "ordinary",
            world_size=3,
            steps=5_000,
            output_dir=tmp_path / "out",
            dataset_root=tmp_path / "dataset",
            model_path=tmp_path / "model",
            sidecar=tmp_path / "missing.npz",
        )


def test_rejects_public_recipe_drift(tmp_path: Path) -> None:
    path = public_recipe(tmp_path)
    payload = json.loads(path.read_text())
    payload["optimizer"]["lr"] = 1e-4
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="recipe drifted"):
        build_config(
            path,
            "ordinary",
            world_size=4,
            steps=5_000,
            output_dir=tmp_path / "out",
            dataset_root=tmp_path / "dataset",
            model_path=tmp_path / "model",
            sidecar=tmp_path / "missing.npz",
        )
