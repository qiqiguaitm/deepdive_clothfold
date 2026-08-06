from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pi05_predictive_adapter_p0 import paired_episode_bootstrap, resolve_norm_assets_dir


def test_paired_episode_bootstrap_uses_episode_means():
    episodes = np.asarray([1, 1, 1, 2], dtype=np.int32)
    normal = np.asarray([1.0, 1.0, 1.0, 0.8], dtype=np.float32)
    control = np.asarray([0.5, 0.5, 0.5, 0.4], dtype=np.float32)
    result = paired_episode_bootstrap(episodes, normal, control, draws=2_000, seed=5)
    assert result["episode_count"] == 2
    assert np.isclose(result["mean_difference"], 0.45)
    assert result["ci95_low"] > 0.0


def test_norm_assets_default_to_checkpoint_and_allow_explicit_override(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint_norm = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    checkpoint_norm.parent.mkdir(parents=True)
    checkpoint_norm.write_text("{}")
    assert resolve_norm_assets_dir(checkpoint, None) == checkpoint / "assets"

    override = tmp_path / "override"
    override_norm = override / "robotwin2.0_absolute_meanstd/norm_stats.json"
    override_norm.parent.mkdir(parents=True)
    override_norm.write_text("{}")
    assert resolve_norm_assets_dir(checkpoint, override) == override


def test_norm_assets_missing_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="normalization statistics missing"):
        resolve_norm_assets_dir(tmp_path / "checkpoint", None)
