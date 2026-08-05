from __future__ import annotations

import numpy as np

from analyze_pi05_crave_r0_probe import (
    paired_episode_bootstrap,
    project_features,
    time_features,
)


def test_projection_is_shared_across_action_conditions() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20, 12)).astype(np.float32)
    same = train[:5].copy()
    projected_train, projected = project_features(
        train, {"a": same, "b": same}, seed=7
    )
    assert projected_train.shape == (20, 256)
    np.testing.assert_array_equal(projected["a"], projected["b"])


def test_time_features_are_task_specific_degree_five() -> None:
    labels = {
        "normalized_current_time": np.asarray([0.5, 0.5]),
        "physical_task": np.asarray([0, 1]),
    }
    features = time_features(labels, 2)
    assert features.shape == (2, 12)
    assert np.count_nonzero(features[0, 6:]) == 0
    assert np.count_nonzero(features[1, :6]) == 0


def test_episode_bootstrap_detects_lower_normal_loss() -> None:
    result = paired_episode_bootstrap(
        np.repeat(np.arange(20), 2),
        np.zeros(40),
        np.ones(40),
        draws=2000,
        seed=3,
    )
    assert result["ci95_low"] > 0
