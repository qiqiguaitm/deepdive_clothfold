from __future__ import annotations

import numpy as np

from build_pi05_r2_causal_readout import (
    boundary_proximity,
    phase_boundary_anchors,
    roc_auc,
    stable_sample,
)


def test_roc_auc_handles_ties() -> None:
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.0, 0.5, 0.5, 1.0])
    assert roc_auc(labels, scores) == 0.875


def test_boundary_anchors_are_medians_by_phase() -> None:
    progress = np.asarray([0.1, 0.3, 0.4, 0.7, 0.8])
    phase = np.asarray([0, 1, 1, 2, 2])
    boundary = np.asarray([False, True, True, True, False])
    assert np.allclose(phase_boundary_anchors(progress, phase, boundary), [0.35, 0.7])


def test_boundary_proximity_uses_only_next_anchor() -> None:
    result = boundary_proximity(np.asarray([0.30, 0.49, 0.60]), np.asarray([0.50]))
    assert result[0] < result[1]
    assert result[2] == 0.0


def test_stable_sample_is_deterministic_and_bounded() -> None:
    rows = np.arange(100)
    assert np.array_equal(stable_sample(rows, 10, task="a"), stable_sample(rows, 10, task="a"))
    assert len(stable_sample(rows, 10, task="a")) == 10
