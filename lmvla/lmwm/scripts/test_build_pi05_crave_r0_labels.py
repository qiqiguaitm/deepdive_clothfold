from __future__ import annotations

import numpy as np

from build_pi05_crave_r0_labels import (
    episode_boundaries,
    fit_reference_sigma,
    query_recurrence_fields,
)


def test_query_fields_follow_cross_episode_reference_progress() -> None:
    references = [
        np.asarray([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.99, 0.1], [0.6, 0.8], [0.1, 0.99]], dtype=np.float32),
    ]
    sigma, reference, ranges, _, _, _ = fit_reference_sigma(
        references, device="cpu", chunk_size=2
    )
    density, progress = query_recurrence_fields(
        references[0], reference, ranges, sigma, device="cpu", chunk_size=2
    )
    assert np.isfinite(density).all()
    assert np.all(np.diff(progress) > 0)
    assert progress[0] < 0.1
    assert progress[-1] > 0.9


def test_density_valley_becomes_phase_boundary() -> None:
    density = np.asarray([0.9, 0.8, 0.2, 0.8, 0.9], dtype=np.float32)
    boundaries = episode_boundaries(density)
    np.testing.assert_array_equal(boundaries, [2])
