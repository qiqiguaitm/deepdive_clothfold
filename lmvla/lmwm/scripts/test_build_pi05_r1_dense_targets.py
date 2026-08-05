from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from build_pi05_r1_dense_targets import build


def write_reference(path: Path, frames: np.ndarray | None = None) -> None:
    episode = np.repeat(np.asarray([10, 11], dtype=np.int32), 6)
    frame = np.tile(np.arange(6, dtype=np.int32), 2) if frames is None else frames
    progress = np.tile(np.linspace(0.0, 1.0, 6, dtype=np.float32), 2)
    phase = np.tile(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int16), 2)
    np.savez_compressed(
        path,
        episode=episode,
        frame=frame,
        physical_task=np.repeat(np.asarray([0, 1], dtype=np.int16), 6),
        recurrence_density=np.full(12, 0.5, dtype=np.float32),
        progress=progress,
        phase_boundary=phase.astype(bool),
        phase=phase,
    )


def test_builds_every_valid_fixed_horizon_row(tmp_path: Path) -> None:
    source = tmp_path / "reference.npz"
    output = tmp_path / "dense.npz"
    manifest = tmp_path / "dense.json"
    write_reference(source)
    result = build(source, output, manifest, horizon=2)
    dense = np.load(output)
    assert result["episode_count"] == 2
    assert result["target_rows"] == 8
    np.testing.assert_array_equal(dense["cur_fi"], [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(dense["tgt_fi"], [2, 3, 4, 5, 2, 3, 4, 5])
    np.testing.assert_allclose(dense["progress_change"], 0.4)
    np.testing.assert_array_equal(
        dense["phase_boundary_crossing"],
        [False, True, True, False, False, True, True, False],
    )
    assert json.loads(manifest.read_text())["dense_targets_sha256"]


def test_rejects_noncontiguous_reference_frames(tmp_path: Path) -> None:
    source = tmp_path / "reference.npz"
    frames = np.tile(np.arange(6, dtype=np.int32), 2)
    frames[3] = 9
    write_reference(source, frames)
    with pytest.raises(ValueError, match="not contiguous"):
        build(source, tmp_path / "dense.npz", tmp_path / "dense.json", horizon=2)
