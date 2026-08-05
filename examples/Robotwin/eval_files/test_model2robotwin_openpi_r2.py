from __future__ import annotations

import json

import numpy as np

from examples.Robotwin.eval_files.model2robotwin_openpi_r2 import load_task_readout


def test_load_task_readout_requires_accepted_manifest(tmp_path) -> None:
    readout_path = tmp_path / "readout.npz"
    np.savez_compressed(readout_path, placeholder=np.asarray(1))
    (tmp_path / "readout_manifest.json").write_text(
        json.dumps({"acceptance": {"accepted": False}, "tasks": {}})
    )
    try:
        load_task_readout(readout_path, "task")
    except RuntimeError as error:
        assert "not accepted" in str(error)
    else:
        raise AssertionError("rejected readout was loaded")


def test_load_task_readout_roundtrip(tmp_path) -> None:
    readout_path = tmp_path / "readout.npz"
    np.savez_compressed(
        readout_path,
        task0_features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
        task0_episode_offsets=np.asarray([0, 1, 2], dtype=np.int32),
        task0_progress=np.asarray([0.0, 1.0], dtype=np.float32),
        task0_density=np.asarray([1.0, 1.0], dtype=np.float32),
        task0_sigma=np.asarray(0.5, dtype=np.float32),
        task0_density_calibration=np.asarray([0.1, 0.9], dtype=np.float32),
        task0_boundary_progress=np.asarray([0.5], dtype=np.float32),
    )
    (tmp_path / "readout_manifest.json").write_text(
        json.dumps(
            {
                "acceptance": {"accepted": True},
                "tasks": {"task": {"task_id": 0}},
            }
        )
    )
    output = load_task_readout(readout_path, "task").query(np.asarray([1.0, 0.0]))
    assert output["progress"] < 0.5
