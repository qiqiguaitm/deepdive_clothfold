from __future__ import annotations

import pytest

from build_pi05_r4_scene_manifest import build


def source() -> dict:
    return {
        "protocol": "source",
        "eval_seeds": {
            str(seed): {
                "task": [seed * 100 + index for index in range(5)]
            }
            for seed in range(4)
        },
    }


def test_build_creates_disjoint_split() -> None:
    result = build(source(), 3)
    assert result["split_by_eval_seed"] == {"train": [0, 1], "eval": [2, 3]}
    assert all(
        len(tasks["task"]) == 3 for tasks in result["eval_seeds"].values()
    )


def test_build_rejects_scene_reuse() -> None:
    payload = source()
    payload["eval_seeds"]["1"]["task"][0] = 0
    with pytest.raises(ValueError, match="scene reused"):
        build(payload, 3)
