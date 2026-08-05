from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("build_pi05_r4_support_scene_manifest.py")
SPEC = importlib.util.spec_from_file_location("build_pi05_r4_support_scene_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(values0: list[int], values1: list[int]) -> dict:
    return {"eval_seeds": {"0": {"task": values0}, "1": {"task": values1}}}


def test_build_selects_every_unused_train_scene() -> None:
    result = MODULE.build(
        payload([10, 11, 12], [20, 21, 22]),
        payload([10], [20]),
        task="task",
    )

    assert result["episodes_per_cell"] == 2
    assert result["split_by_eval_seed"] == {"train": [0, 1], "eval": []}
    assert result["eval_seeds"] == {
        "0": {"task": [11, 12]},
        "1": {"task": [21, 22]},
    }
    assert "all source scenes" in result["selection_rule"]


def test_build_rejects_unequal_supplement_sizes() -> None:
    with pytest.raises(ValueError, match="cell sizes differ"):
        MODULE.build(
            payload([10, 11], [20, 21, 22]),
            payload([10], [20]),
            task="task",
        )


def test_build_rejects_non_subset_base() -> None:
    with pytest.raises(ValueError, match="not a subset"):
        MODULE.build(
            payload([10, 11], [20, 21]),
            payload([99], [20]),
            task="task",
        )
