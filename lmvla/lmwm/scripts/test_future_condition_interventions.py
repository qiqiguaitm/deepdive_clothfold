from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "lawam/starVLA/model/framework/latent_world/runtime/future_condition_interventions.py"
)
SPEC = importlib.util.spec_from_file_location("future_condition_interventions", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FutureConditionIntervention = MODULE.FutureConditionIntervention
future_off_condition = MODULE.future_off_condition
future_off_zero_tether = MODULE.future_off_zero_tether


def context(*, scene_seed: int, query_index: int = 0) -> dict:
    return {
        "task": "stack_blocks_two",
        "eval_seed": 0,
        "scene_seed": scene_seed,
        "episode_id": scene_seed - 100000,
        "query_index": query_index,
    }


def test_normal_capture_is_behavior_preserving(tmp_path) -> None:
    predicted = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    current = torch.full_like(predicted, -1)
    hook = FutureConditionIntervention(mode="normal", capture_root=tmp_path)

    output = hook.apply(
        predicted=predicted,
        current=current,
        contexts=[context(scene_seed=100000), context(scene_seed=100001)],
    )

    assert output.data_ptr() == predicted.data_ptr()
    captured = np.load(
        tmp_path
        / "stack_blocks_two"
        / "eval_seed_0"
        / "scene_seed_100001"
        / "query_000000.npy"
    )
    np.testing.assert_array_equal(captured, predicted[1].numpy())


def test_null_and_rms_matched_persistence() -> None:
    predicted = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
    current = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    contexts = [context(scene_seed=100000)]

    null = FutureConditionIntervention(mode="null").apply(
        predicted=predicted, current=current, contexts=contexts
    )
    persistence = FutureConditionIntervention(mode="persistence").apply(
        predicted=predicted, current=current, contexts=contexts
    )

    assert torch.count_nonzero(null) == 0
    assert persistence.shape == predicted.shape
    torch.testing.assert_close(
        persistence.float().square().mean().sqrt(),
        predicted.float().square().mean().sqrt(),
    )


def test_shuffled_uses_prespecified_different_scene_and_frozen_modulo(tmp_path) -> None:
    capture = FutureConditionIntervention(mode="normal", capture_root=tmp_path)
    current = torch.zeros((1, 2, 3))
    source_values = [torch.full((1, 2, 3), 11.0), torch.full((1, 2, 3), 12.0)]
    for query_index, value in enumerate(source_values):
        capture.apply(
            predicted=value,
            current=current,
            contexts=[context(scene_seed=100001, query_index=query_index)],
        )

    manifest = tmp_path / "shuffle.json"
    manifest.write_text(
        json.dumps(
            {
                "mapping": {
                    "stack_blocks_two": {"0": {"100000": 100001, "100001": 100000}}
                }
            }
        ),
        encoding="utf-8",
    )
    shuffled = FutureConditionIntervention(
        mode="shuffled", capture_root=tmp_path, shuffle_manifest=manifest
    )
    output = shuffled.apply(
        predicted=torch.zeros((1, 2, 3)),
        current=current,
        contexts=[context(scene_seed=100000, query_index=3)],
    )
    torch.testing.assert_close(output, source_values[1])


def test_shuffled_rejects_self_match(tmp_path) -> None:
    manifest = tmp_path / "shuffle.json"
    manifest.write_text(
        json.dumps(
            {"mapping": {"stack_blocks_two": {"0": {"100000": 100000}}}}
        ),
        encoding="utf-8",
    )
    hook = FutureConditionIntervention(
        mode="shuffled", capture_root=tmp_path, shuffle_manifest=manifest
    )
    with pytest.raises(ValueError, match="self-match"):
        hook.apply(
            predicted=torch.zeros((1, 2, 3)),
            current=torch.zeros((1, 2, 3)),
            contexts=[context(scene_seed=100000)],
        )


def test_future_off_is_zero_condition_with_zero_gradient_tether() -> None:
    predicted = torch.randn(2, 3, 4, requires_grad=True)
    condition = future_off_condition(predicted, enabled=True)
    tether = future_off_zero_tether(predicted, enabled=True)
    assert torch.count_nonzero(condition) == 0
    tether.backward()
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad) == 0
