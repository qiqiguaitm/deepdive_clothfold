from __future__ import annotations

import numpy as np
import pytest

from pi05_r2_adaptive_execution import (
    CausalExecutionController,
    CausalRecurrenceReadout,
    R2ScheduleConfig,
)


def make_readout() -> CausalRecurrenceReadout:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.7, 0.3],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return CausalRecurrenceReadout(
        reference_features=features,
        episode_offsets=np.asarray([0, 3, 6]),
        reference_progress=np.asarray([0.0, 0.5, 1.0, 0.0, 0.5, 1.0]),
        reference_density=np.ones(6),
        sigma=0.5,
        density_calibration=np.linspace(0.1, 1.0, 10),
        boundary_progress=np.asarray([0.48, 0.52]),
    )


def test_readout_is_equal_episode_and_boundary_aware() -> None:
    output = make_readout().query(np.asarray([0.75, 0.25]))
    assert 0.35 < output["progress"] < 0.65
    assert 0.0 <= output["confidence"] <= 1.0
    assert output["boundary_distance"] >= 0.0
    assert output["boundary_proximity"] > 0.5


def test_readout_rejects_wrong_feature_shape() -> None:
    with pytest.raises(ValueError, match="current feature"):
        make_readout().query(np.zeros(3))


def test_stable_phase_uses_full_pi05_horizon() -> None:
    controller = CausalExecutionController()
    controller.observe(
        step=8,
        progress=0.2,
        density=0.8,
        confidence=0.8,
        boundary_proximity=0.1,
    )
    decision = controller.decide(cache_remaining=0)
    assert decision.force_replan
    assert decision.horizon == 8
    assert decision.reason == "stable"


def test_new_regression_forces_one_step_replan() -> None:
    controller = CausalExecutionController()
    controller.observe(step=0, progress=0.5, density=0.8, confidence=0.8, boundary_proximity=0.0)
    controller.decide(cache_remaining=0)
    controller.observe(step=8, progress=0.1, density=0.8, confidence=0.8, boundary_proximity=0.0)
    decision = controller.decide(cache_remaining=6)
    assert decision.force_replan
    assert decision.horizon == 1
    assert decision.reason == "regression"


def test_stall_uses_only_past_fifty_steps() -> None:
    controller = CausalExecutionController(R2ScheduleConfig(progress_ema_alpha=1.0))
    for step in range(51):
        signal = controller.observe(
            step=step,
            progress=0.20 + step * 0.0001,
            density=0.8,
            confidence=0.8,
            boundary_proximity=0.0,
        )
    assert signal.stall


def test_query_budget_allows_one_emergency_debt_then_defers() -> None:
    controller = CausalExecutionController(R2ScheduleConfig(progress_ema_alpha=1.0))
    controller.observe(step=0, progress=0.5, density=0.8, confidence=0.8, boundary_proximity=0.0)
    first = controller.decide(cache_remaining=0)
    assert first.force_replan
    controller.observe(step=1, progress=0.2, density=0.8, confidence=0.8, boundary_proximity=0.0)
    emergency = controller.decide(cache_remaining=7)
    assert emergency.force_replan
    assert emergency.query_debt_after_replan == 1
    controller.observe(step=2, progress=0.5, density=0.01, confidence=0.0, boundary_proximity=0.0)
    deferred = controller.decide(cache_remaining=1)
    assert not deferred.force_replan
    assert deferred.reason == "budget_defer"


def test_empty_cache_queries_and_repays_existing_debt() -> None:
    controller = CausalExecutionController(R2ScheduleConfig(progress_ema_alpha=1.0))
    controller.observe(step=0, progress=0.5, density=0.8, confidence=0.8, boundary_proximity=0.0)
    controller.decide(cache_remaining=0)
    controller.observe(step=1, progress=0.2, density=0.8, confidence=0.8, boundary_proximity=0.0)
    controller.decide(cache_remaining=3)
    controller.observe(step=2, progress=0.2, density=0.8, confidence=0.8, boundary_proximity=0.0)
    decision = controller.decide(cache_remaining=0)
    assert decision.force_replan
    assert decision.reason == "budget_repay"
    assert decision.horizon == 8


def test_schedule_cannot_exceed_public_pi05_action_horizon() -> None:
    with pytest.raises(ValueError, match="execution horizons"):
        R2ScheduleConfig(stable_horizon=9).validate()
