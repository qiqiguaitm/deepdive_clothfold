from pathlib import Path

import numpy as np

from build_pi05_r3_semantic_vocabulary import (
    TASK_PLANS,
    action_signature,
    monotonic_event_assignment,
    play_once_trace,
)


def test_all_semantic_plans_match_simulator_execution_order() -> None:
    root = Path("/vePFS/HuanQian/RoboTwin/envs")
    for task, plan in TASK_PLANS.items():
        assert play_once_trace(root / f"{task}.py") == plan["source_trace"]


def test_monotonic_alignment_uses_every_event_without_moving_boundaries() -> None:
    assignment = monotonic_event_assignment(
        np.asarray([0.05, 0.22, 0.48, 0.62, 0.91]), event_count=3
    )
    assert len(assignment) == 5
    assert np.all(np.diff(assignment) >= 0)
    assert set(map(int, assignment)) == {0, 1, 2}


def test_action_signature_identifies_active_arm_and_gripper_close() -> None:
    action = np.zeros((4, 14), dtype=np.float32)
    action[:, 6] = [1.0, 0.5, 0.0, 0.0]
    action[:, 13] = 1.0
    action[:, 0] = [0.0, 0.2, 0.4, 0.6]
    result = action_signature(action)
    assert result["active_arm"] == "left"
    assert result["left_gripper_close"] == 1.0
    assert result["right_gripper_close"] == 0.0
