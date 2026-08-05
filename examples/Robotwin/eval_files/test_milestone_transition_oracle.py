import json

import numpy as np

from examples.Robotwin.eval_files.milestone_transition_oracle import FrozenStageProfile, MonotonicExpertJointTracker
from examples.Robotwin.eval_files.robotwin_batch_bridge import _play_once_with_event_trace


def test_frozen_profile_and_content_controls(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps({"episode_index": episode, "length": 11}) for episode in (10, 11, 20, 21)
        )
        + "\n"
    )
    pairs = tmp_path / "pairs.npz"
    np.savez(
        pairs,
        cur_ep=np.asarray([10, 10, 11, 11, 20, 20, 21, 21]),
        cur_fi=np.asarray([1, 8, 1, 8, 2, 7, 2, 7]),
        cur_ms=np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
        pair_task=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    profile = FrozenStageProfile(pairs, episodes)

    correct = profile.label(0, 0.1)
    within = profile.label(0, 0.1, "within-task")
    cross = profile.label(0, 0.1, "cross-task")
    null = profile.label(0, 0.1, "null")
    assert (correct.task, correct.current, correct.next) == (0, 0, 1)
    assert within.task == correct.task and within.current != correct.current
    assert cross.task != correct.task
    assert not null.available


def test_expert_alignment_is_state_based_and_monotonic():
    trajectory = np.arange(20, dtype=np.float64).reshape(10, 2)
    tracker = MonotonicExpertJointTracker(trajectory)
    assert tracker.update(trajectory[6]) == 6 / 9
    assert tracker.update(trajectory[2]) == 6 / 9
    assert tracker.update(trajectory[9]) == 1.0


def test_expert_event_trace_wraps_program_without_manual_stages():
    class Robot:
        value = 0.0

        def get_left_arm_jointState(self):
            return [self.value] * 7

        def get_right_arm_jointState(self):
            return [-self.value] * 7

    class Task:
        def __init__(self):
            self.robot = Robot()

        def move(self):
            self.robot.value += 1.0

        def play_once(self):
            self.move()
            self.move()
            return {"info": {}}

    info, trace = _play_once_with_event_trace(Task())
    assert info == {"info": {}}
    assert trace.shape == (3, 14)
    np.testing.assert_array_equal(trace[:, 0], [0.0, 1.0, 2.0])
