import numpy as np

from analyze_pi05_crave_r0_rollouts import (
    bootstrap_mean_difference,
    detection_report,
    summarize_outcomes,
    trajectory_events,
)


def test_stall_and_regression_events_use_fixed_action_step_thresholds() -> None:
    progress = np.asarray([0.0, 0.1, 0.2, 0.3, 0.32, 0.30, 0.05, 0.0, 0.0])
    density = np.asarray([0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.3, 0.2, 0.1])
    frame_index = np.arange(len(progress), dtype=np.int32) * 10
    result = trajectory_events(
        progress, density, frame_index, stride=10, density_floor=0.25
    )
    assert result["regression_detected"] is True
    assert result["first_regression_step"] == 60
    assert result["low_density_detected"] is True
    assert result["first_low_density_step"] == 70


def test_bootstrap_and_detection_are_outcome_stratified() -> None:
    difference = bootstrap_mean_difference(
        np.asarray([0.8, 0.9, 1.0]), np.asarray([0.0, 0.1, 0.2]), draws=1000
    )
    assert difference["estimate"] > 0.7
    assert difference["ci_low"] > 0

    report = detection_report(
        [
            {"success": False, "event": True},
            {"success": False, "event": False},
            {"success": True, "event": True},
            {"success": True, "event": False},
        ],
        "event",
    )
    assert report["failure_recall"] == 0.5
    assert report["success_false_positive_rate"] == 0.5
    assert report["precision"] == 0.5


def test_per_task_summary_marks_single_outcome_as_not_estimable() -> None:
    summary = summarize_outcomes([{"success": True}, {"success": True}])
    assert summary["estimable"] is False
    assert summary["successes"] == 2
    assert summary["failures"] == 0
    assert summary["separation"] == {}
