import json
from pathlib import Path

from audit_robotwin_rollout_artifacts import audit, render_markdown


def test_summary_only_is_not_trajectory_evidence(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "summary.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": 0, "seed": 1, "success": True, "steps": 10},
                    {"episode_id": 1, "seed": 2, "success": False, "steps": 20},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = audit(tmp_path)
    assert result["episode_count"] == 2
    assert result["supports_success_failure_duration_analysis"] is True
    assert result["supports_crave_rollout_metrics"] is False
    assert "duration analysis only" in render_markdown(result)


def test_video_and_outcomes_enable_rollout_analysis(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "summary.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": 0, "success": True, "steps": 10},
                    {"episode_id": 1, "success": False, "steps": 20},
                ]
            }
        ),
        encoding="utf-8",
    )
    (task / "episode0.mp4").write_bytes(b"stub")

    result = audit(tmp_path)
    assert result["trajectory_file_count"] == 1
    assert result["supports_crave_rollout_metrics"] is True
