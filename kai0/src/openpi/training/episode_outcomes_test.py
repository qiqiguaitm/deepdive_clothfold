import json

import pytest

from openpi.training.episode_outcomes import iter_episode_outcomes
from openpi.training.episode_outcomes import normalize_rollout_outcome


def test_legacy_success_is_normalized():
    outcome = normalize_rollout_outcome({"episode_id": 3, "success": False})
    assert outcome["label"] == "failure"
    assert outcome["rollout_mode"] == "demonstration"
    assert outcome["schema_version"] == 1


def test_v1_outcome_preserves_recap_fields(tmp_path):
    episode = {
        "episode_id": 4,
        "rollout_outcome": {
            "schema_version": 1,
            "label": "partial_success",
            "rollout_mode": "intervention",
            "failure_modes": ["missed_grasp"],
            "intervention_count": 2,
            "recovery_success": True,
            "unsafe_event": False,
            "time_limit_reached": True,
            "stage_outcomes": [{"stage": "flatten", "success": True}],
        },
    }
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    loaded_episode, outcome = next(iter(iter_episode_outcomes(manifest)))
    assert loaded_episode["episode_id"] == 4
    assert outcome["failure_modes"] == ["missed_grasp"]
    assert outcome["intervention_count"] == 2
    assert outcome["stage_outcomes"][0]["stage"] == "flatten"


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        normalize_rollout_outcome({"rollout_outcome": {"schema_version": 9}})
