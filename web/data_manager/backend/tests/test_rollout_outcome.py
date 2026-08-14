from app.models import SaveRecordingReq


def test_legacy_save_payload_remains_valid():
    req = SaveRecordingReq(success=False, note="legacy", scene_tags=["desk_a"])
    assert req.success is False
    assert req.note == "legacy"
    assert req.scene_tags == ["desk_a"]
    assert req.rollout_outcome() == {
        "schema_version": 1,
        "label": "failure",
        "rollout_mode": "demonstration",
        "stage_outcomes": [],
        "failure_modes": [],
        "intervention_count": 0,
        "recovery_success": None,
        "unsafe_event": False,
        "time_limit_reached": False,
    }


def test_empty_legacy_save_payload_keeps_previous_defaults():
    req = SaveRecordingReq()
    assert req.success is True
    assert req.note == ""
    assert req.scene_tags == []
    assert req.rollout_outcome()["label"] == "success"


def test_new_outcome_fields_are_namespaced():
    req = SaveRecordingReq(
        success=False,
        outcome="partial_success",
        rollout_mode="intervention",
        failure_modes=[" missed_grasp ", "missed_grasp"],
        intervention_count=2,
        recovery_success=True,
        stage_outcomes=[{"stage": "flatten", "success": True, "progress": 1.0}],
    )
    outcome = req.rollout_outcome()
    assert outcome["label"] == "partial_success"
    assert outcome["failure_modes"] == ["missed_grasp"]
    assert outcome["stage_outcomes"] == [
        {"stage": "flatten", "success": True, "progress": 1.0, "failure_mode": None}
    ]
