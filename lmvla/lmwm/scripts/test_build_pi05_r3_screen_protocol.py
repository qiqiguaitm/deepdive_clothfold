from build_pi05_r3_screen_protocol import CONDITIONS, build_manifest


def test_screen_manifest_is_exact_prefix_of_frozen_manifest() -> None:
    source = {"eval_seeds": {"0": {"a": [1, 2, 3], "b": [4, 5, 6]}, "1": {"a": [7, 8, 9], "b": [10, 11, 12]}}}
    result = build_manifest(source, 2)
    assert result["episodes_per_cell"] == 2
    assert result["eval_seeds"]["0"]["a"] == [1, 2]
    assert result["eval_seeds"]["1"]["b"] == [10, 11]


def test_five_preregistered_conditions_are_distinct() -> None:
    assert set(CONDITIONS) == {
        "semantic_next",
        "generic_stage",
        "semantic_current",
        "shuffled_semantic",
        "no_subtask",
    }
    assert CONDITIONS["shuffled_semantic"]["tracker_intervention"] == "within-task"
    assert CONDITIONS["no_subtask"]["prompt_mode"] == "none"
