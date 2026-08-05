from analyze_pi05_r3_semantic_screen import analyze


def report(outcomes: list[bool]) -> dict:
    return {
        "tasks": {
            "task": {
                "cells": [
                    {
                        "eval_seed": 0,
                        "episode_outcomes": [
                            {"scene_seed": index, "success": success}
                            for index, success in enumerate(outcomes)
                        ],
                    }
                ]
            }
        }
    }


def test_strong_paired_semantic_gain_passes_gate() -> None:
    target = [True] * 80 + [False] * 20
    baseline = [True] * 60 + [False] * 40
    reports = {
        "semantic_next": report(target),
        "no_subtask": report(baseline),
        "generic_stage": report([True] * 70 + [False] * 30),
        "semantic_current": report([True] * 69 + [False] * 31),
        "shuffled_semantic": report([True] * 65 + [False] * 35),
    }
    result = analyze(reports, resamples=2000, seed=7)
    assert result["gate"]["accepted"] is True


def test_unmatched_scene_identity_is_rejected() -> None:
    reports = {name: report([True, False]) for name in ("semantic_next", "no_subtask", "generic_stage", "semantic_current", "shuffled_semantic")}
    reports["generic_stage"]["tasks"]["task"]["cells"][0]["episode_outcomes"][1]["scene_seed"] = 99
    try:
        analyze(reports, resamples=100, seed=1)
    except ValueError as exc:
        assert "scene identity differs" in str(exc)
    else:
        raise AssertionError("unmatched scenes were accepted")
