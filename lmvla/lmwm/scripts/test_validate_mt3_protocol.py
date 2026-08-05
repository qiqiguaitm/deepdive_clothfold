from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("validate_mt3_protocol.py")
SPEC = importlib.util.spec_from_file_location("validate_mt3_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def protocol():
    return {
        "shared_encoder": {
            "external_encoder": None,
            "tracker_only_checkpoint": "same raw pi05_base used to initialize joint policy training",
        },
        "tracker_only_training": {
            "selection_uses_closed_loop_success": False,
            "random_seed": 1000,
        },
        "joint_policy_training": {
            "vision_encoder_frozen": False,
            "tracker_receives_live_current_encoder_features": True,
            "tracker_gradient_into_vision_encoder": False,
            "policy_updates": 50000,
            "policy_batch_size": 16,
            "routing_scope": {
                "episode_ranges_by_task_id": {
                    str(task): [task * 10, task * 10 + 9] for task in range(6)
                },
                "covered_episodes": 60,
                "outside_scope_transition": "null",
            },
        },
    }


def split():
    return {"train_episodes": list(range(960)), "val_episodes": list(range(960, 1200))}


def audit():
    return {
        "checks": {
            "episode_frame_unique": True,
            "episode_split_leakage": False,
            "frame_within_episode_length": True,
            "parquet_complete": True,
            "three_view_video_complete": True,
            "base_history_cache_complete": True,
        },
        "episodes": {"total": 1200, "train": 960, "validation": 240},
        "history_offsets_at_50hz": [-15, -7, 0],
        "rows": {
            "total": 420238,
            "train": 336124,
            "validation": 84114,
            "history_frame_references": 1260714,
        },
    }


def test_live_encoder_invariants_are_enforced():
    value = protocol()
    value["shared_encoder"]["external_encoder"] = "So400m"
    try:
        MODULE.validate_invariants(value, split(), audit())
    except ValueError as error:
        assert "external visual encoder" in str(error)
    else:
        raise AssertionError("external encoder should fail")


def test_valid_frozen_protocol_invariants_pass():
    MODULE.validate_invariants(protocol(), split(), audit())


def test_episode_leakage_is_rejected():
    value = split()
    value["val_episodes"][0] = value["train_episodes"][0]
    try:
        MODULE.validate_invariants(protocol(), value, audit())
    except ValueError as error:
        assert "episode leakage" in str(error)
    else:
        raise AssertionError("episode leakage should fail")


def test_incomplete_tracker_inputs_are_rejected():
    value = audit()
    value["checks"]["three_view_video_complete"] = False
    try:
        MODULE.validate_invariants(protocol(), split(), value)
    except ValueError as error:
        assert "incomplete tracker inputs" in str(error)
    else:
        raise AssertionError("incomplete tracker data should fail")
