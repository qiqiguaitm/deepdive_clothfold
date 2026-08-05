from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("select_mt_stage_tracker.py")
SPEC = importlib.util.spec_from_file_location("select_mt_stage_tracker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def report(f1: float, next_accuracy: float, ece: float, split: str = "frozen"):
    return {
        "protocol": {"split_sha256": split},
        "task_macro": {"current_macro_f1": f1, "next_accuracy": next_accuracy},
        "pooled": {"current_ece_15bin": ece},
    }


def test_f1_is_primary_selection_metric():
    result = MODULE.select(
        {
            "current_frame": report(0.80, 0.90, 0.01),
            "history_proprio": report(0.82, 0.70, 0.20),
        }
    )
    assert result["selected"] == "history_proprio"


def test_next_accuracy_then_calibration_break_near_f1_ties():
    result = MODULE.select(
        {
            "current_frame": report(0.800, 0.80, 0.01),
            "history_proprio": report(0.804, 0.82, 0.20),
        }
    )
    assert result["selected"] == "history_proprio"
