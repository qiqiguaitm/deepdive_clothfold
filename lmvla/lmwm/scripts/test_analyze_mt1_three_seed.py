from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_mt1_three_seed.py")
SPEC = importlib.util.spec_from_file_location("analyze_mt1_three_seed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_final_gate_requires_positive_interval_and_two_task_improvements():
    contrast = {
        "available": True,
        "point_estimate_macro_delta": 0.03,
        "ci95": [0.01, 0.05],
    }
    accepted = MODULE.gate_decision(
        contrast,
        {"handover_block": 0.01, "stack_blocks_two": 0.03, "stack_blocks_three": -0.01},
    )
    assert accepted["accepted"]

    rejected = MODULE.gate_decision(
        {**contrast, "ci95": [-0.01, 0.05]},
        {"handover_block": 0.01, "stack_blocks_two": 0.03, "stack_blocks_three": -0.01},
    )
    assert not rejected["accepted"]
