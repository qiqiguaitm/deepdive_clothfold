from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("decide_mt3_seed1000_gate.py")
SPEC = importlib.util.spec_from_file_location("decide_mt3_seed1000_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def analysis(deltas):
    rows = []
    for control in ("a0", "null", "within_task"):
        rows.append({"control": control, "scope": "pooled", "success_rate_delta": deltas[control]})
        for task in MODULE.PREDECLARED_MULTISTAGE_TASKS:
            rows.append(
                {
                    "control": control,
                    "scope": task,
                    "success_rate_delta": deltas.get(task, 0.0) if control == "a0" else 0.1,
                }
            )
    return {"tests": rows}


def test_mt3_pilot_requires_content_baselines_and_two_multistage_tasks():
    accepted = MODULE.decide(
        analysis(
            {
                "a0": 0.03,
                "null": 0.02,
                "within_task": 0.01,
                "handover_block": 0.02,
                "stack_blocks_two": 0.04,
                "stack_blocks_three": -0.01,
            }
        )
    )
    assert accepted["accepted_for_replication"]

    rejected = MODULE.decide(
        analysis(
            {
                "a0": 0.03,
                "null": 0.0,
                "within_task": 0.01,
                "handover_block": 0.02,
                "stack_blocks_two": 0.04,
                "stack_blocks_three": -0.01,
            }
        )
    )
    assert not rejected["accepted_for_replication"]
