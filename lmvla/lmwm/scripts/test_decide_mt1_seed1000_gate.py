from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("decide_mt1_seed1000_gate.py")
SPEC = importlib.util.spec_from_file_location("decide_mt1_seed1000_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def analysis(deltas):
    tasks = ("handover_block", "stack_blocks_two", "stack_blocks_three")
    rows = []
    for control in ("a0", "null_input", "within_task", "null_trained"):
        rows.append({"control": control, "scope": "pooled", "success_rate_delta": deltas[control]})
        for task in tasks:
            rows.append(
                {
                    "control": control,
                    "scope": task,
                    "success_rate_delta": deltas.get(task, 0.0) if control == "a0" else 0.1,
                }
            )
    return {"tests": rows}


def test_gate_accepts_only_positive_content_capacity_and_task_checks():
    accepted = MODULE.decide(
        analysis(
            {
                "a0": 0.03,
                "null_input": 0.02,
                "within_task": 0.01,
                "null_trained": 0.02,
                "handover_block": 0.01,
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
                "null_input": 0.0,
                "within_task": 0.01,
                "null_trained": 0.02,
                "handover_block": 0.01,
                "stack_blocks_two": 0.04,
                "stack_blocks_three": -0.01,
            }
        )
    )
    assert not rejected["accepted_for_replication"]


def test_atomic_write_replaces_complete_file(tmp_path):
    output = tmp_path / "nested" / "gate.json"
    MODULE.atomic_write_text(output, '{"accepted": false}\n')
    MODULE.atomic_write_text(output, '{"accepted": true}\n')
    assert output.read_text() == '{"accepted": true}\n'
    assert not list(output.parent.glob(".*.tmp"))
