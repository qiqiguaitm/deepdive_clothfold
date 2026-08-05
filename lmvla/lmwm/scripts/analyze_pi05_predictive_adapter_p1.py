#!/usr/bin/env python3
"""Apply the preregistered seed-1000 predictive-adapter policy gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import analyze, atomic_write_text, outcome_map  # noqa: E402


def validate_report(name: str, report: dict) -> None:
    if report.get("summary_count") != 24:
        raise ValueError(f"{name}: expected 24 cells, got {report.get('summary_count')}")
    if report.get("total_episodes") != 1200:
        raise ValueError(
            f"{name}: expected 1200 episodes, got {report.get('total_episodes')}"
        )
    if len(report.get("tasks", {})) != 6:
        raise ValueError(f"{name}: expected six tasks")


def apply_gate(normal: dict, controls: dict[str, dict]) -> dict:
    for name, report in {"normal": normal, **controls}.items():
        validate_report(name, report)
    normal_keys = set(outcome_map(normal))
    for name, report in controls.items():
        if set(outcome_map(report)) != normal_keys:
            raise ValueError(f"{name}: scene keys do not exactly match normal")

    comparison = analyze(normal, controls)
    macro_deltas = {
        name: normal["macro_success_rate"] - report["macro_success_rate"]
        for name, report in controls.items()
    }
    task_deltas_vs_a0 = {
        task: normal["tasks"][task]["mean_success_rate"]
        - controls["a0"]["tasks"][task]["mean_success_rate"]
        for task in sorted(normal["tasks"])
    }
    checks = {
        "normal_exceeds_a0": macro_deltas["a0"] > 0.0,
        "normal_exceeds_zero_gate": macro_deltas["zero_gate"] > 0.0,
        "normal_exceeds_shuffled": macro_deltas["shuffled"] > 0.0,
        "normal_exceeds_masked": macro_deltas["masked"] > 0.0,
        "no_task_regression_over_5pp": min(task_deltas_vs_a0.values()) >= -0.05,
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_action_adapter_p1_seed1000_gate_v1",
        "complete": True,
        "normal_macro_success_rate": normal["macro_success_rate"],
        "control_macro_success_rates": {
            name: report["macro_success_rate"] for name, report in controls.items()
        },
        "macro_deltas": macro_deltas,
        "task_deltas_vs_a0": task_deltas_vs_a0,
        "checks": checks,
        "paired_analysis": comparison,
        "accepted": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--zero-gate", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--masked", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def load(path: Path) -> dict:
        return json.loads(path.read_text())

    result = apply_gate(
        load(args.normal),
        {
            "a0": load(args.a0),
            "zero_gate": load(args.zero_gate),
            "shuffled": load(args.shuffled),
            "masked": load(args.masked),
        },
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": result["checks"], "accepted": result["accepted"]}, indent=2))


if __name__ == "__main__":
    main()
