#!/usr/bin/env python3
"""Materialize the complete task-level R4 seed-1000 evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from analyze_pi05_r4_formal import _load_report


ARMS = ("ordinary", "outcome_free_crave", "terminal_outcome")
CONTROLS = ("ordinary", "outcome_free_crave")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _task_rates(cells: dict[str, dict[int, dict[int, int]]]) -> dict[str, float]:
    return {
        task: float(
            np.mean(
                [
                    outcome
                    for eval_seed in task_cells.values()
                    for outcome in eval_seed.values()
                ]
            )
        )
        for task, task_cells in sorted(cells.items())
    }


def finalize(report_paths: dict[str, Path], gate_path: Path) -> dict:
    if set(report_paths) != set(ARMS):
        raise ValueError(f"reports must be exactly {ARMS}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("protocol") != "pi05_r4_formal_eval_protocol_v1":
        raise ValueError("unexpected R4 gate protocol")

    macros = {}
    rates = {}
    for arm in ARMS:
        cells, macros[arm] = _load_report(report_paths[arm])
        rates[arm] = _task_rates(cells)
    if any(set(rates[arm]) != set(rates[ARMS[0]]) for arm in ARMS[1:]):
        raise ValueError("R4 reports have different task sets")
    if not np.isclose(gate["terminal_macro_success_rate"], macros["terminal_outcome"]):
        raise ValueError("gate terminal macro does not match the source report")

    tasks = {}
    for task in sorted(rates["ordinary"]):
        row = {arm: rates[arm][task] for arm in ARMS}
        row.update(
            {
                "terminal_minus_ordinary": row["terminal_outcome"] - row["ordinary"],
                "terminal_minus_outcome_free_crave": (
                    row["terminal_outcome"] - row["outcome_free_crave"]
                ),
            }
        )
        for control in CONTROLS:
            expected = gate["comparisons"][control]["task_deltas"][task]
            if not np.isclose(row[f"terminal_minus_{control}"], expected):
                raise ValueError(f"gate task delta mismatch for {task} against {control}")
        tasks[task] = row

    return {
        "schema_version": 1,
        "protocol": "pi05_r4_seed1000_complete_evidence_v1",
        "training_seed": 1000,
        "checkpoint_step": 5000,
        "cells_per_arm": 24,
        "episodes_per_arm": 1200,
        "macros": macros,
        "tasks": tasks,
        "comparisons": gate["comparisons"],
        "accepted": bool(gate["accepted"]),
        "claim_boundary": (
            "Terminal-outcome and outcome-free CRAVE are sample-weighting signals over "
            "expert demonstrations; this evidence does not estimate Q-values, action "
            "advantages, a reward model, a world critic, or model-predictive control."
        ),
        "sources": {
            "gate": {"path": str(gate_path), "sha256": _sha256(gate_path)},
            "reports": {
                arm: {"path": str(path), "sha256": _sha256(path)}
                for arm, path in sorted(report_paths.items())
            },
        },
    }


def markdown(payload: dict) -> str:
    decision = "accepted" if payload["accepted"] else "rejected"
    lines = [
        "# R4 Seed-1000 Fixed-Checkpoint Evidence",
        "",
        f"Decision: **{decision}** under the preregistered macro and task-safety gate.",
        "",
        "| Task | Ordinary | Outcome-free CRAVE | Terminal outcome | Terminal - ordinary | Terminal - CRAVE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, row in payload["tasks"].items():
        lines.append(
            f"| {task} | {100*row['ordinary']:.1f} | "
            f"{100*row['outcome_free_crave']:.1f} | {100*row['terminal_outcome']:.1f} | "
            f"{100*row['terminal_minus_ordinary']:+.1f} | "
            f"{100*row['terminal_minus_outcome_free_crave']:+.1f} |"
        )
    macros = payload["macros"]
    lines.extend(
        [
            f"| **Macro** | **{100*macros['ordinary']:.1f}** | "
            f"**{100*macros['outcome_free_crave']:.1f}** | "
            f"**{100*macros['terminal_outcome']:.1f}** | "
            f"**{100*(macros['terminal_outcome']-macros['ordinary']):+.1f}** | "
            f"**{100*(macros['terminal_outcome']-macros['outcome_free_crave']):+.1f}** |",
            "",
            "Paired hierarchical 95% intervals:",
        ]
    )
    for control in CONTROLS:
        comparison = payload["comparisons"][control]
        low, high = comparison["hierarchical_paired_bootstrap_95"]
        lines.append(
            f"- Terminal outcome minus {control}: [{100*low:+.2f}, {100*high:+.2f}] points; "
            f"task safety={comparison['no_task_regression_below_minus_0_05']}."
        )
    lines.extend(["", f"Claim boundary: {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--outcome-free-crave", type=Path, required=True)
    parser.add_argument("--terminal-outcome", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        {
            "ordinary": args.ordinary,
            "outcome_free_crave": args.outcome_free_crave,
            "terminal_outcome": args.terminal_outcome,
        },
        args.gate,
    )
    _atomic_text(args.output_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_text(args.output_md, markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
