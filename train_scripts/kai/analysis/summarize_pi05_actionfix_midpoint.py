#!/usr/bin/env python3
"""Summarize absolute-action-corrected pi0.5 midpoint diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path


NAME_RE = re.compile(
    r"^pi05_(a2_abs|a3_live)_seed1000_step20000_(.+)_probe"
    r"(?:(?:_(correct|zero|current))_actionfix_cnsh|_actionfix)$"
)
PROGRESS_RE = re.compile(
    r"Success rate:\s*(\d+)/(\d+).*?progress:\s*(\d+)/(\d+)"
)
EXPECTED_TASKS = {
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
}


def load_episode_outcomes(result_root: Path, task: str) -> dict[int, bool]:
    summaries = sorted(result_root.glob(f"**/tasks/{task}/summary.json"))
    if len(summaries) != 1:
        raise ValueError(
            f"expected one summary for {result_root.name}/{task}, got {len(summaries)}"
        )
    summary = json.loads(summaries[0].read_text())
    episodes = summary.get("episodes", [])
    outcomes = {int(episode["seed"]): bool(episode["success"]) for episode in episodes}
    if len(outcomes) != int(summary.get("n_episodes", -1)):
        raise ValueError(f"duplicate or missing episode seeds in {summaries[0]}")
    return outcomes


def exact_mcnemar_p(correct_only: int, control_only: int) -> float:
    discordant = correct_only + control_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(correct_only, control_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_comparison(
    correct: dict[int, bool], control: dict[int, bool], *, bootstrap_seed: int
) -> dict[str, object]:
    if correct.keys() != control.keys():
        raise ValueError("paired intervention cells do not use identical scene seeds")
    seeds = sorted(correct)
    differences = [int(correct[seed]) - int(control[seed]) for seed in seeds]
    correct_only = sum(value == 1 for value in differences)
    control_only = sum(value == -1 for value in differences)
    rng = random.Random(bootstrap_seed)
    bootstrap = sorted(
        sum(rng.choice(differences) for _ in seeds) / len(seeds) for _ in range(20_000)
    )
    return {
        "n_paired_scenes": len(seeds),
        "success_rate_delta": sum(differences) / len(differences),
        "paired_bootstrap_95_ci": [bootstrap[499], bootstrap[19_499]],
        "correct_only_successes": correct_only,
        "control_only_successes": control_only,
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(correct_only, control_only),
    }


def summarize(root: Path) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for result_root in sorted(root.glob("pi05_*_seed1000_step20000_*_probe*actionfix*")):
        match = NAME_RE.fullmatch(result_root.name)
        if match is None:
            continue
        arm, task, condition = match.groups()
        condition = condition or "correct"
        run_logs = sorted(result_root.glob("**/run.log"))
        if not run_logs:
            continue
        matches = PROGRESS_RE.findall(run_logs[-1].read_text(errors="replace"))
        if not matches:
            continue
        successes, attempts, progress, expected = map(int, matches[-1])
        cells.append(
            {
                "arm": arm,
                "task": task,
                "condition": condition,
                "successes": successes,
                "attempts": attempts,
                "progress": progress,
                "expected": expected,
                "success_rate": successes / attempts if attempts else None,
                "result_root": str(result_root),
            }
        )

    expected_keys = {
        (arm, task, "correct")
        for arm in ("a2_abs", "a3_live")
        for task in EXPECTED_TASKS
    }
    expected_keys.update(
        (arm, "stack_blocks_two", condition)
        for arm in ("a2_abs", "a3_live")
        for condition in ("zero", "current")
    )
    observed_keys = {(cell["arm"], cell["task"], cell["condition"]) for cell in cells}

    aggregates: dict[str, dict[str, object]] = {}
    for arm in ("a2_abs", "a3_live"):
        selected = [
            cell for cell in cells
            if cell["arm"] == arm and cell["condition"] == "correct"
        ]
        successes = sum(int(cell["successes"]) for cell in selected)
        attempts = sum(int(cell["attempts"]) for cell in selected)
        complete = len(selected) == len(EXPECTED_TASKS) and all(
            int(cell["progress"]) >= int(cell["expected"]) for cell in selected
        )
        aggregates[arm] = {
            "successes": successes,
            "attempts": attempts,
            "complete": complete,
            "success_rate": successes / attempts if complete and attempts else None,
            "partial_pooled_rate": successes / attempts if attempts else None,
            "partial_rate_comparable_across_arms": False if not complete else True,
            "cells": len(selected),
        }

    intervention_comparisons: dict[str, dict[str, object]] = {}
    cells_by_key = {
        (str(cell["arm"]), str(cell["task"]), str(cell["condition"])): cell
        for cell in cells
    }
    for arm_index, arm in enumerate(("a2_abs", "a3_live")):
        correct_cell = cells_by_key.get((arm, "stack_blocks_two", "correct"))
        if correct_cell is None:
            continue
        correct = load_episode_outcomes(Path(str(correct_cell["result_root"])), "stack_blocks_two")
        controls: dict[str, object] = {}
        for control_index, condition in enumerate(("zero", "current")):
            control_cell = cells_by_key.get((arm, "stack_blocks_two", condition))
            if control_cell is None:
                continue
            control = load_episode_outcomes(
                Path(str(control_cell["result_root"])), "stack_blocks_two"
            )
            controls[f"correct_minus_{condition}"] = paired_comparison(
                correct,
                control,
                bootstrap_seed=20_000 + 10 * arm_index + control_index,
            )
        intervention_comparisons[arm] = controls

    return {
        "protocol": "step-20000 diagnostic; absolute actions; mean/std; frozen seed-0 scenes",
        "admissibility": "diagnostic only; not a substitute for the step-49999 T1 matrix",
        "complete": observed_keys == expected_keys
        and all(int(cell["progress"]) >= int(cell["expected"]) for cell in cells),
        "expected_cells": len(expected_keys),
        "observed_cells": len(observed_keys),
        "missing_cells": ["/".join(key) for key in sorted(expected_keys - observed_keys)],
        "aggregates_correct": aggregates,
        "stack2_intervention_paired": intervention_comparisons,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
