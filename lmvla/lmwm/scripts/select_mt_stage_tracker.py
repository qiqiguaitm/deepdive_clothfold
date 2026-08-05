#!/usr/bin/env python3
"""Select the MT3 tracker without consulting closed-loop policy results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


F1_TIE_MARGIN = 0.005
NEXT_ACCURACY_TIE_MARGIN = 0.005


def select(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(reports) != {"current_frame", "history_proprio"}:
        raise ValueError("selection requires exactly current_frame and history_proprio reports")
    split_hashes = {report["protocol"]["split_sha256"] for report in reports.values()}
    if len(split_hashes) != 1 or None in split_hashes:
        raise ValueError("tracker reports must reference the same frozen split")

    rows = []
    for name, report in reports.items():
        rows.append(
            {
                "name": name,
                "current_macro_f1": float(report["task_macro"]["current_macro_f1"]),
                "next_accuracy": float(report["task_macro"]["next_accuracy"]),
                "current_ece_15bin": float(report["pooled"]["current_ece_15bin"]),
            }
        )
    best_f1 = max(row["current_macro_f1"] for row in rows)
    finalists = [row for row in rows if best_f1 - row["current_macro_f1"] <= F1_TIE_MARGIN]
    best_next = max(row["next_accuracy"] for row in finalists)
    finalists = [row for row in finalists if best_next - row["next_accuracy"] <= NEXT_ACCURACY_TIE_MARGIN]
    finalists.sort(key=lambda row: (row["current_ece_15bin"], row["name"] != "current_frame"))
    return {
        "selected": finalists[0]["name"],
        "selection_rule": (
            "maximize task-macro current-stage F1; within 0.005 maximize task-macro next-transition "
            "accuracy; within 0.005 minimize pooled 15-bin current-stage ECE; exact ties prefer current_frame"
        ),
        "split_sha256": next(iter(split_hashes)),
        "candidates": sorted(rows, key=lambda row: row["name"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-frame", type=Path, required=True)
    parser.add_argument("--history-proprio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        {
            "current_frame": json.loads(args.current_frame.read_text()),
            "history_proprio": json.loads(args.history_proprio.read_text()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
