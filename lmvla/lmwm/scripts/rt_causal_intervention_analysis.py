#!/usr/bin/env python3
"""Pair RoboTwin intervention episodes and report exact McNemar tests."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def load_episodes(root: str | list[str]) -> dict[tuple[int, str, int], bool]:
    episodes: dict[tuple[int, str, int], bool] = {}
    roots = [root] if isinstance(root, str) else root
    for root_item in roots:
        for summary_path in glob.glob(f"{root_item}/**/summary.json", recursive=True):
            path = Path(summary_path)
            try:
                summary = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            task = summary.get("task_name")
            match = re.search(r"/seed(\d+)(?:/|$)", summary_path)
            if not task or not match or not isinstance(summary.get("episodes"), list):
                continue
            eval_seed = int(match.group(1))
            for episode in summary["episodes"]:
                episode_seed = episode.get("seed")
                success = episode.get("success")
                if episode_seed is None or not isinstance(success, bool):
                    continue
                key = (eval_seed, task, int(episode_seed))
                if key in episodes and episodes[key] != success:
                    raise ValueError(
                        f"conflicting duplicate episode {key} under {roots}"
                    )
                episodes[key] = success
    return episodes


def exact_mcnemar(correct_only: int, control_only: int) -> float:
    discordant = correct_only + control_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(correct_only, control_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare(
    correct: dict[tuple[int, str, int], bool],
    control: dict[tuple[int, str, int], bool],
    common: set[tuple[int, str, int]] | None = None,
) -> list[dict[str, int | float | str]]:
    common = common if common is not None else set(correct) & set(control)
    by_task: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for key in common:
        by_task[key[1]].append(key)

    rows: list[dict[str, int | float | str]] = []
    for task, keys in sorted(by_task.items()):
        rows.append(make_row(task, keys, correct, control))
    rows.append(make_row("POOLED", sorted(common), correct, control))
    return rows


def add_holm_adjustment(
    results: dict[str, list[dict[str, int | float | str]]],
) -> None:
    """Adjust every reported McNemar test as one predeclared family."""
    indexed = [
        (float(row["mcnemar_p"]), label, index)
        for label, rows in results.items()
        for index, row in enumerate(rows)
    ]
    indexed.sort(key=lambda item: item[0])
    running_max = 0.0
    family_size = len(indexed)
    for rank, (p_value, label, index) in enumerate(indexed):
        adjusted = min(1.0, (family_size - rank) * p_value)
        running_max = max(running_max, adjusted)
        results[label][index]["mcnemar_p_holm"] = running_max


def make_row(
    task: str,
    keys: list[tuple[int, str, int]],
    correct: dict[tuple[int, str, int], bool],
    control: dict[tuple[int, str, int], bool],
) -> dict[str, int | float | str]:
    n = len(keys)
    correct_successes = sum(correct[key] for key in keys)
    control_successes = sum(control[key] for key in keys)
    correct_only = sum(correct[key] and not control[key] for key in keys)
    control_only = sum(control[key] and not correct[key] for key in keys)
    return {
        "task": task,
        "common": n,
        "correct_sr": 100.0 * correct_successes / n if n else 0.0,
        "control_sr": 100.0 * control_successes / n if n else 0.0,
        "delta_pp": 100.0 * (control_successes - correct_successes) / n if n else 0.0,
        "correct_only": correct_only,
        "control_only": control_only,
        "mcnemar_p": exact_mcnemar(correct_only, control_only),
    }


def print_markdown(label: str, rows: list[dict[str, int | float | str]]) -> None:
    print(f"\n### {label}")
    print("| Task | Common | Correct SR | Control SR | Delta | Correct-only / control-only | Exact p | Holm p |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['task']} | {row['common']} | {row['correct_sr']:.2f}% | "
            f"{row['control_sr']:.2f}% | {row['delta_pp']:+.2f} pp | "
            f"{row['correct_only']} / {row['control_only']} | {row['mcnemar_p']:.4f} | "
            f"{row['mcnemar_p_holm']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct-root", required=True)
    parser.add_argument(
        "--control",
        action="append",
        required=True,
        metavar="LABEL=ROOT",
        help="May be repeated for zero, shuffled, and other-task roots.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Use a separate correct/control intersection instead of one cohort shared by all controls.",
    )
    args = parser.parse_args()

    correct = load_episodes(args.correct_root.split(","))
    if not correct:
        raise SystemExit(f"no episodes found under correct root: {args.correct_root}")

    controls = {}
    for item in args.control:
        if "=" not in item:
            raise SystemExit(f"control must be LABEL=ROOT: {item}")
        label, root = item.split("=", 1)
        control = load_episodes(root.split(","))
        if not control:
            raise SystemExit(f"no episodes found under control root: {root}")
        controls[label] = control

    shared = set(correct)
    if not args.pairwise:
        for control in controls.values():
            shared &= set(control)
        if not shared:
            raise SystemExit("no episodes are common to correct and all control roots")

    results = {
        label: compare(correct, control, None if args.pairwise else shared)
        for label, control in controls.items()
    }
    add_holm_adjustment(results)

    if args.as_json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for label, rows in results.items():
            print_markdown(label, rows)


if __name__ == "__main__":
    main()
