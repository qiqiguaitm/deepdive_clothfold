#!/usr/bin/env python3
"""Combine per-evaluation RoboTwin reports across training seeds."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPORT_RE = re.compile(r"^rt_all6_v2_(.+)_seed(\d+)_unseen\.json$")
METHODS = {"nowm", "local", "absolute", "residual", "isolation", "combo"}


def summarize(report_dir: Path) -> dict[str, Any]:
    by_method: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    source_files = []
    for path_text in sorted(glob.glob(str(report_dir / "rt_all6_v2_*_seed*_unseen.json"))):
        path = Path(path_text)
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        method, seed_text = match.groups()
        if method not in METHODS:
            continue
        seed = int(seed_text)
        report = json.loads(path.read_text())
        if report.get("summary_count") != 24 or report.get("task_count") != 6:
            continue
        by_method[method][seed] = report
        source_files.append(path.name)

    methods = {}
    for method, seed_reports in sorted(by_method.items()):
        seed_rows = {}
        task_values: dict[str, list[float]] = defaultdict(list)
        macro_values = []
        for seed, report in sorted(seed_reports.items()):
            macro = float(report["macro_success_rate"])
            macro_values.append(macro)
            task_rates = {
                task: float(task_report["mean_success_rate"])
                for task, task_report in sorted(report["tasks"].items())
            }
            for task, rate in task_rates.items():
                task_values[task].append(rate)
            seed_rows[str(seed)] = {
                "macro_success_rate": macro,
                "total_episodes": report["total_episodes"],
                "task_success_rates": task_rates,
            }
        methods[method] = {
            "training_seed_count": len(seed_rows),
            "mean_macro_success_rate": mean(macro_values),
            "macro_population_std": pstdev(macro_values) if len(macro_values) > 1 else None,
            "task_mean_success_rates": {
                task: mean(values) for task, values in sorted(task_values.items())
            },
            "seeds": seed_rows,
        }
    return {"source_files": source_files, "method_count": len(methods), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.report_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
