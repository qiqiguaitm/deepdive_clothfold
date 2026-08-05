#!/usr/bin/env python3
"""Evaluate preregistered CRAVE progress, stall, and regression on policy rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from build_pi05_crave_r0_labels import query_recurrence_fields, reference_ranges


STALL_WINDOW_STEPS = 50
STALL_PROGRESS_EPSILON = 0.02
REGRESSION_PROGRESS_DROP = 0.10
LOW_DENSITY_QUANTILE = 0.05
SMOOTHING_SIGMA_SAMPLES = 1.4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260804


def first_true_step(mask: np.ndarray, frame_index: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(frame_index[indices[0]]) if len(indices) else None


def trajectory_events(
    progress: np.ndarray,
    density: np.ndarray,
    frame_index: np.ndarray,
    *,
    stride: int,
    density_floor: float,
) -> dict[str, Any]:
    progress = np.asarray(progress, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    frame_index = np.asarray(frame_index, dtype=np.int32)
    if not (len(progress) == len(density) == len(frame_index)) or len(progress) == 0:
        raise ValueError("trajectory arrays must be nonempty and aligned")
    smooth_progress = gaussian_filter1d(progress, SMOOTHING_SIGMA_SAMPLES)
    window = max(1, math.ceil(STALL_WINDOW_STEPS / stride))
    stall = np.zeros(len(progress), dtype=bool)
    if len(progress) > window:
        stall[window:] = (
            smooth_progress[window:] - smooth_progress[:-window]
            <= STALL_PROGRESS_EPSILON
        )
    running_max = np.maximum.accumulate(smooth_progress)
    regression = running_max - smooth_progress >= REGRESSION_PROGRESS_DROP
    low_density = density < density_floor
    first_stall = first_true_step(stall, frame_index)
    first_regression = first_true_step(regression, frame_index)
    first_low_density = first_true_step(low_density, frame_index)
    return {
        "initial_progress": float(smooth_progress[0]),
        "terminal_progress": float(smooth_progress[-1]),
        "progress_gain": float(smooth_progress[-1] - smooth_progress[0]),
        "minimum_density": float(np.min(density)),
        "mean_density": float(np.mean(density)),
        "stall_fraction": float(np.mean(stall)),
        "regression_fraction": float(np.mean(regression)),
        "low_density_fraction": float(np.mean(low_density)),
        "first_stall_step": first_stall,
        "first_regression_step": first_regression,
        "first_low_density_step": first_low_density,
        "stall_detected": first_stall is not None,
        "regression_detected": first_regression is not None,
        "low_density_detected": first_low_density is not None,
    }


def bootstrap_mean_difference(
    positive: np.ndarray, negative: np.ndarray, *, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if len(positive) == 0 or len(negative) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = (
            rng.choice(positive, len(positive), replace=True).mean()
            - rng.choice(negative, len(negative), replace=True).mean()
        )
    return {
        "estimate": float(positive.mean() - negative.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def detection_report(records: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    success = [record for record in records if record["success"]]
    failure = [record for record in records if not record["success"]]
    true_positive = sum(bool(record[key]) for record in failure)
    false_positive = sum(bool(record[key]) for record in success)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "failure_count": len(failure),
        "success_count": len(success),
        "failure_recall": true_positive / len(failure) if failure else None,
        "success_false_positive_rate": false_positive / len(success) if success else None,
        "precision": true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None,
    }


def summarize_outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    success = [record for record in records if record["success"]]
    failure = [record for record in records if not record["success"]]
    if not success or not failure:
        return {
            "estimable": False,
            "episode_count": len(records),
            "successes": len(success),
            "failures": len(failure),
            "separation": {},
            "detection": {},
            "failed_stall_lead_steps_median": None,
            "failed_regression_lead_steps_median": None,
        }
    separation_specs = {
        "terminal_progress_success_minus_failure": ("terminal_progress", success, failure),
        "progress_gain_success_minus_failure": ("progress_gain", success, failure),
        "minimum_density_success_minus_failure": ("minimum_density", success, failure),
        "stall_fraction_failure_minus_success": ("stall_fraction", failure, success),
        "regression_fraction_failure_minus_success": ("regression_fraction", failure, success),
        "low_density_fraction_failure_minus_success": ("low_density_fraction", failure, success),
    }
    separation = {
        name: bootstrap_mean_difference(
            np.asarray([record[field] for record in positive]),
            np.asarray([record[field] for record in negative]),
        )
        for name, (field, positive, negative) in separation_specs.items()
    }
    detection = {
        name: detection_report(records, f"{name}_detected")
        for name in ("stall", "regression", "low_density")
    }
    failed_stall_lead = [
        record["stall_lead_steps"]
        for record in failure
        if record["stall_lead_steps"] is not None
    ]
    failed_regression_lead = [
        record["regression_lead_steps"]
        for record in failure
        if record["regression_lead_steps"] is not None
    ]
    return {
        "estimable": True,
        "episode_count": len(records),
        "successes": len(success),
        "failures": len(failure),
        "separation": separation,
        "detection": detection,
        "failed_stall_lead_steps_median": float(np.median(failed_stall_lead))
        if failed_stall_lead
        else None,
        "failed_regression_lead_steps_median": float(np.median(failed_regression_lead))
        if failed_regression_lead
        else None,
    }


def load_feature_records(feature_root: Path) -> tuple[list[dict[str, Any]], int]:
    manifests = sorted(feature_root.glob("shard*.json"))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests under {feature_root}")
    records = []
    stride_values = set()
    expected_total = set()
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        stride_values.add(int(payload["stride"]))
        expected_total.add(int(payload["all_episode_count"]))
        records.extend(payload["records"])
    if len(stride_values) != 1 or len(expected_total) != 1:
        raise ValueError("inconsistent rollout feature manifests")
    identities = {
        (record["task"], int(record["simulator_seed"]), int(record["episode_id"]))
        for record in records
    }
    expected = expected_total.pop()
    if len(records) != expected or len(identities) != expected:
        raise ValueError(f"feature coverage {len(records)}/{len(identities)} != {expected}")
    return records, stride_values.pop()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# pi0.5 CRAVE R0 Rollout Diagnostics",
        "",
        f"- Episodes: {report['episode_count']}",
        f"- Success/failure: {report['successes']}/{report['failures']}",
        f"- Feature stride: {report['protocol']['feature_stride_steps']} action steps",
        f"- Stall: progress gain <= {STALL_PROGRESS_EPSILON} over {STALL_WINDOW_STEPS} steps",
        f"- Regression: drop >= {REGRESSION_PROGRESS_DROP} from running maximum",
        "",
        "## Aggregate",
        "",
    ]
    for name, value in report["separation"].items():
        lines.append(
            f"- {name}: {value['estimate']:.4f} "
            f"(95% CI {value['ci_low']:.4f}, {value['ci_high']:.4f})"
        )
    for name, value in report["detection"].items():
        lines.append(
            f"- {name}: failure recall={value['failure_recall']}, "
            f"success FPR={value['success_false_positive_rate']}, precision={value['precision']}"
        )
    lines.extend(["", "## Per Task", ""])
    for task, value in report["per_task"].items():
        lines.append(
            f"- {task}: {value['successes']} success / {value['failures']} failure; "
            f"estimable={str(value['estimable']).lower()}"
        )
    lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reference-feature-dir", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--probe-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    source_records, stride = load_feature_records(args.feature_root)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    labels_manifest = json.loads(args.labels_manifest.read_text(encoding="utf-8"))
    probe = np.load(args.probe_labels)
    task_ids = labels_manifest["physical_task_ids"]
    records = []
    for task in sorted(selection["tasks"]):
        task_rows = [record for record in source_records if record["task"] == task]
        if not task_rows:
            raise ValueError(f"no rollout records for {task}")
        reference_episodes = selection["tasks"][task]["reference_episodes"]
        reference_features = [
            np.load(args.reference_feature_dir / f"ep{episode}.npz")["pooled"]
            for episode in reference_episodes
        ]
        reference, ranges = reference_ranges(reference_features)
        sigma = float(labels_manifest["tasks"][task]["sigma"])
        task_id = int(task_ids[task])
        calibration_density = probe["current_recurrence_density"][
            probe["physical_task"] == task_id
        ]
        density_floor = float(np.quantile(calibration_density, LOW_DENSITY_QUANTILE))
        for source in task_rows:
            feature_path = (
                args.feature_root
                / f"seed{source['simulator_seed']}"
                / task
                / f"episode{source['episode_id']}.npz"
            )
            payload = np.load(feature_path)
            density, progress = query_recurrence_fields(
                payload["pooled"],
                reference,
                ranges,
                sigma,
                device=args.device,
                chunk_size=args.chunk_size,
            )
            event = trajectory_events(
                progress,
                density,
                payload["frame_index"],
                stride=stride,
                density_floor=density_floor,
            )
            terminal_step = int(source["steps"])
            for name in ("stall", "regression", "low_density"):
                first = event[f"first_{name}_step"]
                event[f"{name}_lead_steps"] = terminal_step - first if first is not None else None
            records.append({**source, "density_floor": density_floor, **event})

    success = [record for record in records if record["success"]]
    failure = [record for record in records if not record["success"]]
    if not success or not failure:
        raise ValueError("rollout diagnostics require both successes and failures")
    aggregate = summarize_outcomes(records)
    separation = aggregate["separation"]
    detection = aggregate["detection"]
    per_task = {
        task: summarize_outcomes([record for record in records if record["task"] == task])
        for task in sorted(selection["tasks"])
    }
    informative = any(
        metric["ci_low"] > 0 for metric in separation.values() if np.isfinite(metric["ci_low"])
    )
    interpretation = (
        "At least one preregistered trajectory statistic separates outcomes with a positive "
        "episode-bootstrap interval. This supports further causal readout validation, not a value claim."
        if informative
        else "No preregistered trajectory statistic has a positive episode-bootstrap interval. "
        "CRAVE is retained as a phase teacher only and is not interpreted as failure value."
    )
    report = {
        "schema_version": 1,
        "protocol": {
            "name": "pi05_crave_r0_rollout_diagnostics_v1",
            "feature_stride_steps": stride,
            "stall_window_steps": STALL_WINDOW_STEPS,
            "stall_progress_epsilon": STALL_PROGRESS_EPSILON,
            "regression_progress_drop": REGRESSION_PROGRESS_DROP,
            "low_density_quantile": LOW_DENSITY_QUANTILE,
            "smoothing_sigma_samples": SMOOTHING_SIGMA_SAMPLES,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "episode_count": len(records),
        "successes": len(success),
        "failures": len(failure),
        "separation": separation,
        "detection": detection,
        "failed_stall_lead_steps_median": aggregate["failed_stall_lead_steps_median"],
        "failed_regression_lead_steps_median": aggregate[
            "failed_regression_lead_steps_median"
        ],
        "per_task": per_task,
        "interpretation": interpretation,
        "records": records,
    }
    atomic_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(args.markdown, render_markdown(report))
    atomic_text(args.marker, f"completed=1\nreport={args.output}\n")
    print(json.dumps({key: report[key] for key in ("episode_count", "successes", "failures", "separation", "detection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
