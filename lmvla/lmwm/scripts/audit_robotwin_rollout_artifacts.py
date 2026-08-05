#!/usr/bin/env python3
"""Audit whether saved RoboTwin evaluations support trajectory-level analyses."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


TRAJECTORY_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".npz", ".npy", ".h5", ".hdf5", ".parquet"}
FRAME_KEYS = {"observations", "observation", "frames", "images", "trajectory", "transitions"}
ACTION_KEYS = {"actions", "action", "joint_actions"}
STATE_KEYS = {"states", "state", "robot_states", "joint_states"}


def _mean(values: list[int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def audit(root: Path) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    suffix_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    trajectory_files = [path for path in files if path.suffix.lower() in TRAJECTORY_SUFFIXES]
    summaries = sorted(root.rglob("summary.json"))

    valid_summaries = 0
    malformed_summaries: list[str] = []
    episode_count = 0
    successes = 0
    failures = 0
    success_steps: list[int] = []
    failure_steps: list[int] = []
    episode_keys: set[str] = set()
    payload_keys: set[str] = set()
    frame_fields: set[str] = set()
    action_fields: set[str] = set()
    state_fields: set[str] = set()

    for path in summaries:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            episodes = payload.get("episodes", [])
            if not isinstance(episodes, list):
                raise TypeError("episodes is not a list")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            malformed_summaries.append(f"{path}: {exc}")
            continue

        valid_summaries += 1
        payload_keys.update(str(key) for key in payload)
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            keys = {str(key) for key in episode}
            episode_keys.update(keys)
            frame_fields.update(keys & FRAME_KEYS)
            action_fields.update(keys & ACTION_KEYS)
            state_fields.update(keys & STATE_KEYS)
            episode_count += 1
            success = bool(episode.get("success", False))
            steps = episode.get("steps")
            if success:
                successes += 1
                if isinstance(steps, int):
                    success_steps.append(steps)
            else:
                failures += 1
                if isinstance(steps, int):
                    failure_steps.append(steps)

    has_frame_observations = bool(trajectory_files or frame_fields)
    supports_crave_rollout_metrics = bool(has_frame_observations and successes and failures)
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "summary_count": len(summaries),
        "valid_summary_count": valid_summaries,
        "malformed_summaries": malformed_summaries,
        "episode_count": episode_count,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / episode_count if episode_count else None,
        "mean_success_steps": _mean(success_steps),
        "mean_failure_steps": _mean(failure_steps),
        "summary_payload_keys": sorted(payload_keys),
        "episode_keys": sorted(episode_keys),
        "trajectory_file_count": len(trajectory_files),
        "trajectory_file_examples": [str(path.relative_to(root)) for path in trajectory_files[:20]],
        "frame_fields": sorted(frame_fields),
        "action_fields": sorted(action_fields),
        "state_fields": sorted(state_fields),
        "has_frame_observations": has_frame_observations,
        "supports_success_failure_duration_analysis": bool(successes and failures and success_steps and failure_steps),
        "supports_crave_rollout_metrics": supports_crave_rollout_metrics,
        "limitations": [] if supports_crave_rollout_metrics else [
            "Saved summaries contain episode outcomes and lengths but no frame observations.",
            "CRAVE progress, recurrence density, stall lead time, and regression detection cannot be reconstructed.",
            "A new outcome-labeled rollout collection with saved visual observations is required.",
        ],
    }


def render_markdown(result: dict[str, Any]) -> str:
    rate = result["success_rate"]
    rate_text = "n/a" if rate is None else f"{100.0 * rate:.2f}%"
    lines = [
        "# RoboTwin Rollout Artifact Audit",
        "",
        f"- Root: `{result['root']}`",
        f"- Valid summaries: {result['valid_summary_count']}/{result['summary_count']}",
        f"- Episodes: {result['episode_count']} ({result['successes']} success, {result['failures']} failure; {rate_text})",
        f"- Mean steps, success/failure: {result['mean_success_steps']} / {result['mean_failure_steps']}",
        f"- Trajectory-like files: {result['trajectory_file_count']}",
        f"- Episode fields: `{', '.join(result['episode_keys'])}`",
        "",
        "## Verdict",
        "",
    ]
    if result["supports_crave_rollout_metrics"]:
        lines.append("Saved artifacts contain outcomes and frame observations required for CRAVE rollout metrics.")
    else:
        lines.append(
            "Historical artifacts support outcome and episode-duration analysis only. They do not support "
            "post-hoc CRAVE progress, density, stall-lead-time, or regression analyses."
        )
        lines.extend(["", "## Required Follow-up", ""])
        lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    result = audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
