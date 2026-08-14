"""Backward-compatible reader for the KAI0 rollout outcome contract.

The contract lives in ``meta/episodes.jsonl`` rather than parquet so outcome
labels can be added after recording without rewriting frame payloads.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VALID_LABELS = frozenset({"success", "partial_success", "failure", "aborted"})
VALID_MODES = frozenset({"demonstration", "autonomous", "intervention", "recovery"})


def normalize_rollout_outcome(episode: dict[str, Any]) -> dict[str, Any]:
    """Normalize a v1 outcome, or derive one from legacy ``success`` metadata."""
    raw = episode.get("rollout_outcome")
    if raw is None:
        raw = {
            "schema_version": SCHEMA_VERSION,
            "label": "success" if episode.get("success", True) else "failure",
            "rollout_mode": "demonstration",
        }
    if not isinstance(raw, dict):
        raise ValueError("rollout_outcome must be an object")
    version = int(raw.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported rollout outcome schema_version={version}")
    label = str(raw.get("label", ""))
    mode = str(raw.get("rollout_mode", ""))
    if label not in VALID_LABELS:
        raise ValueError(f"invalid rollout outcome label={label!r}")
    if mode not in VALID_MODES:
        raise ValueError(f"invalid rollout mode={mode!r}")
    return {
        "schema_version": version,
        "label": label,
        "rollout_mode": mode,
        "stage_outcomes": list(raw.get("stage_outcomes") or []),
        "failure_modes": list(raw.get("failure_modes") or []),
        "intervention_count": int(raw.get("intervention_count", 0)),
        "recovery_success": raw.get("recovery_success"),
        "unsafe_event": bool(raw.get("unsafe_event", False)),
        "time_limit_reached": bool(raw.get("time_limit_reached", False)),
    }


def iter_episode_outcomes(path: str | Path) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(episode_metadata, normalized_outcome)`` from a JSONL manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                yield episode, normalize_rollout_outcome(episode)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid episode outcome at {path}:{line_number}: {exc}") from exc
