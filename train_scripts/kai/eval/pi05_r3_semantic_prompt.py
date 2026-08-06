#!/usr/bin/env python3
"""Format preregistered R3 semantic-subtask prompt interventions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_MODES = {"none", "generic-stage", "semantic-current", "semantic-next"}


class SemanticPromptFormatter:
    def __init__(self, vocabulary_path: Path, task_map_path: Path, mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"unsupported R3 prompt mode {mode!r}; expected {sorted(VALID_MODES)}")
        self.mode = mode
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        task_map = json.loads(task_map_path.read_text(encoding="utf-8"))
        self.task_name_by_id = {int(value): str(key) for key, value in task_map.items()}
        self.events: dict[int, dict[int, str]] = {}
        for row in vocabulary:
            task_id = int(task_map[row["task"]])
            self.events.setdefault(task_id, {})[int(row["local_id"])] = str(row["name"])
        missing = set(self.task_name_by_id) - set(self.events)
        if missing:
            raise ValueError(f"vocabulary has no events for task IDs {sorted(missing)}")

    @staticmethod
    def _scalar(observation: dict[str, Any], key: str, default: int = -1) -> int:
        value = observation.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _readable(name: str) -> str:
        return name.replace("_", " ")

    def format(self, base_prompt: str, observation: dict[str, Any]) -> str:
        base_prompt = str(base_prompt).strip()
        if self.mode == "none" or not bool(observation.get("lmwm_transition_mask", False)):
            return base_prompt

        task_id = self._scalar(observation, "lmwm_transition_task")
        current = self._scalar(observation, "lmwm_transition_current")
        next_stage = self._scalar(observation, "lmwm_transition_next")
        if task_id not in self.events:
            raise ValueError(f"unknown semantic task ID {task_id}")
        event_map = self.events[task_id]

        if self.mode == "generic-stage":
            suffix = f"Current stage ID: {current}. Next stage ID: {next_stage}."
        elif self.mode == "semantic-current":
            if current not in event_map:
                raise ValueError(f"task {task_id} has no semantic event {current}")
            suffix = f"Current subtask: {self._readable(event_map[current])}."
        else:
            if next_stage in event_map:
                target = self._readable(event_map[next_stage])
            elif next_stage == len(event_map):
                target = "complete the task"
            else:
                raise ValueError(f"task {task_id} has no next semantic event {next_stage}")
            suffix = f"Next subtask: {target}."
        return f"{base_prompt} {suffix}".strip()
