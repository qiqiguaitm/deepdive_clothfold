from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TransitionLabel:
    task: int
    current: int
    next: int
    available: bool


class FrozenStageProfile:
    """Task-local stage boundaries mined only from the frozen recurrence artifact."""

    def __init__(self, pairs_path: str | Path, episodes_path: str | Path):
        lengths = {}
        with Path(episodes_path).open() as stream:
            for line in stream:
                episode = json.loads(line)
                lengths[int(episode["episode_index"])] = int(episode["length"])

        pairs = np.load(pairs_path)
        task = np.asarray(pairs["pair_task"], dtype=np.int64)
        episode = np.asarray(pairs["cur_ep"], dtype=np.int64)
        frame = np.asarray(pairs["cur_fi"], dtype=np.float64)
        stage = np.asarray(pairs["cur_ms"], dtype=np.int64)
        episode_length = np.asarray([lengths[int(value)] for value in episode], dtype=np.float64)
        progress = frame / np.maximum(episode_length - 1.0, 1.0)

        self._centers: dict[int, np.ndarray] = {}
        self._stage_ids: dict[int, np.ndarray] = {}
        for task_id in np.unique(task):
            task_mask = task == task_id
            stage_ids = np.unique(stage[task_mask])
            centers = np.asarray(
                [np.median(progress[task_mask & (stage == stage_id)]) for stage_id in stage_ids],
                dtype=np.float64,
            )
            order = np.argsort(centers)
            self._centers[int(task_id)] = centers[order]
            self._stage_ids[int(task_id)] = stage_ids[order]

    def stage_count(self, task_id: int) -> int:
        return int(self._stage_ids[int(task_id)].size)

    def stage_at(self, task_id: int, progress: float) -> int:
        task_id = int(task_id)
        centers = self._centers[task_id]
        stage_ids = self._stage_ids[task_id]
        if centers.size == 1:
            return int(stage_ids[0])
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        index = int(np.searchsorted(boundaries, np.clip(progress, 0.0, 1.0), side="right"))
        return int(stage_ids[index])

    def label(self, task_id: int, progress: float, intervention: str = "correct") -> TransitionLabel:
        task_id = int(task_id)
        current = self.stage_at(task_id, progress)
        if intervention == "null":
            return TransitionLabel(task=-1, current=-1, next=-1, available=False)
        if intervention == "within-task":
            count = self.stage_count(task_id)
            current = (current + max(1, count // 2)) % count
        elif intervention == "cross-task":
            task_ids = sorted(self._stage_ids)
            task_id = task_ids[(task_ids.index(task_id) + 1) % len(task_ids)]
            current = self.stage_at(task_id, progress)
        elif intervention != "correct":
            raise ValueError(f"unsupported transition intervention: {intervention}")
        return TransitionLabel(
            task=task_id,
            current=current,
            next=min(current + 1, self.stage_count(task_id)),
            available=True,
        )


class MonotonicExpertJointTracker:
    """Align policy state to a privileged same-scene expert trajectory without using rollout time."""

    def __init__(self, trajectory: np.ndarray, *, search_window: int | None = None):
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2:
            raise ValueError(f"expert trajectory must be [T,D] with T>=2, got {trajectory.shape}")
        self.trajectory = trajectory
        self.search_window = search_window
        self.cursor = 0
        self.scale = np.maximum(np.std(trajectory, axis=0), 1e-3)

    def update(self, state: np.ndarray) -> float:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape[0] != self.trajectory.shape[1]:
            raise ValueError(
                f"state width {state.shape[0]} does not match trajectory width {self.trajectory.shape[1]}"
            )
        stop = self.trajectory.shape[0]
        if self.search_window is not None:
            stop = min(stop, self.cursor + int(self.search_window) + 1)
        candidates = self.trajectory[self.cursor : stop]
        distances = np.mean(np.square((candidates - state[None]) / self.scale[None]), axis=1)
        self.cursor += int(np.argmin(distances))
        return self.cursor / (self.trajectory.shape[0] - 1)
