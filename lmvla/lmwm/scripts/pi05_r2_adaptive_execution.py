#!/usr/bin/env python3
"""Causal recurrence signals and query-budgeted action execution for pi0.5 R2."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class R2ScheduleConfig:
    action_horizon: int = 8
    fixed_horizon: int = 4
    stable_horizon: int = 8
    caution_horizon: int = 2
    emergency_horizon: int = 1
    stall_window_steps: int = 50
    stall_progress_epsilon: float = 0.02
    regression_progress_drop: float = 0.10
    confidence_floor: float = 0.05
    boundary_proximity_threshold: float = 0.50
    progress_ema_alpha: float = 0.35
    max_query_debt: int = 1

    def validate(self) -> None:
        horizons = (
            self.fixed_horizon,
            self.stable_horizon,
            self.caution_horizon,
            self.emergency_horizon,
        )
        if any(value <= 0 or value > self.action_horizon for value in horizons):
            raise ValueError(f"execution horizons must be in [1, {self.action_horizon}]")
        if self.stable_horizon < self.fixed_horizon:
            raise ValueError("stable horizon must be at least the fixed baseline horizon")
        if self.caution_horizon > self.fixed_horizon:
            raise ValueError("caution horizon must not exceed the fixed baseline horizon")
        if self.emergency_horizon > self.caution_horizon:
            raise ValueError("emergency horizon must not exceed the caution horizon")
        if self.stall_window_steps <= 0:
            raise ValueError("stall window must be positive")
        if not 0.0 < self.progress_ema_alpha <= 1.0:
            raise ValueError("EMA alpha must be in (0, 1]")
        if self.max_query_debt < 0:
            raise ValueError("max query debt must be nonnegative")


@dataclass(frozen=True)
class RecurrenceSignal:
    step: int
    progress: float
    density: float
    confidence: float
    boundary_proximity: float
    smoothed_progress: float
    stall: bool
    regression: bool
    low_confidence: bool


@dataclass(frozen=True)
class ExecutionDecision:
    horizon: int
    force_replan: bool
    reason: str
    query_allowance: int
    queries_after_replan: int
    query_debt_after_replan: int


class CausalRecurrenceReadout:
    """Equal-episode CRAVE readout packaged for one RoboTwin task.

    The caller supplies the current frozen DINO feature only. Reference vectors,
    progress, density, and boundary anchors are all fitted from train-reference
    demonstrations; no held-out or future observation enters a query.
    """

    def __init__(
        self,
        *,
        reference_features: np.ndarray,
        episode_offsets: np.ndarray,
        reference_progress: np.ndarray,
        reference_density: np.ndarray,
        sigma: float,
        density_calibration: np.ndarray,
        boundary_progress: np.ndarray,
    ) -> None:
        features = np.asarray(reference_features, dtype=np.float32)
        offsets = np.asarray(episode_offsets, dtype=np.int64)
        progress = np.asarray(reference_progress, dtype=np.float32)
        density = np.asarray(reference_density, dtype=np.float32)
        calibration = np.sort(np.asarray(density_calibration, dtype=np.float32))
        boundaries = np.sort(np.asarray(boundary_progress, dtype=np.float32))
        if features.ndim != 2 or features.shape[1] <= 0:
            raise ValueError(f"reference features must be [N,D], got {features.shape}")
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(features):
            raise ValueError("episode offsets must span every reference row")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("reference episodes must be nonempty")
        if progress.shape != (len(features),) or density.shape != (len(features),):
            raise ValueError("reference fields must align with reference features")
        if not np.isfinite(features).all() or not np.isfinite(progress).all() or not np.isfinite(density).all():
            raise ValueError("reference readout contains non-finite values")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"invalid recurrence sigma: {sigma}")
        if len(calibration) == 0 or not np.isfinite(calibration).all():
            raise ValueError("density calibration must be finite and nonempty")
        if len(boundaries) and not np.isfinite(boundaries).all():
            raise ValueError("boundary anchors must be finite")

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.reference_features = features / np.maximum(norms, 1e-9)
        self.episode_offsets = offsets
        self.reference_progress = progress
        self.reference_density = density
        self.sigma = float(sigma)
        self.density_calibration = calibration
        self.boundary_progress = boundaries

    def query(self, feature: np.ndarray) -> dict[str, float]:
        current = np.asarray(feature, dtype=np.float32).reshape(-1)
        if current.shape != (self.reference_features.shape[1],):
            raise ValueError(
                f"current feature has shape {current.shape}; expected {(self.reference_features.shape[1],)}"
            )
        if not np.isfinite(current).all():
            raise ValueError("current feature contains non-finite values")
        current = current / max(float(np.linalg.norm(current)), 1e-9)
        distance = np.sqrt(
            np.maximum(2.0 - 2.0 * (self.reference_features @ current), 0.0)
        )
        episode_distance = []
        nearest_rows = []
        for lower, upper in zip(self.episode_offsets[:-1], self.episode_offsets[1:], strict=True):
            relative = int(np.argmin(distance[lower:upper]))
            row = int(lower + relative)
            episode_distance.append(float(distance[row]))
            nearest_rows.append(row)
        weights = np.exp(
            -(np.square(np.asarray(episode_distance, dtype=np.float64)))
            / (2.0 * self.sigma**2)
        )
        denominator = max(float(weights.sum()), 1e-12)
        rows = np.asarray(nearest_rows, dtype=np.int64)
        progress = float(np.dot(weights, self.reference_progress[rows]) / denominator)
        density = float(weights.mean())
        rank = int(np.searchsorted(self.density_calibration, density, side="right"))
        confidence = rank / len(self.density_calibration)
        ahead = self.boundary_progress[self.boundary_progress >= progress]
        if len(ahead):
            boundary_distance = float(ahead[0] - progress)
            boundary_proximity = float(np.exp(-boundary_distance / 0.05))
        else:
            boundary_distance = float("inf")
            boundary_proximity = 0.0
        return {
            "progress": progress,
            "density": density,
            "confidence": confidence,
            "boundary_distance": boundary_distance,
            "boundary_proximity": boundary_proximity,
        }


class CausalExecutionController:
    """Streaming event detector and adaptive action-chunk controller.

    Query allowance follows the preregistered fixed-four baseline. One emergency
    query may borrow from the future; stable eight-step chunks then repay that
    debt. This prevents an adaptive arm from obtaining unconstrained policy
    inference while still allowing immediate reaction to a newly observed event.
    """

    def __init__(self, config: R2ScheduleConfig | None = None) -> None:
        self.config = config or R2ScheduleConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self._smoothed_progress: float | None = None
        self._running_max = -np.inf
        self._progress_history: deque[tuple[int, float]] = deque()
        self._last_signal: RecurrenceSignal | None = None
        self._last_event_reason: str | None = None
        self._queries = 0
        self._executed_steps = 0
        self._decision_counts: dict[str, int] = {
            "initial": 0,
            "stable": 0,
            "boundary": 0,
            "low_confidence": 0,
            "stall": 0,
            "regression": 0,
            "budget_defer": 0,
            "budget_repay": 0,
        }

    def observe(
        self,
        *,
        step: int,
        progress: float,
        density: float,
        confidence: float,
        boundary_proximity: float,
    ) -> RecurrenceSignal:
        values = np.asarray([progress, density, confidence, boundary_proximity], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("recurrence signal contains non-finite values")
        if step < self._executed_steps:
            raise ValueError(f"nonmonotonic action step {step} < {self._executed_steps}")
        self._executed_steps = int(step)
        alpha = self.config.progress_ema_alpha
        if self._smoothed_progress is None:
            self._smoothed_progress = float(progress)
        else:
            self._smoothed_progress = alpha * float(progress) + (1.0 - alpha) * self._smoothed_progress
        self._running_max = max(self._running_max, self._smoothed_progress)
        self._progress_history.append((int(step), self._smoothed_progress))
        lower = int(step) - self.config.stall_window_steps
        while len(self._progress_history) > 1 and self._progress_history[1][0] <= lower:
            self._progress_history.popleft()
        stall = False
        if self._progress_history and self._progress_history[0][0] <= lower:
            stall = (
                self._smoothed_progress - self._progress_history[0][1]
                <= self.config.stall_progress_epsilon
            )
        regression = (
            self._running_max - self._smoothed_progress
            >= self.config.regression_progress_drop
        )
        signal = RecurrenceSignal(
            step=int(step),
            progress=float(progress),
            density=float(density),
            confidence=float(confidence),
            boundary_proximity=float(boundary_proximity),
            smoothed_progress=float(self._smoothed_progress),
            stall=bool(stall),
            regression=bool(regression),
            low_confidence=bool(confidence < self.config.confidence_floor),
        )
        self._last_signal = signal
        return signal

    def query_allowance(self, step: int | None = None) -> int:
        resolved_step = self._executed_steps if step is None else int(step)
        return 1 + resolved_step // self.config.fixed_horizon

    def decide(self, *, cache_remaining: int) -> ExecutionDecision:
        signal = self._last_signal
        if signal is None:
            reason = "initial"
            horizon = self.config.fixed_horizon
            event = True
        elif signal.regression:
            reason = "regression"
            horizon = self.config.emergency_horizon
            event = True
        elif signal.stall:
            reason = "stall"
            horizon = self.config.emergency_horizon
            event = True
        elif signal.low_confidence:
            reason = "low_confidence"
            horizon = self.config.caution_horizon
            event = True
        elif signal.boundary_proximity >= self.config.boundary_proximity_threshold:
            reason = "boundary"
            horizon = self.config.caution_horizon
            event = True
        else:
            reason = "stable"
            horizon = self.config.stable_horizon
            event = False

        event_edge = bool(event and reason != self._last_event_reason)
        cache_empty = int(cache_remaining) <= 0
        wants_query = cache_empty or event_edge
        allowance = self.query_allowance()
        maximum = allowance + self.config.max_query_debt
        debt_before = max(0, self._queries - allowance)
        if cache_empty and debt_before > 0 and event and not event_edge:
            reason = "budget_repay"
            horizon = self.config.stable_horizon
        force_replan = bool(cache_empty or (event_edge and self._queries < maximum))
        if event_edge and not cache_empty and not force_replan:
            reason = "budget_defer"
            horizon = max(1, int(cache_remaining))
        if force_replan:
            self._queries += 1
            self._decision_counts[reason] = self._decision_counts.get(reason, 0) + 1
        self._last_event_reason = reason if event else None
        debt = max(0, self._queries - allowance)
        return ExecutionDecision(
            horizon=int(horizon),
            force_replan=force_replan,
            reason=reason,
            query_allowance=allowance,
            queries_after_replan=self._queries,
            query_debt_after_replan=debt,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "executed_steps": self._executed_steps,
            "queries": self._queries,
            "query_allowance": self.query_allowance(),
            "query_debt": max(0, self._queries - self.query_allowance()),
            "decision_counts": dict(self._decision_counts),
            "last_signal": asdict(self._last_signal) if self._last_signal is not None else None,
        }
