#!/usr/bin/env python3
"""Annotate the frozen six-task P0 panel with preregistered CRAVE-v2 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


VALLEY_PROMINENCE = 0.03
VALLEY_SMOOTHING_SIGMA = 1.4
PROBE_TRAIN_PAIRS_PER_TASK = 1365
PROBE_SELECTION_SEED = 20260804


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def reference_ranges(reference_features: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    offsets = np.cumsum([0, *[len(values) for values in reference_features]])
    ranges = [(int(offsets[i]), int(offsets[i + 1])) for i in range(len(reference_features))]
    return l2_normalize(np.concatenate(reference_features)), ranges


def per_episode_nearest(
    query: np.ndarray,
    reference: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distance and within-episode nearest-frame index for every reference episode."""
    reference_tensor = torch.from_numpy(reference).to(device)
    all_distance = []
    all_index = []
    query = l2_normalize(query)
    with torch.inference_mode():
        for start in range(0, len(query), chunk_size):
            current = torch.from_numpy(query[start : start + chunk_size]).to(device)
            # L2 distance between unit vectors, without materializing a quadratic
            # reference-reference matrix.
            distance = torch.sqrt(
                torch.clamp(2.0 - 2.0 * (current @ reference_tensor.T), min=0.0)
            )
            chunk_distance = []
            chunk_index = []
            for lower, upper in ranges:
                value, index = torch.min(distance[:, lower:upper], dim=1)
                chunk_distance.append(value)
                chunk_index.append(index)
            all_distance.append(torch.stack(chunk_distance, dim=1).cpu().numpy())
            all_index.append(torch.stack(chunk_index, dim=1).cpu().numpy())
    return np.concatenate(all_distance), np.concatenate(all_index)


def fit_reference_sigma(
    reference_features: list[np.ndarray], *, device: str, chunk_size: int
) -> tuple[
    float,
    np.ndarray,
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    reference, ranges = reference_ranges(reference_features)
    distance, nearest_index = per_episode_nearest(
        reference, reference, ranges, device=device, chunk_size=chunk_size
    )
    episode_index = np.empty(len(reference), dtype=np.int32)
    for index, (lower, upper) in enumerate(ranges):
        episode_index[lower:upper] = index
    distance[np.arange(len(reference)), episode_index] = np.nan
    sigma = float(np.nanmedian(distance))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"invalid CRAVE median-heuristic sigma: {sigma}")
    return sigma, reference, ranges, distance, nearest_index, episode_index


def fields_from_nearest(
    distance: np.ndarray,
    nearest_index: np.ndarray,
    ranges: list[tuple[int, int]],
    sigma: float,
    *,
    exclude_episode: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.exp(-(distance**2) / (2.0 * sigma**2)).astype(np.float64)
    if exclude_episode is not None:
        weights[np.arange(len(weights)), exclude_episode] = 0.0
    reference_time = np.stack(
        [
            nearest_index[:, index] / max(1, upper - lower - 1)
            for index, (lower, upper) in enumerate(ranges)
        ],
        axis=1,
    )
    denominator = np.maximum(weights.sum(axis=1), 1e-12)
    density_denominator = weights.shape[1] - (1 if exclude_episode is not None else 0)
    density = weights.sum(axis=1) / density_denominator
    progress = (weights * reference_time).sum(axis=1) / denominator
    return density.astype(np.float32), progress.astype(np.float32)


def query_recurrence_fields(
    query: np.ndarray,
    reference: np.ndarray,
    ranges: list[tuple[int, int]],
    sigma: float,
    *,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance, nearest_index = per_episode_nearest(
        query, reference, ranges, device=device, chunk_size=chunk_size
    )
    return fields_from_nearest(
        distance, nearest_index, ranges, sigma
    )


def stable_probe_pairs(
    episodes: list[int], lengths: list[int], horizon: int, count: int
) -> list[tuple[int, int, int]]:
    candidates = [
        (episode, frame, frame + horizon)
        for episode, length in zip(episodes, lengths, strict=True)
        for frame in range(max(0, length - horizon))
    ]
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} valid probe pairs for requested {count}")
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            f"{PROBE_SELECTION_SEED}:{row[0]}:{row[1]}:{row[2]}".encode()
        ).digest(),
    )[:count]


def episode_boundaries(density: np.ndarray) -> np.ndarray:
    if len(density) < 3:
        return np.empty(0, dtype=np.int32)
    valleys, _ = find_peaks(
        -gaussian_filter1d(density.astype(np.float64), VALLEY_SMOOTHING_SIGMA),
        prominence=VALLEY_PROMINENCE,
        distance=max(2, len(density) // 12),
    )
    return valleys.astype(np.int32)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    panel_npz = np.load(args.panel)
    panel = {key: panel_npz[key] for key in panel_npz.files}
    count = len(panel["cur_ep"])
    outputs = {
        "current_progress": np.full(count, np.nan, dtype=np.float32),
        "target_progress": np.full(count, np.nan, dtype=np.float32),
        "progress_change": np.full(count, np.nan, dtype=np.float32),
        "current_recurrence_density": np.full(count, np.nan, dtype=np.float32),
        "target_recurrence_density": np.full(count, np.nan, dtype=np.float32),
        "phase_boundary_crossing": np.zeros(count, dtype=bool),
        "current_phase": np.full(count, -1, dtype=np.int16),
        "target_phase": np.full(count, -1, dtype=np.int16),
        "normalized_current_time": np.full(count, np.nan, dtype=np.float32),
        "normalized_target_time": np.full(count, np.nan, dtype=np.float32),
        "physical_task": np.full(count, -1, dtype=np.int16),
    }
    task_reports = {}
    probe_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "cur_ep",
            "cur_fi",
            "tgt_fi",
            "physical_task",
            "current_progress",
            "target_progress",
            "progress_change",
            "current_recurrence_density",
            "target_recurrence_density",
            "phase_boundary_crossing",
            "current_phase",
            "target_phase",
            "normalized_current_time",
            "normalized_target_time",
        )
    }
    reference_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "episode",
            "frame",
            "physical_task",
            "recurrence_density",
            "progress",
            "phase_boundary",
            "phase",
        )
    }
    task_names = sorted(selection["tasks"])
    task_to_id = {task: index for index, task in enumerate(task_names)}
    horizon_values = np.unique(panel["tgt_fi"] - panel["cur_fi"])
    if not np.array_equal(horizon_values, [50]):
        raise ValueError(f"R0 protocol requires the frozen 50-frame horizon, got {horizon_values}")
    horizon = 50

    for task in task_names:
        task_spec = selection["tasks"][task]
        task_id = task_to_id[task]
        reference_episodes = list(map(int, task_spec["reference_episodes"]))
        heldout_episodes = list(map(int, task_spec["heldout_episodes"]))
        reference_features = [
            np.load(args.feature_dir / f"ep{episode}.npz")["pooled"]
            for episode in reference_episodes
        ]
        sigma, reference, ranges, ref_distance, ref_nearest, ref_episode_index = fit_reference_sigma(
            reference_features, device=args.device, chunk_size=args.chunk_size
        )
        reference_density, reference_progress = fields_from_nearest(
            ref_distance,
            ref_nearest,
            ranges,
            sigma,
            exclude_episode=ref_episode_index,
        )
        reference_fields = {}
        for index, episode in enumerate(reference_episodes):
            lower, upper = ranges[index]
            ep_density = reference_density[lower:upper]
            ep_progress = reference_progress[lower:upper]
            boundaries = episode_boundaries(ep_density)
            reference_fields[episode] = (ep_density, ep_progress, boundaries)
            frame = np.arange(len(ep_density), dtype=np.int32)
            boundary_mask = np.zeros(len(ep_density), dtype=bool)
            boundary_mask[boundaries] = True
            reference_parts["episode"].append(
                np.full(len(ep_density), episode, dtype=np.int32)
            )
            reference_parts["frame"].append(frame)
            reference_parts["physical_task"].append(
                np.full(len(ep_density), task_id, dtype=np.int16)
            )
            reference_parts["recurrence_density"].append(ep_density.astype(np.float32))
            reference_parts["progress"].append(ep_progress.astype(np.float32))
            reference_parts["phase_boundary"].append(boundary_mask)
            reference_parts["phase"].append(
                np.searchsorted(boundaries, frame, side="right").astype(np.int16)
            )
        heldout_features = [
            np.load(args.feature_dir / f"ep{episode}.npz")["pooled"]
            for episode in heldout_episodes
        ]
        heldout_offsets = np.cumsum([0, *[len(values) for values in heldout_features]])
        density, progress = query_recurrence_fields(
            np.concatenate(heldout_features),
            reference,
            ranges,
            sigma,
            device=args.device,
            chunk_size=args.chunk_size,
        )
        episode_fields = {}
        boundary_total = 0
        for ep_index, episode in enumerate(heldout_episodes):
            lower, upper = int(heldout_offsets[ep_index]), int(heldout_offsets[ep_index + 1])
            boundaries = episode_boundaries(density[lower:upper])
            boundary_total += len(boundaries)
            episode_fields[episode] = (
                density[lower:upper], progress[lower:upper], boundaries
            )

        task_rows = np.flatnonzero(np.isin(panel["cur_ep"], heldout_episodes))
        outputs["physical_task"][task_rows] = task_id
        for row in task_rows:
            episode = int(panel["cur_ep"][row])
            current = int(panel["cur_fi"][row])
            target = int(panel["tgt_fi"][row])
            ep_density, ep_progress, boundaries = episode_fields[episode]
            if not (0 <= current < len(ep_density) and 0 <= target < len(ep_density)):
                raise IndexError(f"{task} ep{episode}: pair {current}->{target} outside {len(ep_density)}")
            outputs["current_progress"][row] = ep_progress[current]
            outputs["target_progress"][row] = ep_progress[target]
            outputs["progress_change"][row] = ep_progress[target] - ep_progress[current]
            outputs["current_recurrence_density"][row] = ep_density[current]
            outputs["target_recurrence_density"][row] = ep_density[target]
            outputs["phase_boundary_crossing"][row] = bool(
                np.any((boundaries > current) & (boundaries <= target))
            )
            outputs["current_phase"][row] = int(np.searchsorted(boundaries, current, side="right"))
            outputs["target_phase"][row] = int(np.searchsorted(boundaries, target, side="right"))
            outputs["normalized_current_time"][row] = current / max(1, len(ep_density) - 1)
            outputs["normalized_target_time"][row] = target / max(1, len(ep_density) - 1)

        probe_pairs = stable_probe_pairs(
            reference_episodes,
            [len(values) for values in reference_features],
            horizon,
            PROBE_TRAIN_PAIRS_PER_TASK,
        )
        probe = {name: [] for name in probe_parts}
        for episode, current, target in probe_pairs:
            ep_density, ep_progress, boundaries = reference_fields[episode]
            probe["cur_ep"].append(episode)
            probe["cur_fi"].append(current)
            probe["tgt_fi"].append(target)
            probe["physical_task"].append(task_id)
            probe["current_progress"].append(ep_progress[current])
            probe["target_progress"].append(ep_progress[target])
            probe["progress_change"].append(ep_progress[target] - ep_progress[current])
            probe["current_recurrence_density"].append(ep_density[current])
            probe["target_recurrence_density"].append(ep_density[target])
            probe["phase_boundary_crossing"].append(
                bool(np.any((boundaries > current) & (boundaries <= target)))
            )
            probe["current_phase"].append(int(np.searchsorted(boundaries, current, side="right")))
            probe["target_phase"].append(int(np.searchsorted(boundaries, target, side="right")))
            probe["normalized_current_time"].append(current / max(1, len(ep_density) - 1))
            probe["normalized_target_time"].append(target / max(1, len(ep_density) - 1))
        for name in probe_parts:
            dtype = bool if name == "phase_boundary_crossing" else np.float32
            if name in {"cur_ep", "cur_fi", "tgt_fi"}:
                dtype = np.int32
            elif name in {"physical_task", "current_phase", "target_phase"}:
                dtype = np.int16
            probe_parts[name].append(np.asarray(probe[name], dtype=dtype))

        task_reports[task] = {
            "sigma": sigma,
            "reference_episodes": len(reference_episodes),
            "heldout_episodes": len(heldout_episodes),
            "pair_rows": len(task_rows),
            "probe_train_pairs": len(probe_pairs),
            "detected_boundaries": boundary_total,
            "mean_boundaries_per_episode": boundary_total / len(heldout_episodes),
            "progress_vs_normalized_time_pearson": pearson(
                outputs["current_progress"][task_rows],
                outputs["normalized_current_time"][task_rows],
            ),
            "progress_change_vs_normalized_time_change_pearson": pearson(
                outputs["progress_change"][task_rows],
                outputs["normalized_target_time"][task_rows]
                - outputs["normalized_current_time"][task_rows],
            ),
        }
        print(json.dumps({task: task_reports[task]}, sort_keys=True), flush=True)

    for key, values in outputs.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"non-finite or uncovered R0 target: {key}")
        if key in {"current_phase", "target_phase", "physical_task"} and np.any(values < 0):
            raise ValueError(f"uncovered R0 phase target: {key}")

    args.output.mkdir(parents=True, exist_ok=True)
    labels_path = args.output / "labels.npz"
    temporary_labels = args.output / f".labels.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary_labels, **panel, **outputs)
    temporary_labels.replace(labels_path)
    probe_path = args.output / "probe_train.npz"
    temporary_probe = args.output / f".probe_train.{os.getpid()}.tmp.npz"
    probe_arrays = {name: np.concatenate(parts) for name, parts in probe_parts.items()}
    np.savez_compressed(temporary_probe, **probe_arrays)
    temporary_probe.replace(probe_path)
    reference_path = args.output / "reference_trajectories.npz"
    temporary_reference = args.output / f".reference_trajectories.{os.getpid()}.tmp.npz"
    reference_arrays = {
        name: np.concatenate(parts) for name, parts in reference_parts.items()
    }
    np.savez_compressed(temporary_reference, **reference_arrays)
    temporary_reference.replace(reference_path)
    report = {
        "schema_version": 1,
        "protocol": "pi05_crave_r0_labels_v1",
        "teacher_scope": "offline full-episode annotation; runtime predictors remain causal",
        "recurrence_density": "CRAVE-v2 equal-episode Gaussian kernel over cross-reference nearest-frame L2 distances",
        "progress": "equal-episode kernel-weighted nearest-reference normalized progress with no temporal band",
        "boundary": "per-heldout-episode valleys of recurrence density",
        "valley_prominence": VALLEY_PROMINENCE,
        "valley_smoothing_sigma": VALLEY_SMOOTHING_SIGMA,
        "time_role": "normalized time is recorded only as an explicit shortcut control",
        "pair_rows": count,
        "probe_train_pair_rows": int(len(probe_arrays["cur_ep"])),
        "reference_trajectory_rows": int(len(reference_arrays["episode"])),
        "reference_trajectory_episodes": int(
            len(np.unique(reference_arrays["episode"]))
        ),
        "probe_train_pairs_per_task": PROBE_TRAIN_PAIRS_PER_TASK,
        "probe_selection_seed": PROBE_SELECTION_SEED,
        "probe_split": "train-reference episodes only; no heldout episode enters readout fitting",
        "physical_task_ids": task_to_id,
        "coverage": 1.0,
        "positive_boundary_crossing_rate": float(outputs["phase_boundary_crossing"].mean()),
        "selection_manifest_sha256": sha256(args.selection),
        "source_panel_sha256": sha256(args.panel),
        "labels_sha256": sha256(labels_path),
        "probe_train_sha256": sha256(probe_path),
        "reference_trajectories_sha256": sha256(reference_path),
        "tasks": task_reports,
    }
    atomic_text(
        args.output / "labels_manifest.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(args.output / "READY_LABELS", "ready\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
