#!/usr/bin/env python3
"""Freeze source-grounded semantic events on unmodified CRAVE boundaries."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


TASK_PLANS: dict[str, dict[str, Any]] = {
    "beat_block_hammer": {
        "source_trace": [("grasp_actor", "hammer"), ("place_actor", "hammer")],
        "events": [
            ("acquire_hammer", "hammer_on_table", "hammer_held_by_active_arm"),
            ("strike_block_with_hammer", "block_unstruck", "hammer_contacts_block"),
        ],
    },
    "blocks_ranking_rgb": {
        "source_trace": [
            ("pick_and_place_block", "block1"),
            ("pick_and_place_block", "block2"),
            ("pick_and_place_block", "block3"),
        ],
        "events": [
            ("place_red_block_left", "red_block_unranked", "red_block_leftmost"),
            ("place_green_block_middle", "green_block_unranked", "red_left_of_green"),
            ("place_blue_block_right", "blue_block_unranked", "red_left_of_green_left_of_blue"),
        ],
    },
    "blocks_ranking_size": {
        "source_trace": [
            ("pick_and_place_block", "block3"),
            ("pick_and_place_block", "block2"),
            ("pick_and_place_block", "block1"),
        ],
        "events": [
            ("place_small_block_right", "small_block_unranked", "small_block_rightmost"),
            ("place_medium_block_middle", "medium_block_unranked", "medium_left_of_small"),
            ("place_large_block_left", "large_block_unranked", "large_left_of_medium_left_of_small"),
        ],
    },
    "handover_block": {
        "source_trace": [
            ("grasp_actor", "box"),
            ("place_actor", "box"),
            ("grasp_actor", "box"),
            ("open_gripper", "grasp_arm_tag"),
            ("place_actor", "box"),
        ],
        "events": [
            ("acquire_red_block_left", "red_block_on_table", "red_block_held_left"),
            ("transfer_red_block_left_to_right", "red_block_held_left", "red_block_held_right"),
            ("place_red_block_on_blue_pad", "red_block_held_right", "red_block_on_blue_pad"),
        ],
    },
    "stack_blocks_two": {
        "source_trace": [
            ("pick_and_place_block", "block1"),
            ("pick_and_place_block", "block2"),
        ],
        "events": [
            ("place_red_base_at_center", "red_block_on_table", "red_block_center_base"),
            ("stack_green_on_red", "green_block_on_table", "green_on_red"),
        ],
    },
    "stack_blocks_three": {
        "source_trace": [
            ("pick_and_place_block", "block1"),
            ("pick_and_place_block", "block2"),
            ("pick_and_place_block", "block3"),
        ],
        "events": [
            ("place_red_base_at_center", "red_block_on_table", "red_block_center_base"),
            ("stack_green_on_red", "green_block_on_table", "green_on_red"),
            ("stack_blue_on_green", "blue_block_on_table", "blue_on_green_on_red"),
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expression_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


def play_once_trace(source: Path) -> list[tuple[str, str]]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    play_once = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "play_once"
    )
    relevant = {"pick_and_place_block", "grasp_actor", "place_actor", "open_gripper"}
    calls = []
    for node in ast.walk(play_once):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        operation = node.func.attr
        if operation not in relevant or not node.args:
            continue
        calls.append((node.lineno, operation, expression_name(node.args[0])))
    return [(operation, target) for _, operation, target in sorted(calls)]


def monotonic_event_assignment(progress: np.ndarray, event_count: int) -> np.ndarray:
    """Assign every CRAVE segment to an ordered event, using all events exactly once."""
    progress = np.asarray(progress, dtype=np.float64)
    segment_count = len(progress)
    if event_count <= 0 or segment_count < event_count:
        raise ValueError(f"cannot align {segment_count} segments to {event_count} events")
    centers = (np.arange(event_count, dtype=np.float64) + 0.5) / event_count
    cost = (progress[:, None] - centers[None, :]) ** 2
    infinity = float("inf")
    dp = np.full((segment_count, event_count), infinity)
    previous = np.full((segment_count, event_count), -1, dtype=np.int32)
    dp[0, 0] = cost[0, 0]
    for segment in range(1, segment_count):
        for event in range(event_count):
            for prior in (event, event - 1):
                if prior < 0 or not np.isfinite(dp[segment - 1, prior]):
                    continue
                candidate = dp[segment - 1, prior] + cost[segment, event]
                if candidate < dp[segment, event]:
                    dp[segment, event] = candidate
                    previous[segment, event] = prior
    if not np.isfinite(dp[-1, -1]):
        raise ValueError("no surjective monotonic event alignment")
    assignment = np.empty(segment_count, dtype=np.int16)
    event = event_count - 1
    for segment in range(segment_count - 1, -1, -1):
        assignment[segment] = event
        if segment:
            event = int(previous[segment, event])
    if set(map(int, assignment)) != set(range(event_count)):
        raise AssertionError("alignment did not cover every semantic event")
    return assignment


def action_signature(action: np.ndarray) -> dict[str, float | str]:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"expected [T,14] action, got {action.shape}")
    delta = np.abs(np.diff(action, axis=0)) if len(action) > 1 else np.zeros((1, 14))
    left_energy = float(delta[:, :6].sum())
    right_energy = float(delta[:, 7:13].sum())
    peak = max(left_energy, right_energy)
    if peak < 1e-5:
        active_arm = "none"
    elif min(left_energy, right_energy) / peak >= 0.8:
        active_arm = "both"
    else:
        active_arm = "left" if left_energy > right_energy else "right"
    return {
        "left_motion_energy": left_energy,
        "right_motion_energy": right_energy,
        "left_gripper_close": float(max(0.0, action[0, 6] - np.min(action[:, 6]))),
        "left_gripper_open": float(max(0.0, np.max(action[:, 6]) - action[0, 6])),
        "right_gripper_close": float(max(0.0, action[0, 13] - np.min(action[:, 13]))),
        "right_gripper_open": float(max(0.0, np.max(action[:, 13]) - action[0, 13])),
        "active_arm": active_arm,
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-trajectories", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-source-root", type=Path, required=True)
    parser.add_argument("--instruction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels_manifest = json.loads(args.labels_manifest.read_text(encoding="utf-8"))
    task_ids = labels_manifest["physical_task_ids"]
    trajectories = np.load(args.reference_trajectories)
    vocabulary = []
    task_reports = {}
    segment_rows: list[dict[str, Any]] = []
    profile_task_chunks: list[np.ndarray] = []
    profile_episode_chunks: list[np.ndarray] = []
    profile_frame_chunks: list[np.ndarray] = []
    profile_stage_chunks: list[np.ndarray] = []
    profile_episodes: list[dict[str, int]] = []
    source_hashes = {}
    instruction_hashes = {}
    event_offset = 0

    for task in sorted(TASK_PLANS):
        plan = TASK_PLANS[task]
        source = args.task_source_root / f"{task}.py"
        instruction = args.instruction_root / f"{task}.json"
        trace = play_once_trace(source)
        if trace != plan["source_trace"]:
            raise ValueError(f"{task}: source trace drift: {trace} != {plan['source_trace']}")
        source_hashes[task] = sha256(source)
        instruction_hashes[task] = sha256(instruction)
        events = []
        for local_id, (name, before, after) in enumerate(plan["events"]):
            event = {
                "id": event_offset + local_id,
                "task": task,
                "local_id": local_id,
                "name": name,
                "before_relation": before,
                "after_relation": after,
                "boundary_source": "CRAVE-v2 recurrence-density valleys only",
                "arm_source": "per-episode 14-DoF action and gripper transitions",
            }
            vocabulary.append(event)
            events.append(event)

        task_id = int(task_ids[task])
        task_mask = trajectories["physical_task"] == task_id
        episodes = np.unique(trajectories["episode"][task_mask])
        accepted = 0
        excluded = []
        segment_count = 0
        for episode in episodes:
            mask = task_mask & (trajectories["episode"] == episode)
            frame = trajectories["frame"][mask].astype(np.int32)
            progress = trajectories["progress"][mask].astype(np.float32)
            density = trajectories["recurrence_density"][mask].astype(np.float32)
            boundary = trajectories["phase_boundary"][mask].astype(bool)
            order = np.argsort(frame)
            frame, progress, density, boundary = (
                value[order] for value in (frame, progress, density, boundary)
            )
            if not np.array_equal(frame, np.arange(len(frame))):
                raise ValueError(f"{task} ep{episode}: non-contiguous reference frames")
            starts = np.concatenate(([0], np.flatnonzero(boundary))).astype(np.int32)
            starts = np.unique(starts)
            ends = np.concatenate((starts[1:], [len(frame)])).astype(np.int32)
            if len(starts) < len(events):
                excluded.append({"episode": int(episode), "reason": "fewer_crave_segments_than_events"})
                continue
            segment_progress = np.asarray(
                [float(np.mean(progress[start:end])) for start, end in zip(starts, ends, strict=True)]
            )
            assignment = monotonic_event_assignment(segment_progress, len(events))
            parquet = (
                args.dataset
                / "data"
                / f"chunk-{int(episode) // 1000:03d}"
                / f"episode_{int(episode):06d}.parquet"
            )
            action = np.asarray(pq.read_table(parquet, columns=["action"])["action"].to_pylist())
            if len(action) != len(frame):
                raise ValueError(f"{task} ep{episode}: action/CRAVE length {len(action)}/{len(frame)}")
            for segment, (start, end, local_event) in enumerate(
                zip(starts, ends, assignment, strict=True)
            ):
                signature = action_signature(action[start:end])
                event = events[int(local_event)]
                segment_rows.append(
                    {
                        "episode": int(episode),
                        "task": task,
                        "physical_task": task_id,
                        "segment": segment,
                        "start": int(start),
                        "end": int(end),
                        "event_id": int(event["id"]),
                        "event_name": event["name"],
                        "mean_progress": float(np.mean(progress[start:end])),
                        "mean_density": float(np.mean(density[start:end])),
                        **signature,
                    }
                )
                count = int(end - start)
                profile_task_chunks.append(np.full(count, task_id, dtype=np.int16))
                profile_episode_chunks.append(np.full(count, int(episode), dtype=np.int32))
                profile_frame_chunks.append(np.arange(start, end, dtype=np.int32))
                profile_stage_chunks.append(np.full(count, int(local_event), dtype=np.int16))
            profile_episodes.append({"episode_index": int(episode), "length": int(len(frame))})
            accepted += 1
            segment_count += len(starts)
        task_reports[task] = {
            "source_trace": trace,
            "event_count": len(events),
            "episodes": len(episodes),
            "accepted_episodes": accepted,
            "excluded_episodes": excluded,
            "coverage": accepted / len(episodes),
            "segments": segment_count,
        }
        event_offset += len(events)

    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary_path = args.output / "vocabulary.json"
    segments_path = args.output / "segments.jsonl"
    profile_pairs_path = args.output / "semantic_profile_pairs.npz"
    profile_episodes_path = args.output / "semantic_profile_episodes.jsonl"
    task_map_path = args.output / "task_map.json"
    atomic_text(vocabulary_path, json.dumps(vocabulary, indent=2, sort_keys=True) + "\n")
    atomic_text(
        segments_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in segment_rows),
    )
    if not profile_task_chunks:
        raise ValueError("semantic profile contains no accepted frames")
    temporary_pairs = profile_pairs_path.with_name(f".{profile_pairs_path.name}.{os.getpid()}.tmp")
    with temporary_pairs.open("wb") as stream:
        np.savez_compressed(
            stream,
            pair_task=np.concatenate(profile_task_chunks),
            cur_ep=np.concatenate(profile_episode_chunks),
            cur_fi=np.concatenate(profile_frame_chunks),
            cur_ms=np.concatenate(profile_stage_chunks),
        )
    temporary_pairs.replace(profile_pairs_path)
    atomic_text(
        profile_episodes_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in profile_episodes),
    )
    atomic_text(task_map_path, json.dumps(task_ids, indent=2, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "protocol": "pi05_r3_source_grounded_semantic_vocabulary_v1",
        "semantic_event_count": len(vocabulary),
        "segment_count": len(segment_rows),
        "boundary_policy": "use CRAVE-v2 valley boundaries unchanged; no semantic model may add or move a boundary",
        "alignment_policy": "surjective monotonic minimum-cost assignment from segment mean CRAVE progress to source-ordered events",
        "naming_policy": "deterministic names from simulator source and instruction schema; no per-frame VLM prompts",
        "source_sha256": source_hashes,
        "instruction_sha256": instruction_hashes,
        "reference_trajectories_sha256": sha256(args.reference_trajectories),
        "labels_manifest_sha256": sha256(args.labels_manifest),
        "vocabulary_sha256": sha256(vocabulary_path),
        "segments_sha256": sha256(segments_path),
        "semantic_profile_pairs_sha256": sha256(profile_pairs_path),
        "semantic_profile_episodes_sha256": sha256(profile_episodes_path),
        "task_map_sha256": sha256(task_map_path),
        "semantic_profile_frame_count": int(sum(len(chunk) for chunk in profile_frame_chunks)),
        "semantic_profile_episode_count": len(profile_episodes),
        "tasks": task_reports,
    }
    atomic_text(args.output / "manifest.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(args.output / "READY", "ready\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
