#!/usr/bin/env python3
"""Materialize audited R4 policy queries as a direct-chunk LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CHUNK_SIZE = 50
ACTION_DIM = 14


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_spec(image_shape: tuple[int, int, int]) -> dict:
    return {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "action": {
            "dtype": "float32",
            "shape": (CHUNK_SIZE, ACTION_DIM),
            "names": None,
        },
        **{
            f"observation.images.{camera}": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            }
            for camera in CAMERAS
        },
        "sample_weight": {"dtype": "float32", "shape": (1,), "names": None},
        "action_valid": {"dtype": "bool", "shape": (CHUNK_SIZE,), "names": None},
        "terminal_success": {"dtype": "bool", "shape": (1,), "names": None},
        "scene_seed": {"dtype": "int64", "shape": (1,), "names": None},
        "source_query_frame": {"dtype": "int64", "shape": (1,), "names": None},
    }


def validate_arrays(payload: np.lib.npyio.NpzFile) -> int:
    required = {
        "state",
        "action",
        "action_valid",
        "task",
        "scene_seed",
        "success",
        "record_index",
        "query_index",
        "query_frame",
        "query_observations",
        "instruction",
        "ordinary_weight",
        "outcome_calibrated_weight",
    }
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"R4 chunks are missing arrays: {sorted(missing)}")
    count = len(payload["state"])
    if payload["state"].shape != (count, 14):
        raise ValueError(f"invalid state shape: {payload['state'].shape}")
    if payload["action"].shape != (count, CHUNK_SIZE, ACTION_DIM):
        raise ValueError(f"invalid action shape: {payload['action'].shape}")
    if payload["action_valid"].shape != (count, CHUNK_SIZE):
        raise ValueError(f"invalid action-valid shape: {payload['action_valid'].shape}")
    if not np.isfinite(payload["state"]).all() or not np.isfinite(payload["action"]).all():
        raise ValueError("R4 state/action arrays contain non-finite values")
    weights = np.asarray(payload["outcome_calibrated_weight"], dtype=np.float32)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("R4 outcome-calibrated weights must be finite positive scalars")
    for task in np.unique(payload["task"]):
        mask = payload["task"] == task
        if not np.isclose(weights[mask].mean(), 1.0, atol=1e-5):
            raise ValueError(f"outcome weights are not task-normalized for {task}")
        if len(np.unique(payload["success"][mask])) != 2:
            raise ValueError(f"task lacks success/failure support: {task}")
    return count


def resolve_query_artifact(chunks_path: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else (chunks_path.parent / path).resolve()


def build(
    chunks_path: Path,
    chunks_report_path: Path,
    output_root: Path,
    *,
    repo_id: str,
    image_writer_threads: int = 8,
) -> dict:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # pragma: no cover - compatibility with pre-0.4 layouts.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    chunks_path = chunks_path.resolve()
    chunks_report_path = chunks_report_path.resolve()
    output_root = output_root.resolve()
    chunks_report = json.loads(chunks_report_path.read_text(encoding="utf-8"))
    if chunks_report.get("chunks_sha256") != sha256(chunks_path):
        raise ValueError("R4 chunks hash does not match its audit report")
    if chunks_report.get("source_audit", {}).get("accepted") is not True:
        raise ValueError("R4 chunks report does not contain an accepted source audit")
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing R4 dataset: {output_root}")

    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    try:
        with np.load(chunks_path, allow_pickle=False) as payload:
            sample_count = validate_arrays(payload)
            if sample_count != int(chunks_report["sample_count"]):
                raise ValueError("R4 chunk sample count does not match its report")
            first_query = resolve_query_artifact(chunks_path, str(payload["query_observations"][0]))
            with np.load(first_query, allow_pickle=False) as query:
                image_shape = tuple(int(value) for value in query[CAMERAS[0]].shape[1:])
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=staging,
                fps=1,
                robot_type="aloha",
                features=feature_spec(image_shape),
                use_videos=False,
                image_writer_threads=image_writer_threads,
            )
            record_ids = np.asarray(payload["record_index"], dtype=np.int64)
            unique_records = np.unique(record_ids)
            for record_id in unique_records:
                indices = np.flatnonzero(record_ids == record_id)
                indices = indices[np.argsort(payload["query_index"][indices])]
                query_path = resolve_query_artifact(
                    chunks_path, str(payload["query_observations"][indices[0]])
                )
                with np.load(query_path, allow_pickle=False) as query:
                    for sample_index in indices:
                        query_index = int(payload["query_index"][sample_index])
                        if int(query["query_frame_index"][query_index]) != int(
                            payload["query_frame"][sample_index]
                        ):
                            raise ValueError(f"query frame mismatch in {query_path}")
                        state = np.asarray(query["query_states"][query_index], dtype=np.float32)
                        if not np.array_equal(state, payload["state"][sample_index]):
                            raise ValueError(f"query state mismatch in {query_path}")
                        frame = {
                            "observation.state": state,
                            "action": np.asarray(payload["action"][sample_index], dtype=np.float32),
                            "sample_weight": np.asarray(
                                [payload["outcome_calibrated_weight"][sample_index]],
                                dtype=np.float32,
                            ),
                            "action_valid": np.asarray(
                                payload["action_valid"][sample_index], dtype=bool
                            ),
                            "terminal_success": np.asarray(
                                [payload["success"][sample_index]], dtype=bool
                            ),
                            "scene_seed": np.asarray(
                                [payload["scene_seed"][sample_index]], dtype=np.int64
                            ),
                            "source_query_frame": np.asarray(
                                [payload["query_frame"][sample_index]], dtype=np.int64
                            ),
                            "task": str(payload["instruction"][sample_index]),
                        }
                        for camera in CAMERAS:
                            image = np.asarray(query[camera][query_index], dtype=np.uint8)
                            if image.shape != image_shape:
                                raise ValueError(f"camera shape mismatch in {query_path}: {camera}")
                            frame[f"observation.images.{camera}"] = image
                        dataset.add_frame(frame)
                dataset.save_episode()
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                finalize()
        staging.rename(output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    info_path = output_root / "meta/info.json"
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_direct_action_chunk_lerobot_v1",
        "interpretation": (
            "sample_weight is task-normalized terminal-outcome weighting; it is not "
            "an action advantage, Q-value, or world-critic estimate"
        ),
        "chunks": str(chunks_path),
        "chunks_sha256": sha256(chunks_path),
        "chunks_report": str(chunks_report_path),
        "chunks_report_sha256": sha256(chunks_report_path),
        "dataset_root": str(output_root),
        "dataset_info_sha256": sha256(info_path),
        "repo_id": repo_id,
        "episodes": int(chunks_report["record_count"]),
        "samples": int(chunks_report["sample_count"]),
        "fps_semantics": "one frame per real policy query; action is already a 50x14 chunk",
        "action_sequence_keys": [],
        "ordinary_arm": "ignore sample_weight",
        "outcome_calibrated_arm": "enable per-sample loss weighting",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--chunks-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pi05-r4-query-train-v1")
    parser.add_argument("--image-writer-threads", type=int, default=8)
    args = parser.parse_args()
    report = build(
        args.chunks,
        args.chunks_report,
        args.output_root,
        repo_id=args.repo_id,
        image_writer_threads=args.image_writer_threads,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"episodes": report["episodes"], "samples": report["samples"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
