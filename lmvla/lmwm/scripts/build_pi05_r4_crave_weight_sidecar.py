#!/usr/bin/env python3
"""Build outcome-free CRAVE progress-change weights for R4 policy queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from build_pi05_crave_r0_labels import query_recurrence_fields, reference_ranges
from build_pi05_r4_outcome_free_manifest import FORBIDDEN_KEYS, OUTPUT_PROTOCOL, sha256


ENCODER = "dinov3-base"
FEATURE_DIM = 768


def normalized_progress_weights(
    tasks: np.ndarray,
    progress_change: np.ndarray,
    target_mask: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Exponentiate centered CRAVE deltas and preserve per-task mean loss scale."""
    tasks = np.asarray(tasks)
    progress_change = np.asarray(progress_change, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=bool)
    if not (tasks.shape == progress_change.shape == target_mask.shape):
        raise ValueError("task, progress-change, and target-mask arrays must align")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("CRAVE temperature must be finite and positive")
    if np.any(~np.isfinite(progress_change[target_mask])):
        raise ValueError("labeled CRAVE progress changes must be finite")

    weights = np.ones(len(tasks), dtype=np.float64)
    for task in np.unique(tasks):
        task_rows = tasks == task
        labeled = task_rows & target_mask
        if not np.any(labeled):
            raise ValueError(f"task has no CRAVE-labeled query transitions: {task}")
        centered = progress_change[labeled] - progress_change[labeled].mean()
        weights[labeled] = np.exp(centered / temperature)
        weights[task_rows] /= weights[task_rows].mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("CRAVE weights must be finite and strictly positive")
    return weights.astype(np.float32)


def validate_outcome_free_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != OUTPUT_PROTOCOL:
        raise ValueError(f"unexpected outcome-free protocol: {payload.get('protocol')!r}")
    if payload.get("record_count") != len(payload.get("records", [])) or not payload["records"]:
        raise ValueError("invalid outcome-free record count")
    for record in payload["records"]:
        leaked = FORBIDDEN_KEYS & set(record)
        if leaked:
            raise ValueError(f"outcome fields leaked into CRAVE input: {sorted(leaked)}")
    return payload


def encode_query_images(
    records: list[dict],
    manifest_path: Path,
    encoder,
    *,
    batch_size: int,
) -> list[np.ndarray]:
    image_parts: list[np.ndarray] = []
    lengths: list[int] = []
    source_manifest = Path(json.loads(manifest_path.read_text())["source_manifest"])
    for record in records:
        query_path = Path(str(record["query_observations"]))
        if not query_path.is_absolute():
            # Paths retain the original query-manifest base, not the derived manifest base.
            query_path = (source_manifest.parent / query_path).resolve()
        if not query_path.is_file() or sha256(query_path) != record["query_observations_sha256"]:
            raise ValueError(f"query artifact hash mismatch: {query_path}")
        with np.load(query_path, allow_pickle=False) as query:
            images = np.asarray(query["cam_high"], dtype=np.uint8)
        if len(images) == 0:
            raise ValueError(f"query artifact contains no images: {query_path}")
        image_parts.append(images)
        lengths.append(len(images))
    all_images = np.concatenate(image_parts)
    pooled_parts = []
    for start in range(0, len(all_images), batch_size):
        stop = min(start + batch_size, len(all_images))
        grid = encoder.encode_grid(list(all_images[start:stop]))
        if hasattr(grid, "detach"):
            pooled = grid.detach().float().mean(dim=(2, 3)).cpu().numpy()
        else:
            pooled = np.asarray(grid).mean(axis=(2, 3))
        pooled_parts.append(np.asarray(pooled, dtype=np.float32))
        print(f"ENCODE queries={stop}/{len(all_images)}", flush=True)
    all_features = np.concatenate(pooled_parts)
    if all_features.shape != (len(all_images), FEATURE_DIM) or not np.isfinite(all_features).all():
        raise ValueError(f"invalid pooled query features: {all_features.shape}")
    offsets = np.cumsum([0, *lengths])
    return [all_features[lower:upper] for lower, upper in zip(offsets[:-1], offsets[1:], strict=True)]


def verify_chunk_alignment(sidecar: dict[str, np.ndarray], chunks_path: Path) -> None:
    with np.load(chunks_path, allow_pickle=False) as chunks:
        expected = {
            "task": np.asarray(chunks["task"]),
            "scene_seed": np.asarray(chunks["scene_seed"]),
            "query_index": np.asarray(chunks["query_index"]),
            "query_frame": np.asarray(chunks["query_frame"]),
        }
    for field, values in expected.items():
        if not np.array_equal(sidecar[field], values):
            raise ValueError(f"CRAVE sidecar does not align with action chunks: {field}")


def build(
    manifest_path: Path,
    selection_path: Path,
    labels_manifest_path: Path,
    reference_root: Path,
    chunks_path: Path,
    output: Path,
    report_path: Path,
    *,
    temperature: float,
    device: str,
    batch_size: int,
    encoder_factory: Callable | None = None,
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest = validate_outcome_free_manifest(manifest_path)
    records = manifest["records"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    labels_manifest = json.loads(labels_manifest_path.read_text(encoding="utf-8"))
    expected_tasks = sorted(selection["tasks"])
    if sorted({str(row["task"]) for row in records}) != expected_tasks:
        raise ValueError("outcome-free records do not cover the frozen six-task selection")

    if encoder_factory is None:
        crave_src = Path(__file__).resolve().parents[2] / "crave/src"
        sys.path.insert(0, str(crave_src))
        from crave.encoders import load_encoder

        encoder_factory = load_encoder
    encoder = encoder_factory(ENCODER, dtype="bf16")
    encoded = encode_query_images(records, manifest_path, encoder, batch_size=batch_size)

    task_out: list[str] = []
    scene_out: list[int] = []
    query_index_out: list[int] = []
    query_frame_out: list[int] = []
    current_progress_out: list[float] = []
    target_progress_out: list[float] = []
    progress_change_out: list[float] = []
    target_mask_out: list[bool] = []
    reference_inventory = []

    for task in expected_tasks:
        reference_paths = [
            reference_root / f"ep{int(episode)}.npz"
            for episode in selection["tasks"][task]["reference_episodes"]
        ]
        reference_features = []
        for path in reference_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                values = np.asarray(payload["pooled"], dtype=np.float32)
            if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.isfinite(values).all():
                raise ValueError(f"invalid reference features: {path}")
            reference_features.append(values)
            reference_inventory.append((str(path.resolve()), sha256(path)))
        reference, ranges = reference_ranges(reference_features)
        sigma = float(labels_manifest["tasks"][task]["sigma"])
        task_rows = [index for index, row in enumerate(records) if row["task"] == task]
        task_features = np.concatenate([encoded[index] for index in task_rows])
        _, progress = query_recurrence_fields(
            task_features,
            reference,
            ranges,
            sigma,
            device=device,
            chunk_size=256,
        )
        offset = 0
        for record_index in task_rows:
            record = records[record_index]
            length = len(encoded[record_index])
            values = progress[offset : offset + length]
            offset += length
            query_path = Path(str(record["query_observations"]))
            if not query_path.is_absolute():
                query_path = (Path(manifest["source_manifest"]).parent / query_path).resolve()
            with np.load(query_path, allow_pickle=False) as query:
                frames = np.asarray(query["query_frame_index"], dtype=np.int32)
            for query_index, (frame, current) in enumerate(zip(frames, values, strict=True)):
                has_target = query_index + 1 < len(values)
                target = float(values[query_index + 1]) if has_target else float("nan")
                task_out.append(task)
                scene_out.append(int(record["scene_seed"]))
                query_index_out.append(query_index)
                query_frame_out.append(int(frame))
                current_progress_out.append(float(current))
                target_progress_out.append(target)
                progress_change_out.append(target - float(current) if has_target else 0.0)
                target_mask_out.append(has_target)

    tasks = np.asarray(task_out)
    delta = np.asarray(progress_change_out, dtype=np.float32)
    target_mask = np.asarray(target_mask_out, dtype=bool)
    weights = normalized_progress_weights(tasks, delta, target_mask, temperature)
    arrays = {
        "task": tasks,
        "scene_seed": np.asarray(scene_out, dtype=np.int64),
        "query_index": np.asarray(query_index_out, dtype=np.int16),
        "query_frame": np.asarray(query_frame_out, dtype=np.int32),
        "current_progress": np.asarray(current_progress_out, dtype=np.float32),
        "target_progress": np.asarray(target_progress_out, dtype=np.float32),
        "progress_change": delta,
        "target_mask": target_mask,
        "weight": weights,
    }
    verify_chunk_alignment(arrays, chunks_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)

    per_task = {}
    for task in expected_tasks:
        mask = tasks == task
        per_task[task] = {
            "queries": int(mask.sum()),
            "labeled_transitions": int(target_mask[mask].sum()),
            "neutral_terminal_queries": int((~target_mask[mask]).sum()),
            "progress_change_mean": float(delta[mask & target_mask].mean()),
            "progress_change_std": float(delta[mask & target_mask].std()),
            "weight_mean": float(weights[mask].mean()),
            "weight_min": float(weights[mask].min()),
            "weight_max": float(weights[mask].max()),
        }
    inventory_digest = hashlib.sha256()
    for path, digest in sorted(reference_inventory):
        inventory_digest.update(f"{path}\t{digest}\n".encode())
    report = {
        "schema_version": 1,
        "protocol": "pi05_r4_outcome_free_crave_weight_sidecar_v1",
        "interpretation": (
            "Weights use only frozen demonstration-derived CRAVE recurrence progress and consecutive "
            "observations from the same behavior-policy rollout. They do not consume outcome, reward, "
            "return, or future observations beyond the executed action chunk."
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "selection_sha256": sha256(selection_path),
        "labels_manifest_sha256": sha256(labels_manifest_path),
        "chunks": str(chunks_path.resolve()),
        "chunks_sha256": sha256(chunks_path),
        "reference_feature_inventory_sha256": inventory_digest.hexdigest(),
        "reference_feature_count": len(reference_inventory),
        "encoder": f"{ENCODER} pooled patch grid",
        "temperature": temperature,
        "unlabeled_policy": "last query in each rollout receives neutral raw weight before task normalization",
        "sample_count": len(weights),
        "per_task": per_task,
        "sidecar": str(output.resolve()),
        "sidecar_sha256": sha256(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    report = build(
        args.manifest,
        args.selection,
        args.labels_manifest,
        args.reference_root,
        args.chunks,
        args.output,
        args.report,
        temperature=args.temperature,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({"samples": report["sample_count"], "sidecar": report["sidecar"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
