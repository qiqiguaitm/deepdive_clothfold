#!/usr/bin/env python3
"""Export deterministic same-task, foreign-episode LaWAM hint controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


TASK_NAMES = {
    0: "beat_block_hammer",
    1: "stack_blocks_two",
    2: "stack_blocks_three",
    3: "blocks_ranking_rgb",
    4: "blocks_ranking_size",
    5: "handover_block",
}


def load_rows(path: Path, key: str, indices: list[int]) -> list[np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        member = archive.open(f"{key}.npy")
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
        if fortran:
            raise ValueError(f"{path}:{key} is Fortran ordered")
        offset = member.tell()
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        rows = []
        for index in indices:
            member.seek(offset + index * row_bytes)
            raw = member.read(row_bytes)
            if len(raw) != row_bytes:
                raise EOFError(f"short read for {path}:{key}[{index}]")
            rows.append(np.frombuffer(raw, dtype=dtype).reshape(row_shape).copy())
        return rows


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["blocks_ranking_size", "handover_block", "stack_blocks_three"],
    )
    args = parser.parse_args()

    pairs_path = args.artifact / "pairs.npz"
    compact_path = args.artifact / "target_compact.npz"
    pairs = np.load(pairs_path)
    compact = np.load(compact_path)
    compact_eps = compact["ep"]
    compact_frames = compact["tgt_fi"]
    args.output.mkdir(parents=True, exist_ok=True)

    name_to_id = {name: task_id for task_id, name in TASK_NAMES.items()}
    manifest: dict[str, object] = {
        "source_pairs": str(pairs_path.resolve()),
        "source_compact": str(compact_path.resolve()),
        "selection": "lowest episode with at least two milestone targets",
        "controls": {},
    }
    controls = manifest["controls"]
    assert isinstance(controls, dict)

    for task_name in args.tasks:
        task_id = name_to_id[task_name]
        task_eps = np.unique(pairs["cur_ep"][pairs["pair_task"] == task_id])
        selected_episode = None
        selected_rows: np.ndarray | None = None
        for episode in task_eps:
            rows = np.flatnonzero(compact_eps == episode)
            if len(rows) >= 2:
                selected_episode = int(episode)
                selected_rows = rows[:2]
                break
        if selected_rows is None or selected_episode is None:
            raise RuntimeError(f"no two-target episode found for {task_name}")

        first, second = load_rows(compact_path, "feat", selected_rows.tolist())
        absolute = second.astype(np.float32)
        residual = second.astype(np.float32) - first.astype(np.float32)
        absolute_path = args.output / f"{task_name}_absolute.npy"
        residual_path = args.output / f"{task_name}_residual.npy"
        np.save(absolute_path, absolute)
        np.save(residual_path, residual)
        controls[task_name] = {
            "task_id": task_id,
            "episode": selected_episode,
            "rows": selected_rows.tolist(),
            "frames": compact_frames[selected_rows].astype(int).tolist(),
            "absolute": {"path": str(absolute_path), "sha256": digest(absolute)},
            "residual": {"path": str(residual_path), "sha256": digest(residual)},
        }

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
