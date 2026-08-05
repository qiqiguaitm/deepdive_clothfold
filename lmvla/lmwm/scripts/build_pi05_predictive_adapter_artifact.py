#!/usr/bin/env python3
"""Freeze the pi0.5 predictive-adapter +1 s pairs and episode split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def split_episode(episode: int, seed: int, heldout_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{episode}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return unit < heldout_fraction


def select_heldout_eval_rows(
    arrays: dict[str, np.ndarray], *, sample_count: int, seed: int
) -> dict[str, np.ndarray]:
    """Freeze a paired evaluation panel with every held-out episode represented."""
    heldout_rows = np.flatnonzero(arrays["heldout"])
    if sample_count < 1 or sample_count > heldout_rows.size:
        raise ValueError(
            f"eval sample_count must be in [1, {heldout_rows.size}], got {sample_count}"
        )
    episodes = arrays["cur_ep"][heldout_rows]
    unique_episodes, starts, counts = np.unique(
        episodes, return_index=True, return_counts=True
    )
    if sample_count < unique_episodes.size:
        raise ValueError(
            "eval sample_count must cover every held-out episode: "
            f"{sample_count} < {unique_episodes.size}"
        )

    rng = np.random.default_rng(seed)
    offsets = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:{int(episode)}".encode()).digest()[:8],
                "big",
            )
            % int(count)
            for episode, count in zip(unique_episodes, counts, strict=True)
        ],
        dtype=np.int64,
    )
    mandatory = heldout_rows[starts + offsets]
    selected_mask = np.zeros(arrays["cur_ep"].shape[0], dtype=np.bool_)
    selected_mask[mandatory] = True
    remaining_count = sample_count - mandatory.size
    if remaining_count:
        candidates = heldout_rows[~selected_mask[heldout_rows]]
        additional = rng.choice(candidates, size=remaining_count, replace=False)
        selected = np.concatenate([mandatory, additional])
    else:
        selected = mandatory
    selected = selected[rng.permutation(selected.size)]

    result = {"row_index": selected.astype(np.int64)}
    for name, value in arrays.items():
        if value.ndim and value.shape[0] == arrays["cur_ep"].shape[0]:
            result[name] = value[selected]
    return result


def build_artifact(
    dataset: Path,
    source_pairs: Path,
    *,
    split_seed: int,
    heldout_fraction: float,
    action_horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between zero and one")
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    tasks_path = dataset / "meta/tasks.jsonl"
    info = json.loads(info_path.read_text())
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    task_by_text = {
        row["task"]: int(row["task_index"])
        for row in map(json.loads, tasks_path.read_text().splitlines())
    }
    episode_by_id = {int(row["episode_index"]): row for row in episodes}
    if len(episode_by_id) != len(episodes):
        raise ValueError("duplicate episode_index in metadata")

    heldout_episodes = sorted(
        episode for episode in episode_by_id if split_episode(episode, split_seed, heldout_fraction)
    )
    train_episodes = sorted(set(episode_by_id) - set(heldout_episodes))
    if not train_episodes or not heldout_episodes:
        raise ValueError("deterministic split produced an empty partition")
    heldout_set = set(heldout_episodes)

    source = np.load(source_pairs)
    required = {"cur_ep", "cur_fi", "tgt_fi", "horizon_frames", "fps", "horizon_seconds"}
    if missing := required - set(source.files):
        raise ValueError(f"source pairs missing arrays: {sorted(missing)}")
    cur_ep = np.asarray(source["cur_ep"], dtype=np.int32)
    cur_fi = np.asarray(source["cur_fi"], dtype=np.int32)
    tgt_fi = np.asarray(source["tgt_fi"], dtype=np.int32)
    horizon_frames = int(source["horizon_frames"])
    if not np.array_equal(tgt_fi, cur_fi + horizon_frames):
        raise ValueError("source pairs contain clamped or non-fixed targets")

    lengths = np.asarray([int(episode_by_id[int(ep)]["length"]) for ep in cur_ep], dtype=np.int32)
    if np.any(tgt_fi >= lengths):
        raise ValueError("future target crosses an episode boundary")
    action_end_fi = np.minimum(cur_fi + action_horizon - 1, lengths - 1).astype(np.int32)
    action_padding = (action_horizon - (action_end_fi - cur_fi + 1)).astype(np.int16)
    heldout_mask = np.asarray([int(ep) in heldout_set for ep in cur_ep], dtype=np.bool_)

    task_ids = np.asarray(
        [task_by_text[episode_by_id[int(ep)]["tasks"][0]] for ep in cur_ep], dtype=np.int32
    )
    arrays = {
        "cur_ep": cur_ep,
        "cur_fi": cur_fi,
        "tgt_fi": tgt_fi,
        "target_valid": np.ones(cur_ep.shape, dtype=np.bool_),
        "heldout": heldout_mask,
        "task_index": task_ids,
        "action_start_fi": cur_fi.copy(),
        "action_end_fi": action_end_fi,
        "action_padding": action_padding,
        "horizon_frames": np.asarray(horizon_frames, dtype=np.int32),
        "fps": np.asarray(int(source["fps"]), dtype=np.int32),
        "horizon_seconds": np.asarray(float(source["horizon_seconds"]), dtype=np.float32),
        "action_horizon": np.asarray(action_horizon, dtype=np.int32),
    }

    episode_manifest = []
    for episode in episodes:
        episode_id = int(episode["episode_index"])
        task_text = episode["tasks"][0]
        record = {
            "episode_index": episode_id,
            "length": int(episode["length"]),
            "task_index": task_by_text[task_text],
            "task": task_text,
            "split": "heldout" if episode_id in heldout_set else "train",
        }
        record["sha256"] = canonical_hash(record)
        episode_manifest.append(record)

    task_pair_counts = Counter(map(int, task_ids))
    audit = {
        "protocol": "pi05_predictive_action_adapter_p0_v1",
        "dataset": str(dataset.resolve()),
        "source_pairs": str(source_pairs.resolve()),
        "dataset_info_sha256": sha256(info_path),
        "episodes_sha256": sha256(episodes_path),
        "tasks_sha256": sha256(tasks_path),
        "episodes": len(episodes),
        "dataset_frames": int(info["total_frames"]),
        "fps": int(source["fps"]),
        "horizon_seconds": float(source["horizon_seconds"]),
        "horizon_frames": horizon_frames,
        "action_horizon": action_horizon,
        "valid_pairs": int(cur_ep.size),
        "train_pairs": int(np.count_nonzero(~heldout_mask)),
        "heldout_pairs": int(np.count_nonzero(heldout_mask)),
        "excluded_tail_frames": int(info["total_frames"] - cur_ep.size),
        "tail_policy": "no clamp: tail frames have target mask false and are absent from pair rows",
        "action_alignment": "action_start_fi equals cur_fi; right-edge action chunks use dataset padding only",
        "split": {
            "unit": "episode",
            "seed": split_seed,
            "heldout_fraction": heldout_fraction,
            "train_episodes": len(train_episodes),
            "heldout_episodes": len(heldout_episodes),
            "leakage": bool(set(train_episodes) & set(heldout_episodes)),
        },
        "task_coverage": {
            "metadata_tasks": int(info["total_tasks"]),
            "tasks_with_valid_pairs": len(task_pair_counts),
            "min_pairs_per_covered_task": min(task_pair_counts.values()),
            "max_pairs_per_covered_task": max(task_pair_counts.values()),
        },
        "preregistered_metric": {
            "name": "mean_patch_cosine_similarity",
            "paired_unit": "heldout episode mean over frozen sampled pairs",
            "bootstrap": "paired episode-cluster bootstrap, 20000 draws, seed 20260804",
            "gate": "normal minus shuffled and normal minus masked each have paired 95% CI > 0",
        },
    }
    split = {
        "protocol": audit["protocol"],
        "split_seed": split_seed,
        "train_episodes": train_episodes,
        "heldout_episodes": heldout_episodes,
    }
    return arrays, {"audit": audit, "split": split}, episode_manifest


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--eval-size", type=int, default=8192)
    parser.add_argument("--eval-seed", type=int, default=20260804)
    args = parser.parse_args()

    arrays, metadata, episode_manifest = build_artifact(
        args.dataset,
        args.source_pairs,
        split_seed=args.split_seed,
        heldout_fraction=args.heldout_fraction,
        action_horizon=args.action_horizon,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "pairs.npz"
    temporary = pairs_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(pairs_path)
    eval_arrays = select_heldout_eval_rows(
        arrays, sample_count=args.eval_size, seed=args.eval_seed
    )
    eval_path = args.output_dir / "heldout_eval.npz"
    temporary_eval = eval_path.with_suffix(".npz.tmp")
    with temporary_eval.open("wb") as stream:
        np.savez_compressed(stream, **eval_arrays)
    temporary_eval.replace(eval_path)
    episode_path = args.output_dir / "episode_manifest.jsonl"
    atomic_text(episode_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_manifest))
    split_path = args.output_dir / "episode_split.json"
    atomic_text(split_path, json.dumps(metadata["split"], indent=2, sort_keys=True) + "\n")
    audit = metadata["audit"]
    audit.update(
        {
            "pairs": str(pairs_path.resolve()),
            "pairs_sha256": sha256(pairs_path),
            "heldout_eval": str(eval_path.resolve()),
            "heldout_eval_sha256": sha256(eval_path),
            "heldout_eval_seed": args.eval_seed,
            "heldout_eval_pairs": int(eval_arrays["row_index"].size),
            "heldout_eval_episodes": int(np.unique(eval_arrays["cur_ep"]).size),
            "heldout_eval_tasks": int(np.unique(eval_arrays["task_index"]).size),
            "episode_manifest": str(episode_path.resolve()),
            "episode_manifest_sha256": sha256(episode_path),
            "episode_split": str(split_path.resolve()),
            "episode_split_sha256": sha256(split_path),
        }
    )
    audit_path = args.output_dir / "audit.json"
    atomic_text(audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
