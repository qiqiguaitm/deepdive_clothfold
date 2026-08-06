"""Merge P0 evaluation shards and apply the preregistered causal gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pi05_predictive_adapter_p0 import paired_episode_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = [json.loads(path.read_text()) for path in args.shards]
    invariant_keys = (
        "protocol",
        "checkpoint",
        "checkpoint_metadata_sha256",
        "pairs_sha256",
        "eval_panel_sha256",
        "episode_split_sha256",
        "norm_stats_sha256",
        "evaluator_sha256",
        "model_source_sha256",
        "interventions",
        "full_panel_sample_count",
        "shard_count",
    )
    for key in invariant_keys:
        if len({json.dumps(shard[key], sort_keys=True) for shard in shards}) != 1:
            raise ValueError(f"shard invariant mismatch: {key}")
    expected_shards = int(shards[0]["shard_count"])
    if sorted(int(shard["shard_index"]) for shard in shards) != list(range(expected_shards)):
        raise ValueError("missing or duplicate evaluation shard")
    if not all(shard["shard_complete"] for shard in shards):
        raise ValueError("cannot merge incomplete evaluation shard")

    rows = np.concatenate([np.asarray(shard["samples"]["panel_row"], dtype=np.int64) for shard in shards])
    order = np.argsort(rows)
    rows = rows[order]
    full_count = int(shards[0]["full_panel_sample_count"])
    if not np.array_equal(rows, np.arange(full_count)):
        raise ValueError("merged shard rows do not exactly cover the frozen panel")
    episodes = np.concatenate([np.asarray(shard["samples"]["episode_index"], dtype=np.int64) for shard in shards])[
        order
    ]
    frames = np.concatenate([np.asarray(shard["samples"]["frame_index"], dtype=np.int64) for shard in shards])[order]
    scores = {
        name: np.concatenate([np.asarray(shard["samples"][name], dtype=np.float32) for shard in shards])[order]
        for name in ("normal", "shuffled", "masked")
    }
    comparisons = {
        control: paired_episode_bootstrap(
            episodes,
            scores["normal"],
            scores[control],
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed + offset,
        )
        for offset, control in enumerate(("shuffled", "masked"), start=1)
    }
    accepted = all(value["ci95_low"] > 0.0 for value in comparisons.values())
    result = {
        "schema_version": 1,
        "protocol": shards[0]["protocol"],
        "checkpoint": shards[0]["checkpoint"],
        "checkpoint_metadata_sha256": shards[0]["checkpoint_metadata_sha256"],
        "pairs_sha256": shards[0]["pairs_sha256"],
        "eval_panel_sha256": shards[0]["eval_panel_sha256"],
        "episode_split_sha256": shards[0]["episode_split_sha256"],
        "norm_stats_sha256": shards[0]["norm_stats_sha256"],
        "evaluator_sha256": shards[0]["evaluator_sha256"],
        "model_source_sha256": shards[0]["model_source_sha256"],
        "interventions": shards[0]["interventions"],
        "source_shards": [str(path.resolve()) for path in args.shards],
        "sample_count": full_count,
        "complete": True,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "aggregate": {name: float(np.mean(value)) for name, value in scores.items()},
        "comparisons": comparisons,
        "accepted": accepted,
        "samples": {
            "panel_row": rows.tolist(),
            "episode_index": episodes.tolist(),
            "frame_index": frames.tolist(),
            **{name: value.tolist() for name, value in scores.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("aggregate", "comparisons", "accepted")}, indent=2))


if __name__ == "__main__":
    main()
