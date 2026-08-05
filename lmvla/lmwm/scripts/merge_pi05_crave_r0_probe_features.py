#!/usr/bin/env python3
"""Merge exact-row shards from the frozen CRAVE R0 feature export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.with_suffix(".json").read_text()) for path in args.shards]
    for key in ("protocol", "checkpoint", "checkpoint_metadata_sha256", "norm_stats_sha256", "extractor_sha256", "model_source_sha256", "feature", "future_image_used", "interventions", "shard_count"):
        if len({json.dumps(report[key], sort_keys=True) for report in reports}) != 1:
            raise ValueError(f"shard invariant mismatch: {key}")
    expected = int(reports[0]["shard_count"])
    if sorted(int(report["shard_index"]) for report in reports) != list(range(expected)):
        raise ValueError("missing or duplicate feature shard")

    merged = {}
    coverage = {}
    for split in ("train", "eval"):
        rows = np.concatenate([np.load(path)[f"{split}_row"] for path in args.shards])
        order = np.argsort(rows)
        rows = rows[order]
        full_rows = int(reports[0]["splits"][split]["full_rows"])
        if not np.array_equal(rows, np.arange(full_rows)):
            raise ValueError(f"{split}: shards do not exactly cover frozen rows")
        merged[f"{split}_row"] = rows
        for condition in ("current", "normal", "shuffled", "masked"):
            values = np.concatenate(
                [np.load(path)[f"{split}_{condition}"] for path in args.shards]
            )[order]
            if values.ndim != 2 or len(values) != full_rows or not np.isfinite(values).all():
                raise ValueError(f"{split}/{condition}: invalid merged features {values.shape}")
            merged[f"{split}_{condition}"] = values
        coverage[split] = full_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **merged)
    temporary.replace(args.output)
    report = {
        "schema_version": 1,
        "protocol": reports[0]["protocol"],
        "checkpoint": reports[0]["checkpoint"],
        "checkpoint_metadata_sha256": reports[0]["checkpoint_metadata_sha256"],
        "norm_stats_sha256": reports[0]["norm_stats_sha256"],
        "extractor_sha256": reports[0]["extractor_sha256"],
        "model_source_sha256": reports[0]["model_source_sha256"],
        "feature": reports[0]["feature"],
        "future_image_used": False,
        "source_shards": [str(path.resolve()) for path in args.shards],
        "coverage": coverage,
        "output_sha256": sha256(args.output),
    }
    report_path = args.output.with_suffix(".json")
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
