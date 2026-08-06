#!/usr/bin/env python3
"""Audit a final predictive-adapter P2 checkpoint without restoring its arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_HANDLERS = {
    "assets": "openpi.training.checkpoints.CallbackHandler",
    "params": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
    "train_state": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
}
ADAPTER_PARTS = (
    "action_in",
    "action_position",
    "action_summary",
    "current_in",
    "predictor_hidden",
    "predictor_out",
    "route_hidden",
    "route_out",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def tree_keys(metadata: dict[str, Any]) -> set[str]:
    tree = metadata.get("tree_metadata")
    return set(tree) if isinstance(tree, dict) else set()


def payload_files(root: Path) -> list[Path]:
    paths = [*root.glob("d/*"), *root.glob("ocdbt.process_*/*")]
    return sorted(path for path in paths if path.is_file() and path.stat().st_size > 0)


def frame_cache_index(root: Path) -> dict[str, Any]:
    rows: list[tuple[str, int]] = []
    episode_ids: list[int] = []
    for path in root.glob("chunk-*/observation.images.cam_high/episode_*.npz"):
        relative = path.relative_to(root).as_posix()
        rows.append((relative, path.stat().st_size))
        episode_ids.append(int(path.stem.removeprefix("episode_")))
    digest = hashlib.sha256()
    for relative, size in sorted(rows):
        digest.update(f"{relative}\t{size}\n".encode())
    sorted_ids = sorted(episode_ids)
    contiguous = bool(sorted_ids) and sorted_ids == list(
        range(sorted_ids[0], sorted_ids[-1] + 1)
    )
    return {
        "files": len(rows),
        "bytes": sum(size for _, size in rows),
        "index_sha256": digest.hexdigest(),
        "first_episode": sorted_ids[0] if sorted_ids else None,
        "last_episode": sorted_ids[-1] if sorted_ids else None,
        "unique_episodes": len(set(episode_ids)),
        "contiguous": contiguous,
    }


def audit(
    *,
    repo: Path,
    seed: int,
    checkpoint: Path,
    reference_checkpoint: Path,
    source_preflight: Path,
    amendment: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["seed"] = {
        "actual": seed,
        "allowed": amendment["authorization"]["training_seeds"],
    }
    checks["seed"]["passed"] = seed in checks["seed"]["allowed"]
    checks["final_step"] = {
        "actual": checkpoint.name,
        "expected": str(amendment["checkpoint_contract"]["final_step"]),
    }
    checks["final_step"]["passed"] = (
        checks["final_step"]["actual"] == checks["final_step"]["expected"]
    )

    required_relative = amendment["checkpoint_contract"]["required_nonempty_files"]
    files = {
        relative: {
            "path": str(checkpoint / relative),
            "bytes": (checkpoint / relative).stat().st_size
            if (checkpoint / relative).is_file()
            else 0,
            "passed": nonempty(checkpoint / relative),
        }
        for relative in required_relative
    }
    checks["required_files"] = {
        "files": files,
        "passed": all(row["passed"] for row in files.values()),
    }

    root_metadata_path = checkpoint / "_CHECKPOINT_METADATA"
    root_metadata = read_json(root_metadata_path) if nonempty(root_metadata_path) else {}
    init_ns = int(root_metadata.get("init_timestamp_nsecs", 0) or 0)
    commit_ns = int(root_metadata.get("commit_timestamp_nsecs", 0) or 0)
    checks["atomic_commit"] = {
        "item_handlers": root_metadata.get("item_handlers"),
        "expected_item_handlers": EXPECTED_HANDLERS,
        "init_timestamp_nsecs": init_ns,
        "commit_timestamp_nsecs": commit_ns,
        "passed": root_metadata.get("item_handlers") == EXPECTED_HANDLERS
        and commit_ns >= init_ns > 0,
    }

    params_path = checkpoint / "params/_METADATA"
    state_path = checkpoint / "train_state/_METADATA"
    ref_params_path = reference_checkpoint / "params/_METADATA"
    ref_state_path = reference_checkpoint / "train_state/_METADATA"
    params_metadata = read_json(params_path) if nonempty(params_path) else {}
    state_metadata = read_json(state_path) if nonempty(state_path) else {}
    ref_params = read_json(ref_params_path) if nonempty(ref_params_path) else {}
    ref_state = read_json(ref_state_path) if nonempty(ref_state_path) else {}
    params_keys = tree_keys(params_metadata)
    state_keys = tree_keys(state_metadata)
    ref_params_keys = tree_keys(ref_params)
    ref_state_keys = tree_keys(ref_state)
    adapter_param_keys = {
        key for key in params_keys if "'predictive_action_adapter'" in key
    }
    missing_adapter_parts = [
        part
        for part in ADAPTER_PARTS
        if not any(f"'{part}'" in key for key in adapter_param_keys)
    ]
    checks["parameter_tree"] = {
        "leaves": len(params_keys),
        "reference_leaves": len(ref_params_keys),
        "adapter_leaves": len(adapter_param_keys),
        "missing_adapter_parts": missing_adapter_parts,
        "matches_reference": bool(params_keys) and params_keys == ref_params_keys,
    }
    checks["parameter_tree"]["passed"] = (
        checks["parameter_tree"]["matches_reference"]
        and not missing_adapter_parts
        and bool(adapter_param_keys)
    )
    optimizer_adapter_mu = {
        key
        for key in state_keys
        if "'opt_state'" in key
        and "'mu'" in key
        and "'predictive_action_adapter'" in key
    }
    optimizer_adapter_nu = {
        key
        for key in state_keys
        if "'opt_state'" in key
        and "'nu'" in key
        and "'predictive_action_adapter'" in key
    }
    checks["optimizer_state"] = {
        "leaves": len(state_keys),
        "reference_leaves": len(ref_state_keys),
        "matches_reference": bool(state_keys) and state_keys == ref_state_keys,
        "has_step": "('step',)" in state_keys,
        "adapter_mu_leaves": len(optimizer_adapter_mu),
        "adapter_nu_leaves": len(optimizer_adapter_nu),
    }
    checks["optimizer_state"]["passed"] = (
        checks["optimizer_state"]["matches_reference"]
        and checks["optimizer_state"]["has_step"]
        and bool(optimizer_adapter_mu)
        and len(optimizer_adapter_mu) == len(optimizer_adapter_nu)
    )

    payload = {
        item: {
            "files": len(payload_files(checkpoint / item)),
            "bytes": sum(path.stat().st_size for path in payload_files(checkpoint / item)),
        }
        for item in ("params", "train_state")
    }
    checks["checkpoint_payload"] = {
        **payload,
        "passed": all(row["files"] > 0 and row["bytes"] > 0 for row in payload.values()),
    }

    source = read_json(source_preflight) if nonempty(source_preflight) else {}
    source_rows = [
        *source.get("source_checks", {}).values(),
        *source.get("artifact_checks", {}).values(),
    ]
    checks["source_freeze"] = {
        "path": str(source_preflight),
        "protocol": source.get("protocol"),
        "reported_passed": source.get("passed") is True,
        "rows": len(source_rows),
        "passed": source.get("passed") is True
        and bool(source_rows)
        and all(row.get("match") is True for row in source_rows),
    }

    data_checks = {}
    for name, spec in amendment["dataset_identity"]["files"].items():
        path = repo / spec["path"] if not Path(spec["path"]).is_absolute() else Path(spec["path"])
        actual = sha256(path) if nonempty(path) else None
        data_checks[name] = {
            "path": str(path),
            "expected_sha256": spec["sha256"],
            "actual_sha256": actual,
            "passed": actual == spec["sha256"],
        }
    cache_spec = amendment["dataset_identity"]["frame_cache"]
    cache_root = repo / cache_spec["path"]
    cache = frame_cache_index(cache_root)
    cache["path"] = str(cache_root)
    cache["passed"] = all(
        cache[key] == cache_spec[key]
        for key in ("files", "bytes", "index_sha256", "first_episode", "last_episode")
    ) and cache["unique_episodes"] == cache_spec["files"] and cache["contiguous"]
    checks["dataset_identity"] = {
        "files": data_checks,
        "frame_cache": cache,
        "passed": all(row["passed"] for row in data_checks.values()) and cache["passed"],
    }

    norm_path = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    expected_norm = amendment["normalization_identity"]["sha256"]
    actual_norm = sha256(norm_path) if nonempty(norm_path) else None
    checks["normalization"] = {
        "path": str(norm_path),
        "expected_sha256": expected_norm,
        "actual_sha256": actual_norm,
        "passed": actual_norm == expected_norm,
    }

    return {
        "schema_version": 1,
        "protocol": amendment["protocol"],
        "seed": seed,
        "checkpoint": str(checkpoint),
        "reference_checkpoint": str(reference_checkpoint),
        "checks": checks,
        "passed": all(check.get("passed") is True for check in checks.values()),
    }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    amendment = read_json(args.amendment)
    result = audit(
        repo=args.repo.resolve(),
        seed=args.seed,
        checkpoint=args.checkpoint.resolve(),
        reference_checkpoint=args.reference_checkpoint.resolve(),
        source_preflight=args.source_preflight.resolve(),
        amendment=amendment,
    )
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        args.marker.unlink(missing_ok=True)
        raise SystemExit(1)
    atomic_write(
        args.marker,
        f"validated_checkpoint={result['checkpoint']}\nreport={args.output.resolve()}\n",
    )


if __name__ == "__main__":
    main()
