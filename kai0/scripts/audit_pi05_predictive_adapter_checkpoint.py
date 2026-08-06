"""Audit P0 inheritance, parameter isolation, and exact-zero policy routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flax.traverse_util
import numpy as np

from openpi.models import model as model_lib


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(flat: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(flat):
        value = np.asarray(flat[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_params = args.base_params.resolve()
    checkpoint = args.checkpoint.resolve()
    checkpoint_params = checkpoint / "params"
    for path in (
        base_params / "_METADATA",
        checkpoint_params / "_METADATA",
        args.norm_stats,
        args.data_audit,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = model_lib.restore_params(base_params, restore_type=np.ndarray)
    trained = model_lib.restore_params(checkpoint_params, restore_type=np.ndarray)
    base_flat = flax.traverse_util.flatten_dict(base, sep="/")
    trained_flat = flax.traverse_util.flatten_dict(trained, sep="/")
    common_names = sorted(set(base_flat) & set(trained_flat))
    adapter_names = sorted(
        name for name in trained_flat if "predictive_action_adapter" in name
    )
    unexpected_new = sorted(set(trained_flat) - set(base_flat) - set(adapter_names))
    missing_in_trained = sorted(set(base_flat) - set(trained_flat))
    mismatched = []
    inherited_dtype_casts = []
    normalized_base = {}
    for name in common_names:
        left = np.asarray(base_flat[name])
        right = np.asarray(trained_flat[name])
        if left.dtype != right.dtype:
            inherited_dtype_casts.append(
                {"name": name, "source_dtype": str(left.dtype), "trained_dtype": str(right.dtype)}
            )
        cast_left = left.astype(right.dtype, copy=False)
        normalized_base[name] = cast_left
        if left.shape != right.shape or not np.array_equal(cast_left, right):
            mismatched.append(name)

    route_output_names = [
        name
        for name in adapter_names
        if "/route_out/" in name and name.endswith(("/kernel", "/bias"))
    ]
    route_output_max_abs = {
        name: float(np.max(np.abs(np.asarray(trained_flat[name], dtype=np.float32))))
        for name in route_output_names
    }
    inherited_trained = {name: trained_flat[name] for name in common_names}
    passed = bool(
        adapter_names
        and route_output_names
        and not unexpected_new
        and not missing_in_trained
        and not mismatched
        and all(value == 0.0 for value in route_output_max_abs.values())
    )
    result = {
        "schema_version": 1,
        "protocol": "pi05_predictive_action_adapter_p0_v1",
        "base_params": str(base_params),
        "base_metadata_sha256": sha256(base_params / "_METADATA"),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": sha256(checkpoint_params / "_METADATA"),
        "norm_stats_sha256": sha256(args.norm_stats),
        "data_audit_sha256": sha256(args.data_audit),
        "base_leaf_count": len(base_flat),
        "checkpoint_leaf_count": len(trained_flat),
        "inherited_leaf_count": len(common_names),
        "adapter_leaf_count": len(adapter_names),
        "inherited_dtype_casts": inherited_dtype_casts,
        "unexpected_new_leaves": unexpected_new,
        "missing_base_leaves": missing_in_trained,
        "modified_inherited_leaves": mismatched,
        "base_cast_to_checkpoint_dtype_tree_sha256": tree_digest(normalized_base),
        "checkpoint_inherited_tree_sha256": tree_digest(inherited_trained),
        "route_output_max_abs": route_output_max_abs,
        "exact_zero_policy_route": bool(
            route_output_names
            and all(value == 0.0 for value in route_output_max_abs.values())
        ),
        "gradient_route_test": {
            "command": "pytest -q src/openpi/models/pi0_test.py",
            "test": "test_predictive_adapter_stops_visual_gradients_and_updates_adapter",
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
