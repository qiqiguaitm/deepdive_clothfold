"""Decide whether the accepted seed-1000 A0 is reusable for the P1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CANDIDATE_ONLY_DIFFS = {
    "/data/lmwm_target_frame_cache_root",
    "/data/lmwm_target_pairs_path",
    "/data/repack_outputs",
    "/exp_name",
    "/initialization/adapter_params",
    "/initialization/loader",
    "/model/predictive_adapter_mode",
    "/name",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def differing_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}/{key}"
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(differing_paths(left[key], right[key], path))
        return paths
    return {prefix} if left != right else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-launch", type=Path, required=True)
    parser.add_argument("--historical-gate", type=Path, required=True)
    parser.add_argument("--a0-dry-run", type=Path, required=True)
    parser.add_argument("--candidate-dry-run", type=Path, required=True)
    parser.add_argument("--current-config-source", type=Path, required=True)
    parser.add_argument("--current-model-source", type=Path, required=True)
    parser.add_argument("--current-weight-loader-source", type=Path, required=True)
    parser.add_argument("--current-train-launcher-source", type=Path, required=True)
    parser.add_argument("--current-base-metadata", type=Path, required=True)
    parser.add_argument("--current-norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = (
        args.historical_launch,
        args.historical_gate,
        args.a0_dry_run,
        args.candidate_dry_run,
        args.current_config_source,
        args.current_model_source,
        args.current_weight_loader_source,
        args.current_train_launcher_source,
        args.current_base_metadata,
        args.current_norm_stats,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    launch = json.loads(args.historical_launch.read_text())
    gate = json.loads(args.historical_gate.read_text())
    a0 = json.loads(args.a0_dry_run.read_text())
    candidate = json.loads(args.candidate_dry_run.read_text())

    differences = differing_paths(a0, candidate)
    unexpected_differences = sorted(differences - EXPECTED_CANDIDATE_ONLY_DIFFS)
    missing_expected_differences = sorted(EXPECTED_CANDIDATE_ONLY_DIFFS - differences)
    protocol_match = not unexpected_differences and not missing_expected_differences

    launch_sources = launch["sha256"]["execution_sources"]
    current_sources = {
        "config.py": sha256(args.current_config_source),
        "pi0.py": sha256(args.current_model_source),
        "weight_loaders.py": sha256(args.current_weight_loader_source),
        "train_pi05_robotwin_confirmatory.py": sha256(
            args.current_train_launcher_source
        ),
    }
    source_matches = {
        name: current_sources[name] == launch_sources.get(name) for name in current_sources
    }
    exact_source_match = all(source_matches.values())

    current_norm_hash = sha256(args.current_norm_stats)
    historical_norm_hash = gate["hashes_sha256"]["norm_stats"]
    norm_match = current_norm_hash == historical_norm_hash
    historical_base_verified = bool(gate["protocol_checks"].get("pi05_base_init"))
    # The historical launch record verifies the official base identity but does
    # not contain its metadata hash, so exact byte identity cannot be proven.
    historical_base_metadata_hash = launch.get("sha256", {}).get("base_metadata")
    current_base_metadata_hash = sha256(args.current_base_metadata)
    exact_base_match_proven = bool(
        historical_base_metadata_hash
        and historical_base_metadata_hash == current_base_metadata_hash
    )

    historical_gate_accepted = bool(gate.get("accepted"))
    reusable = bool(
        historical_gate_accepted
        and protocol_match
        and norm_match
        and exact_source_match
        and exact_base_match_proven
    )
    reasons = []
    if not exact_source_match:
        reasons.append("current execution sources are not byte-identical to accepted A0")
    if not exact_base_match_proven:
        reasons.append("historical launch lacks the base metadata hash required for exact identity")
    if not norm_match:
        reasons.append("normalization artifact differs from accepted A0")
    if not protocol_match:
        reasons.append("A0 and candidate differ outside preregistered adapter-only fields")

    result = {
        "schema_version": 1,
        "protocol": "pi05_predictive_action_adapter_p1_baseline_audit_v1",
        "historical_a0": {
            "accepted": historical_gate_accepted,
            "checkpoint": str(
                Path("kai0/checkpoints/pi05_robotwin_a0_public_exact_bj")
                / "pi05_robotwin_a0_public_exact_seed1000/49999"
            ),
            "macro_success_rate": gate.get("macro_success_rate"),
            "launch_manifest_sha256": sha256(args.historical_launch),
            "gate_sha256": sha256(args.historical_gate),
        },
        "matched_recipe": {
            "candidate_only_differences": sorted(differences),
            "expected_candidate_only_differences": sorted(EXPECTED_CANDIDATE_ONLY_DIFFS),
            "unexpected_differences": unexpected_differences,
            "missing_expected_differences": missing_expected_differences,
            "passed": protocol_match,
        },
        "source_identity": {
            "historical": launch_sources,
            "current": current_sources,
            "matches": source_matches,
            "exact_match": exact_source_match,
        },
        "base_identity": {
            "historical_official_base_verified": historical_base_verified,
            "historical_metadata_sha256": historical_base_metadata_hash,
            "current_metadata_sha256": current_base_metadata_hash,
            "exact_match_proven": exact_base_match_proven,
        },
        "normalization_identity": {
            "historical_sha256": historical_norm_hash,
            "current_sha256": current_norm_hash,
            "exact_match": norm_match,
        },
        "historical_a0_reusable": reusable,
        "decision": "reuse_historical_a0" if reusable else "retrain_current_source_a0",
        "reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
