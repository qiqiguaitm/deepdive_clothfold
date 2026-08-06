from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audit_pi05_predictive_adapter_p2_checkpoint import (
    audit,
    frame_cache_index,
    normalization_identity_check,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(checkpoint: Path, *, include_optimizer: bool = True) -> None:
    handlers = {
        "assets": "openpi.training.checkpoints.CallbackHandler",
        "params": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
        "train_state": "orbax.checkpoint._src.handlers.pytree_checkpoint_handler.PyTreeCheckpointHandler",
    }
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(
        json.dumps(
            {
                "item_handlers": handlers,
                "init_timestamp_nsecs": 1,
                "commit_timestamp_nsecs": 2,
            }
        )
    )
    adapter = {
        f"('params', 'predictive_action_adapter', '{part}', 'value')": {}
        for part in (
            "action_in",
            "action_position",
            "action_summary",
            "current_in",
            "predictor_hidden",
            "predictor_out",
            "route_hidden",
            "route_out",
        )
    }
    state = {"('step',)": {}}
    if include_optimizer:
        for moment in ("mu", "nu"):
            state[
                f"('opt_state', '1', '0', '{moment}', 'predictive_action_adapter', 'route_out', 'value')"
            ] = {}
    for item, tree in (("params", adapter), ("train_state", state)):
        root = checkpoint / item
        (root / "d").mkdir(parents=True)
        (root / "_METADATA").write_text(json.dumps({"tree_metadata": tree}))
        (root / "_sharding").write_text("sharding")
        (root / "manifest.ocdbt").write_text("manifest")
        (root / "d/payload").write_text("payload")
    norm = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"norm": 1}')


def fixture(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    repo = tmp_path / "repo"
    checkpoint = repo / "checkpoints/49999"
    reference = repo / "checkpoints/25000"
    write_metadata(checkpoint)
    write_metadata(reference)
    preflight = repo / "source.json"
    preflight.write_text(
        json.dumps(
            {
                "protocol": "source",
                "passed": True,
                "source_checks": {"source": {"match": True}},
                "artifact_checks": {"artifact": {"match": True}},
            }
        )
    )
    pairs = repo / "pairs.npz"
    episodes = repo / "episodes.jsonl"
    info = repo / "info.json"
    for path, text in ((pairs, "pairs"), (episodes, "episodes"), (info, "info")):
        path.write_text(text)
    cache = repo / "cache/chunk-000/observation.images.cam_high"
    cache.mkdir(parents=True)
    (cache / "episode_000000.npz").write_text("zero")
    (cache / "episode_000001.npz").write_text("one")
    cache_summary = frame_cache_index(repo / "cache")
    amendment = {
        "protocol": "test",
        "authorization": {"training_seeds": [1001, 1002]},
        "checkpoint_contract": {
            "final_step": 49999,
            "required_nonempty_files": [
                "_CHECKPOINT_METADATA",
                "params/_METADATA",
                "params/_sharding",
                "params/manifest.ocdbt",
                "train_state/_METADATA",
                "train_state/_sharding",
                "train_state/manifest.ocdbt",
                "assets/robotwin2.0_absolute_meanstd/norm_stats.json",
            ],
        },
        "normalization_identity": {
            "sha256": digest(
                checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
            )
        },
        "dataset_identity": {
            "files": {
                name: {"path": path.relative_to(repo).as_posix(), "sha256": digest(path)}
                for name, path in (("pairs", pairs), ("episodes", episodes), ("info", info))
            },
            "frame_cache": {
                "path": "cache",
                **{
                    key: cache_summary[key]
                    for key in ("files", "bytes", "index_sha256", "first_episode", "last_episode")
                },
            },
        },
    }
    return amendment, {
        "repo": repo,
        "checkpoint": checkpoint,
        "reference": reference,
        "preflight": preflight,
    }


def run_audit(amendment: dict, paths: dict[str, Path]) -> dict:
    return audit(
        repo=paths["repo"],
        seed=1001,
        checkpoint=paths["checkpoint"],
        reference_checkpoint=paths["reference"],
        source_preflight=paths["preflight"],
        amendment=amendment,
    )


def test_complete_checkpoint_passes(tmp_path: Path) -> None:
    amendment, paths = fixture(tmp_path)
    result = run_audit(amendment, paths)
    assert result["passed"]
    assert result["checks"]["parameter_tree"]["matches_reference"]
    assert result["checks"]["optimizer_state"]["adapter_mu_leaves"] == 1


def test_missing_optimizer_state_fails(tmp_path: Path) -> None:
    amendment, paths = fixture(tmp_path)
    write_metadata_path = paths["checkpoint"] / "train_state/_METADATA"
    metadata = json.loads(write_metadata_path.read_text())
    metadata["tree_metadata"] = {"('step',)": {}}
    write_metadata_path.write_text(json.dumps(metadata))
    assert not run_audit(amendment, paths)["checks"]["optimizer_state"]["passed"]


def test_dataset_drift_fails(tmp_path: Path) -> None:
    amendment, paths = fixture(tmp_path)
    (paths["repo"] / "pairs.npz").write_text("drift")
    assert not run_audit(amendment, paths)["checks"]["dataset_identity"]["passed"]


def test_incomplete_commit_fails(tmp_path: Path) -> None:
    amendment, paths = fixture(tmp_path)
    root = paths["checkpoint"] / "_CHECKPOINT_METADATA"
    metadata = json.loads(root.read_text())
    metadata["commit_timestamp_nsecs"] = 0
    root.write_text(json.dumps(metadata))
    assert not run_audit(amendment, paths)["checks"]["atomic_commit"]["passed"]


def test_normalization_accepts_equal_json_with_different_whitespace(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    reference = repo / "source/norm_stats.json"
    checkpoint = repo / "checkpoint/norm_stats.json"
    reference.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    reference.write_text('{"mean": [1, 2], "std": [3, 4]}\n')
    checkpoint.write_text('{\n  "std": [3, 4],\n  "mean": [1, 2]\n}\n')

    result = normalization_identity_check(
        repo=repo,
        norm_path=checkpoint,
        identity={
            "sha256": digest(reference),
            "semantic_reference_path": "source/norm_stats.json",
            "expected_canonical_sha256": (
                "165835227445183191585a86f9350cac7855018361f11baa21b33c7d4aed60dd"
            ),
        },
    )

    assert result["passed"]
    assert result["semantic_match"]
    assert not result["raw_bytes_match"]
    assert result["canonical_contract_match"]
    assert result["canonical_actual_sha256"] == result["canonical_expected_sha256"]


def test_normalization_rejects_json_value_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    reference = repo / "source/norm_stats.json"
    checkpoint = repo / "checkpoint/norm_stats.json"
    reference.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    reference.write_text('{"mean": [1, 2]}\n')
    checkpoint.write_text('{"mean": [1, 3]}\n')

    result = normalization_identity_check(
        repo=repo,
        norm_path=checkpoint,
        identity={
            "sha256": digest(reference),
            "semantic_reference_path": "source/norm_stats.json",
        },
    )

    assert not result["passed"]
    assert not result["semantic_match"]


def test_normalization_rejects_wrong_declared_canonical_hash(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    reference = repo / "source/norm_stats.json"
    checkpoint = repo / "checkpoint/norm_stats.json"
    reference.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    reference.write_text('{"mean": [1, 2]}\n')
    checkpoint.write_text('{"mean": [1, 2]}\n')

    result = normalization_identity_check(
        repo=repo,
        norm_path=checkpoint,
        identity={
            "sha256": digest(reference),
            "semantic_reference_path": "source/norm_stats.json",
            "expected_canonical_sha256": "0" * 64,
        },
    )

    assert result["semantic_match"]
    assert not result["canonical_contract_match"]
    assert not result["passed"]
