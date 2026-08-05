#!/usr/bin/env python3
"""Verify frozen R1 sources and generated P0/R0 teacher evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checked = {}
    for relative, expected in protocol["source_sha256"].items():
        actual = sha256(repo / relative)
        if actual != expected:
            raise ValueError(f"R1 source drift: {relative}: {actual} != {expected}")
        checked[relative] = actual
    scene = repo / protocol["scene_manifest"]
    if sha256(scene) != protocol["scene_manifest_sha256"]:
        raise ValueError("R1 scene manifest drift")
    immutable = {
        "base_params_metadata": repo / "kai0/checkpoints/pi05_base/params/_METADATA",
        "norm_stats": repo
        / "kai0/assets/pi05_robotwin_a0_public_exact_bj/robotwin2.0_absolute_meanstd/norm_stats.json",
        "dense_targets": repo / protocol["teacher"]["artifact"],
        "dense_targets_manifest": repo / protocol["teacher"]["artifact_manifest"],
    }
    for name, path in immutable.items():
        if sha256(path) != protocol["immutable_artifact_sha256"][name]:
            raise ValueError(f"R1 immutable artifact drift: {name}")
    dense_manifest = json.loads(immutable["dense_targets_manifest"].read_text())
    if dense_manifest.get("dense_targets_sha256") != sha256(
        immutable["dense_targets"]
    ):
        raise ValueError("R1 dense-target manifest mismatch")
    if (
        dense_manifest.get("episode_count") != 1200
        or dense_manifest.get("physical_task_count") != 6
        or dense_manifest.get("horizon_frames") != 50
        or int(dense_manifest.get("target_rows", 0)) < 300_000
    ):
        raise ValueError("R1 dense-target coverage contract failed")

    p0_gate = repo / "logs/predictive/p0_eval/p0_gate.accepted"
    r0_gate = repo / "logs/crave_r0/probe_gate/r0_gate.accepted"
    if not p0_gate.is_file() or not r0_gate.is_file():
        raise RuntimeError("R1 requires accepted P0 and R0 gates")
    p0_audit_path = (
        repo
        / "lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p0_final_audit.json"
    )
    p0_audit = json.loads(p0_audit_path.read_text(encoding="utf-8"))
    if not p0_audit.get("passed") or not p0_audit.get("exact_zero_policy_route"):
        raise RuntimeError("R1 P0 adapter audit is not accepted")
    checkpoint_metadata = (
        repo
        / "kai0/checkpoints/pi05_predictive_adapter_p0"
        / "pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999/params/_METADATA"
    )
    if sha256(checkpoint_metadata) != p0_audit["checkpoint_metadata_sha256"]:
        raise ValueError("R1 P0 checkpoint metadata drift")

    labels_root = repo / "lmvla/lmwm/data/pi05_crave_r0_v1"
    labels_manifest = json.loads(
        (labels_root / "labels_manifest.json").read_text(encoding="utf-8")
    )
    generated = {
        "labels_sha256": labels_root / "labels.npz",
        "probe_train_sha256": labels_root / "probe_train.npz",
        "reference_trajectories_sha256": labels_root / "reference_trajectories.npz",
    }
    for key, path in generated.items():
        if sha256(path) != labels_manifest[key]:
            raise ValueError(f"R1 generated R0 artifact drift: {key}")
    return {
        "accepted": True,
        "protocol_sha256": sha256(protocol_path),
        "p0_audit_sha256": sha256(p0_audit_path),
        "r0_labels_manifest_sha256": sha256(labels_root / "labels_manifest.json"),
        "dense_targets_sha256": sha256(immutable["dense_targets"]),
        "dense_target_rows": int(dense_manifest["target_rows"]),
        "source_sha256": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
