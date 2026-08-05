#!/usr/bin/env python3
"""Verify frozen R3 source identity, scene identity, and semantic profile integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(
    repo: Path,
    protocol_path: Path,
    artifact: Path,
    condition: str,
    model_root: Path | None = None,
    tokenizer_root: Path | None = None,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if condition not in protocol["conditions"]:
        raise ValueError(f"condition {condition!r} is absent from frozen protocol")
    source_results = {}
    for relative, expected in protocol["source_sha256"].items():
        path = repo / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"R3 source drift: {relative}: {actual} != {expected}")
        source_results[relative] = actual

    scene_manifest = repo / "lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json"
    scene_hash = sha256(scene_manifest)
    if scene_hash != protocol["scene_manifest_sha256"]:
        raise ValueError("R3 scene manifest drift")

    public_results = {}
    if "public_artifact_sha256" in protocol:
        if model_root is None or tokenizer_root is None:
            raise ValueError("public checkpoint verification requires model and tokenizer roots")
        public_paths = {
            "config.json": model_root / "config.json",
            "model.safetensors": model_root / "model.safetensors",
            "tokenizer.model": tokenizer_root / "tokenizer.model",
        }
        for name, expected in protocol["public_artifact_sha256"].items():
            actual = sha256(public_paths[name])
            if actual != expected:
                raise ValueError(f"public pi0.5 artifact drift: {name}")
            public_results[name] = actual

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = {
        "vocabulary": artifact / "vocabulary.json",
        "segments": artifact / "segments.jsonl",
        "semantic_profile_pairs": artifact / "semantic_profile_pairs.npz",
        "semantic_profile_episodes": artifact / "semantic_profile_episodes.jsonl",
        "task_map": artifact / "task_map.json",
    }
    for name, path in artifact_files.items():
        expected = manifest[f"{name}_sha256"]
        if sha256(path) != expected:
            raise ValueError(f"R3 semantic artifact drift: {name}")
    if int(manifest["semantic_event_count"]) != 16:
        raise ValueError(f"expected 16 semantic events, got {manifest['semantic_event_count']}")
    if set(manifest["tasks"]) != set(json.loads(artifact_files["task_map"].read_text())):
        raise ValueError("semantic task coverage differs from task map")

    pairs = np.load(artifact_files["semantic_profile_pairs"])
    required_fields = {"pair_task", "cur_ep", "cur_fi", "cur_ms"}
    if set(pairs.files) != required_fields:
        raise ValueError(f"semantic profile fields {pairs.files} != {sorted(required_fields)}")
    lengths = {len(pairs[name]) for name in required_fields}
    if lengths != {int(manifest["semantic_profile_frame_count"])}:
        raise ValueError(f"semantic profile row mismatch: {lengths}")
    if np.any(np.asarray(pairs["cur_ms"]) < 0):
        raise ValueError("semantic profile contains negative event IDs")

    return {
        "accepted": True,
        "condition": condition,
        "condition_config": protocol["conditions"][condition],
        "protocol_sha256": sha256(protocol_path),
        "scene_manifest_sha256": scene_hash,
        "semantic_manifest_sha256": sha256(manifest_path),
        "semantic_profile_frames": int(next(iter(lengths))),
        "semantic_profile_episodes": int(manifest["semantic_profile_episode_count"]),
        "source_sha256": source_results,
        "public_artifact_sha256": public_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.repo.resolve(),
        args.protocol.resolve(),
        args.artifact.resolve(),
        args.condition,
        args.model_root.resolve(),
        args.tokenizer_root.resolve(),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
