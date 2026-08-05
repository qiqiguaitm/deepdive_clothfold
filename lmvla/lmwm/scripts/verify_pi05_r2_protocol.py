#!/usr/bin/env python3
"""Verify frozen R2 sources, public pi0.5 artifacts, scenes, and causal readout."""

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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(
    *,
    repo: Path,
    protocol_path: Path,
    condition: str,
    readout_path: Path,
    model_root: Path,
    tokenizer_root: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if condition not in protocol["conditions"]:
        raise ValueError(f"condition {condition!r} is absent from the R2 protocol")
    checked = {}
    for relative, expected in protocol["source_sha256"].items():
        actual = sha256(repo / relative)
        if actual != expected:
            raise ValueError(f"R2 source drift: {relative}: {actual} != {expected}")
        checked[relative] = actual
    scene_path = repo / protocol["scene_manifest"]
    scene_hash = sha256(scene_path)
    if scene_hash != protocol["scene_manifest_sha256"]:
        raise ValueError("R2 scene manifest drift")
    public_paths = {
        "config.json": model_root / "config.json",
        "model.safetensors": model_root / "model.safetensors",
        "tokenizer.model": tokenizer_root / "tokenizer.model",
    }
    public = {}
    for name, expected in protocol["public_artifact_sha256"].items():
        actual = sha256(public_paths[name])
        if actual != expected:
            raise ValueError(f"public pi0.5 artifact drift: {name}")
        public[name] = actual

    readout_manifest_path = readout_path.with_name("readout_manifest.json")
    gate_path = readout_path.with_name("r2_readout.accepted")
    readout_manifest = json.loads(readout_manifest_path.read_text(encoding="utf-8"))
    if not readout_manifest["acceptance"]["accepted"] or not gate_path.is_file():
        raise RuntimeError("R2 causal readout is not accepted")
    if sha256(readout_path) != readout_manifest["readout_sha256"]:
        raise ValueError("R2 causal readout hash mismatch")
    for name, relative in protocol["readout_sources"].items():
        expected = readout_manifest["source_sha256"][name]
        if sha256(repo / relative) != expected:
            raise ValueError(f"R2 readout source drift: {name}")
    return {
        "accepted": True,
        "condition": condition,
        "protocol_sha256": sha256(protocol_path),
        "scene_manifest_sha256": scene_hash,
        "readout_sha256": readout_manifest["readout_sha256"],
        "source_sha256": checked,
        "public_artifact_sha256": public,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        repo=args.repo.resolve(),
        protocol_path=args.protocol.resolve(),
        condition=args.condition,
        readout_path=args.readout.resolve(),
        model_root=args.model_root.resolve(),
        tokenizer_root=args.tokenizer_root.resolve(),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
