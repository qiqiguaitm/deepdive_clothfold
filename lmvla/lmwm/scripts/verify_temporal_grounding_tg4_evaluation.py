#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "temporal_grounding_tg4_evaluation_v1":
        raise ValueError("unexpected TG4 evaluation protocol")
    if manifest.get("frozen") is not True:
        raise ValueError("TG4 evaluation manifest is not frozen")
    panels = manifest.get("panels", {})
    if panels.get("normal", {}).get("training_seeds") != [1100, 1101, 1102]:
        raise ValueError("unexpected normal training seeds")
    if panels.get("shuffled", {}).get("arms") != ["full"]:
        raise ValueError("shuffled panel must be full-only")
    if manifest.get("runtime", {}).get("evaluation_seeds") != [0, 1, 2, 3]:
        raise ValueError("unexpected evaluation seeds")
    if manifest.get("analysis", {}).get("bootstrap_samples") != 20_000:
        raise ValueError("unexpected bootstrap sample count")
    for relative, expected in manifest.get("sha256", {}).items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {relative}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    verify(args.repo.resolve(), args.manifest.resolve())
    print("TG4_EVALUATION_BUNDLE_VERIFIED protocol=temporal_grounding_tg4_evaluation_v1")


if __name__ == "__main__":
    main()
