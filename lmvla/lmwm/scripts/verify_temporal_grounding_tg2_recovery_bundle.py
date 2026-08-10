#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_OUTER = "11fb84349809b30ddc785dc99105080540d000c2"
EXPECTED_LAWAM = "71803a3f8b0e55679a4557ef6af80a76604f277a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify(repo: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "temporal_grounding_tg2_recovery_v1":
        raise ValueError("Unexpected TG2R protocol")
    if not manifest.get("frozen") or manifest.get("manual_execution_authorized"):
        raise ValueError("TG2R must remain frozen and scheduler-owned")
    if manifest.get("only_training_change") != {
        "datasets.vla_data.in_order": [False, True]
    }:
        raise ValueError("TG2R must change only DataLoader in_order")

    outer = git_head(repo)
    lawam = git_head(repo / "lmvla/lawam")
    if outer != EXPECTED_OUTER or lawam != EXPECTED_LAWAM:
        raise ValueError(
            f"North source mismatch: outer={outer} lawam={lawam}"
        )

    checked = {}
    for relative, expected in manifest["file_sha256"].items():
        path = repo / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"TG2R file drift: {relative}: {actual} != {expected}")
        checked[relative] = actual
    return {
        "protocol": manifest["protocol"],
        "outer_commit": outer,
        "lawam_commit": lawam,
        "verified_files": len(checked),
        "only_training_change": manifest["only_training_change"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo.resolve(), args.manifest.resolve()), indent=2))


if __name__ == "__main__":
    main()
