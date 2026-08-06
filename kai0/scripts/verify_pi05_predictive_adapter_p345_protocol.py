#!/usr/bin/env python3
"""Verify immutable files and runtime identity for P3, P4, or P5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("p3", "p4", "p5"), required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("phase") != args.phase:
        raise ValueError(f"phase mismatch: {manifest.get('phase')} != {args.phase}")
    if not manifest.get("frozen"):
        raise ValueError("protocol is not frozen")
    failures = []
    for relative, expected in manifest["file_sha256"].items():
        path = args.repo / relative
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            failures.append({"path": relative, "expected": expected, "observed": observed})
    if failures:
        raise ValueError(f"frozen protocol drift: {failures}")
    print(json.dumps({"phase": args.phase, "verified_files": len(manifest["file_sha256"])}))


if __name__ == "__main__":
    main()
