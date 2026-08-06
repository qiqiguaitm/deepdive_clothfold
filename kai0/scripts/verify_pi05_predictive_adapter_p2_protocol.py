#!/usr/bin/env python3
"""Verify every file frozen by the predictive-adapter P2 protocol manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, manifest: dict) -> dict:
    checks = {}
    for relative_path, expected in manifest["file_sha256"].items():
        path = repo / relative_path
        actual = sha256(path)
        checks[relative_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        }
    return {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "file_checks": checks,
        "passed": bool(checks) and all(row["match"] for row in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve(), json.loads(args.manifest.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
