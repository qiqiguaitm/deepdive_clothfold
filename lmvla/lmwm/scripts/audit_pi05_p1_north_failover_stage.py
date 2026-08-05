#!/usr/bin/env python3
"""Verify a staged P1 failover tree without authorizing a launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, exclude_parts: set[str]) -> dict[str, Any]:
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and not exclude_parts.intersection(path.parts)
        )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        relative = path.name if root.is_file() else str(path.relative_to(root))
        digest.update(f"{relative}\t{size}\n".encode())
    return {
        "file_count": len(files),
        "bytes": total,
        "inventory_sha256": digest.hexdigest(),
    }


def audit(
    manifest: dict[str, Any], root: Path, *, use_source_paths: bool = False
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for artifact in manifest["artifacts"]:
        relative_path = (
            artifact.get("source_path", artifact["path"])
            if use_source_paths
            else artifact["path"]
        )
        path = root / relative_path
        observed = inventory(path, set(artifact.get("exclude_parts", []))) if path.exists() else None
        expected = {
            key: artifact[key]
            for key in ("file_count", "bytes", "inventory_sha256")
        }
        key_checks = []
        for relative, expected_sha in artifact.get("key_checksums", {}).items():
            key_path = path if path.is_file() and relative == path.name else path / relative
            observed_sha = sha256(key_path) if key_path.is_file() else None
            key_checks.append(
                {
                    "path": relative,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "passed": observed_sha == expected_sha,
                }
            )
        checks.append(
            {
                "path": artifact["path"],
                "observed_path": str(relative_path),
                "expected": expected,
                "observed": observed,
                "key_checks": key_checks,
                "passed": observed == expected and all(item["passed"] for item in key_checks),
            }
        )

    control_checks = []
    for relative, expected in manifest["control_files"].items():
        path = root / relative
        observed = (
            {"bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.is_file()
            else None
        )
        control_checks.append(
            {
                "path": relative,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )

    accepted = all(item["passed"] for item in checks + control_checks)
    return {
        "protocol": manifest["protocol"],
        "root": str(root),
        "path_mode": "source" if use_source_paths else "stage",
        "stage_verified": accepted,
        "launch_authorized": False,
        "artifact_checks": checks,
        "control_file_checks": control_checks,
        "promotion_requirements": manifest["promotion_requirements"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-tree",
        action="store_true",
        help="resolve artifact source_path entries when auditing the source tree",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = audit(manifest, args.root, use_source_paths=args.source_tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["stage_verified"] else 1)


if __name__ == "__main__":
    main()
