#!/usr/bin/env python3
"""Verify that the isolated source overlay exactly matches the frozen P1 tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, excluded: set[str]) -> dict[str, Any]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded.intersection(path.parts)
    )
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        digest.update(f"{path.relative_to(root)}\t{size}\n".encode())
    return {
        "file_count": len(files),
        "bytes": total,
        "inventory_sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    overlay = args.overlay.resolve()
    manifest_path = (
        repo
        / "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_p1_north_failover_stage_v1.json"
    )
    baseline_audit = (
        repo
        / "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_predictive_adapter_p1_baseline_audit.json"
    )
    manifest = json.loads(manifest_path.read_text())
    expected = next(
        item for item in manifest["artifacts"] if item["path"] == "kai0/src/openpi"
    )
    observed = inventory(
        overlay / "kai0/src/openpi", set(expected.get("exclude_parts", []))
    )
    expected_inventory = {
        key: expected[key]
        for key in ("file_count", "bytes", "inventory_sha256")
    }
    source_output = args.output.with_name("p1_source_freeze.json")
    verifier = repo / "kai0/scripts/verify_pi05_predictive_adapter_source_freeze.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--repo",
            str(overlay),
            "--audit",
            str(baseline_audit),
            "--output",
            str(source_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    source_report = json.loads(source_output.read_text()) if source_output.is_file() else {}
    report = {
        "schema_version": 1,
        "protocol": "pi05_p1_frozen_overlay_preflight_v1",
        "overlay": str(overlay),
        "manifest_sha256": sha256(manifest_path),
        "expected_inventory": expected_inventory,
        "observed_inventory": observed,
        "inventory_passed": observed == expected_inventory,
        "source_freeze_passed": completed.returncode == 0
        and source_report.get("passed") is True,
    }
    report["passed"] = report["inventory_passed"] and report["source_freeze_passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
