#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PROTOCOL = "temporal_grounding_tg1_retry500_v1"
EXPECTED_CAP = 500
EXPECTED_CONDITIONS = {
    "TG1A": ["normal", "null", "persistence", "shuffled"],
    "TG1B": [
        "future_off_e36",
        "future_off_e50",
        "local_wm_e36",
        "local_wm_e50",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, manifest_path: Path, bundle: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("unexpected TG1 retry protocol")
    if not manifest.get("frozen") or manifest.get("manual_execution_authorized"):
        raise ValueError("TG1 retry protocol must be frozen and scheduler-owned")
    if manifest.get("operator_authorized") is not True:
        raise ValueError("TG1 retry protocol lacks explicit operator authorization")
    if manifest.get("experiment_protocol_changed") is not True:
        raise ValueError("TG1 retry protocol must disclose its protocol change")
    if manifest.get("retry_cap", {}).get("new") != EXPECTED_CAP:
        raise ValueError("TG1 retry cap must be exactly 500")
    if manifest.get("conditions") != EXPECTED_CONDITIONS:
        raise ValueError("TG1 retry condition panel drift")
    if bundle not in EXPECTED_CONDITIONS:
        raise ValueError(f"unsupported bundle: {bundle}")

    checked = {}
    for relative, expected in manifest.get("file_sha256", {}).items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"TG1 retry input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"TG1 retry SHA-256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        checked[relative] = actual

    marker_path = repo / manifest["activation"]["marker"]
    if not marker_path.is_file():
        raise FileNotFoundError(f"TG1 retry activation marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("protocol") != EXPECTED_PROTOCOL or marker.get("activated") is not True:
        raise ValueError("TG1 retry activation marker is invalid")
    if marker.get("retry_cap") != EXPECTED_CAP:
        raise ValueError("TG1 retry activation marker has the wrong cap")
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("TG1 retry activation marker does not bind this manifest")
    if marker.get("canonical_roots_empty_after_archive") is not True:
        raise ValueError("TG1 retry activation did not establish empty canonical roots")

    return {
        "schema_version": 1,
        "protocol": EXPECTED_PROTOCOL,
        "bundle": bundle,
        "retry_cap": EXPECTED_CAP,
        "verified_files": len(checked),
        "activation_marker": str(marker_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", choices=tuple(EXPECTED_CONDITIONS), required=True)
    args = parser.parse_args()
    result = verify(args.repo.resolve(), args.manifest.resolve(), args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
