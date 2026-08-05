#!/usr/bin/env python3
"""Validate explicit authorization for the P1 North failover launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_ATTEMPTS = {
    "a0": {"pid": 2541516, "last_known_step": 13500},
    "candidate": {"pid": 675597, "last_known_step": 12400},
}
EXPECTED_SCOPE = "pi05_p1_seed1000_north_failover_pair"
EXPECTED_HOST = "root@14.103.218.231:7777"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def audit(manifest_path: Path, authorization: dict[str, Any]) -> dict[str, Any]:
    manifest_sha = sha256(manifest_path)
    attempts = authorization.get("attempts", {})
    authorized_at = _timestamp(authorization.get("authorized_at"))
    authorized_by = authorization.get("authorized_by")
    checks = {
        "protocol": authorization.get("protocol")
        == "pi05_p1_north_failover_authorization_v1",
        "scope": authorization.get("scope") == EXPECTED_SCOPE,
        "manifest_sha256": authorization.get("manifest_sha256") == manifest_sha,
        "gf1_host": authorization.get("gf1_host") == EXPECTED_HOST,
        "original_attempts_stopped": authorization.get(
            "original_gf1_attempts_stopped"
        )
        is True,
        "authorized_by": isinstance(authorized_by, str) and bool(authorized_by.strip()),
        "authorized_at": authorized_at is not None,
    }
    attempt_checks: dict[str, dict[str, bool]] = {}
    for arm, expected in EXPECTED_ATTEMPTS.items():
        observed = attempts.get(arm, {})
        attempt_checks[arm] = {
            "pid": observed.get("pid") == expected["pid"],
            "last_known_step": observed.get("last_known_step")
            == expected["last_known_step"],
            "status": observed.get("status") == "stopped",
            "verified_at": _timestamp(observed.get("verified_at")) is not None,
        }
    verified_times = [
        _timestamp(attempts.get(arm, {}).get("verified_at"))
        for arm in EXPECTED_ATTEMPTS
    ]
    checks["authorization_after_verification"] = authorized_at is not None and all(
        verified is not None and verified <= authorized_at for verified in verified_times
    )
    launch_authorized = all(checks.values()) and all(
        all(rows.values()) for rows in attempt_checks.values()
    )
    return {
        "schema_version": 1,
        "protocol": "pi05_p1_north_failover_authorization_audit_v1",
        "manifest_sha256": manifest_sha,
        "checks": checks,
        "attempt_checks": attempt_checks,
        "launch_authorized": launch_authorized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, json.loads(args.authorization.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["launch_authorized"] else 1)


if __name__ == "__main__":
    main()
