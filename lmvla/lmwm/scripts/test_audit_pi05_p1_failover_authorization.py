from __future__ import annotations

import hashlib
from pathlib import Path

from audit_pi05_p1_failover_authorization import EXPECTED_ATTEMPTS, audit


def _authorization(manifest: Path) -> dict:
    return {
        "protocol": "pi05_p1_north_failover_authorization_v1",
        "scope": "pi05_p1_seed1000_north_failover_pair",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "gf1_host": "root@14.103.218.231:7777",
        "original_gf1_attempts_stopped": True,
        "authorized_by": "operator",
        "authorized_at": "2026-08-04T12:00:00Z",
        "attempts": {
            arm: {
                **expected,
                "status": "stopped",
                "verified_at": "2026-08-04T11:59:00Z",
            }
            for arm, expected in EXPECTED_ATTEMPTS.items()
        },
    }


def test_complete_authorization_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")

    report = audit(manifest, _authorization(manifest))

    assert report["launch_authorized"] is True
    assert all(report["checks"].values())


def test_unverified_original_attempt_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    authorization = _authorization(manifest)
    authorization["attempts"]["candidate"]["status"] = "unreachable"

    report = audit(manifest, authorization)

    assert report["launch_authorized"] is False
    assert report["attempt_checks"]["candidate"]["status"] is False


def test_manifest_mutation_invalidates_authorization(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    authorization = _authorization(manifest)
    manifest.write_text('{"changed": true}\n')

    report = audit(manifest, authorization)

    assert report["launch_authorized"] is False
    assert report["checks"]["manifest_sha256"] is False


def test_null_operator_and_postdated_verification_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    authorization = _authorization(manifest)
    authorization["authorized_by"] = None
    authorization["attempts"]["a0"]["verified_at"] = "2026-08-04T12:01:00Z"

    report = audit(manifest, authorization)

    assert report["launch_authorized"] is False
    assert report["checks"]["authorized_by"] is False
    assert report["checks"]["authorization_after_verification"] is False
