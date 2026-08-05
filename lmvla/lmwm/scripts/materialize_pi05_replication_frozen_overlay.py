#!/usr/bin/env python3
"""Build the gate-conditioned P2/R1 training source overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from materialize_pi05_r1_frozen_overlay import materialize as materialize_r1


R1_PROTOCOL = Path("lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json")
P2_PROTOCOL = Path(
    "lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json"
)
FROZEN_ROOTS = (
    Path("lmvla/paper_iclr_lmvla/frozen_sources/pi05_replication_v1"),
    Path("lmvla/paper_iclr_lmvla/frozen_sources/pi05_r1_v1"),
)
DEFAULT_OUTPUT = Path("logs/frozen_source_overlays/pi05_replication_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_source(repo: Path, overlay: Path, relative: Path, expected: str) -> Path:
    candidates = [overlay / relative]
    candidates.extend(repo / root / relative for root in FROZEN_ROOTS)
    candidates.append(repo / relative)
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == expected:
            return candidate
    observed = {
        str(candidate): sha256(candidate) if candidate.is_file() else "missing"
        for candidate in candidates
    }
    raise ValueError(
        f"exact replication source unavailable: {relative}: {observed}"
    )


def materialize(
    repo: Path,
    output: Path,
    base_materializer: Callable[[Path, Path], dict] = materialize_r1,
) -> dict:
    repo = repo.resolve()
    output = output.resolve()
    managed_root = (repo / "logs/frozen_source_overlays").resolve()
    if managed_root not in output.parents:
        raise ValueError(f"overlay must be below {managed_root}: {output}")

    r1_protocol_path = repo / R1_PROTOCOL
    p2_protocol_path = repo / P2_PROTOCOL
    r1_protocol = json.loads(r1_protocol_path.read_text(encoding="utf-8"))
    p2_protocol = json.loads(p2_protocol_path.read_text(encoding="utf-8"))

    managed_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".replication.", dir=managed_root))
    stage = temporary_root / "overlay"
    try:
        base_audit = base_materializer(repo, stage)
        checks = []
        for relative_text, expected in p2_protocol["file_sha256"].items():
            relative = Path(relative_text)
            source = exact_source(repo, stage, relative, expected)
            destination = stage / relative
            if source != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            checks.append(
                {
                    "path": relative_text,
                    "source": str(source),
                    "sha256": sha256(destination),
                }
            )

        audit = {
            "schema_version": 1,
            "protocol": "pi05_gate_conditioned_replication_overlay_v1",
            "canonical_repo": str(repo),
            "overlay": str(output),
            "r1_protocol_sha256": sha256(r1_protocol_path),
            "p2_protocol_sha256": sha256(p2_protocol_path),
            "r1_source_count": len(r1_protocol["source_sha256"]),
            "p2_source_checks": checks,
            "base_overlay_protocol": base_audit["protocol"],
            "passed": True,
        }
        audit_path = stage / "replication_overlay_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "REPLICATION_READY").write_text(
            f"r1_protocol_sha256={audit['r1_protocol_sha256']}\n"
            f"p2_protocol_sha256={audit['p2_protocol_sha256']}\n"
            f"audit_sha256={sha256(audit_path)}\n",
            encoding="utf-8",
        )

        if output.exists():
            marker = output / "replication_overlay_audit.json"
            if not marker.is_file():
                raise RuntimeError(f"refusing to replace unmanaged overlay: {output}")
            shutil.rmtree(output)
        os.replace(stage, output)
        return audit
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / DEFAULT_OUTPUT
    print(json.dumps(materialize(repo, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
