#!/usr/bin/env python3
"""Materialize the exact R1 source view without mutating the shared checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


FROZEN_RELATIVE = Path("lmvla/paper_iclr_lmvla/frozen_sources/pi05_r1_v1")
PROTOCOL_RELATIVE = Path("lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json")
DEFAULT_OUTPUT_RELATIVE = Path("logs/frozen_source_overlays/pi05_r1_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def materialize(repo: Path, output: Path) -> dict:
    repo = repo.resolve()
    output = output.resolve()
    managed_root = (repo / "logs/frozen_source_overlays").resolve()
    if managed_root not in output.parents:
        raise ValueError(f"overlay must be below {managed_root}: {output}")

    protocol_path = repo / PROTOCOL_RELATIVE
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_root = repo / FROZEN_RELATIVE
    source_checks = []

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as tmp:
        stage = Path(tmp)
        shutil.copytree(
            repo / "kai0/src/openpi",
            stage / "kai0/src/openpi",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        for relative, expected in protocol["source_sha256"].items():
            relative_path = Path(relative)
            frozen = frozen_root / relative_path
            source = frozen if frozen.is_file() else repo / relative_path
            actual = sha256(source)
            if actual != expected:
                raise ValueError(
                    f"source identity unavailable: {relative}: {actual} != {expected}"
                )
            destination = stage / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_checks.append(
                {
                    "path": relative,
                    "source": str(source),
                    "sha256": actual,
                    "frozen_override": source == frozen,
                }
            )

        evidence = {
            protocol["scene_manifest"],
            "kai0/checkpoints/pi05_base/params/_METADATA",
            "kai0/assets/pi05_robotwin_a0_public_exact_bj/robotwin2.0_absolute_meanstd/norm_stats.json",
            protocol["teacher"]["artifact"],
            protocol["teacher"]["artifact_manifest"],
            "logs/predictive/p0_eval/p0_gate.accepted",
            "logs/crave_r0/probe_gate/r0_gate.accepted",
            "lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p0_final_audit.json",
            "kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999/params/_METADATA",
            "lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json",
            "lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz",
            "lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz",
            "lmvla/lmwm/data/pi05_crave_r0_v1/reference_trajectories.npz",
        }
        for relative in sorted(evidence):
            source = repo / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = stage / relative
            if not destination.exists():
                link_file(source, destination)

        audit = {
            "schema_version": 1,
            "protocol": "pi05_r1_frozen_source_overlay_v1",
            "canonical_repo": str(repo),
            "overlay": str(output),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "source_checks": source_checks,
            "frozen_override_count": sum(
                int(item["frozen_override"]) for item in source_checks
            ),
            "passed": True,
        }
        (stage / "overlay_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "READY").write_text(
            f"protocol_sha256={audit['protocol_sha256']}\n", encoding="utf-8"
        )

        if output.exists():
            marker = output / "overlay_audit.json"
            if not marker.is_file():
                raise RuntimeError(f"refusing to replace unmanaged overlay: {output}")
            shutil.rmtree(output)
        os.replace(stage, output)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output or repo / DEFAULT_OUTPUT_RELATIVE
    print(json.dumps(materialize(repo, output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
