#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify(
    repo: Path,
    manifest_path: Path,
    bundle: str,
    amendment_path: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle") != bundle:
        raise ValueError(
            f"Bundle mismatch: requested {bundle!r}, manifest has {manifest.get('bundle')!r}"
        )
    if not manifest.get("frozen") or manifest.get("manual_execution_authorized"):
        raise ValueError("Temporal-grounding bundle must be frozen and scheduler-owned.")

    source = manifest["source"]
    implementation_commit = source["outer_implementation_commit"]
    subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        check=True,
    )
    inner_repo = repo / "lmvla/lawam"
    inner_commit = git(inner_repo, "rev-parse", "HEAD")
    amendment = None
    amendment_files: dict[str, str] = {}
    if amendment_path is not None:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if (
            amendment.get("experiment_protocol_changed") is not False
            or amendment.get("operator_authorized") is not True
        ):
            raise ValueError("Runtime amendment must be authorized and protocol-preserving.")
        if bundle not in amendment.get("applies_to_bundles", []):
            raise ValueError(f"Runtime amendment is not admitted for bundle {bundle}.")
        identity = amendment.get("source_identity") or {}
        if identity.get("parent_lawam_commit") != source["lawam_commit"]:
            raise ValueError("Runtime amendment does not extend the frozen LaWAM commit.")
        if identity.get("lawam_commit") != inner_commit:
            raise ValueError(
                "Runtime amendment LaWAM mismatch: "
                f"expected {identity.get('lawam_commit')}, got {inner_commit}"
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(inner_repo),
                "merge-base",
                "--is-ancestor",
                source["lawam_commit"],
                inner_commit,
            ],
            check=True,
        )
        changed = {
            line
            for line in git(
                inner_repo,
                "diff",
                "--name-only",
                f"{source['lawam_commit']}..{inner_commit}",
            ).splitlines()
            if line
        }
        expected_changed = set(identity.get("lawam_changed_files") or [])
        if changed != expected_changed:
            raise ValueError(
                "Runtime amendment LaWAM diff mismatch: "
                f"expected {sorted(expected_changed)}, got {sorted(changed)}"
            )
        amendment_files = amendment.get("files") or {}
    elif inner_commit != source["lawam_commit"]:
        raise ValueError(
            f"LaWAM commit mismatch: expected {source['lawam_commit']}, got {inner_commit}"
        )

    checked = {}
    for relative, expected in manifest["file_sha256"].items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen bundle input is missing: {path}")
        actual = sha256_file(path)
        amended_expected = amendment_files.get(relative)
        if actual != expected and actual != amended_expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}")
        checked[relative] = actual

    for relative, expected in amendment_files.items():
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Runtime amendment input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Runtime amendment SHA-256 mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        checked[relative] = actual

    result = {
        "schema_version": 1,
        "bundle": bundle,
        "manifest": str(manifest_path),
        "outer_head": git(repo, "rev-parse", "HEAD"),
        "outer_implementation_commit": implementation_commit,
        "lawam_commit": inner_commit,
        "verified_files": len(checked),
    }
    if amendment_path is not None:
        result["runtime_amendment"] = str(amendment_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", choices=("TG1A", "TG1B", "TG2"), required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    amendment = args.amendment
    if amendment is None and os.environ.get("TEMPORAL_GROUNDING_RUNTIME_AMENDMENT"):
        amendment = Path(os.environ["TEMPORAL_GROUNDING_RUNTIME_AMENDMENT"])
    result = verify(
        args.repo.resolve(),
        args.manifest.resolve(),
        args.bundle,
        amendment.resolve() if amendment is not None else None,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
