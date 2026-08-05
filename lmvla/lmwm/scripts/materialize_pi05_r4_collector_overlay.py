#!/usr/bin/env python3
"""Materialize the isolated RoboTwin trajectory collector from frozen sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


LAWAM_COMMIT = "865e0b631c67cc5463feab04e34056a5538186c5"
PATCH_RELATIVE = Path(
    "lmvla/paper_iclr_lmvla/frozen_sources/pi05_r4_collector_v1/"
    "robotwin_batch_bridge.patch"
)
PATCH_SHA256 = "3a39b0c77077561a85922a9aef1a9828900626557e0f7e14e48be9a58f77058b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
        count += 1
        total_bytes += size
    return count, total_bytes, digest.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive:
        target = (destination / member.name).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.name}")
        archive.extract(member, destination, filter="data")


def materialize(repo: Path, source: Path, output: Path) -> dict:
    patch = repo / PATCH_RELATIVE
    if sha256(patch) != PATCH_SHA256:
        raise ValueError("R4 collector patch hash mismatch")
    source_commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != LAWAM_COMMIT:
        raise ValueError(f"lawam source commit mismatch: {source_commit}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        process = subprocess.Popen(
            ["git", "-C", str(source), "archive", "--format=tar", LAWAM_COMMIT],
            stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            safe_extract(archive, temporary)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(patch)],
            cwd=temporary,
            check=True,
        )
        file_count, total_bytes, inventory_sha256 = inventory(temporary)
        ready = {
            "schema_version": 1,
            "protocol": "pi05_r4_trajectory_collector_overlay_v1",
            "lawam_commit": LAWAM_COMMIT,
            "patch": str(PATCH_RELATIVE),
            "patch_sha256": PATCH_SHA256,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "inventory_sha256": inventory_sha256,
        }
        (temporary / "COLLECTOR_READY").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        return ready
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = (args.source or repo / "lmvla/lawam").resolve()
    output = (
        args.output
        or repo / "logs/frozen_source_overlays/pi05_r4_collector_v1/lawam"
    ).resolve()
    print(json.dumps(materialize(repo, source, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
