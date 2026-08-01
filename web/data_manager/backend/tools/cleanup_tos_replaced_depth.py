#!/usr/bin/env python3
"""Delete TOS v5 Zarr ZIPs only after their FFV1 replacements are verified."""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--chunk", required=True, help="Exact owner chunk, e.g. chunk-000")
    parser.add_argument("--version", default="v5")
    parser.add_argument("--prefix", default="KAI0")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-machine")
    parser.add_argument("--confirm-chunk")
    return parser.parse_args()


def _safe(value: str, label: str) -> str:
    value = value.strip().strip("/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _list(client, bucket: str, prefix: str) -> dict[str, int]:
    token = ""
    rows: dict[str, int] = {}
    while True:
        out = client.list_objects_type2(
            bucket, prefix=prefix, continuation_token=token, max_keys=1000
        )
        rows.update((obj.key, int(getattr(obj, "size", 0))) for obj in out.contents)
        if not out.is_truncated:
            return rows
        token = out.next_continuation_token


def main() -> int:
    args = _args()
    root = args.data_root.expanduser().resolve()
    machine = _safe(args.machine_id, "machine id")
    chunk = _safe(args.chunk, "chunk")
    version = _safe(args.version, "version")
    prefix = _safe(args.prefix, "prefix")
    if not re.fullmatch(r"chunk-\d{3}", chunk):
        raise ValueError(f"invalid chunk: {chunk!r}")
    if args.apply and (
        args.confirm_machine != machine or args.confirm_chunk != chunk
    ):
        raise RuntimeError("--apply requires matching machine and chunk confirmations")
    for name in ("KAI0_TOS_AK", "KAI0_TOS_SK"):
        if not os.environ.get(name):
            raise RuntimeError(f"missing environment: {name}")

    import tos

    bucket = os.environ.get("KAI0_TOS_BUCKET", "transfer-shanghai")
    client = tos.TosClientV2(
        os.environ["KAI0_TOS_AK"], os.environ["KAI0_TOS_SK"],
        os.environ.get("KAI0_TOS_ENDPOINT", "tos-cn-shanghai.volces.com"),
        os.environ.get("KAI0_TOS_REGION", "cn-shanghai"),
    )

    local: dict[str, int] = {}
    leaf_by_key: dict[str, str] = {}
    for mkv in root.glob(
        f"*/*/{version}/*-{version}/videos/{chunk}/observation.depth.*/*.mkv"
    ):
        rel = mkv.relative_to(root).as_posix()
        key = f"{prefix}/{rel}"
        local[key] = mkv.stat().st_size
        leaf_by_key[key] = "/".join(rel.split("/")[:4])
    if not local:
        raise RuntimeError(f"no local {version} MKV files for {chunk}")

    remote: dict[str, int] = {}
    for leaf in sorted(set(leaf_by_key.values())):
        remote.update(_list(client, bucket, f"{prefix}/{leaf}/videos/{chunk}/"))

    unverified = [
        key for key, size in local.items() if remote.get(key) != size
    ]
    if unverified:
        print(f"BLOCKED: {len(unverified)} MKV replacements missing or size-mismatched")
        for key in unverified[:20]:
            print(f"  local={local[key]} remote={remote.get(key)} {key}")
        return 1

    candidates = sorted(
        (key[:-len(".mkv")] + ".zarr.zip", remote[key[:-len(".mkv")] + ".zarr.zip"])
        for key in local
        if key[:-len(".mkv")] + ".zarr.zip" in remote
    )
    total = sum(size for _, size in candidates)
    print(
        f"verified_mkv={len(local)} old_zip_candidates={len(candidates)} "
        f"bytes={total} mode={'apply' if args.apply else 'dry-run'}"
    )
    if not args.apply:
        print("dry-run: no TOS objects deleted")
        return 0

    for offset in range(0, len(candidates), 500):
        batch = candidates[offset:offset + 500]
        objects = [tos.models2.ObjectTobeDeleted(key=key) for key, _ in batch]
        out = client.delete_multi_objects(bucket, objects, quiet=False)
        if out.error:
            raise RuntimeError(
                "TOS delete failed: "
                + ", ".join(f"{row.key}:{row.code}" for row in out.error[:10])
            )
        print(f"delete_progress={min(offset + len(batch), len(candidates))}/{len(candidates)}")

    remaining: list[str] = []
    for leaf in sorted(set(leaf_by_key.values())):
        listed = _list(client, bucket, f"{prefix}/{leaf}/videos/{chunk}/")
        remaining.extend(key for key, _ in candidates if key in listed)
    if remaining:
        raise RuntimeError(f"{len(remaining)} old ZIP objects remain after deletion")
    print(f"complete deleted={len(candidates)} bytes={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
