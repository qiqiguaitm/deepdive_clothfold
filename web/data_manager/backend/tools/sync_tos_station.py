#!/usr/bin/env python3
"""Upload one collection station to TOS without propagating deletions.

The uploader is deliberately non-destructive: it never lists local absences as
deletions and never calls a TOS delete API.  Multiple robots can therefore use
one bucket safely either with isolated station prefixes or with shared dataset
leaves whose payloads are separated by ``chunk-XXX``.  In shared mode metadata
is redirected below ``meta/by_station/<machine-id>`` so writers never race on
the canonical metadata files.

    KAI0/stations/ipc01/...
    KAI0/stations/visrobot02/...

Existing objects with the same key and size are skipped so an interrupted run
can be resumed cheaply.  Metadata is always refreshed because its contents can
change without its byte length changing.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LARGE_FILE_BYTES = 64 * 1024 * 1024
PART_SIZE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class UploadItem:
    local: Path
    key: str
    relative: str
    size: int
    metadata: bool


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-destructively upload a station dataset tree to TOS",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0")),
    )
    parser.add_argument(
        "--machine-id",
        default=os.environ.get("KAI0_MACHINE_ID", socket.gethostname().split(".")[0]),
    )
    parser.add_argument(
        "--prefix",
        help="TOS prefix (default: KAI0/stations/<machine-id>)",
    )
    parser.add_argument(
        "--shared-chunks",
        action="store_true",
        help=(
            "Write payloads below a shared prefix (default KAI0) and redirect "
            "metadata to meta/by_station/<machine-id>"
        ),
    )
    parser.add_argument(
        "--copy-from-prefix",
        help="Server-side copy matching objects from this prefix before uploading",
    )
    parser.add_argument("--task", action="append", help="Only upload this task (repeatable)")
    parser.add_argument(
        "--version", action="append",
        help="Only upload this dataset version, e.g. v5 (repeatable)",
    )
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Upload only metadata files (useful for seeding station sidecars)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_component(value: str, label: str) -> str:
    value = value.strip().strip("/")
    if not value or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _valid_dataset_leaves(
    root: Path, tasks: set[str], versions: set[str]
) -> list[Path]:
    leaves: set[Path] = set()
    for parquet in root.glob("*/*/*/*/data/chunk-*/episode_*.parquet"):
        rel = parquet.relative_to(root)
        if tasks and rel.parts[0] not in tasks:
            continue
        if versions and rel.parts[2] not in versions:
            continue
        leaves.add(parquet.parents[2])
    for parquet in root.glob("*/*/*/data/chunk-*/episode_*.parquet"):
        rel = parquet.relative_to(root)
        if tasks and rel.parts[0] not in tasks:
            continue
        # Legacy three-level leaves do not have a version component.
        if versions:
            continue
        leaves.add(parquet.parents[2])
    return sorted(leaves)


def _items(
    root: Path,
    prefix: str,
    machine: str,
    tasks: set[str],
    versions: set[str],
    shared_chunks: bool,
) -> tuple[list[UploadItem], list[Path]]:
    leaves = _valid_dataset_leaves(root, tasks, versions)
    items: list[UploadItem] = []
    for leaf in leaves:
        for path in leaf.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            is_metadata = "/meta/" in f"/{rel}"
            if shared_chunks and is_metadata:
                leaf_rel = leaf.relative_to(root).as_posix()
                key = f"{prefix}/{leaf_rel}/meta/by_station/{machine}/{path.name}"
            else:
                key = f"{prefix}/{rel}"
            items.append(UploadItem(
                local=path,
                key=key,
                relative=rel,
                size=path.stat().st_size,
                metadata=is_metadata,
            ))
    return items, leaves


def _remote_sizes(client, bucket: str, prefixes: list[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for prefix in prefixes:
        token = ""
        while True:
            out = client.list_objects_type2(
                bucket, prefix=f"{prefix.strip('/')}/",
                continuation_token=token, max_keys=1000,
            )
            for obj in out.contents:
                sizes[obj.key] = int(getattr(obj, "size", 0))
            if not out.is_truncated:
                break
            token = out.next_continuation_token
    return sizes


def main() -> int:
    args = _args()
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: data root does not exist: {root}", file=sys.stderr)
        return 2
    if args.workers < 1 or args.workers > 32:
        print("ERROR: --workers must be in [1, 32]", file=sys.stderr)
        return 2

    machine = _safe_component(args.machine_id, "machine id")
    default_prefix = "KAI0" if args.shared_chunks else f"KAI0/stations/{machine}"
    prefix = _safe_component(args.prefix or default_prefix, "prefix")
    tasks = set(args.task or ())
    versions = {_safe_component(value, "version") for value in (args.version or ())}
    items, leaves = _items(root, prefix, machine, tasks, versions, args.shared_chunks)
    if args.metadata_only:
        items = [item for item in items if item.metadata]
    total_bytes = sum(item.size for item in items)
    print(
        f"station={machine} root={root} prefix={prefix} "
        f"layout={'shared_chunks' if args.shared_chunks else 'station'} "
        f"leaves={len(leaves)} files={len(items)} bytes={total_bytes}",
        flush=True,
    )
    for leaf in leaves:
        print(f"leaf={leaf.relative_to(root)}", flush=True)
    if not items:
        print("ERROR: no complete dataset leaves found", file=sys.stderr)
        return 2
    if args.dry_run:
        print("dry-run: no TOS writes performed", flush=True)
        return 0

    required = ("KAI0_TOS_AK", "KAI0_TOS_SK")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print(f"ERROR: missing environment: {', '.join(missing)}", file=sys.stderr)
        return 2

    import tos

    bucket = os.environ.get("KAI0_TOS_BUCKET", "transfer-shanghai")
    client = tos.TosClientV2(
        os.environ["KAI0_TOS_AK"],
        os.environ["KAI0_TOS_SK"],
        os.environ.get("KAI0_TOS_ENDPOINT", "tos-cn-shanghai.volces.com"),
        os.environ.get("KAI0_TOS_REGION", "cn-shanghai"),
    )
    scan_prefixes = [f"{prefix}/{leaf.relative_to(root).as_posix()}" for leaf in leaves]
    existing = {} if args.metadata_only else _remote_sizes(client, bucket, scan_prefixes)
    copy_prefix = (
        _safe_component(args.copy_from_prefix, "copy-from prefix")
        if args.copy_from_prefix else ""
    )
    copy_sizes = _remote_sizes(
        client, bucket,
        [f"{copy_prefix}/{leaf.relative_to(root).as_posix()}" for leaf in leaves],
    ) if copy_prefix else {}
    pending = [
        item for item in items
        if item.metadata or existing.get(item.key) != item.size
    ]
    skipped = len(items) - len(pending)
    pending_bytes = sum(item.size for item in pending)
    print(
        f"bucket={bucket} existing={len(existing)} skipped={skipped} "
        f"pending={len(pending)} pending_bytes={pending_bytes}",
        flush=True,
    )

    lock = threading.Lock()
    done_files = 0
    done_bytes = 0
    copied_files = 0
    failures: list[tuple[UploadItem, str]] = []
    started = time.monotonic()

    def upload(item: UploadItem) -> tuple[UploadItem, bool]:
        source_key = f"{copy_prefix}/{item.relative}" if copy_prefix else ""
        if source_key and copy_sizes.get(source_key) == item.size:
            client.copy_object(bucket, item.key, bucket, source_key)
            return item, True
        if item.size >= LARGE_FILE_BYTES:
            client.upload_file(
                bucket, item.key, str(item.local), task_num=2,
                part_size=PART_SIZE_BYTES,
            )
        else:
            client.put_object_from_file(bucket, item.key, str(item.local))
        return item, False

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="tos-station") as pool:
        futures = {pool.submit(upload, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                _, copied = future.result()
                with lock:
                    done_files += 1
                    done_bytes += item.size
                    copied_files += int(copied)
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"progress={done_files}/{len(pending)} "
                        f"bytes={done_bytes}/{pending_bytes} rate={done_bytes/elapsed/1e6:.1f}MB/s",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append((item, str(exc)))
                print(f"FAILED key={item.key}: {exc}", file=sys.stderr, flush=True)

    manifest = {
        "machine_id": machine,
        "dataset_chunk": int(os.environ.get("KAI0_DATASET_CHUNK", "0")),
        "source_host": socket.gethostname(),
        "source_data_root": str(root),
        "prefix": prefix,
        "layout": "shared_chunks" if args.shared_chunks else "station",
        "complete_dataset_leaves": [str(p.relative_to(root)) for p in leaves],
        "files": len(items),
        "bytes": total_bytes,
        "delete_propagation": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not failures:
        manifest_key = (
            f"{prefix}/_stations/{machine}.json"
            if args.shared_chunks else f"{prefix}/_station.json"
        )
        client.put_object(
            bucket,
            manifest_key,
            content=json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        print(
            f"complete transferred={done_files} copied={copied_files} skipped={skipped} "
            f"manifest={manifest_key} delete_propagation=false",
            flush=True,
        )
        return 0

    print(f"ERROR: {len(failures)} uploads failed; rerun to resume", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
