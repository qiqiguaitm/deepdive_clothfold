#!/usr/bin/env python3
"""Rank submission targets using live capacity and data locality.

The static resource catalog records queue, filesystem, and development-host
topology. The scheduler snapshot contributes current free-card information.
No credential values are read or printed by this tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
DEFAULT_CATALOG = SCRIPT_DIR / "submission_resource_catalog.json"
DEFAULT_SNAPSHOT = REPO / "logs/resource_scheduler_snapshot.json"


@dataclass(frozen=True)
class LiveCapacity:
    free_gpus: int
    capacity_gpus: int
    queueing: bool
    credential_profile: str | None = None
    submission_enabled: bool = True
    detail: str = ""


@dataclass(frozen=True)
class Recommendation:
    rank: int
    resource: str
    score: int
    region: str
    queue: str | None
    credential_profile: str | None
    free_gpus: int
    capacity_gpus: int
    filesystem: str
    development_host: str
    transfer_required: bool
    immediately_runnable: bool
    reason: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def infer_filesystems(paths: Iterable[str], catalog: dict[str, Any]) -> set[str]:
    locations: set[str] = set()
    filesystems = catalog.get("filesystems", {})
    for raw_path in paths:
        normalized = str(Path(raw_path).expanduser())
        matches = [
            (len(prefix), name)
            for name, spec in filesystems.items()
            for prefix in spec.get("mount_prefixes", [])
            if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/")
        ]
        if not matches:
            raise ValueError(
                f"cannot infer filesystem for {raw_path!r}; use --data-location"
            )
        locations.add(max(matches)[1])
    return locations


def _bounded_free(*values: int) -> int:
    return max(0, min(values))


def live_capacity(resource: str, spec: dict[str, Any], snapshot: dict[str, Any]) -> LiveCapacity:
    resources = snapshot.get("resources", {})
    catalog_capacity = int(spec["capacity_gpus"])
    if resource in ("gf1", "local"):
        live = resources.get(resource, {})
        capacity = int(live.get("count", catalog_capacity))
        submission_enabled = bool(live.get("submission_enabled", True))
        return LiveCapacity(
            free_gpus=int(live.get("free_count", 0)) if submission_enabled else 0,
            capacity_gpus=capacity,
            queueing=False,
            submission_enabled=submission_enabled,
            detail=(
                "direct host"
                if submission_enabled
                else str(live.get("retired_reason", "direct host retired"))
            ),
        )
    if resource == "Robot-East-H20":
        live = resources.get(resource, {})
        capacity = int(live.get("capacity", catalog_capacity))
        active = int(live.get("active_gpus_all_users", capacity))
        queueing = bool(live.get("queueing_all_users", []))
        return LiveCapacity(
            free_gpus=max(0, capacity - active),
            capacity_gpus=capacity,
            queueing=queueing,
            detail=f"all-users active={active}",
        )
    if resource == "robot-task":
        live = resources.get(resource, {})
        capacity = int(live.get("capacity", catalog_capacity))
        if not live.get("submission_enabled", True):
            return LiveCapacity(
                free_gpus=0,
                capacity_gpus=capacity,
                queueing=True,
                submission_enabled=False,
                detail="new submissions temporarily disabled",
            )
        active_all = int(live.get("active_gpus_all_users", capacity))
        active_owned = int(live.get("owned_active_gpus", active_all))
        limit = int(spec.get("personal_limit_gpus", capacity))
        free = _bounded_free(capacity - active_all, limit - active_owned)
        queueing = bool(live.get("queueing_all_users", []))
        return LiveCapacity(
            free_gpus=free,
            capacity_gpus=capacity,
            queueing=queueing,
            detail=f"all-users active={active_all}; owned={active_owned}/{limit}",
        )
    if resource == "Robot-North-H20":
        live = resources.get("beijing", {})
        capacity = int(live.get("capacity", catalog_capacity))
        active_all = int(live.get("active_gpus_all_users", capacity))
        queue_free = max(0, capacity - active_all)
        primary_limit = int(live.get("personal_limit", spec.get("personal_limit_gpus", 20)))
        primary_active = int(live.get("owned_active_gpus", primary_limit))
        primary_queued = int(live.get("owned_queued_gpus", 0))
        primary_queueing = bool(live.get("owned_queueing", []))
        primary_free = _bounded_free(
            queue_free, primary_limit - primary_active - primary_queued
        )

        candidates = [
            (
                primary_free,
                not primary_queueing,
                "primary",
                primary_queueing,
            )
        ]
        backup = live.get("backup", {})
        if (
            backup.get("enabled")
            and backup.get("submission_enabled", backup.get("enabled"))
            and backup.get("available")
        ):
            backup_limit = int(backup.get("personal_limit", 20))
            backup_active = int(
                backup.get("identity_active_gpus", backup.get("managed_active_gpus", backup_limit))
            )
            backup_queued = int(
                backup.get("identity_queued_gpus", backup.get("managed_queued_gpus", 0))
            )
            backup_queueing = bool(
                backup.get("identity_queueing", backup.get("managed_queueing", []))
            )
            backup_free = _bounded_free(
                queue_free, backup_limit - backup_active - backup_queued
            )
            candidates.append(
                (
                    backup_free,
                    not backup_queueing,
                    "backup",
                    backup_queueing,
                )
            )
        free, profile_has_capacity, profile, profile_queueing = max(
            candidates,
            key=lambda item: (item[1], item[0], item[2] == "primary"),
        )
        queueing = bool(live.get("queueing_all_users", [])) or profile_queueing
        return LiveCapacity(
            free_gpus=free,
            capacity_gpus=capacity,
            queueing=queueing or not profile_has_capacity,
            credential_profile=profile,
            detail=(
                f"all-users active={active_all}; profile={profile}; "
                f"identity GPU free={free}"
            ),
        )
    raise ValueError(f"unsupported resource in catalog: {resource}")


def preference_order(gpus: int, catalog: dict[str, Any]) -> list[str]:
    preferences = catalog.get("gpu_count_preferences", {})
    if str(gpus) in preferences:
        return list(preferences[str(gpus)])
    resources = catalog.get("resources", {})
    return sorted(
        (
            name
            for name, spec in resources.items()
            if int(spec["capacity_gpus"]) >= gpus
        ),
        key=lambda name: (
            resources[name]["region"] != "cn-beijing" if gpus >= 16 else False,
            int(resources[name]["capacity_gpus"]),
            name,
        ),
    )


def rank_targets(
    *,
    gpus: int,
    catalog: dict[str, Any],
    snapshot: dict[str, Any],
    data_locations: set[str],
    strict_locality: bool = False,
) -> list[Recommendation]:
    if gpus <= 0:
        raise ValueError("--gpus must be positive")
    resources = catalog["resources"]
    order = preference_order(gpus, catalog)
    scoring = catalog["scoring"]
    ranked: list[tuple[int, int, str, LiveCapacity, bool, bool, list[str]]] = []
    for preference_rank, resource in enumerate(order):
        spec = resources[resource]
        if int(spec["capacity_gpus"]) < gpus:
            continue
        live = live_capacity(resource, spec, snapshot)
        transfer_required = bool(data_locations - {spec["filesystem"]})
        if strict_locality and transfer_required:
            continue
        immediate = live.free_gpus >= gpus and not live.queueing
        score = preference_rank * int(scoring["preference_step"])
        reasons = [f"{gpus}-GPU preference #{preference_rank + 1}"]
        if not live.submission_enabled:
            score += int(scoring["disabled_penalty"])
            reasons.append("new submissions temporarily disabled")
        if transfer_required:
            score += int(scoring["cross_filesystem_penalty"])
            reasons.append("cross-filesystem staging required")
        elif data_locations:
            reasons.append("data-local")
        else:
            reasons.append("data location unknown")
        if live.free_gpus < gpus:
            score += int(scoring["insufficient_live_capacity_penalty"])
            reasons.append(f"only {live.free_gpus} GPUs currently free")
        if live.queueing:
            score += int(scoring["queued_penalty"])
            reasons.append("queue already has waiting work")
        if gpus >= 16 and spec["kind"] == "platform":
            reasons.append("gang-scheduling requires whole 8-GPU nodes")
        reasons.append(live.detail)
        ranked.append(
            (
                score,
                preference_rank,
                resource,
                live,
                transfer_required,
                immediate,
                reasons,
            )
        )
    any_immediate = any(item[5] for item in ranked)
    if any_immediate:
        ranked.sort(key=lambda item: (not item[5], item[0], item[1], item[2]))
    else:
        # When every target must wait, North is the designated queue sink.
        ranked.sort(
            key=lambda item: (
                item[2] != "Robot-North-H20",
                item[0],
                item[1],
                item[2],
            )
        )
    return [
        Recommendation(
            rank=index,
            resource=resource,
            score=score,
            region=resources[resource]["region"],
            queue=resources[resource].get("queue"),
            credential_profile=live.credential_profile,
            free_gpus=live.free_gpus,
            capacity_gpus=live.capacity_gpus,
            filesystem=resources[resource]["filesystem"],
            development_host=resources[resource]["development_host"],
            transfer_required=transfer_required,
            immediately_runnable=immediate,
            reason="; ".join(reasons),
        )
        for index, (
            score,
            _preference_rank,
            resource,
            live,
            transfer_required,
            immediate,
            reasons,
        ) in enumerate(ranked, start=1)
    ]


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def snapshot_age_seconds(snapshot: dict[str, Any]) -> float:
    timestamp = parse_timestamp(str(snapshot["timestamp"]))
    return (datetime.now(timezone.utc) - timestamp).total_seconds()


def print_table(recommendations: list[Recommendation], locations: set[str]) -> None:
    location_text = ",".join(sorted(locations)) if locations else "unknown"
    print(f"Data location: {location_text}")
    print(
        "Rank  Target             Region        Free      Run now  "
        "Transfer  Dev host       Credential"
    )
    for item in recommendations:
        credential = item.credential_profile or "-"
        print(
            f"{item.rank:>4}  {item.resource:<18} {item.region:<13} "
            f"{item.free_gpus:>2}/{item.capacity_gpus:<5} "
            f"{('yes' if item.immediately_runnable else 'no'):<8} "
            f"{('yes' if item.transfer_required else 'no'):<9} "
            f"{item.development_host:<14} {credential}"
        )
        print(f"      {item.reason}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, required=True, help="requested GPU count")
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        help="dataset or checkpoint path; repeat for multiple required trees",
    )
    parser.add_argument(
        "--data-location",
        action="append",
        choices=("east_shared", "north_shared"),
        default=[],
        help="explicit data filesystem when no concrete path is available",
    )
    parser.add_argument(
        "--strict-locality",
        action="store_true",
        help="exclude targets that require cross-filesystem staging",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--max-snapshot-age",
        type=int,
        default=180,
        help="reject a scheduler snapshot older than this many seconds",
    )
    parser.add_argument("--allow-stale", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        catalog = load_json(args.catalog)
        snapshot = load_json(args.snapshot)
        age = snapshot_age_seconds(snapshot)
        if age > args.max_snapshot_age and not args.allow_stale:
            raise ValueError(
                f"scheduler snapshot is stale ({age:.0f}s > {args.max_snapshot_age}s)"
            )
        locations = set(args.data_location)
        locations.update(infer_filesystems(args.data_path, catalog))
        recommendations = rank_targets(
            gpus=args.gpus,
            catalog=catalog,
            snapshot=snapshot,
            data_locations=locations,
            strict_locality=args.strict_locality,
        )
        if not recommendations:
            raise ValueError("no catalog resource can satisfy this request")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_output:
        print(
            json.dumps(
                {
                    "requested_gpus": args.gpus,
                    "data_locations": sorted(locations),
                    "snapshot_timestamp": snapshot["timestamp"],
                    "snapshot_age_seconds": round(age, 1),
                    "recommendations": [asdict(item) for item in recommendations],
                },
                indent=2,
            )
        )
    else:
        print_table(recommendations, locations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
