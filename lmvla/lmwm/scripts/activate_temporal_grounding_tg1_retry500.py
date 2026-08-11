#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "temporal_grounding_tg1_retry500_v1"
ARCHIVE_SUFFIX = ".pre_retry500_v1"
ROOT_SPECS = {
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1a_normal": 20,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1a_null": 19,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1a_persistence": 21,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1a_shuffled": 0,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1b_future_off_e36": 20,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1b_future_off_e50": 0,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1b_local_wm_e36": 0,
    "lmvla/lawam/results/eval_runs/robotwin/temporal_grounding_tg1b_local_wm_e50": 18,
}
LEGACY_FEATURE_ROOT = "logs/temporal_grounding/tg1a/predicted_endpoint_features"
FEATURE_ROOT = "logs/tg1_retry500/predicted_endpoint_features"
CAPTURE_MARKER = (
    "logs/resource_markers/"
    "temporal_grounding_tg1a_retry500_normal_capture_complete.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_summaries(root: Path) -> int:
    return sum(1 for _ in root.glob("seed*/**/tasks/*/summary.json"))


def validate_pre_activation(repo: Path) -> tuple[list[dict], dict]:
    if (repo / CAPTURE_MARKER).exists():
        raise RuntimeError("refusing activation: normal feature capture is marked complete")
    moves = []
    for relative, expected_count in ROOT_SPECS.items():
        source = repo / relative
        archive = source.with_name(source.name + ARCHIVE_SUFFIX)
        if archive.exists():
            raise FileExistsError(f"archive target already exists: {archive}")
        actual_count = count_summaries(source) if source.exists() else 0
        if actual_count != expected_count:
            raise RuntimeError(
                f"partial-root drift for {relative}: expected {expected_count} "
                f"summaries, found {actual_count}"
            )
        if expected_count and not source.is_dir():
            raise RuntimeError(f"expected partial root is missing: {source}")
        if source.exists():
            moves.append({"source": source, "archive": archive, "summaries": actual_count})

    legacy_feature = repo / LEGACY_FEATURE_ROOT
    if not legacy_feature.is_dir() or not any(legacy_feature.iterdir()):
        raise RuntimeError(f"expected non-empty legacy feature root: {legacy_feature}")
    new_feature = repo / FEATURE_ROOT
    if new_feature.exists():
        raise FileExistsError(f"retry500 feature root is not empty: {new_feature}")
    feature_files = sum(1 for path in legacy_feature.rglob("*") if path.is_file())
    feature_record = {
        "source": str(legacy_feature.relative_to(repo)),
        "archive_mode": "retained_in_place_read_only_excluded",
        "files": feature_files,
        "retry500_root": FEATURE_ROOT,
    }
    return moves, feature_record


def activate(repo: Path, manifest_path: Path, *, check_only: bool) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or manifest.get("retry_cap", {}).get("new") != 500:
        raise ValueError("invalid TG1 retry500 amendment")
    marker_path = repo / manifest["activation"]["marker"]
    if marker_path.exists():
        raise FileExistsError(f"TG1 retry500 amendment is already active: {marker_path}")

    for relative, expected in manifest.get("file_sha256", {}).items():
        path = repo / relative
        if sha256_file(path) != expected:
            raise ValueError(f"frozen amendment file drift: {relative}")

    moves, legacy_feature = validate_pre_activation(repo)
    report = {
        "protocol": PROTOCOL,
        "activated": not check_only,
        "retry_cap": 500,
        "manifest": str(manifest_path.relative_to(repo)),
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_roots_empty_after_archive": False,
        "archives": [
            {
                "source": str(item["source"].relative_to(repo)),
                "archive": str(item["archive"].relative_to(repo)),
                "summaries": item["summaries"],
            }
            for item in moves
        ],
        "legacy_feature_capture": legacy_feature,
    }
    if check_only:
        return report

    completed = []
    try:
        for item in moves:
            item["source"].rename(item["archive"])
            completed.append(item)
        nonempty = [str((repo / relative)) for relative in ROOT_SPECS if (repo / relative).exists()]
        if nonempty or (repo / FEATURE_ROOT).exists():
            raise RuntimeError(f"canonical roots not empty after archive: {nonempty}")
    except Exception:
        for item in reversed(completed):
            if item["archive"].exists() and not item["source"].exists():
                item["archive"].rename(item["source"])
        raise

    report["canonical_roots_empty_after_archive"] = True
    report["activated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = activate(
        args.repo.resolve(), args.manifest.resolve(), check_only=args.check_only
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
