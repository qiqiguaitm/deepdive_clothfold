#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_ROOT=${NORTH_ROOT:-/vePFS-North-E/vis_robot}
TRAIN_REPO=$NORTH_ROOT/tim/workspace/deepdive_kai0
EVAL_REPO=$NORTH_ROOT/workspace/deepdive_kai0/.staging/pi05_r4_eval_north_v1/repo
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_replication_eval_north_stage.ok
REMOTE_MARKER=$EVAL_REPO/logs/resource_markers/pi05_r4_replication_eval_north_stage.ok
AMENDMENT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_north_eval_replication_amendment_v1.json

test -s "$AMENDMENT"
test -s "$REPO/logs/resource_markers/pi05_r4_replication_north_stage.ok"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$TRAIN_REPO" "$EVAL_REPO" "$REMOTE_MARKER" <<'REMOTE'
set -Eeuo pipefail
train_repo=$1
eval_repo=$2
marker=$3
protocol=$train_repo/lmvla/paper_iclr_lmvla/manifests/pi05_r4_replication_protocol_v1.json
python=$eval_repo/runtime/python/bin/python3.12
site=$eval_repo/runtime/venv/lib/python3.12/site-packages
lerobot=$eval_repo/runtime/lerobot/src

test -s "$train_repo/logs/resource_markers/pi05_r4_replication_north_stage.ok"
test -s "$eval_repo/logs/resource_markers/pi05_r4_eval_north_stage.ok"
test -x "$python"
PYTHONPATH="$lerobot:$site" "$python" - "$train_repo" "$eval_repo" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

source, destination = map(pathlib.Path, sys.argv[1:])
protocol_path = source / "lmvla/paper_iclr_lmvla/manifests/pi05_r4_replication_protocol_v1.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
rows = [(entry["path"], entry["sha256"]) for entry in protocol["parents"].values()]
rows.extend(protocol["file_sha256"].items())
for relative, expected in rows:
    source_path = source / relative
    actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"source mismatch: {relative}: {actual} != {expected}")
    destination_path = destination / relative
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    copied = hashlib.sha256(destination_path.read_bytes()).hexdigest()
    if copied != expected:
        raise ValueError(f"copied mismatch: {relative}: {copied} != {expected}")
accepted = source / "logs/r4/seed1000/r4_gate.accepted"
accepted_destination = destination / "logs/r4/seed1000/r4_gate.accepted"
accepted_destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(accepted, accepted_destination)
print(f"verified and copied files={len(rows)}")
PY

mkdir -p "$(dirname "$marker")"
printf 'validated=%s\ntrain_repo=%s\neval_repo=%s\n' \
  "$(date -u +%FT%TZ)" "$train_repo" "$eval_repo" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_eval_repo=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$EVAL_REPO" "$REMOTE_MARKER" >"$LOCAL_MARKER"
