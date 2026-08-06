#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_ROOT=${NORTH_ROOT:-/vePFS-North-E/vis_robot}
TRAIN_REPO=$NORTH_ROOT/tim/workspace/deepdive_kai0
EVAL_REPO=$NORTH_ROOT/workspace/deepdive_kai0/.staging/pi05_r4_eval_north_v1/repo
RELATIVE_PROTOCOL=lmvla/paper_iclr_lmvla/manifests/pi05_r4_replication_protocol_v1.json
LOCAL_PROTOCOL=$REPO/$RELATIVE_PROTOCOL
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_replication_eval_protocol_repair.ok
REMOTE_MARKER=$EVAL_REPO/logs/resource_markers/pi05_r4_replication_eval_protocol_repair.ok

test -s "$LOCAL_PROTOCOL"
expected=$(sha256sum "$LOCAL_PROTOCOL" | awk '{print $1}')

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$TRAIN_REPO" "$EVAL_REPO" "$RELATIVE_PROTOCOL" "$expected" \
  "$REMOTE_MARKER" <<'REMOTE'
set -Eeuo pipefail
train_repo=$1
eval_repo=$2
relative_protocol=$3
expected=$4
marker=$5
source=$train_repo/$relative_protocol
destination=$eval_repo/$relative_protocol
python=$eval_repo/runtime/python/bin/python3.12

test -s "$source"
test -x "$python"
mkdir -p "$(dirname "$destination")"
cp -a "$source" "$destination"

"$python" - "$eval_repo" "$destination" "$expected" <<'PY'
import hashlib
import json
import pathlib
import sys

repo, protocol_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
expected = sys.argv[3]
actual = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
if actual != expected:
    raise ValueError(f"protocol mismatch: {actual} != {expected}")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
rows = [(entry["path"], entry["sha256"]) for entry in protocol["parents"].values()]
rows.extend(protocol["file_sha256"].items())
for relative, digest in rows:
    path = repo / relative
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise ValueError(f"staged source mismatch: {relative}: {observed} != {digest}")
print(f"verified protocol={actual} referenced_files={len(rows)}", flush=True)
PY

mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nprotocol=%s\nsha256=%s\n' \
  "$(date -u +%FT%TZ)" "$destination" "$expected" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_marker=%s\nprotocol_sha256=%s\n' \
  "$(date -u +%FT%TZ)" "$REMOTE_MARKER" "$expected" >"$LOCAL_MARKER"
