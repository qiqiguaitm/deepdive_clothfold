#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
STAGE=$NORTH_REPO/.staging/pi05_r4_eval_north_v1/repo
VERIFIER=$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
AMENDMENT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_manifest_set_verifier_amendment_v1.json
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_north_manifest_verifier.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/pi05_r4_north_manifest_verifier.ok
REMOTE_VERIFIER=$STAGE/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py

test -s "$VERIFIER"
test -s "$AMENDMENT"
expected=$(python3 -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["file_sha256_override"]["lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py"])
' "$AMENDMENT")
actual=$(sha256sum "$VERIFIER" | cut -d' ' -f1)
test "$actual" = "$expected"

scp -P "$NORTH_PORT" -q "$VERIFIER" "$NORTH_HOST:$REMOTE_VERIFIER.tmp"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$REMOTE_VERIFIER.tmp" "$REMOTE_VERIFIER" "$REMOTE_MARKER" "$expected" <<'REMOTE'
set -Eeuo pipefail
source_path=$1
destination=$2
marker=$3
expected=$4

echo "$expected  $source_path" | sha256sum -c -
chmod 0644 "$source_path"
mv -f "$source_path" "$destination"
echo "$expected  $destination" | sha256sum -c -
python3 -m py_compile "$destination"
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nverifier=%s\nsha256=%s\nmode=set-identity\n' \
  "$(date -u +%FT%TZ)" "$destination" "$expected" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\nsha256=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" "$expected" >"$LOCAL_MARKER"
