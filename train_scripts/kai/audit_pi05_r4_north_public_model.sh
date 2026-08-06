#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_ROOT=${NORTH_ROOT:-/vePFS-North-E/vis_robot}
LOCAL_MODEL=${LOCAL_MODEL:-/vePFS/tim/hf_models/SidneyXie_pi05_robotwin}
REMOTE_MODEL=${REMOTE_MODEL:-$NORTH_ROOT/tim/hf_models/SidneyXie_pi05_robotwin}
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_replication_north_public_model.ok
REMOTE_MARKER=$NORTH_ROOT/tim/workspace/deepdive_kai0/logs/resource_markers/pi05_r4_replication_north_public_model.ok
AUDIT_DIR=$REPO/logs/r4/north_public_model_audit

test -s "$REPO/logs/resource_markers/pi05_r4_replication_north_stage.ok"
test -d "$LOCAL_MODEL"
mkdir -p "$AUDIT_DIR" "$(dirname "$LOCAL_MARKER")"
local_manifest=$AUDIT_DIR/local.sha256
remote_manifest=$AUDIT_DIR/remote.sha256

(
  cd "$LOCAL_MODEL"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$local_manifest.tmp"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$REMOTE_MODEL" <<'REMOTE' >"$remote_manifest.tmp"
set -Eeuo pipefail
model=$1
test -d "$model"
cd "$model"
find . -type f -print0 | sort -z | xargs -0 sha256sum
REMOTE

cmp "$local_manifest.tmp" "$remote_manifest.tmp"
mv "$local_manifest.tmp" "$local_manifest"
mv "$remote_manifest.tmp" "$remote_manifest"
aggregate=$(sha256sum "$local_manifest" | awk '{print $1}')
files=$(wc -l <"$local_manifest")
bytes=$(du -sb "$LOCAL_MODEL" | awk '{print $1}')
validated=$(date -u +%FT%TZ)

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p $(printf %q "$(dirname "$REMOTE_MARKER")") && printf '%s\n' $(printf %q "validated=$validated") $(printf %q "aggregate_sha256=$aggregate") $(printf %q "files=$files") $(printf %q "bytes=$bytes") > $(printf %q "$REMOTE_MARKER")"
printf 'validated=%s\naggregate_sha256=%s\nfiles=%s\nbytes=%s\nremote_model=%s\nremote_marker=%s\n' \
  "$validated" "$aggregate" "$files" "$bytes" "$REMOTE_MODEL" "$REMOTE_MARKER" \
  >"$LOCAL_MARKER"
printf 'verified public model files=%s bytes=%s aggregate_sha256=%s\n' \
  "$files" "$bytes" "$aggregate"
