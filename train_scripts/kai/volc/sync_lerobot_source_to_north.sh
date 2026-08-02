#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/vePFS/tim
NORTH_HOST=root@124.174.16.237
NORTH_PORT=16370
NORTH_ROOT=/vePFS-North-E/vis_robot/tim
STATUS_DIR=/vePFS/tim/workspace/deepdive_kai0/logs/sync_lerobot_source_north
MAIN_STATUS_DIR=/vePFS/tim/workspace/deepdive_kai0/logs/sync_public_pi05_north
LOCAL_ARCHIVE=/vePFS/tim/tmp/lerobot_source_north_20260731.tar
REMOTE_ARCHIVE=$NORTH_ROOT/.transfer_cache/lerobot_source_north_20260731.tar
TOS_URI=tos://transfer-shanghai/temp/deepdive_kai0/lerobot_source_north_20260731.tar

mkdir -p "$STATUS_DIR" "$(dirname "$LOCAL_ARCHIVE")"
printf 'RUNNING start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"
trap 'rc=$?; printf "FAILED rc=%s end=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"' ERR

cd "$SOURCE_ROOT"
if [ ! -s "$LOCAL_ARCHIVE" ]; then
  tar -cf "$LOCAL_ARCHIVE" workspace/lerobot-main/src
fi
tosutil cp "$LOCAL_ARCHIVE" "$TOS_URI" -p=8 -vchecksum

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p '$NORTH_ROOT' '$(dirname "$REMOTE_ARCHIVE")' && \
   tosutil cp '$TOS_URI' '$REMOTE_ARCHIVE' -f -p=8 -vchecksum && \
   tar -C '$NORTH_ROOT' -xf '$REMOTE_ARCHIVE' && \
   unlink '$REMOTE_ARCHIVE'"

# The larger model archive is transferred independently. Wait until its
# extraction and import check finish so the archived .pth cannot overwrite us.
deadline=$((SECONDS + 7200))
while [ ! -f "$MAIN_STATUS_DIR/complete" ]; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for the public PI0.5 bundle" >&2
    exit 1
  fi
  sleep 1
done
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- "$NORTH_ROOT" <<'REMOTE'
set -euo pipefail
root=$1
pth=$root/workspace/lerobot-pi05-server-venv/lib/python3.12/site-packages/__editable__.lerobot-0.6.1.pth
test -d "$(dirname "$pth")"
printf '%s\n' "$root/workspace/lerobot-main/src" > "$pth"
REMOTE

touch "$STATUS_DIR/complete"
printf 'FINISHED rc=0 end=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"
unlink "$LOCAL_ARCHIVE"
