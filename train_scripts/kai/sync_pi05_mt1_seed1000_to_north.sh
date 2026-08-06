#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
SEED=${SEED:-1000}
case "$SEED" in 1000|1001|1002) ;; *) echo "unsupported MT1 seed: $SEED" >&2; exit 2 ;; esac
SRC=$REPO/kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/pi05_robotwin_mt1_oracle_seed${SEED}/49999
DST=$NORTH_REPO/kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/pi05_robotwin_mt1_oracle_seed${SEED}/49999
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_mt1_seed${SEED}_north_eval_checkpoint.ok}

test -f "$SRC/params/_METADATA"
started_at=$(date -u +%FT%TZ)
started_epoch=$(date +%s)
transport=${SYNC_TRANSPORT:-auto}
case "$transport" in auto|tos|ssh) ;; *) echo "unsupported sync transport: $transport" >&2; exit 2 ;; esac
if [ "$transport" != ssh ]; then
  if SYNC_EVAL_ONLY=1 SRC="$SRC" DST="$DST" \
    bash "$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh"; then
    transport=tos
  elif [ "$transport" = tos ]; then
    exit 1
  else
    echo "TOS sync failed; falling back to verified SSH stream" >&2
    echo "phase=ssh-fallback"
    transport=ssh
  fi
fi
if [ "$transport" = ssh ]; then
  echo "phase=ssh-stream"
  SYNC_EVAL_ONLY=1 SRC="$SRC" DST="$DST" \
    bash "$REPO/train_scripts/kai/sync_tree_to_north_verified.sh"
fi

local_hash=$(sha256sum "$SRC/params/_METADATA" | awk '{print $1}')
remote_hash=$(ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "sha256sum '$DST/params/_METADATA'" | awk '{print $1}')
test "$local_hash" = "$remote_hash"
mkdir -p "$(dirname "$MARKER")"
validated_at=$(date -u +%FT%TZ)
elapsed_seconds=$(($(date +%s) - started_epoch))
marker_tmp="${MARKER}.tmp.$$"
trap 'rm -f -- "$marker_tmp"' EXIT
printf 'validated=%s\nstarted=%s\nelapsed_seconds=%s\nsource=%s\ndestination=%s\nmetadata_sha256=%s\ntransport=%s\n' \
  "$validated_at" "$started_at" "$elapsed_seconds" "$SRC" "$DST" \
  "$local_hash" "$transport" > "$marker_tmp"
mv "$marker_tmp" "$MARKER"
trap - EXIT
