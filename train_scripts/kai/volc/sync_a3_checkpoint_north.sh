#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
SRC=$REPO/kai0/checkpoints/pi05_robotwin_a3_live_residual_prefix_official_east/pi05_robotwin_a3_live_residual_prefix_official/19999
DST=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0/checkpoints/pi05_robotwin_a3_live_residual_prefix_official_east/pi05_robotwin_a3_live_residual_prefix_official/19999
STATUS=$REPO/logs/a3_north_sync.status
MARKER=$REPO/logs/resource_markers/a3_north_sync.ok
LOG=$REPO/logs/a3_north_sync.log

mkdir -p "$(dirname "$MARKER")"
exec >> "$LOG" 2>&1
trap 'rc=$?; echo "FINISHED rc=$rc end=$(date -u +%FT%TZ)" > "$STATUS"' EXIT
if [[ -f "$MARKER" ]]; then
  echo "ALREADY_COMPLETE marker=$MARKER" > "$STATUS"
  exit 0
fi

echo "RUNNING start=$(date -u +%FT%TZ)" > "$STATUS"

transfer_partition() {
  local partition=$1
  {
    cd "$SRC"
    find params assets -type f -print
    printf '%s\n' _CHECKPOINT_METADATA
  } | awk -v partition="$partition" 'NR % 4 == partition' | \
    tar -C "$SRC" -cf - -T - | \
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -p 16370 root@124.174.16.237 \
      "mkdir -p '$DST' && tar -C '$DST' -xf -"
}

pids=()
for index in 0 1 2 3; do
  transfer_partition "$index" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

ssh -o BatchMode=yes -p 16370 root@124.174.16.237 \
  "test -f '$DST/params/_METADATA' && test -f '$DST/assets/robotwin2.0/norm_stats.json'"

printf 'completed=%s\nsource=%s\ndestination=%s\n' \
  "$(date -u +%FT%TZ)" "$SRC" "$DST" > "$MARKER"
