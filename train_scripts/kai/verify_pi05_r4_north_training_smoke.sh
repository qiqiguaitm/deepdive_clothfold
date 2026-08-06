#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/tim/workspace/deepdive_kai0}
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_north_training_smoke.ok
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/pi05_r4_north_training_smoke.ok

result=$(ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$NORTH_REPO" <<'REMOTE'
set -Eeuo pipefail
repo=$1
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  log=$(find "$repo/logs/r4/training" -maxdepth 1 -type f \
    -name 'ordinary-seed1001_*.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -n ${log:-} ]]; then
    step=$(python3 - "$log" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
steps = [int(value) for value in re.findall(r"\bStep (\d+):", text)]
print(max(steps, default=0))
PY
)
    if (( step >= 100 )); then
      printf '%s\t%s\n' "$step" "$log"
      exit 0
    fi
  fi
  sleep 15
done
echo "North R4 smoke did not reach step 100 within 30 minutes" >&2
exit 1
REMOTE
)

IFS=$'\t' read -r step remote_log <<<"$result"
validated=$(date -u +%FT%TZ)
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p $(printf %q "$(dirname "$REMOTE_MARKER")") && printf '%s\n' $(printf %q "validated=$validated") $(printf %q "step=$step") $(printf %q "log=$remote_log") > $(printf %q "$REMOTE_MARKER")"
mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nstep=%s\nremote_log=%s\nremote_marker=%s\n' \
  "$validated" "$step" "$remote_log" "$REMOTE_MARKER" >"$LOCAL_MARKER"
