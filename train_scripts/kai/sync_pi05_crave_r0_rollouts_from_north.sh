#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
RESULT_NAME=${RESULT_NAME:-pi05_crave_r0_public_rollouts_v1}
REMOTE_ROOT=$NORTH_REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/pi05_crave_r0_rollout_collection.ok
MARKER=$REPO/logs/resource_markers/pi05_crave_r0_rollout_collection.ok
AUDIT_JSON=$REPO/logs/crave_r0/rollouts/artifact_audit.json
AUDIT_MD=$REPO/logs/crave_r0/rollouts/artifact_audit.md

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "test -s '$REMOTE_MARKER' && test \"\$(find '$REMOTE_ROOT' -name summary.json -type f | wc -l)\" -eq 12 && test \"\$(find '$REMOTE_ROOT' -name 'episode*.mp4' -type f | wc -l)\" -eq 120"

mkdir -p "$LOCAL_ROOT" "$(dirname "$MARKER")" "$(dirname "$AUDIT_JSON")"
rsync -a --partial -e "ssh -p $NORTH_PORT -o BatchMode=yes" \
  "$NORTH_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"

python "$REPO/lmvla/lmwm/scripts/audit_robotwin_rollout_artifacts.py" \
  --root "$LOCAL_ROOT" --json-out "$AUDIT_JSON" --markdown-out "$AUDIT_MD"
python - "$AUDIT_JSON" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["valid_summary_count"] == 12, audit
assert audit["trajectory_file_count"] == 120, audit
assert audit["successes"] > 0 and audit["failures"] > 0, audit
assert audit["supports_crave_rollout_metrics"], audit
PY

printf 'completed=%s\nsource=north\nroot=%s\naudit=%s\n' \
  "$(date -u +%FT%TZ)" "$LOCAL_ROOT" "$AUDIT_JSON" >"$MARKER"
