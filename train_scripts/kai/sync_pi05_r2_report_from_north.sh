#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
CONDITION=${R2_CONDITION:?set R2_CONDITION}
RESULT=pi05_r2_${CONDITION}_screen_v1
REMOTE_REPORT=$NORTH_REPO/lmvla/lmwm/docs/$RESULT.json
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/$RESULT.ok
LOCAL_REPORT=$REPO/lmvla/lmwm/docs/$RESULT.json
LOCAL_MARKER=$REPO/logs/resource_markers/$RESULT.ok

mkdir -p "$(dirname "$LOCAL_REPORT")" "$(dirname "$LOCAL_MARKER")"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" "test -s '$REMOTE_REPORT' && test -s '$REMOTE_MARKER'"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" "cat '$REMOTE_REPORT'" >"$LOCAL_REPORT.tmp"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" "cat '$REMOTE_MARKER'" >"$LOCAL_MARKER.tmp"
python - "$LOCAL_REPORT.tmp" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d['summary_count']==24 and d['task_count']==6 and d['total_episodes']==240, d
assert len(d['efficiency_cells'])==24 and d['total_model_queries'] > 0, d
PY
mv "$LOCAL_REPORT.tmp" "$LOCAL_REPORT"
mv "$LOCAL_MARKER.tmp" "$LOCAL_MARKER"
