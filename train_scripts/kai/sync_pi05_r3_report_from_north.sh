#!/usr/bin/env bash
set -euo pipefail

CONDITION=${R3_CONDITION:?set R3_CONDITION}
REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
name=pi05_r3_${CONDITION}_screen_v1
report=lmvla/lmwm/docs/$name.json
marker=logs/resource_markers/$name.ok
mkdir -p "$REPO/$(dirname "$report")" "$REPO/$(dirname "$marker")"
temporary_report=$REPO/$report.tmp.$$
temporary_marker=$REPO/$marker.tmp.$$
scp -P "$NORTH_PORT" -q "$NORTH_HOST:$NORTH_REPO/$report" "$temporary_report"
scp -P "$NORTH_PORT" -q "$NORTH_HOST:$NORTH_REPO/$marker" "$temporary_marker"
python3 - "$temporary_report" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d['summary_count']==24 and d['task_count']==6 and d['total_episodes']==240, d
PY
mv "$temporary_report" "$REPO/$report"
mv "$temporary_marker" "$REPO/$marker"
