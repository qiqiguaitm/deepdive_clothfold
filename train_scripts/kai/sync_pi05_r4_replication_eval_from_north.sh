#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_ROOT=${NORTH_ROOT:-/vePFS-North-E/vis_robot}
EVAL_REPO=$NORTH_ROOT/workspace/deepdive_kai0/.staging/pi05_r4_eval_north_v1/repo
ARM=${R4_ARM:?set R4_ARM}
SEED=${R4_SEED:?set R4_SEED}
case "$ARM" in ordinary|terminal_outcome|outcome_free_crave) ;; *) exit 2 ;; esac
case "$SEED" in 1001|1002) ;; *) exit 2 ;; esac

RESULT_NAME=pi05_r4_${ARM}_seed${SEED}
REMOTE_ROOT=$EVAL_REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REMOTE_REPORT=$EVAL_REPO/lmvla/lmwm/docs/$RESULT_NAME.json
REMOTE_MARKER=$EVAL_REPO/logs/resource_markers/$RESULT_NAME.ok
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
LOCAL_REPORT=$REPO/lmvla/lmwm/docs/$RESULT_NAME.json
LOCAL_MARKER=$REPO/logs/resource_markers/$RESULT_NAME.ok
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "test -s $(printf %q "$REMOTE_MARKER") && test -s $(printf %q "$REMOTE_REPORT") && test \"\$(find $(printf %q "$REMOTE_ROOT") -name summary.json -type f | wc -l)\" -eq 24"
env SRC="$REMOTE_ROOT" DST="$LOCAL_ROOT" \
  bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"

temporary_report=$LOCAL_REPORT.tmp.$$
temporary_marker=$LOCAL_MARKER.tmp.$$
trap 'rm -f "$temporary_report" "$temporary_marker"' EXIT
mkdir -p "$(dirname "$LOCAL_REPORT")" "$(dirname "$LOCAL_MARKER")"
scp -P "$NORTH_PORT" -q "$NORTH_HOST:$REMOTE_REPORT" "$temporary_report"
scp -P "$NORTH_PORT" -q "$NORTH_HOST:$REMOTE_MARKER" "$temporary_marker"
python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$LOCAL_ROOT"
python3 - "$temporary_report" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["summary_count"] == 24
assert report["task_count"] == 6
assert report["total_episodes"] == 1200
PY
mv "$temporary_report" "$LOCAL_REPORT"
mv "$temporary_marker" "$LOCAL_MARKER"
