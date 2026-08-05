#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
CONDITION=${PREDICTIVE_P1_CONDITION:?set PREDICTIVE_P1_CONDITION}
RESULT_NAME=${RESULT_NAME:-pi05_predictive_adapter_p1_seed1000_${CONDITION}}
REMOTE_MARKER=${REMOTE_MARKER:-$NORTH_REPO/logs/resource_markers/${RESULT_NAME}.ok}
LOCAL_MARKER=${LOCAL_MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
REMOTE_ROOT=$NORTH_REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json

ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "test -s $(printf %q "$REMOTE_MARKER") && test \"\$(find $(printf %q "$REMOTE_ROOT") -name summary.json -type f | wc -l)\" -ge 24"

env SRC="$REMOTE_ROOT" DST="$LOCAL_ROOT" \
  bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"
python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$LOCAL_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$LOCAL_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\ncondition=%s\nreport=%s\nsource=Robot-North-H20\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$CONDITION" "$REPORT" "$REMOTE_MARKER" >"$LOCAL_MARKER"
