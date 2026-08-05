#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
ARM=${R4_ARM:?set R4_ARM}
case "$ARM" in
  terminal_outcome|outcome_free_crave) ;;
  *) echo "unsupported North R4 arm: $ARM" >&2; exit 2 ;;
esac

STAGE=$NORTH_REPO/.staging/pi05_r4_eval_north_v1/repo
RESULT_NAME=pi05_r4_${ARM}_seed1000
REMOTE_MARKER=$STAGE/logs/resource_markers/${RESULT_NAME}.ok
LOCAL_MARKER=$REPO/logs/resource_markers/${RESULT_NAME}.ok
REMOTE_ROOT=$STAGE/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json

ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "test -s $(printf %q "$REMOTE_MARKER") && test \"\$(find $(printf %q "$REMOTE_ROOT") -name summary.json -type f | wc -l)\" -eq 24"
env SRC="$REMOTE_ROOT" DST="$LOCAL_ROOT" \
  bash "$REPO/train_scripts/kai/sync_tree_from_north_verified.sh"
python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$LOCAL_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$LOCAL_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\narm=%s\nreport=%s\nsource=Robot-North-H20\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$ARM" "$REPORT" "$REMOTE_MARKER" >"$LOCAL_MARKER"
