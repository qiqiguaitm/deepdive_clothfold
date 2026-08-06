#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
RESULT_NAME=${RESULT_NAME:?set RESULT_NAME}
INTERVENTION=${INTERVENTION:?set INTERVENTION}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/pi05_robotwin_mt1_oracle_seed1000/49999}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
REMOTE_ROOT=$NORTH_REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json

remote_count=$(ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "find '$REMOTE_ROOT' -name summary.json -type f | wc -l")
test "$remote_count" -ge 24
mkdir -p "$LOCAL_ROOT"
scp -r -P 16370 -o BatchMode=yes \
  "root@124.174.16.237:$REMOTE_ROOT/seed"'*' "$LOCAL_ROOT/"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" --root "$LOCAL_ROOT"
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$LOCAL_ROOT" --expected-cells 24 > "$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"
mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncheckpoint=%s\nintervention=%s\nreport=%s\nsource=Robot-North-H20\n' \
  "$(date -u +%FT%TZ)" "$CKPT" "$INTERVENTION" "$REPORT" > "$MARKER"
