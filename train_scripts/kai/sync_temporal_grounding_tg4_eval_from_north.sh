#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
ARM=${TG4_ARM:?TG4_ARM is required}
TRAIN_SEED=${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}
CONDITION=${TG4_CONDITION:-normal}
case "$ARM" in
  clean_base|future_off|auxiliary_only|conditioning_only|parameter_matched_null|full) ;;
  *) echo "unsupported TG4_ARM=$ARM" >&2; exit 2 ;;
esac
case "$TRAIN_SEED" in 1100|1101|1102) ;; *) exit 2 ;; esac
test "$CONDITION" = normal

STAGE=$NORTH_REPO/.staging/temporal_grounding_tg4_eval_v1/repo
RUN_ID=temporal_grounding_tg4_${ARM}_seed${TRAIN_SEED}
REMOTE_MARKER=$STAGE/logs/resource_markers/${RUN_ID}_normal_eval.ok
LOCAL_MARKER=$REPO/logs/resource_markers/${RUN_ID}_normal_eval_materialized.ok
REMOTE_ROOT=$STAGE/lmvla/lawam_local/results/eval_runs/robotwin/${RUN_ID}_normal
LOCAL_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/${RUN_ID}_normal
SCENES=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
SYNC=$REPO/train_scripts/kai/sync_tree_from_north_verified.sh
QUARANTINE="$REPO/logs/resource_scheduler_local/tg4_eval_cross_storage_quarantine/${RUN_ID}_normal_$(date -u +%Y%m%dT%H%M%SZ)_$$"

ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "test -s $(printf %q "$REMOTE_MARKER") && test \"\$(find $(printf %q "$REMOTE_ROOT") -type f -name summary.json | wc -l)\" -eq 24"
if [[ -e "$LOCAL_ROOT" ]]; then
  mkdir -p "$QUARANTINE"
  mv "$LOCAL_ROOT" "$QUARANTINE/result_root"
fi
env SRC="$REMOTE_ROOT" DST="$LOCAL_ROOT" bash "$SYNC"
"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$SCENES" --root "$LOCAL_ROOT"
test "$(find "$LOCAL_ROOT" -type f -name summary.json | wc -l)" -eq 24

if [[ "$ARM" == full ]]; then
  REMOTE_FEATURE=$STAGE/logs/temporal_grounding/tg4/features/$RUN_ID
  LOCAL_FEATURE=$REPO/logs/temporal_grounding/tg4/features/$RUN_ID
  REMOTE_CAPTURE=$STAGE/logs/resource_markers/${RUN_ID}_normal_capture_complete.json
  LOCAL_CAPTURE=$REPO/logs/resource_markers/${RUN_ID}_normal_capture_complete.json
  ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
    "test -s $(printf %q "$REMOTE_CAPTURE") && test -d $(printf %q "$REMOTE_FEATURE")"
  if [[ -e "$LOCAL_FEATURE" ]]; then
    mkdir -p "$QUARANTINE"
    mv "$LOCAL_FEATURE" "$QUARANTINE/feature_root"
  fi
  if [[ -e "$LOCAL_CAPTURE" ]]; then
    mkdir -p "$QUARANTINE"
    mv "$LOCAL_CAPTURE" "$QUARANTINE/capture_marker"
  fi
  env SRC="$REMOTE_FEATURE" DST="$LOCAL_FEATURE" bash "$SYNC"
  incoming=$LOCAL_CAPTURE.incoming.$$
  ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
    "cat $(printf %q "$REMOTE_CAPTURE")" > "$incoming"
  "$REPO/kai0/.venv/bin/python" \
    "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_feature_capture.py" \
    --feature-root "$LOCAL_FEATURE" --scene-manifest "$SCENES" --output "$incoming.verified"
  "$REPO/kai0/.venv/bin/python" - "$incoming" "$incoming.verified" <<'PY'
import json
import pathlib
import sys

remote = json.loads(pathlib.Path(sys.argv[1]).read_text())
local = json.loads(pathlib.Path(sys.argv[2]).read_text())
keys = (
    "complete",
    "scene_manifest_sha256",
    "scenes",
    "query_files",
    "total_bytes",
    "feature_shape",
    "queries_per_scene",
    "tree_sha256",
    "checks",
)
if {key: remote[key] for key in keys} != {key: local[key] for key in keys}:
    raise SystemExit("North feature capture changed during materialization")
PY
  mv "$incoming.verified" "$LOCAL_CAPTURE"
  rm -f "$incoming"
fi

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nrun_id=%s\ncondition=normal\nsource=Robot-North-H20\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$RUN_ID" "$REMOTE_MARKER" > "$LOCAL_MARKER"
