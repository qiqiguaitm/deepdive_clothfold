#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
CONDITION=${PREDICTIVE_P1_CONDITION:?set PREDICTIVE_P1_CONDITION}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json

case "$CONDITION" in
  a0)
    CONFIG=pi05_robotwin_a0_public_exact_bj
    INTERVENTION=normal
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999
    ;;
  normal|zero_gate|shuffled|masked)
    CONFIG=pi05_predictive_adapter_p1_eval
    INTERVENTION=$CONDITION
    DEFAULT_CKPT=$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000/49999
    ;;
  *)
    echo "unsupported P1 condition: $CONDITION" >&2
    exit 2
    ;;
esac

CKPT=${CKPT:-$DEFAULT_CKPT}
RESULT_NAME=${RESULT_NAME:-pi05_predictive_adapter_p1_seed1000_${CONDITION}}
MARKER=${MARKER:-$REPO/logs/resource_markers/${RESULT_NAME}.ok}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json

test -f "$CKPT/params/_METADATA"
test -f "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json"
test -f "$MANIFEST"

env \
  PI05_EVAL_CONFIG_NAME="$CONFIG" \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  PREDICTIVE_ACTION_INTERVENTION="$INTERVENTION" \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS="0 1 2 3" \
  LOCAL_GPU_COUNT=${LOCAL_GPU_COUNT:-4} \
  GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0} \
  MAX_PARALLEL_SEEDS=${MAX_PARALLEL_SEEDS:-4} \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-21800} \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" \
  --root "$RESULT_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncondition=%s\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$CONDITION" "$CKPT" "$REPORT" >"$MARKER"
