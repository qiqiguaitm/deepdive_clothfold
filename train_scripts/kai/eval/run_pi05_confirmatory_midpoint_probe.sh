#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
ARM=${ARM:?ARM is required}
GPU_INDEX=${GPU_INDEX:?GPU_INDEX is required}
TASK_NAME=${TASK_NAME:-stack_blocks_three}
STEP=${STEP:-20000}
TEST_NUM=${TEST_NUM:-50}
INTERVENTION=${INTERVENTION:-correct}
RESULT_SUFFIX=${RESULT_SUFFIX:-}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json

case "$ARM" in
  a2_abs)
    CONFIG=pi05_robotwin_a2_prefix_official_eval_bj
    EXTRA_CONFIG=$REPO/train_scripts/kai/volc/config_overrides/pi05_a2_abs_confirmatory_eval.json
    CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a2_abs_confirmatory/pi05_robotwin_a2_abs_seed1000/$STEP
    HINT_ENV=(
      ROBOTWIN_HINT_ENCODER=so400m
      OPENPI_SERVER_HINT_ENCODER=so400m
      EVAL_HINT_RESIDUAL=0
      ROBOTWIN_HINT_INTERVENTION=$INTERVENTION
    )
    PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-23200}
    ;;
  a3_live)
    CONFIG=pi05_robotwin_a3_live_residual_prefix_official_eval
    EXTRA_CONFIG=$REPO/train_scripts/kai/volc/config_overrides/pi05_a3_live_confirmatory_eval.json
    CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a3_live_confirmatory/pi05_robotwin_a3_live_seed1000/$STEP
    HINT_ENV=(LMWM_LIVE_HINT_INTERVENTION=$INTERVENTION)
    PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-23300}
    ;;
  *)
    echo "unsupported ARM: $ARM" >&2
    exit 2
    ;;
esac

RESULT_NAME=pi05_${ARM}_seed1000_step${STEP}_${TASK_NAME}_probe${RESULT_SUFFIX}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MARKER=$REPO/logs/resource_markers/${RESULT_NAME}.ok

test -f "$CKPT/params/_METADATA"
test -f "$CKPT/assets/robotwin2.0_absolute_meanstd/norm_stats.json"

env \
  PI05_EVAL_CONFIG_NAME=$CONFIG \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  OPENPI_EXTRA_CONFIG=$EXTRA_CONFIG \
  "${HINT_ENV[@]}" \
  CKPT=$CKPT \
  RESULT_NAME=$RESULT_NAME \
  ROBOTWIN_TASKS=$TASK_NAME \
  ROBOTWIN_TEST_NUM=$TEST_NUM \
  ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS=0 \
  LOCAL_GPU_COUNT=1 \
  GPU_INDEX_OFFSET=$GPU_INDEX \
  MAX_PARALLEL_SEEDS=1 \
  PORT_BASE_OFFSET=$PORT_BASE_OFFSET \
  bash $REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh

python3 $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest $MANIFEST \
  --root $RESULT_ROOT \
  --allow-partial

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\narm=%s\nstep=%s\ntask=%s\nintervention=%s\n' \
  "$(date -u +%FT%TZ)" "$ARM" "$STEP" "$TASK_NAME" "$INTERVENTION" > "$MARKER"
