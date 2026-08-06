#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
ROBOTWIN_PROBE_TASK=${ROBOTWIN_PROBE_TASK:-stack_blocks_three}
RESULT_NAME=${RESULT_NAME:-pi05_a0_final_stack3_frozen_seed0_probe}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_a0_final_stack3_probe.ok}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-1}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-17700}

env \
  PI05_EVAL_CONFIG_NAME=pi05_robotwin_a0_public_exact_bj \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/pi05_robotwin_a0_public_exact_seed1000/49999 \
  RESULT_NAME=$RESULT_NAME \
  ROBOTWIN_TASKS=$ROBOTWIN_PROBE_TASK \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS=0 \
  LOCAL_GPU_COUNT=1 \
  GPU_INDEX_OFFSET=$GPU_INDEX_OFFSET \
  MAX_PARALLEL_SEEDS=1 \
  PORT_BASE_OFFSET=$PORT_BASE_OFFSET \
  bash $REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh

python3 $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest $MANIFEST \
  --root $RESULT_ROOT \
  --allow-partial

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncells=1\nsource_checkpoint=step49999\ntask=%s\n' \
  "$(date -u +%FT%TZ)" "$ROBOTWIN_PROBE_TASK" > "$MARKER"
