#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
RESULT_NAME=pi05_confirmatory_fixed_scene_preflight_a0s1000_step20000
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
MARKER=$REPO/logs/resource_markers/pi05_confirmatory_fixed_scene_preflight.ok

env \
  PI05_EVAL_CONFIG_NAME=pi05_robotwin_a0_public_exact_bj \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/pi05_robotwin_a0_public_exact_seed1000/20000 \
  RESULT_NAME=$RESULT_NAME \
  ROBOTWIN_TASKS=beat_block_hammer \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS=0 \
  LOCAL_GPU_COUNT=1 \
  MAX_PARALLEL_SEEDS=1 \
  PORT_BASE_OFFSET=17400 \
  bash $REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh

python3 $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest $MANIFEST \
  --root $RESULT_ROOT \
  --allow-partial

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncells=1\nsource_checkpoint=step20000\n' "$(date -u +%FT%TZ)" > "$MARKER"
