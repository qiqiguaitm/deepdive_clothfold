#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/pi05_robotwin_mt1_oracle_seed1000/5000}
RESULT_NAME=${RESULT_NAME:-pi05_mt1_oracle_seed1000_step5000_preflight_v2}
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_mt1_oracle_seed1000_step5000_preflight_v2.ok}
INTERVENTION=${ROBOTWIN_TRANSITION_INTERVENTION:-correct}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_mt_preflight_scene_seeds_v1.json
ARTIFACT=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1

env \
  PI05_EVAL_CONFIG_NAME=${PI05_EVAL_CONFIG_NAME:-pi05_robotwin_mt1_oracle_exact} \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  ROBOTWIN_TRANSITION_ORACLE=1 \
  ROBOTWIN_TRANSITION_INTERVENTION="$INTERVENTION" \
  ROBOTWIN_TRANSITION_PAIRS=$ARTIFACT/pairs.npz \
  ROBOTWIN_TRANSITION_TASK_MAP=$ARTIFACT/eval_task_id.json \
  ROBOTWIN_TRANSITION_EPISODES=/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21/meta/episodes.jsonl \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TASKS=beat_block_hammer \
  ROBOTWIN_TEST_NUM=5 \
  ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS=0 \
  LOCAL_GPU_COUNT=1 \
  GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0} \
  MAX_PARALLEL_SEEDS=1 \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-18600} \
  bash $REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh

python3 $REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py \
  --manifest $MANIFEST \
  --root $REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME \
  --allow-partial

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\ncheckpoint=%s\nintervention=%s\n' \
  "$(date -u +%FT%TZ)" "$CKPT" "$INTERVENTION" > "$MARKER"
