#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
SEED=${SEED:?SEED is required}
GPU_INDEX=${GPU_INDEX:?GPU_INDEX is required}
WORKER_INDEX_OFFSET=${WORKER_INDEX_OFFSET:-$((1000 + SEED * 100))}
PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-18200}
RESULT_NAME=${RESULT_NAME:-pi05_rt_a0_public_exact_seed1000}
PI05_EVAL_CONFIG_NAME=${PI05_EVAL_CONFIG_NAME:-${ROBOTWIN_EVAL_CONFIG:-pi05_robotwin_a0_public_exact_bj}}
PI05_ASSET_ID=${PI05_ASSET_ID:-robotwin2.0_absolute_meanstd}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/pi05_robotwin_a0_public_exact_seed1000/49999}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
CKPT_ALIAS=${ROBOTWIN_CKPT_ALIAS:-$(basename "$(dirname "$(dirname "$CKPT")")")}
RUN_GROUP=${ROBOTWIN_RUN_GROUP:-${CKPT_ALIAS}__demo_clean}
RUN_GROUP_DIR=$RESULT_ROOT/seed$SEED/$RUN_GROUP
if [ -n "${ROBOTWIN_ATTACH_RUN_TAG:-}" ]; then
  RUN_DIR=$RUN_GROUP_DIR/$ROBOTWIN_ATTACH_RUN_TAG
else
  shopt -s nullglob
  scheduler_paths=("$RUN_GROUP_DIR"/*/.task_scheduler.json)
  shopt -u nullglob
  [ "${#scheduler_paths[@]}" -eq 1 ] || {
    echo "expected one active scheduler under $RUN_GROUP_DIR, found ${#scheduler_paths[@]}" >&2
    exit 12
  }
  RUN_DIR=$(dirname "${scheduler_paths[0]}")
fi
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
MARKER_NAME=${ATTACH_MARKER_NAME:-pi05_a0_seed1000_eval_attach_seed${SEED}}
MARKER=$REPO/logs/resource_markers/${MARKER_NAME}.ok

test -f "$RUN_DIR/.task_scheduler.json"
test -w "$RUN_DIR"
test -f "$CKPT/params/_METADATA"

env \
  PI05_EVAL_CONFIG_NAME=$PI05_EVAL_CONFIG_NAME \
  PI05_ASSET_ID=$PI05_ASSET_ID \
  CKPT=$CKPT \
  RESULT_NAME=$RESULT_NAME \
  ROBOTWIN_EPISODE_SEED_MANIFEST=$MANIFEST \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  ROBOTWIN_ATTACH_SCHEDULER=1 \
  ROBOTWIN_RESUME_RUN_DIR=$RUN_DIR \
  ROBOTWIN_WORKER_INDEX_OFFSET=$WORKER_INDEX_OFFSET \
  SEEDS=$SEED \
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
printf 'completed=%s\nseed=%s\nworker_index_offset=%s\n' \
  "$(date -u +%FT%TZ)" "$SEED" "$WORKER_INDEX_OFFSET" > "$MARKER"
