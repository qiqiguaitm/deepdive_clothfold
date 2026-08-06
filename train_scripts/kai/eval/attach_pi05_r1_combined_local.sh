#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${R1_VERIFY_REPO:-$REPO/logs/frozen_source_overlays/pi05_r1_v1}
SEEDS=${SEEDS:-"2 3"}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_r1_combined/pi05_r1_combined_seed1000/49999}
RESULT_NAME=${RESULT_NAME:-pi05_r1_seed1000_combined}
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME

test -s "$VERIFY_REPO/READY"
test -s "$CKPT/params/_METADATA"
for seed in $SEEDS; do
  scheduler=(
    "$RESULT_ROOT/seed$seed"/*/"local-unseen-a3-seed$seed"/.task_scheduler.json
  )
  if [[ ${#scheduler[@]} -ne 1 || ! -s "${scheduler[0]}" ]]; then
    printf 'seed %s requires exactly one existing scheduler, found %s\n' \
      "$seed" "${#scheduler[@]}" >&2
    exit 2
  fi
done

export PYTHONPATH="$VERIFY_REPO/kai0/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
exec env \
  PI05_EVAL_CONFIG_NAME=pi05_r1_combined_eval \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  PREDICTIVE_ACTION_INTERVENTION=normal \
  RECURRENCE_ACTION_INTERVENTION=normal \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  ROBOTWIN_ATTACH_REQUEUE_FAILED=1 \
  ROBOTWIN_WORKER_INDEX_OFFSET=${ROBOTWIN_WORKER_INDEX_OFFSET:-2000} \
  SEEDS="$SEEDS" \
  LOCAL_GPU_COUNT=2 \
  GPU_INDEX_OFFSET=0 \
  MAX_PARALLEL_SEEDS=2 \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-26400} \
  LOCAL_EVAL_LOG_DIR=${LOCAL_EVAL_LOG_DIR:-$REPO/logs/local_eval/r1_combined_attach_2g} \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"
