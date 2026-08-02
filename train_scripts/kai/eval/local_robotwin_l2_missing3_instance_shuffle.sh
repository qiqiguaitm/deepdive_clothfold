#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
variant="${ALL6_EVAL_VARIANT:?set ALL6_EVAL_VARIANT}"
run_basename="${ALL6_EVAL_RUN_BASENAME:?set ALL6_EVAL_RUN_BASENAME}"
train_seed="${ALL6_TRAIN_SEED:-2026}"
result_name="${RESULT_NAME:?set RESULT_NAME}"
hint_root=$repo/lmvla/lmwm/data/lawam_within_task_hint_controls
eval_script=$repo/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh
result_root=$repo/lmvla/lawam/results/eval_runs/robotwin/$result_name
mkdir -p "$result_root"

case "$variant" in
  absolute) hint_form=absolute ;;
  residual|combo) hint_form=residual ;;
  *) echo "unsupported instance-shuffle variant: $variant" >&2; exit 2 ;;
esac

tasks=(blocks_ranking_size handover_block stack_blocks_three)
for task in "${tasks[@]}"; do
  expected=0
  for _ in ${SEEDS:-0 1 2 3}; do expected=$((expected + 1)); done
  completed=$(find "$result_root" \
    -type f -path "*/tasks/$task/summary.json" 2>/dev/null | wc -l)
  if [ "$completed" -ge "$expected" ]; then
    echo "instance-shuffle resume skip task=$task cells=$completed/$expected"
    continue
  fi
  hint=$hint_root/${task}_${hint_form}.npy
  test -s "$hint"
  env \
    ALL6_EVAL_VARIANT="$variant" \
    ALL6_EVAL_RUN_BASENAME="$run_basename" \
    ALL6_TRAIN_SEED="$train_seed" \
    ROBOTWIN_TASKS="$task" \
    LMWM_SWAP_HINT="$hint" \
    SEEDS="${SEEDS:-0 1 2 3}" \
    GPUS_PER_SEED="${GPUS_PER_SEED:-1}" \
    LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-2}" \
    NUM_WORKERS="${NUM_WORKERS:-1}" \
    ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}" \
    RESULT_NAME="$result_name" \
    RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-instance-shuffle}-${task}" \
    bash "$eval_script"
done
