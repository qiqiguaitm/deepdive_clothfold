#!/usr/bin/env bash
set -euo pipefail

repo=/vePFS/tim/workspace/deepdive_kai0
variant="${ALL6_EVAL_VARIANT:?set ALL6_EVAL_VARIANT}"
run_basename="${ALL6_EVAL_RUN_BASENAME:?set ALL6_EVAL_RUN_BASENAME}"
control="${L2_CONTROL:?set L2_CONTROL}"
train_seed="${ALL6_TRAIN_SEED:-2026}"
hint_root=$repo/lmvla/lmwm/data/lawam_within_task_hint_controls
swap_root=$repo/lmvla/lmwm/data/swap_hint_probe_robotwin
base_eval=$repo/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh
instance_eval=$repo/train_scripts/kai/eval/local_robotwin_l2_missing3_instance_shuffle.sh
tasks="blocks_ranking_size handover_block stack_blocks_three"

case "$variant" in
  absolute) form=abs ;;
  residual|combo) form=resid ;;
  *) echo "unsupported L2 variant: $variant" >&2; exit 2 ;;
esac

common=(
  ALL6_EVAL_VARIANT="$variant"
  ALL6_EVAL_RUN_BASENAME="$run_basename"
  ALL6_TRAIN_SEED="$train_seed"
  SEEDS="${SEEDS:-0 1 2 3}"
  GPUS_PER_SEED="${GPUS_PER_SEED:-1}"
  LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-4}"
  NUM_WORKERS="${NUM_WORKERS:-1}"
  ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}"
)

case "$control" in
  zero)
    env "${common[@]}" \
      ROBOTWIN_TASKS="$tasks" \
      LMWM_SWAP_HINT_ZERO=1 \
      RESULT_NAME="rt_all6_v2_${variant}_zerohint_seed${train_seed}_missing3_unseen" \
      RUN_TAG_PREFIX="l2-${variant}-zero-missing3-s${train_seed}" \
      bash "$base_eval"
    result_token=zerohint
    ;;
  cross_task)
    env "${common[@]}" \
      LMWM_SWAP_HINT="$swap_root/other_${form}.npy" \
      ROBOTWIN_TASKS="$tasks" \
      RESULT_NAME="rt_all6_v2_${variant}_othertask_seed${train_seed}_missing3_unseen" \
      RUN_TAG_PREFIX="l2-${variant}-other-missing3-s${train_seed}" \
      bash "$base_eval"
    result_token=othertask
    ;;
  instance_shuffle)
    env "${common[@]}" \
      RESULT_NAME="rt_all6_v2_${variant}_instanceshuffle_seed${train_seed}_missing3_unseen" \
      RUN_TAG_PREFIX="l2-${variant}-instance-missing3-s${train_seed}" \
      bash "$instance_eval"
    result_token=instanceshuffle
    ;;
  *) echo "unsupported L2_CONTROL=$control" >&2; exit 2 ;;
esac

count=$(find "$repo/lmvla/lawam/results/eval_runs/robotwin" -type f -name summary.json \
  -path "*/rt_all6_v2_${variant}_${result_token}_seed${train_seed}_missing3_unseen/*" | wc -l)
if [ "${ROBOTWIN_ATTACH_EXISTING:-0}" = 1 ]; then
  echo "L2 attach worker finished variant=$variant control=$control current_cells=$count/12"
  exit 0
fi
test "$count" -eq 12
echo "L2 control complete variant=$variant control=$control cells=$count"
