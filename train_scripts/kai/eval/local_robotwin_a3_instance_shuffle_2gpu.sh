#!/usr/bin/env bash
set -euo pipefail

# Long evaluations read shell files incrementally. Run an immutable copy so a
# later edit to this shared entrypoint cannot truncate an in-flight task.
if [[ "${A3_INSTANCE_SCRIPT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-a3-instance-shuffle.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  A3_INSTANCE_SCRIPT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=/vePFS/tim/workspace/deepdive_kai0
PY=$REPO/kai0/.venv/bin/python
CKPT=$REPO/kai0/checkpoints/pi05_robotwin_a3_live_residual_prefix_official_east/pi05_robotwin_a3_live_residual_prefix_official/19999
EXTRACT=$REPO/train_scripts/kai/analysis/extract_pi05_a3_other_task_feature.py
EVAL=$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh
PAIRS=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz
OUT_ROOT=$REPO/lmvla/lmwm/data/pi05_hint/a3_within_task_other_episode
RESULT_NAME=pi05_rt_a3_instance_shuffle_causal

declare -a SPECS=(
  "beat_block_hammer:0:550:83"
  "blocks_ranking_rgb:3:1100:102"
  "stack_blocks_two:1:24750:172"
)

for spec in "${SPECS[@]}"; do
  IFS=: read -r task pair_task episode frame <<<"$spec"
  completed_seeds=0
  for seed in 0 1 2 3; do
    if find "$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME/seed$seed" \
      -path "*/tasks/$task/summary.json" -print -quit 2>/dev/null | grep -q .; then
      completed_seeds=$((completed_seeds + 1))
    fi
  done
  if (( completed_seeds == 4 )); then
    echo "skip completed task=$task seeds=$completed_seeds"
    continue
  fi
  chunk=$(printf '%03d' $((episode / 1000)))
  episode_padded=$(printf '%06d' "$episode")
  frame_cache=$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256/chunk-$chunk/observation.images.cam_high/episode_$episode_padded.npz
  out_dir=$OUT_ROOT/$task
  mkdir -p "$out_dir"

  if [[ ! -s "$out_dir/feature.npy" || ! -s "$out_dir/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" "$EXTRACT" \
      --repo "$REPO" \
      --checkpoint "$CKPT" \
      --frame-cache "$frame_cache" \
      --frame "$frame" \
      --pair-artifact "$PAIRS" \
      --pair-task "$pair_task" \
      --episode "$episode" \
      --output "$out_dir/feature.npy" \
      --manifest "$out_dir/manifest.json"
  fi

  env \
    LMWM_LIVE_HINT_INTERVENTION=other-task \
    LMWM_LIVE_OTHER_TASK_FEATURE_PATH="$out_dir/feature.npy" \
    ROBOTWIN_TASKS="$task" \
    ROBOTWIN_TEST_NUM=50 \
    SEEDS="0 1 2 3" \
    GPUS_PER_SEED=1 \
    LOCAL_GPU_COUNT=2 \
    MAX_PARALLEL_SEEDS="${INSTANCE_MAX_PARALLEL_SEEDS:-4}" \
    SEED_STAGGER_SECONDS=20 \
    RESULT_NAME="$RESULT_NAME" \
    bash "$EVAL"
done
