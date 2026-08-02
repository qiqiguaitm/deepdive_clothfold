#!/usr/bin/env bash
set -euo pipefail

# Execute an immutable copy because this orchestrator remains open across
# multiple long task evaluations on the shared filesystem.
if [[ "${A2_INSTANCE_SCRIPT_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-a2-instance-shuffle.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  A2_INSTANCE_SCRIPT_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=/vePFS/tim/workspace/deepdive_kai0
PY=$REPO/kai0/.venv/bin/python
EXTRACT=$REPO/train_scripts/kai/analysis/extract_pi05_a2_instance_features.py
EVAL=$REPO/train_scripts/kai/eval/local_robotwin_a2_official_2gpu.sh
PAIRS=$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz
MODEL=$REPO/lmvla/lmwm/data/hf_so400m
OUT_ROOT=$REPO/lmvla/lmwm/data/pi05_hint/a2_within_task_other_episode
RESULT_NAME=pi05_rt_a2_instance_shuffle_causal

declare -a SPECS=(
  "beat_block_hammer:0:550:83"
  "blocks_ranking_rgb:3:1100:102"
  "stack_blocks_two:1:24750:172"
)

if [[ ! -s "$OUT_ROOT/beat_block_hammer/manifest.json" \
   || ! -s "$OUT_ROOT/blocks_ranking_rgb/manifest.json" \
   || ! -s "$OUT_ROOT/stack_blocks_two/manifest.json" ]]; then
  args=()
  for spec in "${SPECS[@]}"; do
    args+=(--spec "$spec")
  done
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" "$EXTRACT" \
    --repo "$REPO" \
    --pair-artifact "$PAIRS" \
    --model-dir "$MODEL" \
    --output-root "$OUT_ROOT" \
    "${args[@]}"
fi

for spec in "${SPECS[@]}"; do
  IFS=: read -r task _pair_task _episode _frame <<<"$spec"
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
  env \
    ROBOTWIN_HINT_INTERVENTION=override \
    ROBOTWIN_HINT_OVERRIDE_PATH="$OUT_ROOT/$task/feature.npy" \
    ROBOTWIN_TASKS="$task" \
    ROBOTWIN_TEST_NUM=50 \
    SEEDS="0 1 2 3" \
    LOCAL_GPU_COUNT=2 \
    SEED_STAGGER_SECONDS=20 \
    EVAL_NAME="$RESULT_NAME" \
    RUN_TAG_PREFIX="local-a2-instance-$task" \
    bash "$EVAL"
done
