#!/usr/bin/env bash
set -euo pipefail

repo=/vePFS/tim/workspace/deepdive_kai0
result_name="${RESULT_NAME:?set RESULT_NAME}"
requested_tasks="${ROBOTWIN_TASKS:?set ROBOTWIN_TASKS}"
seeds="${SEEDS:-0 1 2 3}"
result_root=$repo/lmvla/lawam/results/eval_runs/robotwin/$result_name
mkdir -p "$result_root"
expected=0
for _ in $seeds; do expected=$((expected + 1)); done

remaining=()
for task in $requested_tasks; do
  count=$(find "$result_root" \
    -type f -path "*/tasks/$task/summary.json" 2>/dev/null | wc -l)
  if [ "$count" -ge "$expected" ]; then
    echo "resume skip task=$task cells=$count/$expected"
  else
    remaining+=("$task")
  fi
done

if [ "${#remaining[@]}" -eq 0 ]; then
  echo "resume complete result=$result_name"
  exit 0
fi

export ROBOTWIN_TASKS="${remaining[*]}"
exec bash "$repo/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh"
