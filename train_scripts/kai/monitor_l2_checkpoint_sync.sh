#!/usr/bin/env bash
set -euo pipefail

repo=/vePFS/tim/workspace/deepdive_kai0
base=$repo/lmvla/lawam/results/Checkpoints/robotwin
marker_dir=$repo/logs/resource_markers
expected_size=7296436418
mkdir -p "$marker_dir"

declare -A runs=(
  [absolute]=20260730_063314+robotwin_all6_v2_absolute_seed2026
  [residual]=20260730_164555+robotwin_all6_v2_residual_seed2026
)

for variant in absolute residual; do
  run=${runs[$variant]}
  model=$base/$run/final_model/pytorch_model.pt
  while true; do
    size=$(stat -c %s "$model" 2>/dev/null || echo 0)
    if [[ "$size" == "$expected_size" ]] \
      && [[ -s "$base/$run/config.yaml" ]] \
      && [[ -s "$base/$run/config.json" ]] \
      && [[ -s "$base/$run/dataset_statistics.json" ]]; then
      printf 'ready variant=%s run=%s size=%s timestamp=%s\n' \
        "$variant" "$run" "$size" "$(date -u +%FT%TZ)" \
        > "$marker_dir/l2_${variant}_checkpoint_synced.ok"
      break
    fi
    sleep 30
  done
done
