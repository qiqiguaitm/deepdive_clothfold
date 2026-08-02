#!/usr/bin/env bash
set -euo pipefail

repo=/vePFS/tim/workspace/deepdive_kai0
remote=root@124.174.16.237
remote_base=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/results/Checkpoints/robotwin
local_base=$repo/lmvla/lawam/results/Checkpoints/robotwin

runs=(
  20260730_063314+robotwin_all6_v2_absolute_seed2026
  20260730_164555+robotwin_all6_v2_residual_seed2026
)
files=(config.yaml config.json dataset_statistics.json final_model/pytorch_model.pt)

for run in "${runs[@]}"; do
  echo "[$(date -u +%FT%TZ)] syncing $run"
  install -d "$local_base/$run/final_model"
  for file in "${files[@]}"; do
    scp -P 16370 -o BatchMode=yes \
      "$remote:$remote_base/$run/$file" "$local_base/$run/$file"
  done
  test "$(stat -c %s "$local_base/$run/final_model/pytorch_model.pt")" -eq 7296436418
  echo "[$(date -u +%FT%TZ)] completed $run"
done
