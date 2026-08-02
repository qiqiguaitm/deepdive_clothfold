#!/usr/bin/env bash
set -euo pipefail

LOCAL=/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/results/Checkpoints/robotwin/20260730_152020+robotwin_all6_v2_combo_seed2026
REMOTE=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/results/Checkpoints/robotwin/20260730_152020+robotwin_all6_v2_combo_seed2026
SSH=(ssh -p 16370 -o BatchMode=yes root@124.174.16.237)
SCP=(scp -P 16370 -o BatchMode=yes)

"${SSH[@]}" "mkdir -p '$REMOTE/final_model'"
"${SCP[@]}" "$LOCAL/config.yaml" "root@124.174.16.237:$REMOTE/config.yaml"
"${SCP[@]}" "$LOCAL/config.json" "root@124.174.16.237:$REMOTE/config.json"
"${SCP[@]}" "$LOCAL/dataset_statistics.json" "root@124.174.16.237:$REMOTE/dataset_statistics.json"
"${SCP[@]}" "$LOCAL/final_model/pytorch_model.pt" \
  "root@124.174.16.237:$REMOTE/final_model/pytorch_model.pt"
"${SSH[@]}" "sha256sum '$REMOTE/final_model/pytorch_model.pt'; touch '$REMOTE/SYNC_COMPLETE'"
