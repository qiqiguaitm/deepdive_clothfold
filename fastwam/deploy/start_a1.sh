#!/usr/bin/env bash
# FastWAM A1 deployment preset; all extra arguments are forwarded.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export KAI0_GRIPPER_DEPLOY_REMAP=0

exec "$REPO_ROOT/fastwam/deploy/start_autonomy_isolated.sh" \
  --server-gpu 2 \
  --weights "$REPO_ROOT/fastwam/runs/task_a1_fold_uncond_from_wan22_1e-4/a1_base422_8h20_s42_from_wan22/checkpoints/weights/step_025000.pt" \
  --stats "$REPO_ROOT/fastwam/data/task_a1_fold/dataset_stats.json" \
  --eval-data task_a1_fold \
  --eval-task task_a1_fold_uncond_from_wan22_1e-4 \
  --t5-cache "$REPO_ROOT/fastwam/data/text_embeds_cache/task_a1_fold/*.pt" \
  --preflight-ref "$REPO_ROOT/fastwam/deploy/refs/task_a1_fold.json" \
  "$@"
