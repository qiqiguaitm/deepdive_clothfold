#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ROLLOUT_ROOT=${ROLLOUT_ROOT:-$REPO/lmvla/lawam/results/eval_runs/robotwin/pi05_crave_r0_public_rollouts_v1}
OUTPUT=${OUTPUT:-$REPO/logs/crave_r0/rollout_features}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
MARKER=${MARKER:-$REPO/logs/resource_markers/pi05_crave_r0_rollout_features.ok}
LOG_DIR=$REPO/logs/crave_r0/rollout_feature_logs
mkdir -p "$OUTPUT" "$LOG_DIR" "$(dirname "$MARKER")"

pids=()
for shard in 0 1; do
  gpu=$((GPU_INDEX_OFFSET + shard))
  CUDA_VISIBLE_DEVICES=$gpu \
    "$REPO/kai0/.venv/bin/python" \
    "$REPO/lmvla/lmwm/scripts/extract_pi05_crave_r0_rollout_features.py" \
    --rollout-root "$ROLLOUT_ROOT" --output "$OUTPUT" \
    --shard "$shard" --num-shards 2 --stride 5 \
    >"$LOG_DIR/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

python - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifests = [json.loads((root / f"shard{i}.json").read_text()) for i in range(2)]
records = [record for manifest in manifests for record in manifest["records"]]
identities = {
    (record["task"], int(record["simulator_seed"]), int(record["episode_id"]))
    for record in records
}
assert {manifest["all_episode_count"] for manifest in manifests} == {120}, manifests
assert {manifest["stride"] for manifest in manifests} == {5}, manifests
assert len(records) == len(identities) == 120
assert len(list(root.glob("seed*/*/episode*.npz"))) == 120
PY

printf 'completed=%s\nroot=%s\nepisodes=120\nstride=5\n' \
  "$(date -u +%FT%TZ)" "$OUTPUT" >"$MARKER"
