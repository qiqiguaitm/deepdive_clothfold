#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999}
OUT=$REPO/logs/crave_r0/probe_features_v2

test -f "$CKPT/_CHECKPOINT_METADATA"
test -f "$CKPT/params/_METADATA"
test -f "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/READY_LABELS"
test -f "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz"
test -f "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz"
mkdir -p "$OUT"

PIDS=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$((GPU_INDEX_OFFSET + shard)) \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  JAX_COMPILATION_CACHE_DIR="/tmp/jax-crave-r0-probe-$shard" \
  "$REPO/kai0/.venv/bin/python" -u \
    "$REPO/kai0/scripts/extract_pi05_crave_r0_probe_features.py" \
    --checkpoint "$CKPT" \
    --data-repo "$REPO/../VLANeXt-main/datasets/robotwin2.0_official_prompts_v21" \
    --pairs "$REPO/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/pairs.npz" \
    --episode-split "$REPO/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/episode_split.json" \
    --frame-cache-root "$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256" \
    --norm-assets-dir "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj" \
    --probe-train "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz" \
    --probe-eval "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz" \
    --batch-size 16 --shard-count 2 --shard-index "$shard" \
    --output "$OUT/shard${shard}.npz" \
    >"$OUT/shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

for shard in 0 1; do
  test -s "$OUT/shard${shard}.npz"
  test -s "$OUT/shard${shard}.json"
done
