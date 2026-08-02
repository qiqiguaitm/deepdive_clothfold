#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${ARM:-s0_privileged}
CONFIG=${CONFIG:-pi05_spatial_s0_privileged}
EXP=${EXP:-pi05_spatial_s0_privileged_preflight_v1}
GPU=${GPU:-0}

mkdir -p "$REPO/kai0/assets/$CONFIG/robotwin2.0_absolute_meanstd" "$REPO/logs/spatial_s0"
cp -f \
  "$REPO/kai0/assets/pi05_robotwin_a0_public_exact_bj/robotwin2.0_absolute_meanstd/norm_stats.json" \
  "$REPO/kai0/assets/$CONFIG/robotwin2.0_absolute_meanstd/norm_stats.json"

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-cache-${USER:-tim}}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

extra=()
if [[ $ARM == s0_privileged ]]; then
  extra+=(
    --target-pairs "$REPO/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
    --frame-cache-root "$REPO/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
  )
fi

cd "$REPO/kai0"
exec .venv/bin/python -u scripts/train_pi05_robotwin_confirmatory.py \
  --arm "$ARM" \
  --config-name "$CONFIG" \
  --exp-name "$EXP" \
  --seed 1000 \
  --data-repo "$REPO/lmvla/lawam/dataset/robotwin2.0" \
  --init-params "$REPO/kai0/checkpoints/pi05_base/params" \
  --asset-id robotwin2.0_absolute_meanstd \
  --episodes-json "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_spatial_s0_episode_split.json" \
  --num-train-steps 1 \
  --num-workers 2 \
  --save-interval 1000 \
  --log-interval 1 \
  "${extra[@]}"
