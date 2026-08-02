#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ARM=${ARM:?set ARM to s0_no_goal, s0_current, or s0_privileged}
GPU=${GPU:-0}
STEPS=${STEPS:-1000}
WORKERS=${WORKERS:-2}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
CONFIG=${CONFIG:-pi05_spatial_${ARM}}
EXP=${EXP:-pi05_spatial_${ARM}_seed1000_steps${STEPS}}

case "$ARM" in
  s0_no_goal|s0_current|s0_privileged) ;;
  *) echo "unsupported ARM=$ARM" >&2; exit 2 ;;
esac

mkdir -p "$REPO/kai0/assets/$CONFIG/robotwin2.0_absolute_meanstd"
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
  --num-train-steps "$STEPS" \
  --num-workers "$WORKERS" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 10 \
  "${extra[@]}"
