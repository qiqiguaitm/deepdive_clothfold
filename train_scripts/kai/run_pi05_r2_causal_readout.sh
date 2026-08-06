#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PYTHON=${R2_PYTHON:-$REPO/kai0/.venv/bin/python}
OUTPUT=${R2_READOUT_OUTPUT:-$REPO/lmvla/lmwm/data/pi05_r2_causal_readout_v1}

cd "$REPO"
exec "$PYTHON" lmvla/lmwm/scripts/build_pi05_r2_causal_readout.py \
  --selection lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json \
  --labels-manifest lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json \
  --labels lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz \
  --probe-labels lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz \
  --reference-trajectories lmvla/lmwm/data/pi05_crave_r0_v1/reference_trajectories.npz \
  --reference-feature-dir lmvla/lmwm/data/robotwin_dinov3base \
  --output-dir "$OUTPUT" \
  --device cuda
