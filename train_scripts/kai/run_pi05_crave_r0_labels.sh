#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}

test -f "$REPO/logs/resource_markers/pi05_crave_r0_features.ok"
test -f "$REPO/lmvla/lmwm/data/pi05_crave_r0_v1/READY_SELECTION"

cd "$REPO"
exec "$REPO/kai0/.venv/bin/python" -u \
  lmvla/lmwm/scripts/build_pi05_crave_r0_labels.py \
  --selection lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json \
  --panel lmvla/lmwm/data/pi05_crave_r0_v1/panel.npz \
  --feature-dir lmvla/lmwm/data/robotwin_dinov3base \
  --output lmvla/lmwm/data/pi05_crave_r0_v1 \
  --device cuda --chunk-size 512
