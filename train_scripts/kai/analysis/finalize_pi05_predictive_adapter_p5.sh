#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
exec "$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/analyze_pi05_predictive_adapter_p5.py" \
  --public "$REPO/lmvla/lmwm/docs/pi05_predictive_adapter_p5_public_paired.json" \
  --candidate "1000=$REPO/lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_normal.json" \
  --candidate "1001=$REPO/lmvla/lmwm/docs/pi05_predictive_adapter_p2_seed1001_normal.json" \
  --candidate "1002=$REPO/lmvla/lmwm/docs/pi05_predictive_adapter_p2_seed1002_normal.json" \
  --scene-manifest "$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json" \
  --audit-output "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p5_public_episode_audit.json" \
  --output "$REPO/lmvla/paper_iclr_lmvla/RESULTS_pi05_predictive_adapter_p5_public_gate.json" \
  --marker "$REPO/logs/resource_markers/pi05_predictive_adapter_p5_public_gate.ok"
