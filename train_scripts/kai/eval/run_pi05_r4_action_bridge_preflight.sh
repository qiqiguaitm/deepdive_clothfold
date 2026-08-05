#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PYTHON=${PUBLIC_PI05_SERVER_PY:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}
CHECKPOINT=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/ordinary-seed1000/checkpoints/005000/pretrained_model
REPORT=$REPO/logs/r4/action_bridge_preflight_v1.json
MARKER=$REPO/logs/resource_markers/pi05_r4_action_bridge_preflight.ok

export PYTHONPATH=$REPO/train_scripts/kai/eval:$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}
exec "$PYTHON" "$REPO/train_scripts/kai/eval/verify_lerobot_pi05_action_bridge.py" \
  --checkpoint "$CHECKPOINT" \
  --report "$REPORT" \
  --marker "$MARKER"
