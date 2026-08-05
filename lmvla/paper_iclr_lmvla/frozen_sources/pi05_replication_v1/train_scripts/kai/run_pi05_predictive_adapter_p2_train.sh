#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
SEED=${SEED:?set SEED to 1001 or 1002}
P1_GATE=${PREDICTIVE_P1_GATE:-$REPO/logs/predictive/p1_eval/p1_gate.accepted}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json

case "$SEED" in
  1001|1002) ;;
  *)
    echo "P2 only permits preregistered seeds 1001 and 1002, got: $SEED" >&2
    exit 2
    ;;
esac

test -f "$P1_GATE"
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$REPO" --manifest "$PROTOCOL"

exec env \
  REPO="$REPO" \
  ARM=candidate \
  SEED="$SEED" \
  PREDICTIVE_P0_GATE="$P1_GATE" \
  bash "$REPO/train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
