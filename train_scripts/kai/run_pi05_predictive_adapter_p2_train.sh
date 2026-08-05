#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
P2_VERIFY_REPO=${P2_VERIFY_REPO:-$REPO}
TRAIN_SOURCE_REPO=${TRAIN_SOURCE_REPO:-$P2_VERIFY_REPO}
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
if [[ "$P2_VERIFY_REPO" != "$REPO" || "$TRAIN_SOURCE_REPO" != "$REPO" ]]; then
  test -s "$P2_VERIFY_REPO/REPLICATION_READY"
  test -s "$TRAIN_SOURCE_REPO/REPLICATION_READY"
fi
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$P2_VERIFY_REPO" --manifest "$PROTOCOL"

exec env \
  REPO="$REPO" \
  ARM=candidate \
  SEED="$SEED" \
  TRAIN_VERIFY_REPO="$P2_VERIFY_REPO" \
  TRAIN_SOURCE_REPO="$TRAIN_SOURCE_REPO" \
  PREDICTIVE_P0_GATE="$P1_GATE" \
  bash "$REPO/train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
