#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P1_VERIFY_REPO:-$REPO/logs/frozen_source_overlays/pi05_r1_v1}
CONDITION=${PREDICTIVE_P1_CONDITION:?set PREDICTIVE_P1_CONDITION}
AUDIT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p1_baseline_audit.json
OUTPUT_DIR=${P1_PROTOCOL_OUTPUT_DIR:-$REPO/logs/p1_runtime}

test -s "$VERIFY_REPO/READY"
mkdir -p "$OUTPUT_DIR"
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_source_freeze.py" \
  --repo "$VERIFY_REPO" \
  --audit "$AUDIT" \
  --output "$OUTPUT_DIR/source_freeze_${CONDITION}.json" >/dev/null

export PYTHONPATH="$VERIFY_REPO/kai0/src:${PYTHONPATH:-}"
exec bash "$REPO/train_scripts/kai/eval/run_pi05_predictive_adapter_p1_formal.sh"
