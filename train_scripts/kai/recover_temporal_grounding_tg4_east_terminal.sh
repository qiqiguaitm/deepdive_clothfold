#!/usr/bin/env bash
set -euo pipefail

: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
: "${TG4_RECOVERY_OUTPUT:?TG4_RECOVERY_OUTPUT is required}"

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
POLL_SECONDS=${TG4_RECOVERY_POLL_SECONDS:-300}
TIMEOUT_SECONDS=${TG4_RECOVERY_TIMEOUT_SECONDS:-86400}
RUN_ID=temporal_grounding_tg4_${TG4_ARM}_seed${TG4_TRAIN_SEED}
CHECKPOINT_ROOT=$REPO/lmvla/lawam/results/Checkpoints/robotwin
VERIFIER=$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_terminal_recovery.py

deadline=$((SECONDS + TIMEOUT_SECONDS))
count=0
while (( SECONDS < deadline )); do
  count=$(find "$CHECKPOINT_ROOT" \
    -path "*+$RUN_ID/final_model/pytorch_model.pt" -type f | wc -l)
  if [[ $count == 1 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
if [[ $count != 1 ]]; then
  echo "timed out waiting for exactly one final checkpoint for $RUN_ID" >&2
  exit 1
fi

python3 "$VERIFIER" \
  --repo "$REPO" \
  --output "$TG4_RECOVERY_OUTPUT" \
  --resource east \
  --cell "$TG4_ARM:$TG4_TRAIN_SEED"
