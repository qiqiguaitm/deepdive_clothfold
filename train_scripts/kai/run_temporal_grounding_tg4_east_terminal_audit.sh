#!/usr/bin/env bash
set -euo pipefail

: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
: "${TG4_RECOVERY_OUTPUT:?TG4_RECOVERY_OUTPUT is required}"

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
RUN_ID=temporal_grounding_tg4_${TG4_ARM}_seed${TG4_TRAIN_SEED}
VERIFIER=$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_terminal_recovery.py
INITIALIZATION=$REPO/logs/temporal_grounding/tg4/initialization/$RUN_ID.json
ORDER_DIR=$REPO/logs/temporal_grounding/tg4/data_order/$RUN_ID
READY_MARKER=$REPO/logs/resource_markers/${RUN_ID}_east_terminal_checkpoint_ready.ok
TEMPORARY=$TG4_RECOVERY_OUTPUT.audit.$$
trap 'rm -f "$TEMPORARY"' EXIT

python3 "$VERIFIER" \
  --repo "$REPO" \
  --output "$TEMPORARY" \
  --resource east \
  --cell "$TG4_ARM:$TG4_TRAIN_SEED" \
  --terminal-ready-marker "$READY_MARKER"

# East training writes these audit sidecars as root. Normalize read permission
# only after strict verification so the joint local integrity gate can inspect them.
chmod 0664 "$INITIALIZATION" "$ORDER_DIR"/rank*.json
mkdir -p "$(dirname "$TG4_RECOVERY_OUTPUT")"
mv "$TEMPORARY" "$TG4_RECOVERY_OUTPUT"
