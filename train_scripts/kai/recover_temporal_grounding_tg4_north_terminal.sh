#!/usr/bin/env bash
set -euo pipefail

: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
: "${TG4_RECOVERY_OUTPUT:?TG4_RECOVERY_OUTPUT is required}"

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/temporal_grounding_11fb843}
POLL_SECONDS=${TG4_RECOVERY_POLL_SECONDS:-300}
TIMEOUT_SECONDS=${TG4_RECOVERY_TIMEOUT_SECONDS:-86400}
RUN_ID=temporal_grounding_tg4_${TG4_ARM}_seed${TG4_TRAIN_SEED}
REMOTE_REPORT=/tmp/${RUN_ID}_terminal_recovery.json
VERIFIER=$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_terminal_recovery.py

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  count=$(ssh -p "$PORT" -o BatchMode=yes "$HOST" \
    "find '$NORTH_REPO/lmvla/lawam/results/Checkpoints/robotwin' -path '*+$RUN_ID/final_model/pytorch_model.pt' -type f | wc -l")
  if [[ $count == 1 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done
if [[ ${count:-0} != 1 ]]; then
  echo "timed out waiting for exactly one final checkpoint for $RUN_ID" >&2
  exit 1
fi

ssh -p "$PORT" -o BatchMode=yes "$HOST" \
  "python3 - --repo '$NORTH_REPO' --output '$REMOTE_REPORT' --resource north --cell '$TG4_ARM:$TG4_TRAIN_SEED'" \
  <"$VERIFIER"

mkdir -p "$(dirname "$TG4_RECOVERY_OUTPUT")"
temporary=$TG4_RECOVERY_OUTPUT.tmp.$$
scp -P "$PORT" -q "$HOST:$REMOTE_REPORT" "$temporary"
chmod 0664 "$temporary"
mv "$temporary" "$TG4_RECOVERY_OUTPUT"
ssh -p "$PORT" -o BatchMode=yes "$HOST" "rm -f '$REMOTE_REPORT'"
