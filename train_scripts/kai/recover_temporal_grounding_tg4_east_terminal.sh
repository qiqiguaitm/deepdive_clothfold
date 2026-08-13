#!/usr/bin/env bash
set -euo pipefail

: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"
: "${TG4_READY_OUTPUT:?TG4_READY_OUTPUT is required}"

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
POLL_SECONDS=${TG4_RECOVERY_POLL_SECONDS:-300}
TIMEOUT_SECONDS=${TG4_RECOVERY_TIMEOUT_SECONDS:-86400}
RUN_ID=temporal_grounding_tg4_${TG4_ARM}_seed${TG4_TRAIN_SEED}
CHECKPOINT_ROOT=$REPO/lmvla/lawam/results/Checkpoints/robotwin
LOG_ROOT=$REPO/logs/temporal_grounding/tg4/entrypoint
EXPECTED_ERROR="line 118: el.future_action_window_size=49: command not found"

deadline=$((SECONDS + TIMEOUT_SECONDS))
count=0
while (( SECONDS < deadline )); do
  count=$(find "$CHECKPOINT_ROOT" \
    -path "*+$RUN_ID/final_model/pytorch_model.pt" -type f | wc -l)
  mapfile -t logs < <(find "$LOG_ROOT" -maxdepth 1 -type f \
    -name "${TG4_ARM}_s${TG4_TRAIN_SEED}_east_*.log" | sort)
  if [[ $count == 1 && ${#logs[@]} == 1 ]] \
      && grep -Fq "$RUN_ID: 100%" "${logs[0]}" \
      && grep -Fq "and that's all" "${logs[0]}" \
      && grep -Fq "$EXPECTED_ERROR" "${logs[0]}"; then
    break
  fi
  sleep "$POLL_SECONDS"
done
if [[ $count != 1 || ${#logs[@]} != 1 ]]; then
  echo "timed out waiting for one complete terminal artifact set for $RUN_ID" >&2
  exit 1
fi

mkdir -p "$(dirname "$TG4_READY_OUTPUT")"
temporary=$TG4_READY_OUTPUT.tmp.$$
printf 'ready=%s\nrun_id=%s\nentrypoint_log=%s\n' \
  "$(date -u +%FT%TZ)" "$RUN_ID" "${logs[0]}" >"$temporary"
chmod 0664 "$temporary"
mv "$temporary" "$TG4_READY_OUTPUT"
