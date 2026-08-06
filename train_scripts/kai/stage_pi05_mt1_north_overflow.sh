#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
SEED=${SEED:?set SEED}
STAGE_MARKER=${STAGE_MARKER:-$REPO/logs/resource_markers/pi05_mt1_seed${SEED}_north_eval_checkpoint.ok}
DECISION_MARKER=${DECISION_MARKER:-$REPO/logs/resource_markers/pi05_mt1_seed${SEED}_north_stage_decided.ok}

mkdir -p "$(dirname "$DECISION_MARKER")"
outcome=skipped
sync_rc=0
if env SEED="$SEED" MARKER="$STAGE_MARKER" \
  bash "$REPO/train_scripts/kai/sync_pi05_mt1_seed1000_to_north.sh"; then
  outcome=staged
else
  sync_rc=$?
  rm -f "$STAGE_MARKER"
fi

temporary="${DECISION_MARKER}.tmp.$$"
printf 'decided=%s\nseed=%s\noutcome=%s\nsync_rc=%s\nstage_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$SEED" "$outcome" "$sync_rc" "$STAGE_MARKER" \
  > "$temporary"
mv "$temporary" "$DECISION_MARKER"

if [ "$outcome" = skipped ]; then
  echo "North staging failed with rc=$sync_rc; GF1/East fallback remains enabled" >&2
fi
