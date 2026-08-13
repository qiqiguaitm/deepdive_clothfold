#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
SEED=${TG4_TRAIN_SEED:?set TG4_TRAIN_SEED}
SOURCE=$REPO/train_scripts/kai/sync_temporal_grounding_tg4_checkpoint_from_north.sh

case "$SEED" in
  1100|1101|1102) ;;
  *) exit 2 ;;
esac

test "$(grep -c '^LOCK=\$REPO/logs/locks/temporal_grounding_tg4_materialize.lock$' "$SOURCE")" = 1
slot=$(( (SEED - 1100) % 2 ))
mkdir -p "$REPO/logs/runtime"
runtime=$(mktemp "$REPO/logs/runtime/tg4-materialize.XXXXXX.sh")
trap 'rm -f "$runtime"' EXIT
sed \
  "s|^LOCK=\\\$REPO/logs/locks/temporal_grounding_tg4_materialize.lock$|LOCK=\\\$REPO/logs/locks/temporal_grounding_tg4_materialize.slot${slot}.lock|" \
  "$SOURCE" >"$runtime"
chmod 0555 "$runtime"
bash "$runtime"
