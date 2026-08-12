#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
OUT=$REPO/logs/temporal_grounding/tg4/training_integrity.json
MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_training_integrity.ok
mkdir -p "$(dirname "$OUT")" "$(dirname "$MARKER")"

"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_training.py" \
  --repo "$REPO" --output "$OUT"

cat >"$MARKER" <<EOF
verified=$(date -u +%FT%TZ)
protocol=temporal_grounding_tg4_training_integrity_v1
result=$OUT
EOF
chmod 0664 "$MARKER"
