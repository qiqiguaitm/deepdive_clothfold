#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PY=$REPO/kai0/.venv/bin/python
OUT=$REPO/logs/temporal_grounding/tg2r
MARKER=$REPO/logs/resource_markers/temporal_grounding_tg2r_training_integrity.ok
mkdir -p "$OUT" "$(dirname "$MARKER")"

"$PY" "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg2_recovery_training_v2.py" \
  --repo "$REPO" \
  --output "$OUT/training_integrity.json" \
  --seed-output "$OUT/seed_independence.json"

cat >"$MARKER" <<EOF
verified=$(date -u +%FT%TZ)
protocol=temporal_grounding_tg2_recovery_training_integrity_v2
training_integrity=$OUT/training_integrity.json
seed_independence=$OUT/seed_independence.json
EOF
