#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
OUT=$REPO/lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg4.json
MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_analysis.ok

test -f "$REPO/logs/resource_markers/temporal_grounding_tg4_training_integrity.ok"
"$REPO/kai0/.venv/bin/python" \
  "$REPO/lmvla/lmwm/scripts/analyze_temporal_grounding_tg4.py" \
  --repo "$REPO" \
  --output "$OUT" \
  --marker "$MARKER"
