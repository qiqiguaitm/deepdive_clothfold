#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
DOCS=$REPO/lmvla/lmwm/docs
"$REPO/kai0/.venv/bin/python" "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$REPO" \
  --manifest "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p3_analysis_protocol.json" \
  --phase p3
exec "$REPO/kai0/.venv/bin/python" "$REPO/lmvla/lmwm/scripts/analyze_pi05_predictive_adapter_p3.py" \
  --a0 1000="$DOCS/pi05_predictive_adapter_p1_seed1000_a0.json" \
  --a0 1001="$DOCS/pi05_predictive_adapter_p3_a0_seed1001.json" \
  --a0 1002="$DOCS/pi05_predictive_adapter_p3_a0_seed1002.json" \
  --candidate 1000="$DOCS/pi05_predictive_adapter_p1_seed1000_normal.json" \
  --candidate 1001="$DOCS/pi05_predictive_adapter_p2_seed1001_normal.json" \
  --candidate 1002="$DOCS/pi05_predictive_adapter_p2_seed1002_normal.json" \
  --bootstrap-samples 20000 \
  --output "$DOCS/pi05_predictive_adapter_p3_matched_seed_gate.json" \
  --marker "$REPO/logs/resource_markers/pi05_predictive_adapter_p3_analysis.ok"
