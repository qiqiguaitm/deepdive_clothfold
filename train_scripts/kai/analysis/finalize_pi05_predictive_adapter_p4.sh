#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
DOCS=$REPO/lmvla/lmwm/docs
"$REPO/kai0/.venv/bin/python" "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p345_protocol.py" \
  --repo "$REPO" \
  --manifest "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p4_analysis_protocol.json" \
  --phase p4
args=()
for seed in 1000 1001 1002; do
  if [[ "$seed" == 1000 ]]; then
    prefix=pi05_predictive_adapter_p1_seed1000
  else
    prefix=pi05_predictive_adapter_p2_seed${seed}
  fi
  args+=(--normal "$seed=$DOCS/${prefix}_normal.json")
  for condition in shuffled zero_gate masked; do
    if [[ "$seed" == 1000 ]]; then
      path=$DOCS/pi05_predictive_adapter_p1_seed1000_${condition}.json
    else
      path=$DOCS/pi05_predictive_adapter_p4_seed${seed}_${condition}.json
    fi
    args+=(--${condition//_/-} "$seed=$path")
  done
done
exec "$REPO/kai0/.venv/bin/python" "$REPO/lmvla/lmwm/scripts/analyze_pi05_predictive_adapter_p4.py" \
  "${args[@]}" --bootstrap-samples 20000 \
  --output "$DOCS/pi05_predictive_adapter_p4_intervention_gate.json" \
  --marker "$REPO/logs/resource_markers/pi05_predictive_adapter_p4_analysis.ok"
