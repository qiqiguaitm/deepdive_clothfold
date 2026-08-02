#!/usr/bin/env bash
set -uo pipefail

repo=/vePFS/tim/workspace/deepdive_kai0
output=$repo/logs/l2_six_task_intervention_analysis.json
log=$repo/logs/l2_six_task_analysis_monitor.log
authoritative_markers=(
  l2_strict_absolute_zero.ok
  l2_strict_absolute_cross_task.ok
  l2_strict_absolute_within_task_shuffle.ok
  l2_strict_residual_zero.ok
  l2_strict_residual_cross_task.ok
  l2_strict_residual_within_task_shuffle.ok
  l2_strict_combo_zero.ok
  l2_strict_combo_cross_task.ok
  l2_strict_combo_within_task_shuffle.ok
)

mkdir -p "$(dirname "$output")"
while true; do
  timestamp=$(date -u +%FT%TZ)
  if python3 "$repo/train_scripts/kai/analysis/analyze_robotwin_l2_six_task.py" \
    --repo "$repo" --output "$output" >> "$log" 2>&1; then
    echo "[$timestamp] complete output=$output" >> "$log"
    exit 0
  fi
  marker_count=0
  for marker in "${authoritative_markers[@]}"; do
    if [ -f "$repo/logs/resource_markers/$marker" ]; then
      marker_count=$((marker_count + 1))
    fi
  done
  summary_count=$(find "$repo/lmvla/lawam/results/eval_runs/robotwin" -type f \
    -path '*seed2026_strict_unseen*/summary.json' | wc -l)
  echo "[$timestamp] waiting strict-markers=$marker_count/9 strict-summaries=$summary_count/216" >> "$log"
  sleep 120
done
