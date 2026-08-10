#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
LAWAM=$REPO/lmvla/lawam
PY=$REPO/kai0/.venv/bin/python
PROBE=$REPO/lmvla/lmwm/scripts/probe_temporal_grounding_tg2_data_order.py
OUT=$REPO/logs/temporal_grounding/tg2/data_order_recovery_probe_v1
ACCELERATE_CONFIG=$LAWAM/starVLA/config/accelerate/ddp_bf16.yaml
MICROBATCHES=256

rm -rf "$OUT/a" "$OUT/b" "$OUT/matched.json"
mkdir -p "$OUT"
cd "$LAWAM"
for label in a b; do
  port=29600
  if [[ "$label" == b ]]; then
    port=29601
  fi
  "$REPO/kai0/.venv/bin/accelerate" launch \
    --config_file "$ACCELERATE_CONFIG" \
    --main_process_ip 127.0.0.1 \
    --main_process_port "$port" \
    --num_machines 1 \
    --num_processes 4 \
    "$PROBE" --repo "$REPO" --label "$label" --microbatches "$MICROBATCHES"
done
"$PY" "$PROBE" --repo "$REPO" --compare --microbatches "$MICROBATCHES"
