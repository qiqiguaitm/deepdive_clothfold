#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P2_VERIFY_REPO:-$REPO}
GATE=$REPO/logs/predictive/p2_eval/p2_gate.accepted
LATENCY=$REPO/logs/efficiency/pi05_predictive_adapter_latency.json
MEMORY=$REPO/logs/efficiency/pi05_predictive_adapter_train_memory.json
OUTPUT=$REPO/lmvla/paper_iclr_lmvla/RESULTS_pi05_predictive_adapter_p2_efficiency.json
MARKER=$REPO/logs/resource_markers/pi05_predictive_adapter_p2_efficiency.ok
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json

test -f "$GATE"
test -s "$LATENCY"
test -s "$MEMORY"
if [[ "$VERIFY_REPO" != "$REPO" ]]; then
  test -s "$VERIFY_REPO/REPLICATION_READY"
fi
python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$VERIFY_REPO" --manifest "$PROTOCOL"
python3 - "$GATE" "$LATENCY" "$MEMORY" "$OUTPUT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

gate, latency, memory, output = map(Path, sys.argv[1:])
gate_sha256 = hashlib.sha256(gate.read_bytes()).hexdigest()
latency_report = json.loads(latency.read_text())
memory_report = json.loads(memory.read_text())
if latency_report.get("p2_gate_sha256") != gate_sha256:
    raise ValueError("latency report was not produced from the accepted P2 gate")
if memory_report.get("p2_gate_sha256") != gate_sha256:
    raise ValueError("training-memory report was not produced from the accepted P2 gate")
latency_delta = latency_report.get("delta_candidate_minus_a0", {})
required_latency = {
    "parameters",
    "flops",
    "direct_model_mean_ms",
    "websocket_roundtrip_mean_ms",
    "websocket_throughput_requests_per_second",
}
if not required_latency.issubset(latency_delta):
    raise ValueError("latency report is missing required parameter/FLOP/latency metrics")
if "delta_candidate_minus_a0_mib" not in memory_report:
    raise ValueError("training-memory report is missing the matched peak-memory delta")
report = {
    "schema_version": 1,
    "protocol": "pi05_predictive_adapter_p2_efficiency_v1",
    "complete": True,
    "p2_gate_sha256": gate_sha256,
    "latency_and_compute": latency_report,
    "training_memory": memory_report,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\nreport=%s\n' "$(date -u +%FT%TZ)" "$OUTPUT" >"$MARKER"
