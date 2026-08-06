#!/usr/bin/env bash
set -euo pipefail

if [[ "${PREDICTIVE_EFFICIENCY_SNAPSHOT_ACTIVE:-0}" != 1 ]]; then
  snapshot=$(mktemp /tmp/pi05-predictive-efficiency.XXXXXX.sh)
  cp "${BASH_SOURCE[0]}" "$snapshot"
  chmod 700 "$snapshot"
  PREDICTIVE_EFFICIENCY_SNAPSHOT_ACTIVE=1 exec bash "$snapshot" "$@"
fi

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
P2_GATE=${PREDICTIVE_P2_GATE:-$REPO/logs/predictive/p2_eval/p2_gate.accepted}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json
PY=$REPO/kai0/.venv/bin/python
BENCH=$REPO/train_scripts/kai/analysis/benchmark_pi05_policy_latency.py
OUT_DIR=$REPO/logs/efficiency/pi05_predictive_adapter
FINAL=$REPO/logs/efficiency/pi05_predictive_adapter_latency.json
A0=$OUT_DIR/a0.json
CANDIDATE=$OUT_DIR/candidate.json

test -f "$P2_GATE"
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$REPO" --manifest "$PROTOCOL"
mkdir -p "$OUT_DIR"
export OPENPI_DATA_HOME=$REPO/openpi_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$REPO/kai0/src:$REPO/kai0/packages/openpi-client/src:${PYTHONPATH:-}"

if [[ ! -s "$A0" ]]; then
  "$PY" "$BENCH" \
    --arm predictive_p1_a0 \
    --config pi05_predictive_adapter_p1_a0_exact \
    --checkpoint "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000/49999" \
    --warmup 5 --trials 30 --output "$A0"
fi

if [[ ! -s "$CANDIDATE" ]]; then
  PREDICTIVE_ACTION_INTERVENTION=normal "$PY" "$BENCH" \
    --arm predictive_p1_candidate \
    --config pi05_predictive_adapter_p1_eval \
    --checkpoint "$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000/49999" \
    --warmup 5 --trials 30 --output "$CANDIDATE"
fi

export A0 CANDIDATE FINAL P2_GATE
"$PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

a0 = json.loads(Path(os.environ["A0"]).read_text())
candidate = json.loads(Path(os.environ["CANDIDATE"]).read_text())
gate = Path(os.environ["P2_GATE"])
report = {
    "benchmark": "pi05_predictive_adapter_matched_efficiency",
    "p2_gate": str(gate),
    "p2_gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
    "protocol": {
        "hardware_comparison": "same allocated 80-GiB GPU and process environment",
        "warmup": 5,
        "trials": 30,
        "flops_source": "compiled JAX executable cost_analysis",
    },
    "arms": {"a0": a0, "candidate": candidate},
    "delta_candidate_minus_a0": {
        "parameters": candidate["model_parameter_count"] - a0["model_parameter_count"],
        "flops": candidate["xla_cost_analysis"]["flops"] - a0["xla_cost_analysis"]["flops"],
        "warm_gpu_memory_mib": candidate["gpu_memory_mib"]["after_warmup"] - a0["gpu_memory_mib"]["after_warmup"],
        "direct_model_mean_ms": candidate["direct_model"]["mean_ms"] - a0["direct_model"]["mean_ms"],
        "websocket_roundtrip_mean_ms": candidate["websocket_roundtrip"]["mean_ms"] - a0["websocket_roundtrip"]["mean_ms"],
        "websocket_throughput_requests_per_second": candidate["websocket_throughput_requests_per_second"] - a0["websocket_throughput_requests_per_second"],
    },
}
final = Path(os.environ["FINAL"])
temporary = final.with_suffix(final.suffix + ".tmp")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
temporary.replace(final)
PY
