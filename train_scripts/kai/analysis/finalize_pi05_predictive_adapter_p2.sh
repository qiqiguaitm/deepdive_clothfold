#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
REPORT_DIR=$REPO/lmvla/lmwm/docs
OUTPUT=$REPO/lmvla/paper_iclr_lmvla/RESULTS_pi05_predictive_adapter_p2_gate.json
MARKER_DIR=$REPO/logs/predictive/p2_eval
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json

test -f "$REPO/logs/predictive/p1_eval/p1_gate.accepted"
python3 "$REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$REPO" --manifest "$PROTOCOL"
mkdir -p "$MARKER_DIR"
python3 "$REPO/lmvla/lmwm/scripts/analyze_pi05_predictive_adapter_p2.py" \
  --a0 "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_a0.json" \
  --candidate "1000=$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_normal.json" \
  --candidate "1001=$REPORT_DIR/pi05_predictive_adapter_p2_seed1001_normal.json" \
  --candidate "1002=$REPORT_DIR/pi05_predictive_adapter_p2_seed1002_normal.json" \
  --output "$OUTPUT"

python3 - "$OUTPUT" "$MARKER_DIR" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text())
marker_dir = pathlib.Path(sys.argv[2])
status = "accepted" if result["accepted"] else "rejected"
other = marker_dir / f"p2_gate.{('rejected' if status == 'accepted' else 'accepted')}"
other.unlink(missing_ok=True)
temporary = marker_dir / f".p2_gate.{status}.tmp"
temporary.write_text(json.dumps({"accepted": result["accepted"]}) + "\n")
temporary.replace(marker_dir / f"p2_gate.{status}")
PY
