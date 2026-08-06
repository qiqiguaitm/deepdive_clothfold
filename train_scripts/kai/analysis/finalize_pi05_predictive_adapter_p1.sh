#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
REPORT_DIR=$REPO/lmvla/lmwm/docs
OUTPUT=$REPO/lmvla/paper_iclr_lmvla/RESULTS_pi05_predictive_adapter_p1_seed1000_gate.json
MARKER_DIR=$REPO/logs/predictive/p1_eval

mkdir -p "$MARKER_DIR"
python3 "$REPO/lmvla/lmwm/scripts/analyze_pi05_predictive_adapter_p1.py" \
  --normal "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_normal.json" \
  --a0 "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_a0.json" \
  --zero-gate "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_zero_gate.json" \
  --shuffled "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_shuffled.json" \
  --masked "$REPORT_DIR/pi05_predictive_adapter_p1_seed1000_masked.json" \
  --output "$OUTPUT"

python3 - "$OUTPUT" "$MARKER_DIR" <<'PY'
import json
import pathlib
import sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text())
marker_dir = pathlib.Path(sys.argv[2])
status = "accepted" if result["accepted"] else "rejected"
other = marker_dir / f"p1_gate.{('rejected' if status == 'accepted' else 'accepted')}"
other.unlink(missing_ok=True)
temporary = marker_dir / f".p1_gate.{status}.tmp"
temporary.write_text(json.dumps({"accepted": result["accepted"]}) + "\n")
temporary.replace(marker_dir / f"p1_gate.{status}")
PY
