#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
REMOTE_STAGE=$NORTH_REPO/.staging/pi05_r4_sidecar_v1/repo
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/pi05_r4_crave_sidecar.ok
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_crave_sidecar.ok
REMOTE_OUT=$REMOTE_STAGE/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz
REMOTE_REPORT=$REMOTE_STAGE/logs/r4/training/crave_weights_report.json
LOCAL_OUT=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz
LOCAL_REPORT=$REPO/logs/r4/training/crave_weights_report.json
CHUNKS=$REPO/lmvla/lmwm/data/pi05_r4_training_v1/query_action_chunks.npz

ssh -p 16370 -o BatchMode=yes root@124.174.16.237 \
  "test -s $(printf %q "$REMOTE_MARKER") && test -s $(printf %q "$REMOTE_OUT") && test -s $(printf %q "$REMOTE_REPORT")"
mkdir -p "$(dirname "$LOCAL_OUT")" "$(dirname "$LOCAL_REPORT")"
scp -P 16370 -o BatchMode=yes "root@124.174.16.237:$REMOTE_OUT" "$LOCAL_OUT.tmp"
scp -P 16370 -o BatchMode=yes "root@124.174.16.237:$REMOTE_REPORT" "$LOCAL_REPORT.tmp"

python3 - "$LOCAL_OUT.tmp" "$LOCAL_REPORT.tmp" "$CHUNKS" <<'PY'
import hashlib
import json
import numpy as np
import sys

sidecar_path, report_path, chunks_path = sys.argv[1:]
report = json.load(open(report_path))
digest = hashlib.sha256(open(sidecar_path, "rb").read()).hexdigest()
if report.get("protocol") != "pi05_r4_outcome_free_crave_weight_sidecar_v1":
    raise SystemExit("unexpected R4 sidecar protocol")
if report.get("sample_count") != 6313 or report.get("sidecar_sha256") != digest:
    raise SystemExit("R4 sidecar count or hash mismatch")
with np.load(sidecar_path, allow_pickle=False) as sidecar, np.load(chunks_path, allow_pickle=False) as chunks:
    if len(sidecar["weight"]) != len(chunks["task"]):
        raise SystemExit("R4 sidecar length mismatch")
    for field in ("task", "scene_seed", "query_index", "query_frame"):
        if not np.array_equal(sidecar[field], chunks[field]):
            raise SystemExit(f"R4 sidecar alignment mismatch: {field}")
    weights = sidecar["weight"]
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise SystemExit("R4 sidecar contains invalid weights")
PY
mv "$LOCAL_OUT.tmp" "$LOCAL_OUT"
mv "$LOCAL_REPORT.tmp" "$LOCAL_REPORT"
mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nsidecar=%s\nreport=%s\nsource=Robot-North-H20\n' \
  "$(date -u +%FT%TZ)" "$LOCAL_OUT" "$LOCAL_REPORT" >"$LOCAL_MARKER"
