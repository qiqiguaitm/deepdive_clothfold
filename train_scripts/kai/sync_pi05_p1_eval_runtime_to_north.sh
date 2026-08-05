#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
MANIFEST_REL=lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_eval_amendment_v1.json
OVERLAY_REL=logs/frozen_source_overlays/pi05_r1_v1
LOCAL_DIR=$REPO/logs/p1_north_eval_stage
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_p1_north_eval_stage.ok
REMOTE_DIR=$NORTH_REPO/logs/p1_north_eval_stage
REMOTE_MARKER=$NORTH_REPO/logs/resource_markers/pi05_p1_north_eval_stage.ok

files=(
  "$MANIFEST_REL"
  lmvla/paper_iclr_lmvla/manifests/pi05_p1_frozen_overlay_amendment_v1.json
  lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_failover_stage_v1.json
  lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p1_baseline_audit.json
  lmvla/lmwm/scripts/preflight_pi05_p1_frozen_overlay.py
  lmvla/lmwm/scripts/summarize_robotwin_eval.py
  lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
  lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
  kai0/scripts/serve_policy.py
  kai0/scripts/verify_pi05_predictive_adapter_source_freeze.py
  train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh
  train_scripts/kai/eval/run_pi05_predictive_adapter_p1_formal.sh
  train_scripts/kai/eval/run_pi05_predictive_adapter_p1_frozen.sh
  train_scripts/kai/volc/pi05_predictive_adapter_p1_eval_north_4h20.yaml
  lmvla/lawam/examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh
  lmvla/lawam/examples/Robotwin/eval_files/robotwin_batch_bridge.py
  lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi.py
)

cd "$REPO"
for path in "${files[@]}"; do
  test -f "$path"
done
test -s "$OVERLAY_REL/READY"
mkdir -p "$LOCAL_DIR" "$(dirname "$LOCAL_MARKER")"

printf 'phase=sync-runtime\n'
tar --exclude='__pycache__' --exclude='*.pyc' -cf - \
  "${files[@]}" "$OVERLAY_REL" | \
  ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
    "mkdir -p $(printf %q "$NORTH_REPO") && tar -C $(printf %q "$NORTH_REPO") -xf -"

printf 'phase=verify-remote\n'
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  env NORTH_REPO="$NORTH_REPO" MANIFEST_REL="$MANIFEST_REL" \
  OVERLAY_REL="$OVERLAY_REL" REMOTE_DIR="$REMOTE_DIR" \
  REMOTE_MARKER="$REMOTE_MARKER" bash -s <<'REMOTE'
set -Eeuo pipefail
mkdir -p "$REMOTE_DIR" "$(dirname "$REMOTE_MARKER")"
python3 - "$NORTH_REPO" "$MANIFEST_REL" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest_path = root / sys.argv[2]
manifest = json.loads(manifest_path.read_text())
failures = []
for relative, expected in manifest["synced_file_sha256"].items():
    path = root / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    if actual != expected:
        failures.append(f"{relative}:{actual}!={expected}")
for absolute, expected in manifest["north_prerequisite_sha256"].items():
    path = pathlib.Path(absolute)
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    if actual != expected:
        failures.append(f"{absolute}:{actual}!={expected}")
if failures:
    raise SystemExit("\n".join(failures))
print(f"verified_files={len(manifest['synced_file_sha256'])}")
PY

python3 "$NORTH_REPO/lmvla/lmwm/scripts/preflight_pi05_p1_frozen_overlay.py" \
  --repo "$NORTH_REPO" \
  --overlay "$NORTH_REPO/$OVERLAY_REL" \
  --output "$REMOTE_DIR/overlay_preflight.json" >"$REMOTE_DIR/overlay_preflight.stdout"
python3 - "$REMOTE_DIR/overlay_preflight.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if report.get("passed") is not True:
    raise SystemExit(f"overlay preflight rejected: {report}")
PY

manifest_sha=$(sha256sum "$NORTH_REPO/$MANIFEST_REL" | awk '{print $1}')
printf 'validated=%s\nmanifest_sha256=%s\noverlay=%s\n' \
  "$(date -u +%FT%TZ)" "$manifest_sha" "$NORTH_REPO/$OVERLAY_REL" >"$REMOTE_MARKER"
REMOTE

scp -P "$NORTH_PORT" -o BatchMode=yes \
  "$NORTH_HOST:$REMOTE_DIR/overlay_preflight.json" "$LOCAL_DIR/overlay_preflight.json"
cp "$REPO/$MANIFEST_REL" "$LOCAL_DIR/amendment.json"
printf 'validated=%s\nremote_marker=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$REMOTE_MARKER" "$LOCAL_DIR/overlay_preflight.json" \
  >"$LOCAL_MARKER"
printf 'phase=complete marker=%s\n' "$LOCAL_MARKER"
