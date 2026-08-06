#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
VERIFY_REPO=${P2_VERIFY_REPO:-$REPO}
SEED=${SEED:?set SEED to 1001 or 1002}
P1_GATE=${PREDICTIVE_P1_GATE:-$REPO/logs/predictive/p1_eval/p1_gate.accepted}
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_protocol.json
INTEGRITY_AMENDMENT=${P2_INTEGRITY_AMENDMENT:-$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p2_integrity_amendment_v1.json}
MANIFEST=$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json

case "$SEED" in
  1001|1002) ;;
  *)
    echo "P2 only permits preregistered seeds 1001 and 1002, got: $SEED" >&2
    exit 2
    ;;
esac

if [[ "$VERIFY_REPO" != "$REPO" ]]; then
  test -s "$VERIFY_REPO/REPLICATION_READY"
fi

CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed${SEED}/49999}
RESULT_NAME=pi05_predictive_adapter_p2_seed${SEED}_normal
RESULT_ROOT=$REPO/lmvla/lawam/results/eval_runs/robotwin/$RESULT_NAME
REPORT=$REPO/lmvla/lmwm/docs/${RESULT_NAME}.json
MARKER=$REPO/logs/resource_markers/${RESULT_NAME}.ok

test -f "$P1_GATE"
python3 "$VERIFY_REPO/kai0/scripts/verify_pi05_predictive_adapter_p2_protocol.py" \
  --repo "$VERIFY_REPO" --manifest "$PROTOCOL"
python3 - "$REPO" "$INTEGRITY_AMENDMENT" <<'PY'
import hashlib
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1])
amendment_path = pathlib.Path(sys.argv[2])
amendment = json.loads(amendment_path.read_text())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

for spec in amendment["parents"].values():
    path = repo / spec["path"]
    actual = sha256(path)
    if actual != spec["sha256"]:
        raise SystemExit(f"P2 integrity parent drift: {path}: {actual} != {spec['sha256']}")
for relative, expected in amendment["runtime_file_sha256"].items():
    path = repo / relative
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"P2 integrity runtime drift: {path}: {actual} != {expected}")
PY

AUDIT_REPORT=$REPO/logs/predictive/p2_audit/seed${SEED}_final_checkpoint.json
AUDIT_MARKER=$REPO/logs/resource_markers/pi05_predictive_adapter_p2_seed${SEED}_checkpoint_audit.ok
python3 "$REPO/lmvla/lmwm/scripts/audit_pi05_predictive_adapter_p2_checkpoint.py" \
  --repo "$REPO" \
  --seed "$SEED" \
  --checkpoint "$CKPT" \
  --reference-checkpoint "${CKPT%/49999}/25000" \
  --source-preflight "$REPO/logs/predictive/p1_preflight/source_freeze_candidate_seed${SEED}.json" \
  --amendment "$INTEGRITY_AMENDMENT" \
  --output "$AUDIT_REPORT" \
  --marker "$AUDIT_MARKER"
test -f "$MANIFEST"

export PYTHONPATH="$VERIFY_REPO/kai0/src:${PYTHONPATH:-}"
env \
  PI05_EVAL_CONFIG_NAME=pi05_predictive_adapter_p1_eval \
  PI05_ASSET_ID=robotwin2.0_absolute_meanstd \
  PREDICTIVE_ACTION_INTERVENTION=normal \
  CKPT="$CKPT" \
  RESULT_NAME="$RESULT_NAME" \
  ROBOTWIN_TEST_NUM=50 \
  ROBOTWIN_EPISODE_SEED_MANIFEST="$MANIFEST" \
  ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500 \
  SEEDS="0 1 2 3" \
  LOCAL_GPU_COUNT=${LOCAL_GPU_COUNT:-4} \
  GPU_INDEX_OFFSET=${GPU_INDEX_OFFSET:-0} \
  MAX_PARALLEL_SEEDS=${MAX_PARALLEL_SEEDS:-4} \
  PORT_BASE_OFFSET=${PORT_BASE_OFFSET:-22000} \
  bash "$REPO/train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"

python3 "$REPO/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py" \
  --manifest "$MANIFEST" \
  --root "$RESULT_ROOT"
python3 "$REPO/lmvla/lmwm/scripts/summarize_robotwin_eval.py" \
  "$RESULT_ROOT" --expected-cells 24 >"$REPORT.tmp"
mv "$REPORT.tmp" "$REPORT"

mkdir -p "$(dirname "$MARKER")"
printf 'validated=%s\nseed=%s\ncheckpoint=%s\nreport=%s\n' \
  "$(date -u +%FT%TZ)" "$SEED" "$CKPT" "$REPORT" >"$MARKER"
