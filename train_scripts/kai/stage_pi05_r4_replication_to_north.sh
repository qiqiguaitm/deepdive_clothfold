#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
NORTH_ROOT=${NORTH_ROOT:-/vePFS-North-E/vis_robot}
REMOTE_REPO=$NORTH_ROOT/tim/workspace/deepdive_kai0
PAYLOAD=$REPO/logs/r4/north_replication_stage/payload
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_replication_north_stage.ok
REMOTE_MARKER=$REMOTE_REPO/logs/resource_markers/pi05_r4_replication_north_stage.ok
PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_replication_protocol_v1.json
AMENDMENT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_north_training_amendment_v1.json
SYNC=$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh
LOCAL_SITE=/vePFS/tim/workspace/lerobot-main/.venv/lib/python3.12/site-packages
REMOTE_PUBLIC=$NORTH_ROOT/tim/hf_models/SidneyXie_pi05_robotwin

test -s "$PROTOCOL"
test -s "$AMENDMENT"
test -x "$REPO/train_scripts/kai/runtime/pi05_r4_north_training_python.sh"
test -d "$REPO/lmvla/lmwm/data/pi05_r4_training_v1/lerobot_query_chunks"
test -s "$REPO/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz"
test -d "$LOCAL_SITE/accelerate"
test -d "$LOCAL_SITE/accelerate-1.14.0.dist-info"
test -d "$LOCAL_SITE/sentencepiece"
test -d "$LOCAL_SITE/sentencepiece-0.2.1.dist-info"

rm -rf "$PAYLOAD"
mkdir -p "$PAYLOAD"
python3 - "$REPO" "$PAYLOAD" "$PROTOCOL" <<'PY'
import json
import pathlib
import shutil
import sys

repo, payload, protocol_path = map(pathlib.Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
paths = {entry["path"] for entry in protocol["parents"].values()}
paths.update(protocol["file_sha256"])
for relative in sorted(paths):
    source = repo / relative
    destination = payload / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
PY

mkdir -p \
  "$PAYLOAD/lmvla/lmwm/data" \
  "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests" \
  "$PAYLOAD/logs/r4/seed1000" \
  "$PAYLOAD/logs/resource_markers" \
  "$PAYLOAD/runtime/pi05_r4_north_training/site-packages" \
  "$PAYLOAD/train_scripts/kai/runtime"
cp -a "$REPO/lmvla/lmwm/data/pi05_r4_training_v1" "$PAYLOAD/lmvla/lmwm/data/"
cp -a "$AMENDMENT" "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests/"
cp -a "$REPO/logs/r4/seed1000/r4_gate.accepted" "$PAYLOAD/logs/r4/seed1000/"
for marker in \
  pi05_r4_training_runtime.ok \
  pi05_r4_matched_runtime.ok \
  pi05_r4_crave_sidecar.ok; do
  test -s "$REPO/logs/resource_markers/$marker"
  cp -a "$REPO/logs/resource_markers/$marker" "$PAYLOAD/logs/resource_markers/"
done
cp -a \
  "$LOCAL_SITE/accelerate" \
  "$LOCAL_SITE/accelerate-1.14.0.dist-info" \
  "$LOCAL_SITE/sentencepiece" \
  "$LOCAL_SITE/sentencepiece-0.2.1.dist-info" \
  "$PAYLOAD/runtime/pi05_r4_north_training/site-packages/"
cp -a "$REPO/train_scripts/kai/runtime/pi05_r4_north_training_python.sh" \
  "$PAYLOAD/train_scripts/kai/runtime/"

env SRC="$PAYLOAD" DST="$REMOTE_REPO" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-replication-north-v1 \
  bash "$SYNC"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$REMOTE_REPO" "$REMOTE_PUBLIC" "$REMOTE_MARKER" <<'REMOTE'
set -Eeuo pipefail
repo=$1
public_model=$2
marker=$3
wrapper=$repo/train_scripts/kai/runtime/pi05_r4_north_training_python.sh

chmod 0755 "$wrapper"
mkdir -p "$repo/kai0/.venv/bin" "$(dirname "$repo")/lerobot-main/.venv/bin"
ln -sfn "$wrapper" "$repo/kai0/.venv/bin/python"
ln -sfn "$wrapper" "$(dirname "$repo")/lerobot-main/.venv/bin/python"
test -s "$public_model/model.safetensors"
test -s "$public_model/train_config.json"

PI05_R4_MOUNT_ROOT=/vePFS-North-E/vis_robot "$wrapper" - "$repo" <<'PY'
import hashlib
import json
import pathlib
import sys

import accelerate
import google.protobuf
import lerobot
import sentencepiece
import torch

repo = pathlib.Path(sys.argv[1])
protocol = json.loads(
    (repo / "lmvla/paper_iclr_lmvla/manifests/pi05_r4_replication_protocol_v1.json").read_text()
)
rows = [(entry["path"], entry["sha256"]) for entry in protocol["parents"].values()]
rows.extend(protocol["file_sha256"].items())
for relative, expected in rows:
    actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"frozen source mismatch: {relative}: {actual} != {expected}")
assert accelerate.__version__ == "1.14.0"
assert sentencepiece.__version__ == "0.2.1"
assert google.protobuf.__version__ == "7.35.1"
assert torch.__version__ == "2.11.0+cu128"
print(f"verified sources={len(rows)} lerobot={pathlib.Path(lerobot.__file__).resolve()}")
PY

config=$repo/logs/r4/north_replication_stage/preflight_config.json
mkdir -p "$(dirname "$config")"
PI05_R4_MOUNT_ROOT=/vePFS-North-E/vis_robot "$wrapper" \
  "$repo/lmvla/lmwm/scripts/build_pi05_r4_replication_config.py" \
  --public-config "$public_model/train_config.json" --arm ordinary --seed 1001 \
  --world-size 4 --steps 5000 \
  --output-dir "$repo/logs/r4/north_replication_stage/preflight_output" \
  --dataset-root "$repo/lmvla/lmwm/data/pi05_r4_training_v1/lerobot_query_chunks" \
  --model-path "$public_model" \
  --sidecar "$repo/lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz" \
  --output "$config"
PI05_R4_MOUNT_ROOT=/vePFS-North-E/vis_robot "$wrapper" - "$config" <<'PY'
import sys
from lerobot.configs.train import TrainPipelineConfig

config = TrainPipelineConfig.from_pretrained(sys.argv[1])
config.validate()
assert config.batch_size == 4
assert config.batch_size * 4 == 16
assert config.seed == 1001
assert config.steps == 5000
PY
PI05_R4_MOUNT_ROOT=/vePFS-North-E/vis_robot PI05_R4_TRAINING_RUNTIME=1 \
  PYTHONPATH="$repo/lmvla/lmwm/runtime/pi05_r4_training" \
  "$wrapper" "$repo/lmvla/lmwm/runtime/pi05_r4_training/train_entrypoint.py" --check-binding

mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nrepo=%s\npublic_model=%s\nworld_size=4\neffective_batch=16\naccelerate=1.14.0\nsentencepiece=0.2.1\nprotobuf=7.35.1\ntorch=2.11.0+cu128\n' \
  "$(date -u +%FT%TZ)" "$repo" "$public_model" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_repo=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$REMOTE_REPO" "$REMOTE_MARKER" >"$LOCAL_MARKER"
