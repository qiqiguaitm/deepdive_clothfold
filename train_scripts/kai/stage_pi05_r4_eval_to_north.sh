#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
NORTH_REPO=${NORTH_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
NORTH_HOST=${NORTH_HOST:-root@124.174.16.237}
NORTH_PORT=${NORTH_PORT:-16370}
STAGE=$NORTH_REPO/.staging/pi05_r4_eval_north_v1/repo
PAYLOAD=$REPO/logs/r4/north_eval_stage/payload
LOCAL_MARKER=$REPO/logs/resource_markers/pi05_r4_eval_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/pi05_r4_eval_north_stage.ok
AMENDMENT=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_north_eval_amendment_v1.json
INTEGRITY=$REPO/logs/r4/checkpoint_integrity_v1.json
SYNC=$REPO/train_scripts/kai/sync_tree_to_north_verified_tos.sh
RUNTIME=/vePFS/tim/workspace/lerobot-pi05-server-venv
PYTHON_RUNTIME=/vePFS/tim/.uv_python/cpython-3.12.13-linux-x86_64-gnu
LEROBOT_SRC=/vePFS/tim/workspace/lerobot-main/src
TOKENIZER=/vePFS/tim/hf_models/paligemma_tokenizer

test -s "$AMENDMENT"
test -s "$INTEGRITY"
test -s "$SYNC"
test -x "$RUNTIME/bin/python"
test -x "$PYTHON_RUNTIME/bin/python3.12"
test -f "$LEROBOT_SRC/lerobot/__init__.py"
test -s "$TOKENIZER/tokenizer.model"

rm -rf "$PAYLOAD"
mkdir -p \
  "$PAYLOAD/train_scripts/kai/eval" \
  "$PAYLOAD/train_scripts/kai/volc" \
  "$PAYLOAD/lmvla/lawam/examples/Robotwin" \
  "$PAYLOAD/lmvla/lawam" \
  "$PAYLOAD/lmvla/lmwm/data" \
  "$PAYLOAD/lmvla/lmwm/scripts" \
  "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests" \
  "$PAYLOAD/kai0/src" \
  "$PAYLOAD/kai0/packages/openpi-client" \
  "$PAYLOAD/logs/resource_markers" \
  "$PAYLOAD/logs/r4"

cp -a "$REPO/lmvla/lawam/deployment" "$PAYLOAD/lmvla/lawam/"
cp -a "$REPO/lmvla/lawam/starVLA" "$PAYLOAD/lmvla/lawam/"
cp -a "$REPO/lmvla/lawam/examples/Robotwin/eval_files" \
  "$PAYLOAD/lmvla/lawam/examples/Robotwin/"
cp -a "$REPO/kai0/src/openpi" "$PAYLOAD/kai0/src/"
cp -a "$REPO/kai0/packages/openpi-client/src" \
  "$PAYLOAD/kai0/packages/openpi-client/"

for file in \
  train_scripts/kai/eval/lerobot_pi05_action_bridge.py \
  train_scripts/kai/eval/robotwin_python_wrapper_north.sh \
  train_scripts/kai/eval/run_pi05_r4_formal_eval.sh \
  train_scripts/kai/eval/serve_lerobot_pi05.py; do
  cp -a "$REPO/$file" "$PAYLOAD/$(dirname "$file")/"
done
for file in \
  lmvla/lmwm/scripts/summarize_robotwin_eval.py \
  lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py; do
  cp -a "$REPO/$file" "$PAYLOAD/lmvla/lmwm/scripts/"
done
cp -a "$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json" \
  "$PAYLOAD/lmvla/lmwm/data/"
cp -a \
  "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_formal_eval_protocol_v1.json" \
  "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_local_eval_parallelism_amendment_v1.json" \
  "$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r4_action_bridge_amendment_v1.json" \
  "$AMENDMENT" \
  "$PAYLOAD/lmvla/paper_iclr_lmvla/manifests/"
cp -a "$INTEGRITY" "$PAYLOAD/logs/r4/"
for marker in \
  pi05_r4_checkpoint_permissions.ok \
  pi05_r4_action_bridge_preflight.ok \
  pi05_r4_terminal_outcome-seed1000.ok \
  pi05_r4_outcome_free_crave-seed1000.ok; do
  test -s "$REPO/logs/resource_markers/$marker"
  cp -a "$REPO/logs/resource_markers/$marker" "$PAYLOAD/logs/resource_markers/"
done

env SRC="$PAYLOAD" DST="$STAGE" NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-code \
  bash "$SYNC"

for arm in terminal_outcome outcome_free_crave; do
  source_model=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/$arm-seed1000/checkpoints/005000/pretrained_model
  remote_model=$STAGE/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/$arm-seed1000/checkpoints/005000/pretrained_model
  test -s "$source_model/model.safetensors"
  env SRC="$source_model" DST="$remote_model" SYNC_EVAL_ONLY=1 \
    NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-$arm bash "$SYNC"
done

env SRC="$RUNTIME" DST="$STAGE/runtime/venv" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-venv bash "$SYNC"
env SRC="$PYTHON_RUNTIME" DST="$STAGE/runtime/python" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-python bash "$SYNC"
env SRC="$LEROBOT_SRC" DST="$STAGE/runtime/lerobot/src" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-lerobot bash "$SYNC"
env SRC="$TOKENIZER" DST="$STAGE/runtime/tokenizer" \
  NORTH_TOS_PREFIX=temp/deepdive_kai0/pi05-r4-eval-north-v1-tokenizer bash "$SYNC"

expected_terminal=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["arms"]["terminal_outcome"]["model_sha256"])' "$INTEGRITY")
expected_crave=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["arms"]["outcome_free_crave"]["model_sha256"])' "$INTEGRITY")

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- \
  "$STAGE" "$REMOTE_MARKER" "$expected_terminal" "$expected_crave" <<'REMOTE'
set -Eeuo pipefail
stage=$1
marker=$2
expected_terminal=$3
expected_crave=$4
site=$stage/runtime/venv/lib/python3.12/site-packages
python=$stage/runtime/python/bin/python3.12
export PYTHONPATH="$stage/runtime/lerobot/src:$site:$stage/kai0/src:$stage/kai0/packages/openpi-client/src"
test -x "$python"
test -s "$stage/runtime/tokenizer/tokenizer.model"
echo "$expected_terminal  $stage/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/terminal_outcome-seed1000/checkpoints/005000/pretrained_model/model.safetensors" | sha256sum -c -
echo "$expected_crave  $stage/lmvla/lmwm/checkpoints/pi05_r4_matched_v1/outcome_free_crave-seed1000/checkpoints/005000/pretrained_model/model.safetensors" | sha256sum -c -
"$python" - <<'PY'
import pathlib
import lerobot
import openpi
import torch
import transformers
print('lerobot', pathlib.Path(lerobot.__file__).resolve())
print('openpi', pathlib.Path(openpi.__file__).resolve())
print('torch', torch.__version__)
print('transformers', transformers.__version__)
PY
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nstage=%s\nterminal_sha256=%s\noutcome_free_crave_sha256=%s\n' \
  "$(date -u +%FT%TZ)" "$stage" "$expected_terminal" "$expected_crave" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
