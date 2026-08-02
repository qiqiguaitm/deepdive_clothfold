#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/vePFS/tim
NORTH_HOST=root@124.174.16.237
NORTH_PORT=16370
NORTH_ROOT=/vePFS-North-E/vis_robot/tim
STATUS_DIR=/vePFS/tim/workspace/deepdive_kai0/logs/sync_public_pi05_north
LOCAL_ARCHIVE=/vePFS/tim/tmp/pi05_public_north_bundle_20260731.tar
REMOTE_ARCHIVE=$NORTH_ROOT/.transfer_cache/pi05_public_north_bundle_20260731.tar
TOS_URI=tos://transfer-shanghai/temp/deepdive_kai0/pi05_public_north_bundle_20260731.tar

mkdir -p "$STATUS_DIR"
printf 'RUNNING start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"
trap 'rc=$?; printf "FAILED rc=%s end=%s\n" "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"' ERR

cd "$SOURCE_ROOT"
mkdir -p "$(dirname "$LOCAL_ARCHIVE")"
if [ ! -s "$LOCAL_ARCHIVE" ]; then
  tar -cf "$LOCAL_ARCHIVE" \
    hf_models/SidneyXie_pi05_robotwin \
    hf_models/paligemma_tokenizer \
    workspace/lerobot-main/src \
    workspace/lerobot-pi05-server-venv \
    .uv_python/cpython-3.12.13-linux-x86_64-gnu
fi
if [ "${SKIP_TOS_UPLOAD:-0}" != 1 ]; then
  tosutil cp "$LOCAL_ARCHIVE" "$TOS_URI" -p=8 -vchecksum
fi

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" \
  "mkdir -p '$NORTH_ROOT' '$(dirname "$REMOTE_ARCHIVE")' && \
   tosutil cp '$TOS_URI' '$REMOTE_ARCHIVE' -f -p=8 -vchecksum && \
   tar -C '$NORTH_ROOT' -xf '$REMOTE_ARCHIVE'"

ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" bash -s -- "$NORTH_ROOT" <<'REMOTE'
set -euo pipefail
root=$1
venv=$root/workspace/lerobot-pi05-server-venv
lerobot_src=$root/workspace/lerobot-main/src
python_root=$root/.uv_python/cpython-3.12.13-linux-x86_64-gnu
ln -sfn "$python_root/bin/python3.12" "$venv/bin/python3.12"
ln -sfn python3.12 "$venv/bin/python3"
ln -sfn python3 "$venv/bin/python"
sed -i "s#^home = .*#home = $python_root/bin#" "$venv/pyvenv.cfg"
printf '%s\n' "$lerobot_src" > "$venv/lib/python3.12/site-packages/__editable__.lerobot-0.6.1.pth"
test -s "$root/hf_models/SidneyXie_pi05_robotwin/model.safetensors"
test -s "$root/hf_models/paligemma_tokenizer/tokenizer.model"
"$venv/bin/python" - <<'PY'
import torch
import lerobot
from lerobot.policies.pi05 import PI05Policy

print("PUBLIC_PI05_NORTH_READY", torch.__version__, lerobot.__version__)
PY
REMOTE

touch "$STATUS_DIR/complete"
printf 'FINISHED rc=0 end=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/status"
unlink "$LOCAL_ARCHIVE"
ssh -p "$NORTH_PORT" -o BatchMode=yes "$NORTH_HOST" "unlink '$REMOTE_ARCHIVE'"
