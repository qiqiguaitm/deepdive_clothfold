#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
readonly PYTHON=${LEROBOT_TRAINING_PYTHON:-/vePFS/tim/workspace/lerobot-main/.venv/bin/python}
readonly REQUIREMENTS=$REPO/lmvla/lmwm/runtime/pi05_r4_training/requirements.lock

test -x "$PYTHON"
test -s "$REQUIREMENTS"
UV_CACHE_DIR=${UV_CACHE_DIR:-/vePFS/tim/workspace/.uv-cache} \
  UV_LINK_MODE=copy \
  uv pip install --python "$PYTHON" --requirement "$REQUIREMENTS"
"$PYTHON" -c 'import sentencepiece; print(sentencepiece.__version__)'
