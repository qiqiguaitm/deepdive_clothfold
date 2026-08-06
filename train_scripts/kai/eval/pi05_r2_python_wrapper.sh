#!/usr/bin/env bash
set -euo pipefail

REPO=${RT_REPO:-/vePFS/tim/workspace/deepdive_kai0}
REAL_PYTHON=${R2_REAL_PYTHON:-/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python}

if [[ $# -gt 0 && "$1" == */batched_eval_runner.py ]]; then
  base_runner=$1
  shift
  exec "$REAL_PYTHON" "$REPO/train_scripts/kai/eval/run_pi05_r2_batched_eval_runner.py" \
    "$base_runner" -- "$@"
fi

exec "$REAL_PYTHON" "$@"
