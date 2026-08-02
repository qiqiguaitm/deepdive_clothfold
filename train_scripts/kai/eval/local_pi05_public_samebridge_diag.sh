#!/usr/bin/env bash
set -euo pipefail

REPO=/vePFS/tim/workspace/deepdive_kai0
LAWAM=$REPO/lmvla/lawam
MODEL=/vePFS/tim/hf_models/SidneyXie_pi05_robotwin
TOKENIZER=/vePFS/tim/hf_models/paligemma_tokenizer
SERVER_PY=/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG=$LAWAM/logs/local_rteval/pi05_public_samebridge_diag_${STAMP}.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_EXTENSIONS_DIR=/vePFS/tim/runtime/torch_extensions/a100_sm80_py310
export TORCHINDUCTOR_CACHE_DIR=/vePFS/tim/runtime/torchinductor/pi05_a100_torch211
mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
test -f "$MODEL/model.safetensors"
test -f "$TOKENIZER/tokenizer.model"
test -x "$SERVER_PY"
bash "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh"
source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
"$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh" -c \
  'import sapien.core as sapien; sapien.SapienRenderer(); print("SAPIEN_RENDER_OK")'

export STAR_VLA_PYTHON=$SERVER_PY
export ROBOTWIN_SERVER_BACKEND=openpi
export ROBOTWIN_OPENPI_CONFIG=lerobot_pi05
export OPENPI_SERVE_SCRIPT=$REPO/train_scripts/kai/eval/serve_lerobot_pi05.py
export PALIGEMMA_TOKENIZER_PATH=$TOKENIZER
export KAI0_ROOT=$REPO
export ROBOTWIN_MODEL_INTERFACE=openpi
export ROBOTWIN_PATH=/vePFS/HuanQian/RoboTwin
export ROBOTWIN_PYTHON=$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh
export PYTHONPATH="$REPO/kai0/packages/openpi-client/src:$REPO/kai0/src:${PYTHONPATH:-}"
export ROBOTWIN_TASKS=beat_block_hammer
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=1
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_REPLAN_STEPS=50
export ROBOTWIN_ACTION_ENSEMBLE=0
export SEED=0
export PORT_BASE=10300
export ROBOTWIN_CKPT_ALIAS=SidneyXie_pi05_robotwin
export ROBOTWIN_EVAL_ROOT=$LAWAM/results/eval_runs/robotwin/pi05_public_samebridge_diag_v1

cd "$LAWAM"
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  "$MODEL" "$TASK_CONFIG" public-samebridge-diag-seed0-local

summary_count=$(find "$ROBOTWIN_EVAL_ROOT" -name summary.json -type f | wc -l)
test "$summary_count" -ge 1
mkdir -p "$REPO/logs/resource_markers"
printf 'validated=%s\nsummaries=%s\nhost=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$summary_count" "$(hostname)" \
  > "$REPO/logs/resource_markers/pi05_public_samebridge_diag.ok"
