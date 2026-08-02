#!/usr/bin/env bash
set -euo pipefail

REPO="${PI05_NATIVE_REPO:-/vePFS/tim/workspace/deepdive_kai0}"
LAWAM="$REPO/lmvla/lawam"
RT="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
RUNTIME_ROOT="${PI05_NATIVE_RUNTIME_ROOT:-$REPO/.runtime/robotwin_native}"
CKPT="$REPO/kai0/checkpoints/pi05_robotwin_a0_official_bj/pi05_robotwin_a0_official/19999"
CFG="$REPO/train_scripts/kai/eval/pi05_robotwin_a0_official_deploy.yml"
RT_PY="$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh"
OPENPI_PY="$REPO/kai0/.venv/bin/python"
SETTING="${PI05_NATIVE_CKPT_SETTING:-a0-official-native-v2}"
TASKS="${PI05_NATIVE_TASKS:-beat_block_hammer stack_blocks_two}"
GPUS="${PI05_NATIVE_GPUS:-0 1}"
PORT_BASE="${PI05_NATIVE_PORT_BASE:-11600}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/lmvla/lawam/logs/official_native_eval"
mkdir -p "$LOG_DIR" "$RUNTIME_ROOT/eval_result"

# The shared RoboTwin checkout is root-owned. Keep its code and assets intact,
# but run from a writable mirror so the native evaluator can create eval_result.
for entry in "$RT"/*; do
  name="$(basename "$entry")"
  [ "$name" = eval_result ] && continue
  ln -sfn "$entry" "$RUNTIME_ROOT/$name"
done

test -d "$CKPT/params"
test -f "$CKPT/assets/robotwin2.0/norm_stats.json"
test -f "$CFG"
test -x "$RT_PY"
test -x "$OPENPI_PY"

source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$RUNTIME_ROOT"

run_task() {
  local task="$1"
  local gpu="$2"
  local port="$3"
  local server_log="$LOG_DIR/${task}_${SETTING}_server_${STAMP}.log"
  local eval_log="$LOG_DIR/${task}_${SETTING}_eval_${STAMP}.log"
  local marker="$LOG_DIR/${task}_${SETTING}_marker_${STAMP}"
  touch "$marker"

  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$OPENPI_PY" "$REPO/kai0/scripts/serve_policy.py" \
      --port "$port" \
      policy:checkpoint \
      --policy.config pi05_robotwin_a0_official_bj \
      --policy.dir "$CKPT" \
      >"$server_log" 2>&1 &
  local server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if "$RT_PY" -c \
      "import socket; s=socket.create_connection(('127.0.0.1',$port),2); s.close()" \
      >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$server_log"
      return 1
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return 1
  fi

  local task_status=0
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  OPENPI_WS_PORT="$port" \
  ROBOTWIN_REPLAN_STEPS=50 \
  PYTHONPATH="$REPO/kai0/packages/openpi-client/src:$LAWAM/examples/Robotwin/eval_files:/vePFS/tim/workspace/robotwin-ws-deps:$RT:${PYTHONPATH:-}" \
    "$RT_PY" "$RT/script/eval_policy.py" \
      --config "$CFG" \
      --overrides \
      --task_name "$task" \
      --task_config demo_clean \
      --train_config_name pi05_robotwin_a0_official_bj \
      --model_name kai0_a0_official_ws \
      --ckpt_setting "$SETTING" \
      --seed 0 \
      --policy_name official_openpi_ws_policy \
      >"$eval_log" 2>&1 || task_status=$?

  if ! find "$RUNTIME_ROOT/eval_result/$task/official_openpi_ws_policy/demo_clean/$SETTING" \
      -type f -name _result.txt -newer "$marker" -print -quit 2>/dev/null | grep -q .; then
    echo "native evaluator exited without a fresh result" >>"$eval_log"
    task_status=1
  fi
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  return "$task_status"
}

read -r -a task_list <<<"$TASKS"
read -r -a gpu_list <<<"$GPUS"
if [[ "${#task_list[@]}" -ne "${#gpu_list[@]}" ]]; then
  echo "PI05_NATIVE_TASKS and PI05_NATIVE_GPUS must have the same length" >&2
  exit 2
fi

pids=()
for i in "${!task_list[@]}"; do
  run_task "${task_list[$i]}" "${gpu_list[$i]}" "$((PORT_BASE + i * 100))" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

result_count="$(find "$RUNTIME_ROOT/eval_result" -path "*/official_openpi_ws_policy/demo_clean/$SETTING/*/_result.txt" -type f | wc -l)"
echo "native pi0.5 A0 protocol eval status=$status results=$result_count stamp=$STAMP"
test "$result_count" -ge "${PI05_NATIVE_MIN_RESULTS:-${#task_list[@]}}"
exit "$status"
