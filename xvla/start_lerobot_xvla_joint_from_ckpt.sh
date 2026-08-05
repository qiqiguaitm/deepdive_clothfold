#!/bin/bash
# Start a LeRobot XVLA checkpoint that emits kai0 14D joint actions.
#
# Usage:
#   ./xvla/start_lerobot_xvla_joint_from_ckpt.sh
#   ./xvla/start_lerobot_xvla_joint_from_ckpt.sh foldDATA_lerobot_xvla_030000
#   ./xvla/start_lerobot_xvla_joint_from_ckpt.sh foldDATA_lerobot_xvla_030000 --execute
#
# Default is observe-only. Add --execute only when the robot is homed and supervised.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_BASE="$REPO_ROOT/xvla/ckpts"
DEFAULT_CKPT_NAME="foldDATA_lerobot_xvla_030000"
PORT="${KAI0_LEROBOT_XVLA_PORT:-8004}"
LOG_DIR="${KAI0_LEROBOT_XVLA_LOG_DIR:-/tmp/lerobot_xvla_joint}"
PY="$REPO_ROOT/kai0/.venv_xvla/bin/python"
TOKENIZER="$REPO_ROOT/xvla/assets/bart-large-tokenizer"

ARGS=()
for a in "$@"; do
  ARGS+=("$a")
done

CKPT_ARG="${ARGS[0]-}"
if [ -n "$CKPT_ARG" ] && [[ "$CKPT_ARG" != --* ]]; then
  shift || true
else
  CKPT_ARG="$DEFAULT_CKPT_NAME"
fi

if [ -d "$CKPT_ARG/pretrained_model" ]; then
  CKPT_ROOT="$(cd "$CKPT_ARG" && pwd)"
elif [ -f "$CKPT_ARG/model.safetensors" ]; then
  CKPT_ROOT="$(cd "$(dirname "$CKPT_ARG")" && pwd)"
elif [ -d "$CKPT_BASE/$CKPT_ARG/pretrained_model" ]; then
  CKPT_ROOT="$CKPT_BASE/$CKPT_ARG"
else
  echo "ERROR: cannot find LeRobot XVLA ckpt '$CKPT_ARG'" >&2
  echo "       expected dir with pretrained_model/ or a name under $CKPT_BASE" >&2
  exit 1
fi
CKPT_DIR="$CKPT_ROOT/pretrained_model"

EXECUTE_FLAG=""
CLIENT_ARGS=()
for a in "$@"; do
  case "$a" in
    --execute) EXECUTE_FLAG="--execute" ;;
    *) CLIENT_ARGS+=("$a") ;;
  esac
done

[ -x "$PY" ] || { echo "ERROR: missing python env: $PY" >&2; exit 1; }
[ -f "$CKPT_DIR/model.safetensors" ] || { echo "ERROR: $CKPT_DIR/model.safetensors missing" >&2; exit 1; }
[ -f "$CKPT_DIR/policy_postprocessor_step_0_unnormalizer_processor.safetensors" ] || {
  echo "ERROR: $CKPT_DIR action unnormalizer stats missing" >&2
  exit 1
}
[ -f "$TOKENIZER/tokenizer.json" ] || { echo "ERROR: tokenizer missing: $TOKENIZER" >&2; exit 1; }

if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
  echo "ERROR: port :$PORT is already in use. Stop the old server or set KAI0_LEROBOT_XVLA_PORT." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server_${PORT}.log"
CLIENT_LOG="$LOG_DIR/client_${PORT}.log"

GPU="${KAI0_LEROBOT_XVLA_GPU_ID:-}"
if [ -z "$GPU" ]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
  GPU="${GPU:-0}"
fi

SERVER_PID=""
CLIENT_PID=""
cleanup() {
  trap - EXIT INT TERM
  echo
  echo "[lerobot-xvla-joint] stopping client/server ..."
  [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null || true
  pkill -f "autonomy_launch|policy_inference_node|multi_camera_node|rerun_viz_node|arm_reader_node" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  pkill -f "serve_policy_lerobot_xvla_joint.py" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  LeRobot XVLA joint stack"
echo "    ckpt:       $CKPT_ROOT"
echo "    server:     :$PORT (action_kind=joint, 14D)"
echo "    gpu:        $GPU"
echo "    execute:    ${EXECUTE_FLAG:-observe-only}"
echo "    server log: $SERVER_LOG"
echo "    client log: $CLIENT_LOG"
echo "============================================================"

: > "$SERVER_LOG"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$REPO_ROOT/xvla/serve/serve_policy_lerobot_xvla_joint.py" \
  --ckpt_dir "$CKPT_DIR" \
  --tokenizer "$TOKENIZER" \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo -n "[lerobot-xvla-joint] waiting for server :$PORT"
for i in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "ERROR: server exited early. Last 40 lines:" >&2
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if ss -ltn 2>/dev/null | grep -q ":$PORT\b"; then
    echo " ready"
    grep -E "warmup|Serving LeRobot XVLA|action_kind" "$SERVER_LOG" | tail -10 || true
    break
  fi
  echo -n "."
  sleep 1
  [ "$i" -eq 180 ] && {
    echo
    echo "ERROR: server did not become ready in 180s. Last 40 lines:" >&2
    tail -40 "$SERVER_LOG" >&2 || true
    exit 1
  }
done

# This launcher starts a WebSocket-only server on $PORT. Using shm would make
# the client wait forever for /dev/shm/kai0_v1_{obs,chunk}, so keep the client
# transport aligned with the server.
echo "[lerobot-xvla-joint] starting kai0 autonomy client"
"$REPO_ROOT/start_scripts/kai/start_autonomy.sh" \
  --mode websocket \
  --ws-port "$PORT" \
  --execution-mode joint \
  $EXECUTE_FLAG \
  "inference_rate:=10.0" \
  "latency_k:=3" \
  "min_smooth_steps:=4" \
  "max_smooth_steps:=8" \
  "rtc_execute_horizon:=8" \
  "publish_rate:=30" \
  "cam_fps:=30" \
  "enable_head_depth:=false" \
  "fast_obs_pipeline:=true" \
  "pipelined_obs:=true" \
  "transport:=websocket" \
  "${CLIENT_ARGS[@]}" \
  >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

wait "$CLIENT_PID"
