#!/usr/bin/env bash
# FastWAM A1 diagnostic: execute all 48 predicted steps before re-inference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WS_PORT="${KAI0_FASTWAM_FULL_CHUNK_WS_PORT:-8017}"
if pgrep -f "serve_fastwam_ws_isolated.py.*--port ${WS_PORT}([[:space:]]|$)" >/dev/null; then
  SERVER_MODE=(--reuse-server)
  echo "[A1 full-chunk] 检测到 :${WS_PORT} 模型服务，核对配置后复用。"
else
  SERVER_MODE=(--keep-server)
  echo "[A1 full-chunk] 未检测到 :${WS_PORT} 模型服务，首次加载并保留。"
fi

echo "[A1 full-chunk] 🔴 DIRECT EXECUTE"
echo "[A1 full-chunk] 48 步 @30Hz 全部执行 → 队列清空 → 下一次推理。"
echo "[A1 full-chunk] overlap/latency trim/EMA/RTC/speed-resample 均关闭。"

exec "$REPO_ROOT/fastwam/deploy/start_a1.sh" \
  --execute \
  "${SERVER_MODE[@]}" \
  --ws-port "$WS_PORT" \
  --disable-gripper-bootstrap \
  --publish-rate 30 \
  --speed-factor 1.0 \
  --smooth-alpha 1.0 \
  --min-smooth-steps 1 \
  --max-smooth-steps 0 \
  latency_k:=0 \
  enable_rtc:=false \
  proprio_cmd_feedback:=false \
  gripper_offset:=0.0 \
  gripper_close_snap:=false \
  xvla_sequential:=true \
  "$@"
