#!/usr/bin/env bash
# One-command supervised real-robot execution for the FastWAM A1 preset.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WS_PORT="${KAI0_FASTWAM_ISOLATED_WS_PORT:-8016}"
if pgrep -f "serve_fastwam_ws_isolated.py.*--port ${WS_PORT}([[:space:]]|$)" >/dev/null; then
  SERVER_MODE=(--reuse-server)
  echo "[A1 execute] 检测到 :${WS_PORT} 模型服务，核对配置后复用。"
else
  SERVER_MODE=(--keep-server)
  echo "[A1 execute] 未检测到 :${WS_PORT} 模型服务，首次加载并在退出后保留。"
fi

echo "[A1 execute] 🔴 DIRECT EXECUTE：策略栈就绪后机械臂会立即运动。"
echo "[A1 execute] 参数：publish_rate=30Hz, speed_factor=0.5（验证阶段 1）。"

exec "$REPO_ROOT/fastwam/deploy/start_a1.sh" \
  --execute \
  "${SERVER_MODE[@]}" \
  "$@"
