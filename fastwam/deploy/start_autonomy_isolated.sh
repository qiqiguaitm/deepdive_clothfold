#!/usr/bin/env bash
# FastWAM 真机隔离部署入口。
#
# 只使用 fastwam/scripts/serve_fastwam_ws_isolated.py 的一次性夹爪启动后处理；
# 不修改原 FastWAM 服务、kai0/dagger 启动脚本或共享 policy_inference_node。
#
# 架构(与 gwp 一致,仅 ws-port 后面的模型不同):
#   FastWAM 推理 = 独立 venv(gwp_eval_env)的 openpi-WebSocket server(serve_fastwam_ws.py, opt infer_action ~90ms);
#   控制 = 现有 kai0 policy_inference_node + start_autonomy.sh(--mode websocket 连该 port),继承全套控制参数。
#   FastWAM 的 action expert 只读首帧 KV(不 rollout 视频)→ 天然回避 gwp_ans 的闭环视频塌缩。
#
# 用法:
#   ./fastwam/deploy/start_autonomy_isolated.sh --server-gpu 2
#   ./fastwam/deploy/start_autonomy_isolated.sh --check-only
#   python3 fastwam/deploy/isolated_control.py on
#   python3 fastwam/deploy/isolated_control.py off
#   旋钮: --nfe 4(去噪步) --opt-tier exact|fp8 --inference-rate 10 --debug-dump DIR
#         --gripper-lookahead 24 --gripper-bootstrap-prefix 8
#         --gripper-open-trigger 0.01 --gripper-open-confirm 0.015
#         --gripper-close-trigger 0.005 --gripper-no-close-warn-after 120
#         --debug-dump-n 200
#         --speed-factor 0.50 --smooth-alpha 0.3 --max-smooth-steps 12
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

SERVER_GPU="${KAI0_FASTWAM_ISOLATED_SERVER_GPU:-2}"
WS_PORT="${KAI0_FASTWAM_ISOLATED_WS_PORT:-8016}"
NFE=4; OPT_TIER="exact"
INFER_RATE=8; EXEC_HORIZON=8; PUBLISH_RATE=30
# 共享 policy_inference_node 会把低于 0.50 的值钳制到 0.50。隔离入口显式使用
# 真实可生效下限，避免终端显示 0.35、硬件实际却跑 0.50 的危险错觉。
SPEED_FACTOR=0.50; SMOOTH_ALPHA=0.3; MIN_SMOOTH_STEPS=8; MAX_SMOOTH_STEPS=12
EXECUTE_FLAG=""
RECORD_ENABLE=false
CHECK_ONLY=false
KEEP_SERVER=false
REUSE_SERVER=false
SERVER_READY_TIMEOUT=240
# obs dump 默认开: 2026-07-28 复盘时因为没有 dump, "夹爪指令到底有没有要求张开"这个
# 关键问题无法从 log 判定 (log 只打 act[0])。存 ref_*.png + io_*.npz (完整 [48,14] chunk)
# 成本可忽略 (前 15 次推理), 但事后能一眼定位。--debug-dump '' 可关。
DEBUG_DUMP="log/fastwam_isolated_obs_dump"; DEBUG_DUMP_N=200; GRIP_NEUTRAL=""
GRIP_LOOKAHEAD=24; GRIP_OPEN_TRIGGER=0.01
GRIP_BOOTSTRAP_PREFIX=8; GRIP_OPEN_CONFIRM=0.015
GRIP_CLOSE_TRIGGER=0.005; GRIP_NO_CLOSE_WARN_AFTER=120
DISABLE_GRIP_BOOTSTRAP=false
FW_VENV_PY="${FW_VENV_PY:-/home/tim/gwp_eval_env/venv/bin/python}"
FW_REPO="${FW_REPO:-$PWD/fastwam}"
WEIGHTS="${FASTWAM_WEIGHTS:-$FW_REPO/runs/visrobot01_v3_fold_1e-4/aihc_5n8g_v6/checkpoints/weights/step_050000.pt}"
STATS="${FASTWAM_STATS:-$FW_REPO/data/visrobot01_v3_fold/dataset_stats.json}"
EVAL_DATA_NAME="${FASTWAM_EVAL_DATA:-visrobot01_v3_fold}"
EVAL_TASK_NAME="${FASTWAM_EVAL_TASK:-visrobot01_v3_fold_1e-4}"
T5_CACHE="${FASTWAM_T5_CACHE:-$FW_REPO/data/text_embeds_cache/visrobot01_v3_fold/*.pt}"
PREFLIGHT_REF="${FASTWAM_PREFLIGHT_REF:-$PWD/fastwam/deploy/refs/visrobot01_v3.json}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-gpu)     SERVER_GPU="$2"; shift 2 ;;
    --ws-port)        WS_PORT="$2"; shift 2 ;;
    --nfe)            NFE="$2"; shift 2 ;;
    --opt-tier)       OPT_TIER="$2"; shift 2 ;;
    --inference-rate) INFER_RATE="$2"; shift 2 ;;
    --exec-horizon)   EXEC_HORIZON="$2"; shift 2 ;;
    --publish-rate)   PUBLISH_RATE="$2"; shift 2 ;;
    --speed-factor)   SPEED_FACTOR="$2"; shift 2 ;;
    --smooth-alpha)   SMOOTH_ALPHA="$2"; shift 2 ;;
    --min-smooth-steps) MIN_SMOOTH_STEPS="$2"; shift 2 ;;
    --max-smooth-steps) MAX_SMOOTH_STEPS="$2"; shift 2 ;;
    --gripper-lookahead) GRIP_LOOKAHEAD="$2"; shift 2 ;;
    --gripper-open-trigger) GRIP_OPEN_TRIGGER="$2"; shift 2 ;;
    --gripper-bootstrap-prefix) GRIP_BOOTSTRAP_PREFIX="$2"; shift 2 ;;
    --gripper-open-confirm) GRIP_OPEN_CONFIRM="$2"; shift 2 ;;
    --gripper-close-trigger) GRIP_CLOSE_TRIGGER="$2"; shift 2 ;;
    --gripper-no-close-warn-after) GRIP_NO_CLOSE_WARN_AFTER="$2"; shift 2 ;;
    --disable-gripper-bootstrap) DISABLE_GRIP_BOOTSTRAP=true; shift ;;
    --weights)        WEIGHTS="$2"; shift 2 ;;
    --stats)          STATS="$2"; shift 2 ;;
    --eval-data)      EVAL_DATA_NAME="$2"; shift 2 ;;
    --eval-task)      EVAL_TASK_NAME="$2"; shift 2 ;;
    --t5-cache)       T5_CACHE="$2"; shift 2 ;;
    --preflight-ref)  PREFLIGHT_REF="$2"; shift 2 ;;
    --debug-dump)     DEBUG_DUMP="$2"; shift 2 ;;
    --debug-dump-n)   DEBUG_DUMP_N="$2"; shift 2 ;;
    # 夹爪闩锁缓解: 仅当夹爪 proprio 卡死(硬件不动作)时才用。见 serve_fastwam_ws.py infer()
    # 内注释 + 2026-07-28 复盘。离线最优 0.04; 夹爪硬件正常时【不要开】(真实 proprio 更好)。
    --gripper-neutral) GRIP_NEUTRAL="$2"; shift 2 ;;
    --execute)        EXECUTE_FLAG="--execute"; shift ;;
    --no-execute)     EXECUTE_FLAG=""; shift ;;
    --record)         RECORD_ENABLE=true; shift ;;
    --no-record)      RECORD_ENABLE=false; shift ;;
    --check-only)     CHECK_ONLY=true; shift ;;
    --keep-server)    KEEP_SERVER=true; shift ;;
    --reuse-server)   REUSE_SERVER=true; KEEP_SERVER=true; shift ;;
    --server-ready-timeout) SERVER_READY_TIMEOUT="$2"; shift 2 ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done
case "$INFER_RATE" in *.*) ;; *) INFER_RATE="${INFER_RATE}.0" ;; esac   # 节点声明 DOUBLE, 必须带小数
if ! awk -v value="$SPEED_FACTOR" 'BEGIN { exit !(value >= 0.50) }'; then
  echo "ERROR: --speed-factor=${SPEED_FACTOR} 不会真实生效；共享控制器下限是 0.50。"
  echo "       隔离入口拒绝启动，避免显示值与实际速度不一致。"
  exit 2
fi

# 控制参数(与 gwp/kai0 一致);enable_rtc=false(FastWAM 不消费 RTC);obs_image 近原生(server 侧拼 384x320)
CTRL_ARGS=( "latency_k:=6" "min_smooth_steps:=${MIN_SMOOTH_STEPS}" "max_smooth_steps:=${MAX_SMOOTH_STEPS}"
            "publish_smooth_alpha:=${SMOOTH_ALPHA}" "speed_factor:=${SPEED_FACTOR}"
            "enable_rtc:=false" "proprio_cmd_feedback:=false"
            "cam_fps:=30" "fast_obs_pipeline:=true" "obs_image_h:=480" "obs_image_w:=640"
            "inference_rate:=${INFER_RATE}" "rtc_execute_horizon:=${EXEC_HORIZON}" "publish_rate:=${PUBLISH_RATE}"
            "record_enable:=${RECORD_ENABLE}" )

echo "=========================================================="
echo " FastWAM ISOLATED autonomy: nfe=${NFE} tier=${OPT_TIER}"
echo " 控制: infer_rate=${INFER_RATE}Hz publish_rate=${PUBLISH_RATE}Hz speed=${SPEED_FACTOR}x"
echo " 平滑: min/max=${MIN_SMOOTH_STEPS}/${MAX_SMOOTH_STEPS} EMA-alpha=${SMOOTH_ALPHA} exec_horizon=${EXEC_HORIZON}"
echo " ws server: GPU${SERVER_GPU} :${WS_PORT}  execute=${EXECUTE_FLAG:-observe-only}"
echo " 模型配置: data=${EVAL_DATA_NAME} task=${EVAL_TASK_NAME}"
echo " 权重/统计: ${WEIGHTS} / ${STATS}"
echo " 文本缓存: ${T5_CACHE}"
echo " 起始位参考: ${PREFLIGHT_REF}"
echo " 夹爪启动: initial-only lookahead=${GRIP_LOOKAHEAD} prefix=${GRIP_BOOTSTRAP_PREFIX}"
echo " 夹爪 bootstrap: $($DISABLE_GRIP_BOOTSTRAP && echo OFF/raw-model || echo ON)"
echo "           trigger=${GRIP_OPEN_TRIGGER}m confirm=${GRIP_OPEN_CONFIRM}m；确认张开后永久透传原指令"
echo " 夹爪诊断: close<=${GRIP_CLOSE_TRIGGER}m；连续 ${GRIP_NO_CLOSE_WARN_AFTER} chunks 无闭合预测则告警"
echo " obs dump: ${DEBUG_DUMP:-<off>} (前 ${DEBUG_DUMP_N} chunks)   recorder=${RECORD_ENABLE}"
echo " 夹爪 proprio 中性化: ${GRIP_NEUTRAL:-<off>}"
echo "----------------------------------------------------------"
echo " ⚠️  每次新的真机 trial 必须先闭合夹爪回到起始位。新 WebSocket 会话会"
echo "     自动复位一次性夹爪状态；--reuse-server 只复用权重/编译图，不复用 trial 状态。"
echo "----------------------------------------------------------"
echo " ⚠️  上真机前 (翻 /policy/execute 之前) 请在【另一个终端】跑前置体检:"
echo "       cd $PWD && source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash"
  echo "       python3 fastwam/deploy/preflight.py --ref $PREFLIGHT_REF"
echo "     它只读不驱动, 检查 3 项 (相机分辨率/letterbox · 夹爪闩锁风险 · 起始位偏离)。"
echo "     体检需要相机+臂 topic 已在跑, 所以必须等本脚本把栈起起来之后再跑。"
echo "=========================================================="
if [[ -n "$EXECUTE_FLAG" ]]; then
  echo " 🔴 --execute 已开: 手臂会立即运动。建议使用 observe-only 起栈 → 跑 preflight"
  echo "    → 再用 fastwam_isolated_control.py on/off（带订阅者确认与动作流回执）。"
  echo "=========================================================="
fi
SERVER_SCRIPT="$FW_REPO/scripts/serve_fastwam_ws_isolated.py"
CONTROL_SCRIPT="$PWD/fastwam/deploy/isolated_control.py"
for p in "$FW_VENV_PY" "$WEIGHTS" "$STATS" "$PREFLIGHT_REF" "$SERVER_SCRIPT" "$CONTROL_SCRIPT"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done
for p in "$FW_REPO/configs/data/${EVAL_DATA_NAME}.yaml" "$FW_REPO/configs/task/${EVAL_TASK_NAME}.yaml"; do
  [[ -e "$p" ]] || { echo "ERROR: missing $p"; exit 3; }
done
T5_CACHE_FIRST="$(compgen -G "$T5_CACHE" | head -n 1)"
[[ -n "$T5_CACHE_FIRST" ]] || { echo "ERROR: no T5 cache matches $T5_CACHE"; exit 3; }
if $CHECK_ONLY; then
  "$FW_VENV_PY" -m py_compile "$SERVER_SCRIPT"
  python3 -m py_compile "$CONTROL_SCRIPT"
  echo "[check-only] PASS: 路径、权重、统计文件与隔离 Python 脚本检查通过。"
  echo "[check-only] 未启动模型、相机、ROS、机械臂或任何执行话题。"
  exit 0
fi

SERVER_PID=""
SERVER_STARTED=false
SERVER_LOG="log/fastwam_isolated_server.log"
cleanup() {
  if $SERVER_STARTED && ! $KEEP_SERVER && [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ---- 1. FastWAM openpi-WebSocket server ----
mkdir -p log
if $REUSE_SERVER; then
  SERVER_PID="$(pgrep -f "serve_fastwam_ws_isolated.py.*--port ${WS_PORT}([[:space:]]|$)" | head -n 1 || true)"
  if [[ -z "$SERVER_PID" || ! -r "/proc/$SERVER_PID/cmdline" ]]; then
    echo "ERROR: --reuse-server requested but no server is running on port $WS_PORT"
    exit 4
  fi
  SERVER_CMD="$(tr '\0' ' ' < "/proc/$SERVER_PID/cmdline")"
  for expected in "$WEIGHTS" "$STATS" "$T5_CACHE"; do
    [[ "$SERVER_CMD" == *"$expected"* ]] || {
      echo "ERROR: running server pid=$SERVER_PID does not match requested artifact: $expected"
      exit 4
    }
  done
  if $DISABLE_GRIP_BOOTSTRAP; then
    [[ "$SERVER_CMD" == *"--disable_gripper_bootstrap"* ]] || {
      echo "ERROR: running server pid=$SERVER_PID has gripper bootstrap enabled; refusing raw-mode reuse"
      exit 4
    }
  elif [[ "$SERVER_CMD" == *"--disable_gripper_bootstrap"* ]]; then
    echo "ERROR: running server pid=$SERVER_PID is raw gripper mode; refusing bootstrap-mode reuse"
    exit 4
  fi
  "$FW_VENV_PY" - "$WS_PORT" <<'PY' || {
import sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as response:
    if response.status != 200:
        raise SystemExit(f"health check returned {response.status}")
PY
    echo "ERROR: existing server pid=$SERVER_PID failed /healthz"
    exit 4
  }
  echo "[fastwam] reusing ready server pid=$SERVER_PID on :$WS_PORT (artifacts matched)"
else
  echo "[fastwam] starting ws server (first load/compile/warmup may take 1-2min)..."
  : > "$SERVER_LOG"
  ( cd "$FW_REPO" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
    PYTHONPATH=src:scripts EVAL_DATA="$EVAL_DATA_NAME" EVAL_TASK="$EVAL_TASK_NAME" TORCHINDUCTOR_CACHE_DIR=/data2/gwp_eval/.inductor_fastwam_v6 \
    "$FW_VENV_PY" scripts/serve_fastwam_ws_isolated.py \
      --weights "$WEIGHTS" --stats "$STATS" --t5_cache "$T5_CACHE" --nfe "$NFE" --opt_tier "$OPT_TIER" \
      --port "$WS_PORT" --warmup 2 \
      --gripper_open_lookahead "$GRIP_LOOKAHEAD" \
      --gripper_open_trigger "$GRIP_OPEN_TRIGGER" \
      --gripper_bootstrap_prefix "$GRIP_BOOTSTRAP_PREFIX" \
      --gripper_open_confirm "$GRIP_OPEN_CONFIRM" \
      --gripper_close_trigger "$GRIP_CLOSE_TRIGGER" \
      --gripper_no_close_warn_after "$GRIP_NO_CLOSE_WARN_AFTER" \
      $($DISABLE_GRIP_BOOTSTRAP && echo --disable_gripper_bootstrap) \
      --debug_dump_n "$DEBUG_DUMP_N" \
      ${DEBUG_DUMP:+--debug_dump_dir "$DEBUG_DUMP"} \
      ${GRIP_NEUTRAL:+--gripper_proprio_neutral "$GRIP_NEUTRAL"} \
  ) >> "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED=true
  echo "[fastwam] isolated server pid=$SERVER_PID, log: $SERVER_LOG"

  echo -n "[fastwam] waiting for server ready"
  WAIT_TICKS=$(( SERVER_READY_TIMEOUT * 2 ))
  for i in $(seq 1 "$WAIT_TICKS"); do
    grep -q "ready, listening" "$SERVER_LOG" 2>/dev/null && { echo " OK"; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo; echo "ERROR: server died"; tail -20 "$SERVER_LOG"; exit 4; }
    (( i % 10 == 0 )) && echo -n " $((i / 2))s" || echo -n "."
    sleep 0.5
    [[ $i -eq "$WAIT_TICKS" ]] && {
      echo; echo "ERROR: server not ready after ${SERVER_READY_TIMEOUT}s"; tail -20 "$SERVER_LOG"; exit 4;
    }
  done
fi

# ---- 2. cameras + arms + kai0 policy_inference_node (websocket -> fastwam server) ----
echo "[fastwam] launching kai0 autonomy stack (websocket :$WS_PORT)..."
# 不 exec: 保留 cleanup trap, Ctrl-C 时杀 fastwam server 防孤儿残留
./start_scripts/kai/start_autonomy.sh \
    --mode websocket --ws-port "$WS_PORT" --execution-mode joint $EXECUTE_FLAG \
    "${CTRL_ARGS[@]}" "${EXTRA_ARGS[@]}"

if $KEEP_SERVER; then
  echo "[fastwam] model server kept alive as pid=$SERVER_PID; next trial add --reuse-server"
fi
