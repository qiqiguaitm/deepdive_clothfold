#!/bin/bash
# 本机 gf0 (2×A100) 快速评 pi05 A0: 2 路(GPU0/1, seed0/1), cnsh config + 本地拉取的 a0_bj ckpt.
#   norm 从 ckpt assets 读; LIBERO 本机渲染(MUJOCO_GL=egl). 用法: bash local_eval_a0.sh [NTASKS] [NTRIALS]
set -uo pipefail
REPO=/vePFS/tim/workspace/deepdive_kai0
NTASKS=${1:--1}     # -1=全10任务
NTRIALS=${2:-5}
STAMP=$(date +%Y%m%d_%H%M%S)
CKPT=$REPO/kai0/checkpoints/pi05_libero_a0_bj/pi05_libero_a0/29999
LOG=$REPO/lmvla/lmwm/logs; mkdir -p "$LOG"
OUT=$REPO/lmvla/lmwm/data/pi05_libero_eval/local_a0_${STAMP}
PY=$REPO/kai0/.venv/bin/python
export LIBERO_HOME=$REPO/lmvla/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export PYTHONPATH="$REPO/kai0/src:$REPO/lmvla/lawam:$LIBERO_HOME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false
export EVAL_IMG_FLIP=both EVAL_GRIPPER=inv_pm1 PYTHONUNBUFFERED=1

echo "=== 本地 A0 eval start $(date) tasks=$NTASKS trials=$NTRIALS ==="
pids=""
for i in 0 1; do
  PORT=$((8000 + i))
  ( CUDA_VISIBLE_DEVICES=$i $PY $REPO/kai0/scripts/serve_policy.py --port $PORT \
      policy:checkpoint --policy.config pi05_libero_a0 --policy.dir "$CKPT" \
      > "$LOG/local_server${i}_${STAMP}.log" 2>&1 ) &
  sleep 3
  ( for w in $(seq 1 120); do $PY -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2)" 2>/dev/null && break; sleep 5; done
    CUDA_VISIBLE_DEVICES=$i MUJOCO_EGL_DEVICE_ID=$i $PY $REPO/lmvla/lawam/examples/LIBERO/eval_files/eval_libero_openpi.py \
      --host 127.0.0.1 --port $PORT --task-suite-name libero_10 --num-trials-per-task $NTRIALS --num-tasks $NTASKS \
      --seed $i --replan-steps 5 --output-dir "$OUT/seed${i}" > "$LOG/local_client${i}_${STAMP}.log" 2>&1
  ) &
  pids="$pids $!"
  sleep 8
done
for p in $pids; do wait $p; done
echo "=== 本地 eval 完成, 聚合 $(date) ==="
$PY - "$OUT" <<'PYEOF'
import sys, os, json, glob, statistics
out=sys.argv[1]; per_seed={}
for sd in sorted(glob.glob(os.path.join(out,"seed*"))):
    f=os.path.join(sd,"suites","libero_10","summary.json")
    if os.path.exists(f): per_seed[os.path.basename(sd)]=json.load(open(f))
if not per_seed: print("NO RESULTS"); sys.exit()
tasks=sorted({int(t) for v in per_seed.values() for t in v["per_task_SR"]})
print("=== per-task SR (2 seed 均值) ===")
for t in tasks:
    vals=[v["per_task_SR"].get(str(t), v["per_task_SR"].get(t)) for v in per_seed.values()]
    vals=[x for x in vals if x is not None]
    m=statistics.mean(vals); mark=" <== t6" if t==6 else ""
    print(f"  t{t}: {m*100:.1f}{mark}")
aggs=[v["aggregate_SR"] for v in per_seed.values()]
print(f"=== 聚合 SR: {statistics.mean(aggs)*100:.2f} (n={len(aggs)}) OUT={out} ===")
PYEOF
echo "STAMP=$STAMP OUT=$OUT"
