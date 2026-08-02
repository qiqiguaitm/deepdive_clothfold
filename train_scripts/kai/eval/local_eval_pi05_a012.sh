#!/bin/bash
# 本机 gf0 重评 pi05 A0/A1/A2(在线 hint + gripper inv_pm1 修法), libero_10.
# A0 基线(无hint) / A1 DINOv3 hint / A2 so400m hint. 用 local_eval_hint.py(在线 HintComputer)。
set -uo pipefail
REPO=/vePFS/tim/workspace/deepdive_kai0
TRIALS=${1:-5}
STAMP=$(date +%Y%m%d_%H%M%S)
PY=$REPO/kai0/.venv/bin/python
LOG=$REPO/lmvla/lmwm/logs; mkdir -p "$LOG"
export LIBERO_HOME=$REPO/lmvla/LIBERO
export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
export PYTHONPATH="$REPO/kai0/src:$REPO/lmvla/lawam:$LIBERO_HOME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
cd $REPO
run(){  # arm cfg enc ckpt gpu
  local arm=$1 cfg=$2 enc=$3 ckpt=$4 gpu=$5
  local out=$REPO/lmvla/lmwm/data/pi05_libero_eval/reeval_${arm}_${STAMP}
  echo "[$arm] cfg=$cfg enc=$enc gpu=$gpu -> $out"
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu $PY $REPO/lmvla/lawam/examples/LIBERO/eval_files/local_eval_hint.py \
    --config "$cfg" --ckpt "$ckpt" --encoder "$enc" --tasks -1 --trials $TRIALS --seed 0 --replan 5 --gpu $gpu \
    --out "$out" > "$LOG/reeval_${arm}_${STAMP}.log" 2>&1
  echo "[$arm] done rc=$?"
}
# A0 (gpu0) + A1 (gpu1) 并行
run A0 pi05_libero_a0 none $REPO/kai0/checkpoints/pi05_libero_a0_bj/pi05_libero_a0/29999 0 &
P0=$!
run A1 pi05_libero_a1_prefix_eval dinov3-base $REPO/kai0/checkpoints/pi05_libero_a1_prefix_bj/pi05_libero_a1_prefix/29999 1 &
P1=$!
wait $P0 $P1
# A2 (gpu0) 之后
run A2 pi05_libero_a2_prefix_eval so400m $REPO/kai0/checkpoints/pi05_libero_a2_prefix_bj/pi05_libero_a2_prefix/29999 0
echo "=== 三臂重评完成 STAMP=$STAMP; 聚合 ==="
$PY - "$STAMP" <<'PYEOF'
import sys,glob,json,os,statistics as st
stamp=sys.argv[1]; base="/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/pi05_libero_eval"
for arm in ["A0","A1","A2"]:
    fs=glob.glob(f"{base}/reeval_{arm}_{stamp}/**/summary.json",recursive=True)
    sr=[]
    for f in fs:
        d=json.load(open(f)); x=d.get("aggregate_SR") or d.get("total_success_rate")
        if x is not None: sr.append(100*float(x) if x<=1 else float(x))
    print(f"  {arm}: {[round(x,1) for x in sr]} 均值={st.mean(sr):.2f}" if sr else f"  {arm}: 无结果")
print("PI05_REEVAL_DONE")
PYEOF
