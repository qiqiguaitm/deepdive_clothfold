#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ATTACH_SEEDS=${ATTACH_SEEDS:-"2 3"}
read -r -a attach_seed_array <<< "$ATTACH_SEEDS"
ATTACH_GPU_COUNT=${ATTACH_GPU_COUNT:-${#attach_seed_array[@]}}
ATTACH_GPU_INDEX_BASE=${ATTACH_GPU_INDEX_BASE:-0}
WORKER_INDEX_BASE=${WORKER_INDEX_BASE:-4000}
ATTACH_GROUP_NAME=${ATTACH_GROUP_NAME:-platform3}
ATTACH_MARKER_PREFIX=${ATTACH_MARKER_PREFIX:-pi05_a0_seed1000_eval_attach}
ATTACH_RUN_TAG_PREFIX=${ATTACH_RUN_TAG_PREFIX:-}
RESULT_NAME=${RESULT_NAME:-pi05_rt_a0_public_exact_seed1000}
PI05_EVAL_CONFIG_NAME=${PI05_EVAL_CONFIG_NAME:-${ROBOTWIN_EVAL_CONFIG:-pi05_robotwin_a0_public_exact_bj}}
PI05_ASSET_ID=${PI05_ASSET_ID:-robotwin2.0_absolute_meanstd}
CKPT=${CKPT:-$REPO/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/pi05_robotwin_a0_public_exact_seed1000/49999}
LOG_DIR=${ATTACH_LOG_DIR:-$REPO/lmvla/lawam/logs/volc_rteval}
STAMP=$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/${ATTACH_MARKER_PREFIX}_${ATTACH_GROUP_NAME}_${STAMP}.log" 2>&1
set -x

if [ -f "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh" ]; then
  bash "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh"
fi
if [ -f "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh" ]; then
  source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
fi
ROBOTWIN_PY=${ROBOTWIN_PY:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}
[ -x "$ROBOTWIN_PY" ] || { echo "FATAL: RoboTwin wrapper is not executable: $ROBOTWIN_PY" >&2; exit 13; }
"$ROBOTWIN_PY" -c \
  'import sapien.core as sapien; import mplib; sapien.SapienRenderer(); print("SAPIEN_RENDER_OK")'
nvidia-smi -L

pids=""
status=0
[ "$ATTACH_GPU_COUNT" -gt 0 ]
[ "$ATTACH_GPU_COUNT" -le "${#attach_seed_array[@]}" ]
for ((gpu_index=0; gpu_index<ATTACH_GPU_COUNT; gpu_index++)); do
  (
    for ((seed_index=gpu_index; seed_index<${#attach_seed_array[@]}; seed_index+=ATTACH_GPU_COUNT)); do
      seed=${attach_seed_array[$seed_index]}
      env SEED=$seed GPU_INDEX=$((ATTACH_GPU_INDEX_BASE + gpu_index)) RESULT_NAME=$RESULT_NAME \
        ROBOTWIN_ATTACH_RUN_TAG="${ATTACH_RUN_TAG_PREFIX:+${ATTACH_RUN_TAG_PREFIX}${seed}}" \
        PI05_EVAL_CONFIG_NAME=$PI05_EVAL_CONFIG_NAME PI05_ASSET_ID=$PI05_ASSET_ID \
        CKPT=$CKPT \
        WORKER_INDEX_OFFSET=$((WORKER_INDEX_BASE + seed_index * 100)) \
        PORT_BASE_OFFSET=22200 \
        ATTACH_MARKER_NAME=${ATTACH_MARKER_PREFIX}_${ATTACH_GROUP_NAME}_seed${seed} \
        bash "$REPO/train_scripts/kai/eval/attach_pi05_a0_confirmatory_local.sh" \
        > "$LOG_DIR/${ATTACH_MARKER_PREFIX}_${ATTACH_GROUP_NAME}_${STAMP}_seed${seed}.log" 2>&1
    done
  ) &
  pids="$pids $!"
  sleep 15
done
for pid in $pids; do wait "$pid" || status=1; done

if [ "$status" -eq 0 ]; then
  mkdir -p "$REPO/logs/resource_markers"
  printf 'completed=%s\nseeds=%s\n' "$(date -u +%FT%TZ)" "$ATTACH_SEEDS" \
    > "$REPO/logs/resource_markers/${ATTACH_MARKER_PREFIX}_${ATTACH_GROUP_NAME}.ok"
fi
exit "$status"
