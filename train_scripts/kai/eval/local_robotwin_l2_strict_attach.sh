#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
lawam=$repo/lmvla/lawam
method="${L2_STRICT_METHOD:?set L2_STRICT_METHOD}"
control="${L2_STRICT_CONTROL:?set L2_STRICT_CONTROL}"
offset="${ROBOTWIN_WORKER_INDEX_OFFSET:-16000}"
base_eval=$repo/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh
manifest=$repo/lmvla/lmwm/data/robotwin_l2_seed_manifests/${method}_correct_seed2026.json
PY="${ROBOTWIN_PYTHON:-python3}"

case "$method" in
  absolute)
    run_basename=20260730_063314+robotwin_all6_v2_absolute_seed2026
    hint_form=abs
    ;;
  combo)
    run_basename=20260730_152020+robotwin_all6_v2_combo_seed2026
    hint_form=resid
    ;;
  residual)
    run_basename=20260730_164555+robotwin_all6_v2_residual_seed2026
    hint_form=resid
    ;;
  *) echo "unsupported strict attach method: $method" >&2; exit 2 ;;
esac

export ROBOTWIN_EPISODE_SEED_MANIFEST=$manifest
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS="${ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS:-50}"
common=(
  ALL6_EVAL_VARIANT="$method"
  ALL6_EVAL_RUN_BASENAME="$run_basename"
  ALL6_TRAIN_SEED=2026
  SEEDS="${SEEDS:-0 1}"
  GPUS_PER_SEED="${GPUS_PER_SEED:-1}"
  LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-2}"
  GPU_INDEX_OFFSET="${GPU_INDEX_OFFSET:-0}"
  NUM_WORKERS=1
  ROBOTWIN_NUM_SLOTS=1
  ROBOTWIN_TEST_NUM=50
  ROBOTWIN_ATTACH_EXISTING=1
  ROBOTWIN_ATTACH_REQUEUE_FAILED=1
  ROBOTWIN_WORKER_INDEX_OFFSET="$offset"
)
requested_tasks="${ROBOTWIN_TASKS:-beat_block_hammer blocks_ranking_size blocks_ranking_rgb handover_block stack_blocks_two stack_blocks_three}"

case "$control" in
  zero)
    result=rt_all6_v2_${method}_zerohint_seed2026_strict_unseen
    env -u LMWM_SWAP_HINT -u LMWM_SWAP_HINT_SHUFFLE \
      "${common[@]}" LMWM_SWAP_HINT_ZERO=1 RESULT_NAME="$result" \
      RUN_TAG_PREFIX="strict-attach-${method}-zero" bash "$base_eval"
    ;;
  cross_task)
    result=rt_all6_v2_${method}_othertask_seed2026_strict_unseen
    hint=$repo/lmvla/lmwm/data/swap_hint_probe_robotwin/other_${hint_form}.npy
    test -s "$hint"
    env -u LMWM_SWAP_HINT_ZERO -u LMWM_SWAP_HINT_SHUFFLE \
      "${common[@]}" LMWM_SWAP_HINT="$hint" RESULT_NAME="$result" \
      RUN_TAG_PREFIX="strict-attach-${method}-cross" bash "$base_eval"
    ;;
  spatial_shuffle)
    result=rt_all6_v2_${method}_shuffledhint_seed2026_strict_unseen
    requested_tasks="beat_block_hammer blocks_ranking_rgb stack_blocks_two"
    env -u LMWM_SWAP_HINT -u LMWM_SWAP_HINT_ZERO \
      "${common[@]}" ROBOTWIN_TASKS="$requested_tasks" \
      LMWM_SWAP_HINT_SHUFFLE=1 RESULT_NAME="$result" \
      RUN_TAG_PREFIX="strict-attach-${method}-spatial" bash "$base_eval"
    ;;
  *) echo "unsupported strict attach control: $control" >&2; exit 2 ;;
esac

# An attach worker may legitimately exit while another worker owns the remaining
# cells, but it must never publish success after one of its requested cells
# entered the terminal failed bucket.
for seed in ${SEEDS:-0 1}; do
  eval_root=$lawam/results/eval_runs/robotwin/$result/seed$seed
  mapfile -t scheduler_files < <(find "$eval_root" -type f -name .task_scheduler.json)
  [ "${#scheduler_files[@]}" -eq 1 ] || {
    echo "postcondition requires exactly one scheduler under $eval_root; found ${#scheduler_files[@]}" >&2
    exit 4
  }
  "$PY" - "${scheduler_files[0]}" $requested_tasks <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
requested = set(sys.argv[2:])
failed = sorted(requested.intersection(state.get("failed", {})))
pending = sorted(requested.intersection(state.get("pending", [])))
if failed:
    raise SystemExit(f"attach postcondition failed; terminal failed tasks: {failed}")
if pending:
    raise SystemExit(f"attach postcondition failed; pending tasks remain: {pending}")
PY
done

mkdir -p "$repo/logs/resource_markers"
printf 'completed=%s method=%s control=%s\n' \
  "$(date -u +%FT%TZ)" "$method" "$control" \
  > "$repo/logs/resource_markers/${ATTACH_MARKER_NAME:-l2_strict_attach_${method}_${control}}.ok"
