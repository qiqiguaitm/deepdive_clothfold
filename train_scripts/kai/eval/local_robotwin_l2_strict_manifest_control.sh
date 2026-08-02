#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
method="${L2_STRICT_METHOD:?set L2_STRICT_METHOD}"
control="${L2_STRICT_CONTROL:?set L2_STRICT_CONTROL}"
manifest=$repo/lmvla/lmwm/data/robotwin_l2_seed_manifests/${method}_correct_seed2026.json
base_eval=$repo/train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh
instance_eval=$repo/train_scripts/kai/eval/local_robotwin_l2_missing3_instance_shuffle.sh
verify=$repo/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
result_base=$repo/lmvla/lawam/results/eval_runs/robotwin
marker_dir=$repo/logs/resource_markers

case "$method" in
  absolute) run_basename=20260730_063314+robotwin_all6_v2_absolute_seed2026; hint_form=abs ;;
  residual) run_basename=20260730_164555+robotwin_all6_v2_residual_seed2026; hint_form=resid ;;
  combo) run_basename=20260730_152020+robotwin_all6_v2_combo_seed2026; hint_form=resid ;;
  *) echo "unsupported strict L2 method: $method" >&2; exit 2 ;;
esac

test -s "$manifest"
test -s "$repo/lmvla/lawam/results/Checkpoints/robotwin/$run_basename/final_model/pytorch_model.pt"
export ROBOTWIN_EPISODE_SEED_MANIFEST=$manifest
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS="${ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS:-5}"
common=(
  ALL6_EVAL_VARIANT="$method"
  ALL6_EVAL_RUN_BASENAME="$run_basename"
  ALL6_TRAIN_SEED=2026
  SEEDS="${SEEDS:-0 1 2 3}"
  GPUS_PER_SEED="${GPUS_PER_SEED:-1}"
  LOCAL_GPU_COUNT="${LOCAL_GPU_COUNT:-4}"
  NUM_WORKERS="${NUM_WORKERS:-1}"
  ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-1}"
  ROBOTWIN_TEST_NUM=50
)

roots=()
case "$control" in
  zero)
    result=rt_all6_v2_${method}_zerohint_seed2026_strict_unseen
    env -u LMWM_SWAP_HINT -u LMWM_SWAP_HINT_SHUFFLE \
      "${common[@]}" LMWM_SWAP_HINT_ZERO=1 RESULT_NAME="$result" \
      RUN_TAG_PREFIX="strict-${method}-zero" bash "$base_eval"
    roots+=("$result_base/$result")
    ;;
  cross_task)
    result=rt_all6_v2_${method}_othertask_seed2026_strict_unseen
    hint=$repo/lmvla/lmwm/data/swap_hint_probe_robotwin/other_${hint_form}.npy
    test -s "$hint"
    env -u LMWM_SWAP_HINT_ZERO -u LMWM_SWAP_HINT_SHUFFLE \
      "${common[@]}" LMWM_SWAP_HINT="$hint" RESULT_NAME="$result" \
      RUN_TAG_PREFIX="strict-${method}-cross" bash "$base_eval"
    roots+=("$result_base/$result")
    ;;
  within_task_shuffle)
    spatial=rt_all6_v2_${method}_shuffledhint_seed2026_strict_unseen
    env -u LMWM_SWAP_HINT -u LMWM_SWAP_HINT_ZERO \
      "${common[@]}" ROBOTWIN_TASKS="beat_block_hammer blocks_ranking_rgb stack_blocks_two" \
      LMWM_SWAP_HINT_SHUFFLE=1 RESULT_NAME="$spatial" \
      RUN_TAG_PREFIX="strict-${method}-spatial" bash "$base_eval"
    instance=rt_all6_v2_${method}_instanceshuffle_seed2026_strict_unseen
    env -u LMWM_SWAP_HINT_ZERO -u LMWM_SWAP_HINT_SHUFFLE \
      "${common[@]}" RESULT_NAME="$instance" \
      RUN_TAG_PREFIX="strict-${method}-instance" bash "$instance_eval"
    roots+=("$result_base/$spatial" "$result_base/$instance")
    ;;
  *) echo "unsupported strict L2 control: $control" >&2; exit 2 ;;
esac

verify_args=()
for root in "${roots[@]}"; do verify_args+=(--root "$root"); done
python3 "$verify" --manifest "$manifest" "${verify_args[@]}"
mkdir -p "$marker_dir"
printf 'completed=%s method=%s control=%s manifest=%s\n' \
  "$(date -u +%FT%TZ)" "$method" "$control" "$(sha256sum "$manifest" | awk '{print $1}')" \
  > "$marker_dir/l2_strict_${method}_${control}.ok"
