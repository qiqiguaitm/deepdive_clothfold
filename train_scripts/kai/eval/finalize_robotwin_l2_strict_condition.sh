#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
method="${L2_STRICT_METHOD:?set L2_STRICT_METHOD}"
control="${L2_STRICT_CONTROL:?set L2_STRICT_CONTROL}"
manifest=$repo/lmvla/lmwm/data/robotwin_l2_seed_manifests/${method}_correct_seed2026.json
result_base=$repo/lmvla/lawam/results/eval_runs/robotwin
verify=$repo/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py
marker_dir=$repo/logs/resource_markers

case "$control" in
  zero)
    roots=("$result_base/rt_all6_v2_${method}_zerohint_seed2026_strict_unseen")
    ;;
  cross_task)
    roots=("$result_base/rt_all6_v2_${method}_othertask_seed2026_strict_unseen")
    ;;
  within_task_shuffle)
    roots=(
      "$result_base/rt_all6_v2_${method}_shuffledhint_seed2026_strict_unseen"
      "$result_base/rt_all6_v2_${method}_instanceshuffle_seed2026_strict_unseen"
    )
    ;;
  *)
    echo "unsupported strict L2 control: $control" >&2
    exit 2
    ;;
esac

verify_args=()
for root in "${roots[@]}"; do
  verify_args+=(--root "$root")
done
python3 "$verify" --manifest "$manifest" "${verify_args[@]}"

mkdir -p "$marker_dir"
printf 'completed=%s method=%s control=%s manifest=%s\n' \
  "$(date -u +%FT%TZ)" "$method" "$control" \
  "$(sha256sum "$manifest" | awk '{print $1}')" \
  > "$marker_dir/l2_strict_${method}_${control}.ok"
