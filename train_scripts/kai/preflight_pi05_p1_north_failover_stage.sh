#!/usr/bin/env bash
set -Eeuo pipefail

STAGE_ROOT=${STAGE_ROOT:?set STAGE_ROOT to the verified failover stage}
RUNTIME_REPO=${RUNTIME_REPO:-/vePFS-North-E/vis_robot/workspace/deepdive_kai0}
PYTHON=${PYTHON:-$RUNTIME_REPO/kai0/.venv/bin/python}
REPORT=${REPORT:-$STAGE_ROOT/pi05_p1_north_runtime_preflight.json}
STAGE_REPORT=${STAGE_REPORT:-$STAGE_ROOT/north_stage_report.json}
MANIFEST=$STAGE_ROOT/lmvla/paper_iclr_lmvla/manifests/pi05_p1_north_failover_stage_v1.json
AUDIT=$STAGE_ROOT/lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p1_baseline_audit.json
DATA_REPO=$STAGE_ROOT/datasets/robotwin2.0_official_prompts_v21
NORM_ASSETS=$STAGE_ROOT/kai0/assets/pi05_robotwin_a0_public_exact_bj
BASE_PARAMS=$STAGE_ROOT/kai0/checkpoints/pi05_base/params
P0_ADAPTER=$STAGE_ROOT/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999
PAIRS=$STAGE_ROOT/lmvla/lmwm/data/pi05_predictive_adapter_p0_v1/pairs.npz
FRAME_CACHE=$STAGE_ROOT/lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256
PREFLIGHT_DIR=$STAGE_ROOT/logs/pi05_p1_failover/runtime_preflight

test -x "$PYTHON"
test "$(jq -r '.stage_verified' "$STAGE_REPORT")" = true
test "$(jq -r '.launch_authorized' "$MANIFEST")" = false
mkdir -p "$PREFLIGHT_DIR" "$(dirname "$REPORT")"

export PYTHONPATH=$STAGE_ROOT/kai0/src
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENPI_DATA_HOME=$STAGE_ROOT/openpi_cache

"$PYTHON" "$STAGE_ROOT/kai0/scripts/verify_pi05_predictive_adapter_source_freeze.py" \
  --repo "$STAGE_ROOT" \
  --audit "$AUDIT" \
  --output "$PREFLIGHT_DIR/source_freeze.json" >/dev/null

COMMON_ARGS=(
  --seed 1000
  --data-repo "$DATA_REPO"
  --init-params "$BASE_PARAMS"
  --norm-assets-dir "$NORM_ASSETS"
  --num-train-steps 50000
  --save-interval 5000
  --num-workers 8
  --fsdp-devices 1
  --resume
  --dry-run
)

"$PYTHON" "$STAGE_ROOT/kai0/scripts/train_pi05_robotwin_confirmatory.py" \
  --arm a0 \
  --config-name pi05_predictive_adapter_p1_a0_exact \
  --exp-name pi05_predictive_adapter_p1_a0_seed1000 \
  "${COMMON_ARGS[@]}" > "$PREFLIGHT_DIR/a0_dry_run.json"

"$PYTHON" "$STAGE_ROOT/kai0/scripts/train_pi05_robotwin_confirmatory.py" \
  --arm p1_predictive \
  --config-name pi05_predictive_adapter_p1 \
  --exp-name pi05_predictive_adapter_p1_seed1000 \
  --adapter-checkpoint "$P0_ADAPTER" \
  --target-pairs "$PAIRS" \
  --frame-cache-root "$FRAME_CACHE" \
  "${COMMON_ARGS[@]}" > "$PREFLIGHT_DIR/candidate_dry_run.json"

source "$STAGE_ROOT/train_scripts/kai/lib/checkpoint_resume.sh"
A0_RESUME=()
CANDIDATE_RESUME=()
checkpoint_resume_args \
  "$STAGE_ROOT/kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact/pi05_predictive_adapter_p1_a0_seed1000" \
  A0_RESUME
checkpoint_resume_args \
  "$STAGE_ROOT/kai0/checkpoints/pi05_predictive_adapter_p1/pi05_predictive_adapter_p1_seed1000" \
  CANDIDATE_RESUME
test "${A0_RESUME[*]}" = --resume
test "${CANDIDATE_RESUME[*]}" = --resume

jq -e --arg root "$STAGE_ROOT" '
  .name == "pi05_predictive_adapter_p1_a0_exact"
  and .seed == 1000
  and .training.batch_size == 16
  and .training.num_train_steps == 50000
  and .model.predictive_adapter_mode == "none"
  and .data.repo_id == ($root + "/datasets/robotwin2.0_official_prompts_v21")
' "$PREFLIGHT_DIR/a0_dry_run.json" >/dev/null

jq -e --arg root "$STAGE_ROOT" '
  .name == "pi05_predictive_adapter_p1"
  and .seed == 1000
  and .training.batch_size == 16
  and .training.num_train_steps == 50000
  and .model.predictive_adapter_mode == "joint"
  and .data.repo_id == ($root + "/datasets/robotwin2.0_official_prompts_v21")
  and .initialization.adapter_params == ($root + "/kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999/params")
' "$PREFLIGHT_DIR/candidate_dry_run.json" >/dev/null

tmp=${REPORT}.tmp.$$
jq -n \
  --arg timestamp "$(date -u +'%FT%TZ')" \
  --arg stage_root "$STAGE_ROOT" \
  --arg python "$PYTHON" \
  '{
    schema_version: 1,
    timestamp: $timestamp,
    stage_root: $stage_root,
    python: $python,
    stage_verified: true,
    source_freeze_passed: true,
    a0_dry_run_passed: true,
    candidate_dry_run_passed: true,
    a0_resume_from_10000: true,
    candidate_resume_from_10000: true,
    runtime_preflight_passed: true,
    launch_authorized: false
  }' > "$tmp"
mv "$tmp" "$REPORT"
