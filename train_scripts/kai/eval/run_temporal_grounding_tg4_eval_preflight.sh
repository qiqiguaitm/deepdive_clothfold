#!/usr/bin/env bash
set -euo pipefail
umask 0002

readonly REPO="${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}"
readonly LAWAM="$REPO/lmvla/lawam"
readonly RUN_ID=temporal_grounding_tg4_full_seed1100
readonly PREFLIGHT_LABEL="${TG4_PREFLIGHT_LABEL:-gf1}"
readonly SCENES="$REPO/lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
readonly EVAL_MANIFEST="$REPO/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg4_evaluation_v1.json"
[[ "$PREFLIGHT_LABEL" =~ ^[a-z0-9_]+$ ]] || {
  echo "TG4_PREFLIGHT_LABEL must contain only lowercase letters, digits, or underscores" >&2
  exit 2
}
readonly ROOT="$REPO/logs/resource_scheduler_local/tg4_eval_preflight_${PREFLIGHT_LABEL}"
readonly RESULT_ROOT="$ROOT/result"
readonly FEATURE_ROOT="$ROOT/features"
readonly MARKER="$REPO/logs/resource_markers/temporal_grounding_tg4_eval_${PREFLIGHT_LABEL}_preflight.ok"
readonly LOG="$ROOT/eval.log"
readonly CONTROL_PYTHON="${TG4_CONTROL_PYTHON:-$REPO/kai0/.venv/bin/python}"

test ! -e "$MARKER" || exit 0
test ! -e "$RESULT_ROOT" || {
  echo "refusing to mix TG4 evaluator preflight with existing result root: $RESULT_ROOT" >&2
  exit 3
}

bash "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh"
source "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh"
"$CONTROL_PYTHON" \
  "$REPO/lmvla/lmwm/scripts/verify_temporal_grounding_tg4_evaluation.py" \
  --repo "$REPO" --manifest "$EVAL_MANIFEST"

mapfile -t run_dirs < <(find "$LAWAM/results/Checkpoints/robotwin" -maxdepth 1 \
  -type d -name "*+${RUN_ID}" -print | sort)
[[ ${#run_dirs[@]} -eq 1 ]] || {
  echo "expected exactly one accepted preflight checkpoint, found ${#run_dirs[@]}" >&2
  exit 13
}
readonly CKPT="${run_dirs[0]}/final_model/pytorch_model.pt"
test -f "$CKPT"
test -f "$SCENES"
mkdir -p "$RESULT_ROOT/seed0" "$FEATURE_ROOT"

export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/vePFS/tim/workspace/miniconda3_gf0/envs/lawam/bin/python}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/vePFS/HuanQian/RoboTwin}"
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-$REPO/lmvla/lmwam/scripts/robotwin_python_wrapper.sh}"
export ROBOTWIN_TASKS=beat_block_hammer
export TASK_CONFIG=demo_clean
export ROBOTWIN_TEST_NUM=1
export ROBOTWIN_NUM_SLOTS=1
export NUM_WORKERS=1
export ROBOTWIN_SAVE_VIDEO=0
export ROBOTWIN_INSTRUCTION_TYPE=unseen
export ROBOTWIN_REPLAN_STEPS=36
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN=1
export ROBOTWIN_ACTION_ENSEMBLE=0
export ROBOTWIN_EPISODE_SEED_MANIFEST="$SCENES"
export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=500
export ROBOTWIN_TASK_SCOPED_SERVER=1
export LAWAM_FUTURE_INTERVENTION=normal
export LAWAM_FUTURE_CAPTURE_ROOT="$FEATURE_ROOT"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export USE_BF16=1
export GPU_IDS=0
export SEED=0
export PORT_BASE=28600
export ROBOTWIN_CKPT_ALIAS="tg4_full_s1100_${PREFLIGHT_LABEL}_preflight"
export ROBOTWIN_EVAL_ROOT="$RESULT_ROOT/seed0"
unset LAWAM_FUTURE_OFF LAWAM_AUXILIARY_OFF LAWAM_CONDITIONING_OFF
unset LAWAM_FUTURE_SHUFFLE_MANIFEST
unset LMWM_CKPT LMWM_MILESTONE_TARGET LMWM_TARGET_COMPACT LMWM_FEAT_DIR
unset LMWM_ADAPTER_DIR LMWM_SWAP_TEACHER LMWM_FEAT_STRIDE
unset LMWM_HINT_DROPOUT LMWM_DUAL LMWM_DUAL_2Q LMWM_MS_TSCHED
unset LMWM_MS_RESIDUAL LMWM_MS_RESID_SCALE LMWM_MS_ABS_SCALE
unset LMWM_MS_GATE LMWM_MS_DETACH_BACKBONE LMWM_LOCAL_DETACH_BACKBONE

cd "$LAWAM"
bash examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh \
  "$CKPT" "$TASK_CONFIG" "tg4-full-s1100-${PREFLIGHT_LABEL}-preflight" >"$LOG" 2>&1

mapfile -t summaries < <(find "$RESULT_ROOT" -type f -name summary.json -print | sort)
[[ ${#summaries[@]} -eq 1 ]] || {
  echo "expected one preflight summary, found ${#summaries[@]}" >&2
  exit 14
}
find "$FEATURE_ROOT" -type f -print -quit | grep -q . || {
  echo "TG4 full-arm preflight did not capture any future features" >&2
  exit 15
}

"$CONTROL_PYTHON" - "$CKPT" "${summaries[0]}" "$MARKER" "$PREFLIGHT_LABEL" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

checkpoint, summary_path, marker_path = map(Path, sys.argv[1:4])
preflight_label = sys.argv[4]
summary = json.loads(summary_path.read_text())
if summary.get("task_name") != "beat_block_hammer":
    raise SystemExit("unexpected preflight task")
if summary.get("n_episodes") != 1 or len(summary.get("episodes", [])) != 1:
    raise SystemExit("preflight did not execute exactly one episode")
if int(summary.get("model_queries", 0)) <= 0:
    raise SystemExit("preflight policy server handled no model queries")
if int(summary.get("obs_fetch_count", 0)) <= 0:
    raise SystemExit("preflight simulator produced no observations")
payload = {
    "complete": True,
    "protocol": "temporal_grounding_tg4_gf1_eval_preflight_v1",
    "claim_bearing": False,
    "resource_preflight": preflight_label,
    "checkpoint": str(checkpoint),
    "checkpoint_bytes": checkpoint.stat().st_size,
    "summary": str(summary_path),
    "task": summary["task_name"],
    "episodes": summary["n_episodes"],
    "model_queries": summary["model_queries"],
    "obs_fetch_count": summary["obs_fetch_count"],
    "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}
marker_path.parent.mkdir(parents=True, exist_ok=True)
temporary = marker_path.with_suffix(marker_path.suffix + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, marker_path)
PY

echo "TG4 $PREFLIGHT_LABEL evaluator preflight complete marker=$MARKER"
