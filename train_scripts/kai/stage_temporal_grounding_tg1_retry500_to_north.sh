#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
HOST=${NORTH_HOST:-root@124.174.16.237}
PORT=${NORTH_PORT:-16370}
NORTH_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0
STAGE=$NORTH_ROOT/.staging/temporal_grounding_tg1_retry500_v1
LAWAM_COMMIT=7ca170c9b98e15026ecd9004d91f0ad9c73ba5f7
LAWAM_PARENT=71803a3f8b0e55679a4557ef6af80a76604f277a
OUTER_IMPLEMENTATION=db88e943ddfecf25be0ee83a332b542b5d4419ae
OUTER_HEAD=$(git -C "$REPO" rev-parse HEAD)
LOCAL_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg1_retry500_north_stage.ok
REMOTE_MARKER=$STAGE/logs/resource_markers/temporal_grounding_tg1_retry500_north_stage.ok
SSH=(ssh -p "$PORT" -o BatchMode=yes "$HOST")
RSYNC_RSH="ssh -p $PORT -o BatchMode=yes"

OUTER_FILES=(
  lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json
  lmvla/lmwm/data/robotwin_milestone_all6_v2/lmwm.pt
  lmvla/lmwam/env/heal_lawam_symlinks.sh
  lmvla/lmwm/scripts/activate_temporal_grounding_tg1_retry500.py
  lmvla/lmwm/scripts/analyze_temporal_grounding_tg1a.py
  lmvla/lmwm/scripts/analyze_temporal_grounding_tg1b.py
  lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py
  lmvla/lmwm/scripts/verify_temporal_grounding_feature_capture.py
  lmvla/lmwm/scripts/verify_temporal_grounding_tg1_retry500.py
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v11.json
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_shuffle_v1.json
  lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1b_admission_v1.json
  logs/resource_markers/temporal_grounding_tg1_retry500_activation_v1.json
  train_scripts/kai/eval/local_robotwin_all6_combo_seed2026_2gpu.sh
  train_scripts/kai/eval/run_temporal_grounding_tg1a_formal.sh
  train_scripts/kai/eval/run_temporal_grounding_tg1a_retry500_formal.sh
  train_scripts/kai/eval/run_temporal_grounding_tg1b_formal.sh
  train_scripts/kai/eval/run_temporal_grounding_tg1b_retry500_formal.sh
  train_scripts/kai/volc/temporal_grounding_tg1a_east_4h20.yaml
  train_scripts/kai/volc/temporal_grounding_tg1a_retry500_east_4h20.yaml
  train_scripts/kai/volc/temporal_grounding_tg1b_east_4h20.yaml
  train_scripts/kai/volc/temporal_grounding_tg1b_retry500_east_4h20.yaml
)
CHECKPOINT_DIRS=(
  lawam_robotwin_sft_release
  20260731_172204+robotwin_all6_v2_nowm_seed2027
  20260730_234942+robotwin_all6_v2_local_seed2027
)

test "$(git -C "$REPO/lmvla/lawam" rev-parse HEAD)" = "$LAWAM_COMMIT"
for relative in "${OUTER_FILES[@]}"; do
  test -s "$REPO/$relative"
done

bundle_dir=$(mktemp -d)
trap 'rm -rf "$bundle_dir"' EXIT
git -C "$REPO" bundle create "$bundle_dir/outer.bundle" \
  HEAD "^$OUTER_IMPLEMENTATION"
git -C "$REPO/lmvla/lawam" bundle create \
  "$bundle_dir/lawam.bundle" lmvla-eval-bridge "^$LAWAM_PARENT"
scp -P "$PORT" -o BatchMode=yes "$bundle_dir/outer.bundle" \
  "$bundle_dir/lawam.bundle" "$HOST:/tmp/"

"${SSH[@]}" "set -euo pipefail
  rm -rf '$STAGE.incoming'
  git init -q '$STAGE.incoming'
  mkdir -p '$STAGE.incoming/.git/objects/info'
  printf '%s\n' '$NORTH_ROOT/.git/objects' >'$STAGE.incoming/.git/objects/info/alternates'
  git -C '$STAGE.incoming' fetch -q /tmp/outer.bundle '$OUTER_HEAD'
  git -C '$STAGE.incoming' checkout -q --detach FETCH_HEAD
  rm -rf '$STAGE.incoming/lmvla/lawam'
  git init -q '$STAGE.incoming/lmvla/lawam'
  mkdir -p '$STAGE.incoming/lmvla/lawam/.git/objects/info'
  printf '%s\n' '$NORTH_ROOT/.git/modules/lmvla/lawam/objects' >'$STAGE.incoming/lmvla/lawam/.git/objects/info/alternates'
  git -C '$STAGE.incoming/lmvla/lawam' fetch -q /tmp/lawam.bundle '$LAWAM_COMMIT'
  git -C '$STAGE.incoming/lmvla/lawam' checkout -q --detach FETCH_HEAD
  rm -f /tmp/outer.bundle /tmp/lawam.bundle"
tar -C "$REPO" -cf - "${OUTER_FILES[@]}" | \
  "${SSH[@]}" "tar -C '$STAGE.incoming' -xf -"

"${SSH[@]}" "set -euo pipefail
  mkdir -p '$STAGE.incoming/kai0' '$STAGE.incoming/logs/tg1_retry500' '$STAGE.incoming/logs/resource_markers'
  ln -s '$NORTH_ROOT/kai0/.venv' '$STAGE.incoming/kai0/.venv'
  ln -s '$NORTH_ROOT/lmvla/lawam_local' '$STAGE.incoming/lmvla/lawam_local'
  ln -s '$NORTH_ROOT/lmvla/lawam/robotwin_python_wrapper_northe.sh' '$STAGE.incoming/lmvla/lawam/robotwin_python_wrapper_northe.sh'
  printf '%s\n' '$LAWAM_COMMIT' >'$STAGE.incoming/lmvla/lawam/.source_commit'
  rm -rf '$STAGE.previous'
  test ! -e '$STAGE' || mv '$STAGE' '$STAGE.previous'
  mv '$STAGE.incoming' '$STAGE'"

for run in "${CHECKPOINT_DIRS[@]}"; do
  local_run=$REPO/lmvla/lawam/results/Checkpoints/robotwin/$run
  remote_run=$NORTH_ROOT/lmvla/lawam_local/results/Checkpoints/robotwin/$run
  "${SSH[@]}" "mkdir -p '$remote_run/final_model'"
  rsync -a --partial --append-verify -e "$RSYNC_RSH" \
    "$local_run/final_model/pytorch_model.pt" "$HOST:$remote_run/final_model/"
  for metadata in config.yaml config.json dataset_statistics.json; do
    if [[ -f "$local_run/$metadata" ]]; then
      rsync -a -e "$RSYNC_RSH" "$local_run/$metadata" "$HOST:$remote_run/"
    fi
  done
done

rsync -a --delete -e "$RSYNC_RSH" \
  "$REPO/logs/tg1_retry500/predicted_endpoint_features/" \
  "$HOST:$STAGE/logs/tg1_retry500/predicted_endpoint_features/"
rsync -a -e "$RSYNC_RSH" \
  "$REPO/logs/resource_markers/temporal_grounding_tg1a_retry500_normal_capture_complete.json" \
  "$HOST:$STAGE/logs/resource_markers/"

"${SSH[@]}" bash -s -- "$STAGE" "$REMOTE_MARKER" "$LAWAM_COMMIT" <<'REMOTE'
set -euo pipefail
stage=$1
marker=$2
lawam_commit=$3
test "$(cat "$stage/lmvla/lawam/.source_commit")" = "$lawam_commit"
bash "$stage/lmvla/lmwam/env/heal_lawam_symlinks.sh"
export TEMPORAL_GROUNDING_RUNTIME_AMENDMENT="$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_runtime_amendment_v11.json"
"$stage/kai0/.venv/bin/python" \
  "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$stage" --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1a_admission_v1.json" --bundle TG1A
"$stage/kai0/.venv/bin/python" \
  "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_bundle.py" \
  --repo "$stage" --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1b_admission_v1.json" --bundle TG1B
for bundle in TG1A TG1B; do
  "$stage/kai0/.venv/bin/python" \
    "$stage/lmvla/lmwm/scripts/verify_temporal_grounding_tg1_retry500.py" \
    --repo "$stage" --manifest "$stage/lmvla/paper_iclr_lmvla/manifests/temporal_grounding_tg1_retry500_amendment_v1.json" --bundle "$bundle"
done
mkdir -p "$(dirname "$marker")"
printf 'validated=%s\nstage=%s\nlawam_commit=%s\n' \
  "$(date -u +%FT%TZ)" "$stage" "$lawam_commit" >"$marker"
REMOTE

mkdir -p "$(dirname "$LOCAL_MARKER")"
printf 'validated=%s\nremote_stage=%s\nremote_marker=%s\n' \
  "$(date -u +%FT%TZ)" "$STAGE" "$REMOTE_MARKER" >"$LOCAL_MARKER"
