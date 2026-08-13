#!/usr/bin/env bash
set -euo pipefail

repo=${REPO_ROOT:-/vePFS/tim/workspace/deepdive_kai0}
: "${TG4_ARM:?TG4_ARM is required}"
: "${TG4_TRAIN_SEED:?TG4_TRAIN_SEED is required}"

runtime_dir="$repo/logs/temporal_grounding/tg4/runtime_snapshots"
mkdir -p "$runtime_dir"
runtime_runner="$runtime_dir/${TG4_ARM}_s${TG4_TRAIN_SEED}_${HOSTNAME:-unknown}_$$.sh"
install -m 0555 \
  "$repo/train_scripts/kai/run_temporal_grounding_tg4_train.sh" \
  "$runtime_runner"
exec bash "$runtime_runner"
