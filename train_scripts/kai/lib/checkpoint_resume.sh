#!/usr/bin/env bash

# Populate the named array with --resume when a complete numeric checkpoint exists.
checkpoint_resume_args() {
  if [[ $# -ne 2 ]]; then
    printf 'usage: checkpoint_resume_args CHECKPOINT_ROOT OUTPUT_ARRAY\n' >&2
    return 2
  fi

  local checkpoint_root=$1
  local output_name=$2
  local -n output_args=$output_name
  local -a checkpoint_metadata=()
  local path step latest_step=-1

  output_args=()
  shopt -s nullglob
  checkpoint_metadata=("$checkpoint_root"/[0-9]*/_CHECKPOINT_METADATA)
  shopt -u nullglob

  for path in "${checkpoint_metadata[@]}"; do
    step=$(basename "$(dirname "$path")")
    [[ $step =~ ^[0-9]+$ ]] || continue
    if (( 10#$step > latest_step )); then
      latest_step=$((10#$step))
    fi
  done

  if (( latest_step >= 0 )); then
    output_args=(--resume)
    printf 'resuming from checkpoint step %d under %s\n' \
      "$latest_step" "$checkpoint_root" >&2
  fi
}
