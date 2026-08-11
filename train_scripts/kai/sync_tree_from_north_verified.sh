#!/usr/bin/env bash
set -euo pipefail

src="${SRC:?set SRC to an absolute North directory}"
dst="${DST:?set DST to a shared vePFS directory}"
host="${NORTH_HOST:-root@124.174.16.237}"
port="${NORTH_PORT:-16370}"

case "$src" in
  /vePFS-North-E/vis_robot/*) ;;
  *) echo "refusing non-North source: $src" >&2; exit 2 ;;
esac
case "$dst" in
  /vePFS/tim/workspace/deepdive_kai0/*) ;;
  *) echo "refusing destination outside the shared repository: $dst" >&2; exit 2 ;;
esac

parent=$(dirname "$dst")
name=$(basename "$dst")
incoming="$parent/.${name}.incoming.$(date -u +%Y%m%d_%H%M%S).$$"
manifest=$(mktemp)
trap 'rm -f "$manifest"; rm -rf "$incoming"' EXIT

find_command='find . -type f -print0'
tar_args=(-C "$src" -cf -)
parallel_large_files=()
if [ "${SYNC_EVAL_ONLY:-0}" = 1 ]; then
  find_command='find . -path ./train_state -prune -o -type f -print0'
  tar_args=(-C "$src" --exclude=./train_state -cf -)
fi
if [ "${SYNC_PARALLEL_LARGE_FILES:-0}" = 1 ]; then
  parallel_large_files=(
    final_model/pytorch_model.pt
    checkpoints/steps_20000_state/optimizer.bin
    checkpoints/steps_20000_state/pytorch_model.bin
  )
  for relative in "${parallel_large_files[@]}"; do
    ssh -p "$port" -o BatchMode=yes "$host" \
      "test -f $(printf %q "$src/$relative")"
    tar_args+=("--exclude=./$relative")
  done
fi
tar_args+=(.)

ssh -p "$port" -o BatchMode=yes "$host" "test -d $(printf %q "$src")"
ssh -p "$port" -o BatchMode=yes "$host" \
  "cd $(printf %q "$src") && $find_command | sort -z | xargs -0 sha256sum" \
  > "$manifest"
test -s "$manifest"

mkdir -p "$parent"
rm -rf "$incoming"
mkdir -p "$incoming"
parallel_pids=()
for relative in "${parallel_large_files[@]}"; do
  mkdir -p "$incoming/$(dirname "$relative")"
  (
    ssh -p "$port" -o BatchMode=yes "$host" \
      "cat $(printf %q "$src/$relative")" >"$incoming/$relative"
  ) &
  parallel_pids+=("$!")
done
parallel_failed=0
for pid in "${parallel_pids[@]}"; do
  wait "$pid" || parallel_failed=1
done
test "$parallel_failed" = 0
ssh -p "$port" -o BatchMode=yes "$host" \
  "tar $(printf '%q ' "${tar_args[@]}")" | tar -C "$incoming" -xf -

(
  cd "$incoming"
  sha256sum -c "$manifest"
)

rm -rf "$dst"
mv "$incoming" "$dst"
trap 'rm -f "$manifest"' EXIT

printf 'verified_reverse_sync source=%s destination=%s files=%s completed=%s\n' \
  "$src" "$dst" "$(wc -l < "$manifest")" "$(date -u +%FT%TZ)"
