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
tar_args=(-C "$src" -cf - .)
if [ "${SYNC_EVAL_ONLY:-0}" = 1 ]; then
  find_command='find . -path ./train_state -prune -o -type f -print0'
  tar_args=(-C "$src" --exclude=./train_state -cf - .)
fi

ssh -p "$port" -o BatchMode=yes "$host" "test -d $(printf %q "$src")"
ssh -p "$port" -o BatchMode=yes "$host" \
  "cd $(printf %q "$src") && $find_command | sort -z | xargs -0 sha256sum" \
  > "$manifest"
test -s "$manifest"

mkdir -p "$parent"
rm -rf "$incoming"
mkdir -p "$incoming"
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
