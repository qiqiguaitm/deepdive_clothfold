#!/usr/bin/env bash
set -euo pipefail

src="${SRC:?set SRC to a local directory}"
dst="${DST:?set DST to an absolute North directory}"
host="${NORTH_HOST:-root@124.174.16.237}"
port="${NORTH_PORT:-16370}"

test -d "$src"
case "$dst" in
  /vePFS-North-E/vis_robot/*) ;;
  *) echo "refusing non-North destination: $dst" >&2; exit 2 ;;
esac

parent=$(dirname "$dst")
name=$(basename "$dst")
incoming="$parent/.${name}.incoming.$(date -u +%Y%m%d_%H%M%S).$$"
manifest=$(mktemp)
trap 'rm -f "$manifest"' EXIT

find_args=(. -type f -print0)
tar_args=(-C "$src" -cf - .)
if [ "${SYNC_EVAL_ONLY:-0}" = 1 ]; then
  # Evaluation only needs the model parameters and assets. Excluding the
  # optimizer state avoids transferring roughly 19 GiB for intermediate
  # checkpoints while preserving the same per-file integrity verification.
  find_args=(. -path ./train_state -prune -o -type f -print0)
  tar_args=(-C "$src" --exclude=./train_state -cf - .)
fi

(
  cd "$src"
  find "${find_args[@]}" | sort -z | xargs -0 sha256sum
) > "$manifest"
test -s "$manifest"

ssh -p "$port" -o BatchMode=yes "$host" \
  "mkdir -p $(printf %q "$parent"); rm -rf $(printf %q "$incoming"); mkdir -p $(printf %q "$incoming")"

tar "${tar_args[@]}" | ssh -p "$port" -o BatchMode=yes "$host" \
  "tar -C $(printf %q "$incoming") -xf -"

ssh -p "$port" -o BatchMode=yes "$host" \
  "cd $(printf %q "$incoming") && sha256sum -c -" < "$manifest"

ssh -p "$port" -o BatchMode=yes "$host" \
  "rm -rf $(printf %q "$dst"); mv $(printf %q "$incoming") $(printf %q "$dst")"

printf 'verified_sync source=%s destination=%s files=%s completed=%s\n' \
  "$src" "$dst" "$(wc -l < "$manifest")" "$(date -u +%FT%TZ)"
