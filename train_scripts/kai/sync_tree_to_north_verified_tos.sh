#!/usr/bin/env bash
set -euo pipefail

src="${SRC:?set SRC to a local directory}"
dst="${DST:?set DST to an absolute North directory}"
host="${NORTH_HOST:-root@124.174.16.237}"
port="${NORTH_PORT:-16370}"
bucket="${NORTH_TOS_BUCKET:-transfer-shanghai}"
prefix="${NORTH_TOS_PREFIX:-temp/deepdive_kai0/north-sync}"
jobs="${NORTH_TOS_JOBS:-8}"
parallels="${NORTH_TOS_PARALLELS:-8}"

test -d "$src"
case "$dst" in
  /vePFS-North-E/vis_robot/*) ;;
  *) echo "refusing non-North destination: $dst" >&2; exit 2 ;;
esac
command -v tosutil >/dev/null
ssh -p "$port" -o BatchMode=yes "$host" 'command -v tosutil >/dev/null'

parent=$(dirname "$dst")
name=$(basename "$dst")
stamp="$(date -u +%Y%m%d_%H%M%S).$$"
incoming="$parent/.${name}.incoming.tos.$stamp"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
stage_parent="${LOCAL_TOS_STAGE_ROOT:-$repo/logs/sync/tos_staging}"
mkdir -p "$stage_parent"
local_stage="$stage_parent/${name}.tos-stage.$stamp"
object="tos://$bucket/$prefix/${name}-$stamp/"
manifest=$(mktemp)
committed=0

cleanup() {
  if [ -d "$local_stage" ]; then
    rm -rf -- "$local_stage"
  fi
  tosutil rm "$object" -r -f -j="$jobs" >/dev/null 2>&1 || true
  if [ "$committed" -eq 0 ]; then
    ssh -p "$port" -o BatchMode=yes "$host" \
      "rm -rf -- $(printf %q "$incoming")" >/dev/null 2>&1 || true
  fi
  rm -f -- "$manifest"
}
trap cleanup EXIT

mkdir -p "$local_stage"
# The training process may create root-owned checkpoint files. Linux protected
# hardlinks then reject cp -al for the unprivileged scheduler user. Reflink
# when supported and otherwise perform a regular same-vePFS copy.
cp -a --reflink=auto "$src/." "$local_stage/"
if [ "${SYNC_EVAL_ONLY:-0}" = 1 ] && [ -d "$local_stage/train_state" ]; then
  rm -rf -- "$local_stage/train_state"
fi

(
  cd "$local_stage"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$manifest"
test -s "$manifest"

echo "phase=tos-upload"
tosutil cp "$local_stage/" "$object" -r -flat -vchecksum \
  -j="$jobs" -p="$parallels" -threshold=52428800

echo "phase=tos-download"
ssh -p "$port" -o BatchMode=yes "$host" \
  "mkdir -p $(printf %q "$parent"); rm -rf -- $(printf %q "$incoming"); mkdir -p $(printf %q "$incoming"); \
   tosutil cp $(printf %q "$object") $(printf %q "$incoming/") -r -f -flat -vchecksum \
     -j=$(printf %q "$jobs") -p=$(printf %q "$parallels") -threshold=52428800"

echo "phase=sha256-verify"
ssh -p "$port" -o BatchMode=yes "$host" \
  "cd $(printf %q "$incoming") && sha256sum -c -" < "$manifest"

echo "phase=atomic-commit"
ssh -p "$port" -o BatchMode=yes "$host" \
  "rm -rf -- $(printf %q "$dst"); mv $(printf %q "$incoming") $(printf %q "$dst")"
committed=1

printf 'verified_sync transport=tos source=%s destination=%s files=%s completed=%s\n' \
  "$src" "$dst" "$(wc -l < "$manifest")" "$(date -u +%FT%TZ)"
