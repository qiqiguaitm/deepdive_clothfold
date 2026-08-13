#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
PYTHON=$REPO/kai0/.venv/bin/python
FINALIZER=$REPO/lmvla/lmwm/scripts/finalize_temporal_grounding_tg4_todo.py
REPORT=$REPO/lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg4.json
ANALYSIS_MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_analysis.ok
TODO=$REPO/lmvla/paper_iclr_lmvla/PAPER_TODO.md
SUMMARY=$REPO/lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg4.md
MARKER=$REPO/logs/resource_markers/temporal_grounding_tg4_todo_finalize.ok
PROVISIONAL=$MARKER.provisional
LOCK=$REPO/logs/locks/temporal_grounding_tg4_todo_finalize.lock

if [[ -e "$PROVISIONAL" ]]; then
  mv "$PROVISIONAL" "$PROVISIONAL.stale.$(date -u +%Y%m%d_%H%M%S)"
fi
"$PYTHON" "$FINALIZER" \
  --report "$REPORT" \
  --analysis-marker "$ANALYSIS_MARKER" \
  --todo "$TODO" \
  --summary "$SUMMARY" \
  --completion-marker "$PROVISIONAL" \
  --lock "$LOCK"

test "$(git -C "$REPO" branch --show-current)" = main
if ! git -C "$REPO" diff --quiet -- "$TODO" "$SUMMARY"; then
  git -C "$REPO" commit --only -m "Finalize TG4 source decomposition" -- \
    "$TODO" "$SUMMARY"
fi
git -C "$REPO" push origin main
head_commit=$(git -C "$REPO" rev-parse HEAD)
remote_commit=$(git -C "$REPO" rev-parse origin/main)
test "$head_commit" = "$remote_commit"
printf 'git_commit=%s\n' "$head_commit" >>"$PROVISIONAL"
mv "$PROVISIONAL" "$MARKER"
