#!/usr/bin/env bash
# Compatibility entry point for the FastWAM A1 strict full-chunk test.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$REPO_ROOT/fastwam/deploy/run_a1_full_chunk_execute.sh" "$@"
