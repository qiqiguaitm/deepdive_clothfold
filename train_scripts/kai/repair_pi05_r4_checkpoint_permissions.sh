#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}
ROOT=$REPO/lmvla/lmwm/checkpoints/pi05_r4_matched_v1
REPORT=$REPO/logs/r4/checkpoint_integrity_v1.json
MARKER=$REPO/logs/resource_markers/pi05_r4_checkpoint_permissions.ok
ARMS=(ordinary terminal_outcome outcome_free_crave)

if [[ $(id -u) -ne 0 ]]; then
  echo "R4 checkpoint permission normalization must run in the platform root container" >&2
  exit 77
fi

for arm in "${ARMS[@]}"; do
  run=$ROOT/${arm}-seed1000
  marker=$REPO/logs/resource_markers/pi05_r4_${arm}-seed1000.ok
  model=$run/checkpoints/005000/pretrained_model/model.safetensors
  config=$run/checkpoints/005000/pretrained_model/config.json
  step=$run/checkpoints/005000/training_state/training_step.json
  test -s "$marker"
  test -s "$model"
  test -s "$config"
  test -s "$step"
done

# The platform trainer creates mode-0600 root-owned tensors. This changes only
# access bits so the same immutable checkpoints can be audited and evaluated
# from the shared development host.
for arm in "${ARMS[@]}"; do
  chmod -R a+rX "$ROOT/${arm}-seed1000"
done

mkdir -p "$(dirname "$REPORT")" "$(dirname "$MARKER")"
tmp=$(mktemp "${REPORT}.tmp.XXXXXX")
python - "$ROOT" "$tmp" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
arms = ("ordinary", "terminal_outcome", "outcome_free_crave")
records = {}
for arm in arms:
    checkpoint = root / f"{arm}-seed1000" / "checkpoints" / "005000"
    model = checkpoint / "pretrained_model" / "model.safetensors"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    step = json.loads(
        (checkpoint / "training_state" / "training_step.json").read_text()
    )
    records[arm] = {
        "checkpoint": str(checkpoint),
        "model_bytes": model.stat().st_size,
        "model_mode": oct(model.stat().st_mode & 0o777),
        "model_sha256": digest.hexdigest(),
        "readable": os.access(model, os.R_OK),
        "training_step": step,
    }

assert all(record["readable"] for record in records.values())
assert all(record["model_mode"] in {"0o644", "0o664"} for record in records.values())
assert all(record["training_step"]["step"] == 5000 for record in records.values())
output.write_text(json.dumps({"schema_version": 1, "arms": records}, indent=2) + "\n")
PY
mv "$tmp" "$REPORT"
chmod a+r "$REPORT"
printf 'completed=%s\nreport=%s\n' "$(date -u +%FT%TZ)" "$REPORT" >"$MARKER"
chmod a+r "$MARKER"
