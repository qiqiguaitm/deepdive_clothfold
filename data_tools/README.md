# KAI0 data tools

This directory is the stable home for reusable dataset processing. Historical
experiment/build scripts remain at their old paths so existing commands and
research records do not break; new scripts should import `data_tools.lerobot`
instead of copying helpers.

## Commands

```bash
# Find and classify existing scripts
python -m data_tools inventory
python -m data_tools inventory --output data_tools/SCRIPT_INVENTORY.md

# Validate action/parquet/video alignment (report-only)
python -m data_tools quality /path/to/lerobot-leaf \
  --output /tmp/quality.jsonl

# Replace both AgileX pick-place/nail-painting converters with one command
python -m data_tools normalize \
  --src /data/day1 --src /data/day2 --dst /data/normalized \
  --task "nail painting" --use-latest-good

# Locate every >=50-frame stationary run, including interior pauses
python -m data_tools static /path/to/lerobot-leaf \
  --min-frames 50 --ideal-only --output /tmp/static_segments.jsonl

# Detect 50Hz-related flicker / rolling brightness bands in one video
python -m data_tools flicker /path/to/episode_000001.mp4 --mains-hz 50

# Sample 20 top-head videos from a dataset
python -m data_tools flicker /path/to/dataset --camera top_head --sample 20 \
  --output /tmp/flicker.jsonl

# Unified read-only audit: one day, several days, a range, or all days
python -m data_tools audit /data1/DATA_IMP/KAI0/Task_A1 \
  --date 2026-07-23 --output /tmp/audit-0723.json
python -m data_tools audit /data1/DATA_IMP/KAI0/Task_A1 \
  --date 2026-07-23 --date 2026-07-24 --output /tmp/audit-days.json
python -m data_tools audit /data1/DATA_IMP/KAI0/Task_A1 \
  --date-from 2026-07-20 --date-to 2026-07-31 --output /tmp/audit-range.json
python -m data_tools audit /data1/DATA_IMP/KAI0/Task_A1 \
  --output /tmp/audit-all.json

# Install and use the tested Forge backend
python -m pip install -r data_tools/requirements-forge.txt
python -m data_tools forge inspect /path/to/dataset
python -m data_tools forge convert input.hdf5 output/ --format lerobot-v3
```

`quality` writes a JSONL report and a sibling `*.good_episodes.txt` compatible
with the AgileX filtering/conversion workflow. It never quarantines or deletes
source artifacts. `normalize` also refuses to overwrite an existing output;
direct Forge passthrough commands retain Forge's own behavior.

`audit` checks parquet/action/video integrity and frame-count mismatches,
trajectory velocity/acceleration spikes, all stationary runs of at least 50
frames (leading, interior, and trailing), plus sampled image blur and 50 Hz
flicker. It only writes the requested JSON report. Use `--visual-sample 0` to
decode every video instead of the default 12 representative videos per leaf.

## Organization policy

- `data_tools/`: shared production-safe primitives and format adapters.
- `web/data_manager/backend/tools/`: online recorder maintenance only.
- `train_scripts/kai/data/`: reproducible experiment recipes and dataset builds.
- `start_scripts/data_fix/`: legacy compatibility entry points; migrate their
  implementation into this library when next touched.

See `THIRD_PARTY.md` for AgileX archive and Forge provenance.

## Consolidated operational scripts

The implementations below now live in this directory. Their previous paths are
compatibility wrappers only.

```bash
# Safe leading/trailing idle trim (dry-run unless --apply)
python -m data_tools.edge_trim --leaf /path/to/leaf --chunk 0

# DAgger frame classification
python -m data_tools.dagger_classify /path/to/leaf --dry-run

# Remove classified non-ideal frames from stitched chunk-001 data
python -m data_tools.dagger_trim --date 2026-06-24 --dry-run

# Legacy destructive finalizer (discouraged; retained for reproducibility)
python -m data_tools.legacy_dagger_finalize
```

The historical multi-purpose `build_no_release.py` implementation is retained
as `legacy_build_no_release.py` because many experiment recipes import its
statistics and frame-selection helpers. Its old path re-exports all names for
compatibility; new code should use focused `data_tools` modules instead.
