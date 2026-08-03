# Migration map

The first consolidation pass keeps old paths operational. Use this table when
touching an older script; move shared implementation into `data_tools` and leave
only a small argument-compatible wrapper at the original path.

| Previous script family | Consolidated API / command |
|---|---|
| AgileX `filter_pick_place_episodes.py` (3 identical copies) | `python -m data_tools quality` / `data_tools.quality` |
| AgileX `convert_pickplace_to_lerobot.py` | `python -m data_tools normalize --task "pick and place in box"` |
| AgileX `convert_nailpainting_to_lerobot.py` | `python -m data_tools normalize --task "nail painting"` |
| Repeated JSONL/layout/link/reindex helpers in `train_scripts/kai/data` | `data_tools.lerobot` |
| Cross-format inspect/convert/quality/segment | `python -m data_tools forge ...` |
| Script discovery across legacy directories | `python -m data_tools inventory` |
| Hard-coded `finalize_dagger_dataset.py` static CSV generation | `python -m data_tools static --min-frames 50` |
| `web/data_manager/backend/tools/trim_station_edges.py` | `python -m data_tools.edge_trim` |
| `train_scripts/kai/data/classify_dagger_frames.py` | `python -m data_tools.dagger_classify` |
| `train_scripts/kai/data/trim_stitched_episodes.py` | `python -m data_tools.dagger_trim` |
| `train_scripts/kai/data/build_no_release.py` | `python -m data_tools.legacy_build_no_release` |

Not migrated automatically:

- one-off research analyses and figures;
- task-specific sampling recipes whose hard-coded episode lists are part of an
  experiment record;
- online recorder maintenance scripts that import backend internals;
- destructive AgileX quarantine mode. The consolidated quality command is
  intentionally report-only.

Forge 0.2.0 does not currently replace KAI0 merge/reindex logic. Its upstream
README/roadmap advertises merge and split as planned work, so KAI0 uses Forge for
format-neutral inspection/conversion and retains `normalize` for deterministic
LeRobot v2.1 merging.
