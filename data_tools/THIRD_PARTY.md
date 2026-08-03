# Third-party provenance

## Forge Robotics

- Project: https://github.com/arpitg1304/forge
- Package: `forge-robotics==0.2.0`
- Reviewed upstream commit: `29e1df0bf0ff4dbd21f39cf7b59d6cd97620f2be`
- Upstream commit date: 2026-07-22
- License: MIT
- Copyright: 2025 Arpit Gupta

KAI0 does not vendor Forge source. `forge_adapter.py` invokes the separately
installed package and pins the version tested by this repository. The upstream
license permits integration; keeping it external avoids copying a fast-moving
alpha codebase and its optional TensorFlow/ROS/video dependency graph.

## AgileX data scripts (2026-07-31 archive)

Source archive: `/home/tim/workspace/resource/agilex_data_scripts_20260731.zip`.

The archive contained:

- two near-duplicate LeRobot converters for pick-place and nail-painting;
- three byte-identical copies of `filter_pick_place_episodes.py`;
- three machine-specific shell wrappers with `/home/agilex/...` defaults.

Their reusable filesystem, JSONL, episode reindexing, artifact placement and
structural quality checks are consolidated into `lerobot.py` and `quality.py`.
Machine-specific absolute paths were intentionally not imported.
