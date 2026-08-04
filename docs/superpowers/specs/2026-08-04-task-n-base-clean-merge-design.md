# Task_N `base_clean` merge design

## Objective

Create `/home/tim/workspace/deepdive_kai0/kai0/data/Task_N/base_clean` as one
standard LeRobot v2.1 leaf containing the union of:

- 481 cleaned episodes downloaded from the specified TOS payload; and
- 341 cleaned episodes currently present under `kai0/data/Task_N/base/v5`.

The source datasets are immutable. The output must be reproducible, globally
reindexed, structurally valid, and traceable to every source episode.

## Verified input contract

Only the four TOS leaves with real artifacts are inputs: `2026-07-29-v2`,
`2026-07-30-v2`, `2026-07-31-v2`, and `2026-08-03-v2`. Historical TOS leaves
that contain metadata but no parquet/video artifacts are excluded.

The complete preflight established:

- TOS: 481 episodes, 405,575 frames, 14-dimensional state/action;
- local v5: 341 episodes, 244,061 frames, 32-dimensional state/action;
- both inputs: 30 FPS, task `nail painting`, the same parquet columns, and the
  common RGB cameras `top_head`, `hand_left`, and `hand_right`;
- 2,466 required input videos, with no missing artifact;
- continuous source `frame_index` values and no non-finite values in the first
  14 state/action dimensions; and
- 822 unique trajectory hashes, with no within-source or cross-source duplicate.

The expected output is therefore exactly 822 episodes and 649,636 frames.

## Chosen approach

Add a Task_N-specific build recipe under `train_scripts/kai/data` and reuse the
filesystem primitives in `data_tools.lerobot`. Do not broaden the generic
`data_tools.normalize` interface: it assumes episode IDs are unique within a
source root and has no 32D-to-14D projection contract, while local v5 uses
compound date/station/chunk identities.

The builder will discover actual parquet artifacts rather than trusting stale
declared totals. It will order the 481 TOS episodes first and the 341 local v5
episodes second. Within each source, ordering is deterministic by date, chunk,
and source episode ID. Every output episode receives a new contiguous ID from
0 through 821.

## Transformation and output layout

For every episode, the builder will:

1. validate required columns, source dimensions, finite values, contiguous
   frame indices, and all three required videos;
2. retain the standard LeRobot columns;
3. keep the first 14 values of `observation.state` and `action` (a no-op for the
   TOS input and a projection for local v5);
4. rewrite `episode_index`, `frame_index`, global `index`, `timestamp`, and
   `task_index` consistently at 30 FPS;
5. write a compressed parquet under `data/chunk-000`; and
6. hard-link each source video into the canonical
   `videos/chunk-000/observation.images.<camera>` path, falling back to a copy
   only if a hard link is unavailable.

The result is a single LeRobot v2.1 leaf with `meta/info.json`, `tasks.jsonl`,
`episodes.jsonl`, and `episodes_stats.jsonl`. Its feature declaration contains
only the three common RGB cameras and 14-dimensional state/action. A conversion
manifest records the output episode ID, input class, source leaf, station when
available, source chunk/episode ID, frame count, and trajectory hash.

## Safety and failure handling

The builder refuses to overwrite an existing `base_clean`. It first writes to a
uniquely named hidden sibling staging directory under `Task_N`. Any validation,
write, or verification failure leaves the published target absent and reports
the precise source identity. After all checks pass, the staging directory is
renamed atomically to `base_clean`.

Neither TOS staging nor local v5 source artifacts are modified. Hard-linked
videos remain valid independently of their source directory entries.

## Test and verification strategy

Implementation follows test-driven development. Focused fixtures will first
define and fail on the required behaviors: compound source identity discovery,
32D-to-14D projection, deterministic global reindexing, canonical video
placement, metadata totals, and refusal to overwrite an existing destination.
The minimal builder implementation will then make those tests pass.

Before publication, a real-data dry run must reproduce the preflight totals.
After the build, verification must confirm:

- 822 parquet files, 649,636 rows, and 2,466 videos;
- episode IDs `0..821`, global frame indices `0..649635`, and per-episode frame
  indices/timestamps that are contiguous and consistent with 30 FPS;
- state/action shape 14 and finite values in every output parquet;
- every output video exists and is readable;
- metadata totals and per-episode lengths match physical artifacts;
- the source manifest covers every output episode exactly once; and
- the published directory is a complete atomic rename of the verified staging
  directory.

The existing read-only audit can then be run on `base_clean` as an additional
report. Its trajectory warnings are quality signals rather than structural
merge failures; integrity or artifact mismatches are release blockers.
