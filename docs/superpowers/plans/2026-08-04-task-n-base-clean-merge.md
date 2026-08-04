# Task_N `base_clean` Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish one verified LeRobot v2.1 `base_clean` leaf containing all 481 TOS episodes and all 341 local cleaned episodes.

**Architecture:** A Task_N-specific recipe discovers physical artifacts through compound source identities, validates and fingerprints every trajectory, projects state/action to 14D, and writes a globally reindexed staging dataset. It reuses `data_tools.lerobot` for layout and safe video placement, then verifies the complete staging tree before an atomic rename.

**Tech Stack:** Python 3.11, unittest, NumPy, pandas, PyArrow, PyAV, existing `data_tools.lerobot` helpers, TOS staging on vePFS.

---

## File structure

- Create `train_scripts/kai/data/build_task_n_base_clean.py` for source discovery, validation, projection, fingerprinting, construction, verification, and atomic publication.
- Create `train_scripts/kai/data/test_build_task_n_base_clean.py` for unit and fixture-level tests.
- Use `docs/superpowers/specs/2026-08-04-task-n-base-clean-merge-design.md` as the behavior contract.
- Generate `kai0/data/Task_N/base_clean/conversion_manifest.json` and standard LeRobot metadata at runtime; dataset artifacts are not committed.

### Task 1: Define compound source discovery

**Files:**
- Create: `train_scripts/kai/data/test_build_task_n_base_clean.py`
- Create: `train_scripts/kai/data/build_task_n_base_clean.py`

- [ ] **Step 1: Write the failing discovery test**

Create fixture helpers that write one parquet, one metadata row, and three legacy-layout video files. Load the not-yet-created builder through `importlib` and fail explicitly with `builder module missing` when absent:

```python
def test_discovery_orders_tos_before_local_and_preserves_compound_identity(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tos = root / "tos"
        local = root / "local"
        make_leaf(tos / "2026-07-29-v2", episode_id=7, width=14)
        make_leaf(local / "2026-07-31-v5", episode_id=7, width=32, station="ipc01")
        builder = load_builder()
        episodes = builder.discover_inputs(tos, local)
        self.assertEqual([item.source_kind for item in episodes], ["tos", "local_v5"])
        self.assertEqual(len({item.identity for item in episodes}), 2)
        self.assertEqual(episodes[0].expected_width, 14)
        self.assertEqual(episodes[1].expected_width, 32)
```

- [ ] **Step 2: Run the discovery test and verify RED**

```bash
cd /home/tim/workspace/deepdive_kai0
kai0/.venv/bin/python -m unittest \
  train_scripts.kai.data.test_build_task_n_base_clean.TaskNBaseCleanTest.test_discovery_orders_tos_before_local_and_preserves_compound_identity -v
```

Expected: `FAIL` containing `builder module missing`.

- [ ] **Step 3: Implement the minimal source model and discovery**

Create this interface:

```python
@dataclass(frozen=True)
class SourceEpisode:
    source_kind: str
    leaf: str
    station: str | None
    chunk: str
    source_episode_id: int
    expected_width: int
    parquet: Path
    videos: tuple[Path, Path, Path]
    source_meta: dict

    @property
    def identity(self) -> str:
        station = self.station or "unknown"
        return (
            f"{self.source_kind}/{self.leaf}/{station}/{self.chunk}/"
            f"episode_{self.source_episode_id:06d}"
        )


def discover_inputs(tos_root: Path, local_root: Path) -> list[SourceEpisode]:
    tos = _discover_tree(tos_root, "*-v2", "tos", 14)
    local = _discover_tree(local_root, "*-v5", "local_v5", 32)
    episodes = tos + local
    identities = [item.identity for item in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate compound source identity")
    return episodes
```

`_discover_tree` must enumerate actual `data/chunk-*/episode_*.parquet` files, map metadata by `(chunk, episode_id)`, support both `episodes.jsonl` and `multistation_episodes.jsonl`, and resolve all three cameras in canonical then legacy layout. Reject a missing metadata row or required video.

- [ ] **Step 4: Run the discovery test and verify GREEN**

Run the Step 2 command. Expected: one test passes.

- [ ] **Step 5: Commit source discovery**

```bash
git add train_scripts/kai/data/build_task_n_base_clean.py \
  train_scripts/kai/data/test_build_task_n_base_clean.py
git commit -m "feat(data): discover Task_N base_clean sources"
```

### Task 2: Project, reindex, and fingerprint tables

**Files:**
- Modify: `train_scripts/kai/data/build_task_n_base_clean.py`
- Modify: `train_scripts/kai/data/test_build_task_n_base_clean.py`

- [ ] **Step 1: Write failing transform tests**

```python
def test_transform_projects_joint14_and_reindexes(self):
    builder = load_builder()
    table = make_table(rows=3, width=32, episode_id=9)
    output, digest = builder.transform_table(table, new_episode=2, global_offset=10)
    self.assertEqual(output["episode_index"].to_pylist(), [2, 2, 2])
    self.assertEqual(output["frame_index"].to_pylist(), [0, 1, 2])
    self.assertEqual(output["index"].to_pylist(), [10, 11, 12])
    np.testing.assert_allclose(output["timestamp"].to_numpy(), [0.0, 1 / 30, 2 / 30])
    self.assertEqual(len(output["observation.state"][0].as_py()), 14)
    self.assertEqual(len(output["action"][0].as_py()), 14)
    self.assertEqual(digest, builder.trajectory_sha256(output))
```

Add separate rejection tests for non-contiguous `frame_index`, a non-finite first-14D value, and an unexpected source width.

- [ ] **Step 2: Run transform tests and verify RED**

```bash
kai0/.venv/bin/python -m unittest \
  train_scripts.kai.data.test_build_task_n_base_clean.TaskNBaseCleanTest.test_transform_projects_joint14_and_reindexes -v
```

Expected: `FAIL` because `transform_table` is absent.

- [ ] **Step 3: Implement validation, projection, and fingerprinting**

```python
KEEP_COLUMNS = (
    "observation.state", "action", "timestamp", "frame_index",
    "episode_index", "index", "task_index",
)
FPS = 30


def trajectory_sha256(table: pa.Table) -> str:
    state = np.asarray(table["observation.state"].to_pylist(), dtype="<f4")
    action = np.asarray(table["action"].to_pylist(), dtype="<f4")
    digest = hashlib.sha256()
    digest.update(np.asarray([len(table)], dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(state[:, :14]).tobytes())
    digest.update(np.ascontiguousarray(action[:, :14]).tobytes())
    return digest.hexdigest()


def transform_table(table, new_episode, global_offset, expected_width=None):
    missing = [name for name in KEEP_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    width = expected_width if expected_width is not None else state.shape[1]
    if state.shape != (len(table), width) or action.shape != (len(table), width):
        raise ValueError(f"unexpected state/action shape: {state.shape}/{action.shape}")
    if not np.isfinite(state[:, :14]).all() or not np.isfinite(action[:, :14]).all():
        raise ValueError("non-finite state/action in first 14 dimensions")
    source_frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(source_frames, np.arange(len(table))):
        raise ValueError("non-contiguous source frame_index")
    frame = table.select(KEEP_COLUMNS).to_pandas()
    frame["observation.state"] = [row[:14] for row in state]
    frame["action"] = [row[:14] for row in action]
    frame["timestamp"] = np.arange(len(frame), dtype=np.float32) / FPS
    frame["frame_index"] = np.arange(len(frame), dtype=np.int64)
    frame["episode_index"] = np.full(len(frame), new_episode, dtype=np.int64)
    frame["index"] = np.arange(global_offset, global_offset + len(frame), dtype=np.int64)
    frame["task_index"] = np.zeros(len(frame), dtype=np.int64)
    output = pa.Table.from_pandas(frame, preserve_index=False)
    return output, trajectory_sha256(output)
```

- [ ] **Step 4: Run all current tests**

```bash
kai0/.venv/bin/python -m unittest train_scripts.kai.data.test_build_task_n_base_clean -v
```

Expected: discovery, rejection, projection, fingerprint, and index tests pass.

- [ ] **Step 5: Commit table transformation**

```bash
git add train_scripts/kai/data/build_task_n_base_clean.py \
  train_scripts/kai/data/test_build_task_n_base_clean.py
git commit -m "feat(data): project and reindex Task_N episodes"
```

### Task 3: Build and atomically publish a fixture dataset

**Files:**
- Modify: `train_scripts/kai/data/build_task_n_base_clean.py`
- Modify: `train_scripts/kai/data/test_build_task_n_base_clean.py`

- [ ] **Step 1: Write the failing end-to-end fixture test**

```python
def test_build_dataset_writes_verified_atomic_leaf(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tos = root / "tos"
        local = root / "local"
        destination = root / "Task_N" / "base_clean"
        tos_video = make_leaf(tos / "2026-07-29-v2", 1, 14)
        make_leaf(local / "2026-07-31-v5", 2, 32, station="ipc01")
        builder = load_builder()
        result = builder.build_and_publish(tos, local, destination)
        self.assertEqual(result, {"episodes": 2, "frames": 6, "videos": 6})
        info = json.loads((destination / "meta/info.json").read_text())
        self.assertEqual(info["features"]["action"]["shape"], [14])
        manifest = json.loads((destination / "conversion_manifest.json").read_text())
        self.assertEqual([row["source_kind"] for row in manifest["episodes"]], ["tos", "local_v5"])
        linked = destination / "videos/chunk-000/observation.images.top_head/episode_000000.mp4"
        self.assertEqual(tos_video.stat().st_ino, linked.stat().st_ino)
        with self.assertRaises(FileExistsError):
            builder.build_and_publish(tos, local, destination)
```

- [ ] **Step 2: Run the fixture test and verify RED**

```bash
kai0/.venv/bin/python -m unittest \
  train_scripts.kai.data.test_build_task_n_base_clean.TaskNBaseCleanTest.test_build_dataset_writes_verified_atomic_leaf -v
```

Expected: `FAIL` because `build_and_publish` is absent.

- [ ] **Step 3: Implement dataset writing and metadata**

Implement `build_and_publish(tos_root, local_root, destination, dry_run=False, expected_episodes=None, expected_frames=None)` with this sequence:

1. Refuse an existing destination.
2. Discover and validate all sources.
3. Calculate trajectory hashes and reject any duplicate hash.
4. Return counts without creating output when `dry_run=True`.
5. Create a hidden sibling through `tempfile.mkdtemp(prefix=".base_clean.building-", dir=destination.parent)`.
6. Write zstd parquet files and place videos through `data_tools.lerobot.place_file(source_video, target_video, mode="hardlink")`.
7. Write `info.json`, `tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`, and:

```python
manifest = {
    "format": "lerobot-v2.1",
    "task": "nail painting",
    "fps": 30,
    "source_order": ["tos", "local_v5"],
    "episodes": manifest_rows,
    "totals": {
        "episodes": len(episodes),
        "frames": total_frames,
        "videos": len(episodes) * 3,
    },
}
```

8. Verify staging.
9. Publish through `os.replace(staging, destination)` only after successful verification.

On failure, print the exact unpublished staging path and source identity.

- [ ] **Step 4: Implement structural verification**

`verify_dataset(root, expected_episodes, expected_frames, decode_videos=False)` must assert:

```python
assert episode_ids == list(range(expected_episodes))
assert total_rows == expected_frames
assert global_indices == list(range(expected_frames))
assert required_video_count == expected_episodes * len(CAMERAS)
assert len(manifest["episodes"]) == expected_episodes
assert info["total_episodes"] == expected_episodes
assert info["total_frames"] == expected_frames
```

When `decode_videos=True`, open every video with PyAV and decode its first frame; identify any empty or unreadable path.

- [ ] **Step 5: Run the full test module and verify GREEN**

```bash
kai0/.venv/bin/python -m unittest train_scripts.kai.data.test_build_task_n_base_clean -v
```

Expected: all discovery, validation, transform, build, metadata, hard-link, verification, and overwrite tests pass.

- [ ] **Step 6: Commit the complete builder**

```bash
git add train_scripts/kai/data/build_task_n_base_clean.py \
  train_scripts/kai/data/test_build_task_n_base_clean.py
git commit -m "feat(data): build verified Task_N base_clean dataset"
```

### Task 4: Dry-run and build the real dataset

**Files:**
- Runtime input: `.runtime/task_n_base_clean_merge/20260804.eRdaa3/tos_full/base`
- Runtime input: `kai0/data/Task_N/base/v5`
- Runtime output: `kai0/data/Task_N/base_clean`

- [ ] **Step 1: Confirm target absence and source counts**

```bash
cd /home/tim/workspace/deepdive_kai0
test ! -e kai0/data/Task_N/base_clean
find .runtime/task_n_base_clean_merge/20260804.eRdaa3/tos_full/base \
  -path '*/data/chunk-*/episode_*.parquet' -type f | wc -l
find kai0/data/Task_N/base/v5 \
  -path '*/data/chunk-*/episode_*.parquet' -type f | wc -l
```

Expected: `481` and `341`.

- [ ] **Step 2: Run the real-data dry run**

```bash
KAI0_ROOT=/home/tim/workspace/deepdive_kai0/kai0 \
TOS_TASK_N_ROOT=/home/tim/workspace/deepdive_kai0/.runtime/task_n_base_clean_merge/20260804.eRdaa3/tos_full/base \
kai0/.venv/bin/python train_scripts/kai/data/build_task_n_base_clean.py --dry-run
```

Expected:

```json
{"episodes": 822, "frames": 649636, "videos": 2466, "duplicates": 0}
```

- [ ] **Step 3: Build and atomically publish**

Run the Step 2 command without `--dry-run`. Expected: `PUBLISHED` with 822 episodes, 649,636 frames, and 2,466 videos.

- [ ] **Step 4: Decode every output video and revalidate tables**

```bash
KAI0_ROOT=/home/tim/workspace/deepdive_kai0/kai0 \
kai0/.venv/bin/python train_scripts/kai/data/build_task_n_base_clean.py \
  --verify-only --decode-videos
```

Expected: `VERIFY_OK` with all exact totals.

### Task 5: Audit and hand off

**Files:**
- Create runtime report: `.runtime/task_n_base_clean_merge/20260804.eRdaa3/base_clean_audit.json`
- Update execution notes: `.planning/2026-08-04-task-n-base-clean-822-merge/`

- [ ] **Step 1: Run the read-only audit on the published leaf**

```bash
cd /home/tim/workspace/deepdive_kai0
kai0/.venv/bin/python -m data_tools audit kai0/data/Task_N/base_clean \
  --output .runtime/task_n_base_clean_merge/20260804.eRdaa3/base_clean_audit.json
```

Expected: one leaf, 822 episodes, and zero integrity failures. Trajectory warnings remain quality signals rather than structural merge failures.

- [ ] **Step 2: Record final structure and disk behavior**

```bash
find kai0/data/Task_N/base_clean -maxdepth 4 -type d | sort
find kai0/data/Task_N/base_clean/data -type f -name '*.parquet' | wc -l
find kai0/data/Task_N/base_clean/videos -type f -name '*.mp4' | wc -l
du -sh kai0/data/Task_N/base_clean
```

Expected: 822 parquet files and 2,466 videos. Note that hard links share data blocks with sources even if `du` attributes apparent size to the output tree.

- [ ] **Step 3: Deliver verified paths and totals**

Report the published path, source ordering, exact counts, local 32D-to-14D projection, duplicate count, audit integrity result, manifest path, and retained recoverable TOS staging path.
