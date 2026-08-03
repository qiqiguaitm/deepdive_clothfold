from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from data_tools.inventory import classify
from data_tools.lerobot import DatasetLayout, discover_episodes, read_jsonl, write_jsonl


class DataToolsTest(unittest.TestCase):
    def test_layout_and_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = DatasetLayout(root)
            parquet = layout.parquet(7, chunk=2)
            parquet.parent.mkdir(parents=True)
            parquet.touch()
            self.assertEqual(discover_episodes(root), [7])
            self.assertIn("observation.images.hand_left", str(layout.video("hand_left", 7, 2)))

    def test_jsonl_roundtrip_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta" / "episodes.jsonl"
            rows = [{"episode_id": 1, "prompt": "折叠"}, {"episode_id": 2}]
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())

    def test_inventory_categories(self) -> None:
        self.assertEqual(classify("build_task_a.py"), "build")
        self.assertEqual(classify("validate_episode_pts.py"), "validate")
        self.assertEqual(classify("reset_video_pts.py"), "repair")

    def test_normalize_reindexes_agilex_legacy_video_layout(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        from data_tools.normalize import normalize

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            target = base / "target"
            layout = DatasetLayout(source)
            parquet = layout.parquet(4)
            parquet.parent.mkdir(parents=True)
            pq.write_table(
                pa.table({
                    "action": [[0.1, 0.2], [0.2, 0.3]],
                    "observation.state": [[1.0, 2.0], [1.1, 2.1]],
                }),
                parquet,
            )
            (source / "meta").mkdir()
            info = {
                "fps": 30,
                "features": {
                    "action": {"dtype": "float32", "shape": [2]},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "observation.images.top_head": {"dtype": "video", "shape": [480, 640, 3]},
                },
            }
            (source / "meta" / "info.json").write_text(json.dumps(info))
            write_jsonl(source / "meta" / "episodes.jsonl", [{"episode_id": 4, "operator": "test"}])
            # AgileX archive used videos/chunk-000/<camera>/..., without the
            # canonical observation.images prefix on the source side.
            for camera in ("top_head", "hand_left", "hand_right"):
                video = source / "videos" / "chunk-000" / camera / "episode_000004.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"fixture")

            result = normalize([source], target, task="pick and place")
            self.assertEqual(result["episodes"], 1)
            output = pq.read_table(DatasetLayout(target).parquet(0))
            self.assertEqual(output["episode_index"].to_pylist(), [0, 0])
            self.assertEqual(output["frame_index"].to_pylist(), [0, 1])
            self.assertTrue(DatasetLayout(target).video("hand_left", 0).exists())
            self.assertEqual(read_jsonl(target / "meta" / "tasks.jsonl")[0]["task"], "pick and place")

    def test_static_segments_include_interior_run_at_50_frames(self) -> None:
        import numpy as np
        from data_tools.static_segments import detect_static_segments

        # moving 10, static 50, moving 10: the middle run must be reported with
        # exact inclusive boundaries rather than only an episode-level count.
        values = np.zeros((70, 14), dtype=np.float64)
        values[:10] = np.arange(10)[:, None] * 0.01
        values[10:60] = values[9]
        values[60:] = values[9] + np.arange(1, 11)[:, None] * 0.01
        segments = detect_static_segments(values, episode_id=3, min_frames=50)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].position, "interior")
        self.assertEqual(segments[0].start_frame, 10)
        self.assertEqual(segments[0].end_frame, 59)
        self.assertEqual(segments[0].frames, 50)

    def test_flicker_metrics_separate_rolling_bands_from_stable_rows(self) -> None:
        import numpy as np
        from data_tools.flicker import flicker_metrics

        frames, rows = 120, 24
        luma = np.full(frames, 100.0)
        stable = np.zeros((frames, rows))
        y = np.arange(rows)
        rolling = np.asarray([
            12.0 * np.sin(2 * np.pi * (y / rows + frame / 6.0))
            for frame in range(frames)
        ])
        stable_metrics = flicker_metrics(luma, stable, 30.0)
        rolling_metrics = flicker_metrics(luma, rolling, 30.0)
        self.assertLess(stable_metrics["row_delta_median"], 0.1)
        self.assertGreater(rolling_metrics["row_delta_median"], 3.0)
        self.assertGreater(rolling_metrics["row_low_correlation_fraction"], 0.02)

    def test_audit_date_selection_supports_exact_and_range(self) -> None:
        from data_tools.audit import discover_leaves

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("2026-07-21-v4", "2026-07-23-v4", "not-dated"):
                parquet = root / name / "data/chunk-000/episode_000000.parquet"
                parquet.parent.mkdir(parents=True)
                parquet.touch()
            quarantined = root / ".deleted/2026-07-23-v4/data/chunk-000/episode_000001.parquet"
            quarantined.parent.mkdir(parents=True)
            quarantined.touch()
            exact = discover_leaves(root, dates=["2026-07-23"])
            self.assertEqual([path.name for path in exact], ["2026-07-23-v4"])
            ranged = discover_leaves(root, date_from="2026-07-20", date_to="2026-07-22")
            self.assertEqual([path.name for path in ranged], ["2026-07-21-v4"])

    def test_trajectory_spike_is_reported(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow not installed")
        from data_tools.audit import inspect_trajectory

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode_000001.parquet"
            values = [[i * 0.001, 0.0] for i in range(20)]
            values[10][0] += 0.5
            pq.write_table(pa.table({"observation.state": values}), path)
            issues = inspect_trajectory(path, 1)
            self.assertTrue(any(row.issue == "trajectory_velocity_spike" for row in issues))


if __name__ == "__main__":
    unittest.main()
