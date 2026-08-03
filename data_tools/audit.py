"""Read-only, date-selectable audit for KAI0 LeRobot dataset leaves."""
from __future__ import annotations
import json, re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable
import numpy as np
from .flicker import inspect_video
from .lerobot import discover_episodes
from .quality import scan_dataset
from .static_segments import scan_static_segments

DATE_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:-[^/]*)?$")

@dataclass(frozen=True)
class TrajectoryIssue:
    episode_id: int
    issue: str
    frame: int
    score: float

def _leaf_date(path: Path) -> date | None:
    match = DATE_RE.search(path.name)
    if not match: return None
    try: return date.fromisoformat(match.group("date"))
    except ValueError: return None

def discover_leaves(root: Path, *, dates: Iterable[str] = (), date_from: str | None = None,
                    date_to: str | None = None) -> list[Path]:
    """Find dataset leaves and select exact dates or an inclusive date range."""
    root = root.resolve()
    candidates = {
        p.parents[2]
        for p in root.glob("**/data/chunk-*/episode_*.parquet")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    }
    if (root / "data").is_dir(): candidates.add(root)
    exact = {date.fromisoformat(value) for value in dates}
    lower = date.fromisoformat(date_from) if date_from else None
    upper = date.fromisoformat(date_to) if date_to else None
    if lower and upper and lower > upper: raise ValueError("--date-from must not be after --date-to")
    selected = []
    for leaf in candidates:
        value = _leaf_date(leaf)
        if exact and value not in exact: continue
        if lower and (value is None or value < lower): continue
        if upper and (value is None or value > upper): continue
        selected.append(leaf)
    return sorted(selected)

def inspect_trajectory(parquet: Path, episode_id: int) -> list[TrajectoryIssue]:
    """Flag isolated velocity/acceleration spikes using robust per-dimension limits."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet, columns=["observation.state"])
    values = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    if len(values) < 4 or values.ndim != 2: return []
    velocity = np.diff(values, axis=0)
    acceleration = np.diff(velocity, axis=0)
    issues = []
    for name, signal, offset in (("trajectory_velocity_spike", velocity, 1),
                                 ("trajectory_acceleration_spike", acceleration, 2)):
        magnitude = np.max(np.abs(signal), axis=1)
        median = float(np.median(magnitude)); mad = float(np.median(np.abs(magnitude - median)))
        threshold = max(median + 12.0 * 1.4826 * mad, 0.08 if "velocity" in name else 0.12)
        for index in np.flatnonzero(magnitude > threshold):
            issues.append(TrajectoryIssue(episode_id, name, int(index + offset),
                                          round(float(magnitude[index]), 8)))
    return issues

def inspect_blur(video: Path, *, max_frames: int = 180, threshold: float = 35.0) -> dict:
    """Estimate blur from grayscale gradient energy."""
    import av
    scores = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            gray = frame.to_ndarray(format="gray")[::4, ::4].astype(np.float32)
            scores.append(float((np.diff(gray, axis=1).var() + np.diff(gray, axis=0).var()) / 2.0))
            if max_frames and len(scores) >= max_frames: break
    median = float(np.median(scores)) if scores else 0.0
    low_fraction = float(np.mean(np.asarray(scores) < threshold)) if scores else 1.0
    return {"video": str(video), "frames": len(scores), "median_sharpness": median,
            "blurred_fraction": low_fraction, "flagged": not scores or low_fraction >= 0.5}

def _sample(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or len(items) <= count: return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[i] for i in sorted(set(indices.tolist()))]

def audit_leaf(leaf: Path, *, visual_sample: int = 12, min_static_frames: int = 50,
               mains_hz: float = 50.0, max_visual_frames: int = 180) -> dict:
    episodes = discover_episodes(leaf)
    camera_dirs = sorted({path.parent.name.removeprefix("observation.images.")
                          for path in leaf.glob("videos/chunk-*/*/episode_*.mp4")
                          if "depth" not in path.parent.name})
    quality = scan_dataset(leaf, cameras=camera_dirs)
    static = scan_static_segments(leaf, min_frames=min_static_frames)
    trajectory = []
    for episode in episodes:
        parquet = next(iter(leaf.glob(f"data/chunk-*/episode_{episode:06d}.parquet")))
        trajectory.extend(inspect_trajectory(parquet, episode))
    videos = _sample(sorted(leaf.glob("videos/chunk-*/*/episode_*.mp4")), visual_sample)
    flicker, blur = [], []
    for video in videos:
        try:
            fresult = inspect_video(video, mains_hz=mains_hz, max_frames=max_visual_frames)
            if fresult.flagged: flicker.append(asdict(fresult))
            bresult = inspect_blur(video, max_frames=max_visual_frames)
            if bresult["flagged"]: blur.append(bresult)
        except Exception as exc:
            blur.append({"video": str(video), "flagged": True, "error": f"{type(exc).__name__}: {exc}"})
    issues = {"integrity": [asdict(row) for row in quality if not row.good],
              "static_segments": [asdict(row) for row in static],
              "trajectory": [asdict(row) for row in trajectory], "flicker": flicker, "blur": blur}
    leaf_day = _leaf_date(leaf)
    return {"leaf": str(leaf), "date": leaf_day.isoformat() if leaf_day else None,
            "episodes": len(episodes), "visual_videos_checked": len(videos),
            "issue_counts": {key: len(value) for key, value in issues.items()}, "issues": issues}

def run_audit(root: Path, *, output: Path, dates: Iterable[str] = (), date_from: str | None = None,
              date_to: str | None = None, visual_sample: int = 12, min_static_frames: int = 50,
              mains_hz: float = 50.0, max_visual_frames: int = 180) -> dict:
    dates = tuple(dates)
    leaves = discover_leaves(root, dates=dates, date_from=date_from, date_to=date_to)
    if not leaves: raise ValueError("no dataset leaves matched the requested path/date selection")
    reports = [audit_leaf(leaf, visual_sample=visual_sample, min_static_frames=min_static_frames,
                          mains_hz=mains_hz, max_visual_frames=max_visual_frames) for leaf in leaves]
    keys = ("integrity", "static_segments", "trajectory", "flicker", "blur")
    report = {"format": "kai0.dataset-audit.v1", "root": str(root.resolve()),
              "selection": {"dates": list(dates), "date_from": date_from, "date_to": date_to},
              "leaves": len(reports), "episodes": sum(x["episodes"] for x in reports),
              "issue_counts": {key: sum(x["issue_counts"][key] for x in reports) for key in keys},
              "datasets": reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
