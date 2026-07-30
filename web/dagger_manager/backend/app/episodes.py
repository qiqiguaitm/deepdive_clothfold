"""History episode browsing + replay for dagger_manager.

Reads the layouts written by dagger_recorder_node:
    <DATA_ROOT>/<task>/<subset>/<date>-vN/
        ├── data/chunk-000/episode_NNNNNN.parquet          ← 单段 (Form C)
        ├── data/chunk-001/episode_NNNNNN.parquet          ← 拼接段 (直采 / 离线 stitch)
        ├── videos/chunk-0NN/<feature-key>/episode_NNNNNN.mp4
        ├── meta/episodes.jsonl                            ← chunk-000 的 meta
        └── meta/episodes_stitched.jsonl                   ← chunk-001 的 meta

一个 episode 唯一键 = (subset, date, chunk, episode_id) —— episode_id 在【每个
chunk 内】各自从 0 重排, 所以 chunk-000 #0 与 chunk-001 #0 是不同 episode, chunk
必须进 key。subset ∈ {dagger, inference}; chunk-001 目前只在 dagger 下。

All path building whitelists each URL component (SAFE_NAME) and verifies the
resolved path stays under DATA_ROOT — same traversal defense as data_manager.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from .stack import DATA_ROOT

SUBSETS = ("dagger", "inference")
C_CAM = ("top_head", "hand_left", "hand_right")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-.]+$")

# chunk → 该 chunk 的 meta 文件名。chunk-000 = 单段 (Form C); chunk-001 = 拼接段
# (dagger_recorder 直采 或 stitch_dagger_episodes.py 离线拼), 两者都用同一份
# episodes_stitched.jsonl 契约。未列出的 chunk 一律回退到 stitched (未来多 chunk 拼接)。
_META_BY_CHUNK = {0: "episodes.jsonl", 1: "episodes_stitched.jsonl"}


def _meta_name(chunk: int) -> str:
    return _META_BY_CHUNK.get(chunk, "episodes_stitched.jsonl")


def _cdir(chunk: int) -> str:
    return f"chunk-{int(chunk):03d}"


def _camera_video_path(date_root: Path, cam: str, ep: int, chunk: int = 0) -> Path:
    """Try `observation.images.<cam>` (v4+) then bare `<cam>` (v3)."""
    name = f"episode_{ep:06d}.mp4"
    seg = _cdir(chunk)
    p = date_root / "videos" / seg / f"observation.images.{cam}" / name
    if p.exists():
        return p
    return date_root / "videos" / seg / cam / name


def _camera_video_candidates(date_root: Path, cam: str, ep: int, chunk: int = 0) -> list[Path]:
    """All possible video paths for deletion."""
    name = f"episode_{ep:06d}.mp4"
    seg = _cdir(chunk)
    return [
        date_root / "videos" / seg / f"observation.images.{cam}" / name,
        date_root / "videos" / seg / cam / name,
    ]


def _camera_depth_candidates(date_root: Path, cam: str, ep: int, chunk: int = 0) -> list[Path]:
    """All possible depth paths for deletion (chunk-001 直采不写 depth, 但离线
    路径可能有 → 仍逐一探测删除)."""
    stem, zarr_n = f"episode_{ep:06d}", f"episode_{ep:06d}.zarr"
    seg = _cdir(chunk)
    return [
        date_root / "videos" / seg / f"observation.depth.{cam}" / zarr_n,
        date_root / "videos" / seg / f"observation.depth.{cam}" / (stem + ".zarr.zip"),
        date_root / "videos" / seg / f"observation.depth.{cam}" / (stem + ".mkv"),
        date_root / "videos" / seg / f"{cam}_depth" / zarr_n,
        date_root / "videos" / seg / f"{cam}_depth" / (stem + ".zarr.zip"),
        date_root / "videos" / seg / f"{cam}_depth" / (stem + ".mkv"),
    ]


def _safe(*parts: str) -> None:
    for p in parts:
        if not SAFE_NAME.match(p):
            raise HTTPException(400, f"unsafe path component: {p!r}")


_VER_RE = re.compile(r"v\d+$")


def _has_any_meta(dd: Path) -> bool:
    """A browsable date dir has at least one chunk's meta file (chunk-000 单段
    或 chunk-001 拼接段)。纯 chunk-001 直采日期没有 episodes.jsonl, 只 gate
    episodes.jsonl 会把它整个漏掉。"""
    return any((dd / "meta" / _meta_name(c)).is_file() for c in _META_BY_CHUNK)


def _iter_date_dirs(subset_root: Path):
    """Yield (date_leaf, date_dir) for BOTH layouts:
      - nested (2026-06-15+): <subset>/<vN>/<date>-vN/   (date dirs under a version dir)
      - legacy flat:          <subset>/<date>-v2/         (date dirs directly under subset)
    Only dirs containing at least one chunk's meta file are yielded."""
    if not subset_root.is_dir():
        return
    for child in sorted(subset_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if _VER_RE.fullmatch(child.name):
            # version dir → its children are the date dirs
            for dd in sorted(child.iterdir(), reverse=True):
                if dd.is_dir() and _has_any_meta(dd):
                    yield dd.name, dd
        elif _has_any_meta(child):
            yield child.name, child  # legacy flat date dir


def _date_root(task: str, subset: str, date: str) -> Path:
    _safe(task, subset, date)
    if subset not in SUBSETS:
        raise HTTPException(400, f"unknown subset {subset!r}")
    subset_root = DATA_ROOT / task / subset
    # Candidates: nested <subset>/<vN>/<date> (vN parsed from the date's own -vN
    # suffix, e.g. 2026-06-15-v3 → v3), then legacy flat <subset>/<date>.
    candidates: list[Path] = []
    m = re.search(r"-(v\d+)$", date)
    if m:
        candidates.append(subset_root / m.group(1) / date)
    candidates.append(subset_root / date)
    for c in candidates:
        full = c.resolve()
        if not str(full).startswith(str(DATA_ROOT.resolve())):
            raise HTTPException(400, "path escapes DATA_ROOT")
        if full.is_dir():
            return full
    # last resort: scan version dirs for a matching leaf
    if subset_root.is_dir():
        for vdir in subset_root.iterdir():
            if vdir.is_dir() and _VER_RE.fullmatch(vdir.name):
                full = (vdir / date).resolve()
                if str(full).startswith(str(DATA_ROOT.resolve())) and full.is_dir():
                    return full
    # nothing on disk — return the primary candidate (callers 404 on missing meta)
    return candidates[0].resolve()


def list_tasks() -> list[dict]:
    """Every Task_* dir under DATA_ROOT, with a flag for whether it has any
    dagger/inference data (so the UI can grey out empty tasks)."""
    out: list[dict] = []
    if not DATA_ROOT.is_dir():
        return out
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Task_"):
            has_data = any((d / s).is_dir() for s in SUBSETS)
            out.append({"task": d.name, "has_data": has_data})
    return out


def list_episodes(task: str = "Task_A") -> list[dict]:
    """Enumerate every episode across both subsets and all date dirs.

    Sorted newest-first by (date desc, episode_id desc) so the most recent
    captures bubble to the top of the UI list.
    """
    out: list[dict] = []
    task_root = DATA_ROOT / task
    if not task_root.is_dir():
        return out
    for subset in SUBSETS:
        for _leaf, date_dir in _iter_date_dirs(task_root / subset):
            for chunk in sorted(_META_BY_CHUNK):
                meta_fp = date_dir / "meta" / _meta_name(chunk)
                if not meta_fp.is_file():
                    continue
                video_dir = date_dir / "videos" / _cdir(chunk)
                for line in meta_fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ep = int(d.get("episode_id", -1))
                    if ep < 0:
                        continue
                    # Has at least the head-cam mp4? (v4+ feature-key first, then bare v3 key)
                    head_mp4_v4 = video_dir / "observation.images.top_head" / f"episode_{ep:06d}.mp4"
                    head_mp4_v3 = video_dir / "top_head" / f"episode_{ep:06d}.mp4"
                    out.append({
                        "subset": subset,
                        "date": date_dir.name,
                        "chunk": chunk,
                        "episode_id": ep,
                        "length": int(d.get("length", 0)),
                        "duration_s": float(d.get("duration_s", 0.0)),
                        "operator": d.get("operator", ""),
                        "prompt": d.get("prompt", ""),
                        "success": bool(d.get("success", True)),
                        "note": d.get("note", ""),
                        "created_at": d.get("created_at"),
                        "has_video": head_mp4_v4.is_file() or head_mp4_v3.is_file(),
                        # 拼接段特有: 人接管次数 + 人控帧数 (chunk-000 无此键 → 缺省)
                        "n_takeovers": d.get("n_takeovers"),
                        "human_frames": d.get("human_frames"),
                        # 油门加速标识: 本段 rollout 是否踩过油门 (整段标记) + 峰值倍率
                        "used_throttle": bool(d.get("used_throttle", False)),
                        "speed_factor": float(d.get("speed_factor", 1.0)),
                    })
    # newest-first; chunk breaks ties so chunk-001 (拼接段) 排在同号 chunk-000 之上
    out.sort(key=lambda e: (e["date"], e["episode_id"], e["chunk"]), reverse=True)
    return out


def episode_video_path(task: str, subset: str, date: str, episode_id: int,
                       camera: str, chunk: int = 0) -> Path:
    if camera not in C_CAM:
        raise HTTPException(400, f"unknown camera {camera!r}")
    root = _date_root(task, subset, date)
    return _camera_video_path(root, camera, episode_id, chunk)


def episode_meta(task: str, subset: str, date: str, episode_id: int,
                 chunk: int = 0) -> dict:
    meta_fp = _date_root(task, subset, date) / "meta" / _meta_name(chunk)
    if not meta_fp.is_file():
        raise HTTPException(404, "meta not found")
    for line in meta_fp.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(d.get("episode_id", -1)) == episode_id:
            return d
    raise HTTPException(404, "episode not found")


def delete_episode(task: str, subset: str, date: str, episode_id: int,
                   chunk: int = 0) -> None:
    """Remove parquet + per-camera mp4 + meta line for ONE chunk's episode.
    Irreversible. chunk-001 (拼接段) 删的是整条拼接 episode, 不影响 chunk-000。"""
    root = _date_root(task, subset, date)
    pq = root / "data" / _cdir(chunk) / f"episode_{episode_id:06d}.parquet"
    if pq.exists():
        pq.unlink()
    for cam in C_CAM:
        for vp in _camera_video_candidates(root, cam, episode_id, chunk):
            if vp.exists():
                vp.unlink()
        import shutil as _sh
        for dp in _camera_depth_candidates(root, cam, episode_id, chunk):
            if dp.exists():
                if dp.suffix == ".zarr" and dp.is_dir():
                    _sh.rmtree(dp, ignore_errors=True)
                else:
                    dp.unlink()
    meta_fp = root / "meta" / _meta_name(chunk)
    if meta_fp.is_file():
        keep: list[str] = []
        for line in meta_fp.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if int(d.get("episode_id", -1)) == episode_id:
                    continue
            except json.JSONDecodeError:
                pass
            keep.append(line)
        meta_fp.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
