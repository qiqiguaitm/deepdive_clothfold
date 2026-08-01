"""LeRobot v2.1 episode writer + meta-update helpers.

Extracted from recorder.py so both the teleop backend (FastAPI-driven) and the
autonomy ROS2 recorder node can produce binary-identical dataset bytes.

NO FastAPI / ros_bridge / pydantic dependencies — safe to import from any venv
that has av + pyarrow + zarr.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import zarr
    _HAS_ZARR = True
    _ZARR_MAJOR = int(zarr.__version__.split(".")[0])
except ImportError:
    _HAS_ZARR = False
    _ZARR_MAJOR = 0

try:
    import numcodecs
    _HAS_NUMCODECS = True
except ImportError:
    _HAS_NUMCODECS = False

from .depth_archive import (
    convert_zarr_dir_to_ffv1,
    ffv1_path_for,
    pack_zarr_dir,
    pending_path_for,
    zip_path_for,
)
from .eef_kinematics import (
    STATE_DIM as EEF_STATE_DIM,
    append_absolute_eef,
    apply_relative_eef_actions,
    write_modality_json,
)
from .layout import compound_to_subset_root, new_task_subset_root, today_compound


def _open_depth_zarr(path, h, w):
    """Open a uint16 depth zarr array shape (0, h, w) chunks (1, h, w) with
    blosc/zstd/bitshuffle. Returns a zarr.Array that supports append-via-resize.

    Forces on-disk format zarr_format=2 so depth files are interchangeable
    with teleop's data_manager backend (which is pinned to zarr 2.x).
    """
    if _ZARR_MAJOR >= 3:
        # numcodecs.Blosc is the underlying compressor, works across versions.
        # `compressors=` (plural, list) is the zarr 3 signature.
        comp = numcodecs.Blosc(cname="zstd", clevel=3,
                               shuffle=numcodecs.Blosc.BITSHUFFLE) if _HAS_NUMCODECS else None
        return zarr.create_array(
            store=str(path), shape=(0, h, w), chunks=(1, h, w),
            dtype="uint16", zarr_format=2, compressors=comp, overwrite=True,
        )
    # zarr 2.x — original API.
    try:
        comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    except Exception:
        comp = None
    return zarr.open(str(path), mode="w",
                     shape=(0, h, w), chunks=(1, h, w),
                     dtype="uint16", compressor=comp)


def _append_depth_frame(z, frame):
    """Append one (h, w) uint16 frame to a depth zarr array.

    Hides the zarr 2 vs 3 API split: 2.x has `.append()`; 3.x removed it,
    use `.resize()` + slice assignment instead.
    """
    if _ZARR_MAJOR >= 3:
        t = z.shape[0]
        z.resize((t + 1, z.shape[1], z.shape[2]))
        z[t] = frame
    else:
        z.append(frame[None, :, :])


def _load_depth_flags() -> tuple[str, ...]:
    """Read config/camera_depth_flags.py by probing upward from this file.

    The data_manager backend isn't a child of /config/, so a plain relative
    import won't reach it. Falls back to () if the macro file is missing.
    """
    here = Path(__file__).resolve()
    cams: tuple[str, ...] = ()
    for parent in here.parents:
        candidate = parent / "config" / "camera_depth_flags.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "kai0_camera_depth_flags", candidate)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            cams = tuple(mod.DEPTH_CAMERAS)
            break
    # Per-run head-depth override: KAI0_HEAD_DEPTH=0 (set by start_dagger_collect.sh
    # for V1/v0 dagger) drops top_head from the recorded set, so we don't subscribe
    # to / allocate a top_head depth zarr that would otherwise fill with zeros once
    # multi_camera stops publishing head depth. Env UNSET → keep the file default
    # (teleop path is unaffected). Only "0" disables; any other value is a no-op.
    if os.environ.get("KAI0_HEAD_DEPTH") == "0":
        cams = tuple(c for c in cams if c != "top_head")
    return cams


CAMERAS = (
    ("top_head", "mid_head", "hand_left", "hand_right")
    if os.environ.get("KAI0_ENABLE_MID_HEAD", "1") == "1"
    else ("top_head", "hand_left", "hand_right")
)
DEPTH_CAMERAS = _load_depth_flags()
FPS = 30
WIDTH = 640
HEIGHT = 480


def eef_recording_enabled() -> bool:
    """EEF schema is opt-in so existing v2/v3/v4 and DAgger writers stay 14-D."""
    return os.environ.get("KAI0_RECORD_EEF", "0") == "1"


def depth_ffv1_enabled() -> bool:
    """Online FFV1 is explicit so autonomy/DAgger legacy schemas stay unchanged."""
    return os.environ.get("KAI0_DEPTH_FFV1", "0") == "1"

# ── V3 online front-trim (leading-idle trim at record time) ──
# Same semantics/constants as train_scripts/kai/data/build_no_release.py
# (motion_onset + cut = max(0, onset - MARGIN)). Lets the collection pipeline
# emit V3 datasets directly instead of a post-hoc build_no_release pass.
TRIM_ARM_DIMS = list(range(0, 6)) + list(range(7, 13))  # 12 arm dims (exclude grippers 6,13)
TRIM_THR = 3e-3   # rad/frame: sustained mean |Δaction| over arm dims => "moving"
TRIM_WIN = 10     # frames of sustained motion to call it the onset
TRIM_MARGIN = 15  # keep this many frames before onset (lead-in; NOT a full delete)

# ── V3 online tail-trim (trailing post-task idle cap at record time) ──
# Mirrors build_no_release.py::tail_cap_keep_indices: a trailing frame is "idle"
# only when BOTH arm AND gripper are static, so a final gripper release/place is
# NEVER dropped; the long post-completion hold is capped to TAIL_CAP terminal
# settle frames. ONLY the trailing run is touched — interior idle streams as-is
# (no middle thinning), so per-chunk task-motion displacement is unchanged.
TRIM_GRIP_DIMS = [6, 13]   # L/R gripper action dims (excluded from TRIM_ARM_DIMS)
TRIM_GRIP_THR = 0.02       # |Δgrip| above this => gripper acting (grasp/release)
TAIL_CAP = 15              # keep this many trailing-idle frames as terminal settle (~0.5s @30Hz)


def _tail_keep_count(action: list[list[float]], tail_cap: int = TAIL_CAP) -> int:
    """Return the contiguous prefix length after terminal-idle capping.

    This is intentionally equivalent to
    build_no_release.tail_cap_keep_indices(), but returns only keep_end so the
    live writer can trim video/depth/parquet to one shared length.
    """
    total = len(action)
    if total <= 1:
        return total
    arr = np.asarray(action, dtype=np.float64)
    d_arm = np.abs(np.diff(arr[:, TRIM_ARM_DIMS], axis=0)).mean(axis=1)
    d_grip = np.abs(np.diff(arr[:, TRIM_GRIP_DIMS], axis=0)).max(axis=1)
    active = np.concatenate([
        np.asarray([True]),
        (d_arm > TRIM_THR) | (d_grip > TRIM_GRIP_THR),
    ])
    tail = 0
    for idx in range(total - 1, -1, -1):
        if active[idx]:
            break
        tail += 1
    return total if tail <= tail_cap else total - (tail - tail_cap)

# ── 人接管段 (dagger segment) 的裁剪口径 ──
# episode 级的 TRIM_MARGIN / TAIL_CAP 是"留一段 lead-in / terminal settle"的采集约定,
# 对【段内】的遥操静止伪影不适用: 离线 stitch_dagger_episodes.classify_segment 对 dag 段
# 是全裁的 (hes_end 无 margin, stat_start 无 cap) —— 人握上主臂还没动 / 松手前不动的那些帧
# 既不是策略行为也不是纠错行为, 一帧都不该留。常量与离线脚本同名同值。
#
# 关键: 起手判定必须【带夹爪】—— 人的第一个动作常常是先合爪抓住布料, 手臂几乎不动。
# 只看 12 个臂关节会把"抓取瞬间"当静止裁掉。离线同口径 (arm_slow AND grip_slow 才算迟疑)。
HESITATION_THR = 5e-3        # rad/frame — arm 低于此 = 迟疑/慢速起手
GRIP_HESITATION_THR = 0.01   # |Δgrip| 低于此 = 夹爪没在抓/放
HESITATION_WIN = 3           # 连续这么多帧超阈值 = 真正起手

# ── chunk-001 (直接采集拼接段) 的 dagger_frame_class 回溯打标窗 ──
# Sirius 人反应时间 0.75s (原文 ℓ=15 帧 @20Hz) → 30fps 需 22 帧。每个 0→1
# (policy→human) 边界之前这么多帧的 intervention==0 帧标 preintv(2) = 失败先兆。
# 语义/权重见 docs/training/analysis/chunk001_schema.md。
#
# ⚠️ 这里【没有】离线 stitch 的 COAST_TRIM: 那 15 帧滑行伪影是离线脚本假设
# "_do_takeover 期间 writer 仍在写" 才要裁的, 而 dagger_recorder_node 在发
# execute=false 之前就已切 ALIGNING → _on_record_tick 静默, 滑行帧从未落盘。
# 在线路径直接从最后一帧真实策略行为往前数 PREINTV_MARGIN 即可。
PREINTV_MARGIN = round(0.75 * FPS)   # =22 @30fps

log = logging.getLogger(__name__)

# One low-priority depth-finalizer for the whole backend.  New captures become
# lossless FFV1/gray16le MKV.  Serializing jobs prevents post-save work
# from competing with the live writer and recreating the old 10-20s drain stall.
_DEPTH_FINALIZE_QUEUE: queue.Queue[tuple[tuple[Path, ...], bool]] = queue.Queue()
_DEPTH_FINALIZE_THREAD: threading.Thread | None = None
_DEPTH_FINALIZE_THREAD_LOCK = threading.Lock()
_DEPTH_ENQUEUED: set[Path] = set()
_DEPTH_ENQUEUED_LOCK = threading.Lock()


def _finalize_depth_dirs(dirs: tuple[Path, ...] | list[Path], *, ffv1: bool) -> None:
    for dpath in dirs:
        dpath = Path(dpath)
        marker = pending_path_for(dpath)
        try:
            if not ffv1:
                pack_zarr_dir(dpath, remove_dir=True)
                continue
            before = sum(f.stat().st_size for f in dpath.rglob("*") if f.is_file())
            dst = convert_zarr_dir_to_ffv1(
                dpath,
                remove_dir=True,
                verify_pixels=os.environ.get("KAI0_DEPTH_VERIFY_PIXELS", "1") == "1",
                fps=FPS,
            )
            log.info(
                "[depth-ffv1] %s -> %s %.1fMB -> %.1fMB",
                dpath.name, dst.name, before / 1e6, dst.stat().st_size / 1e6,
            )
            marker.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            if ffv1:
                # Keep both source and marker. Startup recovery retries, while
                # TOS waits instead of mixing depth formats inside v5.
                log.error("depth FFV1 failed for %s; keeping pending job",
                          dpath, exc_info=True)
            else:
                log.error("depth zarr.zip pack failed for %s; keeping zarr dir",
                          dpath, exc_info=True)
        finally:
            with _DEPTH_ENQUEUED_LOCK:
                _DEPTH_ENQUEUED.discard(dpath)


def _depth_finalize_worker() -> None:
    # CPU and IO priority are per-thread on Linux when addressed by native TID.
    # Best effort: unsupported systems still benefit from serialization.
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass
    ionice = shutil.which("ionice")
    if ionice:
        try:
            subprocess.run(
                [ionice, "-c", "3", "-p", str(threading.get_native_id())],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
    while True:
        dirs, ffv1 = _DEPTH_FINALIZE_QUEUE.get()
        try:
            _finalize_depth_dirs(dirs, ffv1=ffv1)
        finally:
            _DEPTH_FINALIZE_QUEUE.task_done()


def _enqueue_depth_finalize(dirs: list[Path], *, ffv1: bool,
                            create_marker: bool | None = None) -> int:
    """Queue unique depth dirs and return the number of newly queued jobs."""
    global _DEPTH_FINALIZE_THREAD
    if create_marker is None:
        create_marker = ffv1
    queued: list[Path] = []
    with _DEPTH_ENQUEUED_LOCK:
        for raw in dirs:
            dpath = Path(raw)
            if not dpath.is_dir() or dpath in _DEPTH_ENQUEUED:
                continue
            if create_marker:
                marker = pending_path_for(dpath)
                marker.write_text(f"{time.time()}\n", encoding="utf-8")
            _DEPTH_ENQUEUED.add(dpath)
            queued.append(dpath)
    if not queued:
        return 0
    with _DEPTH_FINALIZE_THREAD_LOCK:
        if _DEPTH_FINALIZE_THREAD is None or not _DEPTH_FINALIZE_THREAD.is_alive():
            _DEPTH_FINALIZE_THREAD = threading.Thread(
                target=_depth_finalize_worker, name="depth-finalize-worker", daemon=True,
            )
            _DEPTH_FINALIZE_THREAD.start()
    _DEPTH_FINALIZE_QUEUE.put((tuple(queued), ffv1))
    return len(queued)


def recover_pending_depth_jobs(data_root: Path) -> int:
    """Resume only explicitly marked online jobs after a backend restart."""
    dirs: list[Path] = []
    for marker in data_root.glob("*/*/v*/*/videos/chunk-*/observation.depth.*/episode_*.zarr.ffv1.pending"):
        zarr_dir = Path(str(marker)[:-len(".ffv1.pending")])
        if zarr_dir.is_dir():
            dirs.append(zarr_dir)
        elif ffv1_path_for(zarr_dir).is_file() or zip_path_for(zarr_dir).is_file():
            marker.unlink(missing_ok=True)
    return _enqueue_depth_finalize(dirs, ffv1=True, create_marker=False)

# Queue sentinel for segment-boundary control ops (see EpisodeWriter.segment_control).
# A real tick's item[0] is the cam_arrs dict, so an `is` check can never collide.
_CTL = object()


def task_subset_root(task: str, subset: str) -> Path:
    """`<DATA_ROOT>/<task>/<subset>/<today-v2>` for new episodes (v2 layout)."""
    return new_task_subset_root(task, subset)


def pick_codec() -> tuple[str, str, dict]:
    """Pick video codec — h264 (default, broad compatibility), av1 (compact),
    or nvenc (GPU hardware H.264, keeps the mp4 encode off the CPU).

    KAI0_VIDEO_CODEC:
      h264   (default) — libx264 veryfast (CPU).
      av1              — libsvtav1 → libaom-av1 → falls back to h264 (CPU).
      nvenc | gpu      — h264_nvenc (GPU). Encodes on KAI0_NVENC_GPU (default
                         '0' = first CUDA-visible device); point it at an *idle*
                         card so the encode steals neither inference CPU cores
                         nor the inference GPU. Falls back to libx264 when the
                         linked PyAV/ffmpeg has no NVENC — e.g. kai0/.venv pins
                         av==13 (no nvenc), while backend/.venv PyAV 17 has it,
                         so the teleop recorder gets GPU encode and the dagger
                         recorder degrades gracefully to libx264.
    """
    choice = os.environ.get("KAI0_VIDEO_CODEC", "h264").lower()
    avail = set(av.codecs_available)
    if choice in ("nvenc", "gpu", "h264_nvenc"):
        if "h264_nvenc" in avail:
            gpu = os.environ.get("KAI0_NVENC_GPU", "0")
            # p4 = balanced preset; vbr+cq for constant-quality ≈ libx264 crf 23.
            return "h264", "h264_nvenc", {
                "preset": "p4", "tune": "ll", "rc": "vbr", "cq": "23",
                "gpu": str(gpu),
            }
        log.warning("h264_nvenc not in this PyAV build, falling back to libx264")
    if choice == "av1":
        if "libsvtav1" in avail:
            return "av1", "libsvtav1", {"preset": "8", "crf": "32"}
        if "libaom-av1" in avail:
            return "av1", "libaom-av1", {"cpu-used": "8", "crf": "32", "b:v": "0"}
        log.warning("AV1 encoder not found, falling back to libx264")
    return "h264", "libx264", {"preset": "veryfast", "crf": "23"}


class EpisodeWriter:
    """Single-episode disk writer: 3 mp4 containers + 1 parquet buffer.

    Optional depth zarr for cameras listed in DEPTH_CAMERAS (D435 head only
    by default).
    """

    def __init__(self, task: str, subset: str, ep: int, prompt: str,
                 template_id: str, operator: str,
                 front_trim: bool | None = None,
                 tail_trim: bool | None = None,
                 chunk: int = 0,
                 frame_class: bool = False,
                 record_depth: bool = True) -> None:
        self.task = task
        self.subset = subset
        self.ep = ep
        self.prompt = prompt
        self.template_id = template_id
        self.operator = operator
        # chunk-000 = 单段 episode (teleop / autonomy / 旧 Form-C 分段采集)。
        # chunk-001 = 一个 rollout 内 INF+DAG 连录的拼接段 (直接采集模式, 见
        # dagger_recorder_node 的 KAI0_DIRECT_CHUNK001)。离线 stitch 产的也是 001,
        # 两条路径的 on-disk 布局必须一致。
        self.chunk = int(chunk)
        self._frame_class = bool(frame_class)
        self._depth_ffv1 = depth_ffv1_enabled()
        _cdir = f"chunk-{self.chunk:03d}"

        self.root = task_subset_root(task, subset)
        self.pq_path = self.root / "data" / _cdir / f"episode_{ep:06d}.parquet"
        # Video/depth dirs are named by the LeRobot feature key (full name) so they match
        # info.json's {video_key} path templates. Using the short cam name here produced
        # top_head/ dirs that the loader (expecting observation.images.top_head/) silently
        # failed to find — the full-vs-short-name loader bug.
        self.video_paths = {
            cam: self.root / "videos" / _cdir / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            for cam in CAMERAS
        }
        # record_depth=False 完全不开 depth zarr (chunk-001 下游一帧不用, 而 depth
        # 是 dagger 采集磁盘占用的大头 — 实测 30G/9.8G RGB)。
        self.depth_paths = {
            cam: self.root / "videos" / _cdir / f"observation.depth.{cam}" / f"episode_{ep:06d}.zarr"
            for cam in (DEPTH_CAMERAS if record_depth else ())
        }
        for p in [self.pq_path.parent, *(v.parent for v in self.video_paths.values()),
                  *(d.parent for d in self.depth_paths.values())]:
            p.mkdir(parents=True, exist_ok=True)

        spec_name, codec_name, codec_opts = pick_codec()
        self._spec_name = spec_name
        self._codec_name = codec_name
        self._containers: dict[str, av.container.OutputContainer] = {}
        self._streams: dict[str, av.video.stream.VideoStream] = {}
        for cam, path in self.video_paths.items():
            container = av.open(str(path), mode="w")
            stream = container.add_stream(codec_name, rate=FPS)
            stream.width = WIDTH
            stream.height = HEIGHT
            stream.pix_fmt = "yuv420p"
            stream.options = dict(codec_opts)
            self._containers[cam] = container
            self._streams[cam] = stream

        # Encoder warmup: nvenc opens an encode session on its FIRST encode
        # (~0.3-0.6s per stream). That stall, paid in the capture loop, makes the
        # 30Hz recorder fall behind and drop the first ~0.5s of frames. Pay it HERE
        # in __init__ (recorder.start(), before the capture thread exists) instead.
        # _skip_packets cancels the one frame nvenc buffers from the warmup so the
        # output stays exactly N frames with 0-based PTS. libx264 has no such stall,
        # so warmup is nvenc-only.
        self._skip_packets: dict[str, int] = {}
        self._force_idr: dict[str, int] = {}
        if codec_name == "h264_nvenc" and os.environ.get("KAI0_ENCODER_WARMUP", "1") == "1":
            self._warmup_encoders()

        self._depth_arrays: dict[str, object] = {}
        if _HAS_ZARR:
            for cam, path in self.depth_paths.items():
                if path.exists():
                    shutil.rmtree(path, ignore_errors=True)
                self._depth_arrays[cam] = _open_depth_zarr(path, HEIGHT, WIDTH)
        else:
            log.warning("zarr not installed, depth recording disabled")

        self._rows_state: list[list[float]] = []
        self._rows_action: list[list[float]] = []
        self._rows_intervention: list[int] = []  # int8: 0=policy, 1=human, -1=N/A
        self._frame_idx = 0   # count of EMITTED (kept) frames → drives pts + parquet
                              # index + the frame_index/fps timestamp (all 0-based)

        # V3 front-trim: rolling-buffer the leading ticks until motion onset,
        # then flush [onset-MARGIN:] and stream the rest. Zero re-encode,
        # memory bounded to (MARGIN+WIN) frames. Default from KAI0_FRONT_TRIM
        # env (off unless a collection entry script opts in — keeps the
        # autonomy diagnostic recorder un-trimmed).
        if front_trim is None:
            front_trim = os.environ.get("KAI0_FRONT_TRIM", "0") == "1"
        self._front_trim = bool(front_trim)
        self._onset_found = not self._front_trim   # off → everything streams now
        self._buf: list[tuple] = []                # pending (cam_arrs, depth_arrs, s, a, ts, iv)
        self._prev_action: np.ndarray | None = None
        self._run = 0                              # consecutive-moving frame counter

        # Episode tail trim is applied at finalize to a contiguous on-disk prefix.
        # Streaming every frame here is essential: an online bounded buffer cannot
        # know whether a long idle interval is terminal or an interior pause, and
        # dropping it early creates visible time jumps if motion later resumes.
        if tail_trim is None:
            tail_trim = os.environ.get(
                "KAI0_TAIL_TRIM", "1" if self._front_trim else "0") == "1"
        self._tail_trim = bool(tail_trim)
        # DAgger human segments use their strict online buffer. Normal episodes
        # use a separate exact bounded tail candidate buffer below, with on-disk
        # trimming only as a fallback for unusually long terminal idles.
        self._segment_tail_trim = False
        self._tail_buf: list[tuple] = []
        self._tail_prev_action: np.ndarray | None = None  # DAgger segment anchor
        self._normal_tail_buf: list[tuple] = []
        self._normal_tail_prev_action: np.ndarray | None = None
        self._normal_tail_spilled = False
        # Raw RGB+depth is large, so cap the exact in-memory candidate tail. Five
        # seconds covers normal pedal reaction/settle; longer unattended idles
        # spill to disk and use the slower but lossless packet-copy fallback.
        self._normal_tail_limit = max(
            TAIL_CAP,
            int(os.environ.get("KAI0_TAIL_BUFFER_FRAMES", str(5 * FPS))),
        )

        # ── Trim profile ──
        # Defaults = the episode-level V3 convention (keep a TRIM_MARGIN lead-in and a
        # TAIL_CAP terminal settle). A multi-segment chunk-001 episode temporarily swaps
        # in the stricter per-segment profile at human-takeover boundaries, where a
        # lead-in/settle would just be teleop artifact — see segment_control().
        # Nothing but segment_control() ever changes these, so single-writer captures
        # (teleop / autonomy / 旧 Form C) behave byte-identically to before.
        self._front_margin = TRIM_MARGIN
        self._front_win = TRIM_WIN
        self._front_thr_arm = TRIM_THR
        self._front_thr_grip: float | None = None  # None = arm-only (legacy)
        self._front_keep: int | None = None        # None = flush whole buffer (legacy)
        self._tail_cap = TAIL_CAP

        # Optional async writer (KAI0_ASYNC_WRITER=1): the capture thread preps +
        # enqueues at 30Hz; this background thread drains the queue and does the
        # heavy front-trim + encode + depth compress, so a slow tick (NVENC/IO
        # stall, GIL contention) never stalls the grab loop → no record-time frame
        # drops. Off by default; sync mode is the legacy path with identical output.
        self._async = os.environ.get("KAI0_ASYNC_WRITER", "0") == "1"
        self._q: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._worker_exc: BaseException | None = None
        self._dropped = 0
        self._queue_peak = 0
        if self._async:
            self._q = queue.Queue(maxsize=512)   # ~17s @30Hz headroom for transient stalls
            self._worker = threading.Thread(target=self._writer_loop,
                                            name=f"epwriter-{task}-{ep}", daemon=True)
            self._worker.start()

    def write_tick(self, frames: dict[str, np.ndarray],
                   state: list[float], action: list[float], ts: float,
                   depth_frames: dict[str, np.ndarray] | None = None,
                   intervention: int = -1) -> None:
        # Prep all per-tick payloads up front. _prep_rgb/_prep_depth copy into
        # contiguous final-size arrays, so the result is self-owned — safe to hand
        # to the async writer thread and frees the bridge's frame buffer now.
        cam_arrs = {cam: self._prep_rgb(frames.get(cam)) for cam in CAMERAS}
        if self._depth_arrays:
            depth_frames = depth_frames or {}
            depth_arrs = {cam: self._prep_depth(depth_frames.get(cam)) for cam in DEPTH_CAMERAS}
        else:
            depth_arrs = {}
        s = [float(x) for x in (list(state)[:14] + [0.0] * max(0, 14 - len(state)))]
        a = [float(x) for x in (list(action)[:14] + [0.0] * max(0, 14 - len(action)))]
        if eef_recording_enabled():
            s = append_absolute_eef(s)
        iv = max(-1, min(1, int(intervention)))  # clamp to int8 (clawvla format)
        item = (cam_arrs, depth_arrs, s, a, ts, iv)

        # Async mode: enqueue and return — the heavy front-trim + encode + depth
        # compress run on the writer thread, so a slow tick never stalls the 30Hz
        # grab loop (record-time frame-drop root fix). Sync mode: process inline.
        if self._async:
            if self._worker_exc is not None:        # writer thread died → surface to caller
                raise self._worker_exc
            try:
                self._q.put_nowait(item)
                self._queue_peak = max(self._queue_peak, self._q.qsize())
            except queue.Full:
                self._dropped += 1
                if self._dropped == 1 or self._dropped % 30 == 0:
                    log.warning("[async-writer] queue full, dropped %d tick(s) "
                                "(writer can't keep up)", self._dropped)
            return
        self._ingest(*item)

    def _ingest(self, cam_arrs: dict, depth_arrs: dict,
                s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Front-trim onset buffering + staging. Runs on the capture thread (sync
        mode) OR the writer thread (async mode) — never both at once, so the
        front-trim state needs no extra lock."""
        # Fast path: front-trim off, or onset already passed → hand to the
        # tail-trim stage immediately (which is a passthrough when tail_trim off).
        if self._onset_found:
            self._stage_tick(cam_arrs, depth_arrs, s, a, ts, iv)
            return

        # ── V3 front-trim: buffer + incremental onset detection ──
        self._buf.append((cam_arrs, depth_arrs, s, a, ts, iv))
        a_np = np.asarray(a, dtype=np.float64)
        if self._prev_action is not None:
            da = float(np.abs(a_np[TRIM_ARM_DIMS] - self._prev_action[TRIM_ARM_DIMS]).mean())
            moving = da > self._front_thr_arm
            if self._front_thr_grip is not None and not moving:
                # 人接管段: 第一个动作常是"手臂几乎不动, 先合爪抓住布料" → 夹爪在动就算起手,
                # 否则会把抓取瞬间当迟疑裁掉。episode 级 profile 不看夹爪 (thr_grip=None)。
                dg = float(np.abs(a_np[TRIM_GRIP_DIMS]
                                  - self._prev_action[TRIM_GRIP_DIMS]).max())
                moving = dg > self._front_thr_grip
            self._run = self._run + 1 if moving else 0
        self._prev_action = a_np

        if self._run >= self._front_win:
            # onset reached; the rolling buffer holds exactly [onset-MARGIN : now]
            # (proof: cap=MARGIN+WIN ⇒ buf_start = max(0, onset-MARGIN) = cut).
            # _front_keep tightens that to the last N entries — the human-segment
            # profile sets it to HESITATION_WIN so ONLY the motion burst survives
            # (zero lead-in), matching the offline classify_segment's hes_end.
            self._onset_found = True
            pending = (self._buf if self._front_keep is None
                       else self._buf[-self._front_keep:])
            for tk in pending:
                self._stage_tick(*tk)
            self._buf = []
            return

        # Cap the rolling window. Dropped frames are provably earlier than any
        # future cut (cut = future_onset - MARGIN > dropped index), so safe.
        if len(self._buf) > self._front_margin + self._front_win:
            self._buf.pop(0)

    def segment_control(self, op: str) -> None:
        """Segment-boundary control for a multi-segment episode (chunk-001 直接采集).

        MUST stay FIFO-ordered with the ticks around it, so in async mode this is
        queued rather than applied on the caller's thread — otherwise a takeover
        thread could re-arm the front-trim while the writer thread is still
        draining the previous segment's frames, and the trim would eat the wrong
        ones.

        Ops:
          'human_seg_begin' — 人接管段开始 (踩踏板)。重新武装 front-trim 并切到人接管
                              profile: 零 lead-in、判定带夹爪 → 起手迟疑一帧不留
                              (离线 class 3 全裁)。
          'human_seg_end'   — 人接管段结束 (松踏板/交还)。尾部静止段一帧不留
                              (离线 class 4 全裁), 然后切回 episode 级 profile,
                              让整集最后仍保留 TAIL_CAP 帧 terminal settle。
          'flush_keep'      — 冲掉所有 pending 缓冲, 【不】做任何裁剪。用在策略段被接管
                              打断处: 那段静止尾巴是"模型卡死"的失败先兆 (preintv 负样本),
                              是最该留的帧。
        """
        if self._async and self._worker is not None:
            if self._worker_exc is not None:
                raise self._worker_exc
            self._q.put((_CTL, op))
        else:
            self._apply_control(op)

    def _apply_control(self, op: str) -> None:
        """Runs on whichever thread owns the trim state (capture in sync mode,
        writer thread in async mode) — same invariant as _ingest."""
        if op == "human_seg_begin":
            # Human takeover proves any normal policy idle immediately before it
            # was interior/failure context, so preserve the complete buffered run.
            for tk in self._normal_tail_buf:
                self._emit_tick(*tk)
            self._normal_tail_buf = []
            self._normal_tail_spilled = False
            self._normal_tail_prev_action = None
            self._segment_tail_trim = True
            if self._front_trim:
                self._onset_found = False
                self._prev_action = None
                self._run = 0
            # 人接管段 profile: margin=0 + keep=HESITATION_WIN → 起手静止全裁,
            # 只留真正动起来的那几帧; 判定带夹爪, 免得把"先合爪"当迟疑。
            self._front_margin = 0
            self._front_win = HESITATION_WIN
            self._front_thr_arm = HESITATION_THR
            self._front_thr_grip = GRIP_HESITATION_THR
            self._front_keep = HESITATION_WIN
            self._tail_cap = 0          # 段尾静止一帧不留
            return
        if op not in ("flush_keep", "human_seg_end"):
            log.warning("unknown segment_control op %r, ignored", op)
            return
        human_end = (op == "human_seg_end")
        # Un-resolved front-trim buffer: the segment ended before motion onset.
        #   flush_keep     → 策略段整段没动 = 模型冻结, 那正是要留的负样本 → 全留。
        #   human_seg_end  → 人接管段整段没动 = 纯迟疑伪影, 无一帧有效纠错 → 全丢
        #                    (离线 find_keep_indices 同样返回空区间)。
        if self._buf:
            if human_end:
                self._buf = []
            else:
                for tk in self._buf:
                    self._stage_tick(*tk)
                self._buf = []
            self._onset_found = True
        if self._tail_buf:
            keep = self._tail_buf[:self._tail_cap] if human_end else self._tail_buf
            for tk in keep:
                self._emit_tick(*tk)
            self._tail_buf = []
        # Next segment's first staged frame becomes the tail-trim anchor again
        # (matches the offline per-segment active[0]=True).
        self._tail_prev_action = None
        if human_end:
            # 切回 episode 级 profile: 后续策略段不做起手裁剪 (策略是在任务中途接着跑,
            # 低速属真实行为), 整集最末仍保留 TAIL_CAP 帧 terminal settle。
            self._front_margin = TRIM_MARGIN
            self._front_win = TRIM_WIN
            self._front_thr_arm = TRIM_THR
            self._front_thr_grip = None
            self._front_keep = None
            self._tail_cap = TAIL_CAP
            self._segment_tail_trim = False
            self._normal_tail_prev_action = None

    def _writer_loop(self) -> None:
        """Async writer thread: drain queued ticks → full front-trim + encode +
        depth pipeline. Exits on the None sentinel, on stop, or on first
        processing error (stored in _worker_exc → surfaced to the caller at the
        next write_tick / at finalize)."""
        while True:
            item = self._q.get()
            try:
                if item is None or self._stop.is_set():
                    return
                if item[0] is _CTL:
                    self._apply_control(item[1])
                    continue
                self._ingest(*item)
            except BaseException as e:  # noqa: BLE001
                self._worker_exc = e
                log.exception("[async-writer] tick processing failed; stopping writer")
                return
            finally:
                self._q.task_done()

    def _stage_tick(self, cam_arrs: dict, depth_arrs: dict,
                    s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Stage one tick without ever deleting an interior pause.

        Normal episodes retain the current idle run exactly in a bounded raw
        buffer. Motion resuming flushes the whole run, proving it was interior.
        Saving emits only the first TAIL_CAP frames. If an idle exceeds the
        configured memory limit, it is streamed to disk and finalize falls back
        to contiguous packet-copy trimming.

        DAgger human segments keep their stricter segment-specific behavior.
        """
        if self._segment_tail_trim:
            a_np = np.asarray(a, dtype=np.float64)
            if self._tail_prev_action is None:
                active = True
            else:
                d_arm = float(np.abs(a_np[TRIM_ARM_DIMS]
                                     - self._tail_prev_action[TRIM_ARM_DIMS]).mean())
                d_grip = float(np.abs(a_np[TRIM_GRIP_DIMS]
                                      - self._tail_prev_action[TRIM_GRIP_DIMS]).max())
                active = (d_arm > TRIM_THR) or (d_grip > TRIM_GRIP_THR)
            self._tail_prev_action = a_np
            if active:
                for tk in self._tail_buf:
                    self._emit_tick(*tk)
                self._tail_buf = []
                self._emit_tick(cam_arrs, depth_arrs, s, a, ts, iv)
            elif len(self._tail_buf) < self._tail_cap:
                self._tail_buf.append((cam_arrs, depth_arrs, s, a, ts, iv))
            return

        if not self._tail_trim:
            self._emit_tick(cam_arrs, depth_arrs, s, a, ts, iv)
            return

        item = (cam_arrs, depth_arrs, s, a, ts, iv)
        a_np = np.asarray(a, dtype=np.float64)
        if self._normal_tail_prev_action is None:
            active = True
        else:
            d_arm = float(np.abs(
                a_np[TRIM_ARM_DIMS]
                - self._normal_tail_prev_action[TRIM_ARM_DIMS]).mean())
            d_grip = float(np.abs(
                a_np[TRIM_GRIP_DIMS]
                - self._normal_tail_prev_action[TRIM_GRIP_DIMS]).max())
            active = (d_arm > TRIM_THR) or (d_grip > TRIM_GRIP_THR)
        self._normal_tail_prev_action = a_np

        if active:
            # Motion resumed: the complete candidate run was interior, so keep it.
            for tk in self._normal_tail_buf:
                self._emit_tick(*tk)
            self._normal_tail_buf = []
            self._normal_tail_spilled = False
            self._emit_tick(*item)
            return

        if self._normal_tail_spilled:
            self._emit_tick(*item)
            return

        self._normal_tail_buf.append(item)
        if len(self._normal_tail_buf) > self._normal_tail_limit:
            # Unusually long idle: bound RAM, preserve every frame, and remember
            # that final trimming must happen on disk if this remains terminal.
            for tk in self._normal_tail_buf:
                self._emit_tick(*tk)
            self._normal_tail_buf = []
            self._normal_tail_spilled = True

    def _warmup_encoders(self) -> None:
        """Pay the nvenc per-session init (the ~0.3-0.6s/stream first-encode stall)
        at construction time so the capture loop never sees it. Feed one throwaway
        black frame per stream with a NEGATIVE pts (so the real frames' pts 0..N-1
        stay strictly increasing); discard whatever it emits now, and record the one
        frame nvenc keeps buffered (1-frame delay) in _skip_packets so _emit_tick
        drops it when it surfaces on the first real encode — output stays exactly N
        frames, 0-based PTS (verified)."""
        black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for cam, stream in self._streams.items():
            try:
                frame = av.VideoFrame.from_ndarray(black, format="rgb24")
                frame.pts = -1
                emitted = sum(1 for _ in stream.encode(frame))
                self._skip_packets[cam] = max(0, 1 - emitted)
                # The warmup frame's keyframe packet is skipped, so the FIRST real
                # frame MUST be forced to an IDR keyframe — otherwise the muxed
                # stream has no keyframe and the whole video is undecodable (black).
                self._force_idr[cam] = 1
            except Exception:  # noqa: BLE001
                log.warning("encoder warmup failed for %s, skipping", cam, exc_info=True)
                self._skip_packets[cam] = 0
                self._force_idr[cam] = 0

    def _emit_tick(self, cam_arrs: dict, depth_arrs: dict,
                   s: list[float], a: list[float], ts: float, iv: int) -> None:
        """Encode one kept tick → mp4 + depth zarr + parquet rows. pts/frame_index
        count only emitted (kept) frames, so trimmed output starts from 0 and the
        video PTS is zeroed by construction (first kept frame → pts 0). The parquet
        timestamp is derived as frame_index/fps in _write_parquet (NOT the wall-clock
        ts arg), keeping it aligned with the zeroed PTS after front/tail trim — see
        docs/deployment/training_ops/dataset_trimming_and_pts.md. `ts` is unused now
        (kept in the signature for callers / future diagnostics)."""
        for cam in CAMERAS:
            arr = cam_arrs.get(cam)
            if arr is None:
                arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = self._frame_idx
            if self._force_idr.get(cam, 0) > 0:
                frame.pict_type = av.video.frame.PictureType.I
                self._force_idr[cam] -= 1
            for packet in self._streams[cam].encode(frame):
                if self._skip_packets.get(cam, 0) > 0:
                    self._skip_packets[cam] -= 1
                    continue
                self._containers[cam].mux(packet)

        if self._depth_arrays:
            for cam in DEPTH_CAMERAS:
                z = self._depth_arrays.get(cam)
                if z is None:
                    continue
                d = depth_arrs.get(cam)
                if d is None:
                    d = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
                _append_depth_frame(z, d)

        self._rows_state.append(s)
        self._rows_action.append(a)
        self._rows_intervention.append(iv)
        self._frame_idx += 1

    def _prep_rgb(self, arr: np.ndarray | None) -> np.ndarray | None:
        return None if arr is None else self._ensure_size(arr)

    @staticmethod
    def _prep_depth(d: np.ndarray | None) -> np.ndarray | None:
        if d is None or getattr(d, "shape", None) != (HEIGHT, WIDTH):
            return None
        return np.ascontiguousarray(d.astype(np.uint16, copy=False))

    @staticmethod
    def _ensure_size(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == HEIGHT and arr.shape[1] == WIDTH and arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
        from PIL import Image
        img = Image.fromarray(arr).resize((WIDTH, HEIGHT))
        return np.asarray(img, dtype=np.uint8)

    def finalize(self) -> None:
        save_t0 = time.perf_counter()
        queued_at_stop = self._q.qsize() if self._q is not None else 0
        # Async: drain all queued ticks (process them) then stop the writer thread
        # BEFORE we touch encoders/buffers on this thread. The recorder already
        # joined its capture thread (no more enqueues arrive), so this is race-free.
        if self._async and self._worker is not None:
            if self._worker_exc is None:
                self._q.put(None)            # FIFO sentinel → drains all real items first
                self._worker.join()
            else:
                self._worker.join(timeout=1.0)
            if self._dropped:
                log.warning("[async-writer] ep=%d: %d tick(s) dropped under backpressure",
                            self.ep, self._dropped)
            if self._worker_exc is not None:
                raise self._worker_exc
        drain_done = time.perf_counter()
        # Front-trim with no motion onset (degenerate/never-moved episode): keep
        # only the last MARGIN frames — matches build_no_release (cut=len-MARGIN).
        # (_front_margin is TRIM_MARGIN for a normal episode; 0 only when the capture
        # died mid-human-segment, where a lead-in of pure hesitation is worth nothing.)
        if self._front_trim and not self._onset_found and self._buf:
            for tk in (self._buf[-self._front_margin:] if self._front_margin > 0 else []):
                self._stage_tick(*tk)
            self._buf = []
            self._onset_found = True
        # A capture ending mid-human-segment may still have a bounded DAgger
        # segment tail. Normal episodes emit only the first terminal-settle
        # window; the rest was never encoded and needs no MP4 rewrite.
        if self._segment_tail_trim and self._tail_buf:
            for tk in self._tail_buf[:self._tail_cap]:
                self._emit_tick(*tk)
            self._tail_buf = []
        elif self._tail_trim and self._normal_tail_buf:
            for tk in self._normal_tail_buf[:self._tail_cap]:
                self._emit_tick(*tk)
            self._normal_tail_buf = []
        disk_tail_trim = self._tail_trim and self._normal_tail_spilled
        for cam, stream in self._streams.items():
            for packet in stream.encode():
                self._containers[cam].mux(packet)
            self._containers[cam].close()
        self._containers.clear()
        self._streams.clear()
        close_done = time.perf_counter()
        if disk_tail_trim:
            self._trim_terminal_idle()
        trim_done = time.perf_counter()
        self._write_parquet()
        parquet_done = time.perf_counter()
        # Structural validation now runs asynchronously immediately before TOS
        # upload.  Keeping it out of save() lets the operator start the next
        # episode without waiting for four full packet scans.
        validate_done = time.perf_counter()
        # Convert depth `.zarr/` to one lossless FFV1/gray16le MKV.  A persistent
        # marker makes TOS wait and lets startup recovery resume an interrupted
        # job.  One low-priority process-wide worker serializes conversions; save
        # returns immediately. KAI0_DEPTH_PACK_SYNC=1 remains a compatibility
        # switch for callers that explicitly require synchronous finalization.
        dirs = [d for d in self.depth_paths.values() if d.is_dir()]
        if dirs:
            if os.environ.get("KAI0_DEPTH_PACK_SYNC", "0") == "1":
                if self._depth_ffv1:
                    for dpath in dirs:
                        pending_path_for(dpath).write_text(f"{time.time()}\n", encoding="utf-8")
                _finalize_depth_dirs(dirs, ffv1=self._depth_ffv1)
            else:
                _enqueue_depth_finalize(dirs, ffv1=self._depth_ffv1)
        log.info(
            "[save-profile] ep=%d queued=%d peak=%d drain=%.3fs close=%.3fs "
            "tail=%.3fs parquet=%.3fs validate=%.3fs total=%.3fs",
            self.ep, queued_at_stop, self._queue_peak,
            drain_done - save_t0, close_done - drain_done,
            trim_done - close_done, parquet_done - trim_done,
            validate_done - parquet_done, time.perf_counter() - save_t0,
        )

    def _trim_terminal_idle(self) -> None:
        """Trim only the final static run, keeping every interior frame.

        Video is already encoded, so ffmpeg packet-copy creates a temporary
        contiguous prefix without quality loss. All four temporary videos are
        validated before any original is replaced. Depth and tabular rows are
        then shortened to exactly the same keep count.
        """
        keep = _tail_keep_count(self._rows_action, self._tail_cap)
        total = self._frame_idx
        if keep >= total:
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            log.warning("[tail-trim] ffmpeg unavailable; keeping full ep=%d", self.ep)
            return

        staged: dict[str, Path] = {}
        tmp_paths: list[Path] = []
        try:
            def stage_video(cam: str, path: Path) -> tuple[str, Path]:
                tmp = path.with_name(f".{path.stem}.tailtrim.tmp.mp4")
                tmp.unlink(missing_ok=True)
                subprocess.run(
                    [
                        ffmpeg, "-v", "error", "-y", "-i", str(path),
                        "-map", "0:v:0", "-c", "copy", "-copyinkf",
                        "-frames:v", str(keep), "-movflags", "+faststart",
                        str(tmp),
                    ],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                )
                with av.open(str(tmp)) as container:
                    packets = sum(
                        1 for packet in container.demux(container.streams.video[0])
                        if packet.pts is not None
                    )
                if packets != keep:
                    raise RuntimeError(
                        f"{cam}: staged packet count {packets} != keep {keep}"
                    )
                return cam, tmp

            tmp_paths = [
                path.with_name(f".{path.stem}.tailtrim.tmp.mp4")
                for path in self.video_paths.values()
            ]
            with ThreadPoolExecutor(
                max_workers=len(self.video_paths),
                thread_name_prefix=f"tailtrim-{self.ep}",
            ) as pool:
                futures = [
                    pool.submit(stage_video, cam, path)
                    for cam, path in self.video_paths.items()
                ]
                for future in futures:
                    cam, tmp = future.result()
                    staged[cam] = tmp
        except Exception:
            log.exception("[tail-trim] ep=%d failed; keeping full episode", self.ep)
            for tmp in tmp_paths:
                tmp.unlink(missing_ok=True)
            return

        # Commit only after every staged video passed validation. Any failure in
        # this phase propagates to recorder.save(), whose abort path removes the
        # incomplete episode instead of publishing mismatched modalities.
        try:
            for z in self._depth_arrays.values():
                if int(z.shape[0]) > keep:
                    z.resize((keep, z.shape[1], z.shape[2]))
            for cam, tmp in staged.items():
                os.replace(tmp, self.video_paths[cam])
            self._rows_state = self._rows_state[:keep]
            self._rows_action = self._rows_action[:keep]
            self._rows_intervention = self._rows_intervention[:keep]
            self._frame_idx = keep
            log.info(
                "[tail-trim] ep=%d contiguous prefix %d→%d (trimmed %d terminal idle)",
                self.ep, total, keep, total - keep,
            )
        finally:
            for tmp in tmp_paths:
                tmp.unlink(missing_ok=True)

    def _validate_alignment(self) -> None:
        """docs/deployment/training_ops/dataset_trimming_and_pts.md §4 checklist:
        every video's first PTS == 0 and frame count == parquet rows. Structurally
        guaranteed by _emit_tick (pts = frame_index, one row per emitted frame), but
        a record-time spot check catches encoder/mux regressions before training.

        Uses packet DEMUX (no decode): one coded packet per frame, so len(packets)
        == frame count, and min(pts) == first displayed frame's pts. ~10-50× cheaper
        than decoding every frame — important because finalize runs under the
        recorder lock, so a slow check would freeze the backend on save. Gated by
        KAI0_VALIDATE_TRIM."""
        n_rows = pq.read_metadata(self.pq_path).num_rows
        for path in self.video_paths.values():
            with av.open(str(path)) as c:
                ptss = [p.pts for p in c.demux(c.streams.video[0]) if p.pts is not None]
            first_pts = min(ptss) if ptss else None
            if first_pts != 0:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: first pts={first_pts} != 0 "
                    f"(video PTS not zeroed → visual↔action skew)")
            if len(ptss) != n_rows:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: video frames {len(ptss)} != parquet rows {n_rows}")
            # Decode the first frame (cheap, 1 frame): catches a missing keyframe /
            # undecodable stream — the demux count+pts check above passes even when
            # the video is all-black (e.g. the encoder-warmup keyframe got skipped).
            with av.open(str(path)) as c:
                first = next(c.decode(c.streams.video[0]), None)
            if first is None:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: no decodable frame (missing keyframe?)")
            if first.to_ndarray(format="rgb24").mean() < 2.0:
                raise RuntimeError(
                    f"[trim-validate] {path.name}: first frame is black (decode/keyframe broken)")
        log.info("[trim-validate] ep=%d OK: first-pts=0, frames==rows==%d, first-frame decodes",
                 self.ep, n_rows)

    def abort(self) -> None:
        # Async: stop the writer thread fast and discard whatever's still queued.
        if self._async and self._worker is not None:
            self._stop.set()
            try:
                self._q.put_nowait(None)     # wake a blocked get()
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
        self._buf = []              # drop any un-flushed front-trim buffer
        self._tail_buf = []         # drop any held DAgger segment buffer
        self._normal_tail_buf = []  # drop normal terminal candidate buffer
        for container in self._containers.values():
            try:
                container.close()
            except Exception:
                pass
        self._containers.clear()
        self._streams.clear()
        self._depth_arrays.clear()
        for path in self.video_paths.values():
            path.unlink(missing_ok=True)
        for d in self.depth_paths.values():
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            zip_path_for(d).unlink(missing_ok=True)  # in case a prior pack ran
            ffv1_path_for(d).unlink(missing_ok=True)
            pending_path_for(d).unlink(missing_ok=True)
        self.pq_path.unlink(missing_ok=True)

    def _write_parquet(self) -> None:
        n = len(self._rows_state)
        actions = (
            apply_relative_eef_actions(self._rows_state, self._rows_action)
            if eef_recording_enabled()
            else self._rows_action
        )
        cols = {
            "observation.state": pa.array(self._rows_state, type=pa.list_(pa.float32())),
            "action": pa.array(actions, type=pa.list_(pa.float32())),
            # lerobot-standard timestamp = frame_index / fps (0-based, contiguous) —
            # matches build_no_release. NOT wall-clock: each emitted tick is exactly
            # one video frame (pts = frame_index, first kept frame pts = 0), so
            # frame_index/fps is the only axis that stays aligned with the zeroed
            # video PTS after front/tail trim. Wall-clock here would re-introduce the
            # §2 PTS-style visual↔action skew (invisible to offline MAE). See
            # docs/deployment/training_ops/dataset_trimming_and_pts.md.
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS, type=pa.float32()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "episode_index": pa.array([self.ep] * n, type=pa.int64()),
            "index": pa.array(list(range(n)), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
        }
        # Only emit intervention column when any tick wrote a non-default value
        # (-1 means "N/A / not applicable", matches clawvla convention for non-DAgger
        # captures). For DAgger episodes intervention rows are 0 (policy) or 1 (human).
        if self._rows_intervention and any(v != -1 for v in self._rows_intervention):
            cols["intervention"] = pa.array(self._rows_intervention, type=pa.int8())
            if self._frame_class:
                cols["dagger_frame_class"] = pa.array(
                    self._derive_frame_class(), type=pa.int8())
        table = pa.table(cols)
        pq.write_table(table, self.pq_path)

    def _derive_frame_class(self) -> np.ndarray:
        """dagger_frame_class {0=robot, 1=intv_core, 2=preintv} from the intervention
        column, at finalize (the label is retroactive — you only know a frame was
        "pre-intervention" once the takeover happens).

        Classes 3 (hesitation) / 4 (stationary_tail) never appear: segment_control
        physically trims those at the segment boundaries, so the on-disk column is a
        true 3-way split. Same rule as train_scripts/kai/data/relabel_chunk001_preintv.py
        → online and offline chunk-001 stay label-identical.
        """
        iv = np.asarray(self._rows_intervention, dtype=np.int8)
        cls = np.zeros(len(iv), dtype=np.int8)
        cls[iv == 1] = 1
        # every policy→human boundary: the PREINTV_MARGIN policy frames before it
        # are the failure precursor. Guard on iv==0 so a neighbouring human segment
        # doesn't get demoted from intv_core.
        (bounds,) = np.nonzero((iv[1:] == 1) & (iv[:-1] == 0))
        for i in bounds + 1:
            lo = max(0, int(i) - PREINTV_MARGIN)
            seg = np.arange(lo, i)
            cls[seg[iv[lo:i] == 0]] = 2
        return cls

    @property
    def frame_count(self) -> int:
        return self._frame_idx


def write_episode_meta(writer: EpisodeWriter, duration: float,
                       success: bool = True, note: str = "",
                       scene_tags: list[str] | None = None,
                       extra: dict | None = None,
                       filename: str = "episodes.jsonl") -> None:
    """Append one record to meta/<filename> + ensure meta/tasks.jsonl has prompt.

    `extra` merges additional keys into the record (e.g. terminal-cause labels
    like {"terminal": "intervention", "intervention_frame_index": N} that the
    dagger recorder attaches to policy-rollout episodes).

    `filename` selects the log: chunk-000 episodes go to episodes.jsonl, chunk-001
    stitched rollouts to episodes_stitched.jsonl — the same split the offline
    stitch_dagger_episodes.py produces, so finalize_dagger_dataset.py and the rest
    of the chunk-001 chain read online and offline output identically."""
    meta_dir = writer.root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if eef_recording_enabled():
        modality_path = write_modality_json(writer.root)
        # eef_kinematics also serves older callers and describes the superset of
        # camera modalities.  For a no-mid run, make this episode root's schema
        # match the videos actually produced instead of advertising a missing
        # observation.images.mid_head stream.
        if "mid_head" not in CAMERAS:
            modality = json.loads(modality_path.read_text(encoding="utf-8"))
            modality.get("video", {}).pop("mid_head", None)
            modality_path.write_text(
                json.dumps(modality, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    rec = {
        "episode_id": writer.ep,
        "episode_chunk": writer.chunk,
        "chunk": f"chunk-{writer.chunk:03d}",
        "length": writer.frame_count,
        "duration_s": round(duration, 3),
        "operator": writer.operator,
        "prompt": writer.prompt,
        "template_id": writer.template_id,
        "success": success,
        "note": note,
        "scene_tags": list(scene_tags or []),
        "created_at": time.time(),
    }
    if extra:
        rec.update(extra)
    with (meta_dir / filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tasks_path = meta_dir / "tasks.jsonl"
    tasks_path.touch()
    existing_prompts = set()
    for ln in tasks_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            existing_prompts.add(json.loads(ln).get("task"))
        except json.JSONDecodeError:
            continue
    if writer.prompt not in existing_prompts:
        with tasks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"task_index": len(existing_prompts), "task": writer.prompt},
                ensure_ascii=False,
            ) + "\n")


def update_info_json(task: str | None, subset: str | None) -> None:
    """Re-aggregate meta/info.json from episodes.jsonl for one (task, subset)."""
    if not task or not subset:
        return
    root = task_subset_root(task, subset)
    info_path = root / "meta" / "info.json"
    ep_log_path = root / "meta" / "episodes.jsonl"
    total_ep = 0
    total_frames = 0
    if ep_log_path.exists():
        for ln in ep_log_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            total_ep += 1
            total_frames += int(d.get("length", 0))

    info = {
        "codebase_version": "v2.1",
        "robot_type": "agilex",
        "total_episodes": total_ep,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_ep * len(CAMERAS),
        "total_chunks": len(list((root / "data").glob("chunk-*"))) if (root / "data").exists() else 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{total_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "depth_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mkv"
            if depth_ffv1_enabled()
            else "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.zarr.zip"
        ),
        "features": features_block(),
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def features_block() -> dict:
    spec_codec = pick_codec()[0]
    img_feat = {
        "dtype": "video",
        "shape": [HEIGHT, WIDTH, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": HEIGHT,
            "video.width": WIDTH,
            "video.codec": spec_codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": FPS,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    if depth_ffv1_enabled():
        depth_feat = {
            "dtype": "uint16_ffv1",
            "shape": [HEIGHT, WIDTH],
            "names": ["height", "width"],
            "info": {
                "container": "matroska",
                "codec": "ffv1",
                "pix_fmt": "gray16le",
                "unit": "millimeter",
                "depth.height": HEIGHT,
                "depth.width": WIDTH,
                "depth.fps": FPS,
            },
        }
    else:
        depth_feat = {
            "dtype": "uint16_zarr",
            "shape": [HEIGHT, WIDTH],
            "names": ["height", "width"],
            "info": {
                "store": "zarr.DirectoryStore",
                "container": "zip",
                "compressor": "blosc.zstd:level3:bitshuffle",
                "unit": "millimeter",
                "depth.height": HEIGHT,
                "depth.width": WIDTH,
                "depth.fps": FPS,
            },
        }
    return {
        **{f"observation.images.{cam}": img_feat for cam in CAMERAS},
        **{f"observation.depth.{cam}": depth_feat for cam in DEPTH_CAMERAS},
        "observation.state": {
            "dtype": "float32",
            "shape": [EEF_STATE_DIM if eef_recording_enabled() else 14],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": [EEF_STATE_DIM if eef_recording_enabled() else 14],
            "names": None,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def next_episode_id(task: str, subset: str, chunk: int = 0) -> int:
    """Scan data/chunk-{chunk:03d}/episode_*.parquet under task_subset_root and
    return max+1 (or 0 if empty). Used by autonomy recorder to auto-pick
    episode_id without needing a UI/state-machine.
    """
    root = task_subset_root(task, subset)
    chunk_dir = root / "data" / f"chunk-{int(chunk):03d}"
    if not chunk_dir.exists():
        return 0
    eps = []
    for p in chunk_dir.glob("episode_*.parquet"):
        try:
            eps.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return (max(eps) + 1) if eps else 0
