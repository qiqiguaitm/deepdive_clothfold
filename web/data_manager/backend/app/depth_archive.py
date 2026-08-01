"""Single-file depth archive: pack a depth `.zarr/` directory (one tiny file per
frame-chunk → ~1.7k files/episode) into one `.zarr.zip` object, and read it back
by extracting to a temp dir on demand.

Why: depth zarr DirectoryStore emits ~1749 tiny files per episode. On object
storage (TOS) that means ~155k objects/day → listing/transfer is the bottleneck
(hot-sync excludes depth entirely for this reason). Training never reads depth;
the only readers are the data_manager viewer + the offline video_publisher replay
node. So we store depth as ONE file per episode and decompress to a temp dir when
a reader actually needs the frames.

Format: ZIP (ZIP_STORED — chunks are already blosc-zstd compressed, so the outer
container adds no recompression CPU) whose root holds the CONTENTS of the `.zarr`
dir (`.zarray`, `.zattrs`, chunk files `0.0.0` …). Extracting to a temp dir T then
`zarr.open(T)` yields the original array.

All readers stay BACKWARD-COMPATIBLE: if the legacy `.zarr/` dir is present it is
used directly; the `.zarr.zip` is preferred when both exist.
"""
from __future__ import annotations

import os
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

ZIP_SUFFIX = ".zip"  # artifact = "<...>.zarr.zip"
FFV1_SUFFIX = ".mkv"
PENDING_SUFFIX = ".ffv1.pending"


def zip_path_for(zarr_dir: Path | str) -> Path:
    """The `.zarr.zip` sibling path for a base `.zarr` dir path."""
    return Path(str(zarr_dir) + ZIP_SUFFIX)


def ffv1_path_for(zarr_dir: Path | str) -> Path:
    """Return the lossless depth-video sibling for an ``episode_X.zarr`` dir."""
    zarr_dir = Path(zarr_dir)
    base = str(zarr_dir)
    if base.endswith(".zarr"):
        base = base[:-len(".zarr")]
    return Path(base + FFV1_SUFFIX)


def pending_path_for(zarr_dir: Path | str) -> Path:
    """Persistent marker used to keep TOS sync behind depth finalization."""
    return Path(str(zarr_dir) + PENDING_SUFFIX)


def convert_zarr_dir_to_ffv1(
    zarr_dir: Path | str,
    *,
    remove_dir: bool = True,
    verify_pixels: bool = True,
    fps: int = 30,
) -> Path:
    """Atomically convert uint16 depth Zarr to lossless FFV1/gray16le MKV.

    Frames are streamed through PyAV so an episode is never materialized fully in
    RAM.  The output is committed only after packet-count/format validation and,
    by default, a decoded-byte SHA256 comparison against the Zarr input.  The
    source directory is removed only after the verified MKV has been renamed into
    place.  A crash therefore leaves either the original Zarr or a harmless temp
    file for startup recovery.
    """
    import av
    import numpy as np
    import zarr

    zarr_dir = Path(zarr_dir)
    if not zarr_dir.is_dir():
        raise FileNotFoundError(f"not a zarr dir: {zarr_dir}")
    arr = zarr.open(str(zarr_dir), mode="r")
    if len(arr.shape) != 3:
        raise RuntimeError(f"invalid depth shape {arr.shape} at {zarr_dir}")
    n, height, width = (int(v) for v in arr.shape)
    dst = ffv1_path_for(zarr_dir)
    tmp = dst.with_name(f".{dst.name}.tmp.mkv")
    tmp.unlink(missing_ok=True)

    input_hash = hashlib.sha256()
    try:
        with av.open(str(tmp), mode="w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "gray16le"
            stream.options = {"level": "3"}
            for idx in range(n):
                frame_arr = np.ascontiguousarray(arr[idx], dtype="<u2")
                input_hash.update(frame_arr.tobytes())
                frame = av.VideoFrame.from_ndarray(frame_arr, format="gray16le")
                frame.pts = idx
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    fields: dict[str, str] = {}
    with av.open(str(tmp)) as container:
        stream = container.streams.video[0]
        fields = {
            "codec_name": stream.codec_context.name,
            "width": str(stream.width),
            "height": str(stream.height),
            "nb_read_packets": str(sum(
                1 for packet in container.demux(stream) if packet.pts is not None
            )),
        }
    expected = {"codec_name": "ffv1", "width": str(width),
                "height": str(height), "nb_read_packets": str(n)}
    if any(fields.get(k) != v for k, v in expected.items()):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffv1 validation failed: got={fields}, expected={expected}")

    if verify_pixels:
        decoded_hash = hashlib.sha256()
        decoded_bytes = 0
        with av.open(str(tmp)) as container:
            for frame in container.decode(video=0):
                raw = np.ascontiguousarray(
                    frame.to_ndarray(format="gray16le"), dtype="<u2"
                ).tobytes()
                decoded_hash.update(raw)
                decoded_bytes += len(raw)
        expected_bytes = n * height * width * 2
        if (decoded_bytes != expected_bytes or decoded_hash.digest() != input_hash.digest()):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                "ffv1 pixel verification failed: "
                f"bytes={decoded_bytes}/{expected_bytes}"
            )

    os.replace(tmp, dst)
    if remove_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)
    return dst


def convert_zarr_zip_to_ffv1(
    zarr_zip: Path | str,
    *,
    remove_zip: bool = True,
    verify_pixels: bool = True,
    fps: int = 30,
) -> Path:
    """Convert a packed ``episode_XXXXXX.zarr.zip`` to verified FFV1.

    Extraction and encoding use a temporary directory beside the source so the
    final rename is atomic on the dataset filesystem.  The packed source is kept
    until the decoded FFV1 pixels have been verified against the extracted Zarr.
    """
    zarr_zip = Path(zarr_zip)
    suffix = ".zarr.zip"
    if not zarr_zip.is_file() or not str(zarr_zip).endswith(suffix):
        raise FileNotFoundError(f"not a zarr.zip file: {zarr_zip}")
    dst = Path(str(zarr_zip)[:-len(suffix)] + FFV1_SUFFIX)

    with tempfile.TemporaryDirectory(
        prefix=f".{zarr_zip.stem}.ffv1-", dir=zarr_zip.parent
    ) as tmp_name:
        tmp_root = Path(tmp_name)
        extracted = tmp_root / "source.zarr"
        extracted.mkdir()
        with zipfile.ZipFile(zarr_zip) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"unsafe zip member {member.filename!r}")
            archive.extractall(extracted)
        tmp_mkv = convert_zarr_dir_to_ffv1(
            extracted,
            remove_dir=False,
            verify_pixels=verify_pixels,
            fps=fps,
        )
        os.replace(tmp_mkv, dst)

    if remove_zip:
        zarr_zip.unlink()
    return dst


def resolve_depth_artifact(zarr_dir: Path | str) -> Path | None:
    """Given the base `.../episode_X.zarr` path, return the artifact that exists:
    prefer the lossless FFV1 `.mkv` (newest, ~58% smaller), then the packed
    `.zarr.zip`, then the legacy `.zarr/` dir, else None."""
    zarr_dir = Path(zarr_dir)
    # Callers may already have resolved a concrete artifact rather than passing
    # the canonical `.zarr` base path.
    if zarr_dir.is_file() and zarr_dir.suffix in {FFV1_SUFFIX, ZIP_SUFFIX}:
        return zarr_dir
    mkv = ffv1_path_for(zarr_dir)
    if mkv.is_file():
        return mkv
    zp = zip_path_for(zarr_dir)
    if zp.is_file():
        return zp
    if zarr_dir.is_dir():
        return zarr_dir
    return None


def pack_zarr_dir(zarr_dir: Path | str, *, remove_dir: bool = True) -> Path:
    """Pack a `.zarr/` directory into a sibling `.zarr.zip` (ZIP_STORED, contents
    at zip root). Atomic: writes to a .tmp then renames. Returns the zip path.
    On `remove_dir`, deletes the source dir only after the zip is in place."""
    zarr_dir = Path(zarr_dir)
    if not zarr_dir.is_dir():
        raise FileNotFoundError(f"not a zarr dir: {zarr_dir}")
    zp = zip_path_for(zarr_dir)
    tmp = Path(str(zp) + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _dirs, files in os.walk(zarr_dir):
            for name in files:
                fp = Path(root) / name
                arc = fp.relative_to(zarr_dir).as_posix()  # contents at zip root
                zf.write(fp, arc)
    os.replace(tmp, zp)
    if remove_dir:
        shutil.rmtree(zarr_dir, ignore_errors=True)
    return zp


def open_depth_readonly(artifact: Path | str):
    """Open a depth artifact (`.zarr.zip` OR legacy `.zarr/` dir) read-only.

    Returns (zarr_array, tmpdir) where tmpdir is a str to rmtree when done
    (None for the legacy dir path, which is opened in place). Caller MUST
    clean up tmpdir (use `close_depth(tmpdir)`)."""
    import zarr  # lazy: only when a reader actually needs frames

    artifact = Path(artifact)
    if artifact.suffix == ".mkv":  # lossless FFV1 gray16le depth video
        import av
        import numpy as np

        c = av.open(str(artifact))
        frames = [f.to_ndarray() for f in c.decode(video=0)]  # each (H,W) uint16
        c.close()
        return np.stack(frames), None  # (T,H,W) uint16 — supports .shape / arr[i]
    if artifact.is_dir():
        return zarr.open(str(artifact), mode="r"), None
    if artifact.suffix == ZIP_SUFFIX:
        tmp = tempfile.mkdtemp(prefix="kai0_depthz_")
        try:
            with zipfile.ZipFile(artifact) as zf:
                zf.extractall(tmp)
            return zarr.open(tmp, mode="r"), tmp
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    raise FileNotFoundError(f"no depth artifact at {artifact}")


def close_depth(tmpdir: str | None) -> None:
    """Remove the temp dir returned by open_depth_readonly (no-op if None)."""
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
