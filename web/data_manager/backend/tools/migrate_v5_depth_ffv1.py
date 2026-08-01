#!/usr/bin/env python3
"""Migrate legacy v5 Zarr ZIP depth artifacts to lossless FFV1 MKV."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.depth_archive import convert_zarr_zip_to_ffv1  # noqa: E402


DEPTH_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/"
    "episode_{episode_index:06d}.mkv"
)
DEPTH_FEATURE = {
    "dtype": "uint16_ffv1",
    "shape": [480, 640],
    "names": ["height", "width"],
    "info": {
        "container": "matroska",
        "codec": "ffv1",
        "pix_fmt": "gray16le",
        "unit": "millimeter",
        "depth.height": 480,
        "depth.width": 640,
        "depth.fps": 30,
    },
}


def _leaves(root: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        leaves = [p.resolve() for p in explicit]
    else:
        leaves = sorted(p for p in root.glob("*/*/v5/*-v5") if p.is_dir())
    return [p for p in leaves if p.parent.name == "v5"]


def _update_info(leaf: Path) -> None:
    path = leaf / "meta" / "info.json"
    if not path.is_file():
        return
    info = json.loads(path.read_text(encoding="utf-8"))
    info["depth_path"] = DEPTH_PATH
    features = info.get("features", {})
    for key in list(features):
        if key.startswith("observation.depth."):
            features[key] = DEPTH_FEATURE
    tmp = path.with_name(f".{path.name}.ffv1.tmp")
    tmp.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _assert_leaf_complete(leaf: Path) -> None:
    leftovers = list(leaf.glob("videos/chunk-*/observation.depth.*/*.zarr.zip"))
    if leftovers:
        raise RuntimeError(f"{leaf}: {len(leftovers)} zarr.zip files remain")
    for parquet in leaf.glob("data/chunk-*/*.parquet"):
        chunk = parquet.parent.name
        for depth_dir in (leaf / "videos" / chunk).glob("observation.depth.*"):
            mkv = depth_dir / f"{parquet.stem}.mkv"
            if not mkv.is_file():
                raise RuntimeError(f"missing migrated depth: {mkv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0")),
    )
    parser.add_argument("--leaf", action="append", type=Path, default=[])
    parser.add_argument(
        "--keep-source", action="store_true",
        help="keep verified .zarr.zip files (default removes each after verification)",
    )
    args = parser.parse_args()

    leaves = _leaves(args.root.resolve(), args.leaf)
    archives = [
        archive
        for leaf in leaves
        for archive in sorted(
            leaf.glob("videos/chunk-*/observation.depth.*/*.zarr.zip")
        )
    ]
    print(f"[v5-depth] leaves={len(leaves)} archives={len(archives)}", flush=True)
    done = 0
    before = sum(path.stat().st_size for path in archives)
    for archive in archives:
        dst = convert_zarr_zip_to_ffv1(
            archive, remove_zip=not args.keep_source, verify_pixels=True
        )
        done += 1
        print(
            f"[v5-depth] {done}/{len(archives)} {archive.name} -> "
            f"{dst.name} ({dst.stat().st_size / 1024**2:.1f} MiB)",
            flush=True,
        )

    for leaf in leaves:
        _assert_leaf_complete(leaf)
        _update_info(leaf)

    after = sum(
        path.stat().st_size
        for leaf in leaves
        for path in leaf.glob("videos/chunk-*/observation.depth.*/*.mkv")
    )
    print(
        f"[v5-depth] complete converted={done} "
        f"source={before / 1024**3:.2f} GiB mkv_total={after / 1024**3:.2f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
