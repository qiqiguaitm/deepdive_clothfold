"""Unified command line entry point: ``python -m data_tools``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m data_tools")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="classify existing project data scripts")
    inventory.add_argument("--repo", type=Path, default=Path.cwd())
    inventory.add_argument("--output", type=Path)

    quality = commands.add_parser("quality", help="check parquet/video/action integrity")
    quality.add_argument("root", type=Path)
    quality.add_argument("--episode", type=int, action="append")
    quality.add_argument("--camera", action="append", dest="cameras")
    quality.add_argument("--output", type=Path)

    normalize = commands.add_parser("normalize", help="merge/reindex AgileX or KAI0 LeRobot leaves")
    normalize.add_argument("--src", type=Path, action="append", required=True)
    normalize.add_argument("--dst", type=Path, required=True)
    normalize.add_argument("--task", required=True)
    normalize.add_argument("--fps", type=int, default=30)
    normalize.add_argument("--camera", action="append", dest="cameras")
    normalize.add_argument("--video-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    normalize.add_argument("--use-latest-good", action="store_true")

    static = commands.add_parser("static", help="locate long leading/interior/trailing static runs")
    static.add_argument("root", type=Path)
    static.add_argument("--episode", type=int, action="append")
    static.add_argument("--min-frames", type=int, default=50)
    static.add_argument("--fps", type=float, default=30.0)
    static.add_argument("--source-column", default="observation.state")
    static.add_argument("--arm-threshold", type=float, default=3e-3)
    static.add_argument("--gripper-threshold", type=float, default=0.02)
    static.add_argument("--ideal-only", action="store_true")
    static.add_argument("--ideal-class", type=int, action="append", dest="ideal_classes")
    static.add_argument("--output", type=Path)

    flicker = commands.add_parser("flicker", help="detect periodic exposure flicker and rolling bands")
    flicker.add_argument("path", type=Path, help="video file or dataset directory")
    flicker.add_argument("--camera")
    flicker.add_argument("--sample", type=int, default=0)
    flicker.add_argument("--mains-hz", type=float, default=50.0)
    flicker.add_argument("--max-frames", type=int, default=0)
    flicker.add_argument("--output", type=Path)

    audit = commands.add_parser("audit", help="audit one or more dated dataset leaves")
    audit.add_argument("root", type=Path)
    audit.add_argument("--date", action="append", default=[], help="exact YYYY-MM-DD; repeatable")
    audit.add_argument("--date-from", help="inclusive YYYY-MM-DD")
    audit.add_argument("--date-to", help="inclusive YYYY-MM-DD")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--visual-sample", type=int, default=12, help="videos per leaf; 0 checks all")
    audit.add_argument("--max-visual-frames", type=int, default=180)
    audit.add_argument("--min-static-frames", type=int, default=50)
    audit.add_argument("--mains-hz", type=float, default=50.0)

    forge = commands.add_parser("forge", help="run the pinned Forge Robotics backend")
    forge.add_argument("arguments", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inventory":
        from .inventory import render_markdown

        report = render_markdown(args.repo.resolve())
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        else:
            print(report, end="")
        return 0
    if args.command == "quality":
        from .quality import scan_dataset

        rows = scan_dataset(
            args.root.resolve(), episodes=args.episode,
            cameras=args.cameras or ("top_head", "hand_left", "hand_right"),
            output=args.output,
        )
        print(json.dumps({"episodes": len(rows), "good": sum(row.good for row in rows)}, indent=2))
        return 0 if all(row.good for row in rows) else 2
    if args.command == "normalize":
        from .normalize import normalize

        result = normalize(
            args.src, args.dst.resolve(), task=args.task, fps=args.fps,
            cameras=args.cameras or ("top_head", "hand_left", "hand_right"),
            video_mode=args.video_mode, use_latest_good=args.use_latest_good,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "static":
        from collections import Counter
        from dataclasses import asdict
        from .static_segments import scan_static_segments

        segments = scan_static_segments(
            args.root.resolve(), episodes=args.episode, min_frames=args.min_frames,
            fps=args.fps, source_column=args.source_column,
            arm_threshold=args.arm_threshold, gripper_threshold=args.gripper_threshold,
            ideal_only=args.ideal_only, ideal_classes=args.ideal_classes or (1, 5),
            output=args.output,
        )
        positions = Counter(segment.position for segment in segments)
        print(json.dumps({
            "segments": len(segments),
            "episodes": len({segment.episode_id for segment in segments}),
            "positions": dict(sorted(positions.items())),
            "items": [asdict(segment) for segment in segments] if not args.output else [],
        }, indent=2))
        return 0
    if args.command == "flicker":
        from dataclasses import asdict
        from .flicker import scan_flicker

        results = scan_flicker(
            args.path.resolve(), camera=args.camera, sample=args.sample,
            mains_hz=args.mains_hz, max_frames=args.max_frames, output=args.output,
        )
        flagged = [result for result in results if result.flagged]
        print(json.dumps({
            "videos": len(results), "flagged": len(flagged),
            "items": [asdict(result) for result in flagged],
        }, indent=2, ensure_ascii=False))
        return 2 if flagged else 0
    if args.command == "forge":
        from .forge_adapter import run_forge

        try:
            return run_forge(args.arguments)
        except RuntimeError as exc:
            print(f"error: {exc}")
            return 2
    if args.command == "audit":
        from .audit import run_audit
        try:
            report = run_audit(args.root, output=args.output, dates=args.date,
                               date_from=args.date_from, date_to=args.date_to,
                               visual_sample=args.visual_sample,
                               min_static_frames=args.min_static_frames,
                               mains_hz=args.mains_hz,
                               max_visual_frames=args.max_visual_frames)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        print(json.dumps({key: report[key] for key in ("leaves", "episodes", "issue_counts")}, indent=2))
        return 2 if any(report["issue_counts"].values()) else 0
    raise AssertionError(args.command)
