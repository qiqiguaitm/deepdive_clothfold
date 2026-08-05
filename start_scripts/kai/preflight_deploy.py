#!/usr/bin/env python3
"""Compatibility entry point for the canonical FastWAM deploy preflight."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "fastwam" / "deploy" / "preflight.py"),
    run_name="__main__",
)
