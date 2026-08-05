#!/usr/bin/env python3
"""Compatibility entry point for :mod:`fastwam.deploy.isolated_control`."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "fastwam" / "deploy" / "isolated_control.py"),
    run_name="__main__",
)
