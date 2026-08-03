"""Thin, version-checked adapter around the optional Forge Robotics CLI."""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

SUPPORTED_FORGE = "0.2.0"


def forge_version() -> str | None:
    try:
        return importlib.metadata.version("forge-robotics")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_forge() -> str:
    version = forge_version()
    if version is None or shutil.which("forge") is None:
        raise RuntimeError(
            f"Forge is optional. Install the tested version with: "
            f"python -m pip install 'forge-robotics[lerobot,video]=={SUPPORTED_FORGE}'"
        )
    if version != SUPPORTED_FORGE:
        raise RuntimeError(f"Forge {version} is installed; KAI0 integration is pinned to {SUPPORTED_FORGE}")
    return version


def run_forge(arguments: Sequence[str], *, check: bool = True) -> int:
    require_forge()
    return subprocess.run(["forge", *arguments], check=check).returncode


def inspect(source: str) -> int:
    return run_forge(["inspect", source])


def convert(source: str, destination: Path, output_format: str, extra: Sequence[str] = ()) -> int:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {destination}")
    return run_forge(["convert", source, str(destination), "--format", output_format, *extra])
