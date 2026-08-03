"""Discover and classify historical data scripts without moving their paths."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

CATEGORIES = {
    "build": ("build_", "prepare_", "merge", "split_"),
    "validate": ("validate", "verify", "sanity", "scan_", "check_"),
    "repair": ("fix_", "trim_", "prune_", "reset_", "relabel_", "rescale_"),
    "stats": ("stats", "norm_", "compute_", "analy", "quality", "filter_"),
    "visualize": ("viz_", "render_", "video", "plot_"),
    "export": ("export", "convert", "to_tos", "from_tos"),
}

SEARCH_ROOTS = (
    "train_scripts/kai/data",
    "start_scripts/data_fix",
    "web/data_manager/backend/tools",
    "kai0/scripts",
)


def classify(name: str) -> str:
    lower = name.lower()
    for category, tokens in CATEGORIES.items():
        if any(token in lower for token in tokens):
            return category
    return "experiment"


def collect(repo: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for relative in SEARCH_ROOTS:
        root = repo / relative
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                rows.append((classify(path.name), path.relative_to(repo)))
    return rows


def render_markdown(repo: Path) -> str:
    rows = collect(repo)
    counts = Counter(category for category, _ in rows)
    lines = ["# Data script inventory", "", f"Total: {len(rows)}", ""]
    lines.extend(f"- `{category}`: {counts[category]}" for category in sorted(counts))
    lines.extend(["", "| Category | Existing compatibility path |", "|---|---|"])
    lines.extend(f"| {category} | `{path}` |" for category, path in rows)
    return "\n".join(lines) + "\n"
