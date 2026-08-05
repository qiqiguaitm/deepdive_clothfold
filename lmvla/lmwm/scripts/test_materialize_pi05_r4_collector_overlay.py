from __future__ import annotations

from pathlib import Path
import tarfile

import pytest

from materialize_pi05_r4_collector_overlay import inventory, safe_extract


def test_inventory_changes_with_file_identity(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("first\n")
    first = inventory(tmp_path)
    (tmp_path / "a.txt").write_text("other\n")
    second = inventory(tmp_path)
    assert first[:2] == second[:2]
    assert first[2] != second[2]


def test_safe_extract_rejects_parent_escape(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar"
    payload = tmp_path / "payload"
    payload.write_text("bad")
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="../escape")
    destination = tmp_path / "out"
    destination.mkdir()
    with tarfile.open(archive_path) as archive:
        with pytest.raises(ValueError, match="escapes destination"):
            safe_extract(archive, destination)
