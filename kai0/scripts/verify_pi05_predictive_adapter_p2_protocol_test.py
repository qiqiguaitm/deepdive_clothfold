from __future__ import annotations

import hashlib
from pathlib import Path

from verify_pi05_predictive_adapter_p2_protocol import verify


def test_protocol_file_hashes_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "protocol.txt"
    path.write_text("frozen")
    manifest = {
        "protocol": "test",
        "file_sha256": {"protocol.txt": hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    assert verify(tmp_path, manifest)["passed"]
    path.write_text("drift")
    assert not verify(tmp_path, manifest)["passed"]
