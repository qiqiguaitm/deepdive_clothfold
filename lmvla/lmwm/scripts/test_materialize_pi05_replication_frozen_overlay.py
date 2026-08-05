import hashlib
import json
from pathlib import Path

from materialize_pi05_replication_frozen_overlay import materialize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replication_overlay_restores_protocol_exact_files(tmp_path: Path) -> None:
    repo = tmp_path
    r1_protocol_path = (
        repo / "lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json"
    )
    r1_protocol_path.parent.mkdir(parents=True)
    r1_protocol_path.write_text(json.dumps({"source_sha256": {}}))

    current = repo / "train_scripts/kai/runtime.sh"
    current.parent.mkdir(parents=True)
    current.write_text("drifted\n")
    frozen = (
        repo
        / "lmvla/paper_iclr_lmvla/frozen_sources/pi05_replication_v1"
        / "train_scripts/kai/runtime.sh"
    )
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen\n")
    p2_protocol_path = repo / (
        "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_predictive_adapter_p2_protocol.json"
    )
    p2_protocol_path.write_text(
        json.dumps(
            {
                "file_sha256": {
                    "train_scripts/kai/runtime.sh": digest(frozen),
                }
            }
        )
    )

    def fake_base_materializer(_repo: Path, output: Path) -> dict:
        output.mkdir(parents=True)
        (output / "READY").write_text("ready\n")
        return {"protocol": "fake-r1"}

    output = repo / "logs/frozen_source_overlays/replication"
    audit = materialize(repo, output, fake_base_materializer)
    assert audit["passed"]
    assert (output / "REPLICATION_READY").is_file()
    assert digest(output / "train_scripts/kai/runtime.sh") == digest(frozen)
