import hashlib
import json
from pathlib import Path

from materialize_pi05_r1_frozen_overlay import materialize


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializer_uses_frozen_override_and_links_evidence(tmp_path: Path) -> None:
    repo = tmp_path
    current = repo / "kai0/src/openpi/training/config.py"
    current.parent.mkdir(parents=True)
    current.write_text("drifted\n")
    frozen = (
        repo
        / "lmvla/paper_iclr_lmvla/frozen_sources/pi05_r1_v1"
        / "kai0/src/openpi/training/config.py"
    )
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen\n")

    evidence = {
        "scene.json": b"{}",
        "kai0/checkpoints/pi05_base/params/_METADATA": b"base",
        "kai0/assets/pi05_robotwin_a0_public_exact_bj/robotwin2.0_absolute_meanstd/norm_stats.json": b"norm",
        "dense.npz": b"dense",
        "dense.json": b"manifest",
        "logs/predictive/p0_eval/p0_gate.accepted": b"p0",
        "logs/crave_r0/probe_gate/r0_gate.accepted": b"r0",
        "lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p0_final_audit.json": b"{}",
        "kai0/checkpoints/pi05_predictive_adapter_p0/pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999/params/_METADATA": b"p0ckpt",
        "lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json": b"{}",
        "lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz": b"labels",
        "lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz": b"probe",
        "lmvla/lmwm/data/pi05_crave_r0_v1/reference_trajectories.npz": b"refs",
    }
    for relative, payload in evidence.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    protocol = {
        "source_sha256": {"kai0/src/openpi/training/config.py": digest(frozen)},
        "scene_manifest": "scene.json",
        "teacher": {"artifact": "dense.npz", "artifact_manifest": "dense.json"},
    }
    protocol_path = (
        repo / "lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json"
    )
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol))

    output = repo / "logs/frozen_source_overlays/test"
    audit = materialize(repo, output)
    assert audit["passed"]
    assert audit["frozen_override_count"] == 1
    assert (output / "READY").is_file()
    assert digest(output / "kai0/src/openpi/training/config.py") == digest(frozen)
    assert (output / "scene.json").is_symlink()
