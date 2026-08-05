import hashlib
import json

import pytest

from verify_pi05_r1_protocol import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    repo = tmp_path
    source = repo / "source.py"
    source.write_text("frozen\n")
    scene = repo / "scene.json"
    scene.write_text("{}")
    base = repo / "kai0/checkpoints/pi05_base/params/_METADATA"
    norm = (
        repo
        / "kai0/assets/pi05_robotwin_a0_public_exact_bj"
        / "robotwin2.0_absolute_meanstd/norm_stats.json"
    )
    base.parent.mkdir(parents=True)
    norm.parent.mkdir(parents=True)
    base.write_text("base")
    norm.write_text("norm")
    dense = repo / "lmvla/lmwm/data/pi05_crave_r0_v1/r1_dense_targets.npz"
    dense_manifest = dense.with_name("r1_dense_targets_manifest.json")
    dense.parent.mkdir(parents=True)
    dense.write_bytes(b"dense")
    dense_manifest.write_text(
        json.dumps(
            {
                "dense_targets_sha256": digest(dense),
                "episode_count": 1200,
                "physical_task_count": 6,
                "horizon_frames": 50,
                "target_rows": 359823,
            }
        )
    )
    protocol = repo / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "source_sha256": {"source.py": digest(source)},
                "scene_manifest": "scene.json",
                "scene_manifest_sha256": digest(scene),
                "immutable_artifact_sha256": {
                    "base_params_metadata": digest(base),
                    "norm_stats": digest(norm),
                    "dense_targets": digest(dense),
                    "dense_targets_manifest": digest(dense_manifest),
                },
                "teacher": {
                    "artifact": str(dense.relative_to(repo)),
                    "artifact_manifest": str(dense_manifest.relative_to(repo)),
                },
            }
        )
    )
    (repo / "logs/predictive/p0_eval").mkdir(parents=True)
    (repo / "logs/predictive/p0_eval/p0_gate.accepted").write_text("accepted")
    (repo / "logs/crave_r0/probe_gate").mkdir(parents=True)
    (repo / "logs/crave_r0/probe_gate/r0_gate.accepted").write_text("accepted")
    checkpoint = (
        repo
        / "kai0/checkpoints/pi05_predictive_adapter_p0"
        / "pi05_predictive_adapter_p0_seed1000_20k_w8clean_8g/19999/params/_METADATA"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("p0")
    audit = (
        repo
        / "lmvla/paper_iclr_lmvla/manifests/pi05_predictive_adapter_p0_final_audit.json"
    )
    audit.parent.mkdir(parents=True)
    audit.write_text(
        json.dumps(
            {
                "passed": True,
                "exact_zero_policy_route": True,
                "checkpoint_metadata_sha256": digest(checkpoint),
            }
        )
    )
    labels = repo / "lmvla/lmwm/data/pi05_crave_r0_v1"
    labels.mkdir(parents=True, exist_ok=True)
    generated = {}
    for key, name in (
        ("labels_sha256", "labels.npz"),
        ("probe_train_sha256", "probe_train.npz"),
        ("reference_trajectories_sha256", "reference_trajectories.npz"),
    ):
        path = labels / name
        path.write_bytes(name.encode())
        generated[key] = digest(path)
    (labels / "labels_manifest.json").write_text(json.dumps(generated))
    return repo, protocol, source, labels


def test_r1_verifier_accepts_exact_generated_evidence(tmp_path):
    repo, protocol, _, _ = fixture(tmp_path)
    result = verify(repo, protocol)
    assert result["accepted"]
    assert result["source_sha256"]["source.py"]


def test_r1_verifier_rejects_source_and_generated_drift(tmp_path):
    repo, protocol, source, labels = fixture(tmp_path)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="source drift"):
        verify(repo, protocol)
    source.write_text("frozen\n")
    (labels / "probe_train.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="generated R0 artifact drift"):
        verify(repo, protocol)


def test_r1_verifier_requires_both_accepted_gates(tmp_path):
    repo, protocol, _, _ = fixture(tmp_path)
    (repo / "logs/crave_r0/probe_gate/r0_gate.accepted").unlink()
    with pytest.raises(RuntimeError, match="accepted P0 and R0"):
        verify(repo, protocol)
