import json
from pathlib import Path

import pytest

import verify_temporal_grounding_tg2_recovery_bundle as verifier


def make_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "lmvla/lawam").mkdir(parents=True)
    payload = repo / "payload.txt"
    payload.write_text("frozen\n", encoding="utf-8")
    manifest = repo / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "temporal_grounding_tg2_recovery_v1",
                "frozen": True,
                "manual_execution_authorized": False,
                "only_training_change": {
                    "datasets.vla_data.in_order": [False, True]
                },
                "file_sha256": {
                    "payload.txt": verifier.sha256_file(payload),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        verifier,
        "git_head",
        lambda path: (
            verifier.EXPECTED_LAWAM
            if path == repo / "lmvla/lawam"
            else verifier.EXPECTED_OUTER
        ),
    )
    return repo, manifest


def test_accepts_only_in_order_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, manifest = make_bundle(tmp_path, monkeypatch)
    result = verifier.verify(repo, manifest)
    assert result["verified_files"] == 1
    assert result["only_training_change"] == {
        "datasets.vla_data.in_order": [False, True]
    }


def test_rejects_additional_protocol_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, manifest = make_bundle(tmp_path, monkeypatch)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["only_training_change"]["trainer.gradient_accumulation_steps"] = [2, 1]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="change only DataLoader"):
        verifier.verify(repo, manifest)
