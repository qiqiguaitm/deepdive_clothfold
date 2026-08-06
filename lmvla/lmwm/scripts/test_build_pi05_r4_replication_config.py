from __future__ import annotations

import json
from pathlib import Path

from build_pi05_r4_replication_config import main


def test_replication_builder_changes_only_seed_identity(tmp_path: Path, monkeypatch) -> None:
    public = tmp_path / "train_config.json"
    public.write_text(
        json.dumps(
            {
                "steps": 50000,
                "batch_size": 16,
                "seed": 1000,
                "dataset": {},
                "policy": {
                    "chunk_size": 50,
                    "freeze_vision_encoder": False,
                    "gradient_checkpointing": True,
                    "compile_model": True,
                    "compile_mode": "max-autotune",
                    "output_features": {"action": {"shape": [14]}},
                },
                "optimizer": {"lr": 2.5e-5, "weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8},
                "scheduler": {"num_warmup_steps": 1000, "num_decay_steps": 30000, "decay_lr": 2.5e-6},
                "wandb": {},
            }
        )
    )
    output = tmp_path / "replication.json"
    dataset = tmp_path / "dataset"
    model = tmp_path / "model"
    dataset.mkdir()
    model.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "builder",
            "--public-config", str(public),
            "--arm", "ordinary",
            "--seed", "1001",
            "--world-size", "4",
            "--steps", "5000",
            "--output-dir", str(tmp_path / "run"),
            "--dataset-root", str(dataset),
            "--model-path", str(model),
            "--sidecar", str(tmp_path / "unused.npz"),
            "--output", str(output),
        ],
    )
    assert main() == 0
    config = json.loads(output.read_text())
    assert config["seed"] == 1001
    assert config["job_name"] == "pi05-r4-ordinary-seed1001"
    assert config["batch_size"] == 4
    assert config["steps"] == 5000
