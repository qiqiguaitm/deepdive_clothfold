#!/usr/bin/env python3
"""Train one frozen-protocol MT3 stage tracker from cached pi0.5 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


NUM_STAGES = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_weights(target: np.ndarray, classes: int = NUM_STAGES) -> np.ndarray:
    count = np.bincount(target.astype(np.int64), minlength=classes).astype(np.float64)
    present = count > 0
    if not np.any(present):
        raise ValueError("cannot compute class weights from an empty target")
    frequency = count[present] / np.sum(count[present])
    raw = frequency**-0.5
    normalized = raw / np.sum(frequency * raw)
    result = np.zeros(classes, dtype=np.float32)
    result[present] = np.clip(normalized, 0.25, 4.0)
    return result


class CurrentFrameTracker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(3 * 2048, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.current_head = nn.Linear(512, NUM_STAGES)
        self.next_head = nn.Linear(512, NUM_STAGES)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features.flatten(start_dim=1))
        return self.current_head(hidden), self.next_head(hidden)


class HistoryProprioTracker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.GRU(2048 + 14, 256, num_layers=1, batch_first=True)
        self.current_head = nn.Linear(256, NUM_STAGES)
        self.next_head = nn.Linear(256, NUM_STAGES)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.temporal(features)
        final = hidden[-1]
        return self.current_head(final), self.next_head(final)


def load_features(root: Path, candidate: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifests = sorted(root.glob("shard-*-of-*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no feature shard manifests under {root}")
    payloads = [json.loads(path.read_text()) for path in manifests]
    num_shards = {int(payload["num_shards"]) for payload in payloads}
    if len(num_shards) != 1 or len(payloads) != next(iter(num_shards)):
        raise ValueError("feature cache does not contain every declared shard")
    if {int(payload["shard_index"]) for payload in payloads} != set(range(len(payloads))):
        raise ValueError("feature shard indices are incomplete or duplicated")
    provenance = {json.dumps(payload["provenance"], sort_keys=True) for payload in payloads}
    if len(provenance) != 1:
        raise ValueError("feature shards do not share checkpoint/pairs/split provenance")

    common_keys = ("episode", "frame", "task", "current_target", "next_target", "split")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in common_keys}
    feature_parts = []
    chunk_hashes = []
    for manifest_path, payload in zip(manifests, payloads, strict=True):
        shard_rows = 0
        for chunk in payload["chunks"]:
            path = manifest_path.parent / chunk["file"]
            if sha256(path) != chunk["sha256"]:
                raise ValueError(f"feature chunk hash mismatch: {path}")
            archive = np.load(path)
            for key in common_keys:
                arrays[key].append(np.asarray(archive[key]))
            if candidate == "current_frame":
                feature_parts.append(np.asarray(archive["current_view_features"]))
            elif candidate == "history_proprio":
                feature_parts.append(
                    np.concatenate(
                        [
                            np.asarray(archive["history_base_features"]),
                            np.asarray(archive["history_proprio"], dtype=np.float16),
                        ],
                        axis=-1,
                    )
                )
            else:
                raise ValueError(f"unknown tracker candidate: {candidate}")
            shard_rows += int(chunk["rows"])
            chunk_hashes.append(chunk["sha256"])
        if shard_rows != int(payload["rows"]):
            raise ValueError(f"feature shard row count mismatch: {manifest_path}")
    result = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    result["features"] = np.concatenate(feature_parts, axis=0)
    rows = len(result["episode"])
    if any(len(value) != rows for value in result.values()):
        raise ValueError("feature cache arrays have inconsistent row counts")
    keys = zip(result["episode"].tolist(), result["frame"].tolist(), strict=True)
    if len(set(keys)) != rows:
        raise ValueError("feature cache contains duplicate episode/frame rows")
    return result, {
        "feature_cache_root": str(root.resolve()),
        "feature_provenance": json.loads(next(iter(provenance))),
        "feature_chunk_sha256": chunk_hashes,
    }


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def predict(
    model: nn.Module,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    current, next_stage = [], []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = torch.as_tensor(features[selected], device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            current_logits, next_logits = model(batch)
        current.append(current_logits.float().cpu().numpy())
        next_stage.append(next_logits.float().cpu().numpy())
    return np.concatenate(current), np.concatenate(next_stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("current_frame", "history_proprio"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        raise ValueError("updates and batch size must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    data, provenance = load_features(args.features, args.candidate)
    train_indices = np.flatnonzero(data["split"] == 0)
    validation_indices = np.flatnonzero(data["split"] == 1)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("feature cache must contain both train and validation rows")
    current_weight = torch.as_tensor(
        class_weights(data["current_target"][train_indices]), device=device
    )
    next_weight = torch.as_tensor(class_weights(data["next_target"][train_indices]), device=device)
    model: nn.Module = (
        CurrentFrameTracker() if args.candidate == "current_frame" else HistoryProprioTracker()
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    model.train()
    for update in range(1, args.updates + 1):
        positions = torch.randint(
            len(train_indices), (args.batch_size,), generator=generator
        ).numpy()
        selected = train_indices[positions]
        features = torch.as_tensor(data["features"][selected], device=device, dtype=torch.float32)
        current_target = torch.as_tensor(
            data["current_target"][selected], device=device, dtype=torch.long
        )
        next_target = torch.as_tensor(data["next_target"][selected], device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            current_logits, next_logits = model(features)
            current_loss = nn.functional.cross_entropy(
                current_logits, current_target, weight=current_weight
            )
            next_loss = nn.functional.cross_entropy(next_logits, next_target, weight=next_weight)
            loss = current_loss + next_loss
        loss.backward()
        optimizer.step()
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            print(
                f"update={update}/{args.updates} loss={loss.item():.6f} "
                f"current={current_loss.item():.6f} next={next_loss.item():.6f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "tracker.pt"
    atomic_torch_save(
        {
            "candidate": args.candidate,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "updates": args.updates,
            "seed": args.seed,
            "provenance": provenance,
        },
        checkpoint,
    )
    current_logits, next_logits = predict(
        model, data["features"], validation_indices, device=device
    )
    predictions = args.output / "validation_predictions.npz"
    atomic_npz(
        predictions,
        episode=data["episode"][validation_indices],
        frame=data["frame"][validation_indices],
        task=data["task"][validation_indices],
        current_target=data["current_target"][validation_indices],
        next_target=data["next_target"][validation_indices],
        current_logits=current_logits,
        next_logits=next_logits,
    )
    report = {
        "version": "pi05-mt3-tracker-training-v1",
        "candidate": args.candidate,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "class_weights": {
            "current": current_weight.cpu().tolist(),
            "next": next_weight.cpu().tolist(),
            "structural_note": "current targets span 0--8 and next targets span 1--9; absent head classes have zero loss weight",
        },
        "provenance": provenance,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "predictions": str(predictions),
        "predictions_sha256": sha256(predictions),
    }
    (args.output / "train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
