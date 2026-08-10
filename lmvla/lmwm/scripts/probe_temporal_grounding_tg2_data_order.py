#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def collect(repo: Path, label: str, microbatches: int) -> None:
    lawam = repo / "lmvla/lawam"
    os.chdir(lawam)
    sys.path.insert(0, str(lawam))
    from starVLA.training.train_starvla import build_accelerator, prepare_data

    cfg = OmegaConf.load(lawam / "starVLA/config/training/train_robotwin.yaml")
    cfg.seed = 1000
    data = cfg.datasets.vla_data
    data.data_mix = "robotwin2_lmwm_all6_v2"
    data.sec_chunk = 1.0
    data.num_frames = 2
    data.per_device_batch_size = 16
    data.num_workers = 8
    data.val_num_workers = 2
    data.enable_video_frame_cache = False
    data.in_order = True
    data.persistent_workers = True
    data.prefetch_factor = 2
    data.drop_last = True
    cfg.trainer.gradient_accumulation_steps = 2
    cfg.trainer.ddp_find_unused_parameters = False

    set_seed(int(cfg.seed))
    accelerator = build_accelerator(cfg)
    dataloader, _ = prepare_data(cfg, accelerator)
    dataloader = accelerator.prepare(dataloader)
    set_seed(int(cfg.seed) + accelerator.process_index)

    digest = hashlib.sha256()
    samples = 0
    iterator = iter(dataloader)
    for _ in range(microbatches):
        batch = next(iterator)
        for key in ("episode_index", "frame_index"):
            value = batch[key].detach().to(device="cpu", dtype=torch.int64).contiguous()
            digest.update(key.encode("ascii") + b"\0")
            digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
            digest.update(value.numpy().tobytes(order="C"))
        samples += int(batch["episode_index"].shape[0])

    output = repo / "logs/temporal_grounding/tg2/data_order_recovery_probe_v1"
    atomic_json(
        output / label / f"rank{accelerator.process_index}.json",
        {
            "schema_version": 1,
            "protocol": "temporal_grounding_tg2_data_order_recovery_probe_v1",
            "label": label,
            "training_seed": 1000,
            "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "microbatches": microbatches,
            "samples": samples,
            "num_workers": 8,
            "in_order": True,
            "sha256": digest.hexdigest(),
        },
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


def compare(repo: Path, microbatches: int) -> None:
    root = repo / "logs/temporal_grounding/tg2/data_order_recovery_probe_v1"
    rows = {}
    for label in ("a", "b"):
        paths = sorted((root / label).glob("rank*.json"))
        if len(paths) != 4:
            raise ValueError(f"Expected four {label} rank records, found {paths}")
        payloads = [json.loads(path.read_text()) for path in paths]
        for rank, payload in enumerate(payloads):
            expected = {
                "label": label,
                "training_seed": 1000,
                "rank": rank,
                "world_size": 4,
                "microbatches": microbatches,
                "num_workers": 8,
                "in_order": True,
            }
            observed = {key: payload.get(key) for key in expected}
            if observed != expected:
                raise ValueError(f"Probe metadata drift: {observed}")
        rows[label] = [payload["sha256"] for payload in payloads]
    if rows["a"] != rows["b"]:
        raise ValueError(f"in_order=true did not reproduce exact rank order: {rows}")
    atomic_json(
        root / "matched.json",
        {
            "schema_version": 1,
            "protocol": "temporal_grounding_tg2_data_order_recovery_probe_v1",
            "complete": True,
            "microbatches": microbatches,
            "sha256_by_rank": rows["a"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--label", choices=("a", "b"))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--microbatches", type=int, default=256)
    args = parser.parse_args()
    if args.compare == (args.label is not None):
        parser.error("provide exactly one of --label or --compare")
    repo = args.repo.resolve()
    if args.compare:
        compare(repo, args.microbatches)
    else:
        collect(repo, args.label, args.microbatches)


if __name__ == "__main__":
    main()
