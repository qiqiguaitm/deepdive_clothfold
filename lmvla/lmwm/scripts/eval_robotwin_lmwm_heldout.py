#!/usr/bin/env python3
"""Train and evaluate LMWM on task-stratified held-out RoboTwin episodes."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p1_train_lmwm_robotwin import (  # noqa: E402
    GridCache,
    InverseEnc,
    MilestoneGenerator,
    MilestonePredictorGrid,
    cosr,
)


TASK_NAMES = {
    0: "beat_block_hammer",
    1: "stack_blocks_two",
    2: "stack_blocks_three",
    3: "blocks_ranking_rgb",
    4: "blocks_ranking_size",
    5: "handover_block",
}


def split_episodes(
    cur_ep: np.ndarray, pair_task: np.ndarray, fold: int, folds: int
) -> tuple[set[int], set[int], dict[str, list[int]]]:
    train: set[int] = set()
    heldout: set[int] = set()
    manifest: dict[str, list[int]] = {}
    for task_id, task_name in TASK_NAMES.items():
        episodes = sorted(set(map(int, cur_ep[pair_task == task_id])))
        task_heldout = episodes[fold::folds]
        heldout.update(task_heldout)
        train.update(set(episodes) - set(task_heldout))
        manifest[task_name] = task_heldout
    return train, heldout, manifest


def grid_batch(
    cache: GridCache,
    cur_ep: np.ndarray,
    cur_fi: np.ndarray,
    tgt_fi: np.ndarray,
    indices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = np.stack(
        [cache.get(int(cur_ep[i]))[int(cur_fi[i])] for i in indices]
    ).astype(np.float32)
    future = np.stack(
        [cache.get(int(cur_ep[i]))[int(tgt_fi[i])] for i in indices]
    ).astype(np.float32)
    return torch.from_numpy(current).to(device), torch.from_numpy(future).to(device)


def episode_groups(
    cur_ep: np.ndarray, indices: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    episodes = np.unique(cur_ep[indices])
    groups = [indices[cur_ep[indices] == episode] for episode in episodes]
    weights = np.asarray([len(group) for group in groups], dtype=np.float64)
    weights /= weights.sum()
    return groups, weights


def sample_episode_local_batch(
    groups: list[np.ndarray],
    weights: np.ndarray,
    batch_size: int,
    episodes_per_batch: int,
    rng: np.random.Generator,
) -> np.ndarray:
    episode_slots = min(episodes_per_batch, batch_size)
    group_ids = rng.choice(len(groups), size=episode_slots, replace=True, p=weights)
    counts = np.full(episode_slots, batch_size // episode_slots, dtype=np.int64)
    counts[: batch_size % episode_slots] += 1
    return np.concatenate(
        [
            rng.choice(groups[group_id], size=int(count), replace=True)
            for group_id, count in zip(group_ids, counts)
        ]
    )


@torch.no_grad()
def predict_grid(
    generator: MilestoneGenerator,
    predictor: MilestonePredictorGrid,
    current: torch.Tensor,
) -> torch.Tensor:
    logits, means, _ = predictor(current)
    code = means[torch.arange(len(current), device=current.device), logits.argmax(1)]
    return generator(current, code)


@torch.no_grad()
def evaluate(
    *,
    generator: MilestoneGenerator,
    predictor: MilestonePredictorGrid,
    cache: GridCache,
    arrays: dict[str, np.ndarray],
    heldout_indices: np.ndarray,
    samples_per_task: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    per_task: dict[str, object] = {}
    pooled_bank: dict[int, tuple[np.ndarray, list[tuple[int, int]]]] = {}

    for task_id, task_name in TASK_NAMES.items():
        task_indices = heldout_indices[arrays["pair_task"][heldout_indices] == task_id]
        chosen = rng.choice(
            task_indices, size=min(samples_per_task, len(task_indices)), replace=False
        )
        chosen = chosen[np.argsort(arrays["cur_ep"][chosen], kind="stable")]
        identities = []
        bank_vectors = []
        for index in chosen:
            identity = (int(arrays["cur_ep"][index]), int(arrays["tgt_fi"][index]))
            if identity in identities:
                continue
            target = cache.get(identity[0])[identity[1]].astype(np.float32)
            identities.append(identity)
            bank_vectors.append(target.mean(axis=(1, 2)))
        bank = np.stack(bank_vectors)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8
        pooled_bank[task_id] = bank, identities

        metrics: dict[str, list[float]] = defaultdict(list)
        for start in range(0, len(chosen), batch_size):
            indices = chosen[start : start + batch_size]
            current, future = grid_batch(
                cache,
                arrays["cur_ep"],
                arrays["cur_fi"],
                arrays["tgt_fi"],
                indices,
                device,
            )
            predicted = predict_grid(generator, predictor, current)
            pred_cos = cosr(predicted.flatten(1), future.flatten(1))
            current_cos = cosr(current.flatten(1), future.flatten(1))
            error = F.smooth_l1_loss(predicted, future, reduction="none").flatten(1).mean(1)
            metrics["latent_cosine"].extend(pred_cos.cpu().tolist())
            metrics["persistence_cosine"].extend(current_cos.cpu().tolist())
            metrics["smooth_l1"].extend(error.cpu().tolist())

            pooled = predicted.mean((2, 3)).cpu().numpy()
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
            similarities = pooled @ bank.T
            order = np.argsort(-similarities, axis=1)
            for row, index in enumerate(indices):
                target_identity = (
                    int(arrays["cur_ep"][index]),
                    int(arrays["tgt_fi"][index]),
                )
                target_position = identities.index(target_identity)
                metrics["retrieval_top1"].append(float(order[row, 0] == target_position))
                metrics["retrieval_top5"].append(float(target_position in order[row, :5]))
                retrieved_episode = identities[int(order[row, 0])][0]
                metrics["retrieval_same_episode"].append(
                    float(retrieved_episode == target_identity[0])
                )

        summary = {name: float(np.mean(values)) for name, values in metrics.items()}
        summary["predictor_lift"] = (
            summary["latent_cosine"] - summary["persistence_cosine"]
        )
        summary["samples"] = int(len(chosen))
        summary["retrieval_bank_size"] = int(len(identities))
        per_task[task_name] = summary

    macro_keys = [
        "latent_cosine",
        "persistence_cosine",
        "predictor_lift",
        "smooth_l1",
        "retrieval_top1",
        "retrieval_top5",
        "retrieval_same_episode",
    ]
    macro = {
        key: float(np.mean([per_task[name][key] for name in TASK_NAMES.values()]))
        for key in macro_keys
    }
    return {"macro": macro, "per_task": per_task}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--episodes-per-batch", type=int, default=4)
    parser.add_argument("--eval-samples-per-task", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.fold < args.folds:
        raise ValueError("fold must be in [0, folds)")

    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)

    pairs = np.load(args.pairs)
    arrays = {name: pairs[name] for name in pairs.files}
    train_eps, heldout_eps, split_manifest = split_episodes(
        arrays["cur_ep"], arrays["pair_task"], args.fold, args.folds
    )
    train_indices = np.flatnonzero(np.isin(arrays["cur_ep"], list(train_eps)))
    heldout_indices = np.flatnonzero(np.isin(arrays["cur_ep"], list(heldout_eps)))
    train_groups, train_group_weights = episode_groups(
        arrays["cur_ep"], train_indices
    )

    sample_episode = int(arrays["cur_ep"][0])
    sample = np.load(args.feat / f"ep{sample_episode}.npz")["grid"]
    din = int(sample.shape[-1])
    cache = GridCache(str(args.feat), args.cache_cap, din)
    inverse = InverseEnc(din, 32).to(args.device)
    generator = MilestoneGenerator(din, 32).to(args.device)
    predictor = MilestonePredictorGrid(din, 32, 4).to(args.device)
    optimizer_teacher = torch.optim.AdamW(
        list(inverse.parameters()) + list(generator.parameters()),
        lr=2e-4,
        weight_decay=1e-5,
    )
    optimizer_predictor = torch.optim.AdamW(
        predictor.parameters(), lr=2e-4, weight_decay=1e-5
    )
    rng = np.random.default_rng(args.seed + args.fold)

    for step in range(args.steps):
        indices = sample_episode_local_batch(
            train_groups,
            train_group_weights,
            args.batch_size,
            args.episodes_per_batch,
            rng,
        )
        current, future = grid_batch(
            cache,
            arrays["cur_ep"],
            arrays["cur_fi"],
            arrays["tgt_fi"],
            indices,
            args.device,
        )
        code = inverse(current, future)
        predicted = generator(current, code)
        reconstruction = F.smooth_l1_loss(predicted, future)
        lift = F.relu(
            cosr(predicted.flatten(1), current.flatten(1))
            - cosr(predicted.flatten(1), future.flatten(1))
        ).mean()
        distillation = predictor.nll(current, code.detach())
        (reconstruction + lift).backward(retain_graph=True)
        optimizer_teacher.step()
        optimizer_teacher.zero_grad()
        distillation.backward()
        optimizer_predictor.step()
        optimizer_predictor.zero_grad()
        if step % 100 == 0 or step + 1 == args.steps:
            print(
                f"fold={args.fold} step={step}/{args.steps} rec={reconstruction.item():.5f} "
                f"lift={lift.item():.5f} dist={distillation.item():.3f}",
                flush=True,
            )

    generator.eval()
    predictor.eval()
    evaluation = evaluate(
        generator=generator,
        predictor=predictor,
        cache=cache,
        arrays=arrays,
        heldout_indices=heldout_indices,
        samples_per_task=args.eval_samples_per_task,
        batch_size=args.eval_batch_size,
        device=args.device,
        seed=args.seed + args.fold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".pt")
    torch.save(
        {
            "inv": inverse.state_dict(),
            "gen": generator.state_dict(),
            "prd": predictor.state_dict(),
            "code_dim": 32,
            "din": din,
            "fold": args.fold,
            "folds": args.folds,
        },
        checkpoint,
    )
    report = {
        "protocol": "task-stratified episode holdout",
        "fold": args.fold,
        "folds": args.folds,
        "train_episodes": len(train_eps),
        "heldout_episodes": len(heldout_eps),
        "train_pairs": int(len(train_indices)),
        "heldout_pairs": int(len(heldout_indices)),
        "split": split_manifest,
        "checkpoint": str(checkpoint),
        **evaluation,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["macro"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
