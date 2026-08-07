#!/usr/bin/env python3
"""Audit LaWAM action-time contracts and frozen RoboTwin milestone horizons.

This is a CPU-only, result-independent audit. It reads frozen configs, source
contracts, dataset metadata, and milestone pair indices; it never loads a
policy, performs inference, or writes an artifact. The JSON printed to stdout
is intended to be captured only after the assertions below pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TASK_NAMES = {
    0: "hammer",
    1: "stack_two",
    2: "stack_three",
    3: "ranking_rgb",
    4: "ranking_size",
    5: "handover",
}

TASK_STRATA = {
    "ordered_construction": [
        "stack_blocks_two",
        "stack_blocks_three",
        "stack_bowls_two",
        "stack_bowls_three",
    ],
    "reactive_contact": ["beat_block_hammer", "click_bell", "stamp_seal"],
    "fine_grained_geometry": ["blocks_ranking_size", "place_object_scale"],
    "relational_transfer": ["handover_block", "handover_mic"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def q(values: np.ndarray, quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)) -> dict[str, float]:
    return {f"q{int(level * 100):02d}": round(float(np.quantile(values, level)), 6) for level in quantiles}


def contract_row(
    *,
    name: str,
    config: dict[str, Any],
    fps: float,
    fps_evidence: str,
    executed_actions_per_query: int | None,
) -> dict[str, Any]:
    action = config["framework"]["action_model"]
    data = config["datasets"]["vla_data"]
    sec_chunk = float(data["sec_chunk"])
    horizon_sec = float(action["flow_cfg"]["horizon_sec"])
    padded = int(action["action_horizon"])
    future = int(action["future_action_window_size"])
    past = int(action["past_action_window_size"])
    valid = int(sec_chunk * fps)
    last_offset = valid - 1
    assert np.isclose(sec_chunk, horizon_sec), (name, sec_chunk, horizon_sec)
    assert padded == future + past + 1, (name, padded, future, past)
    assert 0 < valid <= padded, (name, valid, padded)
    if executed_actions_per_query is not None:
        assert 0 < executed_actions_per_query <= valid
    return {
        "name": name,
        "data_mix": str(data["data_mix"]),
        "fps": float(fps),
        "fps_evidence": fps_evidence,
        "sec_chunk": sec_chunk,
        "padded_action_length": padded,
        "valid_action_count_H": valid,
        "last_valid_offset_h": last_offset,
        "sampled_video_offsets_for_two_frames": [0, last_offset],
        "executed_actions_per_query": executed_actions_per_query,
        "last_executed_offset": None if executed_actions_per_query is None else executed_actions_per_query - 1,
        "target_is_inside_executed_prefix": (
            None if executed_actions_per_query is None else last_offset < executed_actions_per_query
        ),
        "checks": {
            "sec_chunk_equals_flow_horizon_sec": True,
            "padded_length_equals_future_plus_past_plus_one": True,
            "valid_length_fits_padding": True,
        },
    }


def episode_slices(ep: np.ndarray) -> list[slice]:
    starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
    ends = np.r_[starts[1:], len(ep)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends)]


def milestone_audit(pairs_path: Path, source_audit_path: Path, *, training_h: int, execution_h: int) -> dict[str, Any]:
    with np.load(pairs_path, allow_pickle=False) as pairs:
        required = ("cur_ep", "cur_fi", "tgt_fi", "cur_ms", "pair_task")
        assert tuple(pairs.files) == required, pairs.files
        arrays = {key: np.asarray(pairs[key], dtype=np.int64) for key in required}

    n = len(arrays["cur_ep"])
    assert n > 0 and all(value.shape == (n,) for value in arrays.values())
    order = np.lexsort((arrays["cur_fi"], arrays["cur_ep"]))
    ep = arrays["cur_ep"][order]
    cur = arrays["cur_fi"][order]
    tgt = arrays["tgt_fi"][order]
    milestone = arrays["cur_ms"][order]
    task = arrays["pair_task"][order]
    horizon = tgt - cur

    assert np.all(horizon >= 0)
    assert set(np.unique(task).tolist()) == set(TASK_NAMES)

    slices = episode_slices(ep)
    assert len(slices) == 1200
    duplicate_pair_count = 0
    outside_episode_count = 0
    nonmonotone_target_count = 0
    mixed_task_episode_count = 0
    incomplete_frame_coverage_count = 0
    terminal_fallback_episode_count = 0
    adjacent_count = 0
    adjacent_same_target_count = 0
    per_episode: list[dict[str, Any]] = []

    for section in slices:
        e = ep[section]
        c = cur[section]
        t = tgt[section]
        m = milestone[section]
        k = task[section]
        assert np.all(e == e[0])
        duplicate_pair_count += int(len(c) - len(np.unique(c)))
        final_frame = int(c.max())
        outside_episode_count += int(np.sum((t < 0) | (t > final_frame)))
        nonmonotone_target_count += int(np.sum(np.diff(t) < 0))
        mixed_task_episode_count += int(len(np.unique(k)) != 1)
        complete = np.array_equal(c, np.arange(final_frame + 1, dtype=np.int64))
        incomplete_frame_coverage_count += int(not complete)
        terminal_fallback_episode_count += int(t[-1] == final_frame)
        adjacent = np.diff(c) == 1
        adjacent_count += int(adjacent.sum())
        adjacent_same_target_count += int(np.sum(adjacent & (t[1:] == t[:-1])))
        per_episode.append(
            {
                "episode": int(e[0]),
                "task": int(k[0]),
                "frames": len(c),
                "distinct_targets": len(np.unique(t)),
                "distinct_milestone_indices": len(np.unique(m)),
            }
        )

    assert duplicate_pair_count == 0
    assert outside_episode_count == 0
    assert nonmonotone_target_count == 0
    assert mixed_task_episode_count == 0
    assert incomplete_frame_coverage_count == 0
    assert terminal_fallback_episode_count == len(slices)

    source_audit = read_json(source_audit_path)
    pairs_hash_records = {
        key: value
        for key, value in source_audit["hashes_sha256"].items()
        if key.endswith("robotwin_milestone_all6_confirmatory_v1/pairs.npz")
    }
    assert len(pairs_hash_records) == 1, pairs_hash_records
    assert next(iter(pairs_hash_records.values())) == sha256(pairs_path)
    assert source_audit["dataset_episodes"] == len(slices)

    records: dict[str, Any] = {}
    for task_id, task_name in TASK_NAMES.items():
        mask = task == task_id
        h_task = horizon[mask]
        g_train = h_task / float(training_h)
        g_exec = h_task / float(execution_h)
        eps_task = [row for row in per_episode if row["task"] == task_id]
        t_ep = ep[mask]
        t_cur = cur[mask]
        t_tgt = tgt[mask]
        consecutive = (t_ep[1:] == t_ep[:-1]) & (t_cur[1:] == t_cur[:-1] + 1)
        same = consecutive & (t_tgt[1:] == t_tgt[:-1])
        records[task_name] = {
            "episodes": len(eps_task),
            "pairs": int(mask.sum()),
            "mean_episode_frames": round(float(np.mean([row["frames"] for row in eps_task])), 6),
            "horizon_frames": {
                "mean": round(float(h_task.mean()), 6),
                "quantiles": q(h_task),
            },
            "within_training_chunk_rate_h_le_49": round(float(np.mean(h_task <= training_h)), 8),
            "beyond_training_chunk_rate_h_gt_49": round(float(np.mean(h_task > training_h)), 8),
            "within_executed_prefix_rate_h_le_35": round(float(np.mean(h_task <= execution_h)), 8),
            "beyond_executed_prefix_rate_h_gt_35": round(float(np.mean(h_task > execution_h)), 8),
            "g_training_h49": {"mean": round(float(g_train.mean()), 6), "quantiles": q(g_train)},
            "g_execution_h35": {"mean": round(float(g_exec.mean()), 6), "quantiles": q(g_exec)},
            "adjacent_same_target_rate": round(float(same.sum() / consecutive.sum()), 8),
            "mean_distinct_targets_per_episode": round(
                float(np.mean([row["distinct_targets"] for row in eps_task])), 6
            ),
            "mean_distinct_milestone_indices_per_episode": round(
                float(np.mean([row["distinct_milestone_indices"] for row in eps_task])), 6
            ),
        }

    g_training = horizon / float(training_h)
    g_execution = horizon / float(execution_h)
    return {
        "pairs": n,
        "episodes": len(slices),
        "task_count": len(TASK_NAMES),
        "schema_target_episode_semantics": "tgt_fi is indexed within cur_ep; no separate tgt_ep field",
        "checks": {
            "unique_episode_frame_pairs": duplicate_pair_count == 0,
            "complete_per_episode_frame_coverage": incomplete_frame_coverage_count == 0,
            "target_not_before_current": bool(np.all(horizon >= 0)),
            "target_inside_source_episode": outside_episode_count == 0,
            "target_sequence_monotone": nonmonotone_target_count == 0,
            "one_task_per_episode": mixed_task_episode_count == 0,
            "terminal_fallback_on_every_episode": terminal_fallback_episode_count == len(slices),
            "source_audit_hash_matches": True,
        },
        "global": {
            "horizon_frames": {"mean": round(float(horizon.mean()), 6), "quantiles": q(horizon)},
            "within_training_chunk_rate_h_le_49": round(float(np.mean(horizon <= training_h)), 8),
            "beyond_training_chunk_rate_h_gt_49": round(float(np.mean(horizon > training_h)), 8),
            "within_executed_prefix_rate_h_le_35": round(float(np.mean(horizon <= execution_h)), 8),
            "beyond_executed_prefix_rate_h_gt_35": round(float(np.mean(horizon > execution_h)), 8),
            "g_training_h49": {"mean": round(float(g_training.mean()), 6), "quantiles": q(g_training)},
            "g_execution_h35": {"mean": round(float(g_execution.mean()), 6), "quantiles": q(g_execution)},
            "adjacent_same_target_rate": round(float(adjacent_same_target_count / adjacent_count), 8),
            "terminal_target_pair_rate": round(
                float(np.mean(tgt == np.repeat([cur[s].max() for s in slices], [s.stop - s.start for s in slices]))),
                8,
            ),
        },
        "by_task": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    repo = args.repo.resolve()

    paths = {
        "release_checkpoint": repo / "lmvla/lawam/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt",
        "release_config": repo / "lmvla/lawam/results/Checkpoints/robotwin/lawam_robotwin_sft_release/config.yaml",
        "release_statistics": repo / "lmvla/lawam/results/Checkpoints/robotwin/lawam_robotwin_sft_release/dataset_statistics.json",
        "tg1b_local_checkpoint": repo / "lmvla/lawam/results/Checkpoints/robotwin/20260730_234942+robotwin_all6_v2_local_seed2027/final_model/pytorch_model.pt",
        "tg1b_local_config": repo / "lmvla/lawam/results/Checkpoints/robotwin/20260730_234942+robotwin_all6_v2_local_seed2027/config.yaml",
        "tg1b_future_off_checkpoint": repo / "lmvla/lawam/results/Checkpoints/robotwin/20260731_172204+robotwin_all6_v2_nowm_seed2027/final_model/pytorch_model.pt",
        "tg1b_future_off_config": repo / "lmvla/lawam/results/Checkpoints/robotwin/20260731_172204+robotwin_all6_v2_nowm_seed2027/config.yaml",
        "robotwin_config": repo / "lmvla/lawam/starVLA/config/training/train_robotwin.yaml",
        "libero_config": repo / "lmvla/lawam/starVLA/config/training/train_libero.yaml",
        "libero_info": repo / "lmvla/lawam/dataset/libero_merged_no_noops_20hz/meta/info.json",
        "local_robotwin_info": repo / "lmvla/lawam/dataset/robotwin2_lmwm_all6_v2_v30/meta/info.json",
        "pairs": repo / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz",
        "pairs_audit": repo / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/AUDIT.json",
        "loader": repo / "lmvla/lawam/starVLA/dataloader/lerobot_datasets.py",
        "runtime_contracts": repo / "lmvla/lawam/starVLA/model/framework/latent_world/runtime/contracts.py",
        "robotwin_eval": repo / "lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_interface.py",
        "robotwin_eval_launcher": repo / "lmvla/lawam/examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    assert not missing, missing

    loader_text = paths["loader"].read_text()
    runtime_text = paths["runtime_contracts"].read_text()
    eval_text = paths["robotwin_eval"].read_text()
    launcher_text = paths["robotwin_eval_launcher"].read_text()
    assert "chunk_len = int(sec_chunk_f * fps)" in loader_text
    assert "return [int(action_arr[0]), int(action_arr[-1])]" in loader_text
    assert "abs(sec_chunk_f - horizon_sec) > 1e-6" in runtime_text
    assert '"robotwin_eef_30hz": 30.0' in eval_text
    assert '"robotwin2_lmwm_all6_v2": 50.0' in eval_text
    assert 'ROBOTWIN_REPLAN_STEPS="${ROBOTWIN_REPLAN_STEPS:-36}"' in launcher_text

    release_config = read_yaml(paths["release_config"])
    tg1b_local_config = read_yaml(paths["tg1b_local_config"])
    tg1b_future_off_config = read_yaml(paths["tg1b_future_off_config"])
    assert tg1b_local_config["seed"] == tg1b_future_off_config["seed"] == 2027
    assert tg1b_local_config["datasets"]["vla_data"]["data_mix"] == "robotwin2_lmwm_all6_v2"
    assert tg1b_future_off_config["datasets"]["vla_data"]["data_mix"] == "robotwin2_lmwm_all6_v2"
    assert tg1b_local_config["framework"]["action_model"]["future_prediction"] is True
    assert tg1b_local_config["framework"]["action_model"]["enable_loss_distill"] is True
    assert tg1b_future_off_config["framework"]["action_model"]["future_prediction"] is False
    assert tg1b_future_off_config["framework"]["action_model"]["enable_loss_distill"] is False
    local_config = read_yaml(paths["robotwin_config"])
    local_config["datasets"]["vla_data"]["data_mix"] = "robotwin2_lmwm_all6_v2"
    libero_config = read_yaml(paths["libero_config"])
    libero_info = read_json(paths["libero_info"])
    local_info = read_json(paths["local_robotwin_info"])

    release = contract_row(
        name="released_lawam_robotwin_checkpoint",
        config=release_config,
        fps=30.0,
        fps_evidence="frozen evaluator mapping; original release dataset directory is not local",
        executed_actions_per_query=36,
    )
    local = contract_row(
        name="completed_local_lmwm_all6_matrix",
        config=local_config,
        fps=float(local_info["fps"]),
        fps_evidence="local frozen dataset meta/info.json",
        executed_actions_per_query=36,
    )
    libero = contract_row(
        name="released_lawam_libero_recipe",
        config=libero_config,
        fps=float(libero_info["fps"]),
        fps_evidence="local dataset meta/info.json",
        executed_actions_per_query=None,
    )
    assert release["valid_action_count_H"] == 36 and release["last_valid_offset_h"] == 35
    assert release["target_is_inside_executed_prefix"] is True
    assert local["valid_action_count_H"] == 50 and local["last_valid_offset_h"] == 49
    assert local["target_is_inside_executed_prefix"] is False
    assert libero["valid_action_count_H"] == 8 and libero["last_valid_offset_h"] == 7

    result = {
        "protocol": "temporal_grounding_local_audit_v1",
        "date_utc": "2026-08-07",
        "evidence_class": "CPU-only descriptive source/data audit; no policy utility or causal mechanism claim",
        "notation": {
            "H": "number of valid actions in the model action-time grid",
            "h": "H-1, the last valid discrete offset",
            "tau_t": "raw milestone target frame for current frame t",
            "g_training": "(tau(t)-t)/49 for the completed local 50-action training chunk",
            "g_execution": "(tau(t)-t)/35 for the completed 36-action executed prefix",
            "z": "target visual representation",
        },
        "contracts": [release, local, libero],
        "tg1b_checkpoint_pair": {
            "training_seed": 2027,
            "data_mix": "robotwin2_lmwm_all6_v2",
            "local_wm": str(paths["tg1b_local_checkpoint"].relative_to(repo)),
            "future_off": str(paths["tg1b_future_off_checkpoint"].relative_to(repo)),
            "identity_checks": {
                "same_training_seed": True,
                "same_data_mix": True,
                "local_future_prediction_and_distillation_enabled": True,
                "future_off_prediction_and_distillation_disabled": True,
            },
        },
        "milestones": milestone_audit(paths["pairs"], paths["pairs_audit"], training_h=49, execution_h=35),
        "task_strata_frozen_before_new_policy_outcomes": TASK_STRATA,
        "limitations": [
            "The released RoboTwin training dataset is not present locally, so its 30 Hz value is audited from the frozen evaluator mapping rather than original meta/info.json.",
            "The release artifact does not include the exact original training source/config field for num_frames; [0,35] is the endpoint implied by the audited current two-frame loader rule, not an independent reconstruction of the original training sample stream.",
            "The milestone pair schema indexes tgt_fi within cur_ep and does not store an independent tgt_ep field.",
            "Temporal alignment and horizon distributions are descriptive; they do not show that future content causes control utility.",
        ],
        "sha256": {str(path.relative_to(repo)): sha256(path) for path in paths.values()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
