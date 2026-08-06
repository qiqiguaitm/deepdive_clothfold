import hashlib
import json
from pathlib import Path

import pytest

from summarize_pi05_step40000_safety import RESULTS, TASKS, summarize


def test_safety_summary_uses_paired_frozen_scenes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    manifest_path.parent.mkdir(parents=True)
    seeds = [100000 + index for index in range(50)]
    manifest = {"eval_seeds": {"0": {task: seeds for task in TASKS}}}
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    base = tmp_path / "lmvla/lawam/results/eval_runs/robotwin"

    for arm_index, template in enumerate(RESULTS.values()):
        for task in TASKS:
            path = base / template.format(task=task) / "seed0/run/tasks" / task / "summary.json"
            path.parent.mkdir(parents=True)
            episodes = [
                {"seed": seed, "success": index < 20 + 5 * arm_index}
                for index, seed in enumerate(seeds)
            ]
            path.write_text(
                json.dumps(
                    {
                        "task_name": task,
                        "n_episodes": 50,
                        "successes": sum(item["success"] for item in episodes),
                        "fixed_seed_manifest": {"sha256": manifest_sha},
                        "episodes": episodes,
                    }
                )
            )

    report = summarize(tmp_path)
    assert report["complete"] is True
    assert report["paired_vs_a0"]["a2_abs_40k"][TASKS[0]][
        "success_rate_delta_vs_a0"
    ] == 0.1
    assert report["paired_vs_a0"]["a3_live_40k"][TASKS[0]][
        "success_rate_delta_vs_a0"
    ] == 0.2

    corrupted = next(
        (
            tmp_path
            / "lmvla/lawam/results/eval_runs/robotwin"
            / RESULTS["a2_abs_40k"].format(task=TASKS[0])
        ).glob(f"seed0/**/tasks/{TASKS[0]}/summary.json")
    )
    payload = json.loads(corrupted.read_text())
    payload["episodes"][0]["seed"] += 1
    corrupted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen seed order"):
        summarize(tmp_path)
