import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

import pytest


MODULE_PATH = Path(__file__).with_name("resource_aware_scheduler.py")
SPEC = importlib.util.spec_from_file_location("resource_aware_scheduler", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
scheduler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheduler)


def test_primary_north_operational_limits_are_25() -> None:
    assert scheduler.NORTH_PERSONAL_LIMIT == 25
    assert scheduler.NORTH_PRIMARY_MAX_JOBS == 25


def test_readiness_hashes_require_exact_file_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen\n")
    expected = scheduler.sha256_file(source)
    spec = {"ready_hashes": [{"path": str(source), "sha256": expected}]}

    assert scheduler.readiness_spec_satisfied(spec)
    assert scheduler.readiness_hash_failures(spec) == []

    source.write_text("drifted\n")
    assert not scheduler.readiness_spec_satisfied(spec)
    assert scheduler.readiness_hash_failures(spec)[0].startswith(f"{source}:")


def test_readiness_supports_explicit_directories(tmp_path: Path) -> None:
    cache = tmp_path / "frame_cache"
    spec = {"ready_dirs": [str(cache)]}

    assert not scheduler.readiness_spec_satisfied(spec)
    cache.mkdir()
    assert scheduler.readiness_spec_satisfied(spec)


def test_candidate_failure_count_respects_rearm_epoch() -> None:
    task_state = {
        "ignore_failures_before": "2026-08-05T12:00:00Z",
        "attempts": [
            {
                "resource": "local",
                "failure": "old protocol rejection",
                "finished_at": "2026-08-05T11:00:00Z",
            },
            {
                "resource": "local",
                "failure": "new runtime failure",
                "finished_at": "2026-08-05T12:01:00Z",
            },
        ],
    }

    assert scheduler.candidate_failure_count(task_state, {"resource": "local"}) == 1


def test_p2_frame_cache_uses_directory_readiness() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    tasks = {task["id"]: task for task in queue["tasks"]}

    for seed in (1001, 1002):
        task = tasks[f"pi05_predictive_adapter_p2_candidate_seed{seed}_train"]
        assert task["ready_dirs"] == [
            str(
                scheduler.REPO
                / "lmvla/lawam/dataset/robotwin2.0/frame_cache_jpeg256"
            )
        ]
        assert all("frame_cache_jpeg256" not in path for path in task["ready_files"])


def test_frozen_source_readiness_covers_p1_p2_and_r1_gpu_jobs() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_r1_recurrence_aligned_tasks(queue)
    scheduler.apply_frozen_source_readiness(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    p1 = tasks["pi05_predictive_adapter_p1_a0_seed1000_train"]
    p1_eval = tasks["pi05_predictive_adapter_p1_a0_seed1000_eval"]
    p2_eval = tasks["pi05_predictive_adapter_p2_candidate_seed1001_eval"]
    p2_train = tasks["pi05_predictive_adapter_p2_candidate_seed1001_train"]
    r1_train = tasks["pi05_r1_combined_seed1001_train"]
    r1_eval = tasks["pi05_r1_combined_seed1000_eval"]
    r1_unauthorized_eval = tasks["pi05_r1_combined_seed1001_eval"]
    assert len(p1["ready_hashes"]) == 4
    assert p1_eval["ready_hashes"] != p1["ready_hashes"]
    assert any(
        item["path"].startswith(str(scheduler.R1_FROZEN_OVERLAY))
        for item in p1_eval["ready_hashes"]
    )
    assert any(
        item["path"].endswith("run_pi05_predictive_adapter_p1_frozen.sh")
        for item in p1_eval["ready_hashes"]
    )
    assert "run_pi05_predictive_adapter_p1_frozen.sh" in p1_eval["candidates"][0][
        "command"
    ]
    p1_east = next(
        candidate
        for candidate in p1_eval["candidates"]
        if candidate["resource"] == "Robot-East-H20"
    )
    assert p1_east["env"]["P1_VERIFY_REPO"] == str(scheduler.R1_FROZEN_OVERLAY)
    assert p1_east["env"]["ROBOTWIN_ATTACH_REQUEUE_FAILED"] == "1"
    assert p1_east["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert p2_eval["ready_hashes"] != p1["ready_hashes"]
    assert any(
        item["path"].startswith(str(scheduler.REPLICATION_FROZEN_OVERLAY))
        for item in p2_eval["ready_hashes"]
    )
    assert any(
        item["path"].endswith(
            "frozen_sources/pi05_replication_v1/train_scripts/kai/eval/"
            "run_pi05_predictive_adapter_p2_formal.sh"
        )
        for item in p2_eval["ready_hashes"]
    )
    assert "P2_VERIFY_REPO=" in p2_eval["candidates"][0]["command"]
    assert "frozen_sources/pi05_replication_v1" in p2_eval["candidates"][0][
        "command"
    ]
    p2_eval_east = next(
        candidate
        for candidate in p2_eval["candidates"]
        if candidate["resource"] == "Robot-East-H20"
    )
    assert p2_eval_east["env"]["P2_VERIFY_REPO"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY
    )
    assert p2_eval_east["env"]["PYTHONPATH"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY / "kai0/src"
    )
    assert p2_eval_east["env"]["P2_EVAL_LAUNCHER"].endswith(
        "frozen_sources/pi05_replication_v1/train_scripts/kai/eval/"
        "run_pi05_predictive_adapter_p2_formal.sh"
    )
    assert p2_train["ready_hashes"] != p1["ready_hashes"]
    assert any(
        item["path"].startswith(str(scheduler.REPLICATION_FROZEN_OVERLAY))
        for item in p2_train["ready_hashes"]
    )
    p2_east = next(
        candidate
        for candidate in p2_train["candidates"]
        if candidate["resource"] == "Robot-East-H20"
    )
    assert p2_east["env"]["P2_VERIFY_REPO"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY
    )
    assert p2_east["env"]["TRAIN_SOURCE_REPO"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY
    )
    assert len(r1_train["ready_hashes"]) >= 20
    assert r1_eval["ready_hashes"] != r1_train["ready_hashes"]
    assert any(
        item["path"].startswith(str(scheduler.R1_FROZEN_OVERLAY))
        for item in r1_eval["ready_hashes"]
    )
    assert any(
        item["path"].endswith("run_pi05_r1_formal.sh")
        for item in r1_eval["ready_hashes"]
    )
    assert r1_unauthorized_eval["ready_hashes"] != r1_train["ready_hashes"]
    assert any(
        item["path"].startswith(str(scheduler.REPLICATION_FROZEN_OVERLAY))
        for item in r1_train["ready_hashes"]
    )
    r1_train_east = next(
        candidate
        for candidate in r1_train["candidates"]
        if candidate["resource"] == "Robot-East-H20"
    )
    assert r1_train_east["env"]["R1_VERIFY_REPO"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY
    )
    assert r1_train_east["env"]["TRAIN_SOURCE_REPO"] == str(
        scheduler.REPLICATION_FROZEN_OVERLAY
    )
    overlay_ready = str(scheduler.R1_FROZEN_OVERLAY / "READY")
    cpu_preflight = str(scheduler.R1_FROZEN_OVERLAY / "CPU_PREFLIGHT")
    assert r1_eval["ready_files"].count(overlay_ready) == 1
    assert r1_eval["ready_files"].count(cpu_preflight) == 1
    assert "ROBOTWIN_ATTACH_REQUEUE_FAILED=1" in r1_eval["candidates"][0]["command"]
    east_candidate = next(
        candidate
        for candidate in r1_eval["candidates"]
        if candidate["resource"] == "Robot-East-H20"
    )
    assert east_candidate["env"]["ROBOTWIN_ATTACH_REQUEUE_FAILED"] == "1"
    assert east_candidate["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert east_candidate["env"]["TORCH_EXTENSIONS_DIR"] == (
        "/vePFS/tim/runtime/torch_extensions/h20_sm90_py310"
    )
    assert east_candidate["max_failures"] == 6
    assert east_candidate["retry_cooldown_seconds"] == 60
    scheduler.apply_frozen_source_readiness(queue)
    assert r1_eval["ready_files"].count(overlay_ready) == 1
    assert r1_eval["ready_files"].count(cpu_preflight) == 1
    replication_ready = str(
        scheduler.REPLICATION_FROZEN_OVERLAY / "REPLICATION_READY"
    )
    assert p2_eval["ready_files"].count(replication_ready) == 1
    assert p2_train["ready_files"].count(replication_ready) == 1
    assert r1_train["ready_files"].count(replication_ready) == 1
    assert "ready_hashes" not in tasks["pi05_r1_seed1000_gate"]


def test_replication_launchers_keep_canonical_outputs_and_frozen_sources_separate() -> None:
    launchers = {
        "train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh": (
            "TRAIN_VERIFY_REPO",
            "TRAIN_SOURCE_REPO",
        ),
        "train_scripts/kai/run_pi05_predictive_adapter_p2_train.sh": (
            "P2_VERIFY_REPO",
            "TRAIN_SOURCE_REPO",
        ),
        "train_scripts/kai/run_pi05_r1_train.sh": (
            "R1_VERIFY_REPO",
            "TRAIN_SOURCE_REPO",
        ),
    }
    for relative, required in launchers.items():
        text = (scheduler.REPO / relative).read_text()
        assert 'REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}' in text
        assert 'test -s "$' in text and "/REPLICATION_READY\"" in text
        for variable in required:
            assert variable in text

    p1_train_launcher = (
        scheduler.REPO / "train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
    ).read_text()
    assert 'cd "$REPO/kai0"' in p1_train_launcher
    assert 'cd "$TRAIN_SOURCE_REPO/kai0"' not in p1_train_launcher

    eval_launcher = (
        scheduler.REPO
        / "lmvla/paper_iclr_lmvla/frozen_sources/pi05_replication_v1/"
        "train_scripts/kai/eval/run_pi05_predictive_adapter_p2_formal.sh"
    ).read_text()
    assert 'REPO=${REPO:-/vePFS/tim/workspace/deepdive_kai0}' in eval_launcher
    assert 'VERIFY_REPO=${P2_VERIFY_REPO:-$REPO}' in eval_launcher
    assert 'test -s "$VERIFY_REPO/REPLICATION_READY"' in eval_launcher
    assert '--repo "$VERIFY_REPO"' in eval_launcher
    assert 'RESULT_ROOT=$REPO/' in eval_launcher
    assert 'CKPT=${CKPT:-$REPO/' in eval_launcher


def test_r4_collection_is_smoke_gated_and_isolated() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_r4_outcome_collection_tasks(queue)
    scheduler.add_pi05_r4_outcome_collection_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}
    smoke = tasks["pi05_r4_outcome_collection_smoke"]
    formal = tasks["pi05_r4_outcome_collection_formal"]
    support = tasks["pi05_r4_beat_train_support_supplement"]
    balanced_support = tasks["pi05_r4_balanced_train_support_supplement"]
    finalize = tasks["pi05_r4_outcome_dataset_finalize"]
    query_smoke = tasks["pi05_r4_query_collection_smoke_v3"]
    query_base = tasks["pi05_r4_query_base_train_collection"]
    query_support = tasks["pi05_r4_query_beat_support_collection"]
    query_balanced = tasks["pi05_r4_query_balanced_support_collection"]
    query_finalize = tasks["pi05_r4_query_dataset_finalize"]
    training_chunks = tasks["pi05_r4_training_chunks_build"]
    lerobot_build = tasks["pi05_r4_lerobot_dataset_build"]
    runtime_verify = tasks["pi05_r4_training_runtime_verify"]
    outcome_free = tasks["pi05_r4_outcome_free_manifest_build"]
    crave_sidecar = tasks["pi05_r4_crave_sidecar_build"]
    matched_runtime = tasks["pi05_r4_matched_runtime_verify"]
    training_smoke = tasks["pi05_r4_training_smoke"]
    checkpoint_permissions = tasks["pi05_r4_checkpoint_permissions"]
    formal_training = {
        arm: tasks[task_id]
        for arm, task_id in {
            "ordinary": "pi05_r4_ordinary_seed1000_train",
            "terminal_outcome": "pi05_r4_terminal_outcome_seed1000_train",
            "outcome_free_crave": "pi05_r4_outcome_free_crave_seed1000_train",
        }.items()
    }
    formal_eval = {
        arm: tasks[f"pi05_r4_{arm}_seed1000_eval"]
        for arm in ("ordinary", "terminal_outcome", "outcome_free_crave")
    }
    formal_gate = tasks["pi05_r4_seed1000_gate"]
    north_stage = tasks["pi05_r4_eval_north_stage"]
    north_materializers = {
        arm: tasks[f"pi05_r4_{arm}_eval_materialize_north"]
        for arm in ("terminal_outcome", "outcome_free_crave")
    }

    assert smoke["candidates"][0]["resource"] == "local"
    assert smoke["candidates"][0]["gpus"] == 1
    assert "R4_FINALIZE_DATASET=0" in smoke["candidates"][0]["command"]
    assert formal["candidates"][0]["resource"] == "Robot-East-H20"
    assert formal["candidates"][0]["gpus"] == 4
    assert formal["completion_glob"].endswith("pi05_r4_outcomes_public_v1/dataset_manifest.json")
    assert formal["candidates"][0]["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert formal["candidates"][0]["env"]["TORCH_EXTENSIONS_DIR"].endswith(
        "h20_sm90_py310"
    )
    assert any(path.endswith("pi05_r4_outcome_collection_smoke.ok") for path in formal["ready_files"])
    assert all(
        not item["path"].startswith(str(scheduler.REPO / "lmvla/lawam/"))
        for item in formal["ready_hashes"]
    )
    assert any(
        "frozen_source_overlays/pi05_r4_collector_v1" in item["path"]
        for item in formal["ready_hashes"]
    )
    assert support["candidates"][0]["resource"] == "local"
    assert support["candidates"][0]["gpus"] == 2
    assert "ROBOTWIN_TEST_NUM=40" in support["candidates"][0]["command"]
    assert "R4_FINALIZE_DATASET=0" in support["candidates"][0]["command"]
    assert sum(path.endswith("beat_block_hammer/summary.json") for path in support["ready_files"]) == 4
    assert any(
        item["path"].endswith("pi05_r4_beat_train_support_supplement_v1.json")
        for item in support["ready_hashes"]
    )
    assert any(
        item["path"].endswith("pi05_r4_outcome_support_amendment_v1.json")
        for item in support["ready_hashes"]
    )
    assert balanced_support["candidates"][0]["resource"] == "Robot-East-H20"
    assert balanced_support["candidates"][0]["gpus"] == 4
    assert balanced_support["completion_min_count"] == 2
    assert any(
        item["path"].endswith("dataset_audit_combined_v1.json")
        for item in balanced_support["ready_hashes"]
    )
    assert finalize["candidates"][0]["gpus"] == 0
    assert finalize["candidates"][0]["resource"] == "local"
    assert finalize["rearm_after_ready_file"].endswith(
        "pi05_r4_balanced_support_a.ok"
    )
    assert any(path.endswith("pi05_r4_beat_train_support_supplement.ok") for path in finalize["ready_files"])
    assert finalize["completion_glob"].endswith("pi05_r4_outcome_collection.ok")
    assert any(
        item["path"].endswith("pi05_r4_outcome_merge_amendment_v1.json")
        for item in finalize["ready_hashes"]
    )
    assert query_smoke["candidates"][0]["resource"] == "local"
    assert query_smoke["candidates"][0]["gpus"] == 1
    assert "run_pi05_r4_query_collection.sh" in query_smoke["candidates"][0]["command"]
    assert "pi05_r4_query_smoke_scene_seeds_v1.json" in query_smoke["candidates"][0]["command"]
    assert "pi05_r4_query_smoke_scene_seeds_v1.json" not in smoke["candidates"][0]["command"]
    assert query_base["candidates"][0]["resource"] == "Robot-East-H20"
    assert query_base["candidates"][0]["gpus"] == 4
    assert query_base["completion_min_count"] == 2
    assert any(path.endswith("pi05_r4_outcome_collection.ok") for path in query_base["ready_files"])
    assert query_support["candidates"][0]["resource"] == "local"
    assert query_support["candidates"][0]["gpus"] == 2
    assert "ROBOTWIN_TEST_NUM=40" in query_support["candidates"][0]["command"]
    assert query_balanced["candidates"][0]["resource"] == "Robot-East-H20"
    assert query_balanced["candidates"][0]["gpus"] == 4
    assert query_balanced["completion_min_count"] == 2
    assert query_finalize["candidates"][0]["gpus"] == 0
    assert query_finalize["completion_glob"].endswith("pi05_r4_query_dataset.ok")
    for task in (query_smoke, query_base, query_support, query_balanced, query_finalize):
        assert any(
            item["path"].endswith("pi05_r4_query_base_train_east_4h20.yaml")
            for item in task["ready_hashes"]
        )
    assert training_chunks["candidates"][0]["resource"] == "local"
    assert training_chunks["candidates"][0]["gpus"] == 0
    assert training_chunks["completion_glob"].endswith("pi05_r4_training_chunks.ok")
    assert any(
        path.endswith("pi05_r4_outcome_collection.ok")
        for path in training_chunks["ready_files"]
    )
    assert any(
        path.endswith("pi05_r4_query_dataset.ok")
        for path in training_chunks["ready_files"]
    )
    command = training_chunks["candidates"][0]["command"]
    assert "build_pi05_r4_training_chunks.py" in command
    assert "query_action_chunks.npz" in command
    assert "does not authorize policy training" in training_chunks["description"]
    assert lerobot_build["candidates"][0]["resource"] == "local"
    assert lerobot_build["candidates"][0]["gpus"] == 0
    assert lerobot_build["completion_glob"].endswith("pi05_r4_lerobot_dataset.ok")
    assert any(
        path.endswith("pi05_r4_training_chunks.ok")
        for path in lerobot_build["ready_files"]
    )
    lerobot_command = lerobot_build["candidates"][0]["command"]
    assert "build_pi05_r4_lerobot_dataset.py" in lerobot_command
    assert "lerobot_query_chunks" in lerobot_command
    assert "does not authorize policy training" in lerobot_build["description"]
    assert runtime_verify["candidates"][0]["resource"] == "local"
    assert runtime_verify["candidates"][0]["gpus"] == 0
    assert runtime_verify["completion_glob"].endswith("pi05_r4_training_runtime.ok")
    assert lerobot_build["completion_glob"] in runtime_verify["ready_files"]
    runtime_command = runtime_verify["candidates"][0]["command"]
    assert "PI05_R4_TRAINING_RUNTIME=1" in runtime_command
    assert "--load-policy" in runtime_command
    assert "&& exec env" not in runtime_command
    assert "pi05_r4_training_runtime.ok" in runtime_command
    assert "does not authorize policy training" in runtime_verify["description"]
    assert outcome_free["candidates"][0]["resource"] == "local"
    assert outcome_free["candidates"][0]["gpus"] == 0
    assert outcome_free["completion_glob"].endswith(
        "pi05_r4_outcome_free_manifest.ok"
    )
    outcome_free_command = outcome_free["candidates"][0]["command"]
    assert "build_pi05_r4_outcome_free_manifest.py" in outcome_free_command
    assert "outcome_free_query_manifest.json" in outcome_free_command
    assert any(
        item["path"].endswith("query_action_chunks.npz")
        and item["sha256"]
        == "ef47ce3cae6449bb440db6ecd502c687205eed51364fc40574c023c01c33c966"
        for item in outcome_free["ready_hashes"]
    )
    assert [candidate["resource"] for candidate in crave_sidecar["candidates"]] == [
        "local",
        "Robot-East-H20",
    ]
    assert all(candidate["gpus"] == 1 for candidate in crave_sidecar["candidates"])
    assert crave_sidecar["completion_glob"].endswith("pi05_r4_crave_sidecar.ok")
    assert any(
        path.endswith("pi05_r4_outcome_free_manifest.ok")
        for path in crave_sidecar["ready_files"]
    )
    assert any(
        path.endswith("pi05_r4_training_chunks.ok")
        for path in crave_sidecar["ready_files"]
    )
    assert "crave_weights.npz" in crave_sidecar["candidates"][0]["command"]
    assert matched_runtime["candidates"][0]["resource"] == "local"
    assert matched_runtime["candidates"][0]["gpus"] == 0
    assert matched_runtime["completion_glob"].endswith("pi05_r4_matched_runtime.ok")
    assert crave_sidecar["completion_glob"] in matched_runtime["ready_files"]
    assert runtime_verify["completion_glob"] in matched_runtime["ready_files"]
    matched_command = matched_runtime["candidates"][0]["command"]
    assert "--sidecar" in matched_command
    assert "--load-policy" in matched_command
    assert "&& exec env" not in matched_command
    assert "pi05_r4_matched_runtime.ok" in matched_command
    assert training_smoke["candidates"][0]["resource"] == "Robot-East-H20"
    assert training_smoke["candidates"][0]["gpus"] == 4
    assert training_smoke["completion_glob"].endswith(
        "pi05_r4_smoke-ordinary-4g.ok"
    )
    assert training_smoke["rearm_after_ready_file"].endswith(
        "pi05_r4_training_smoke_amendment_v1.json"
    )
    assert runtime_verify["completion_glob"] in training_smoke["ready_files"]
    assert matched_runtime["completion_glob"] in training_smoke["ready_files"]
    assert crave_sidecar["completion_glob"] in training_smoke["ready_files"]
    assert "formal R4 training remains blocked" in training_smoke["description"]
    assert any(
        item["path"].endswith("pi05_r4_training_smoke_amendment_v1.json")
        for item in training_smoke["ready_hashes"]
    )
    expected_yamls = {
        "ordinary": "pi05_r4_train_ordinary_east_4h20.yaml",
        "terminal_outcome": "pi05_r4_train_terminal_east_4h20.yaml",
        "outcome_free_crave": "pi05_r4_train_crave_east_4h20.yaml",
    }
    for arm, task in formal_training.items():
        candidate = task["candidates"][0]
        assert task["priority"] == 1
        assert candidate["resource"] == "Robot-East-H20"
        assert candidate["gpus"] == 4
        assert candidate["max_failures"] == 1
        assert candidate["yaml"].endswith(expected_yamls[arm])
        assert training_smoke["completion_glob"] in task["ready_files"]
        assert any(
            item["path"].endswith("pi05_r4_formal_training_amendment_v1.json")
            for item in task["ready_hashes"]
        )
        assert "policy-effect claims remain blocked" in task["description"]
        assert task["progress_logs"][0]["label"] == "step"
        assert task["progress_logs"][0]["regex"] == (
            r"Training:[^\r\n]*?(\d+)/5000"
        )
    for arm, task in formal_eval.items():
        assert task["priority"] == 2
        assert task["prefer_max_gpus_when_immediate"] is True
        expected_resources = ["Robot-East-H20", "local"]
        expected_gpus = [4, 2]
        if arm != "ordinary":
            expected_resources.append("Robot-North-H20")
            expected_gpus.append(4)
        assert [candidate["resource"] for candidate in task["candidates"]] == expected_resources
        assert [candidate["gpus"] for candidate in task["candidates"]] == expected_gpus
        assert task["candidates"][0]["env"]["R4_ARM"] == arm
        assert f"R4_ARM={arm}" in task["candidates"][1]["command"]
        assert "ROBOTWIN_NUM_SLOTS=2" in task["candidates"][1]["command"]
        assert "NUM_WORKERS=2" in task["candidates"][1]["command"]
        assert "ROBOTWIN_NUM_SLOTS" not in task["candidates"][0]["env"]
        assert any(
            item["path"].endswith("pi05_r4_formal_eval_protocol_v1.json")
            for item in task["ready_hashes"]
        )
        assert any(
            item["path"].endswith(
                "pi05_r4_local_eval_parallelism_amendment_v1.json"
            )
            for item in task["ready_hashes"]
        )
        assert any(
            item["path"].endswith("run_pi05_r4_formal_eval.sh")
            and item["sha256"]
            == "e91ab34a71ae82b3e49099a5fb73154c6493fb2c73954324d4b775dec074bb86"
            for item in task["ready_hashes"]
        )
        assert any(
            item["path"].endswith("pi05_r4_action_bridge_amendment_v1.json")
            for item in task["ready_hashes"]
        )
        assert any(
            item["path"].endswith("serve_lerobot_pi05.py")
            and item["sha256"]
            == "acf914ba7038463e5412ec0cdc05a153dc25af666274dad519fee4692eb3d0d5"
            for item in task["ready_hashes"]
        )
        assert any(
            path.endswith("pi05_r4_action_bridge_preflight.ok")
            for path in task["ready_files"]
        )
        assert checkpoint_permissions["completion_glob"] in task["ready_files"]
        assert task["progress_globs"][0]["expected"] == 24
        if arm != "ordinary":
            assert any(
                item["path"].endswith("pi05_r4_north_eval_amendment_v1.json")
                for item in task["ready_hashes"]
            )
            assert {location["label"] for location in task["completion_locations"]} == {
                "shared",
                "north",
            }
    assert north_stage["priority"] == 0
    assert north_stage["candidates"][0]["resource"] == "local"
    assert north_stage["candidates"][0]["gpus"] == 0
    assert north_stage["rearm_after_ready_file"].endswith(
        "pi05_r4_north_eval_amendment_v1.json"
    )
    assert "stage_pi05_r4_eval_to_north.sh" in north_stage["candidates"][0]["command"]
    assert "R4_NORTH_VALIDATE_ONLY=1" in north_stage["candidates"][0]["command"]
    north_stage_script = (
        scheduler.REPO / "train_scripts/kai/stage_pi05_r4_eval_to_north.sh"
    ).read_text()
    assert north_stage_script.index('chmod 0755 "$python"') < north_stage_script.index(
        'test -x "$python"'
    )
    assert 'chmod 0755 "$stage/train_scripts/kai/eval/robotwin_python_wrapper_north.sh"' in (
        north_stage_script
    )
    assert any(
        item["path"].endswith("pi05_r4_eval_north_4h20.yaml")
        for item in north_stage["ready_hashes"]
    )
    for arm, materialize in north_materializers.items():
        assert materialize["materialize_north_result_for"] == formal_eval[arm]["id"]
        assert materialize["candidates"][0]["gpus"] == 0
        assert f"R4_ARM={arm}" in materialize["candidates"][0]["command"]
        assert "sync_pi05_r4_eval_from_north.sh" in materialize["candidates"][0]["command"]
    assert checkpoint_permissions["priority"] == 0
    assert checkpoint_permissions["candidates"][0]["resource"] == "Robot-East-H20"
    assert checkpoint_permissions["candidates"][0]["gpus"] == 1
    assert checkpoint_permissions["completion_glob"].endswith(
        "pi05_r4_checkpoint_permissions.ok"
    )
    assert checkpoint_permissions["artifact_progress"][0]["glob"].endswith(
        "checkpoint_integrity_v1.json"
    )
    assert all(
        any(marker in path for path in checkpoint_permissions["ready_files"])
        for marker in (
            "pi05_r4_ordinary-seed1000.ok",
            "pi05_r4_terminal_outcome-seed1000.ok",
            "pi05_r4_outcome_free_crave-seed1000.ok",
        )
    )
    assert any(
        item["path"].endswith("pi05_r4_checkpoint_permissions_amendment_v1.json")
        for item in checkpoint_permissions["ready_hashes"]
    )
    assert formal_gate["candidates"][0]["resource"] == "local"
    assert formal_gate["candidates"][0]["gpus"] == 0
    assert formal_gate["priority"] == 2
    assert formal_gate["completion_glob"].endswith(
        "RESULTS_pi05_r4_seed1000_gate.json"
    )
    assert "--accepted-marker" in formal_gate["candidates"][0]["command"]
    assert all(
        f"pi05_r4_{arm}_seed1000.json" in formal_gate["candidates"][0]["command"]
        for arm in ("ordinary", "terminal_outcome", "outcome_free_crave")
    )


def test_r4_sidecar_north_stage_is_exact_and_materialized() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_r4_outcome_collection_tasks(queue)

    scheduler.add_pi05_r4_sidecar_north_tasks(queue)
    scheduler.add_pi05_r4_sidecar_north_tasks(queue)
    scheduler.validate_queue(queue)

    tasks = {task["id"]: task for task in queue["tasks"]}
    stage = tasks["pi05_r4_sidecar_north_stage"]
    assert stage["candidates"][0]["resource"] == "local"
    assert stage["candidates"][0]["gpus"] == 0
    assert "stage_pi05_r4_sidecar_to_north.sh" in stage["candidates"][0]["command"]
    assert any(
        item["path"].endswith("pi05_r4_crave_sidecar_north_1h20.yaml")
        for item in stage["ready_hashes"]
    )

    parent = tasks["pi05_r4_crave_sidecar_build"]
    north = [
        candidate
        for candidate in parent["candidates"]
        if candidate["resource"] == "Robot-North-H20"
    ]
    assert len(north) == 1
    candidate = north[0]
    assert candidate["gpus"] == 1
    assert candidate["ready_files"] == [
        str(scheduler.REPO / "logs/resource_markers/pi05_r4_sidecar_north_stage.ok")
    ]
    assert any(
        path.endswith("pi05_r4_sidecar_north_stage.ok")
        for path in candidate["ready_files_remote"]
    )
    assert {location["label"] for location in parent["completion_locations"]} == {
        "shared",
        "north",
    }

    materialize = tasks["pi05_r4_sidecar_materialize_north"]
    assert materialize["materialize_north_result_for"] == parent["id"]
    assert materialize["candidates"][0]["gpus"] == 0
    assert "sync_pi05_r4_sidecar_from_north.sh" in materialize["candidates"][0][
        "command"
    ]


def north_snapshot(
    *,
    primary: int = 20,
    all_users: int = 46,
    backup: int = 0,
    backup_enabled: bool = True,
    queueing: bool = False,
) -> dict:
    return {
        "resources": {
            "beijing": {
                "available": True,
                "capacity": 56,
                "personal_limit": 20,
                "owned_active_gpus": primary,
                "owned_queued_gpus": 0,
                "active_gpus_all_users": all_users,
                "owned_queueing": ["primary-job"] if queueing else [],
                "owned_submitted_jobs": 0,
                "max_submitted_jobs": 20,
                "queueing_all_users": ["other-job"] if queueing else [],
                "backup": {
                    "enabled": backup_enabled,
                    "available": True,
                    "managed_active_gpus": backup,
                    "managed_queued_gpus": 0,
                    "managed_queueing": [],
                    "managed_submitted_jobs": 0,
                    "max_submitted_jobs": 20,
                    "personal_limit": 20,
                },
            }
        }
    }


def north_candidate(gpus: int = 4) -> dict:
    return {
        "kind": "platform",
        "resource": "Robot-North-H20",
        "gpus": gpus,
    }


def test_north_candidate_respects_configured_submitted_job_quota() -> None:
    candidate = north_candidate(4)
    snapshot = north_snapshot(primary=0, all_users=0)
    beijing = snapshot["resources"]["beijing"]
    beijing["owned_submitted_jobs"] = 3
    beijing["max_submitted_jobs"] = 3
    assert not scheduler.candidate_available(candidate, snapshot, "primary")

    backup = beijing["backup"]
    backup["managed_submitted_jobs"] = 1
    backup["max_submitted_jobs"] = 2
    assert scheduler.candidate_available(candidate, snapshot, "backup")
    assert scheduler.candidate_credential_profile(candidate, snapshot) == "backup"
    scheduler.reserve_dispatched_candidate(snapshot, candidate, "backup")
    assert backup["managed_submitted_jobs"] == 2
    assert not scheduler.candidate_available(candidate, snapshot, "backup")


def test_submitted_job_states_include_all_pre_running_wait_states() -> None:
    assert scheduler.SUBMITTED_JOB_STATES == (
        "Running",
        "Deploying",
        "Creating",
        "Waiting",
        "Queueing",
    )


def test_permanent_resource_policy_removes_gf1_and_disables_orphans() -> None:
    queue = {
        "tasks": [
            {
                "id": "mixed",
                "candidates": [
                    {"resource": "gf1", "kind": "ssh", "gpus": 4},
                    {"resource": "Robot-East-H20", "kind": "platform", "gpus": 4},
                ],
            },
            {
                "id": "orphan",
                "candidates": [
                    {"resource": "gf1", "kind": "ssh", "gpus": 4},
                ],
            },
        ]
    }

    scheduler.apply_permanent_resource_policy(queue)

    mixed, orphan = queue["tasks"]
    assert [candidate["resource"] for candidate in mixed["candidates"]] == [
        "Robot-East-H20"
    ]
    assert mixed["retired_resource_candidates"] == ["gf1"]
    assert mixed.get("enabled", True)
    assert orphan["candidates"] == []
    assert not orphan["enabled"]
    assert "permanent host shutdown" in orphan["disabled_reason"]


def test_load_state_retires_running_gf1_attempt(monkeypatch, tmp_path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "tasks": {
                    "mixed": {
                        "status": "running",
                        "attempts": [
                            {
                                "resource": "gf1",
                                "kind": "ssh",
                                "pid": "123",
                                "monitor_error": "host unreachable",
                            }
                        ],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(scheduler, "STATE_PATH", state_path)
    queue = {
        "tasks": [
            {
                "id": "mixed",
                "candidates": [
                    {"resource": "Robot-East-H20", "kind": "platform", "gpus": 4}
                ],
            }
        ]
    }

    state = scheduler.load_state(queue)

    task_state = state["tasks"]["mixed"]
    assert task_state["status"] == "pending"
    assert "replacement resource" in task_state["waiting_reason"]
    assert task_state["attempts"][-1]["terminal_reason"] == (
        "resource permanently retired by operator"
    )
    assert "monitor_error" not in task_state["attempts"][-1]


def test_gf1_snapshot_disable_is_a_dispatch_backstop() -> None:
    candidate = {"resource": "gf1", "kind": "ssh", "gpus": 4}
    snapshot = {
        "resources": {
            "gf1": {
                "available": True,
                "submission_enabled": False,
                "count": 8,
                "free_count": 8,
                "gpus": [],
            }
        }
    }

    assert not scheduler.candidate_available(candidate, snapshot)
    with pytest.raises(RuntimeError, match="permanent host shutdown"):
        scheduler.launch_gf1(candidate)


def test_dispatch_order_uses_router_gpu_count_preferences() -> None:
    candidates = [
        {"resource": "Robot-North-H20", "gpus": 2},
        {"resource": "gf1", "gpus": 2},
        {"resource": "local", "gpus": 2},
        {"resource": "robot-task", "gpus": 2},
        {"resource": "Robot-East-H20", "gpus": 2},
    ]
    assert [
        candidate["resource"]
        for candidate in scheduler.ordered_dispatch_candidates(
            {"candidates": candidates}
        )
    ] == [
        "local",
        "Robot-North-H20",
        "gf1",
        "Robot-East-H20",
        "robot-task",
    ]

    for candidate in candidates:
        candidate["gpus"] = 4
    assert [
        candidate["resource"]
        for candidate in scheduler.ordered_dispatch_candidates(
            {"candidates": candidates}
        )
    ] == [
        "gf1",
        "Robot-East-H20",
        "Robot-North-H20",
        "robot-task",
        "local",
    ]


def test_dispatch_order_applies_router_locality_and_live_capacity() -> None:
    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["timestamp"] = "2026-08-03T13:00:00Z"
    snapshot["resources"].update(
        {
            "gf1": {"count": 8, "free_count": 4},
            "local": {"count": 2, "free_count": 2},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 8,
                "queueing_all_users": [],
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 32,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
        }
    )
    north_checkpoint = "/vePFS-North-E/vis_robot/checkpoints/model/49999"
    task = {
        "ready_files": [north_checkpoint + "/_CHECKPOINT_METADATA"],
        "ready_files_remote": [north_checkpoint + "/_CHECKPOINT_METADATA"],
        "candidates": [
            {"resource": "gf1", "kind": "ssh", "gpus": 4},
            {"resource": "Robot-North-H20", "kind": "platform", "gpus": 4},
        ],
    }
    assert [
        candidate["resource"]
        for candidate in scheduler.ordered_dispatch_candidates(task, snapshot)
    ] == ["Robot-North-H20", "gf1"]


def test_dispatch_order_can_prefer_larger_immediate_allocation() -> None:
    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["timestamp"] = "2026-08-04T05:55:00Z"
    snapshot["resources"].update(
        {
            "local": {"count": 2, "free_count": 2},
            "gf1": {"count": 8, "free_count": 0},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 0,
                "queueing_all_users": [],
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 32,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
        }
    )
    task = {
        "prefer_max_gpus_when_immediate": True,
        "candidates": [
            {"resource": "local", "kind": "local", "gpus": 2},
            {"resource": "gf1", "kind": "ssh", "gpus": 4},
            {"resource": "Robot-East-H20", "kind": "platform", "gpus": 4},
        ],
    }

    ordered = scheduler.ordered_dispatch_candidates(task, snapshot)
    assert [(candidate["resource"], candidate["gpus"]) for candidate in ordered] == [
        ("Robot-East-H20", 4),
        ("local", 2),
        ("gf1", 4),
    ]

    snapshot["resources"]["Robot-East-H20"]["active_gpus_all_users"] = 8
    ordered = scheduler.ordered_dispatch_candidates(task, snapshot)
    assert ordered[0]["resource"] == "local"


def test_dispatch_order_can_prefer_smaller_immediate_allocation() -> None:
    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["timestamp"] = "2026-08-05T11:40:00Z"
    task = {
        "prefer_min_gpus_when_immediate": True,
        "candidates": [
            {"resource": "Robot-North-H20", "kind": "platform", "gpus": 8},
            {"resource": "Robot-North-H20", "kind": "platform", "gpus": 4},
        ],
    }

    ordered = scheduler.ordered_dispatch_candidates(task, snapshot)
    assert [candidate["gpus"] for candidate in ordered] == [4, 8]

    task["prefer_max_gpus_when_immediate"] = True
    with pytest.raises(ValueError, match="cannot prefer both"):
        scheduler.ordered_dispatch_candidates(task, snapshot)


def test_mt3_mixed_gpu_candidates_prefer_8g_only_when_all_cards_are_free() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    task = next(
        task
        for task in queue["tasks"]
        if task["id"] == "pi05_mt3_learned_seed1000_train"
    )
    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["timestamp"] = "2026-08-03T14:45:00Z"
    snapshot["resources"].update(
        {
            "gf1": {"count": 8, "free_count": 8},
            "local": {"count": 2, "free_count": 2},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 0,
                "queueing_all_users": [],
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 0,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
        }
    )

    ordered = scheduler.ordered_dispatch_candidates(task, snapshot)
    assert [(candidate["resource"], candidate["gpus"]) for candidate in ordered] == [
        ("gf1", 8),
        ("gf1", 4),
        ("Robot-East-H20", 4),
    ]

    snapshot["resources"]["gf1"]["free_count"] = 4
    ordered = scheduler.ordered_dispatch_candidates(task, snapshot)
    assert [(candidate["resource"], candidate["gpus"]) for candidate in ordered] == [
        ("gf1", 4),
        ("Robot-East-H20", 4),
        ("gf1", 8),
    ]


def test_north_queue_sink_respects_primary_and_backup_job_limits() -> None:
    candidate = north_candidate(4)
    snapshot = north_snapshot(primary=0, all_users=56, queueing=True)
    assert not scheduler.candidate_available(candidate, snapshot, "primary")
    assert scheduler.north_queue_credential_profile(candidate, snapshot) == "primary"
    scheduler.reserve_queued_north_candidate(snapshot, candidate, "primary")
    assert snapshot["resources"]["beijing"]["owned_submitted_jobs"] == 1
    assert snapshot["resources"]["beijing"]["owned_queued_gpus"] == 4

    beijing = snapshot["resources"]["beijing"]
    beijing["owned_submitted_jobs"] = beijing["max_submitted_jobs"]
    assert scheduler.north_queue_credential_profile(candidate, snapshot) == "backup"
    scheduler.reserve_queued_north_candidate(snapshot, candidate, "backup")
    assert beijing["backup"]["managed_submitted_jobs"] == 1
    assert beijing["backup"]["managed_queued_gpus"] == 4

    beijing["backup"]["managed_submitted_jobs"] = beijing["backup"][
        "max_submitted_jobs"
    ]
    assert scheduler.north_queue_credential_profile(candidate, snapshot) is None


def test_north_queue_sink_spills_projected_gpu_quota_to_backup() -> None:
    snapshot = north_snapshot(primary=4, all_users=56, queueing=True)
    profiles = []
    for gpus in [2, 4, 4, 4, 4]:
        candidate = north_candidate(gpus)
        profile = scheduler.north_queue_credential_profile(candidate, snapshot)
        assert profile is not None
        profiles.append(profile)
        scheduler.reserve_queued_north_candidate(snapshot, candidate, profile)
    assert profiles == ["primary", "primary", "primary", "primary", "backup"]
    beijing = snapshot["resources"]["beijing"]
    assert beijing["owned_active_gpus"] == 4
    assert beijing["owned_queued_gpus"] == 14
    assert beijing["active_gpus_all_users"] == 56
    assert beijing["backup"]["managed_queued_gpus"] == 4


def test_persistent_north_queue_sink_is_not_reclaimed_by_short_timeout() -> None:
    attempt = {
        "started_at": "2026-08-01T00:00:00Z",
        "queue_timeout_seconds": 1,
        "persistent_north_queue_sink": True,
    }
    assert not scheduler.queued_attempt_timed_out(attempt)


def test_robot_task_queueing_is_reclaimed_immediately() -> None:
    attempt = {
        "resource": "robot-task",
        "started_at": "2099-01-01T00:00:00Z",
        "queue_timeout_seconds": 900,
    }
    assert scheduler.queued_attempt_timed_out(attempt)


def test_robot_task_fragmentation_failure_blocks_equivalent_shapes_globally() -> None:
    state = {
        "tasks": {
            "probe": {
                "attempts": [
                    {
                        "resource": "robot-task",
                        "gpus": 4,
                        "active_gpus_at_dispatch": 24,
                        "failure": (
                            "reclaimed after queueing because Shanghai queueing is disabled"
                        ),
                    }
                ]
            }
        }
    }
    snapshot = robot_task_snapshot(active=24)
    assert scheduler.robot_task_fragmentation_blocked(
        {"resource": "robot-task", "gpus": 4}, state, snapshot
    )
    assert scheduler.robot_task_fragmentation_blocked(
        {"resource": "robot-task", "gpus": 8}, state, snapshot
    )
    assert not scheduler.robot_task_fragmentation_blocked(
        {"resource": "robot-task", "gpus": 2}, state, snapshot
    )
    snapshot["resources"]["robot-task"]["active_gpus_all_users"] = 20
    assert not scheduler.robot_task_fragmentation_blocked(
        {"resource": "robot-task", "gpus": 4}, state, snapshot
    )


def test_dispatch_submits_persistent_north_queue_sink_without_gpu_reservation(
    tmp_path, monkeypatch
) -> None:
    task = {
        "id": "north-queue-sink",
        "priority": 1,
        "description": "queue sink test",
        "enabled": True,
        "candidates": [
            {
                "kind": "platform",
                "resource": "Robot-North-H20",
                "region": "cn-beijing",
                "gpus": 4,
            }
        ],
    }
    state = {"tasks": {task["id"]: {"status": "pending", "attempts": []}}}
    snapshot = north_snapshot(primary=0, all_users=56, queueing=True)
    submitted = []
    monkeypatch.setattr(scheduler, "ready", lambda _task: True)
    monkeypatch.setattr(scheduler, "candidate_exhausted", lambda *_args: False)
    monkeypatch.setattr(scheduler, "candidate_in_cooldown", lambda *_args: False)
    monkeypatch.setattr(
        scheduler,
        "capture_submission_recommendation",
        lambda *_args: (
            tmp_path / "recommendation.json",
            {
                "global_recommendation": "Robot-North-H20",
                "task_eligible_recommendation": "Robot-North-H20",
                "selected_resource": "Robot-North-H20",
                "selection_analysis": "North queue sink",
            },
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "submit_platform",
        lambda _candidate, profile: submitted.append(profile) or "job-queued",
    )
    for name in (
        "capture_pi05_confirmatory_launch",
        "capture_pi05_confirmatory_eval_launch",
        "capture_pi05_mt3_eval_launch",
        "capture_pi05_mt12_training_launch",
        "capture_pi05_mt12_eval_launch",
    ):
        monkeypatch.setattr(scheduler, name, lambda *_args: None)
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args: None)

    scheduler.dispatch({"tasks": [task]}, state, snapshot)

    assert submitted == ["primary"]
    assert state["tasks"][task["id"]]["status"] == "running"
    attempt = state["tasks"][task["id"]]["attempts"][-1]
    assert attempt["persistent_north_queue_sink"] is True
    assert attempt["job_id"] == "job-queued"
    beijing = snapshot["resources"]["beijing"]
    assert beijing["owned_submitted_jobs"] == 1
    assert beijing["active_gpus_all_users"] == 56
    assert beijing["owned_active_gpus"] == 0


def test_submission_recommendation_audit_records_locality_and_selection(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(scheduler, "RECOMMENDATION_LOG_DIR", tmp_path)
    snapshot = {
        "timestamp": "2026-08-03T12:00:00Z",
        "resources": {
            "gf1": {"count": 8, "free_count": 4},
            "local": {"count": 2, "free_count": 2},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 8,
                "queueing_all_users": [],
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 0,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
            "beijing": {
                "capacity": 56,
                "active_gpus_all_users": 40,
                "personal_limit": 20,
                "owned_active_gpus": 0,
                "owned_queueing": ["queued"],
                "queueing_all_users": ["queued"],
                "owned_submitted_jobs": 1,
                "max_submitted_jobs": 20,
                "backup": {"enabled": False},
            },
        },
    }
    checkpoint = str(scheduler.REPO / "kai0/checkpoints/example/49999")
    gf1 = {
        "kind": "ssh",
        "resource": "gf1",
        "gpus": 4,
        "command": f"env CKPT={checkpoint} bash run.sh",
    }
    task = {
        "id": "recommendation-test",
        "ready_files": [f"{checkpoint}/_CHECKPOINT_METADATA"],
        "candidates": [
            gf1,
            {"kind": "platform", "resource": "robot-task", "gpus": 4},
        ],
    }

    path, payload = scheduler.capture_submission_recommendation(task, gf1, snapshot)

    assert path.is_file()
    assert payload["data_locations"] == ["east_shared"]
    assert payload["global_recommendation"] == "gf1"
    assert payload["task_eligible_recommendation"] == "gf1"
    assert payload["selected_resource"] == "gf1"
    assert payload["selected_matches_task_recommendation"] is True
    assert json.loads(path.read_text()) == payload


def test_dispatch_rechecks_completion_after_recommendation(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "canonical.ok"
    task = {
        "id": "completion-race",
        "priority": 0,
        "description": "completion race test",
        "completion_locations": [
            {"label": "canonical", "glob": str(marker), "remote": False}
        ],
        "completion_min_count": 1,
        "candidates": [
            {
                "kind": "local",
                "resource": "local",
                "gpus": 1,
                "gpu_indices": [0],
                "status_dir": str(tmp_path / "status"),
                "command": "true",
            }
        ],
    }
    state = {"tasks": {task["id"]: {"status": "pending", "attempts": []}}}
    snapshot = {
        "resources": {
            "local": {
                "available": True,
                "count": 1,
                "free_count": 1,
                "managed_reserved_indices": [],
                "gpus": [{"index": 0, "memory_used_mib": 0}],
            }
        }
    }
    launched = []

    monkeypatch.setattr(scheduler, "ready", lambda _task: True)
    monkeypatch.setattr(scheduler, "candidate_exhausted", lambda *_args: False)
    monkeypatch.setattr(scheduler, "candidate_in_cooldown", lambda *_args: False)

    def recommend(*_args):
        marker.write_text("complete\n")
        return (
            tmp_path / "recommendation.json",
            {
                "global_recommendation": "local",
                "task_eligible_recommendation": "local",
                "selected_resource": "local",
                "selection_analysis": "test",
            },
        )

    monkeypatch.setattr(scheduler, "capture_submission_recommendation", recommend)
    monkeypatch.setattr(
        scheduler,
        "launch_local",
        lambda candidate: launched.append(candidate) or "123",
    )
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args: None)

    scheduler.dispatch({"tasks": [task]}, state, snapshot)

    assert launched == []
    assert state["tasks"][task["id"]]["status"] == "completed"
    assert state["tasks"][task["id"]]["artifacts_complete"] is True


def test_submission_recommendation_failure_prevents_launch(monkeypatch) -> None:
    task = {
        "id": "recommendation-required",
        "priority": 1,
        "description": "test",
        "enabled": True,
        "candidates": [
            {
                "kind": "local",
                "resource": "local",
                "gpus": 1,
                "status_dir": "/tmp/recommendation-required",
                "command": "true",
            }
        ],
    }
    state = {"tasks": {task["id"]: {"status": "pending", "attempts": []}}}
    launched = []
    monkeypatch.setattr(scheduler, "ready", lambda _task: True)
    monkeypatch.setattr(scheduler, "candidate_available", lambda *_args: True)
    monkeypatch.setattr(scheduler, "candidate_exhausted", lambda *_args: False)
    monkeypatch.setattr(scheduler, "candidate_in_cooldown", lambda *_args: False)
    monkeypatch.setattr(
        scheduler,
        "capture_submission_recommendation",
        lambda *_args: (_ for _ in ()).throw(ValueError("router unavailable")),
    )
    monkeypatch.setattr(
        scheduler, "launch_local", lambda _candidate: launched.append(True)
    )
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args: None)

    scheduler.dispatch({"tasks": [task]}, state, {"resources": {}})

    assert launched == []
    attempt = state["tasks"][task["id"]]["attempts"][0]
    assert "router unavailable" in attempt["failure"]


def test_dispatch_reconciles_completion_without_attempt_history(
    tmp_path, monkeypatch
) -> None:
    marker = tmp_path / "already-complete.ok"
    marker.touch()
    task = {
        "id": "recovered-completion",
        "priority": 1,
        "description": "test",
        "enabled": True,
        "completion_glob": str(marker),
        "ready_files": [str(tmp_path / "missing-input")],
        "candidates": [],
    }
    state = {"tasks": {task["id"]: {"status": "pending", "attempts": []}}}
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args: None)

    scheduler.dispatch({"tasks": [task]}, state, {"resources": {}})

    recovered = state["tasks"][task["id"]]
    assert recovered["status"] == "completed"
    assert recovered["artifacts_complete"] is True
    assert "local=1/1" in recovered["artifact_progress"]
    assert "waiting_reason" not in recovered


def test_robot_task_queue_reclaim_retries_after_cooldown_from_zero_usage() -> None:
    candidate = {
        "kind": "platform",
        "resource": "robot-task",
        "gpus": 8,
        "retry_cooldown_seconds": 300,
    }
    task_state = {
        "attempts": [
            {
                "resource": "robot-task",
                "credential_profile": "primary",
                "active_gpus_at_dispatch": 0,
                "failure": "reclaimed after queueing for more than 120 seconds",
                "finished_at": "2020-01-01T00:00:00Z",
            }
        ]
    }
    snapshot = {"resources": {"robot-task": {"active_gpus_all_users": 0}}}

    assert not scheduler.candidate_in_cooldown(task_state, candidate, snapshot)


def test_robot_task_queue_reclaim_retries_when_usage_drops() -> None:
    candidate = {
        "kind": "platform",
        "resource": "robot-task",
        "gpus": 8,
        "retry_cooldown_seconds": 300,
    }
    task_state = {
        "attempts": [
            {
                "resource": "robot-task",
                "credential_profile": "primary",
                "active_gpus_at_dispatch": 8,
                "failure": "reclaimed after queueing for more than 120 seconds",
                "finished_at": scheduler.utc_now(),
            }
        ]
    }
    snapshot = {"resources": {"robot-task": {"active_gpus_all_users": 4}}}

    assert not scheduler.candidate_in_cooldown(task_state, candidate, snapshot)


def test_gf1_launcher_requires_three_explicit_dead_probes(monkeypatch) -> None:
    task = {"id": "gf1-task"}
    state = {
        "status": "running",
        "waiting_reason": "stale pre-dispatch state",
        "attempts": [
            {
                "kind": "ssh",
                "pid": "123",
                "status_dir": "/tmp/gf1-task",
            }
        ],
    }
    probe = {"result": "error"}

    def fake_ssh(_host, command, timeout=60):
        if command.startswith("cat "):
            return "RUNNING start=now host=gf1"
        if probe["result"] == "error":
            raise RuntimeError("transient connection failure")
        return probe["result"]

    monkeypatch.setattr(scheduler, "ssh", fake_ssh)
    monkeypatch.setattr(scheduler, "log", lambda _message: None)

    scheduler.check_managed_task(task, state)
    assert state["status"] == "running"
    assert "waiting_reason" not in state
    assert "probe unavailable" in state["attempts"][0]["monitor_error"]

    probe["result"] = "DEAD"
    for expected in (1, 2):
        scheduler.check_managed_task(task, state)
        assert state["status"] == "running"
        assert state["attempts"][0]["launcher_dead_confirmations"] == expected
    scheduler.check_managed_task(task, state)
    assert state["status"] == "pending"
    assert "three consecutive polls" in state["attempts"][0]["failure"]


def test_gf1_alive_probe_clears_dead_confirmation(monkeypatch) -> None:
    task = {"id": "gf1-task"}
    attempt = {
        "kind": "ssh",
        "pid": "123",
        "status_dir": "/tmp/gf1-task",
        "launcher_dead_confirmations": 2,
    }
    state = {"status": "running", "attempts": [attempt]}

    def fake_ssh(_host, command, timeout=60):
        return "RUNNING start=now host=gf1" if command.startswith("cat ") else "ALIVE"

    monkeypatch.setattr(scheduler, "ssh", fake_ssh)
    scheduler.check_managed_task(task, state)
    assert state["status"] == "running"
    assert "launcher_dead_confirmations" not in attempt


def test_gf1_completion_artifact_does_not_release_gpus_before_launcher_exit(
    tmp_path, monkeypatch
) -> None:
    sentinel = tmp_path / "_CHECKPOINT_METADATA"
    sentinel.write_text("committed\n")
    task = {
        "id": "pi05_mt1_oracle_seed1001_train",
        "completion_glob": str(sentinel),
        "completion_min_count": 1,
    }
    attempt = {
        "kind": "ssh",
        "pid": "123",
        "status_dir": "/tmp/gf1-task",
    }
    state = {"status": "running", "attempts": [attempt]}
    status = {"value": "RUNNING start=now host=gf1"}

    def fake_ssh(_host, command, timeout=60):
        if command.startswith("cat "):
            return status["value"]
        return "ALIVE"

    monkeypatch.setattr(scheduler, "ssh", fake_ssh)
    scheduler.check_managed_task(task, state)
    assert state["status"] == "running"

    status["value"] = "FINISHED rc=0 start=now end=now host=gf1"
    scheduler.check_managed_task(task, state)
    assert state["status"] == "completed"


def test_mt1_final_evals_have_ordered_north_fallback_and_remote_evidence() -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.validate_queue(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}
    for suffix in ("correct", "null", "within_task", "cross_task"):
        task = tasks[f"pi05_mt1_oracle_seed1000_{suffix}_eval"]
        resources = [candidate["resource"] for candidate in task["candidates"]]
        assert resources == [
            "gf1",
            "Robot-East-H20",
            "Robot-North-H20",
            "robot-task",
        ]
        north = next(
            candidate
            for candidate in task["candidates"]
            if candidate["resource"] == "Robot-North-H20"
        )
        robot_task = task["candidates"][-1]
        assert robot_task["gpus"] == 4
        assert robot_task["yaml"].endswith("pi05_mt_transition_eval_cnsh_4a100.yaml")
        assert any(
            path.endswith("/49999/params/_METADATA")
            for path in north["ready_files_remote"]
        )
        assert any(
            path.endswith("/49999/_CHECKPOINT_METADATA")
            for path in north["ready_files_remote"]
        )
        assert not any(
            path.endswith("/49999/train_state/_METADATA")
            for path in north["ready_files_remote"]
        )
        assert any(
            path.endswith("/49999/assets/robotwin2.0_absolute_meanstd/norm_stats.json")
            for path in north["ready_files_remote"]
        )
        assert any(
            path.endswith("/49999/_CHECKPOINT_METADATA") for path in task["ready_files"]
        )
        assert any(
            path.endswith("/49999/train_state/_METADATA")
            for path in task["ready_files"]
        )
        assert any(
            path.endswith("/49999/assets/robotwin2.0_absolute_meanstd/norm_stats.json")
            for path in task["ready_files"]
        )
        assert any(location.get("remote") for location in task["completion_locations"])


def test_north_materialization_is_only_required_for_completed_north_parent() -> None:
    assert scheduler.north_materialization_required({"status": "pending"}) is None
    assert (
        scheduler.north_materialization_required(
            {"status": "completed", "attempts": [{"resource": "gf1"}]}
        )
        is False
    )
    assert (
        scheduler.north_materialization_required(
            {"status": "completed", "attempts": [{"resource": "Robot-North-H20"}]}
        )
        is True
    )


def test_dispatch_does_not_materialize_pending_north_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {"id": "parent", "priority": 0, "enabled": False}
    materialize = {
        "id": "materialize",
        "priority": 0,
        "enabled": True,
        "materialize_north_result_for": "parent",
        "completion_glob": "/tmp/failed-parent-report.json",
        "candidates": [{"kind": "local", "resource": "local", "gpus": 0}],
    }
    state = {
        "tasks": {
            "parent": {"status": "pending", "attempts": []},
            "materialize": {"status": "pending", "attempts": []},
        }
    }
    monkeypatch.setattr(
        scheduler,
        "ordered_dispatch_candidates",
        lambda *_args: pytest.fail("pending materializer reached candidate selection"),
    )

    scheduler.dispatch({"tasks": [parent, materialize]}, state, {"resources": {}})

    materialize_state = state["tasks"]["materialize"]
    assert materialize_state["status"] == "pending"
    assert materialize_state["attempts"] == []
    assert materialize_state["waiting_reason"] == (
        "waiting for North parent task to complete: parent"
    )


def test_north_parent_completion_rearms_only_precompletion_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = {"id": "parent", "priority": 0, "enabled": False}
    materialize = {
        "id": "materialize",
        "priority": 0,
        "enabled": True,
        "materialize_north_result_for": "parent",
        "completion_glob": "/tmp/not-created-materializer-report.json",
        "candidates": [{"kind": "local", "resource": "local", "gpus": 0}],
    }
    state = {
        "tasks": {
            "parent": {
                "status": "completed",
                "completed_at": "2026-08-05T12:00:00Z",
                "attempts": [{"resource": "Robot-North-H20"}],
            },
            "materialize": {
                "status": "pending",
                "attempts": [
                    {
                        "resource": "local",
                        "failure": "old failure",
                        "finished_at": "2026-08-05T02:00:00Z",
                    }
                ],
                "exhausted_resources": {"local": {"failures": 3, "limit": 3}},
            },
        }
    }
    monkeypatch.setattr(scheduler, "ordered_dispatch_candidates", lambda *_args: [])

    scheduler.dispatch({"tasks": [parent, materialize]}, state, {"resources": {}})
    materialize_state = state["tasks"]["materialize"]
    assert "exhausted_resources" not in materialize_state
    assert materialize_state["rearmed_after_parent_completion"]
    assert materialize_state["ignore_failures_before"] == "2026-08-05T12:00:00Z"
    assert scheduler.candidate_failure_count(
        materialize_state, materialize["candidates"][0]
    ) == 0

    materialize_state["attempts"].append(
        {
            "resource": "local",
            "failure": "new failure",
            "finished_at": "2026-08-05T12:01:00Z",
        }
    )
    scheduler.dispatch({"tasks": [parent, materialize]}, state, {"resources": {}})
    assert scheduler.candidate_failure_count(
        materialize_state, materialize["candidates"][0]
    ) == 1


def test_mt1_north_materialization_tasks_match_parent_outputs() -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    tasks = {task["id"]: task for task in json.loads(queue_path.read_text())["tasks"]}
    for suffix, intervention in (
        ("correct", "correct"),
        ("null", "null"),
        ("within_task", "within-task"),
        ("cross_task", "cross-task"),
    ):
        parent_id = f"pi05_mt1_oracle_seed1000_{suffix}_eval"
        parent = tasks[parent_id]
        sync = tasks[f"pi05_mt1_{suffix}_sync_from_north"]
        assert sync["materialize_north_result_for"] == parent_id
        remote = next(
            location
            for location in parent["completion_locations"]
            if location["remote"]
        )
        assert sync["ready_files_remote"] == [remote["glob"]]
        parent_result = re.search(
            r"(?:^| )RESULT_NAME=([^ ]+)", parent["candidates"][0]["command"]
        )
        assert parent_result is not None
        command = sync["candidates"][0]["command"]
        assert f"RESULT_NAME={parent_result.group(1)}" in command
        assert f"INTERVENTION={intervention}" in command


def test_mt3_tracker_pipeline_is_fully_staged_behind_protocol_gate() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    assert len(queue["tasks"]) == 19
    tasks = {task["id"]: task for task in queue["tasks"]}
    protocol = tasks["pi05_mt3_protocol_validate"]
    assert protocol["candidates"][0]["gpus"] == 0
    assert protocol["candidates"][0]["resource"] == "local"
    assert any(
        path.endswith("pi05_mt1_three_seed_gate.ok") for path in protocol["ready_files"]
    )
    assert any(
        path.endswith("RESULTS_pi05_mt1_three_seed.json")
        for path in protocol["ready_files"]
    )
    assert protocol["completion_glob"].endswith("pi05_mt3_protocol.ok")
    assert "validate_mt3_protocol.py" in protocol["candidates"][0]["command"]
    for index in range(8):
        task = tasks[f"pi05_mt3_feature_shard{index}_of8"]
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "gf1",
            "Robot-East-H20",
        ]
        assert any(
            path.endswith("pi05_mt3_protocol.ok") for path in task["ready_files"]
        )
        assert task["completion_glob"].endswith(
            f"shard-{index:02d}-of-08/manifest.json"
        )
    assert set(tasks) >= {
        "pi05_mt3_tracker_current_frame_train",
        "pi05_mt3_tracker_history_proprio_train",
        "pi05_mt3_tracker_current_frame_metrics",
        "pi05_mt3_tracker_history_proprio_metrics",
        "pi05_mt3_tracker_select",
        "pi05_mt3_learned_seed1000_train",
    }
    policy = tasks["pi05_mt3_learned_seed1000_train"]
    assert [candidate["resource"] for candidate in policy["candidates"]] == [
        "gf1",
        "gf1",
        "Robot-East-H20",
    ]
    assert policy["candidates"][0]["gpus"] == 8
    assert policy["candidates"][0]["gpu_indices"] == list(range(8))
    assert "WORKERS=16" in policy["candidates"][0]["command"]
    assert policy["candidates"][1]["gpus"] == 4
    assert "WORKERS=8" in policy["candidates"][1]["command"]
    assert any(
        path.endswith("pi05_mt3_tracker_selected.ok") for path in policy["ready_files"]
    )
    for intervention in ("predicted", "within_task", "null", "oracle"):
        task = tasks[f"pi05_mt3_learned_seed1000_{intervention}_eval"]
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "gf1",
            "Robot-East-H20",
        ]
        assert any(
            path.endswith("pi05_mt3_tracker_selected.ok")
            for path in task["ready_files"]
        )
        assert any(
            path.endswith("49999/params/_METADATA") for path in task["ready_files"]
        )
        assert len(task["produces_files"]) == 1
        assert task["produces_files"][0].endswith(".json")


def test_mt4_replication_is_staged_behind_seed1000_pilot_gate() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt4_replication_tasks(queue)
    assert len(queue["tasks"]) == 13
    tasks = {task["id"]: task for task in queue["tasks"]}

    pilot = tasks["pi05_mt3_seed1000_control_analysis"]
    assert pilot["candidates"][0]["resource"] == "local"
    assert pilot["candidates"][0]["gpus"] == 0
    assert any(path.endswith("seed1000_oracle.json") for path in pilot["ready_files"])
    assert any(
        path.endswith("logs/eval_reports/pi05_rt_a0_public_exact_seed1000.json")
        for path in pilot["ready_files"]
    )

    for seed in (1001, 1002):
        train = tasks[f"pi05_mt3_learned_seed{seed}_train"]
        evaluations = [
            tasks[f"pi05_mt3_learned_seed{seed}_{intervention}_eval"]
            for intervention in ("predicted", "null", "within_task")
        ]
        for task in (train, *evaluations):
            assert any(
                path.endswith("pi05_mt3_seed1000_replication_gate.ok")
                for path in task["ready_files"]
            )
            assert any(
                path.endswith("RESULTS_pi05_mt3_seed1000_gate.json")
                for path in task["ready_files"]
            )
            assert [candidate["resource"] for candidate in task["candidates"]] == [
                "gf1",
                "Robot-East-H20",
            ]
        for task in evaluations:
            assert len(task["produces_files"]) == 1
            assert task["produces_files"][0].endswith(".json")
        assert train["candidates"][1]["env"]["PILOT_GATE"].endswith(
            "pi05_mt3_seed1000_replication_gate.ok"
        )

    final = tasks["pi05_mt3_three_seed_analysis"]
    assert final["candidates"][0]["resource"] == "local"
    assert len(final["ready_files"]) == 9
    for seed in (1000, 1001, 1002):
        assert any(
            path.endswith(f"logs/eval_reports/pi05_rt_a0_public_exact_seed{seed}.json")
            for path in final["ready_files"]
        )
    for control in ("null", "within_task"):
        analysis = tasks[f"pi05_mt3_three_seed_vs_{control}_analysis"]
        assert len(analysis["ready_files"]) == 9
        assert analysis["candidates"][0]["gpus"] == 0
    gate = tasks["pi05_mt4_three_seed_content_gate"]
    assert len(gate["ready_files"]) == 6
    assert sum(path.endswith(".ok") for path in gate["ready_files"]) == 3
    assert sum(path.endswith(".json") for path in gate["ready_files"]) == 3
    assert gate["completion_glob"].endswith("RESULTS_pi05_mt4_content_gate.json")
    assert gate["produces_files"][0].endswith("pi05_mt4_three_seed_content.ok")


def test_mt6_scope_analysis_is_staged_behind_confirmed_content_gate() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt6_scope_task(queue)
    assert len(queue["tasks"]) == 1
    task = queue["tasks"][0]
    assert task["id"] == "pi05_mt6_scope_analysis"
    assert task["priority"] == 11
    assert task["candidates"][0]["resource"] == "local"
    assert task["candidates"][0]["gpus"] == 0
    assert len(task["ready_files"]) == 13
    assert any(
        path.endswith("pi05_mt4_three_seed_content.ok") for path in task["ready_files"]
    )
    assert any(
        path.endswith("RESULTS_pi05_mt4_content_gate.json")
        for path in task["ready_files"]
    )
    assert any(
        path.endswith("robotwin_mt6_scope_v1.json") for path in task["ready_files"]
    )
    assert task["completion_glob"].endswith("RESULTS_pi05_mt6_scope.json")


def test_mt5_frozen_two_by_two_is_fully_staged_behind_content_gate() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt5_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}
    assert len(tasks) == 13
    for arm in ("local", "combined"):
        for seed in (1000, 1001, 1002):
            train = tasks[f"pi05_mt5_{arm}_seed{seed}_train"]
            evaluate = tasks[f"pi05_mt5_{arm}_seed{seed}_eval"]
            assert [candidate["resource"] for candidate in train["candidates"]] == [
                "gf1",
                "Robot-East-H20",
                "robot-task",
            ]
            assert [candidate["resource"] for candidate in evaluate["candidates"]] == [
                "gf1",
                "Robot-East-H20",
                "robot-task",
            ]
            for task in (train, evaluate):
                assert any(
                    path.endswith("pi05_mt4_three_seed_content.ok")
                    for path in task["ready_files"]
                )
                assert any(
                    path.endswith("RESULTS_pi05_mt4_content_gate.json")
                    for path in task["ready_files"]
                )
    analysis = tasks["pi05_mt5_complementarity_analysis"]
    assert analysis["completion_glob"].endswith("RESULTS_pi05_mt5_three_seed.json")
    assert len(analysis["ready_files"]) == 16
    assert analysis["produces_files"][0].endswith("pi05_mt5_complementarity.ok")


def test_mt6_efficiency_is_gated_and_uses_resource_priority() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt6_efficiency_task(queue)
    assert len(queue["tasks"]) == 1
    task = queue["tasks"][0]
    assert task["id"] == "pi05_mt6_selected_efficiency"
    assert [candidate["resource"] for candidate in task["candidates"]] == [
        "gf1",
        "Robot-East-H20",
        "local",
    ]
    assert any(
        path.endswith("pi05_mt4_three_seed_content.ok") for path in task["ready_files"]
    )
    assert any(
        path.endswith("RESULTS_pi05_mt4_content_gate.json")
        for path in task["ready_files"]
    )
    assert any(
        path.endswith("pi05_mt3_tracker_selected.ok") for path in task["ready_files"]
    )
    assert task["completion_glob"].endswith("pi05_mt6_selected.json")


def test_mt6_train_memory_is_gated_and_matched_on_four_a100s() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt6_train_memory_task(queue)
    assert len(queue["tasks"]) == 1
    task = queue["tasks"][0]
    assert task["id"] == "pi05_mt6_selected_train_memory"
    assert [candidate["resource"] for candidate in task["candidates"]] == [
        "gf1",
        "robot-task",
    ]
    assert all(candidate["gpus"] == 4 for candidate in task["candidates"])
    assert len(task["ready_any"]) == 2
    assert all(
        alternative["ready_files"][0].endswith("tracker.pt")
        for alternative in task["ready_any"]
    )
    assert any(
        path.endswith("pi05_mt4_three_seed_content.ok") for path in task["ready_files"]
    )
    assert any(
        path.endswith("RESULTS_pi05_mt4_content_gate.json")
        for path in task["ready_files"]
    )
    assert task["completion_glob"].endswith("pi05_mt6_train_memory_selected.json")


def test_mt_gate_consumer_requires_auditable_decision_json() -> None:
    marker = str(
        scheduler.REPO / "logs/resource_markers/pi05_mt1_seed1000_replication_gate.ok"
    )
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_bad_gate_consumer",
                "ready_files": [marker],
                "candidates": [],
            }
        ]
    }
    with pytest.raises(ValueError, match="without gate decision"):
        scheduler.validate_queue(queue)


def test_gate_producer_must_complete_on_final_decision_output() -> None:
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_bad_gate_producer",
                "completion_glob": "/tmp/intermediate.json",
                "candidates": [
                    {
                        "resource": "local",
                        "command": (
                            "python decide.py --analysis /tmp/intermediate.json "
                            "--output /tmp/decision.json "
                            "--accepted-marker /tmp/accepted.ok"
                        ),
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="complete on final decision output"):
        scheduler.validate_queue(queue)


def test_static_mt_gate_consumers_include_decision_json() -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.validate_queue(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}
    expected = {
        "pi05_mt1_oracle_seed1001_train": "RESULTS_pi05_mt1_seed1000_gate.json",
        "pi05_mt1_oracle_seed1002_train": "RESULTS_pi05_mt1_seed1000_gate.json",
        "pi05_mt1_three_seed_analysis": "RESULTS_pi05_mt1_seed1000_gate.json",
        "pi05_mt3_protocol_gate": "RESULTS_pi05_mt1_three_seed.json",
    }
    for task_id, decision_name in expected.items():
        assert any(
            path.endswith(decision_name) for path in tasks[task_id]["ready_files"]
        )


def test_negative_gate_closes_transitive_artifact_dependencies(
    tmp_path: Path, monkeypatch
) -> None:
    marker = str(tmp_path / "pilot.ok")
    decision = tmp_path / "decision.json"
    decision.write_text('{"accepted": false}\n')
    monkeypatch.setattr(
        scheduler,
        "GATE_DECISION_SPECS",
        {marker: (decision, ("accepted",))},
    )
    stage = str(tmp_path / "stage.ok")
    child = str(tmp_path / "child.ok")
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_stage",
                "ready_files": [marker, str(decision)],
                "completion_glob": stage,
            },
            {
                "id": "pi05_mt_child",
                "ready_files": [stage],
                "completion_glob": child,
            },
            {
                "id": "pi05_mt_grandchild",
                "ready_files": [child],
                "completion_glob": str(tmp_path / "grandchild.ok"),
            },
        ]
    }
    closed = scheduler.gate_rejection_closure(queue)
    assert set(closed) == {
        "pi05_mt_stage",
        "pi05_mt_child",
        "pi05_mt_grandchild",
    }
    assert all(reason == str(decision) for reason in closed.values())


def test_accepted_gate_does_not_close_branch(tmp_path: Path, monkeypatch) -> None:
    marker = str(tmp_path / "pilot.ok")
    decision = tmp_path / "decision.json"
    decision.write_text('{"accepted": true}\n')
    monkeypatch.setattr(
        scheduler,
        "GATE_DECISION_SPECS",
        {marker: (decision, ("accepted",))},
    )
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_stage",
                "ready_files": [marker, str(decision)],
                "completion_glob": str(tmp_path / "stage.ok"),
            }
        ]
    }
    assert scheduler.gate_rejection_closure(queue) == {}


def test_mt1_pilot_rejection_closes_full_mt3_mt4_branch(
    tmp_path: Path, monkeypatch
) -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    scheduler.add_pi05_mt4_replication_tasks(queue)
    scheduler.add_pi05_mt5_tasks(queue)
    scheduler.add_pi05_mt6_scope_task(queue)
    scheduler.add_pi05_mt6_efficiency_task(queue)
    scheduler.add_pi05_mt6_train_memory_task(queue)
    scheduler.add_pi05_mt3_eval_attach_tasks(queue)
    marker = str(
        scheduler.REPO / "logs/resource_markers/pi05_mt1_seed1000_replication_gate.ok"
    )
    decision = tmp_path / "decision.json"
    decision.write_text('{"accepted": false}\n')
    monkeypatch.setattr(
        scheduler,
        "GATE_DECISION_SPECS",
        {marker: (decision, ("accepted",))},
    )
    closed = scheduler.gate_rejection_closure(queue)
    assert {
        "pi05_mt1_oracle_seed1001_train",
        "pi05_mt1_oracle_seed1002_train",
        "pi05_mt1_three_seed_analysis",
        "pi05_mt3_protocol_gate",
        "pi05_mt3_seed1000_control_analysis",
        "pi05_mt3_three_seed_analysis",
        "pi05_mt4_three_seed_content_gate",
        "pi05_mt5_local_seed1000_train",
        "pi05_mt5_combined_seed1002_eval",
        "pi05_mt5_complementarity_analysis",
        "pi05_mt6_scope_analysis",
        "pi05_mt6_selected_efficiency",
        "pi05_mt6_selected_train_memory",
    } <= set(closed)


def test_load_state_persists_and_clears_gate_disabled_status(
    tmp_path: Path, monkeypatch
) -> None:
    marker = str(tmp_path / "pilot.ok")
    decision = tmp_path / "decision.json"
    state_path = tmp_path / "state.json"
    decision.write_text('{"accepted": false}\n')
    monkeypatch.setattr(scheduler, "STATE_PATH", state_path)
    monkeypatch.setattr(
        scheduler,
        "GATE_DECISION_SPECS",
        {marker: (decision, ("accepted",))},
    )
    stage = str(tmp_path / "stage.ok")
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_stage",
                "ready_files": [marker, str(decision)],
                "completion_glob": stage,
            },
            {
                "id": "pi05_mt_child",
                "ready_files": [stage],
                "completion_glob": str(tmp_path / "child.ok"),
            },
            {
                "id": "pi05_mt_independent",
                "ready_files": [],
                "completion_glob": str(tmp_path / "independent.ok"),
            },
        ]
    }
    rejected = scheduler.load_state(queue)
    for task_id in ("pi05_mt_stage", "pi05_mt_child"):
        assert rejected["tasks"][task_id]["status"] == "disabled"
        assert str(decision) in rejected["tasks"][task_id]["disabled_reason"]
    assert rejected["tasks"]["pi05_mt_independent"]["status"] == "pending"

    state_path.write_text(json.dumps(rejected))
    decision.write_text('{"accepted": true}\n')
    accepted = scheduler.load_state(queue)
    for task_id in ("pi05_mt_stage", "pi05_mt_child"):
        assert accepted["tasks"][task_id]["status"] == "pending"
        assert "disabled_reason" not in accepted["tasks"][task_id]


def test_load_state_preserves_configured_disabled_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler, "STATE_PATH", tmp_path / "state.json")
    queue = {
        "tasks": [
            {
                "id": "closed_branch",
                "enabled": False,
                "disabled_reason": "scientific screen rejected",
            }
        ]
    }

    state = scheduler.load_state(queue)

    assert state["tasks"]["closed_branch"]["status"] == "disabled"
    assert (
        state["tasks"]["closed_branch"]["disabled_reason"]
        == "scientific screen rejected"
    )


@pytest.mark.parametrize(
    ("marker_name", "expected_closed", "expected_open"),
    [
        (
            "pi05_mt1_three_seed_gate.ok",
            {
                "pi05_mt3_protocol_gate",
                "pi05_mt3_learned_seed1000_train",
                "pi05_mt4_three_seed_content_gate",
                "pi05_mt6_scope_analysis",
                "pi05_mt6_selected_efficiency",
                "pi05_mt6_selected_train_memory",
            },
            {"pi05_mt1_oracle_seed1001_train"},
        ),
        (
            "pi05_mt3_seed1000_replication_gate.ok",
            {
                "pi05_mt3_learned_seed1001_train",
                "pi05_mt3_three_seed_analysis",
                "pi05_mt4_three_seed_content_gate",
                "pi05_mt6_scope_analysis",
                "pi05_mt6_selected_efficiency",
                "pi05_mt6_selected_train_memory",
            },
            {"pi05_mt3_learned_seed1000_train"},
        ),
        (
            "pi05_mt3_three_seed_beats_null.ok",
            {
                "pi05_mt4_three_seed_content_gate",
                "pi05_mt6_scope_analysis",
                "pi05_mt6_selected_efficiency",
                "pi05_mt6_selected_train_memory",
            },
            {"pi05_mt3_three_seed_vs_null_analysis"},
        ),
        (
            "pi05_mt4_three_seed_content.ok",
            {
                "pi05_mt5_local_seed1000_train",
                "pi05_mt5_combined_seed1002_eval",
                "pi05_mt5_complementarity_analysis",
                "pi05_mt6_scope_analysis",
                "pi05_mt6_selected_efficiency",
                "pi05_mt6_selected_train_memory",
            },
            {"pi05_mt4_three_seed_content_gate"},
        ),
    ],
)
def test_downstream_gate_rejection_closes_only_its_branch(
    tmp_path: Path,
    monkeypatch,
    marker_name: str,
    expected_closed: set[str],
    expected_open: set[str],
) -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    scheduler.add_pi05_mt4_replication_tasks(queue)
    scheduler.add_pi05_mt5_tasks(queue)
    scheduler.add_pi05_mt6_scope_task(queue)
    scheduler.add_pi05_mt6_efficiency_task(queue)
    scheduler.add_pi05_mt6_train_memory_task(queue)
    scheduler.add_pi05_mt3_eval_attach_tasks(queue)
    marker = str(scheduler.REPO / "logs/resource_markers" / marker_name)
    decision = tmp_path / "decision.json"
    decision.write_text('{"accepted": false}\n')
    monkeypatch.setattr(
        scheduler,
        "GATE_DECISION_SPECS",
        {marker: (decision, ("accepted",))},
    )
    closed = scheduler.gate_rejection_closure(queue)
    assert expected_closed <= set(closed)
    assert not (expected_open & set(closed))


def test_mt1_control_analysis_uses_authoritative_a0_report() -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.validate_queue(queue)
    task = next(
        task
        for task in queue["tasks"]
        if task["id"] == "pi05_mt1_seed1000_control_analysis"
    )
    authoritative = str(
        scheduler.REPO / "logs/eval_reports/pi05_rt_a0_public_exact_seed1000.json"
    )
    assert task["completion_glob"].endswith("RESULTS_pi05_mt1_seed1000_gate.json")
    assert task["produces_files"] == [
        str(
            scheduler.REPO
            / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_seed1000_controls.json"
        )
    ]
    assert authoritative in task["ready_files"]
    command = task["candidates"][0]["command"]
    assert f"--control a0={authoritative}" in command
    assert "lmvla/lmwm/docs/pi05_rt_a0_public_exact" not in command


def test_mt_queue_validation_rejects_resource_missing_from_router() -> None:
    queue = {
        "tasks": [
            {
                "id": "pi05_mt_future_task",
                "candidates": [
                    {"resource": "unregistered-cluster", "gpus": 1},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="resource absent from submission router"):
        scheduler.validate_queue(queue)


def test_shared_eval_cell_state_requires_four_complete_schedulers(
    tmp_path: Path,
) -> None:
    for seed in range(4):
        run_dir = tmp_path / f"seed{seed}" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / ".task_scheduler.json").write_text(
            json.dumps(
                {
                    "completed": {f"task{index}": {} for index in range(6)},
                    "in_progress": {},
                    "pending": [],
                    "failed": {},
                }
            )
        )
    assert scheduler.shared_eval_cell_state(tmp_path) == {
        "schedulers": 4,
        "completed": 24,
        "in_progress": 0,
        "pending": 0,
        "failed": 0,
    }


def test_shared_finalizer_publishes_report_and_marker(
    tmp_path: Path, monkeypatch
) -> None:
    result_name = "pi05_mt1_oracle_seed1001_correct"
    root = tmp_path / "lmvla/lawam/results/eval_runs/robotwin" / result_name
    for seed in range(4):
        run_dir = root / f"seed{seed}" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / ".task_scheduler.json").write_text(
            json.dumps(
                {
                    "completed": {f"task{index}": {} for index in range(6)},
                    "in_progress": {},
                    "pending": [],
                    "failed": {},
                }
            )
        )
    calls = []

    def fake_run(command, *, timeout=60, env_overrides=None):
        calls.append((command, timeout, env_overrides))
        return (
            '{"accepted": true}\n' if "summarize_robotwin_eval.py" in command[1] else ""
        )

    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(scheduler, "LOG_PATH", tmp_path / "scheduler-test.log")
    monkeypatch.setattr(
        scheduler,
        "PI05_MT12_SHARED_FINALIZERS",
        {"pi05_mt1_oracle_seed1001_correct_eval": (result_name, "correct")},
    )
    monkeypatch.setattr(scheduler, "run", fake_run)
    scheduler.refresh_pi05_mt12_shared_finalizers()

    report = tmp_path / "lmvla/lmwm/docs" / f"{result_name}.json"
    marker = tmp_path / "logs/resource_markers" / f"{result_name}.ok"
    assert report.read_text() == '{"accepted": true}\n'
    assert "intervention=correct" in marker.read_text()
    assert len(calls) == 2
    assert all(timeout == 180 for _command, timeout, _env in calls)


def test_mt_eval_attach_workers_preserve_transition_protocol_and_resource_order() -> (
    None
):
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.add_pi05_mt_eval_attach_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    expected = {
        "pi05_mt1_correct_eval_attach": "correct",
        "pi05_mt1_null_eval_attach": "null",
        "pi05_mt1_within_eval_attach": "within-task",
        "pi05_mt1_cross_eval_attach": "cross-task",
        "pi05_mt2_null_eval_attach": "null",
    }
    for prefix, intervention in expected.items():
        gf1_task = tasks[f"{prefix}_gf1g4"]
        east_task = tasks[f"{prefix}_east4g"]
        platform_task = tasks[f"{prefix}_cnsh4g"]
        local_task = tasks[f"{prefix}_local2g"]
        assert [candidate["resource"] for candidate in gf1_task["candidates"]] == [
            "gf1",
            "gf1",
        ]
        assert [candidate["gpu_indices"] for candidate in gf1_task["candidates"]] == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
        assert "ATTACH_GPU_INDEX_BASE=0" in gf1_task["candidates"][0]["command"]
        assert "ATTACH_GPU_INDEX_BASE=4" in gf1_task["candidates"][1]["command"]
        assert east_task["candidates"][0]["resource"] == "Robot-East-H20"
        assert platform_task["candidates"][0]["resource"] == "robot-task"
        assert local_task["candidates"][0]["resource"] == "local"
        assert local_task["candidates"][0]["gpu_indices"] == [0, 1]
        assert "ATTACH_LOG_DIR=" in local_task["candidates"][0]["command"]
        assert "chmod -R a+rwX" in local_task["candidates"][0]["command"]
        assert [
            gf1_task["priority"],
            east_task["priority"],
            platform_task["priority"],
            local_task["priority"],
        ] == [3, 4, 5, 6]
        for task in (gf1_task, east_task, platform_task, local_task):
            assert (
                len(
                    [
                        path
                        for path in task["ready_files"]
                        if path.endswith(".task_scheduler.json")
                    ]
                )
                == 4
            )
        platform = platform_task["candidates"][0]
        assert platform["env"]["ROBOTWIN_TRANSITION_INTERVENTION"] == intervention
        assert platform["env"]["ROBOTWIN_TRANSITION_ORACLE"] == "1"
        assert platform["env"]["ATTACH_RUN_TAG_PREFIX"] == "local-unseen-a3-seed"
        assert platform["env"]["ATTACH_GPU_COUNT"] == "4"
        assert "ROBOTWIN_TRANSITION_PAIRS=" in local_task["candidates"][0]["command"]
        assert "ROBOTWIN_TRANSITION_PAIRS=" in gf1_task["candidates"][0]["command"]
        assert int(platform["env"]["WORKER_INDEX_BASE"]) != int(
            re.search(
                r"WORKER_INDEX_BASE=([0-9]+)",
                local_task["candidates"][0]["command"],
            ).group(1)
        )
        assert (
            len(
                {
                    task["satisfied_by_task"]
                    for task in (gf1_task, east_task, platform_task, local_task)
                }
            )
            == 1
        )
        parent = tasks[platform_task["satisfied_by_task"]]
        parent_command = parent["candidates"][0]["command"]
        match = re.search(r"(?:^| )RESULT_NAME=([^ ]+)", parent_command)
        assert match is not None
        assert platform["env"]["RESULT_NAME"] == match.group(1)


def test_mt1_replication_eval_attach_fans_out_across_idle_resources() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    assert len(queue["tasks"]) == 18
    tasks = {task["id"]: task for task in queue["tasks"]}

    for seed in (1001, 1002):
        groups = (
            "gf1g4",
            "east4g",
            "cnsh4g",
            "cnsh4g_1",
            "cnsh4g_2",
            "cnsh4g_3",
            "local2g",
        )
        seed_tasks = [
            tasks[f"pi05_mt1_oracle_seed{seed}_correct_eval_attach_{group}"]
            for group in groups
        ]
        assert all(
            task["satisfied_by_task"] == f"pi05_mt1_oracle_seed{seed}_correct_eval"
            for task in seed_tasks
        )
        assert [task["candidates"][0]["resource"] for task in seed_tasks] == [
            "gf1",
            "Robot-East-H20",
            "robot-task",
            "robot-task",
            "robot-task",
            "robot-task",
            "local",
        ]
        assert [task["priority"] for task in seed_tasks] == [3, 4, 5, 5, 5, 5, 6]
        for task in seed_tasks:
            scheduler_files = [
                path
                for path in task["ready_files"]
                if path.endswith(".task_scheduler.json")
            ]
            assert len(scheduler_files) == 4
            assert all(
                f"seed{eval_seed}" in scheduler_files[eval_seed]
                for eval_seed in range(4)
            )
            assert all(f"seed{seed}" in path for path in task["ready_files"][:2])
        gf1 = seed_tasks[0]["candidates"][0]
        assert gf1["gpus"] == 4
        assert gf1["gpu_indices"] == list(
            range((1002 - seed) * 4, (1002 - seed) * 4 + 4)
        )

        assert f"ATTACH_GPU_INDEX_BASE={(1002 - seed) * 4}" in gf1["command"]
        east = seed_tasks[1]["candidates"][0]
        assert east["gpus"] == 4
        assert east["queue_timeout_seconds"] == 900
        assert east["env"]["ROBOTWIN_TRANSITION_INTERVENTION"] == "correct"
        assert east["env"]["ROBOTWIN_TRANSITION_ORACLE"] == "1"
        assert east["env"]["ATTACH_RUN_TAG_PREFIX"] == "local-unseen-a3-seed"
        assert east["env"]["ATTACH_GROUP_NAME"] == "east4g"
        local = seed_tasks[-1]["candidates"][0]
        assert local["gpu_indices"] == [0, 1]
        assert "ATTACH_LOG_DIR=" in local["command"]
        assert "chmod -R a+rwX" in local["command"]
        worker_bases = {
            int(task["candidates"][0].get("env", {}).get("WORKER_INDEX_BASE", 0))
            or int(
                re.search(
                    r"WORKER_INDEX_BASE=([0-9]+)",
                    task["candidates"][0]["command"],
                ).group(1)
            )
            for task in seed_tasks
        }
        assert len(worker_bases) == 7
        # attach_pi05_a0_confirmatory_platform.sh uses PORT_BASE_OFFSET=22200;
        # each group adds at most seed=3 (120) and seed_index=3 (300).
        assert max(worker_bases) + 22200 + 120 + 300 < 65536

        assert min(worker_bases) >= 20000 + (seed - 1001) * 4000
        assert max(worker_bases) <= 22400 + (seed - 1001) * 4000
        assert all(
            task["candidates"][0]["queue_timeout_seconds"] == 900
            for task in seed_tasks
            if task["candidates"][0]["kind"] == "platform"
        )
        assert (
            f"pi05_mt1_oracle_seed{seed}_correct_eval"
            in scheduler.PI05_MT12_SHARED_FINALIZERS
        )
    small_robot_tasks = [
        tasks[f"pi05_mt1_oracle_seed1001_correct_eval_attach_cnsh2g_{shard}"]
        for shard in range(4)
    ]
    assert [task["candidates"][0]["gpus"] for task in small_robot_tasks] == [
        2,
        2,
        2,
        2,
    ]
    assert all(
        task["candidates"][0]["resource"] == "robot-task"
        and task["priority"] == 5
        and task["candidates"][0]["yaml"].endswith(
            "pi05_a0_confirmatory_attach_cnsh_2a100.yaml"
        )
        for task in small_robot_tasks
    )
    assert [
        int(task["candidates"][0]["env"]["WORKER_INDEX_BASE"])
        for task in small_robot_tasks
    ] == [22800, 23200, 23600, 24000]
    fallback_snapshot = {
        "resources": {
            "gf1": {
                "available": True,
                "count": 8,
                "free_count": 0,
                "managed_reserved_indices": [],
                "gpus": [
                    {"index": index, "memory_used_mib": 70000} for index in range(8)
                ],
            },
            "Robot-East-H20": {
                "available": True,
                "capacity": 8,
                "active_gpus_all_users": 0,
                "queueing_all_users": [],
            },
        }
    }
    for seed in (1001, 1002):
        parent = next(
            task
            for task in json.loads(scheduler.QUEUE_PATH.read_text())["tasks"]
            if task["id"] == f"pi05_mt1_oracle_seed{seed}_correct_eval"
        )
        east = next(
            candidate
            for candidate in parent["candidates"]
            if candidate["resource"] == "Robot-East-H20"
        )
        assert scheduler.candidate_available(east, fallback_snapshot)
        scheduler.reserve_dispatched_candidate(fallback_snapshot, east)
    assert (
        fallback_snapshot["resources"]["Robot-East-H20"]["active_gpus_all_users"] == 8
    )

    fallback_snapshot["resources"]["gf1"].update(
        {
            "free_count": 8,
            "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
        }
    )
    for seed in (1001, 1002):
        helper = tasks[f"pi05_mt1_oracle_seed{seed}_correct_eval_attach_gf1g4"]
        candidate = helper["candidates"][0]
        assert scheduler.candidate_available(candidate, fallback_snapshot)
        scheduler.reserve_dispatched_candidate(fallback_snapshot, candidate)
    assert fallback_snapshot["resources"]["gf1"]["free_count"] == 0

    seed1001_parent = next(
        task
        for task in json.loads(scheduler.QUEUE_PATH.read_text())["tasks"]
        if task["id"] == "pi05_mt1_oracle_seed1001_correct_eval"
    )
    parent_candidate = seed1001_parent["candidates"][0]
    seed1001_helper = tasks["pi05_mt1_oracle_seed1001_correct_eval_attach_gf1g4"][
        "candidates"
    ][0]
    assert set(parent_candidate["gpu_indices"]).isdisjoint(
        seed1001_helper["gpu_indices"]
    )
    assert sorted(
        parent_candidate["gpu_indices"] + seed1001_helper["gpu_indices"]
    ) == list(range(8))


def test_mt1_seed1002_north_overflow_is_staged_verified_and_materialized(
    tmp_path: Path,
) -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    before = len(queue["tasks"])
    scheduler.add_pi05_mt1_replication_north_overflow(queue)
    assert len(queue["tasks"]) == before + 7
    tasks = {task["id"]: task for task in queue["tasks"]}

    parent = tasks["pi05_mt1_oracle_seed1002_correct_eval"]
    decision_marker = str(
        scheduler.REPO
        / "logs/resource_markers/pi05_mt1_seed1002_north_stage_decided.ok"
    )
    stage_marker = str(
        scheduler.REPO
        / "logs/resource_markers/pi05_mt1_seed1002_north_eval_checkpoint.ok"
    )
    assert decision_marker in parent["ready_files"]
    assert [location["label"] for location in parent["completion_locations"]] == [
        "shared",
        "north",
    ]
    assert [candidate["resource"] for candidate in parent["candidates"]] == [
        "gf1",
        "Robot-East-H20",
        "Robot-North-H20",
    ]
    north = parent["candidates"][-1]
    assert north["gpus"] == 4
    assert north["env"]["CKPT"].startswith(scheduler.NORTH_REPO)
    assert north["ready_files_remote"][0].startswith(scheduler.NORTH_REPO)

    local_decision = tmp_path / "north-stage-decided.ok"
    local_stage = tmp_path / "north-stage-succeeded.ok"
    local_decision.write_text("outcome=skipped\n")
    readiness_parent = copy.deepcopy(parent)
    readiness_parent["ready_files"] = [str(local_decision)]
    readiness_north = readiness_parent["candidates"][-1]
    readiness_north["ready_files"] = [str(local_stage)]
    readiness_north["ready_files_remote"] = []
    assert scheduler.ready(readiness_parent)
    assert scheduler.readiness_spec_satisfied(readiness_parent["candidates"][0])
    assert not scheduler.readiness_spec_satisfied(readiness_north)
    local_stage.write_text("validated=true\n")
    assert scheduler.readiness_spec_satisfied(readiness_north)

    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["timestamp"] = "2026-08-03T15:00:00Z"
    snapshot["resources"].update(
        {
            "gf1": {"count": 8, "free_count": 0},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 8,
                "queueing_all_users": [],
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 32,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
            "local": {"count": 2, "free_count": 2},
        }
    )
    assert scheduler.ordered_dispatch_candidates(parent, snapshot)[0]["resource"] == (
        "Robot-North-H20"
    )
    snapshot["resources"]["gf1"]["free_count"] = 4
    assert (
        scheduler.ordered_dispatch_candidates(parent, snapshot)[0]["resource"] == "gf1"
    )

    stage = tasks["pi05_mt1_seed1002_sync_north_eval_checkpoint"]
    assert stage["candidates"][0]["gpus"] == 0
    assert "SEED=1002" in stage["candidates"][0]["command"]
    assert stage["completion_glob"] == decision_marker
    assert "STAGE_MARKER=" in stage["candidates"][0]["command"]
    assert "DECISION_MARKER=" in stage["candidates"][0]["command"]
    assert stage["candidates"][0]["retry_cooldown_seconds"] == 60
    assert any(
        path.endswith("step49999_checkpoint_audit.ok") for path in stage["ready_files"]
    )
    assert [item["label"] for item in stage["progress_logs"]] == [
        "phase",
        "transfer",
    ]
    assert "phase=([a-z0-9-]+)" in stage["progress_logs"][0]["regex"]
    assert "launcher.log" in stage["progress_logs"][1]["glob"]

    attach = tasks["pi05_mt1_oracle_seed1002_correct_eval_attach_bj2g"]
    assert attach["satisfied_by_task"] == parent["id"]
    assert attach["completion_remote"] is True
    assert attach["candidates"][0]["resource"] == "Robot-North-H20"
    assert attach["candidates"][0]["gpus"] == 2
    assert (
        len(
            [
                path
                for path in attach["ready_files_remote"]
                if path.endswith(".task_scheduler.json")
            ]
        )
        == 4
    )

    four_gpu_attach = [
        tasks[f"pi05_mt1_oracle_seed1002_correct_eval_attach_bj4g{index}"]
        for index in range(4)
    ]
    assert [task["candidates"][0]["gpus"] for task in four_gpu_attach] == [4] * 4
    assert [
        task["candidates"][0]["env"]["WORKER_INDEX_BASE"] for task in four_gpu_attach
    ] == ["12000", "12800", "13600", "14400"]
    assert all(task["satisfied_by_task"] == parent["id"] for task in four_gpu_attach)
    assert all(task["completion_remote"] is True for task in four_gpu_attach)
    assert all(
        task["candidates"][0]["yaml"].endswith("pi05_confirmatory_attach_bj_4h20.yaml")
        for task in four_gpu_attach
    )
    north_yaml = (
        scheduler.REPO / "train_scripts/kai/volc/pi05_confirmatory_attach_bj_4h20.yaml"
    ).read_text()
    assert 'Flavor: "ml.pni3ln.17xlarge"' in north_yaml
    assert "export ATTACH_GPU_COUNT=4" in north_yaml

    worker_bases = [
        int(task["candidates"][0]["env"]["WORKER_INDEX_BASE"])
        for task in [attach, *four_gpu_attach]
    ]
    assert len(worker_bases) == len(set(worker_bases))
    assert 22200 + max(worker_bases) + 120 + 300 < 65536

    capacity = north_snapshot(primary=0, all_users=34)
    parent_candidate = copy.deepcopy(parent["candidates"][-1])
    parent_candidate["ready_files"] = []
    parent_candidate["ready_files_remote"] = []
    assert (
        scheduler.candidate_credential_profile(parent_candidate, capacity) == "primary"
    )
    scheduler.reserve_dispatched_candidate(capacity, parent_candidate, "primary")
    profiles = []
    for helper in [attach, *four_gpu_attach]:
        candidate = copy.deepcopy(helper["candidates"][0])
        candidate["ready_files"] = []
        candidate["ready_files_remote"] = []
        profile = scheduler.candidate_credential_profile(candidate, capacity)
        assert profile is not None
        profiles.append(profile)
        scheduler.reserve_dispatched_candidate(capacity, candidate, profile)
    assert profiles == ["primary", "primary", "primary", "primary", "backup"]
    assert capacity["resources"]["beijing"]["owned_active_gpus"] == 18
    assert capacity["resources"]["beijing"]["backup"]["managed_active_gpus"] == 4
    assert capacity["resources"]["beijing"]["active_gpus_all_users"] == 56

    materialize = tasks["pi05_mt1_seed1002_correct_sync_from_north"]
    assert materialize["materialize_north_result_for"] == parent["id"]
    assert (
        "RESULT_NAME=pi05_mt1_oracle_seed1002_correct"
        in materialize["candidates"][0]["command"]
    )

    sync_script = (
        scheduler.REPO / "train_scripts/kai/sync_pi05_mt1_seed1000_to_north.sh"
    ).read_text()
    assert "SEED=${SEED:-1000}" in sync_script
    assert "pi05_robotwin_mt1_oracle_seed${SEED}" in sync_script
    assert "SYNC_TRANSPORT:-auto" in sync_script
    assert "sync_tree_to_north_verified_tos.sh" in sync_script
    assert "falling back to verified SSH stream" in sync_script
    assert 'echo "phase=ssh-fallback"' in sync_script
    assert "elapsed_seconds=%s" in sync_script
    assert 'marker_tmp="${MARKER}.tmp.$$"' in sync_script
    assert 'mv "$marker_tmp" "$MARKER"' in sync_script
    tos_sync_script = (
        scheduler.REPO / "train_scripts/kai/sync_tree_to_north_verified_tos.sh"
    ).read_text()
    assert 'tosutil cp "$local_stage/" "$object" -r -flat -vchecksum' in tos_sync_script
    assert "logs/sync/tos_staging" in tos_sync_script
    assert 'cp -a --reflink=auto "$src/." "$local_stage/"' in tos_sync_script
    assert "-r -f -flat -vchecksum" in tos_sync_script
    assert 'echo "phase=tos-upload"' in tos_sync_script
    assert 'echo "phase=sha256-verify"' in tos_sync_script
    assert "sha256sum -c -" in tos_sync_script
    assert 'mv $(printf %q "$incoming") $(printf %q "$dst")' in tos_sync_script
    stage_script = (
        scheduler.REPO / "train_scripts/kai/stage_pi05_mt1_north_overflow.sh"
    ).read_text()
    assert "outcome=skipped" in stage_script
    assert "outcome=staged" in stage_script
    assert 'mv "$temporary" "$DECISION_MARKER"' in stage_script
    scheduler.validate_queue(queue)


@pytest.mark.parametrize("sync_succeeds", [False, True])
def test_mt1_north_stage_decision_preserves_fallback(
    tmp_path: Path, sync_succeeds: bool
) -> None:
    fake_sync = tmp_path / "train_scripts/kai/sync_pi05_mt1_seed1000_to_north.sh"
    fake_sync.parent.mkdir(parents=True)
    fake_sync.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n" + ('touch "$MARKER"\n' if sync_succeeds else "exit 7\n")
    )
    env = {
        **os.environ,
        "REPO": str(tmp_path),
        "SEED": "1002",
    }
    result = subprocess.run(
        [
            "bash",
            str(scheduler.REPO / "train_scripts/kai/stage_pi05_mt1_north_overflow.sh"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    marker_root = tmp_path / "logs/resource_markers"
    decision = (marker_root / "pi05_mt1_seed1002_north_stage_decided.ok").read_text()
    expected = "staged" if sync_succeeds else "skipped"
    assert f"outcome={expected}" in decision
    assert f"sync_rc={0 if sync_succeeds else 7}" in decision
    assert (
        marker_root / "pi05_mt1_seed1002_north_eval_checkpoint.ok"
    ).exists() is sync_succeeds


def test_north_stage_progress_reports_tos_phase_and_percentage(tmp_path: Path) -> None:
    log = tmp_path / "launcher.log"
    log.write_text("phase=tos-upload\n[====>____] 42.25% 5.0MB/s\r")
    task = {
        "id": "pi05_mt1_seed1002_sync_north_eval_checkpoint",
        "enabled": True,
        "completion_glob": str(tmp_path / "decision.ok"),
        "completion_min_count": 1,
        "progress_logs": [
            {"label": "phase", "glob": str(log), "regex": r"phase=([a-z0-9-]+)"},
            {"label": "transfer", "glob": str(log), "regex": r"([0-9]+(?:\.[0-9]+)?)%"},
        ],
    }
    state = {"tasks": {task["id"]: {"status": "running", "attempts": [{}]}}}
    scheduler.refresh_running_progress({"tasks": [task]}, state)
    assert state["tasks"][task["id"]]["runtime_progress"] == (
        "phase=tos-upload, transfer=42.25"
    )


def test_mt1_replication_overflow_fills_gf1_then_routes_seed1002_north() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    scheduler.add_pi05_mt1_replication_north_overflow(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}
    snapshot = north_snapshot(primary=0, all_users=27)
    snapshot["timestamp"] = "2026-08-03T15:04:00Z"
    snapshot["resources"].update(
        {
            "gf1": {
                "available": True,
                "count": 8,
                "free_count": 8,
                "managed_reserved_indices": [],
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
            },
            "Robot-East-H20": {
                "available": True,
                "capacity": 8,
                "active_gpus_all_users": 8,
                "queueing_all_users": [],
            },
            "robot-task": {
                "available": True,
                "capacity": 32,
                "active_gpus_all_users": 24,
                "owned_active_gpus": 8,
                "queueing_all_users": [],
            },
            "local": {"available": True, "count": 2, "free_count": 2},
        }
    )

    seed1001 = tasks["pi05_mt1_oracle_seed1001_correct_eval"]
    parent = seed1001["candidates"][0]
    helper = tasks["pi05_mt1_oracle_seed1001_correct_eval_attach_gf1g4"]["candidates"][
        0
    ]
    assert parent["gpu_indices"] == [0, 1, 2, 3]
    assert helper["gpu_indices"] == [4, 5, 6, 7]
    scheduler.reserve_dispatched_candidate(snapshot, parent, "primary")
    assert scheduler.candidate_available(helper, snapshot)
    scheduler.reserve_dispatched_candidate(snapshot, helper, "primary")
    assert snapshot["resources"]["gf1"]["free_count"] == 0

    seed1002 = tasks["pi05_mt1_oracle_seed1002_correct_eval"]
    assert scheduler.ordered_dispatch_candidates(seed1002, snapshot)[0]["resource"] == (
        "Robot-North-H20"
    )
    north = copy.deepcopy(seed1002["candidates"][-1])
    north["ready_files"] = []
    north["ready_files_remote"] = []
    assert scheduler.candidate_credential_profile(north, snapshot) == "primary"


def test_mt1_8g_optimization_probes_cover_input_and_hybrid_fsdp() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt1_8g_optimization_probes(queue)
    assert len(queue["tasks"]) == 2
    observed = {
        (
            int(task["candidates"][0]["env"]["WORKERS"]),
            int(task["candidates"][0]["env"]["FSDP_DEVICES"]),
        )
        for task in queue["tasks"]
    }
    assert observed == {(16, 1), (16, 2)}
    for task in queue["tasks"]:
        candidate = task["candidates"][0]
        assert candidate["resource"] == "robot-task"
        assert candidate["gpus"] == 8
        assert candidate["queue_timeout_seconds"] == 900
        assert candidate["env"]["RESULT"].endswith(".json")


def test_queue_validation_rejects_attach_worker_port_overflow() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    task = next(
        task
        for task in queue["tasks"]
        if task["id"] == "pi05_mt1_oracle_seed1001_correct_eval_attach_east4g"
    )
    task["candidates"][0]["env"]["WORKER_INDEX_BASE"] = "50000"
    with pytest.raises(ValueError, match="attach worker port exceeds TCP range"):
        scheduler.validate_queue(queue)


def test_mt1_replication_fanout_fills_fifty_gpus_in_two_polls(
    monkeypatch,
) -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    prefixes = tuple(
        f"pi05_mt1_oracle_seed{seed}_correct_eval" for seed in (1001, 1002)
    )
    tasks = [
        task
        for task in queue["tasks"]
        if any(
            task["id"] == prefix or task["id"].startswith(prefix + "_attach")
            for prefix in prefixes
        )
    ]
    assert len(tasks) == 20
    state = {
        "tasks": {task["id"]: {"status": "pending", "attempts": []} for task in tasks}
    }
    snapshot = north_snapshot(primary=0, all_users=0)
    snapshot["resources"].update(
        {
            "gf1": {
                "available": True,
                "count": 8,
                "free_count": 8,
                "managed_reserved_indices": [],
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
            },
            "Robot-East-H20": {
                "available": True,
                "capacity": 8,
                "active_gpus_all_users": 0,
                "queueing_all_users": [],
            },
            "robot-task": {
                "available": True,
                "capacity": 32,
                "active_gpus_all_users": 0,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
            "local": {
                "available": True,
                "count": 2,
                "free_count": 2,
                "managed_reserved_indices": [],
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(2)],
            },
        }
    )
    launched = []
    monkeypatch.setattr(scheduler, "ready", lambda _task: True)
    monkeypatch.setattr(
        scheduler,
        "completion_evidence",
        lambda _task: (False, "completion artifacts local=0/1"),
    )
    monkeypatch.setattr(scheduler, "check_managed_task", lambda *_args: None)
    monkeypatch.setattr(scheduler, "candidate_exhausted", lambda *_args: False)
    monkeypatch.setattr(scheduler, "candidate_in_cooldown", lambda *_args: False)
    monkeypatch.setattr(
        scheduler,
        "capture_submission_recommendation",
        lambda task, candidate, _snapshot: (
            Path(f"/tmp/{task['id']}.json"),
            {
                "global_recommendation": candidate["resource"],
                "task_eligible_recommendation": candidate["resource"],
                "selected_resource": candidate["resource"],
                "selection_analysis": "test",
            },
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "launch_gf1",
        lambda candidate: launched.append(candidate["resource"]) or "101",
    )
    monkeypatch.setattr(
        scheduler,
        "launch_local",
        lambda candidate: launched.append(candidate["resource"]) or "102",
    )
    monkeypatch.setattr(
        scheduler,
        "submit_platform",
        lambda candidate, _profile: launched.append(candidate["resource"])
        or f"job-{len(launched)}",
    )
    for name in (
        "capture_pi05_confirmatory_launch",
        "capture_pi05_confirmatory_eval_launch",
        "capture_pi05_mt3_eval_launch",
        "capture_pi05_mt12_training_launch",
        "capture_pi05_mt12_eval_launch",
    ):
        monkeypatch.setattr(scheduler, name, lambda *_args: None)
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args: None)

    scheduler.dispatch({"tasks": tasks}, state, snapshot)
    assert sum(item["status"] == "running" for item in state["tasks"].values()) == 8
    scheduler.dispatch({"tasks": tasks}, state, snapshot)

    running = [
        task_id
        for task_id, task_state in state["tasks"].items()
        if task_state["status"] == "running"
    ]
    pending = [
        task_id
        for task_id, task_state in state["tasks"].items()
        if task_state["status"] == "pending"
    ]
    assert len(running) == 15
    assert pending == [
        "pi05_mt1_oracle_seed1001_correct_eval_attach_gf1g4",
        "pi05_mt1_oracle_seed1002_correct_eval_attach_gf1g4",
        "pi05_mt1_oracle_seed1002_correct_eval_attach_cnsh4g_2",
        "pi05_mt1_oracle_seed1002_correct_eval_attach_cnsh4g_3",
        "pi05_mt1_oracle_seed1002_correct_eval_attach_local2g",
    ]
    assert launched.count("gf1") == 2
    assert launched.count("Robot-East-H20") == 2
    assert launched.count("robot-task") == 10
    assert launched.count("local") == 1
    assert snapshot["resources"]["gf1"]["free_count"] == 0
    assert snapshot["resources"]["Robot-East-H20"]["active_gpus_all_users"] == 8
    assert snapshot["resources"]["robot-task"]["active_gpus_all_users"] == 32
    assert snapshot["resources"]["local"]["free_count"] == 0


def test_robot_task_fragmentation_ignores_legacy_attempt_without_gpu_shape() -> None:
    candidate = {"resource": "robot-task", "gpus": 2}
    state = {
        "tasks": {
            "legacy": {
                "attempts": [
                    {
                        "resource": "robot-task",
                        "active_gpus_at_dispatch": 24,
                        "failure": "reclaimed after queueing for more than 180 seconds",
                    }
                ]
            },
            "four_gpu_probe": {
                "attempts": [
                    {
                        "resource": "robot-task",
                        "gpus": 4,
                        "active_gpus_at_dispatch": 24,
                        "failure": "reclaimed after queueing because Shanghai queueing is disabled",
                    }
                ]
            },
        }
    }
    snapshot = {"resources": {"robot-task": {"active_gpus_all_users": 24}}}
    assert not scheduler.robot_task_fragmentation_blocked(candidate, state, snapshot)
    candidate["gpus"] = 4
    assert scheduler.robot_task_fragmentation_blocked(candidate, state, snapshot)


def test_mt3_tracker_secondary_outputs_are_declared() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    for candidate in ("current_frame", "history_proprio"):
        outputs = tasks[f"pi05_mt3_tracker_{candidate}_train"]["produces_files"]
        assert any(path.endswith("/tracker.pt") for path in outputs)
        assert any(path.endswith("/validation_predictions.npz") for path in outputs)
    selection = tasks["pi05_mt3_tracker_select"]["produces_files"]
    assert selection == [str(scheduler.REPO / "logs/mt_stage_tracker/selection.json")]


def test_mt_final_evals_fill_gf1_then_east_then_north(monkeypatch) -> None:
    monkeypatch.setattr(scheduler, "readiness_spec_satisfied", lambda spec: True)
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    tasks = {task["id"]: task for task in json.loads(queue_path.read_text())["tasks"]}
    task_ids = (
        "pi05_mt1_oracle_seed1000_correct_eval",
        "pi05_mt1_oracle_seed1000_null_eval",
        "pi05_mt2_null_seed1000_eval",
        "pi05_mt1_oracle_seed1000_cross_task_eval",
        "pi05_mt1_oracle_seed1000_within_task_eval",
    )
    snapshot = north_snapshot(primary=0, all_users=34)
    snapshot["resources"].update(
        {
            "gf1": {
                "available": True,
                "count": 8,
                "free_count": 8,
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
            },
            "Robot-East-H20": {
                "available": True,
                "capacity": 8,
                "active_gpus_all_users": 0,
                "queueing_all_users": [],
            },
            "robot-task": {
                "available": True,
                "capacity": 32,
                "active_gpus_all_users": 0,
                "owned_active_gpus": 0,
                "queueing_all_users": [],
            },
            "local": {
                "available": True,
                "count": 2,
                "free_count": 2,
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(2)],
            },
        }
    )

    def select_resources(resource_snapshot):
        selected = {}
        for task_id in task_ids:
            candidates = scheduler.ordered_dispatch_candidates(
                tasks[task_id], resource_snapshot
            )
            for candidate in candidates:
                profile = (
                    scheduler.candidate_credential_profile(candidate, resource_snapshot)
                    if candidate["kind"] == "platform"
                    else "primary"
                )
                if profile is not None and scheduler.candidate_available(
                    candidate, resource_snapshot, profile
                ):
                    selected[task_id] = candidate["resource"]
                    scheduler.reserve_dispatched_candidate(
                        resource_snapshot, candidate, profile
                    )
                    break
        return selected

    initial_snapshot = copy.deepcopy(snapshot)
    selected = select_resources(snapshot)
    assert selected == {
        "pi05_mt1_oracle_seed1000_correct_eval": "gf1",
        "pi05_mt1_oracle_seed1000_null_eval": "gf1",
        "pi05_mt2_null_seed1000_eval": "Robot-East-H20",
        "pi05_mt1_oracle_seed1000_cross_task_eval": "Robot-East-H20",
        "pi05_mt1_oracle_seed1000_within_task_eval": "Robot-North-H20",
    }

    monkeypatch.setattr(
        scheduler,
        "readiness_spec_satisfied",
        lambda spec: spec.get("resource") != "Robot-North-H20",
    )
    selected = select_resources(initial_snapshot)
    assert selected == {
        "pi05_mt1_oracle_seed1000_correct_eval": "gf1",
        "pi05_mt1_oracle_seed1000_null_eval": "gf1",
        "pi05_mt2_null_seed1000_eval": "Robot-East-H20",
        "pi05_mt1_oracle_seed1000_cross_task_eval": "Robot-East-H20",
        "pi05_mt1_oracle_seed1000_within_task_eval": "robot-task",
    }


def test_mt3_eval_attach_is_selection_aware_and_gate_controlled() -> None:
    queue = {"tasks": []}
    scheduler.add_pi05_mt3_eval_attach_tasks(queue)
    assert len(queue["tasks"]) == 10
    tasks = {task["id"]: task for task in queue["tasks"]}
    assert set(tasks) == {
        "pi05_mt3_seed1000_predicted_eval_attach",
        "pi05_mt3_seed1000_within_task_eval_attach",
        "pi05_mt3_seed1000_null_eval_attach",
        "pi05_mt3_seed1000_oracle_eval_attach",
        "pi05_mt3_seed1001_predicted_eval_attach",
        "pi05_mt3_seed1001_null_eval_attach",
        "pi05_mt3_seed1001_within_task_eval_attach",
        "pi05_mt3_seed1002_predicted_eval_attach",
        "pi05_mt3_seed1002_null_eval_attach",
        "pi05_mt3_seed1002_within_task_eval_attach",
    }
    for task in tasks.values():
        assert len(task["ready_any"]) == 2
        assert all(len(spec["ready_files"]) == 4 for spec in task["ready_any"])
        assert any(
            path.endswith("pi05_mt3_tracker_selected.ok")
            for path in task["ready_files"]
        )
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "robot-task",
            "local",
        ]
        assert task["candidates"][0]["yaml"].endswith("pi05_mt3_attach_cnsh_4a100.yaml")
        assert task["candidates"][0]["env"]["EVAL_WORKERS_PER_GPU"] == "2"
        assert "attach_pi05_mt3_formal.sh" in task["candidates"][1]["command"]
        assert "EVAL_WORKERS_PER_GPU=2" in task["candidates"][1]["command"]
        platform_base = int(task["candidates"][0]["env"]["WORKER_INDEX_BASE"])
        local_base = int(
            re.search(
                r"WORKER_INDEX_BASE=([0-9]+)",
                task["candidates"][1]["command"],
            ).group(1)
        )
        assert platform_base != local_base
        assert max(platform_base, local_base) + 22200 + 120 + 300 < 65536

    local_launcher = (
        scheduler.REPO / "train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh"
    ).read_text()
    assert (
        'EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-${NUM_WORKERS:-1}}"'
        in local_launcher
    )
    assert "export ALLOW_GPU_OVERSUBSCRIBE=1" in local_launcher
    formal_launcher = (
        scheduler.REPO / "train_scripts/kai/eval/run_pi05_mt3_formal.sh"
    ).read_text()
    assert "EVAL_WORKERS_PER_GPU=${EVAL_WORKERS_PER_GPU:-2}" in formal_launcher


def test_mt3_eval_attach_result_names_match_parent_evaluations() -> None:
    queue_path = Path(__file__).with_name("resource_scheduler_queue.json")
    queue = json.loads(queue_path.read_text())
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    scheduler.add_pi05_mt4_replication_tasks(queue)
    scheduler.add_pi05_mt3_eval_attach_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    attach_tasks = [
        task
        for task_id, task in tasks.items()
        if task_id.startswith("pi05_mt3_seed") and task_id.endswith("_eval_attach")
    ]
    assert len(attach_tasks) == 10
    for task in attach_tasks:
        parent = tasks[task["satisfied_by_task"]]
        match = re.search(
            r"(?:^| )RESULT_NAME=([^ ]+)", parent["candidates"][0]["command"]
        )
        assert match is not None
        parent_result = match.group(1)
        assert task["candidates"][0]["env"]["RESULT_NAME"] == parent_result
        scheduler_paths = [
            path
            for alternative in task["ready_any"]
            for path in alternative["ready_files"]
        ]
        assert len(scheduler_paths) == 8
        assert all(f"/robotwin/{parent_result}/" in path for path in scheduler_paths)


def test_mt3_eval_launch_provenance_records_frozen_protocol(
    tmp_path, monkeypatch
) -> None:
    selection = tmp_path / "logs/mt_stage_tracker/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"selected": "history_proprio"}))
    protocol = (
        tmp_path / "lmvla/paper_iclr_lmvla/manifests/robotwin_mt3_protocol_v1.json"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps(
            {
                "joint_policy_training": {
                    "action_representation": "absolute joint actions",
                    "normalization": "mean/std",
                    "policy_updates": 50000,
                    "policy_batch_size": 16,
                }
            }
        )
    )
    captured = {}
    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(scheduler, "sha256_file", lambda path: "digest")
    monkeypatch.setattr(
        scheduler, "atomic_json", lambda path, payload: captured.update(payload)
    )
    monkeypatch.setattr(scheduler, "log", lambda message: None)
    task = {
        "id": "pi05_mt3_learned_seed1001_oracle_eval",
        "ready_files": [
            str(
                tmp_path / "kai0/checkpoints/pi05_robotwin_mt3_learned_exact/"
                "pi05_robotwin_mt3_learned_seed1001/49999/params/_METADATA"
            )
        ],
    }
    candidate = {
        "kind": "ssh",
        "resource": "gf1",
        "gpus": 4,
        "gpu_indices": [0, 1, 2, 3],
        "command": "run-mt3",
    }
    scheduler.capture_pi05_mt3_eval_launch(task, candidate, "gf1-123")
    assert captured["training_seed"] == 1001
    assert captured["intervention"] == "oracle"
    assert captured["selected_tracker"] == "history_proprio"
    assert captured["protocol"] == {
        "action_representation": "absolute joint actions",
        "normalization": "mean/std",
        "policy_updates": 50000,
        "policy_batch_size": 16,
        "evaluation_cells": 24,
        "evaluation_episodes_per_cell": 50,
        "history_enabled": True,
        "oracle_enabled": True,
    }
    assert captured["sha256"]["checkpoint_metadata"] == "digest"
    assert captured["sha256"]["launch_command"]


def test_mt1_eval_launch_provenance_records_shared_control_protocol(
    tmp_path, monkeypatch
) -> None:
    captured = {}
    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(
        scheduler,
        "PI05_CONFIRMATORY_SCENE_MANIFEST_SHARED",
        str(tmp_path / "scene_manifest.json"),
    )
    monkeypatch.setattr(scheduler, "sha256_file", lambda path: "digest")
    monkeypatch.setattr(
        scheduler, "atomic_json", lambda path, payload: captured.update(payload)
    )
    monkeypatch.setattr(scheduler, "log", lambda message: None)
    metadata = (
        tmp_path / "kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/"
        "pi05_robotwin_mt1_oracle_seed1000/49999/params/_METADATA"
    )
    task = {
        "id": "pi05_mt1_oracle_seed1002_correct_eval",
        "ready_files": [str(metadata)],
    }
    candidate = {
        "kind": "ssh",
        "resource": "gf1",
        "gpus": 4,
        "gpu_indices": [0, 1, 2, 3],
        "command": "run-mt1-within-task",
    }
    scheduler.capture_pi05_mt12_eval_launch(task, candidate, "gf1-456")
    assert captured["arm"] == "mt1_oracle"
    assert captured["training_seed"] == 1002
    assert captured["intervention"] == "correct"
    assert captured["protocol"] == {
        "action_representation": "absolute joint actions",
        "normalization": "mean/std",
        "scene_cells": 24,
        "episodes_per_cell": 50,
        "fixed_seed_max_attempts": 500,
        "oracle_trajectory_enabled": True,
    }
    assert captured["resource"]["name"] == "gf1"
    assert len(captured["sha256"]["execution_sources"]) == 6
    assert captured["sha256"]["training_launch_manifest"] == "digest"


def test_mt1_eval_launch_provenance_hashes_north_paths(tmp_path, monkeypatch) -> None:
    captured = {}
    remote_calls = []
    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(scheduler, "NORTH_REPO", "/north/repo")
    monkeypatch.setattr(
        scheduler,
        "PI05_CONFIRMATORY_SCENE_MANIFEST_NORTH",
        "/north/repo/scene_manifest.json",
    )
    monkeypatch.setattr(scheduler, "sha256_file", lambda path: "local-digest")

    def remote_sha(paths):
        remote_calls.extend(paths)
        return {path: "remote-digest" for path in paths}

    monkeypatch.setattr(scheduler, "remote_sha256", remote_sha)
    monkeypatch.setattr(
        scheduler, "atomic_json", lambda path, payload: captured.update(payload)
    )
    monkeypatch.setattr(scheduler, "log", lambda message: None)
    metadata = (
        "/north/repo/kai0/checkpoints/pi05_robotwin_mt1_oracle_exact/"
        "pi05_robotwin_mt1_oracle_seed1000/49999/params/_METADATA"
    )
    task = {
        "id": "pi05_mt1_oracle_seed1000_cross_task_eval",
        "ready_files_remote": [metadata],
    }
    candidate = {
        "kind": "platform",
        "resource": "Robot-North-H20",
        "gpus": 4,
        "yaml": "north.yaml",
        "task_name": "north-mt1-cross",
    }
    scheduler.capture_pi05_mt12_eval_launch(task, candidate, "north-job")
    assert captured["intervention"] == "cross_task"
    assert captured["resource"]["name"] == "Robot-North-H20"
    assert captured["sha256"]["checkpoint_metadata"] == "remote-digest"
    assert metadata in remote_calls
    assert all(path.startswith("/north/repo") for path in remote_calls)


def test_completion_root_supports_specific_confirmatory_glob() -> None:
    root = "/tmp/pi05_rt_a0_public_exact_seed1001"
    pattern = root + "/seed*/run/confirmatory-seed*/tasks/*/summary.json"
    assert scheduler.completion_root_from_glob(pattern) == root
    assert scheduler.completion_root_from_glob(root + "/**/summary.json") == root


def test_north_uses_primary_when_task_fits_personal_limit() -> None:
    snapshot = north_snapshot(primary=16)
    assert (
        scheduler.candidate_credential_profile(north_candidate(), snapshot) == "primary"
    )


def test_north_spills_to_backup_when_primary_task_does_not_fit() -> None:
    snapshot = north_snapshot(primary=18)
    assert (
        scheduler.candidate_credential_profile(north_candidate(), snapshot) == "backup"
    )


def test_north_spills_at_full_primary_limit() -> None:
    snapshot = north_snapshot(primary=20)
    assert (
        scheduler.candidate_credential_profile(north_candidate(2), snapshot) == "backup"
    )


def test_north_does_not_submit_without_physical_capacity() -> None:
    snapshot = north_snapshot(primary=20, all_users=55)
    assert scheduler.candidate_credential_profile(north_candidate(2), snapshot) is None


def test_north_primary_does_not_queue_when_personal_capacity_but_no_physical_cards() -> (
    None
):
    snapshot = north_snapshot(primary=18, all_users=56)
    assert scheduler.candidate_credential_profile(north_candidate(2), snapshot) is None


def test_north_does_not_submit_while_queue_has_waiting_job() -> None:
    snapshot = north_snapshot(primary=20, queueing=True)
    assert scheduler.candidate_credential_profile(north_candidate(2), snapshot) is None


def test_north_does_not_use_disabled_backup_identity() -> None:
    snapshot = north_snapshot(primary=20, backup_enabled=False)
    assert scheduler.candidate_credential_profile(north_candidate(2), snapshot) is None


def robot_task_snapshot(*, active: int, queueing: bool = False) -> dict:
    return {
        "resources": {
            "robot-task": {
                "available": True,
                "capacity": 32,
                "active_gpus_all_users": active,
                "owned_active_gpus": active,
                "queueing_all_users": ["queued-job"] if queueing else [],
            }
        }
    }


def test_robot_task_dispatches_only_when_physical_cards_are_free() -> None:
    candidate = {"kind": "platform", "resource": "robot-task", "gpus": 4}
    assert scheduler.candidate_available(candidate, robot_task_snapshot(active=28))
    assert not scheduler.candidate_available(candidate, robot_task_snapshot(active=30))


def test_robot_task_does_not_stack_jobs_behind_existing_queueing() -> None:
    candidate = {"kind": "platform", "resource": "robot-task", "gpus": 4}
    assert not scheduler.candidate_available(
        candidate, robot_task_snapshot(active=0, queueing=True)
    )


def test_robot_task_submission_control_blocks_new_dispatches() -> None:
    candidate = {"kind": "platform", "resource": "robot-task", "gpus": 4}
    snapshot = robot_task_snapshot(active=0)
    snapshot["resources"]["robot-task"]["submission_enabled"] = False
    assert not scheduler.candidate_available(candidate, snapshot)


def test_fixed_gpu_candidate_respects_managed_index_reservation() -> None:
    candidate = {
        "kind": "ssh",
        "resource": "gf1",
        "gpus": 1,
        "gpu_indices": [0],
    }
    snapshot = {
        "resources": {
            "gf1": {
                "available": True,
                "free_count": 8,
                "managed_reserved_indices": [0],
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
            }
        }
    }
    assert not scheduler.candidate_available(candidate, snapshot)
    candidate["gpu_indices"] = [1]
    assert scheduler.candidate_available(candidate, snapshot)


def test_running_attempt_reserves_index_before_memory_is_allocated() -> None:
    candidate = {
        "kind": "ssh",
        "resource": "gf1",
        "gpus": 1,
        "gpu_indices": [0],
    }
    queue = {"tasks": [{"id": "probe", "candidates": [candidate]}]}
    state = {
        "tasks": {
            "probe": {
                "status": "running",
                "attempts": [
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 1,
                        "gpu_indices": [0],
                    }
                ],
            }
        }
    }
    snapshot = {
        "resources": {
            "gf1": {
                "available": True,
                "count": 8,
                "free_count": 8,
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(8)],
            },
            "local": {
                "available": True,
                "count": 2,
                "free_count": 2,
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(2)],
            },
        }
    }
    scheduler.apply_managed_gpu_reservations(queue, state, snapshot)
    assert snapshot["resources"]["gf1"]["managed_reserved_indices"] == [0]
    assert snapshot["resources"]["gf1"]["free_count"] == 7
    assert not scheduler.candidate_available(candidate, snapshot)


def test_same_poll_reservation_prevents_platform_overcommit() -> None:
    snapshot = {
        "resources": {
            "robot-task": {
                "available": True,
                "capacity": 32,
                "active_gpus_all_users": 24,
                "owned_active_gpus": 24,
                "queueing_all_users": [],
            },
            "beijing": {
                "available": True,
                "capacity": 56,
                "personal_limit": 20,
                "active_gpus_all_users": 46,
                "owned_active_gpus": 20,
                "owned_queueing": [],
                "queueing_all_users": [],
                "backup": {
                    "enabled": True,
                    "available": True,
                    "managed_active_gpus": 0,
                    "managed_queueing": [],
                    "personal_limit": 20,
                },
            },
        }
    }
    sh_candidate = {"kind": "platform", "resource": "robot-task", "gpus": 4}
    assert scheduler.candidate_available(sh_candidate, snapshot)
    scheduler.reserve_dispatched_candidate(snapshot, sh_candidate)
    scheduler.reserve_dispatched_candidate(snapshot, sh_candidate)
    assert snapshot["resources"]["robot-task"]["active_gpus_all_users"] == 32
    assert not scheduler.candidate_available(sh_candidate, snapshot)

    north_candidate = {
        "kind": "platform",
        "resource": "Robot-North-H20",
        "gpus": 2,
    }
    assert scheduler.candidate_available(north_candidate, snapshot, "backup")
    for _ in range(5):
        scheduler.reserve_dispatched_candidate(snapshot, north_candidate, "backup")
    assert snapshot["resources"]["beijing"]["active_gpus_all_users"] == 56
    assert snapshot["resources"]["beijing"]["backup"]["managed_active_gpus"] == 10
    assert not scheduler.candidate_available(north_candidate, snapshot, "backup")


def test_dispatch_launches_multiple_nonoverlapping_tasks_per_poll(monkeypatch) -> None:
    tasks = []
    state = {"tasks": {}}
    for index in (0, 1):
        task_id = f"local-{index}"
        tasks.append(
            {
                "id": task_id,
                "priority": 1,
                "description": task_id,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 1,
                        "gpu_indices": [index],
                        "status_dir": f"/tmp/{task_id}",
                        "command": "true",
                    }
                ],
            }
        )
        state["tasks"][task_id] = {
            "status": "pending",
            "attempts": [],
            "artifacts_complete": False,
            "artifact_progress": "stale terminal evidence",
            "artifact_progress_changed_at": "2026-08-05T00:00:00Z",
            "artifact_progress_checked_at": "2026-08-05T00:00:00Z",
        }
    snapshot = {
        "resources": {
            "local": {
                "available": True,
                "count": 2,
                "free_count": 2,
                "managed_reserved_indices": [],
                "gpus": [{"index": index, "memory_used_mib": 0} for index in range(2)],
            }
        }
    }
    launched = []
    monkeypatch.setattr(
        scheduler, "launch_local", lambda candidate: launched.append(candidate) or "1"
    )
    monkeypatch.setattr(scheduler, "atomic_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "log", lambda *_args, **_kwargs: None)
    scheduler.dispatch({"tasks": tasks}, state, snapshot)
    assert len(launched) == 2
    assert all(value["status"] == "running" for value in state["tasks"].values())
    assert all(
        value["artifacts_complete"] is False
        and "artifact_progress" not in value
        and "artifact_progress_changed_at" not in value
        and "artifact_progress_checked_at" not in value
        for value in state["tasks"].values()
    )
    assert snapshot["resources"]["local"]["managed_reserved_indices"] == [0, 1]
    assert snapshot["resources"]["local"]["free_count"] == 0


def test_stop_managed_attempt_terminates_process_groups(monkeypatch) -> None:
    remote = []
    local = []
    monkeypatch.setattr(
        scheduler,
        "ssh",
        lambda command, script, **_kwargs: remote.append((command, script)) or "",
    )
    monkeypatch.setattr(
        scheduler.os, "killpg", lambda pid, sig: local.append((pid, sig))
    )
    scheduler.stop_managed_attempt({"kind": "ssh", "pid": "123"})
    scheduler.stop_managed_attempt({"kind": "local", "pid": "456"})
    assert remote and "kill -TERM -- -123" in remote[0][1]
    assert local == [(456, scheduler.signal.SIGTERM)]


def test_launch_local_uses_non_login_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(scheduler.subprocess, "Popen", fake_popen)
    candidate = {
        "status_dir": str(tmp_path / "status"),
        "command": "printf ready",
    }

    pid = scheduler.launch_local(candidate)

    assert pid == "12345"
    assert calls[0][0][:2] == ["bash", "-c"]
    launcher_body = calls[0][0][2]
    assert "bash -c " in launcher_body
    assert "bash -lc " not in launcher_body


def test_dynamic_attach_tasks_are_idempotent_and_checkpoint_gated() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_shared_eval_attach_tasks(queue)
    scheduler.add_pi05_north_eval_attach_tasks(queue)
    scheduler.add_pi05_step40000_safety_probes(queue)
    first_count = len(queue["tasks"])
    scheduler.add_pi05_shared_eval_attach_tasks(queue)
    scheduler.add_pi05_north_eval_attach_tasks(queue)
    scheduler.add_pi05_step40000_safety_probes(queue)
    assert len(queue["tasks"]) == first_count
    scheduler.validate_queue(queue)

    confirmatory = [
        task
        for task in queue["tasks"]
        if scheduler.PI05_CONFIRMATORY_EVAL_RE.fullmatch(task["id"])
    ]
    assert confirmatory
    for task in confirmatory:
        for candidate in task["candidates"]:
            key = (
                "ready_files_remote"
                if candidate["resource"] == "Robot-North-H20"
                else "ready_files"
            )
            paths = candidate.get(key, [])
            assert any(path.endswith("/params/_METADATA") for path in paths)
            assert any(path.endswith("/_CHECKPOINT_METADATA") for path in paths)
    shared = [task for task in queue["tasks"] if "_eval_attach_cnsh_g" in task["id"]]
    north = [
        task for task in queue["tasks"] if task["id"].endswith("_eval_attach_bj2g")
    ]
    assert len(shared) == 11
    assert len(north) == 5
    assert sum(task["candidates"][0]["gpus"] for task in shared) == 44
    assert sum(task["candidates"][0]["gpus"] for task in north) == 10

    for task in shared:
        assert any(
            path.endswith("/_CHECKPOINT_METADATA") for path in task["ready_files"]
        )
        assert task["ready_any"]
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "gf1",
            "robot-task",
        ]
        assert task["candidates"][0]["gpu_indices"] == [4, 5, 6, 7]

    a0_shared = [task for task in shared if "pi05_a0_s1001" in task["id"]]
    assert len(a0_shared) == 5
    for task in a0_shared:
        assert (
            "ATTACH_RUN_TAG_PREFIX=confirmatory-seed"
            in task["candidates"][0]["command"]
        )
        assert (
            task["candidates"][1]["env"]["ATTACH_RUN_TAG_PREFIX"] == "confirmatory-seed"
        )

    safety = [task for task in queue["tasks"] if task["id"].endswith("_safety_gf1")]
    assert len(safety) == 4
    assert sorted(task["candidates"][0]["gpu_indices"] for task in safety) == [
        [0],
        [1],
        [2],
        [3],
    ]
    for task in safety:
        assert any(
            path.endswith("/_CHECKPOINT_METADATA") for path in task["ready_files"]
        )
        assert task["progress_logs"][0]["total"] == 50
    for task in north:
        assert any(
            path.endswith("/_CHECKPOINT_METADATA")
            for path in task["ready_files_remote"]
        )
        assert task["ready_any"]
        assert task["candidates"][0]["max_failures"] >= 6

    for label in ("a2_s1001", "a2_s1002"):
        task = next(task for task in north if label in task["id"])
        assert any(
            all(
                "unseen-hint-seed" in path for path in alternative["ready_files_remote"]
            )
            for alternative in task["ready_any"]
        )

    attach_script = (
        scheduler.REPO
        / "train_scripts/kai/eval/attach_pi05_a0_confirmatory_platform.sh"
    ).read_text()
    assert 'if [ -f "$REPO/lmvla/lmwam/env/heal_lawam_symlinks.sh" ]' in attach_script
    assert (
        'if [ -f "$REPO/lmvla/lmwam/env/prepare_robotwin_renderer.sh" ]'
        in attach_script
    )
    assert "ATTACH_RUN_TAG_PREFIX" in attach_script

    resume_launchers = [
        scheduler.REPO / "train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh",
        scheduler.REPO
        / "train_scripts/kai/volc/pi05_robotwin_eval_confirmatory_east_4h20.yaml",
        scheduler.REPO
        / "train_scripts/kai/volc/pi05_robotwin_eval_a0_exact_cnsh_4a100.yaml",
        scheduler.REPO
        / "train_scripts/kai/volc/pi05_robotwin_eval_a0_official_x4_bj.yaml",
        scheduler.REPO
        / "train_scripts/kai/volc/pi05_robotwin_eval_hint_official_x4_bj.yaml",
    ]
    for launcher in resume_launchers:
        text = launcher.read_text()
        assert "ROBOTWIN_ATTACH_SCHEDULER=1" in text
        assert "scheduler_paths=" in text
        assert "ambiguous schedulers" in text


def test_full_runtime_queue_validates_after_dynamic_task_injection() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_shared_eval_attach_tasks(queue)
    scheduler.add_pi05_mt_eval_attach_tasks(queue)
    scheduler.add_pi05_mt1_replication_eval_attach_tasks(queue)
    scheduler.add_pi05_mt1_replication_north_overflow(queue)
    scheduler.add_pi05_mt1_8g_optimization_probes(queue)
    scheduler.add_pi05_r1_recurrence_aligned_tasks(queue)
    scheduler.add_pi05_r2_adaptive_execution_tasks(queue)
    scheduler.add_pi05_north_eval_attach_tasks(queue)
    scheduler.add_pi05_step40000_safety_probes(queue)
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    scheduler.add_pi05_mt4_replication_tasks(queue)
    scheduler.add_pi05_mt5_tasks(queue)
    scheduler.add_pi05_mt6_scope_task(queue)
    scheduler.add_pi05_mt6_efficiency_task(queue)
    scheduler.add_pi05_mt6_train_memory_task(queue)
    scheduler.add_pi05_mt3_eval_attach_tasks(queue)
    scheduler.validate_queue(queue)


def test_validation_does_not_require_atomic_sentinel_for_parameter_only_base() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.validate_queue(queue)

    task = next(
        task
        for task in queue["tasks"]
        if task["id"] == "pi05_mt1_oracle_seed1001_train"
    )
    assert (
        str(scheduler.REPO / "kai0/checkpoints/pi05_base/params/_METADATA")
        in task["ready_files"]
    )
    assert (
        str(scheduler.REPO / "kai0/checkpoints/pi05_base/_CHECKPOINT_METADATA")
        not in task["ready_files"]
    )


def test_mt_training_completion_requires_atomic_checkpoint_commit() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_mt3_tracker_tasks(queue)
    scheduler.add_pi05_mt4_replication_tasks(queue)
    scheduler.add_pi05_mt5_tasks(queue)

    training_tasks = [
        task
        for task in queue["tasks"]
        if task["id"].startswith("pi05_mt")
        and task["id"].endswith("_train")
        and "/checkpoints/" in task["completion_glob"]
    ]
    assert training_tasks
    assert all(
        task["completion_glob"].endswith("/49999/_CHECKPOINT_METADATA")
        for task in training_tasks
    )


def test_mt1_replication_evals_wait_for_atomic_checkpoint_commit() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    for seed in (1001, 1002):
        task = next(
            task
            for task in queue["tasks"]
            if task["id"] == f"pi05_mt1_oracle_seed{seed}_correct_eval"
        )
        checkpoint = (
            scheduler.REPO
            / "kai0/checkpoints/pi05_robotwin_mt1_oracle_exact"
            / f"pi05_robotwin_mt1_oracle_seed{seed}/49999"
        )
        assert task["ready_files"] == [
            str(checkpoint / "params/_METADATA"),
            str(checkpoint / "_CHECKPOINT_METADATA"),
            str(checkpoint / "train_state/_METADATA"),
            str(checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"),
            str(
                scheduler.REPO
                / "logs/resource_markers"
                / f"pi05_mt1_oracle_seed{seed}_step49999_checkpoint_audit.ok"
            ),
        ]


def test_mt1_replication_checkpoint_audit_requires_complete_nonempty_artifact(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(scheduler, "REPO", tmp_path)
    monkeypatch.setattr(scheduler, "LOG_PATH", tmp_path / "scheduler.log")
    checkpoint = (
        tmp_path
        / "kai0/checkpoints/pi05_robotwin_mt1_oracle_exact"
        / "pi05_robotwin_mt1_oracle_seed1001/40000"
    )
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "train_state").mkdir()
    norm = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    norm.parent.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("root")
    (checkpoint / "params/_METADATA").write_text("params")
    (checkpoint / "train_state/_METADATA").write_text("state")
    assert scheduler.audit_pi05_mt1_replication_checkpoint(1001, 40000) is None

    norm.write_text('{"robotwin2.0_absolute_meanstd": {}}')
    marker = scheduler.audit_pi05_mt1_replication_checkpoint(1001, 40000)
    assert marker is not None and marker.is_file()
    audit = json.loads(
        (
            tmp_path / "logs/checkpoint_audits/pi05_mt1_oracle_seed1001_step40000.json"
        ).read_text()
    )
    assert audit["accepted"] is True
    assert audit["step"] == 40000
    assert audit["checkpoint_bytes"] > 0
    assert set(audit["sha256"]) == {
        "root_metadata",
        "params_metadata",
        "train_state_metadata",
        "norm_stats",
    }


def test_crave_r0_rollout_collection_is_gated_and_materialized_from_north() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    tasks = {task["id"]: task for task in queue["tasks"]}

    collection = tasks["pi05_crave_r0_rollout_collection"]
    assert len(collection["ready_any"]) == 2
    gate_files = {
        alternative["ready_files"][0] for alternative in collection["ready_any"]
    }
    assert gate_files == {
        str(scheduler.REPO / "logs/predictive/p0_eval/p0_gate.accepted"),
        str(scheduler.REPO / "logs/predictive/p0_eval/p0_gate.rejected"),
    }
    assert {candidate["resource"] for candidate in collection["candidates"]} == {
        "local",
        "Robot-North-H20",
        "Robot-East-H20",
    }
    assert all(candidate["gpus"] == 2 for candidate in collection["candidates"])
    assert any(location["remote"] for location in collection["completion_locations"])
    assert {
        item["label"]: item["expected"] for item in collection["progress_globs"]
    } == {"summaries": 12, "videos": 120}

    materialize = tasks["pi05_crave_r0_rollout_sync_from_north"]
    assert materialize["materialize_north_result_for"] == collection["id"]
    assert materialize["candidates"][0]["gpus"] == 0
    assert (
        "sync_pi05_crave_r0_rollouts_from_north.sh"
        in materialize["candidates"][0]["command"]
    )

    features = tasks["pi05_crave_r0_rollout_features"]
    assert collection["completion_glob"] in features["ready_files"]
    assert all(candidate["gpus"] == 2 for candidate in features["candidates"])
    analysis = tasks["pi05_crave_r0_rollout_analysis"]
    assert features["completion_glob"] in analysis["ready_files"]
    assert (
        str(scheduler.REPO / "lmvla/lmwm/data/pi05_crave_r0_v1/READY_LABELS")
        in analysis["ready_files"]
    )
    assert all(candidate["gpus"] == 1 for candidate in analysis["candidates"])

    vocabulary = tasks["pi05_r3_semantic_vocabulary"]
    assert (
        str(
            scheduler.REPO
            / "lmvla/lmwm/data/pi05_crave_r0_v1/reference_trajectories.npz"
        )
        in vocabulary["ready_files"]
    )
    assert vocabulary["candidates"][0]["gpus"] == 0
    assert (
        "build_pi05_r3_semantic_vocabulary.py" in vocabulary["candidates"][0]["command"]
    )

    r3_conditions = {
        "semantic_next",
        "generic_stage",
        "semantic_current",
        "shuffled_semantic",
        "no_subtask",
    }
    for condition in r3_conditions:
        screen = tasks[f"pi05_r3_{condition}_screen"]
        assert vocabulary["completion_glob"] in screen["ready_files"]
        assert (
            str(scheduler.REPO / "logs/resource_markers/pi05_r3_north_sync.ok")
            in screen["ready_files"]
        )
        assert {candidate["resource"] for candidate in screen["candidates"]} == {
            "gf1",
            "Robot-East-H20",
            "Robot-North-H20",
            "robot-task",
        }
        assert all(candidate["gpus"] == 4 for candidate in screen["candidates"])
        materialize = tasks[f"pi05_r3_{condition}_materialize_north"]
        assert materialize["materialize_north_result_for"] == screen["id"]
        assert materialize["candidates"][0]["gpus"] == 0

    r3_gate = tasks["pi05_r3_semantic_screen_gate"]
    assert len(r3_gate["ready_files"]) == 6
    assert r3_gate["candidates"][0]["gpus"] == 0
    assert "analyze_pi05_r3_semantic_screen.py" in r3_gate["candidates"][0]["command"]


def test_r2_adaptive_execution_dag_is_gated_and_resource_ordered() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_r2_adaptive_execution_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    readout = tasks["pi05_r2_causal_readout"]
    assert (
        str(scheduler.REPO / "lmvla/lmwm/data/pi05_crave_r0_v1/READY_LABELS")
        in readout["ready_files"]
    )
    assert [candidate["resource"] for candidate in readout["candidates"]] == [
        "local",
        "gf1",
        "Robot-East-H20",
        "robot-task",
    ]

    sync = tasks["pi05_r2_north_sync"]
    assert (
        str(
            scheduler.REPO
            / "lmvla/lmwm/data/pi05_r2_causal_readout_v1/r2_readout.accepted"
        )
        in sync["ready_files"]
    )
    assert sync["candidates"][0]["gpus"] == 0

    expected_resources = [
        "gf1",
        "gf1",
        "Robot-East-H20",
        "Robot-North-H20",
        "robot-task",
    ]
    for condition in ("fixed4", "adaptive"):
        screen = tasks[f"pi05_r2_{condition}_screen"]
        assert [
            candidate["resource"] for candidate in screen["candidates"]
        ] == expected_resources
        assert all(candidate["gpus"] == 4 for candidate in screen["candidates"])
        assert (
            str(scheduler.REPO / "logs/resource_markers/pi05_r2_north_sync.ok")
            in screen["ready_files"]
        )
        materialize = tasks[f"pi05_r2_{condition}_materialize_north"]
        assert materialize["materialize_north_result_for"] == screen["id"]

    gate = tasks["pi05_r2_adaptive_screen_gate"]
    assert (
        str(scheduler.REPO / "lmvla/lmwm/docs/pi05_r2_fixed4_screen_v1.json")
        in gate["ready_files"]
    )
    assert (
        str(scheduler.REPO / "lmvla/lmwm/docs/pi05_r2_adaptive_screen_v1.json")
        in gate["ready_files"]
    )
    assert "--fixed-report" in gate["candidates"][0]["command"]
    assert "--adaptive-report" in gate["candidates"][0]["command"]


def test_r2_rejected_readout_disables_all_execution_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = (
        tmp_path
        / "lmvla/lmwm/data/pi05_r2_causal_readout_v1/r2_readout.rejected"
    )
    rejected.parent.mkdir(parents=True)
    rejected.write_text('{"accepted": false}\n')
    monkeypatch.setattr(scheduler, "REPO", tmp_path)

    queue = {"tasks": []}
    scheduler.add_pi05_r2_adaptive_execution_tasks(queue)
    tasks = {task["id"]: task for task in queue["tasks"]}

    assert "enabled" not in tasks["pi05_r2_causal_readout"]
    downstream = {
        "pi05_r2_north_sync",
        "pi05_r2_fixed4_screen",
        "pi05_r2_fixed4_materialize_north",
        "pi05_r2_adaptive_screen",
        "pi05_r2_adaptive_materialize_north",
        "pi05_r2_adaptive_screen_gate",
    }
    assert downstream <= tasks.keys()
    for task_id in downstream:
        assert tasks[task_id]["enabled"] is False
        assert tasks[task_id]["disabled_reason"] == "R2 causal readout gate rejected"


def test_r1_recurrence_aligned_dag_is_double_gated_and_disjoint() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())
    scheduler.add_pi05_r1_recurrence_aligned_tasks(queue)
    scheduler.add_pi05_r1_recurrence_aligned_tasks(queue)
    scheduler.validate_queue(queue)
    tasks = {
        task["id"]: task for task in queue["tasks"] if task["id"].startswith("pi05_r1_")
    }
    assert len(tasks) == 24

    p0_gate = str(scheduler.REPO / "logs/predictive/p0_eval/p0_gate.accepted")
    r0_gate = str(scheduler.REPO / "logs/crave_r0/probe_gate/r0_gate.accepted")
    crave = tasks["pi05_r1_crave_seed1000_train"]
    combined = tasks["pi05_r1_combined_seed1000_train"]
    for task in (crave, combined):
        assert p0_gate in task["ready_files"]
        assert r0_gate in task["ready_files"]
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "gf1",
            "Robot-East-H20",
            "robot-task",
        ]
    assert crave["candidates"][0]["gpu_indices"] == [0, 1, 2, 3]
    assert combined["candidates"][0]["gpu_indices"] == [4, 5, 6, 7]
    for arm in ("crave", "combined"):
        label = f"pi05_r1_{arm}_seed1000"
        assert label in scheduler.EAST_TRAIN_WATCH_TASKS
        assert scheduler.TRAIN_WATCH_MANAGED_TASK_IDS[
            ("Robot-East-H20", label)
        ] == f"pi05_r1_{arm}_seed1000_train"

    for condition in (
        "crave",
        "combined",
        "combined_zero_gate",
        "combined_shuffled",
    ):
        task = tasks[f"pi05_r1_{condition}_seed1000_eval"]
        assert [candidate["resource"] for candidate in task["candidates"]] == [
            "local",
            "gf1",
            "Robot-East-H20",
            "robot-task",
        ]
        assert task["candidates"][0]["gpus"] == 2
        assert task["candidates"][0]["gpu_indices"] == [0, 1]
        assert "MAX_PARALLEL_SEEDS=2" in task["candidates"][0]["command"]
        assert "R1_VERIFY_REPO=" in task["candidates"][0]["command"]
        assert "PYTHONPATH=" in task["candidates"][0]["command"]
        assert task["candidates"][2]["env"]["R1_VERIFY_REPO"] == str(
            scheduler.R1_FROZEN_OVERLAY
        )
        assert all(candidate["gpus"] == 4 for candidate in task["candidates"][1:])
        assert "pi05_r1_protocol_v1.json" in " ".join(task["ready_files"])
    gate = tasks["pi05_r1_seed1000_gate"]
    assert len(gate["ready_files"]) == 7
    assert "zero_route=" in gate["candidates"][0]["command"]
    assert "shuffled_action=" in gate["candidates"][0]["command"]

    replication_gate = str(scheduler.REPO / "logs/r1/seed1000/r1_gate.accepted")
    rejection_reason = (
        "R1 seed-1000 necessary comparison rejected: combined is significantly "
        "worse than CRAVE-only"
    )
    for seed in (1001, 1002):
        for arm in ("a0", "predictive", "crave", "combined"):
            train = tasks[f"pi05_r1_{arm}_seed{seed}_train"]
            evaluate = tasks[f"pi05_r1_{arm}_seed{seed}_eval"]
            assert train["enabled"] is False
            assert evaluate["enabled"] is False
            assert train["disabled_reason"] == rejection_reason
            assert evaluate["disabled_reason"] == rejection_reason
            assert replication_gate in train["ready_files"]
            assert replication_gate in evaluate["ready_files"]
            assert [candidate["resource"] for candidate in train["candidates"]] == [
                "gf1",
                "Robot-East-H20",
                "robot-task",
            ]
            assert [candidate["resource"] for candidate in evaluate["candidates"]] == [
                "local",
                "gf1",
                "Robot-East-H20",
                "robot-task",
            ]
            assert evaluate["candidates"][0]["gpus"] == 2
            assert "MAX_PARALLEL_SEEDS=2" in evaluate["candidates"][0]["command"]
            assert all(candidate["gpus"] == 4 for candidate in evaluate["candidates"][1:])
            assert f"SEED={seed}" in evaluate["candidates"][0]["command"]
            if arm == "predictive":
                p2_task_id = f"pi05_predictive_adapter_p2_candidate_seed{seed}_train"
                p2_train = next(
                    task for task in queue["tasks"] if task["id"] == p2_task_id
                )
                assert train["completion_glob"] == p2_train["completion_glob"]
                assert train["hold_retry_while_running"] == [p2_task_id]
                assert p2_train["hold_retry_while_running"] == [train["id"]]

    final_gate = tasks["pi05_r1_three_seed_gate"]
    assert final_gate["enabled"] is False
    assert final_gate["disabled_reason"] == rejection_reason
    assert len(final_gate["ready_files"]) == 14
    assert "1002:combined=" in final_gate["candidates"][0]["command"]


def test_staged_failover_statuses_reads_local_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "status": "SYNCING",
                "progress_fraction": 0.25,
                "rate_bytes_per_second": 5_000_000,
                "eta_seconds": 3600,
                "stage_verified": False,
                "launch_authorized": False,
            }
        )
    )
    monkeypatch.setattr(scheduler, "P1_NORTH_FAILOVER_PROGRESS_PATH", progress)
    monkeypatch.setattr(
        scheduler,
        "P1_NORTH_FAILOVER_AUTH_AUDIT_PATH",
        tmp_path / "missing_authorization.json",
    )

    statuses = scheduler.staged_failover_statuses()

    assert statuses["pi05_p1_north"]["status"] == "SYNCING"
    assert statuses["pi05_p1_north"]["progress_fraction"] == 0.25
    assert statuses["pi05_p1_north"]["rate_bytes_per_second"] == 5_000_000
    assert statuses["pi05_p1_north"]["eta_seconds"] == 3600
    assert statuses["pi05_p1_north"]["launch_authorized"] is False


def test_staged_failover_statuses_applies_authorization_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"status": "VERIFIED", "launch_authorized": False}))
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"launch_authorized": True}))
    monkeypatch.setattr(scheduler, "P1_NORTH_FAILOVER_PROGRESS_PATH", progress)
    monkeypatch.setattr(
        scheduler, "P1_NORTH_FAILOVER_AUTH_AUDIT_PATH", authorization
    )

    status = scheduler.staged_failover_statuses()["pi05_p1_north"]

    assert status["launch_authorized"] is True
    assert status["authorization_audit"] == str(authorization)


def test_p1_north_failover_pair_is_audited_and_materialized() -> None:
    queue = {"tasks": []}

    scheduler.add_pi05_p1_north_failover_tasks(queue)
    scheduler.add_pi05_p1_north_failover_tasks(queue)

    tasks = {task["id"]: task for task in queue["tasks"]}
    assert set(tasks) == {
        "pi05_p1_north_failover_pair",
        "pi05_p1_north_failover_materialize",
    }
    parent = tasks["pi05_p1_north_failover_pair"]
    assert parent["prefer_min_gpus_when_immediate"] is True
    assert parent["completion_remote"] is True
    assert parent["completion_min_count"] == 2
    assert parent["completion_glob"].endswith(
        "pi05_predictive_adapter_p1*_seed1000/49999/_CHECKPOINT_METADATA"
    )
    assert len(parent["candidates"]) == 2
    recovery = parent["candidates"][0]
    assert recovery["resource"] == "Robot-North-H20"
    assert recovery["gpus"] == 4
    assert recovery["ready_files_remote"] == [
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "kai0/checkpoints/pi05_predictive_adapter_p1"
            / "pi05_predictive_adapter_p1_seed1000/49999/_CHECKPOINT_METADATA"
        ),
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "kai0/checkpoints/pi05_predictive_adapter_p1"
            / "pi05_predictive_adapter_p1_seed1000/49999/params/_METADATA"
        ),
    ]
    candidate = parent["candidates"][1]
    assert candidate["resource"] == "Robot-North-H20"
    assert candidate["gpus"] == 8
    assert candidate["max_failures"] == 6
    assert candidate["env"]["PATH"].startswith(
        f"{scheduler.P1_NORTH_FAILOVER_STAGE}/runtime/bin:"
    )
    assert candidate["env"]["LD_LIBRARY_PATH"] == (
        f"{scheduler.P1_NORTH_FAILOVER_STAGE}/runtime/lib"
    )
    assert candidate["env"]["PYTHONPATH"] == (
        f"{scheduler.P1_NORTH_FAILOVER_STAGE}/kai0/src"
    )
    assert parent["ready_hashes"][0]["path"].endswith(
        "north_container_runtime_amendment.json"
    )
    assert candidate["ready_files_remote"] == [
        str(scheduler.P1_NORTH_FAILOVER_STAGE / "north_stage_report.json"),
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "pi05_p1_north_runtime_preflight.json"
        ),
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "pi05_p1_north_failover_authorization.json"
        ),
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "logs/pi05_p1_failover/authorization_audit.json"
        ),
        str(
            scheduler.P1_NORTH_FAILOVER_STAGE
            / "north_container_runtime_amendment.json"
        ),
    ]
    materialize = tasks["pi05_p1_north_failover_materialize"]
    assert materialize["materialize_north_result_for"] == parent["id"]
    assert materialize["completion_glob"].endswith(
        "logs/resource_markers/pi05_p1_north_failover_materialized.ok"
    )
    assert materialize["completion_glob"] in materialize["produces_files"]
    assert materialize["candidates"][0]["gpus"] == 0
    assert materialize["candidates"][0]["resource"] == "local"


def test_p1_north_eval_is_staged_hash_gated_and_materialized() -> None:
    queue = json.loads(scheduler.QUEUE_PATH.read_text())

    scheduler.add_pi05_p1_north_eval_tasks(queue)
    scheduler.add_pi05_p1_north_eval_tasks(queue)
    scheduler.add_pi05_p1_a0_seed01_east_helper_task(queue)
    scheduler.add_pi05_p1_a0_seed01_east_helper_task(queue)
    scheduler.add_pi05_p1_a0_east_accelerator_task(queue)
    scheduler.add_pi05_p1_a0_east_accelerator_task(queue)
    scheduler.apply_frozen_source_readiness(queue)
    scheduler.validate_queue(queue)

    tasks = {task["id"]: task for task in queue["tasks"]}
    stage = tasks["pi05_p1_north_eval_stage"]
    assert stage["completion_glob"] == str(scheduler.P1_NORTH_EVAL_STAGE_MARKER)
    assert stage["candidates"][0]["gpus"] == 0
    assert "sync_pi05_p1_eval_runtime_to_north.sh" in stage["candidates"][0][
        "command"
    ]

    for condition in ("a0", "normal", "zero_gate", "shuffled", "masked"):
        parent_id = f"pi05_predictive_adapter_p1_{condition}_seed1000_eval"
        parent = tasks[parent_id]
        north = [
            candidate
            for candidate in parent["candidates"]
            if candidate["resource"] == "Robot-North-H20"
        ]
        assert len(north) == 1
        candidate = north[0]
        assert candidate["gpus"] == 4
        assert candidate["ready_files"] == [
            str(scheduler.P1_NORTH_EVAL_STAGE_MARKER)
        ]
        assert str(scheduler.P1_NORTH_EVAL_STAGE_MARKER_REMOTE) in candidate[
            "ready_files_remote"
        ]
        assert candidate["env"]["P1_VERIFY_REPO"] == str(
            scheduler.P1_NORTH_EVAL_OVERLAY
        )
        assert candidate["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
        assert {location["label"] for location in parent["completion_locations"]} == {
            "shared",
            "north",
        }

        materialize = tasks[f"pi05_p1_{condition}_eval_materialize_north"]
        assert materialize["materialize_north_result_for"] == parent_id
        assert materialize["candidates"][0]["gpus"] == 0
        assert "sync_pi05_p1_eval_from_north.sh" in materialize["candidates"][0][
            "command"
        ]

    for condition, priority in (("a0", 0), ("shuffled", 1), ("zero_gate", 2)):
        accelerator = tasks[f"pi05_p1_{condition}_north_accelerator"]
        assert accelerator["priority"] == priority
        assert accelerator["completion_locations"][0]["remote"] is True
        assert accelerator["completion_locations"][0]["glob"].endswith(
            f"pi05_p1_{condition}_north_accelerator.ok"
        )
        assert accelerator["completion_locations"][1]["remote"] is False
        assert accelerator["completion_locations"][1]["glob"].endswith(
            f"pi05_predictive_adapter_p1_seed1000_{condition}.ok"
        )
        candidate = accelerator["candidates"][0]
        assert candidate["resource"] == "Robot-North-H20"
        assert candidate["gpus"] == 4
        assert candidate["env"]["PREDICTIVE_P1_CONDITION"] == condition
        assert candidate["yaml"].endswith(
            "pi05_p1_eval_accelerator_north_4h20.yaml"
        )
        assert sum(
            path.endswith(".task_scheduler.json")
            for path in candidate["ready_files_remote"]
        ) == 4
        if condition == "a0":
            assert "p1_a0_exact" in candidate["env"]["CKPT"]
        else:
            assert "p1_a0_exact" not in candidate["env"]["CKPT"]
        assert any(
            item["path"].endswith(
                "pi05_p1_north_accelerator_amendment_v1.json"
            )
            for item in accelerator["ready_hashes"]
        )

    tail = tasks["pi05_p1_a0_north_tail_accelerator"]
    assert tail["priority"] == 0
    assert {item["label"] for item in tail["completion_locations"]} == {
        "north",
        "canonical",
    }
    tail_candidate = tail["candidates"][0]
    assert tail_candidate["resource"] == "Robot-North-H20"
    assert tail_candidate["gpus"] == 4
    assert tail_candidate["max_failures"] == 1
    assert tail_candidate["env"]["PORT_BASE_OFFSET"] == "26600"
    assert tail_candidate["yaml"].endswith(
        "pi05_p1_a0_tail_accelerator_north_4h20.yaml"
    )
    assert sum(
        path.endswith(".task_scheduler.json")
        for path in tail_candidate["ready_files_remote"]
    ) == 4
    assert any(
        item["path"].endswith(
            "pi05_p1_a0_north_tail_accelerator_amendment_v2.json"
        )
        for item in tail["ready_hashes"]
    )
    tail_script = (
        scheduler.REPO
        / "train_scripts/kai/eval/run_pi05_p1_a0_north_tail_accelerator.sh"
    ).read_text()
    assert "pending == 0" in tail_script
    assert "skipped=no_pending_cells" in tail_script
    assert "pending > 4" in tail_script

    a0_seed01_helper = tasks["pi05_p1_a0_seed01_east_helper"]
    assert a0_seed01_helper["priority"] == 0
    assert {
        item["label"] for item in a0_seed01_helper["completion_locations"]
    } == {"helper", "canonical"}
    assert sum(
        path.endswith(".task_scheduler.json")
        for path in a0_seed01_helper["ready_files"]
    ) == 2
    assert str(scheduler.R1_FROZEN_OVERLAY / "READY") in a0_seed01_helper[
        "ready_files"
    ]
    assert not any(
        path.startswith(str(scheduler.P1_NORTH_EVAL_OVERLAY))
        for path in a0_seed01_helper["ready_files"]
    )
    helper_candidate = a0_seed01_helper["candidates"][0]
    assert helper_candidate["resource"] == "Robot-East-H20"
    assert helper_candidate["gpus"] == 4
    assert helper_candidate["yaml"].endswith(
        "pi05_p1_a0_seed01_helper_east_4h20.yaml"
    )
    assert any(
        item["path"].endswith(
            "pi05_p1_a0_seed01_east_helper_amendment_v1.json"
        )
        for item in a0_seed01_helper["ready_hashes"]
    )

    a0_accelerator = tasks["pi05_p1_a0_east_accelerator"]
    assert a0_accelerator["priority"] == 0
    assert {item["label"] for item in a0_accelerator["completion_locations"]} == {
        "accelerator",
        "canonical",
        "local_accelerator",
    }
    assert sum(
        path.endswith(".task_scheduler.json")
        for path in a0_accelerator["ready_files"]
    ) == 2
    assert str(scheduler.R1_FROZEN_OVERLAY / "READY") in a0_accelerator[
        "ready_files"
    ]
    assert not any(
        path.startswith(str(scheduler.P1_NORTH_EVAL_OVERLAY))
        for path in a0_accelerator["ready_files"]
    )
    candidates = {
        candidate["resource"]: candidate
        for candidate in a0_accelerator["candidates"]
    }
    assert set(candidates) == {"Robot-East-H20", "local"}
    east_candidate = candidates["Robot-East-H20"]
    assert east_candidate["gpus"] == 4
    assert east_candidate["yaml"].endswith(
        "pi05_p1_a0_accelerator_east_4h20.yaml"
    )
    local_candidate = candidates["local"]
    assert local_candidate["gpus"] == 2
    assert local_candidate["gpu_indices"] == [0, 1]
    assert "run_pi05_p1_a0_local_accelerator.sh" in local_candidate["command"]
    assert any(
        item["path"].endswith(
            "pi05_p1_a0_accelerator_amendment_v2.json"
        )
        for item in a0_accelerator["ready_hashes"]
    )

    east_accelerator = tasks["pi05_p1_a0_east_secondary_accelerator"]
    assert east_accelerator["priority"] == 0
    assert {
        item["label"] for item in east_accelerator["completion_locations"]
    } == {"east_accelerator", "canonical"}
    assert sum(
        path.endswith(".task_scheduler.json")
        for path in east_accelerator["ready_files"]
    ) == 2
    assert len(east_accelerator["candidates"]) == 1
    east_candidate = east_accelerator["candidates"][0]
    assert east_candidate["resource"] == "Robot-East-H20"
    assert east_candidate["gpus"] == 4
    assert east_candidate["task_name"] == "pi05-p1-a0-secondary-attach-east4g"


def test_completion_locations_require_artifacts_even_without_completion_glob(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "accelerator.ok"
    task = {
        "id": "attach_only_accelerator",
        "completion_locations": [
            {"label": "accelerator", "glob": str(marker), "remote": False}
        ],
        "completion_min_count": 1,
    }

    complete, evidence = scheduler.completion_evidence(task)

    assert complete is False
    assert evidence == "completion artifacts accelerator=0/1"

    marker.write_text("ok\n")
    complete, evidence = scheduler.completion_evidence(task)

    assert complete is True
    assert evidence == "completion artifacts accelerator=1/1"


def test_running_task_polls_remote_completion_before_local_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_marker = tmp_path / "canonical.ok"
    task = {
        "id": "north_attach_helper",
        "completion_locations": [
            {"label": "north", "glob": "/north/helper.ok", "remote": True},
            {"label": "canonical", "glob": str(local_marker), "remote": False},
        ],
        "completion_min_count": 1,
    }
    attempt = {
        "kind": "platform",
        "job_id": "t-remote",
        "region": "cn-beijing",
        "last_state": "Queueing",
    }
    state = {"status": "running", "attempts": [attempt]}
    probes = []
    stopped = []
    monkeypatch.setattr(
        scheduler,
        "completion_evidence",
        lambda _task: probes.append(_task["id"])
        or (True, "completion artifacts north=1/1, canonical=0/1"),
    )
    monkeypatch.setattr(
        scheduler, "stop_managed_attempt", lambda value: stopped.append(value)
    )

    scheduler.check_managed_task(task, state)

    assert probes == [task["id"]]
    assert stopped == [attempt]
    assert state["status"] == "completed"
    assert attempt["stopped_after_completion_artifact"]


def test_load_state_reopens_terminal_fallback_without_declared_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    marker = tmp_path / "accelerator.ok"
    state_path.write_text(
        json.dumps(
            {
                "tasks": {
                    "attach_only_accelerator": {
                        "status": "completed",
                        "attempts": [{"resource": "local"}],
                        "artifacts_complete": True,
                        "artifact_progress": "platform terminal state",
                        "completed_at": "2026-08-05T18:20:50Z",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(scheduler, "STATE_PATH", state_path)
    queue = {
        "tasks": [
            {
                "id": "attach_only_accelerator",
                "completion_locations": [
                    {
                        "label": "accelerator",
                        "glob": str(marker),
                        "remote": False,
                    }
                ],
                "completion_min_count": 1,
                "candidates": [],
            }
        ]
    }

    state = scheduler.load_state(queue)
    task_state = state["tasks"]["attach_only_accelerator"]

    assert task_state["status"] == "pending"
    assert task_state["artifacts_complete"] is False
    assert "completed_at" not in task_state
    assert "completion_misclassification_repaired" in task_state["attempts"][-1]


def test_load_state_reopens_report_only_p1_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    marker = tmp_path / "materialized.ok"
    state_path.write_text(
        json.dumps(
            {
                "tasks": {
                    "pi05_p1_north_failover_materialize": {
                        "status": "completed",
                        "completed_at": "2026-08-05T13:03:17Z",
                        "artifacts_complete": True,
                        "attempts": [{"resource": "local"}],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(scheduler, "STATE_PATH", state_path)
    queue = {
        "tasks": [
            {
                "id": "pi05_p1_north_failover_materialize",
                "enabled": True,
                "completion_glob": str(marker),
                "completion_min_count": 1,
            }
        ]
    }

    state = scheduler.load_state(queue)

    task_state = state["tasks"]["pi05_p1_north_failover_materialize"]
    assert task_state["status"] == "pending"
    assert task_state["artifacts_complete"] is False
    assert "completed_at" not in task_state
    assert task_state["attempts"][-1]["completion_misclassification_repaired"]
