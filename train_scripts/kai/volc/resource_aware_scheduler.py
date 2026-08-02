#!/usr/bin/env python3
"""Resource-aware dispatcher for the remaining RoboTwin experiment queue.

The dispatcher keeps work local until a target has real capacity. It monitors
Volc queues, gf1 and the two-GPU development host, enforces the 20-GPU Beijing
limit and the physical 32-GPU robot-task capacity, and records an atomic
state/snapshot under ``logs/``.
"""

from __future__ import annotations

import argparse
import configparser
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


REPO = Path(__file__).resolve().parents[3]
QUEUE_PATH = REPO / "train_scripts/kai/volc/resource_scheduler_queue.json"
STATE_PATH = REPO / "logs/resource_scheduler_state.json"
SNAPSHOT_PATH = REPO / "logs/resource_scheduler_snapshot.json"
SNAPSHOT_MARKDOWN_PATH = REPO / "logs/resource_scheduler_snapshot.md"
LOG_PATH = REPO / "logs/resource_scheduler.log"
LOCK_PATH = REPO / "logs/resource_scheduler.lock"
OWNER = "trn:iam::2113249311:user/suiyang.guo"
NORTH_QUEUE = "q-20260516104642-khch9"
SH_QUEUE = "q-20251204185107-fvnpx"
EAST_QUEUE = "q-20260516104437-2ml4v"
SH_CAPACITY = 32
NORTH_CAPACITY = 56
NORTH_PERSONAL_LIMIT = 20
NORTH_BACKUP_PERSONAL_LIMIT = int(os.environ.get("NORTH_BACKUP_PERSONAL_LIMIT", "20"))
BACKUP_CREDENTIALS_PATH = Path(
    os.environ.get(
        "VOLC_BACKUP_CREDENTIALS_FILE",
        "~/.volc/credentials.scheduler-backup",
    )
).expanduser()
BACKUP_CONTROL_PATH = Path(
    os.environ.get(
        "VOLC_BACKUP_CONTROL_FILE",
        "~/.volc/scheduler-backup.conf",
    )
).expanduser()
SH_PERSONAL_LIMIT = int(os.environ.get("SH_PERSONAL_LIMIT", str(SH_CAPACITY)))
SH_MIN_DISPATCH_FREE = int(os.environ.get("SH_MIN_DISPATCH_FREE", "0"))
RETRY_COOLDOWN_SECONDS = 900
MAX_FAILURES_PER_RESOURCE = 3
RESOURCE_DISPATCH_PRIORITY = {
    "gf1": 0,
    "Robot-East-H20": 1,
    "Robot-North-H20": 2,
    "robot-task": 3,
}
PI05_CONFIRMATORY_EVAL_RE = re.compile(
    r"pi05_(a0_public_exact|a2_abs_confirmatory|a3_live_confirmatory)_seed100[012]_eval"
)
PI05_CONFIRMATORY_SCENE_MANIFEST_SHARED = (
    "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/"
    "robotwin_pi05_confirmatory_scene_seeds_v1.json"
)
PI05_CONFIRMATORY_SCENE_MANIFEST_NORTH = (
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lmwm/data/"
    "robotwin_pi05_confirmatory_scene_seeds_v1.json"
)
PI05_CONFIRMATORY_FIXED_SEED_MAX_ATTEMPTS = "500"
GF1 = ["ssh", "-p", "7777", "-o", "BatchMode=yes", "root@14.103.218.231"]
GSY = ["ssh", "-p", "16370", "-o", "BatchMode=yes", "root@124.174.16.237"]
GF1_WATCH_TASKS = {
    "a3_official_all6_eval": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/logs/"
            "local_rteval/gf1_a3_8g/status"
        ),
        "artifact_glob": (
            "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/results/eval_runs/"
            "robotwin/pi05_rt_a3_live_residual_official_gf1_8g/**/summary.json"
        ),
        "expected_artifacts": 24,
    },
}
GF1_TRAIN_WATCH_TASKS = {
    "residual_seed2027": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/logs/"
            "local_train/gf1_residual_s2027_8g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam/logs/"
            "local_train/gf1_residual_s2027_8g/launcher.log"
        ),
        "expected_steps": 20000,
    },
    "pi05_a0_public_exact": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_a0_public_exact_seed1000_4g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_a0_public_exact_seed1000_4g/launcher.log"
        ),
        "expected_steps": 50000,
    },
    "pi05_a0_public_exact_seed1001": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_a0_public_exact_seed1001_4g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_a0_public_exact_seed1001_4g/launcher.log"
        ),
        "expected_steps": 50000,
    },
}
LOCAL_WATCH_TASKS: dict[str, dict[str, Any]] = {}
PLATFORM_TRAIN_WATCH_TASKS = {
    variant: {
        "log_glob": REPO
        / f"lmvla/lawam/logs/volc_robotwin/all6_v2_{variant}_seed2027_*.log",
        "expected_steps": 20000,
    }
    for variant in ("nowm", "absolute", "isolation")
}
EAST_TRAIN_WATCH_TASKS = {
    "pi05_a2_abs_seed1000": {
        "log_globs": [
            REPO / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_confirmatory_20260801_224940.log",
            REPO / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_seed1000_*.log",
        ],
        "expected_steps": 50000,
    },
    "pi05_a3_live_seed1000": {
        "log_globs": [
            REPO / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_confirmatory_20260801_225018.log",
            REPO / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_seed1000_*.log",
        ],
        "expected_steps": 50000,
    },
}
NORTH_TRAIN_WATCH_TASKS = {
    variant: {
        "log_glob": (
            "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/"
            f"volc_robotwin/all6_v2_{variant}_seed2028_*.log"
        ),
        "expected_steps": 20000,
    }
    for variant in ("nowm", "combo")
}
NORTH_TRAIN_WATCH_TASKS["pi05_a0_public_recipe"] = {
    "log_glob": (
        "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/"
        "volc_robotwin/pi05_robotwin_a0_official_bj_*.log"
    ),
    "expected_steps": 50000,
}
NORTH_TRAIN_WATCH_TASKS.update(
    {
        "pi05_a0_public_exact_seed1002": {
            "log_globs": [
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a0_official_bj_20260801_232341.log",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a0_public_exact_seed1002_*.log",
            ],
            "expected_steps": 50000,
        },
        "pi05_a2_abs_seed1001": {
            "log_globs": [
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_confirmatory_20260801_232118.log",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_seed1001_*.log",
            ],
            "expected_steps": 50000,
        },
        "pi05_a2_abs_seed1002": {
            "log_globs": [
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_confirmatory_20260801_232452.log",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_seed1002_*.log",
            ],
            "expected_steps": 50000,
        },
        "pi05_a3_live_seed1001": {
            "log_globs": [
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_confirmatory_20260801_232229.log",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_seed1001_*.log",
            ],
            "expected_steps": 50000,
        },
        "pi05_a3_live_seed1002": {
            "log_globs": [
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_confirmatory_20260801_232609.log",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_seed1002_*.log",
            ],
            "expected_steps": 50000,
        },
    }
)
for fold in (0, 1):
    NORTH_TRAIN_WATCH_TASKS[f"heldout_fold{fold}"] = {
        "log_glob": (
            "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lmwm/logs/"
            f"robotwin_lmwm_heldout_fold{fold}_*.log"
        ),
        "expected_steps": 5000,
    }
TRAIN_WATCH_MANAGED_TASK_IDS = {
    ("Beijing", "combo"): "combo_seed2028_train",
    ("Beijing", "nowm"): "nowm_seed2028_train",
    ("Beijing", "pi05_a0_public_recipe"): "pi05_a0_public_recipe_seed1000_train",
    ("robot-task", "nowm"): "nowm_seed2027_train",
    ("robot-task", "absolute"): "absolute_seed2027_train",
    ("robot-task", "isolation"): "isolation_seed2027_train",
    ("gf1", "residual_seed2027"): "residual_seed2027_train",
    ("gf1", "pi05_a0_public_exact"): "pi05_a0_public_exact_seed1000_train",
    ("gf1", "pi05_a0_public_exact_seed1001"): "pi05_a0_public_exact_seed1001_train",
    ("Robot-East-H20", "pi05_a2_abs_seed1000"): "pi05_a2_abs_confirmatory_seed1000_train",
    ("Robot-East-H20", "pi05_a3_live_seed1000"): "pi05_a3_live_confirmatory_seed1000_train",
    ("Beijing", "pi05_a0_public_exact_seed1002"): "pi05_a0_public_exact_seed1002_train",
    ("Beijing", "pi05_a2_abs_seed1001"): "pi05_a2_abs_confirmatory_seed1001_train",
    ("Beijing", "pi05_a2_abs_seed1002"): "pi05_a2_abs_confirmatory_seed1002_train",
    ("Beijing", "pi05_a3_live_seed1001"): "pi05_a3_live_confirmatory_seed1001_train",
    ("Beijing", "pi05_a3_live_seed1002"): "pi05_a3_live_confirmatory_seed1002_train",
    ("Beijing", "heldout_fold0"): "lawam_heldout_predictor_fold0",
    ("Beijing", "heldout_fold1"): "lawam_heldout_predictor_fold1",
}
NORTH_WATCH_TASKS = {
    "pi05_a2_residual": ("pi05_rt_a2_residual_prefix_official_v4", 24),
    "pi05_a0": ("pi05_rt_a0_official_v2", 24),
    "pi05_a2_absolute": ("pi05_rt_a2_prefix_official_v2", 24),
    "pi05_a1": ("pi05_rt_a1_prefix_official_v2", 24),
    "absolute_zero_hint": ("rt_all6_v2_absolute_zerohint_seed2026_unseen", 12),
    "absolute_shuffled_hint": ("rt_all6_v2_absolute_shuffledhint_seed2026_unseen", 12),
    "absolute_other_task_hint": ("rt_all6_v2_absolute_othertask_seed2026_unseen", 12),
    "residual_zero_hint": ("rt_all6_v2_residual_zerohint_seed2026_unseen", 12),
    "residual_shuffled_hint": ("rt_all6_v2_residual_shuffledhint_seed2026_unseen", 12),
    "residual_other_task_hint": ("rt_all6_v2_residual_othertask_seed2026_unseen", 12),
}
SHARED_EVAL_WATCH_TASKS = {
    "local_seed2027": ("rt_all6_v2_local_seed2027_unseen", 24),
    "combo_seed2027": ("rt_all6_v2_combo_seed2027_unseen", 24),
}
ACTIVE_STATES = ("Running", "Deploying")
TERMINAL_STATES = {"Completed", "Success", "Failed", "Stopped"}
GPU_BY_FLAVOR = {
    "ml.hpcpni3ln.45xlarge": 8,
    "ml.pni3ln.45xlarge": 8,
    "ml.pni3ln.35xlarge": 8,
    "ml.pni3ln.22xlarge": 4,
    "ml.pni3ln.17xlarge": 4,
    "ml.pni3ln.11xlarge": 2,
    "ml.pni3ln.8xlarge": 2,
    "ml.pni3ln.5xlarge": 1,
    "ml.pni3ln.4xlarge": 1,
    "ml.hpcpni2.28xlarge": 8,
    "ml.pni2.14xlarge": 4,
    "ml.pni2.7xlarge": 2,
}
UNKNOWN_GPU_FLAVOR_FALLBACK = 8
UNKNOWN_GPU_FLAVORS_WARNED: set[str] = set()
QUEUE_CONFIG = {
    "Robot-North-H20": {"region": "cn-beijing", "id": NORTH_QUEUE, "capacity": None},
    "Robot-East-H20": {"region": "cn-shanghai", "id": EAST_QUEUE, "capacity": 8},
    "robot-task": {"region": "cn-shanghai", "id": SH_QUEUE, "capacity": SH_CAPACITY},
}
NORTH_REPO = "/vePFS-North-E/vis_robot/workspace/deepdive_kai0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
CAUSAL_REPORTS = {
    label: {
        "correct": f"lmvla/lawam/results/eval_runs/robotwin/rt_all6_v2_{variant}_seed2026_unseen",
        "controls": {
            "zero": f"lmvla/lawam/results/eval_runs/robotwin/rt_all6_v2_{variant}_zerohint_seed2026_unseen",
            "shuffled": f"lmvla/lawam/results/eval_runs/robotwin/rt_all6_v2_{variant}_shuffledhint_seed2026_unseen",
            "other-task": f"lmvla/lawam/results/eval_runs/robotwin/rt_all6_v2_{variant}_othertask_seed2026_unseen",
        },
        "output": REPO / f"logs/robotwin_{label}_causal_final.json",
    }
    for label, variant in (("residual", "residual"), ("combo", "combo"))
}
CAUSAL_REPORTS["pi05_a2"] = {
    "correct": "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_prefix_official_v2",
    "controls": {
        "current": "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_current_hint_causal",
        "zero": "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_zero_hint_causal",
        "shuffled": "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_shuffled_hint_causal",
        "other-task": "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_other_task_hint_causal",
    },
    "output": REPO / "logs/pi05_a2_causal_final.json",
}
EVAL_REPORTS = {
    "rt_all6_v2_combo_oracle_retrieval_seed2026_unseen": {
        "root": (
            "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
            "results/eval_runs/robotwin/"
            "rt_all6_v2_combo_oracle_retrieval_seed2026_unseen"
        ),
        "expected": 24,
        "remote": True,
    },
    "pi05_public_samebridge_4seed_v3": {
        "root": (
            "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
            "results/eval_runs/robotwin/pi05_public_samebridge_4seed_v3"
        ),
        "expected": 24,
        "remote": True,
    },
    "pi05_rt_a0_public_recipe_seed1000": {
        "root": (
            "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
            "results/eval_runs/robotwin/pi05_rt_a0_public_recipe_seed1000"
        ),
        "expected": 24,
        "remote": True,
    },
    "pi05_rt_a0_public_exact_seed1000": {
        "expected": 24,
        "locations": [
            {
                "root": str(
                    REPO
                    / "lmvla/lawam/results/eval_runs/robotwin/"
                    "pi05_rt_a0_public_exact_seed1000"
                ),
                "remote": False,
            },
            {
                "root": (
                    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
                    "results/eval_runs/robotwin/pi05_rt_a0_public_exact_seed1000"
                ),
                "remote": True,
            },
        ],
    },
    **{
        label: {
            "expected": 12,
            "locations": [
                {
                    "root": str(REPO / "lmvla/lawam/results/eval_runs/robotwin" / label),
                    "remote": False,
                },
                {
                    "root": (
                        "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
                        f"results/eval_runs/robotwin/{label}"
                    ),
                    "remote": True,
                },
            ],
        }
        for label in (
            "pi05_rt_a2_current_hint_causal",
            "pi05_rt_a2_zero_hint_causal",
            "pi05_rt_a2_shuffled_hint_causal",
            "pi05_rt_a2_other_task_hint_causal",
            "pi05_rt_a2_instance_shuffle_causal",
        )
    },
    **{
        label: {
            "expected": 12,
            "locations": [
                {
                    "root": str(REPO / "lmvla/lawam/results/eval_runs/robotwin" / label),
                    "remote": False,
                },
                {
                    "root": (
                        "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
                        f"results/eval_runs/robotwin/{label}"
                    ),
                    "remote": True,
                },
            ],
        }
        for label in (
            "pi05_rt_a3_current_hint_causal",
            "pi05_rt_a3_zero_hint_causal",
            "pi05_rt_a3_shuffled_hint_causal",
            "pi05_rt_a3_other_task_hint_causal",
            "pi05_rt_a3_instance_shuffle_causal",
        )
    },
    **{
        f"rt_all6_v2_{variant}_seed2026_unseen": {
            "root": (
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
                "results/eval_runs/robotwin/"
                + (
                    "rt_all6_v2_nowm_v2_seed2026_unseen"
                    if variant == "nowm"
                    else f"rt_all6_v2_{variant}_seed2026_unseen"
                )
            ),
            "expected": 24,
            "remote": True,
        }
        for variant in ("nowm", "local", "absolute", "residual", "isolation", "combo")
    },
    **{
        f"rt_all6_v2_{variant}_seed2027_unseen": {
            "root": str(
                REPO
                / "lmvla/lawam/results/eval_runs/robotwin"
                / f"rt_all6_v2_{variant}_seed2027_unseen"
            ),
            "expected": 24,
            "remote": False,
        }
        for variant in ("nowm", "local", "absolute", "residual", "isolation", "combo")
    },
    **{
        f"rt_all6_v2_{variant}_seed2028_unseen": {
            "root": (
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
                f"results/eval_runs/robotwin/rt_all6_v2_{variant}_seed2028_unseen"
            ),
            "expected": 24,
            "remote": True,
        }
        for variant in ("nowm", "combo")
    },
    **{
        f"rt_all6_v2_{variant}_seed2026_unseen": {
            "root": str(
                REPO
                / "lmvla/lawam/results/eval_runs/robotwin"
                / f"rt_all6_v2_{variant}_seed2026_unseen"
            ),
            "expected": 24,
            "remote": False,
        }
        for variant in ("nowm_resetflow", "neverwm")
    },
}

for method, result_template in (
    ("a0", "pi05_rt_a0_public_exact_seed{seed}"),
    ("a2_abs", "pi05_rt_a2_abs_confirmatory_s{seed}"),
    ("a3_live", "pi05_rt_a3_live_confirmatory_s{seed}"),
):
    for training_seed in (1000, 1001, 1002):
        label = result_template.format(seed=training_seed)
        EVAL_REPORTS[label] = {
            "expected": 24,
            "method": method,
            "training_seed": training_seed,
            "locations": [
                {
                    "root": str(
                        REPO / "lmvla/lawam/results/eval_runs/robotwin" / label
                    ),
                    "remote": False,
                },
                {
                    "root": (
                        "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/"
                        f"lmvla/lawam/results/eval_runs/robotwin/{label}"
                    ),
                    "remote": True,
                },
            ],
        }

# Volc OpenAPI is directly reachable from the development host. Inheriting a
# stale local proxy (commonly 127.0.0.1:7890) makes SDK polling intermittently
# time out and delays task reclamation.
for proxy_key in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ.pop(proxy_key, None)

TRACKED_JOBS = {
    "t-20260731223709-nlv5k": ("cn-beijing", "no-WM seed2026 eval"),
    "t-20260731221831-4wgst": ("cn-beijing", "A2 residual official eval"),
    "t-20260731225645-h49n6": ("cn-beijing", "local-WM seed2026 eval"),
    "t-20260731230433-npm45": ("cn-beijing", "A0 official eval"),
    "t-20260731230549-tvdn5": ("cn-beijing", "A2 absolute official eval"),
    "t-20260731230904-mgksx": ("cn-beijing", "A1 official eval"),
    "t-20260731230929-6xszg": ("cn-beijing", "slots=1 probe"),
    "t-20260731230950-r6wkh": ("cn-beijing", "slots=2 probe"),
    "t-20260731094624-ktxv7": ("cn-shanghai", "combo seed2027 train"),
    "t-20260731231804-wh7l8": ("cn-shanghai", "local-WM seed2027 eval"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def validate_queue(queue: dict[str, Any]) -> None:
    """Reject queue edits that would silently invalidate confirmatory evidence."""
    tasks = queue.get("tasks", [])
    task_ids = [task.get("id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("resource queue contains duplicate task ids")

    tasks_by_id = {task["id"]: task for task in tasks}
    eval_sidecars = {
        "pi05_a2_abs_confirmatory_eval.json": (
            "pi05_robotwin_a2_prefix_official_eval_bj"
        ),
        "pi05_a3_live_confirmatory_eval.json": (
            "pi05_robotwin_a3_live_residual_prefix_official_eval"
        ),
    }
    for filename, expected_base in eval_sidecars.items():
        sidecar = REPO / "train_scripts/kai/volc/config_overrides" / filename
        spec = json.loads(sidecar.read_text())
        if spec.get("base_config_name") != expected_base:
            raise ValueError(f"{filename} must clone inference config {expected_base}")
        if spec.get("override_asset_id") != "robotwin2.0_absolute_meanstd":
            raise ValueError(f"{filename} must select mean/std confirmation assets")
    for task in tasks:
        if not PI05_CONFIRMATORY_EVAL_RE.fullmatch(task["id"]):
            continue
        for candidate in task.get("candidates", []):
            manifest = (
                PI05_CONFIRMATORY_SCENE_MANIFEST_NORTH
                if candidate.get("resource") == "Robot-North-H20"
                else PI05_CONFIRMATORY_SCENE_MANIFEST_SHARED
            )
            fixed_seed_env = {
                "ROBOTWIN_EPISODE_SEED_MANIFEST": manifest,
                "ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS": (
                    PI05_CONFIRMATORY_FIXED_SEED_MAX_ATTEMPTS
                ),
            }
            candidate.setdefault("env", {}).update(fixed_seed_env)
            if candidate.get("kind") in {"ssh", "local"}:
                prefix = " ".join(
                    f"{key}={shlex.quote(value)}" for key, value in fixed_seed_env.items()
                )
                if "ROBOTWIN_EPISODE_SEED_MANIFEST=" not in candidate["command"]:
                    candidate["command"] = f"env {prefix} {candidate['command']}"
            expected = {
                "ROBOTWIN_EPISODE_SEED_MANIFEST": manifest,
                "ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS": "500",
            }
            if any(candidate["env"].get(key) != value for key, value in expected.items()):
                raise ValueError(f"{task['id']} fixed-scene protocol is incomplete")

    seed1001_a0 = tasks_by_id.get("pi05_a0_public_exact_seed1001_eval", {})
    gf1_seed1001 = [
        candidate
        for candidate in seed1001_a0.get("candidates", [])
        if candidate.get("resource") == "gf1"
    ]
    if len(gf1_seed1001) != 1 or gf1_seed1001[0].get("gpu_indices") != [4, 5, 6, 7]:
        raise ValueError("A0 seed1001 gf1 evaluation must reserve GPUs 4-7")
    if "GPU_INDEX_OFFSET=4" not in gf1_seed1001[0].get("command", ""):
        raise ValueError("A0 seed1001 gf1 command must execute on GPUs 4-7")
    for seed in (1000, 1001, 1002):
        task_id = f"pi05_a2_abs_confirmatory_seed{seed}_eval"
        task = tasks_by_id.get(task_id)
        if task is None:
            raise ValueError(f"confirmatory queue is missing {task_id}")
        for candidate in task.get("candidates", []):
            if candidate.get("resource") in {"Robot-North-H20", "Robot-East-H20"}:
                env = candidate.get("env", {})
                expected_yaml = (
                    "pi05_robotwin_eval_hint_official_x4_bj.yaml"
                    if candidate.get("resource") == "Robot-North-H20"
                    else "pi05_robotwin_eval_confirmatory_east_4h20.yaml"
                )
                if not str(candidate.get("yaml", "")).endswith(expected_yaml):
                    raise ValueError(
                        f"{task_id} {candidate['resource']} candidate is not hint-enabled"
                    )
                required = {
                    "ROBOTWIN_EVAL_CONFIG": "pi05_robotwin_a2_prefix_official_eval_bj",
                    "ROBOTWIN_HINT_ENCODER": "so400m",
                    "OPENPI_SERVER_HINT_ENCODER": "so400m",
                    "EVAL_HINT_RESIDUAL": "0",
                    "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
                    "OPENPI_EXTRA_CONFIG": (
                        f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                        "pi05_a2_abs_confirmatory_eval.json"
                        if candidate.get("resource") == "Robot-North-H20"
                        else f"{REPO}/train_scripts/kai/volc/config_overrides/"
                        "pi05_a2_abs_confirmatory_eval.json"
                    ),
                }
                if any(env.get(key) != value for key, value in required.items()):
                    raise ValueError(
                        f"{task_id} {candidate['resource']} hint protocol is incomplete"
                    )
            elif candidate.get("resource") == "gf1":
                command = candidate.get("command", "")
                required_tokens = (
                    "PI05_EVAL_CONFIG_NAME=pi05_robotwin_a2_prefix_official_eval_bj",
                    "OPENPI_SERVER_HINT_ENCODER=so400m",
                    "EVAL_HINT_RESIDUAL=0",
                    "PI05_ASSET_ID=robotwin2.0_absolute_meanstd",
                    "OPENPI_EXTRA_CONFIG=/vePFS/tim/workspace/deepdive_kai0/"
                    "train_scripts/kai/volc/config_overrides/"
                    "pi05_a2_abs_confirmatory_eval.json",
                )
                if any(token not in command for token in required_tokens):
                    raise ValueError(f"{task_id} gf1 hint protocol is incomplete")

    for seed in (1000, 1001, 1002):
        task_id = f"pi05_a3_live_confirmatory_seed{seed}_eval"
        task = tasks_by_id.get(task_id)
        if task is None:
            raise ValueError(f"confirmatory queue is missing {task_id}")
        for candidate in task.get("candidates", []):
            expected_config = "pi05_robotwin_a3_live_residual_prefix_official_eval"
            if candidate.get("kind") == "platform":
                env = candidate.get("env", {})
                expected_sidecar = (
                    f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json"
                    if candidate.get("resource") == "Robot-North-H20"
                    else f"{REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json"
                )
                if env.get("PI05_EVAL_CONFIG_NAME") != expected_config:
                    raise ValueError(
                        f"{task_id} {candidate['resource']} uses a training-time A3 config"
                    )
                if env.get("OPENPI_EXTRA_CONFIG") != expected_sidecar:
                    raise ValueError(f"{task_id} {candidate['resource']} A3 sidecar mismatch")
            else:
                command = candidate.get("command", "")
                required_tokens = (
                    f"PI05_EVAL_CONFIG_NAME={expected_config}",
                    "OPENPI_EXTRA_CONFIG=/vePFS/tim/workspace/deepdive_kai0/"
                    "train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json",
                )
                if any(token not in command for token in required_tokens):
                    raise ValueError(
                        f"{task_id} {candidate['resource']} A3 inference protocol mismatch"
                    )


def load_state(queue: dict[str, Any]) -> dict[str, Any]:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {"tasks": {}, "created_at": utc_now()}
    queue_task_ids = {task["id"] for task in queue["tasks"]}
    for retired_task_id in set(state["tasks"]) - queue_task_ids:
        state["tasks"].pop(retired_task_id)
    for task in queue["tasks"]:
        task_state = state["tasks"].setdefault(
            task["id"], {"status": "pending", "attempts": []}
        )
        if not task.get("enabled", True):
            task_state["status"] = "disabled"
        elif task_state.get("status") == "disabled":
            task_state["status"] = "pending"
    return state


def credential_values(profile: str = "primary") -> tuple[str, str]:
    if profile == "primary":
        return os.environ["VOLC_AK"], os.environ["VOLC_SK"]
    if profile != "backup":
        raise ValueError(f"unknown credential profile: {profile}")
    stat = BACKUP_CREDENTIALS_PATH.stat()
    if stat.st_mode & 0o077:
        raise PermissionError(
            f"backup credential file must be mode 0600: {BACKUP_CREDENTIALS_PATH}"
        )
    parser = configparser.ConfigParser()
    parser.read(BACKUP_CREDENTIALS_PATH)
    section = parser["default"]
    return section["access_key_id"].strip(), section["secret_access_key"].strip()


def backup_credentials_configured() -> bool:
    try:
        ak, sk = credential_values("backup")
    except (OSError, KeyError, configparser.Error, PermissionError):
        return False
    return bool(ak and sk)


def backup_credentials_enabled() -> bool:
    try:
        parser = configparser.ConfigParser()
        if not parser.read(BACKUP_CONTROL_PATH):
            return False
        return parser.getboolean("scheduler", "enabled", fallback=False)
    except (OSError, configparser.Error, ValueError):
        return False


def api(region: str, profile: str = "primary") -> Service:
    ak, sk = credential_values(profile)
    info = ServiceInfo(
        "open.volcengineapi.com",
        {"Accept": "application/json"},
        Credentials(ak, sk, "ml_platform", region),
        10,
        10,
    )
    return Service(
        info,
        {
            "ListJobs": ApiInfo("POST", "/", {"Action": "ListJobs", "Version": "2024-07-01"}, {}, {}),
            "GetJob": ApiInfo("POST", "/", {"Action": "GetJob", "Version": "2024-07-01"}, {}, {}),
            "StopJob": ApiInfo("POST", "/", {"Action": "StopJob", "Version": "2024-07-01"}, {}, {}),
        },
    )


SERVICES: dict[tuple[str, str], Service] = {}


def service(region: str, profile: str = "primary") -> Service:
    return SERVICES.setdefault((profile, region), api(region, profile))


def job_gpus(job: dict[str, Any]) -> int:
    total = 0
    for role in job.get("ResourceConfig", {}).get("Roles", []):
        flavor = role.get("Resource", {}).get("InstanceTypeId", "")
        if flavor not in GPU_BY_FLAVOR:
            if flavor and flavor not in UNKNOWN_GPU_FLAVORS_WARNED:
                log(
                    f"unknown GPU flavor {flavor}; conservatively counting "
                    f"{UNKNOWN_GPU_FLAVOR_FALLBACK} GPUs per replica"
                )
                UNKNOWN_GPU_FLAVORS_WARNED.add(flavor)
            gpus_per_replica = UNKNOWN_GPU_FLAVOR_FALLBACK if flavor else 0
        else:
            gpus_per_replica = GPU_BY_FLAVOR[flavor]
        total += gpus_per_replica * int(role.get("Replicas", 1))
    return total


def list_jobs(region: str, queue_id: str, profile: str = "primary") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in (*ACTIVE_STATES, "Queueing"):
        body = {"ResourceQueueId": queue_id, "PageSize": 100, "State": state}
        raw = service(region, profile).json("ListJobs", {}, json.dumps(body).encode())
        result = json.loads(raw).get("Result", {})
        for job in result.get("Items", result.get("List", [])):
            job = dict(job)
            job["_state"] = state
            job["_gpus"] = job_gpus(job)
            rows.append(job)
    return rows


def get_job(region: str, job_id: str, profile: str = "primary") -> dict[str, Any]:
    raw = service(region, profile).json(
        "GetJob", {}, json.dumps({"Id": job_id}).encode()
    )
    result = json.loads(raw).get("Result", {})
    status = result.get("Status") or {}
    return {
        "id": job_id,
        "name": result.get("Name"),
        "state": status.get("State") or result.get("State"),
        "message": status.get("Message") or "",
        "created_at": result.get("CreateTime"),
        "updated_at": result.get("UpdateTime"),
        "gpus": job_gpus(result),
    }


def run(
    command: list[str],
    *,
    timeout: int = 60,
    env_overrides: dict[str, str] | None = None,
) -> str:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    return subprocess.check_output(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=env,
    )


def ssh(command: list[str], script: str, *, timeout: int = 60) -> str:
    return run([*command, script], timeout=timeout)


def gpu_snapshot(command: list[str] | None = None) -> dict[str, Any]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = run(query, timeout=20) if command is None else ssh(command, " ".join(query), timeout=20)
    rows = []
    for line in output.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        rows.append({"index": values[0], "memory_used_mib": values[1], "memory_total_mib": values[2], "utilization": values[3]})
    return {
        "gpus": rows,
        "count": len(rows),
        "free_count": sum(row["memory_used_mib"] < 1024 for row in rows),
        "available": True,
    }


def safe_gpu_snapshot(
    command: list[str] | None, expected_count: int, label: str
) -> dict[str, Any]:
    try:
        return gpu_snapshot(command)
    except Exception as exc:
        log(f"resource probe unavailable {label}: {type(exc).__name__}: {exc}")
        return {
            "gpus": [],
            "count": expected_count,
            "free_count": 0,
            "available": False,
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def gf1_watched_statuses() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for label, config in GF1_WATCH_TASKS.items():
        path = config["status_path"]
        pattern = config["artifact_glob"]
        program = f"import glob; print(len(glob.glob({pattern!r}, recursive=True)))"
        try:
            status = ssh(
                GF1,
                f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || echo MISSING",
                timeout=20,
            ).strip()
            artifact_count = int(
                ssh(GF1, f"python3 -c {shlex.quote(program)}", timeout=20).strip()
            )
            statuses[label] = {
                "status": status,
                "artifact_count": artifact_count,
                "expected_artifacts": config["expected_artifacts"],
            }
        except Exception as exc:
            statuses[label] = {
                "status": "MONITOR_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return statuses


def gf1_training_statuses() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    progress_pattern = re.compile(
        r"(\d+)(?:it)?/(\d+).*?(?:rate:)?([0-9.]+)(it/s|s/it)"
    )
    separator = "__RESOURCE_SCHEDULER_LOG_TAIL__"
    for label, config in GF1_TRAIN_WATCH_TASKS.items():
        status_path = config["status_path"]
        log_path = config["log_path"]
        try:
            script = (
                f"test -f {shlex.quote(status_path)} && cat {shlex.quote(status_path)} || echo MISSING; "
                f"printf '\n{separator}\n'; "
                f"test -f {shlex.quote(log_path)} && tail -c 1048576 {shlex.quote(log_path)} || true"
            )
            response = None
            for attempt in range(2):
                try:
                    response = ssh(GF1, script, timeout=30)
                    break
                except Exception:
                    if attempt == 0:
                        time.sleep(1)
                    else:
                        raise
            assert response is not None
            remote_status, log_tail = response.split(f"\n{separator}\n", 1)
            remote_status = remote_status.strip()
            log_tail = log_tail.replace("\r", "\n")
            log_tail = re.sub(
                r"([0-9.]+)([kM])it",
                lambda match: str(
                    int(float(match[1]) * {"k": 1_000, "M": 1_000_000}[match[2]])
                ),
                log_tail,
            )
            matches = list(progress_pattern.finditer(log_tail))
            if not matches:
                statuses[label] = {"status": remote_status, "progress": "INITIALIZING"}
                continue
            match = matches[-1]
            step = int(match.group(1))
            expected = int(match.group(2))
            rate_value = float(match.group(3))
            seconds_per_step = (
                1.0 / rate_value if match.group(4) == "it/s" else rate_value
            )
            statuses[label] = {
                "status": remote_status,
                "step": step,
                "expected_steps": expected,
                "seconds_per_step": seconds_per_step,
                "eta_hours": round(max(0, expected - step) * seconds_per_step / 3600, 2),
            }
        except Exception as exc:
            statuses[label] = {
                "status": "MONITOR_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return statuses


def local_watched_statuses() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for label, config in LOCAL_WATCH_TASKS.items():
        path = config["log_path"]
        if not path.is_file():
            statuses[label] = {"status": "MISSING", "log_path": str(path)}
            continue
        text = path.read_text(errors="ignore")[-20000:]
        matches = list(re.finditer(r"step (\d+)/(\d+).*?rate=([0-9.]+)it/s.*?eta=([^\s]+)", text))
        if not matches:
            statuses[label] = {"status": "NO_PROGRESS_LINE", "log_path": str(path)}
            continue
        match = matches[-1]
        statuses[label] = {
            "status": "RUNNING",
            "step": int(match.group(1)),
            "expected_steps": int(match.group(2)),
            "rate_it_s": float(match.group(3)),
            "eta": match.group(4),
            "log_mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    return statuses


def platform_training_statuses(
    watch_tasks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    progress_pattern = re.compile(
        r"([0-9.]+)(?:it)?/([0-9.]+).*?([0-9.]+)(it/s|s/it)"
    )
    for label, config in (watch_tasks or PLATFORM_TRAIN_WATCH_TASKS).items():
        patterns = config.get("log_globs", [config.get("log_glob")])
        paths = sorted(
            {
                path
                for pattern in patterns
                if pattern
                for path in glob.glob(str(pattern))
            },
            key=os.path.getmtime,
        )
        if not paths:
            statuses[label] = {"status": "WAITING_FOR_LOG"}
            continue
        path = Path(paths[-1])
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 1024 * 1024))
            text_tail = stream.read().decode("utf-8", errors="replace").replace("\r", "\n")
        text_tail = re.sub(
            r"([0-9.]+)([kM])it",
            lambda match: str(
                int(float(match[1]) * {"k": 1_000, "M": 1_000_000}[match[2]])
            ),
            text_tail,
        )
        matches = list(progress_pattern.finditer(text_tail))
        if not matches:
            statuses[label] = {"status": "INITIALIZING", "log_path": str(path)}
            continue
        match = matches[-1]
        step = int(float(match.group(1)))
        expected = int(float(match.group(2)))
        rate_value = float(match.group(3))
        seconds_per_step = (
            1.0 / rate_value if match.group(4) == "it/s" else rate_value
        )
        age_seconds = max(0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
        statuses[label] = {
            "status": "RUNNING" if age_seconds < 300 else "STALE_LOG",
            "step": step,
            "expected_steps": expected,
            "seconds_per_step": seconds_per_step,
            "eta_hours": round(max(0, expected - step) * seconds_per_step / 3600, 2),
            "log_mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    return statuses


def north_training_statuses() -> dict[str, Any]:
    program = f"""
import calendar, glob, json, os, re, time
tasks = {NORTH_TRAIN_WATCH_TASKS!r}
pattern = re.compile(
    r'([0-9.]+)(?:it)?/([0-9.]+).*?(?:rate:)?([0-9.]+)(it/s|s/it)'
)
simple_pattern = re.compile(r'step=(\\d+)/(\\d+)')
result = {{}}
for label, config in tasks.items():
    patterns = config.get('log_globs', [config.get('log_glob')])
    paths = sorted(
        {{path for pattern in patterns if pattern for path in glob.glob(pattern)}},
        key=os.path.getmtime,
    )
    if not paths:
        result[label] = {{'status': 'WAITING_FOR_LOG'}}
        continue
    path = paths[-1]
    with open(path, 'rb') as stream:
        stream.seek(max(0, os.path.getsize(path) - 1024 * 1024))
        tail = stream.read().decode('utf-8', errors='replace').replace('\\r', '\\n')
    tail = re.sub(
        r'([0-9.]+)([kM])it',
        lambda match: str(int(float(match[1]) * {{'k': 1_000, 'M': 1_000_000}}[match[2]])),
        tail,
    )
    matches = list(pattern.finditer(tail))
    if not matches:
        simple_matches = list(simple_pattern.finditer(tail))
        if not simple_matches:
            result[label] = {{'status': 'INITIALIZING', 'log_path': path}}
            continue
        match = simple_matches[-1]
        step, expected = int(float(match[1])), int(float(match[2]))
        stamp = re.search(r'_(\\d{{8}}_\\d{{6}})\\.log$', path)
        started_at = (
            calendar.timegm(time.strptime(stamp[1], '%Y%m%d_%H%M%S'))
            if stamp else os.path.getctime(path)
        )
        seconds_per_step = max(0.0, time.time() - started_at) / max(1, step)
    else:
        match = matches[-1]
        step, expected = int(float(match[1])), int(float(match[2]))
        rate_value = float(match[3])
        seconds_per_step = 1.0 / rate_value if match[4] == 'it/s' else rate_value
    age_seconds = max(0, time.time() - os.path.getmtime(path))
    result[label] = {{
        'status': 'RUNNING' if age_seconds < 300 else 'STALE_LOG',
        'step': step,
        'expected_steps': expected,
        'seconds_per_step': seconds_per_step,
        'eta_hours': round(max(0, expected - step) * seconds_per_step / 3600, 2),
        'log_mtime': os.path.getmtime(path),
    }}
print(json.dumps(result))
"""
    try:
        return json.loads(ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=45))
    except Exception as exc:
        return {"monitor_error": {"error": str(exc)}}


def north_watched_statuses() -> dict[str, Any]:
    base = (
        "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
        "results/eval_runs/robotwin"
    )
    program = f"""
import glob, json
base = {base!r}
tasks = {NORTH_WATCH_TASKS!r}
result = {{}}
for label, (result_name, expected) in tasks.items():
    root = base + '/' + result_name
    summaries = glob.glob(root + '/**/summary.json', recursive=True)
    seeds = {{}}
    for path in glob.glob(root + '/**/.task_scheduler.json', recursive=True):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        seed = path.split('/seed', 1)[1].split('/', 1)[0] if '/seed' in path else '?'
        seeds[seed] = {{
            'completed': len(data.get('completed', {{}})),
            'in_progress': list(data.get('in_progress', {{}})),
            'failed': list(data.get('failed', {{}})),
            'updated_at': data.get('updated_at'),
        }}
    result[label] = {{
        'artifact_count': len(summaries),
        'expected_artifacts': expected,
        'seeds': seeds,
    }}
print(json.dumps(result))
"""
    try:
        return json.loads(ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=45))
    except Exception as exc:
        return {"monitor_error": str(exc)}


def shared_eval_statuses() -> dict[str, Any]:
    base = REPO / "lmvla/lawam/results/eval_runs/robotwin"
    result: dict[str, Any] = {}
    for label, (result_name, expected) in SHARED_EVAL_WATCH_TASKS.items():
        root = base / result_name
        summaries = glob.glob(str(root / "**/summary.json"), recursive=True)
        seeds: dict[str, Any] = {}
        for path_text in glob.glob(str(root / "**/.task_scheduler.json"), recursive=True):
            path = Path(path_text)
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            match = re.search(r"/seed([^/]+)/", path_text)
            seed = match.group(1) if match else "?"
            seeds[seed] = {
                "completed": len(data.get("completed", {})),
                "in_progress": list(data.get("in_progress", {})),
                "failed": list(data.get("failed", {})),
                "updated_at": data.get("updated_at"),
            }
        result[label] = {
            "artifact_count": len(summaries),
            "expected_artifacts": expected,
            "seeds": seeds,
        }
    return result


def readiness_spec_satisfied(spec: dict[str, Any]) -> bool:
    if any(not Path(path).is_file() for path in spec.get("ready_files", [])):
        return False
    if any(not glob.glob(pattern) for pattern in spec.get("ready_globs", [])):
        return False
    remote_files = spec.get("ready_files_remote", [])
    if remote_files:
        checks = " && ".join(f"test -f {shlex.quote(path)}" for path in remote_files)
        try:
            ssh(GSY, checks, timeout=30)
        except Exception:
            return False
    remote_globs = spec.get("ready_globs_remote", [])
    if remote_globs:
        program = (
            "import glob,sys; "
            f"sys.exit(0 if all(glob.glob(p) for p in {remote_globs!r}) else 1)"
        )
        try:
            ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=30)
        except Exception:
            return False
    return True


def ready(task: dict[str, Any]) -> bool:
    if not readiness_spec_satisfied(task):
        return False
    alternatives = task.get("ready_any", [])
    return not alternatives or any(readiness_spec_satisfied(spec) for spec in alternatives)


def completion_evidence(task: dict[str, Any]) -> tuple[bool, str]:
    pattern = task.get("completion_glob")
    if not pattern:
        return True, "platform terminal state"
    minimum = int(task.get("completion_min_count", 1))
    locations = task.get("completion_locations") or [
        {
            "label": "remote" if task.get("completion_remote") else "local",
            "glob": pattern,
            "remote": bool(task.get("completion_remote")),
        }
    ]
    evidence = []
    complete = False
    for location in locations:
        location_pattern = location["glob"]
        label = location.get("label", "remote" if location.get("remote") else "local")
        if location.get("remote"):
            program = (
                "import glob; "
                f"print(len(glob.glob({location_pattern!r}, recursive=True)))"
            )
            try:
                count = int(
                    ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=30).strip()
                )
            except Exception as exc:
                evidence.append(f"{label}=error:{type(exc).__name__}")
                continue
        else:
            count = len(glob.glob(location_pattern, recursive=True))
        location_complete = count >= minimum
        verification = ""
        if location_complete and PI05_CONFIRMATORY_EVAL_RE.fullmatch(task["id"]):
            root = location_pattern.split("/**/", 1)[0]
            manifest = (
                PI05_CONFIRMATORY_SCENE_MANIFEST_NORTH
                if location.get("remote")
                else PI05_CONFIRMATORY_SCENE_MANIFEST_SHARED
            )
            verifier = (
                f"{NORTH_REPO}/lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py"
                if location.get("remote")
                else str(REPO / "lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py")
            )
            command = ["python3", verifier, "--manifest", manifest, "--root", root]
            try:
                if location.get("remote"):
                    ssh(GSY, shlex.join(command), timeout=180)
                else:
                    run(command, timeout=180)
                verification = ",fixed-seeds=verified"
            except Exception as exc:
                location_complete = False
                verification = f",fixed-seeds=error:{type(exc).__name__}"
        complete = complete or location_complete
        evidence.append(f"{label}={count}/{minimum}{verification}")
    return complete, "completion artifacts " + ", ".join(evidence)


def record_artifact_progress(
    task_state: dict[str, Any], complete: bool, evidence: str
) -> None:
    if evidence != task_state.get("artifact_progress") or not task_state.get(
        "artifact_progress_changed_at"
    ):
        task_state["artifact_progress_changed_at"] = utc_now()
        task_state.pop("artifact_stale_warning_at", None)
    task_state["artifact_progress"] = evidence
    task_state["artifact_progress_checked_at"] = utc_now()
    task_state["artifacts_complete"] = complete


def mark_task_completed(task: dict[str, Any], task_state: dict[str, Any]) -> None:
    task_state["status"] = "completed"
    task_state["completed_at"] = utc_now()
    marker = task.get("completion_marker")
    if marker:
        path = Path(marker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"completed={task_state['completed_at']}\ntask={task['id']}\n")


def refresh_causal_reports() -> None:
    """Materialize final shared-cohort reports as soon as all controls finish."""
    for label, config in CAUSAL_REPORTS.items():
        output = config["output"]
        if output.is_file():
            continue
        roots = [config["correct"], *config["controls"].values()]
        program = (
            "import glob,json; print(json.dumps(["
            + ",".join(
                f"len(glob.glob({(NORTH_REPO + '/' + root + '/**/summary.json')!r}, recursive=True))"
                for root in roots
            )
            + "]))"
        )
        try:
            counts = json.loads(
                ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=30)
            )
        except Exception:
            continue
        if counts[0] < 12 or any(count < 12 for count in counts[1:]):
            continue
        command = [
            "python3",
            "lmvla/lmwm/scripts/rt_causal_intervention_analysis.py",
            "--correct-root",
            config["correct"],
        ]
        for control, root in config["controls"].items():
            command.extend(["--control", f"{control}={root}"])
        command.append("--json")
        try:
            report = json.loads(
                ssh(
                    GSY,
                    f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}",
                    timeout=90,
                )
            )
        except Exception as exc:
            log(f"causal report generation failed {label}: {type(exc).__name__}: {exc}")
            continue
        atomic_json(output, report)
        log(f"materialized final causal report {label}: {output}")


def sync_north_eval_tree(label: str, expected: int) -> Path | None:
    """Copy a small North evaluation result tree when its summaries are complete."""
    eval_base = REPO / "lmvla/lawam/results/eval_runs/robotwin"
    local_root = eval_base / label
    if len(list(local_root.glob("**/summary.json"))) >= expected:
        return local_root
    remote_root = f"{NORTH_REPO}/lmvla/lawam/results/eval_runs/robotwin/{label}"
    try:
        remote_count = int(
            ssh(
                GSY,
                "python3 -c "
                + shlex.quote(
                    "import glob; print(len(glob.glob(" + repr(remote_root + "/**/summary.json")
                    + ", recursive=True)))"
                ),
                timeout=30,
            ).strip()
        )
    except Exception:
        return None
    if remote_count < expected:
        return None

    local_root.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [
                "scp",
                "-r",
                "-P",
                "16370",
                "-o",
                "BatchMode=yes",
                f"root@124.174.16.237:{remote_root}/seed*",
                str(local_root),
            ],
            timeout=300,
        )
    except Exception as exc:
        log(f"North eval result sync failed {label}: {type(exc).__name__}: {exc}")
        return None
    if len(list(local_root.glob("**/summary.json"))) < expected:
        log(f"North eval result sync incomplete {label}; waiting for the next poll")
        return None
    return local_root


def refresh_l2_strict_north_results() -> None:
    """Sync and verify strict residual controls produced on North vePFS."""
    eval_base = REPO / "lmvla/lawam/results/eval_runs/robotwin"
    manifest = (
        REPO
        / "lmvla/lmwm/data/robotwin_l2_seed_manifests/"
        "residual_correct_seed2026.json"
    )
    verifier = REPO / "lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py"
    marker_dir = REPO / "logs/resource_markers"
    controls = {
        "zero": [
            ("rt_all6_v2_residual_zerohint_seed2026_strict_unseen", 24),
        ],
        "within_task_shuffle": [
            ("rt_all6_v2_residual_shuffledhint_seed2026_strict_unseen", 12),
            ("rt_all6_v2_residual_instanceshuffle_seed2026_strict_unseen", 12),
        ],
    }
    for control, specs in controls.items():
        marker = marker_dir / f"l2_strict_residual_{control}.ok"
        if marker.is_file():
            continue
        roots: list[Path] = []
        for label, expected in specs:
            root = sync_north_eval_tree(label, expected)
            if root is None:
                break
            roots.append(root)
        if len(roots) != len(specs):
            continue
        command = [
            "python3",
            str(verifier),
            "--manifest",
            str(manifest),
        ]
        for root in roots:
            command.extend(["--root", str(root)])
        try:
            run(command, timeout=180)
        except Exception as exc:
            log(
                f"strict residual verification failed {control}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            f"completed={utc_now()} method=residual control={control} "
            f"source=north_verified\n"
        )
        log(f"synced and verified strict residual control: {control}")


def refresh_oracle_retrieval_report() -> None:
    """Sync retrieval outcomes and compare them with the frozen combo policy."""
    output = REPO / "logs/eval_reports/robotwin_combo_oracle_retrieval_paired.json"
    if output.is_file():
        return
    oracle = sync_north_eval_tree(
        "rt_all6_v2_combo_oracle_retrieval_seed2026_unseen", 24
    )
    if oracle is None:
        return
    correct = (
        REPO
        / "lmvla/lawam/results/eval_runs/robotwin/"
        "rt_all6_v2_combo_seed2026_unseen"
    )
    if len(list(correct.glob("**/summary.json"))) < 24:
        return
    command = [
        "python3",
        str(REPO / "lmvla/lmwm/scripts/rt_causal_intervention_analysis.py"),
        "--correct-root",
        str(correct),
        "--control",
        f"retrieval={oracle}",
        "--pairwise",
        "--json",
    ]
    try:
        report = json.loads(run(command, timeout=90))
    except Exception as exc:
        log(
            "oracle retrieval paired report generation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    atomic_json(output, report)
    log(f"materialized oracle retrieval paired report: {output}")


def run_local_causal_report(
    output: Path,
    correct: Path,
    controls: dict[str, Path],
    *,
    correct_expected: int = 12,
) -> None:
    if output.is_file():
        return
    if len(list(correct.glob("**/summary.json"))) < correct_expected:
        return

    command = [
        "python3",
        str(REPO / "lmvla/lmwm/scripts/rt_causal_intervention_analysis.py"),
        "--correct-root",
        str(correct),
    ]
    for name, root in controls.items():
        if len(list(root.glob("**/summary.json"))) < 12:
            return
        command.extend(["--control", f"{name}={root}"])
    command.append("--json")
    try:
        report = json.loads(run(command, timeout=90))
    except Exception as exc:
        log(f"A3 causal report generation failed {output.name}: {type(exc).__name__}: {exc}")
        return
    atomic_json(output, report)
    log(f"materialized local causal report: {output}")


def run_pi05_a3_causal_report(output: Path, controls: dict[str, Path]) -> None:
    correct = (
        REPO
        / "lmvla/lawam/results/eval_runs/robotwin/"
        "pi05_rt_a3_live_residual_official_gf1_8g"
    )
    run_local_causal_report(output, correct, controls, correct_expected=24)


def refresh_pi05_a3_causal_reports() -> None:
    """Merge cross-mount A3 controls and materialize core/final paired reports."""
    eval_base = REPO / "lmvla/lawam/results/eval_runs/robotwin"
    current = sync_north_eval_tree("pi05_rt_a3_current_hint_causal", 12)
    if current is None:
        return
    controls = {
        "current": current,
        "zero": eval_base / "pi05_rt_a3_zero_hint_causal",
        "shuffled": eval_base / "pi05_rt_a3_shuffled_hint_causal",
    }
    run_pi05_a3_causal_report(REPO / "logs/pi05_a3_causal_core.json", controls)

    other_task = sync_north_eval_tree("pi05_rt_a3_other_task_hint_causal", 12)
    if other_task is None:
        return
    run_pi05_a3_causal_report(
        REPO / "logs/pi05_a3_causal_other_task.json",
        {**controls, "other-task": other_task},
    )

    instance_shuffle = eval_base / "pi05_rt_a3_instance_shuffle_causal"
    run_pi05_a3_causal_report(
        REPO / "logs/pi05_a3_causal_final.json",
        {
            **controls,
            "other-task": other_task,
            "instance-shuffle": instance_shuffle,
        },
    )


def refresh_pi05_a2_instance_report() -> None:
    """Sync frozen A2 controls and add the local within-task instance control."""
    output = REPO / "logs/pi05_a2_causal_with_instance.json"
    if output.is_file():
        return
    instance = (
        REPO
        / "lmvla/lawam/results/eval_runs/robotwin/"
        "pi05_rt_a2_instance_shuffle_causal"
    )
    if len(list(instance.glob("**/summary.json"))) < 12:
        return
    labels = {
        "correct": "pi05_rt_a2_prefix_official_v2",
        "current": "pi05_rt_a2_current_hint_causal",
        "zero": "pi05_rt_a2_zero_hint_causal",
        "shuffled": "pi05_rt_a2_shuffled_hint_causal",
        "other-task": "pi05_rt_a2_other_task_hint_causal",
    }
    roots = {name: sync_north_eval_tree(label, 12) for name, label in labels.items()}
    if any(root is None for root in roots.values()):
        return
    run_local_causal_report(
        output,
        roots.pop("correct"),
        {**roots, "instance-shuffle": instance},
    )


def refresh_eval_reports() -> None:
    """Write one local aggregate when any configured result location closes."""
    remote_script = "lmvla/lmwm/scripts/summarize_robotwin_eval.py"
    local_script = REPO / remote_script
    for label, config in EVAL_REPORTS.items():
        output = REPO / "logs/eval_reports" / f"{label}.json"
        if output.is_file():
            continue
        report = None
        for location in config.get("locations", [config]):
            try:
                if location["remote"]:
                    command = [
                        "python3",
                        remote_script,
                        location["root"],
                        "--expected-cells",
                        str(config["expected"]),
                    ]
                    report_text = ssh(
                        GSY,
                        f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}",
                        timeout=90,
                    )
                else:
                    command = [
                        "python3",
                        str(local_script),
                        location["root"],
                        "--expected-cells",
                        str(config["expected"]),
                    ]
                    report_text = run(command, timeout=90)
                report = json.loads(report_text)
                break
            except Exception:
                continue
        if report is None:
            continue
        atomic_json(output, report)
        log(f"materialized eval report {label}: {output}")


def refresh_pi05_corrected_a0_gate() -> None:
    """Materialize the protocol/result gate that unlocks corrected A2/A3."""
    report = REPO / "logs/eval_reports/pi05_rt_a0_public_recipe_seed1000.json"
    output = REPO / "logs/pi05_a0_public_recipe_gate.json"
    if not report.is_file() or output.is_file():
        return
    remote_output = f"{NORTH_REPO}/logs/pi05_a0_public_recipe_gate.json"
    remote_marker = f"{NORTH_REPO}/logs/resource_markers/pi05_a0_public_recipe_gate.ok"
    command = [
        f"{NORTH_REPO}/kai0/.venv/bin/python",
        "train_scripts/kai/analysis/audit_pi05_corrected_a0_gate.py",
        "--repo",
        NORTH_REPO,
        "--eval-root",
        f"{NORTH_REPO}/lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a0_public_recipe_seed1000",
        "--checkpoint",
        f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a0_public_recipe_bj/pi05_robotwin_a0_public_recipe_seed1000/49999",
        "--norm-stats",
        f"{NORTH_REPO}/kai0/assets/pi05_robotwin_a0_public_recipe_bj/robotwin2.0_absolute_meanstd/norm_stats.json",
        "--launch-manifest",
        f"{NORTH_REPO}/lmvla/paper_iclr_lmvla/manifests/pi05_a0_seed1000_launch.json",
        "--output",
        remote_output,
        "--marker",
        remote_marker,
    ]
    try:
        result = json.loads(
            ssh(GSY, f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}", timeout=180)
        )
    except Exception as exc:
        log(f"corrected A0 gate audit failed: {type(exc).__name__}: {exc}")
        return
    atomic_json(output, result)
    log(
        f"materialized corrected A0 gate accepted={result.get('accepted')} "
        f"macro={result.get('macro_success_rate')}: {output}"
    )


def refresh_pi05_exact_a0_gate() -> None:
    """Audit the repaired-prompt, no-augmentation A0 before unlocking A2/A3."""
    report = REPO / "logs/eval_reports/pi05_rt_a0_public_exact_seed1000.json"
    output = REPO / "logs/pi05_a0_public_exact_gate.json"
    if not report.is_file():
        return
    if output.is_file():
        try:
            if json.loads(output.read_text()).get("accepted") is True:
                return
        except Exception:
            pass
    remote_output = f"{NORTH_REPO}/logs/pi05_a0_public_exact_gate.json"
    remote_marker = f"{NORTH_REPO}/logs/resource_markers/pi05_a0_public_exact_gate.ok"
    command = [
        f"{NORTH_REPO}/kai0/.venv/bin/python",
        "train_scripts/kai/analysis/audit_pi05_corrected_a0_gate.py",
        "--repo",
        NORTH_REPO,
        "--eval-root",
        (
            f"{NORTH_REPO}/lmvla/lawam/results/eval_runs/robotwin/"
            "pi05_rt_a0_public_exact_seed1000"
        ),
        "--checkpoint",
        (
            f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/"
            "pi05_robotwin_a0_public_exact_seed1000/49999"
        ),
        "--norm-stats",
        (
            f"{NORTH_REPO}/kai0/assets/pi05_robotwin_a0_public_exact_bj/"
            "robotwin2.0_absolute_meanstd/norm_stats.json"
        ),
        "--launch-manifest",
        (
            f"{NORTH_REPO}/lmvla/paper_iclr_lmvla/manifests/"
            "pi05_a0_exact_seed1000_launch.json"
        ),
        "--launch-config-snapshot",
        (
            f"{NORTH_REPO}/lmvla/paper_iclr_lmvla/manifests/"
            "pi05_a0_exact_seed1000_config_at_launch.py"
        ),
        "--config-name",
        "pi05_robotwin_a0_public_exact_bj",
        "--expected-job-name",
        "pi05-a0-public-exact-s1000-bj4g",
        "--expected-exp-name",
        "pi05_robotwin_a0_public_exact_seed1000",
        "--expected-seed",
        "1000",
        "--dataset-manifest",
        (
            "/vePFS-North-E/vis_robot/huanqian/uniVP/data/robotwin2.0/"
            "robotwin2.0_official_prompts_v21/meta/"
            "official_prompt_repair_manifest.json"
        ),
        "--require-no-augmentation",
        "--output",
        remote_output,
        "--marker",
        remote_marker,
    ]
    try:
        result = json.loads(
            ssh(GSY, f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}", timeout=180)
        )
    except Exception as exc:
        log(f"exact A0 gate audit failed: {type(exc).__name__}: {exc}")
        return
    atomic_json(output, result)
    log(
        f"materialized exact A0 gate accepted={result.get('accepted')} "
        f"macro={result.get('macro_success_rate')}: {output}"
    )


def refresh_method_matrix() -> None:
    """Regenerate the cross-training-seed method matrix when inputs change."""
    report_dir = REPO / "logs/eval_reports"
    script = REPO / "lmvla/lmwm/scripts/summarize_robotwin_method_matrix.py"
    output = report_dir / "robotwin_all6_v2_training_seed_matrix.json"
    try:
        report = json.loads(run(["python3", str(script), str(report_dir)], timeout=90))
    except Exception:
        return
    if output.is_file():
        try:
            if json.loads(output.read_text()) == report:
                return
        except Exception:
            pass
    atomic_json(output, report)
    log(f"materialized training-seed matrix: {output}")


def refresh_pi05_confirmatory_matrix() -> None:
    """Regenerate the matched pi0.5 seed matrix as evaluations close."""
    report_dir = REPO / "logs/eval_reports"
    script = REPO / "lmvla/lmwm/scripts/summarize_pi05_confirmatory_matrix.py"
    output = report_dir / "pi05_confirmatory_training_seed_matrix.json"
    try:
        report = json.loads(run(["python3", str(script), str(report_dir)], timeout=180))
    except Exception:
        return
    if output.is_file():
        try:
            if json.loads(output.read_text()) == report:
                return
        except Exception:
            pass
    atomic_json(output, report)
    log(
        "materialized pi0.5 confirmatory matrix "
        f"complete={report.get('complete')}: {output}"
    )


def submit_platform(candidate: dict[str, Any], credential_profile: str = "primary") -> str:
    command = [
        str(REPO / "kai0/.venv/bin/python"),
        str(REPO / "train_scripts/kai/volc/submit_yaml.py"),
        str(REPO / candidate["yaml"]),
    ]
    if candidate.get("task_name"):
        command.extend(["--task-name", candidate["task_name"]])
    for name, value in candidate.get("env", {}).items():
        command.extend(["--set-env", f"{name}={value}"])
    env_overrides = None
    if credential_profile != "primary":
        ak, sk = credential_values(credential_profile)
        env_overrides = {"VOLC_AK": ak, "VOLC_SK": sk}
    output = run(command, timeout=180, env_overrides=env_overrides)
    match = re.search(r"SUCCESS task_id=(t-[0-9a-z-]+)", output)
    if not match:
        raise RuntimeError(f"could not parse task id: {output[-1500:]}")
    return match.group(1)


def canonical_manifest_sha256(payload: dict[str, Any]) -> str:
    """Hash dataset content while ignoring environment-specific absolute paths."""
    semantic = dict(payload)
    for key in ("builder", "output", "source"):
        semantic.pop(key, None)
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def remote_sha256(paths: list[str]) -> dict[str, str]:
    command = "sha256sum " + " ".join(shlex.quote(path) for path in paths)
    output = ssh(GSY, command, timeout=60)
    result: dict[str, str] = {}
    for line in output.splitlines():
        digest, path = line.split(maxsplit=1)
        result[path.lstrip("* ")] = digest
    return result


def capture_pi05_confirmatory_launch(
    task: dict[str, Any], candidate: dict[str, Any], job_id: str, *, backfill: bool = False
) -> None:
    """Freeze matched A0/A2/A3 launch provenance for every training seed."""
    match = re.fullmatch(
        r"pi05_(a0_public_exact|a2_abs_confirmatory|a3_live_confirmatory)_seed(100[012])_train",
        task["id"],
    )
    if match is None:
        return
    arm_key, seed_text = match.groups()
    arm = {
        "a0_public_exact": "a0",
        "a2_abs_confirmatory": "a2_abs",
        "a3_live_confirmatory": "a3_live",
    }[arm_key]
    manifest_dir = REPO / "lmvla/paper_iclr_lmvla/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"pi05_confirmatory_{arm}_seed{seed_text}_launch.json"
    if manifest_path.is_file():
        return

    shared_dataset_manifest = Path(
        "/vePFS/tim/workspace/VLANeXt-main/datasets/"
        "robotwin2.0_official_prompts_v21/meta/official_prompt_repair_manifest.json"
    )
    runtime_env = {str(k): str(v) for k, v in candidate.get("env", {}).items()}
    config_name = runtime_env.get(
        "PI05_CONFIG_NAME",
        runtime_env.get("PI05_CONFIRM_CONFIG", "pi05_robotwin_a0_public_exact_bj"),
    )
    experiment_name = runtime_env.get(
        "PI05_EXP_NAME",
        runtime_env.get("PI05_CONFIRM_EXP", f"pi05_robotwin_a0_public_exact_seed{seed_text}"),
    )
    is_north = candidate["resource"] == "Robot-North-H20"
    if is_north:
        source_root = NORTH_REPO
        dataset_manifest_path = (
            "/vePFS-North-E/vis_robot/huanqian/uniVP/data/robotwin2.0/"
            "robotwin2.0_official_prompts_v21/meta/official_prompt_repair_manifest.json"
        )
        dataset_payload = json.loads(
            ssh(GSY, f"cat {shlex.quote(dataset_manifest_path)}", timeout=30)
        )
        source_paths = [
            f"{source_root}/kai0/src/openpi/training/config.py",
            f"{source_root}/kai0/src/openpi/models/pi0.py",
        ]
        if arm != "a0":
            source_paths.append(f"{source_root}/kai0/scripts/train_pi05_robotwin_confirmatory.py")
        source_hashes = remote_sha256(source_paths)
        source_hashes = {Path(path).name: digest for path, digest in source_hashes.items()}
        dataset_raw_sha = remote_sha256([dataset_manifest_path])[dataset_manifest_path]
    else:
        dataset_payload = json.loads(shared_dataset_manifest.read_text())
        source_files = [
            REPO / "kai0/src/openpi/training/config.py",
            REPO / "kai0/src/openpi/models/pi0.py",
        ]
        if arm != "a0":
            source_files.append(REPO / "kai0/scripts/train_pi05_robotwin_confirmatory.py")
        source_hashes = {path.name: sha256_file(path) for path in source_files}
        dataset_raw_sha = sha256_file(shared_dataset_manifest)

    artifact_hashes: dict[str, str] = {}
    if arm == "a2_abs":
        hash_record = REPO / "logs/efficiency/a2_hint_sync.sha256"
        artifact_hashes["hint.npz"] = hash_record.read_text().split()[0]
    elif arm == "a3_live":
        artifact_hashes["pairs.npz"] = sha256_file(
            REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
        )

    manifest = {
        "capture_source": (
            "resource_aware_scheduler state backfill"
            if backfill
            else "resource_aware_scheduler dispatch"
        ),
        "captured_at": utc_now(),
        "config_name": config_name,
        "experiment_name": experiment_name,
        "job_id": job_id,
        "job_name": candidate["task_name"],
        "protocol": {
            "arm": arm,
            "action_horizon": 50,
            "actions": "absolute",
            "asset_id": runtime_env.get("PI05_ASSET_ID", "robotwin2.0_absolute_meanstd"),
            "batch_size": 16,
            "evaluation_cells": 24,
            "evaluation_seeds": 4,
            "image_augmentation": "none",
            "lmwm_spatial_condition": "none",
            "normalization": "mean_std",
            "num_train_steps": 50_000,
            "training_seed": int(seed_text),
        },
        "resource": {
            "backend": candidate["kind"],
            "name": candidate["resource"],
            "gpus": int(candidate["gpus"]),
            "preemptible": False,
            "queue_id": (
                QUEUE_CONFIG[candidate["resource"]]["id"]
                if candidate["kind"] == "platform"
                else None
            ),
            "accelerator": "NVIDIA A100-SXM4-80GB"
            if candidate["kind"] == "ssh"
            else "GPU-H3c",
        },
        "runtime_env": runtime_env,
        "sha256": {
            "dataset_manifest_raw": dataset_raw_sha,
            "dataset_manifest_semantic": canonical_manifest_sha256(dataset_payload),
            "execution_sources": source_hashes,
            "conditioning_artifacts": artifact_hashes,
        },
    }
    if is_north:
        manifest["code_equivalence_note"] = (
            "The North pi0.py differs from the shared copy only by spatial-condition "
            "code gated behind lmwm_spatial_condition != 'none'; this run sets 'none'."
        )
    if candidate["kind"] == "platform":
        manifest["sha256"]["submitted_yaml"] = sha256_file(REPO / candidate["yaml"])
    else:
        manifest["sha256"]["launch_command"] = hashlib.sha256(
            candidate["command"].encode()
        ).hexdigest()
    atomic_json(manifest_path, manifest)
    log(f"captured {arm} seed{seed_text} launch provenance job_id={job_id}")


def capture_pi05_confirmatory_eval_launch(
    task: dict[str, Any], candidate: dict[str, Any], job_id: str
) -> None:
    """Freeze the actual evaluator resource, checkpoint, and protocol at dispatch."""
    match = re.fullmatch(
        r"pi05_(a0_public_exact|a2_abs_confirmatory|a3_live_confirmatory)_seed(100[012])_eval",
        task["id"],
    )
    if match is None:
        return
    arm_key, seed_text = match.groups()
    arm = {
        "a0_public_exact": "a0",
        "a2_abs_confirmatory": "a2_abs",
        "a3_live_confirmatory": "a3_live",
    }[arm_key]
    manifest_dir = REPO / "lmvla/paper_iclr_lmvla/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / f"pi05_confirmatory_{arm}_seed{seed_text}_eval_launch.json"
    if output.is_file():
        return

    is_north = candidate["resource"] == "Robot-North-H20"
    local_ready = [
        *task.get("ready_files", []),
        *candidate.get("ready_files", []),
    ]
    remote_ready = [
        *task.get("ready_files_remote", []),
        *candidate.get("ready_files_remote", []),
    ]
    ready_paths = remote_ready if is_north else local_ready
    metadata_path = next(
        (path for path in ready_paths if path.endswith("/49999/params/_METADATA")),
        None,
    )
    if metadata_path is None:
        for alternative in task.get("ready_any", []):
            key = "ready_files_remote" if is_north else "ready_files"
            metadata_path = next(
                (
                    path
                    for path in alternative.get(key, [])
                    if path.endswith("/49999/params/_METADATA")
                ),
                None,
            )
            if metadata_path is not None:
                break
    if metadata_path is None:
        raise ValueError(f"cannot resolve final checkpoint metadata for {task['id']}")
    metadata_sha = (
        remote_sha256([metadata_path])[metadata_path]
        if is_north
        else sha256_file(Path(metadata_path))
    )

    protocol_path = manifest_dir / "pi05_confirmatory_eval_protocol.json"
    training_manifest = manifest_dir / f"pi05_confirmatory_{arm}_seed{seed_text}_launch.json"
    runtime_env = {str(key): str(value) for key, value in candidate.get("env", {}).items()}
    manifest = {
        "capture_source": "resource_aware_scheduler evaluator dispatch",
        "captured_at": utc_now(),
        "arm": arm,
        "training_seed": int(seed_text),
        "job_id": job_id,
        "job_name": candidate.get("task_name", task["id"]),
        "resource": {
            "backend": candidate["kind"],
            "name": candidate["resource"],
            "gpus": int(candidate["gpus"]),
            "gpu_indices": candidate.get("gpu_indices"),
            "queue_id": (
                QUEUE_CONFIG[candidate["resource"]]["id"]
                if candidate["kind"] == "platform"
                else None
            ),
        },
        "runtime_env": runtime_env,
        "checkpoint": {
            "metadata_path": metadata_path,
            "metadata_sha256": metadata_sha,
        },
        "sha256": {
            "evaluation_protocol": sha256_file(protocol_path),
            "training_launch_manifest": sha256_file(training_manifest),
        },
    }
    if candidate["kind"] == "platform":
        manifest["sha256"]["submitted_yaml"] = sha256_file(REPO / candidate["yaml"])
    else:
        manifest["sha256"]["launch_command"] = hashlib.sha256(
            candidate["command"].encode()
        ).hexdigest()
    atomic_json(output, manifest)
    log(f"captured {arm} seed{seed_text} evaluator provenance job_id={job_id}")


def refresh_pi05_launch_provenance(queue: dict[str, Any], state: dict[str, Any]) -> None:
    """Backfill manifests for jobs that were dispatched before unified capture existed."""
    for task in queue["tasks"]:
        if not re.fullmatch(
            r"pi05_(a0_public_exact|a2_abs_confirmatory|a3_live_confirmatory)_seed100[012]_train",
            task["id"],
        ):
            continue
        task_state = state["tasks"].get(task["id"], {})
        attempts = [attempt for attempt in task_state.get("attempts", []) if attempt.get("job_id") or attempt.get("pid")]
        if not attempts:
            continue
        attempt = attempts[-1]
        candidates = [
            candidate
            for candidate in task.get("candidates", [])
            if candidate.get("resource") == attempt.get("resource")
            and candidate.get("kind") == attempt.get("kind")
        ]
        if not candidates:
            continue
        job_id = attempt.get("job_id")
        if job_id is None:
            job_id = f"gf1-{attempt['pid']}"
        try:
            capture_pi05_confirmatory_launch(
                task, candidates[0], str(job_id), backfill=True
            )
        except Exception as exc:
            log(
                f"confirmatory provenance backfill failed {task['id']}: "
                f"{type(exc).__name__}: {exc}"
            )


def launch_gf1(candidate: dict[str, Any]) -> str:
    status_dir = candidate["status_dir"]
    command = candidate["command"]
    body = (
        f"set +e; start=$(date -u +%FT%TZ); "
        f"echo \"RUNNING start=$start host=$(hostname)\" > {shlex.quote(status_dir + '/status')}; "
        f"bash -lc {shlex.quote(command)}; rc=$?; end=$(date -u +%FT%TZ); "
        f"echo \"FINISHED rc=$rc start=$start end=$end host=$(hostname)\" > {shlex.quote(status_dir + '/status')}; exit $rc"
    )
    remote = (
        f"mkdir -p {shlex.quote(status_dir)}; rm -f {shlex.quote(status_dir + '/status')}; "
        f"nohup bash -c {shlex.quote(body)} > {shlex.quote(status_dir + '/launcher.log')} 2>&1 < /dev/null & "
        f"echo $! | tee {shlex.quote(status_dir + '/pid')}"
    )
    return ssh(GF1, remote, timeout=30).strip().splitlines()[-1]


def launch_local(candidate: dict[str, Any]) -> str:
    status_dir = Path(candidate["status_dir"])
    try:
        status_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(status_dir, os.W_OK):
            raise PermissionError(f"status directory is not writable: {status_dir}")
    except PermissionError:
        status_dir = REPO / "logs/resource_scheduler_local" / status_dir.name
        status_dir.mkdir(parents=True, exist_ok=True)
        candidate["status_dir"] = str(status_dir)
    status_path = status_dir / "status"
    status_path.unlink(missing_ok=True)
    command = candidate["command"]
    body = (
        "set +e; start=$(date -u +%FT%TZ); "
        f"echo \"RUNNING start=$start host=$(hostname)\" > {shlex.quote(str(status_path))}; "
        f"bash -lc {shlex.quote(command)}; rc=$?; end=$(date -u +%FT%TZ); "
        f"echo \"FINISHED rc=$rc start=$start end=$end host=$(hostname)\" > {shlex.quote(str(status_path))}; exit $rc"
    )
    stream = (status_dir / "launcher.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-c", body],
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream.close()
    (status_dir / "pid").write_text(f"{process.pid}\n")
    return str(process.pid)


def check_managed_task(task: dict[str, Any], task_state: dict[str, Any]) -> None:
    if task_state.get("status") != "running":
        if task_state.get("artifacts_complete"):
            mark_task_completed(task, task_state)
        return
    attempt = task_state["attempts"][-1]
    if attempt["kind"] == "platform":
        credential_profile = attempt.get("credential_profile", "primary")
        if credential_profile == "backup" and not backup_credentials_enabled():
            attempt["monitor_status"] = "backup credential profile is disabled"
            attempt["last_checked_at"] = utc_now()
            return
        try:
            info = get_job(attempt["region"], attempt["job_id"], credential_profile)
        except Exception as exc:
            attempt["monitor_error"] = type(exc).__name__
            attempt["last_checked_at"] = utc_now()
            return
        attempt.pop("monitor_error", None)
        attempt.pop("monitor_status", None)
        attempt["last_state"] = info["state"]
        attempt["last_checked_at"] = utc_now()
        if info["state"] in {"Completed", "Success"}:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = f"terminal state without complete outputs: {evidence}"
                log(f"retrying {task['id']}: {evidence}")
        elif info["state"] in {"Failed", "Stopped"}:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = info["message"]
        elif info["state"] == "Queueing":
            started = datetime.fromisoformat(attempt["started_at"].replace("Z", "+00:00"))
            queue_timeout = int(attempt.get("queue_timeout_seconds", 300))
            if (datetime.now(timezone.utc) - started).total_seconds() > queue_timeout:
                service(attempt["region"], credential_profile).json(
                    "StopJob", {}, json.dumps({"Id": attempt["job_id"]}).encode()
                )
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = (
                    f"reclaimed after queueing for more than {queue_timeout} seconds"
                )
                log(f"reclaimed queued job {attempt['job_id']} for local task retry")
    elif attempt["kind"] == "ssh":
        status_path = attempt["status_dir"] + "/status"
        try:
            status = ssh(GF1, f"cat {shlex.quote(status_path)}", timeout=20).strip()
        except Exception as exc:
            attempt["monitor_error"] = str(exc)
            return
        attempt["last_status"] = status
        attempt["last_checked_at"] = utc_now()
        match = re.search(r"FINISHED rc=(\d+)", status)
        if match and int(match.group(1)) == 0:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = f"successful process without complete outputs: {evidence}"
        elif match:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = status
        elif status.startswith("RUNNING"):
            try:
                ssh(GF1, f"kill -0 {int(attempt['pid'])}", timeout=20)
            except Exception:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = "launcher disappeared while status remained RUNNING"
                log(f"reclaimed orphaned gf1 launcher for {task['id']}")
    else:
        status_path = Path(attempt["status_dir"]) / "status"
        if not status_path.is_file():
            try:
                os.kill(int(attempt["pid"]), 0)
            except (ProcessLookupError, ValueError):
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = f"launcher disappeared without {status_path}"
                log(f"reclaimed orphaned local launcher for {task['id']}")
            except PermissionError:
                attempt["monitor_error"] = f"cannot inspect launcher pid={attempt['pid']}"
            else:
                attempt["monitor_error"] = f"missing {status_path}"
            return
        status = status_path.read_text().strip()
        attempt["last_status"] = status
        attempt["last_checked_at"] = utc_now()
        match = re.search(r"FINISHED rc=(\d+)", status)
        if match and int(match.group(1)) == 0:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = f"successful process without complete outputs: {evidence}"
        elif match:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = status
        elif status.startswith("RUNNING"):
            try:
                os.kill(int(attempt["pid"]), 0)
            except (ProcessLookupError, ValueError):
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = "launcher disappeared while status remained RUNNING"
                log(f"reclaimed orphaned local launcher for {task['id']}")
            except PermissionError:
                attempt["monitor_error"] = f"cannot inspect launcher pid={attempt['pid']}"


def managed_platform_job_ids(
    state: dict[str, Any] | None, credential_profile: str
) -> set[str]:
    job_ids: set[str] = set()
    if not state:
        return job_ids
    for task_state in state.get("tasks", {}).values():
        attempts = task_state.get("attempts", [])
        if task_state.get("status") != "running" or not attempts:
            continue
        attempt = attempts[-1]
        if (
            attempt.get("kind") == "platform"
            and attempt.get("credential_profile", "primary") == credential_profile
            and attempt.get("job_id")
        ):
            job_ids.add(attempt["job_id"])
    return job_ids


def make_snapshot(state: dict[str, Any] | None = None) -> dict[str, Any]:
    queue_errors: dict[str, str] = {}

    def safe_list(region: str, queue_id: str, label: str) -> list[dict[str, Any]]:
        try:
            return list_jobs(region, queue_id)
        except Exception as exc:
            queue_errors[label] = f"{type(exc).__name__}: {exc}"
            log(f"resource probe unavailable {label}: {queue_errors[label]}")
            return []

    north = safe_list("cn-beijing", NORTH_QUEUE, "beijing")
    shanghai = safe_list("cn-shanghai", SH_QUEUE, "robot-task")
    east = safe_list("cn-shanghai", EAST_QUEUE, "Robot-East-H20")
    north_owned = [job for job in north if job.get("CreatedBy") == OWNER]
    shanghai_owned = [job for job in shanghai if job.get("CreatedBy") == OWNER]
    gf1 = safe_gpu_snapshot(GF1, 8, "gf1")
    gf1["watched_tasks"] = gf1_watched_statuses()
    gf1["watched_tasks"].update(gf1_training_statuses())
    local = safe_gpu_snapshot(None, 2, "local")
    local["watched_tasks"] = local_watched_statuses()
    north_watched = north_watched_statuses()
    north_watched.update(north_training_statuses())
    backup_enabled = backup_credentials_enabled()
    backup_north: dict[str, Any] = {
        "enabled": backup_enabled,
        "configured": BACKUP_CREDENTIALS_PATH.is_file(),
        "available": False,
        "managed_active_gpus": 0,
        "managed_queueing": [],
        "personal_limit": NORTH_BACKUP_PERSONAL_LIMIT,
    }
    if backup_enabled:
        try:
            if not backup_credentials_configured():
                raise RuntimeError("backup credentials are missing or invalid")
            backup_jobs = list_jobs("cn-beijing", NORTH_QUEUE, "backup")
            managed_ids = managed_platform_job_ids(state, "backup")
            managed_jobs = [job for job in backup_jobs if job.get("Id") in managed_ids]
            backup_north.update(
                {
                    "available": True,
                    "managed_active_gpus": sum(
                        job["_gpus"]
                        for job in managed_jobs
                        if job["_state"] in ACTIVE_STATES
                    ),
                    "managed_queueing": [
                        job.get("Id")
                        for job in managed_jobs
                        if job["_state"] == "Queueing"
                    ],
                }
            )
        except Exception as exc:
            backup_north["error_type"] = type(exc).__name__
    tracked = {}
    for job_id, (region, label) in TRACKED_JOBS.items():
        try:
            tracked[job_id] = {"label": label, "region": region, **get_job(region, job_id)}
        except Exception as exc:
            tracked[job_id] = {"label": label, "region": region, "error": str(exc)}
    return {
        "timestamp": utc_now(),
        "resources": {
            "beijing": {
                "available": "beijing" not in queue_errors,
                "owned_active_gpus": sum(job["_gpus"] for job in north_owned if job["_state"] in ACTIVE_STATES),
                "owned_queueing": [job.get("Id") for job in north_owned if job["_state"] == "Queueing"],
                "active_gpus_all_users": sum(
                    job["_gpus"] for job in north if job["_state"] in ACTIVE_STATES
                ),
                "queueing_all_users": [
                    job.get("Id") for job in north if job["_state"] == "Queueing"
                ],
                "capacity": NORTH_CAPACITY,
                "personal_limit": NORTH_PERSONAL_LIMIT,
                "backup": backup_north,
                "watched_tasks": north_watched,
            },
            "robot-task": {
                "available": "robot-task" not in queue_errors,
                "active_gpus_all_users": sum(job["_gpus"] for job in shanghai if job["_state"] in ACTIVE_STATES),
                "owned_active_gpus": sum(
                    job["_gpus"]
                    for job in shanghai_owned
                    if job["_state"] in ACTIVE_STATES
                ),
                "queueing_all_users": [job.get("Id") for job in shanghai if job["_state"] == "Queueing"],
                "owned_queueing": [job.get("Id") for job in shanghai_owned if job["_state"] == "Queueing"],
                "capacity": SH_CAPACITY,
                "watched_tasks": platform_training_statuses(),
                "watched_evals": shared_eval_statuses(),
            },
            "Robot-East-H20": {
                "available": "Robot-East-H20" not in queue_errors,
                "active_gpus_all_users": sum(
                    job["_gpus"] for job in east if job["_state"] in ACTIVE_STATES
                ),
                "queueing_all_users": [
                    job.get("Id") for job in east if job["_state"] == "Queueing"
                ],
                "capacity": 8,
                "watched_tasks": platform_training_statuses(EAST_TRAIN_WATCH_TASKS),
            },
            "gf1": gf1,
            "local": local,
        },
        "tracked_jobs": tracked,
    }


def write_markdown_snapshot(snapshot: dict[str, Any]) -> None:
    resources = snapshot["resources"]
    rows = [
        ("Beijing primary owned", resources["beijing"]["owned_active_gpus"], NORTH_PERSONAL_LIMIT, len(resources["beijing"]["owned_queueing"])),
        ("robot-task owned", resources["robot-task"]["owned_active_gpus"], SH_PERSONAL_LIMIT, len(resources["robot-task"]["owned_queueing"])),
        ("robot-task all users", resources["robot-task"]["active_gpus_all_users"], SH_CAPACITY, len(resources["robot-task"]["queueing_all_users"])),
        ("Robot-East-H20 all users", resources["Robot-East-H20"]["active_gpus_all_users"], 8, len(resources["Robot-East-H20"]["queueing_all_users"])),
        ("gf1", resources["gf1"]["count"] - resources["gf1"]["free_count"], resources["gf1"]["count"], 0),
        ("local", resources["local"]["count"] - resources["local"]["free_count"], resources["local"]["count"], 0),
    ]
    backup = resources["beijing"].get("backup", {})
    if backup.get("enabled"):
        rows.insert(
            1,
            (
                "Beijing backup managed",
                backup.get("managed_active_gpus", 0),
                backup.get("personal_limit", NORTH_BACKUP_PERSONAL_LIMIT),
                len(backup.get("managed_queueing", [])),
            ),
        )
    lines = [
        "# Resource Scheduler Snapshot",
        "",
        f"Updated: `{snapshot['timestamp']}`",
        "",
        "Dispatch priority: `gf1 > Robot-East-H20 > Robot-North-H20 > robot-task`.",
        "",
        "| Resource | Active GPUs | Capacity/limit | Free | Queueing |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {active} | {capacity} | {capacity - active} | {queueing} |"
        for name, active, capacity, queueing in rows
    )
    lines.extend(
        [
            "",
            "## Managed Tasks",
            "",
            "| Task | Status | Progress |",
            "|---|---|---|",
        ]
    )
    for task_id, state in sorted(snapshot.get("scheduler_tasks", {}).items()):
        if state.get("status") in {"completed", "disabled"}:
            continue
        if state.get("status") == "pending" and state.get("waiting_reason"):
            progress = state["waiting_reason"]
        else:
            progress = state.get("runtime_progress") or state.get("artifact_progress", "")
        lines.append(f"| `{task_id}` | {state.get('status', 'unknown')} | {progress} |")
    lines.extend(
        [
            "",
            "## Training Heartbeats",
            "",
            "| Resource | Task | Step | Rate | ETA | Status |",
            "|---|---|---:|---|---|---|",
        ]
    )
    training_groups = (
        ("Beijing", resources["beijing"].get("watched_tasks", {})),
        ("Robot-East-H20", resources["Robot-East-H20"].get("watched_tasks", {})),
        ("robot-task", resources["robot-task"].get("watched_tasks", {})),
        ("gf1", resources["gf1"].get("watched_tasks", {})),
        ("local", resources["local"].get("watched_tasks", {})),
    )
    for resource, tasks in training_groups:
        for task, status in sorted(tasks.items()):
            if not status.get("status"):
                continue
            if str(status["status"]).startswith("FINISHED"):
                continue
            managed_id = TRAIN_WATCH_MANAGED_TASK_IDS.get((resource, task))
            managed = snapshot.get("scheduler_tasks", {}).get(managed_id, {})
            if managed_id and managed.get("status") != "running":
                continue
            rate = status.get("seconds_per_step")
            rate_text = f"{rate:.2f} s/step" if isinstance(rate, (int, float)) else str(status.get("rate_it_s", ""))
            eta = status.get("eta_hours", status.get("eta", ""))
            eta_text = f"{eta:.2f} h" if isinstance(eta, (int, float)) else str(eta)
            lines.append(
                f"| {resource} | `{task}` | {status.get('step', '')} | {rate_text} | {eta_text} | {status.get('status', '')} |"
            )
    atomic_text(SNAPSHOT_MARKDOWN_PATH, "\n".join(lines) + "\n")


def candidate_available(
    candidate: dict[str, Any],
    snapshot: dict[str, Any],
    credential_profile: str = "primary",
) -> bool:
    if not readiness_spec_satisfied(candidate):
        return False
    resource = candidate["resource"]
    gpus = int(candidate["gpus"])
    resources = snapshot["resources"]
    if resource == "gf1":
        state = resources["gf1"]
        required_indices = candidate.get("gpu_indices")
        if required_indices is not None:
            free_indices = {
                row["index"]
                for row in state.get("gpus", [])
                if row["memory_used_mib"] < 1024
            }
            return state.get("available", True) and set(required_indices) <= free_indices
        return state.get("available", True) and state["free_count"] >= gpus
    if resource == "local":
        state = resources["local"]
        required_indices = candidate.get("gpu_indices")
        if required_indices is not None:
            free_indices = {
                row["index"]
                for row in state.get("gpus", [])
                if row["memory_used_mib"] < 1024
            }
            return state.get("available", True) and set(required_indices) <= free_indices
        return state.get("available", True) and state["free_count"] >= gpus
    if resource == "robot-task":
        state = resources["robot-task"]
        # Nominal free cards can be split across nodes. Try once when the
        # nominal count fits, then let queue-timeout recovery hold subsequent
        # retries until active usage actually drops.
        free = state["capacity"] - state["active_gpus_all_users"]
        min_dispatch_free = int(candidate.get("min_dispatch_free", SH_MIN_DISPATCH_FREE))
        return (
            state.get("available", True)
            and not state["queueing_all_users"]
            and state["owned_active_gpus"] + gpus <= SH_PERSONAL_LIMIT
            and free >= max(gpus, min_dispatch_free)
        )
    if resource == "Robot-East-H20":
        state = resources["Robot-East-H20"]
        return (
            state.get("available", True)
            and not state["queueing_all_users"]
            and state["capacity"] - state["active_gpus_all_users"] >= gpus
        )
    if resource == "Robot-North-H20":
        state = resources["beijing"]
        if credential_profile == "primary":
            return (
                state.get("available", True)
                and not state["owned_queueing"]
                and state["owned_active_gpus"] + gpus <= state["personal_limit"]
            )
        backup = state.get("backup", {})
        physical_free = state["capacity"] - state["active_gpus_all_users"]
        return (
            credential_profile == "backup"
            # Spill to the backup identity once the next task no longer fits
            # under the primary identity's personal limit. Requiring the
            # primary usage to equal the limit leaves unusable 1-3 GPU gaps.
            and state.get("available", True)
            and state["owned_active_gpus"] + gpus > state["personal_limit"]
            and backup.get("enabled")
            and backup.get("available")
            and not backup.get("managed_queueing")
            and not state.get("queueing_all_users")
            and backup.get("managed_active_gpus", 0) + gpus
            <= backup.get("personal_limit", NORTH_BACKUP_PERSONAL_LIMIT)
            and physical_free >= gpus
        )
    return False


def candidate_credential_profile(
    candidate: dict[str, Any], snapshot: dict[str, Any]
) -> str | None:
    if candidate.get("kind") != "platform":
        return None
    if candidate.get("resource") != "Robot-North-H20":
        return "primary" if candidate_available(candidate, snapshot) else None
    if candidate_available(candidate, snapshot, "primary"):
        return "primary"
    if candidate_available(candidate, snapshot, "backup"):
        return "backup"
    return None


def candidate_in_cooldown(
    task_state: dict[str, Any],
    candidate: dict[str, Any],
    snapshot: dict[str, Any],
    credential_profile: str = "primary",
) -> bool:
    """Avoid repeatedly launching a broken template on the same resource."""
    now = datetime.now(timezone.utc)
    for attempt in reversed(task_state.get("attempts", [])):
        if (
            attempt.get("resource") != candidate["resource"]
            or attempt.get("credential_profile", "primary") != credential_profile
            or not attempt.get("failure")
        ):
            continue
        timestamp = attempt.get("finished_at") or attempt.get("last_checked_at")
        if not timestamp:
            return True
        if (
            candidate["resource"] == "robot-task"
            and attempt.get("active_gpus_at_dispatch") is not None
            and (
                attempt["failure"].startswith("reclaimed after queueing")
                or "剩余配额不足" in attempt["failure"]
            )
        ):
            current_active = snapshot["resources"]["robot-task"]["active_gpus_all_users"]
            if current_active < int(attempt["active_gpus_at_dispatch"]):
                return False
            return True
        failed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return (now - failed_at).total_seconds() < int(
            candidate.get("retry_cooldown_seconds", RETRY_COOLDOWN_SECONDS)
        )
    return False


def candidate_failure_count(
    task_state: dict[str, Any],
    candidate: dict[str, Any],
    credential_profile: str = "primary",
) -> int:
    """Count runtime/template failures, excluding transient capacity failures."""
    transient_markers = ("reclaimed after queueing", "剩余配额不足")
    return sum(
        1
        for attempt in task_state.get("attempts", [])
        if attempt.get("resource") == candidate["resource"]
        and attempt.get("credential_profile", "primary") == credential_profile
        and attempt.get("failure")
        and not any(marker in attempt["failure"] for marker in transient_markers)
    )


def candidate_exhausted(
    task_state: dict[str, Any],
    candidate: dict[str, Any],
    task_id: str | None = None,
    credential_profile: str = "primary",
) -> bool:
    failures = candidate_failure_count(task_state, candidate, credential_profile)
    limit = int(candidate.get("max_failures", MAX_FAILURES_PER_RESOURCE))
    if failures < limit:
        return False
    exhausted = task_state.setdefault("exhausted_resources", {})
    resource = candidate["resource"]
    resource_key = (
        resource if credential_profile == "primary" else f"{resource}@{credential_profile}"
    )
    if exhausted.get(resource_key, {}).get("failures") != failures:
        exhausted[resource_key] = {
            "failures": failures,
            "limit": limit,
            "updated_at": utc_now(),
        }
        if task_id:
            log(
                f"exhausted {resource_key} for {task_id} after {failures} failures"
            )
    return True


def dispatch(queue: dict[str, Any], state: dict[str, Any], snapshot: dict[str, Any]) -> None:
    tasks = sorted(queue["tasks"], key=lambda item: (item["priority"], item["id"]))

    # Helpers are no longer useful once their authoritative parent task has
    # completed. Close active platform helpers and suppress marker-only retries.
    for task in tasks:
        satisfied_by = task.get("satisfied_by_task")
        if not satisfied_by:
            continue
        parent_state = state["tasks"].get(satisfied_by, {})
        task_state = state["tasks"][task["id"]]
        if parent_state.get("status") != "completed" or task_state.get("status") == "completed":
            continue
        if task_state.get("status") == "running" and task_state.get("attempts"):
            attempt = task_state["attempts"][-1]
            if attempt.get("kind") == "platform" and attempt.get("job_id"):
                profile = attempt.get("credential_profile", "primary")
                try:
                    service(attempt["region"], profile).json(
                        "StopJob", {}, json.dumps({"Id": attempt["job_id"]}).encode()
                    )
                    attempt["stopped_after_dependency"] = satisfied_by
                except Exception as exc:
                    attempt["dependency_stop_error"] = f"{type(exc).__name__}: {exc}"
        mark_task_completed(task, task_state)
        task_state["satisfied_by_task"] = satisfied_by
        task_state.pop("waiting_reason", None)
        log(f"completed helper {task['id']} because {satisfied_by} is complete")

    # Always monitor every active attempt before launching more work. Returning
    # immediately after one dispatch used to starve later running tasks whenever
    # an earlier task was launchable, delaying queue-timeout reclamation.
    for task in tasks:
        if not task.get("enabled", True):
            continue
        task_state = state["tasks"][task["id"]]
        if task_state.get("status") == "running":
            check_managed_task(task, task_state)

    for task in tasks:
        if not task.get("enabled", True):
            continue
        task_state = state["tasks"][task["id"]]
        if task_state["status"] != "pending":
            continue
        if task_state.get("attempts") and task.get("completion_glob"):
            complete, evidence = completion_evidence(task)
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
                task_state.pop("waiting_reason", None)
                log(f"completed before redispatch {task['id']}: {evidence}")
                continue
        active_holds = [
            task_id
            for task_id in task.get("hold_retry_while_running", [])
            if state["tasks"].get(task_id, {}).get("status") == "running"
        ]
        if active_holds:
            task_state["waiting_reason"] = (
                "waiting for active helper tasks: " + ", ".join(active_holds)
            )
            continue
        if not ready(task):
            task_state["waiting_reason"] = "blocked by required input, checkpoint, or gate"
            continue
        task_state["waiting_reason"] = "waiting for an eligible resource"
        candidates = sorted(
            task["candidates"],
            key=lambda candidate: RESOURCE_DISPATCH_PRIORITY.get(
                candidate.get("resource", ""), 4
            ),
        )
        for candidate in candidates:
            if candidate["kind"] == "platform":
                credential_profile = candidate_credential_profile(candidate, snapshot)
                if credential_profile is None:
                    continue
            else:
                credential_profile = "primary"
                if not candidate_available(candidate, snapshot):
                    continue
            if candidate_exhausted(
                task_state,
                candidate,
                task["id"],
                credential_profile,
            ):
                continue
            if candidate_in_cooldown(
                task_state,
                candidate,
                snapshot,
                credential_profile,
            ):
                continue
            attempt = {
                "kind": candidate["kind"],
                "resource": candidate["resource"],
                "gpus": int(candidate["gpus"]),
                "started_at": utc_now(),
            }
            try:
                if candidate["kind"] == "platform":
                    attempt["credential_profile"] = credential_profile
                    if candidate["resource"] == "robot-task":
                        attempt["active_gpus_at_dispatch"] = snapshot["resources"][
                            "robot-task"
                        ]["active_gpus_all_users"]
                    job_id = submit_platform(candidate, credential_profile)
                    attempt.update(
                        {
                            "job_id": job_id,
                            "region": candidate["region"],
                            "queue_timeout_seconds": int(candidate.get("queue_timeout_seconds", 300)),
                        }
                    )
                    try:
                        capture_pi05_confirmatory_launch(task, candidate, job_id)
                        capture_pi05_confirmatory_eval_launch(task, candidate, job_id)
                    except Exception as exc:
                        # Submission already succeeded. Keep tracking the platform job and
                        # surface provenance capture as a recoverable audit error.
                        attempt["provenance_error"] = f"{type(exc).__name__}: {exc}"
                        log(
                            f"confirmatory provenance capture failed job_id={job_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    log(
                        f"dispatched {task['id']} to {candidate['resource']} "
                        f"profile={credential_profile} job_id={job_id}"
                    )
                elif candidate["kind"] == "ssh":
                    pid = launch_gf1(candidate)
                    attempt.update({"pid": pid, "status_dir": candidate["status_dir"]})
                    try:
                        capture_pi05_confirmatory_launch(task, candidate, f"gf1-{pid}")
                        capture_pi05_confirmatory_eval_launch(task, candidate, f"gf1-{pid}")
                    except Exception as exc:
                        attempt["provenance_error"] = f"{type(exc).__name__}: {exc}"
                        log(
                            f"confirmatory provenance capture failed gf1 pid={pid}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    log(f"dispatched {task['id']} to gf1 pid={pid}")
                else:
                    pid = launch_local(candidate)
                    attempt.update({"pid": pid, "status_dir": candidate["status_dir"]})
                    try:
                        capture_pi05_confirmatory_eval_launch(task, candidate, f"local-{pid}")
                    except Exception as exc:
                        attempt["provenance_error"] = f"{type(exc).__name__}: {exc}"
                        log(
                            f"confirmatory evaluator provenance capture failed local pid={pid}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    log(f"dispatched {task['id']} to local pid={pid}")
            except Exception as exc:
                attempt.update(
                    {
                        "failure": f"launch failed: {type(exc).__name__}: {exc}",
                        "finished_at": utc_now(),
                    }
                )
                task_state["attempts"].append(attempt)
                task_state["waiting_reason"] = attempt["failure"]
                atomic_json(STATE_PATH, state)
                log(f"{task['id']} {attempt['failure']}")
                continue
            task_state["attempts"].append(attempt)
            task_state["status"] = "running"
            task_state["description"] = task["description"]
            task_state.pop("waiting_reason", None)
            atomic_json(STATE_PATH, state)
            return


def refresh_running_progress(queue: dict[str, Any], state: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    for task in queue["tasks"]:
        if not task.get("enabled", True):
            continue
        task_state = state["tasks"][task["id"]]
        if task_state.get("status") != "running" or not task.get("completion_glob"):
            continue
        complete, evidence = completion_evidence(task)
        record_artifact_progress(task_state, complete, evidence)
        progress_items = []
        for item in task.get("progress_globs", []):
            count = len(glob.glob(item["glob"], recursive=True))
            expected = item.get("expected")
            value = f"{count}/{expected}" if expected is not None else str(count)
            progress_items.append(f"{item['label']}={value}")
        for item in task.get("progress_logs", []):
            files = [Path(path) for path in glob.glob(item["glob"], recursive=True)]
            if not files:
                continue
            latest = max(files, key=lambda path: path.stat().st_mtime)
            matches = re.findall(item["regex"], latest.read_text(errors="replace"))
            if matches:
                match = matches[-1]
                value = "/".join(match) if isinstance(match, tuple) else str(match)
                progress_items.append(f"{item['label']}={value}")
        if progress_items:
            task_state["runtime_progress"] = ", ".join(progress_items)
        # Artifact completion is not enough while a process is still active;
        # retain it as progress and let the terminal-state check close the task.
        changed_at = task_state.get("artifact_progress_changed_at")
        stale_after = int(task.get("progress_stale_seconds", 7200))
        if int(task.get("completion_min_count", 1)) <= 1 or not changed_at:
            continue
        changed = datetime.fromisoformat(changed_at.replace("Z", "+00:00"))
        stale_seconds = max(0, int((now - changed).total_seconds()))
        task_state["artifact_stale_seconds"] = stale_seconds
        warned_at = task_state.get("artifact_stale_warning_at")
        should_warn = stale_seconds >= stale_after
        if should_warn and warned_at:
            warned = datetime.fromisoformat(warned_at.replace("Z", "+00:00"))
            should_warn = (now - warned).total_seconds() >= stale_after
        if should_warn:
            log(f"stale progress warning {task['id']}: {evidence} unchanged for {stale_seconds}s")
            task_state["artifact_stale_warning_at"] = utc_now()


def apply_managed_gpu_reservations(
    queue: dict[str, Any], state: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    """Keep local/SSH cards reserved across transient model reload gaps."""
    reserved = {"local": 0, "gf1": 0}
    tasks_by_id = {task["id"]: task for task in queue["tasks"]}
    for task_id, task_state in state["tasks"].items():
        if task_state.get("status") != "running" or not task_state.get("attempts"):
            continue
        attempt = task_state["attempts"][-1]
        resource = attempt.get("resource")
        if resource not in reserved:
            continue
        gpus = attempt.get("gpus")
        if gpus is None:
            task = tasks_by_id.get(task_id, {})
            matching = [
                candidate
                for candidate in task.get("candidates", [])
                if candidate.get("resource") == resource
                and candidate.get("kind") == attempt.get("kind")
            ]
            if matching:
                gpus = matching[0].get("gpus", 0)
        reserved[resource] += int(gpus or 0)

    for resource, managed_gpus in reserved.items():
        resource_state = snapshot["resources"][resource]
        count = int(resource_state["count"])
        observed_free = int(resource_state["free_count"])
        observed_busy = count - observed_free
        effective_busy = min(count, max(observed_busy, managed_gpus))
        resource_state["observed_free_count"] = observed_free
        resource_state["managed_reserved_gpus"] = managed_gpus
        resource_state["free_count"] = count - effective_busy


def poll_once(queue: dict[str, Any], state: dict[str, Any]) -> None:
    snapshot = make_snapshot(state)
    apply_managed_gpu_reservations(queue, state, snapshot)
    dispatch(queue, state, snapshot)
    refresh_running_progress(queue, state)
    refresh_causal_reports()
    refresh_l2_strict_north_results()
    refresh_oracle_retrieval_report()
    refresh_pi05_a3_causal_reports()
    refresh_pi05_a2_instance_report()
    refresh_eval_reports()
    refresh_pi05_confirmatory_matrix()
    refresh_pi05_launch_provenance(queue, state)
    refresh_pi05_corrected_a0_gate()
    refresh_pi05_exact_a0_gate()
    refresh_method_matrix()
    snapshot["scheduler_tasks"] = state["tasks"]
    atomic_json(STATE_PATH, state)
    atomic_json(SNAPSHOT_PATH, snapshot)
    write_markdown_snapshot(snapshot)
    resources = snapshot["resources"]
    backup = resources["beijing"].get("backup", {})
    log(
        "resources "
        f"bj={resources['beijing']['owned_active_gpus']}/{NORTH_PERSONAL_LIMIT} queued={len(resources['beijing']['owned_queueing'])} "
        f"bj_backup={backup.get('managed_active_gpus', 0)}/{backup.get('personal_limit', NORTH_BACKUP_PERSONAL_LIMIT)} "
        f"backup_enabled={backup.get('enabled', False)} backup_available={backup.get('available', False)} "
        f"sh={resources['robot-task']['active_gpus_all_users']}/{SH_CAPACITY} queued={len(resources['robot-task']['queueing_all_users'])} "
        f"sh_owned={resources['robot-task']['owned_active_gpus']}/{SH_PERSONAL_LIMIT} "
        f"east={resources['Robot-East-H20']['active_gpus_all_users']}/8 queued={len(resources['Robot-East-H20']['queueing_all_users'])} "
        f"gf1_free={resources['gf1']['free_count']}/{resources['gf1']['count']} "
        f"local_free={resources['local']['free_count']}/{resources['local']['count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another resource-aware scheduler instance already holds the lock")
    lock_stream.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_stream.flush()
    if not os.environ.get("VOLC_AK") or not os.environ.get("VOLC_SK"):
        raise SystemExit("VOLC_AK/VOLC_SK are required")
    queue = json.loads(QUEUE_PATH.read_text())
    validate_queue(queue)
    state = load_state(queue)
    log(f"scheduler start interval={args.interval}s once={args.once}")
    while True:
        try:
            queue = json.loads(QUEUE_PATH.read_text())
            validate_queue(queue)
            state = load_state(queue)
            poll_once(queue, state)
        except Exception as exc:
            log(f"poll error {type(exc).__name__}: {exc}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
