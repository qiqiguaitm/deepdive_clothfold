#!/usr/bin/env python3
"""Resource-aware dispatcher for the remaining RoboTwin experiment queue.

The dispatcher keeps work local until a target has real capacity. It monitors
the Volc queues and the two-GPU development host, enforces the 25-GPU Beijing
primary-account limit and the physical 32-GPU robot-task capacity, and records an atomic
state/snapshot under ``logs/``. The retired gf1 host remains in provenance but
is excluded from probing and dispatch.
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
import signal
import shlex
import statistics
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

try:
    from train_scripts.kai.volc import recommend_submission_target as submission_router
except ModuleNotFoundError:  # Direct execution adds only this directory to sys.path.
    import recommend_submission_target as submission_router


REPO = Path(__file__).resolve().parents[3]
QUEUE_PATH = REPO / "train_scripts/kai/volc/resource_scheduler_queue.json"
STATE_PATH = REPO / "logs/resource_scheduler_state.json"
SNAPSHOT_PATH = REPO / "logs/resource_scheduler_snapshot.json"
SNAPSHOT_MARKDOWN_PATH = REPO / "logs/resource_scheduler_snapshot.md"
LOG_PATH = REPO / "logs/resource_scheduler.log"
LOCK_PATH = REPO / "logs/resource_scheduler.lock"
P1_NORTH_FAILOVER_PROGRESS_PATH = REPO / "logs/pi05_p1_failover/progress.json"
P1_NORTH_FAILOVER_AUTH_AUDIT_PATH = (
    REPO / "logs/pi05_p1_failover/north_authorization_audit.json"
)
P1_NORTH_FAILOVER_STAGE = Path(
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/.staging/"
    "pi05_p1_failover_20260804T1034Z"
)
RECOMMENDATION_LOG_DIR = REPO / "logs/submission_recommendations"
ROBOT_TASK_DISABLE_MARKER = (
    REPO / "logs/resource_controls/robot_task_submission.disabled"
)
PERMANENTLY_DISABLED_RESOURCES = {
    "gf1": "operator retired gf1 after permanent host shutdown on 2026-08-04",
}
OWNER = "trn:iam::2113249311:user/suiyang.guo"
NORTH_QUEUE = "q-20260516104642-khch9"
SH_QUEUE = "q-20251204185107-fvnpx"
EAST_QUEUE = "q-20260516104437-2ml4v"
SH_CAPACITY = 32
NORTH_CAPACITY = 56
NORTH_PERSONAL_LIMIT = 25
NORTH_BACKUP_PERSONAL_LIMIT = int(os.environ.get("NORTH_BACKUP_PERSONAL_LIMIT", "20"))
NORTH_PRIMARY_MAX_JOBS = int(os.environ.get("NORTH_PRIMARY_MAX_JOBS", "25"))
NORTH_BACKUP_MAX_JOBS = int(os.environ.get("NORTH_BACKUP_MAX_JOBS", "20"))
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
REMOTE_LAUNCHER_DEAD_CONFIRMATIONS = 3
MAX_DISPATCHES_PER_POLL = 8
GATE_DECISION_SPECS = {
    str(REPO / "logs/resource_markers/pi05_mt1_seed1000_replication_gate.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_seed1000_gate.json",
        ("accepted_for_replication",),
    ),
    str(REPO / "logs/resource_markers/pi05_mt1_three_seed_gate.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_three_seed.json",
        ("gate", "accepted"),
    ),
    str(REPO / "logs/resource_markers/pi05_mt3_seed1000_replication_gate.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt3_seed1000_gate.json",
        ("accepted_for_replication",),
    ),
    str(REPO / "logs/resource_markers/pi05_mt3_three_seed.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt3_three_seed.json",
        ("gate", "accepted"),
    ),
    str(REPO / "logs/resource_markers/pi05_mt3_three_seed_beats_null.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt3_three_seed_vs_null.json",
        ("gate", "accepted"),
    ),
    str(REPO / "logs/resource_markers/pi05_mt3_three_seed_beats_within_task.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt3_three_seed_vs_within_task.json",
        ("gate", "accepted"),
    ),
    str(REPO / "logs/resource_markers/pi05_mt4_three_seed_content.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt4_content_gate.json",
        ("accepted",),
    ),
    str(REPO / "logs/resource_markers/pi05_mt5_complementarity.ok"): (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt5_three_seed.json",
        ("gate", "accepted"),
    ),
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
TRAIN_HEALTH_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|"
    r"CUDA out of memory|OutOfMemoryError|CUDNN_STATUS|"
    r"NCCL[^\n]*(?:error|failed)|FloatingPointError|"
    r"(?:loss|grad_norm|lmwm_loss|main_loss)=(?:nan|inf|-inf)\b",
    re.IGNORECASE,
)
GF1 = [
    "ssh",
    "-p",
    "7777",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "ConnectionAttempts=1",
    "root@14.103.218.231",
]
GSY = ["ssh", "-p", "16370", "-o", "BatchMode=yes", "root@124.174.16.237"]
PI05_MT12_SHARED_FINALIZERS = {
    "pi05_mt1_oracle_seed1000_correct_eval": (
        "pi05_mt1_oracle_seed1000_correct",
        "correct",
    ),
    "pi05_mt1_oracle_seed1000_null_eval": (
        "pi05_mt1_oracle_seed1000_null",
        "null",
    ),
    "pi05_mt1_oracle_seed1000_within_task_eval": (
        "pi05_mt1_oracle_seed1000_within_task",
        "within-task",
    ),
    "pi05_mt1_oracle_seed1000_cross_task_eval": (
        "pi05_mt1_oracle_seed1000_cross_task",
        "cross-task",
    ),
    "pi05_mt2_null_seed1000_eval": ("pi05_mt2_null_seed1000_eval", "null"),
    "pi05_mt1_oracle_seed1001_correct_eval": (
        "pi05_mt1_oracle_seed1001_correct",
        "correct",
    ),
    "pi05_mt1_oracle_seed1002_correct_eval": (
        "pi05_mt1_oracle_seed1002_correct",
        "correct",
    ),
}
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
    "pi05_mt1_oracle_seed1000": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_mt1_oracle_seed1000_4g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_mt1_oracle_seed1000_4g/launcher.log"
        ),
        "expected_steps": 50000,
    },
    "pi05_mt2_null_seed1000": {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_mt2_null_seed1000_4g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            "gf1_pi05_mt2_null_seed1000_4g/launcher.log"
        ),
        "expected_steps": 50000,
    },
}
for seed in (1001, 1002):
    GF1_TRAIN_WATCH_TASKS[f"pi05_mt1_oracle_seed{seed}"] = {
        "status_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            f"gf1_pi05_mt1_oracle_seed{seed}_4g/status"
        ),
        "log_path": (
            "/vePFS/tim/workspace/deepdive_kai0/logs/local_train/"
            f"gf1_pi05_mt1_oracle_seed{seed}_4g/launcher.log"
        ),
        "expected_steps": 50000,
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
            REPO
            / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_confirmatory_20260801_224940.log",
            REPO / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a2_abs_seed1000_*.log",
        ],
        "expected_steps": 50000,
    },
    "pi05_a3_live_seed1000": {
        "log_globs": [
            REPO
            / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_confirmatory_20260801_225018.log",
            REPO
            / "lmvla/lawam/logs/volc_robotwin/pi05_robotwin_a3_live_seed1000_*.log",
        ],
        "expected_steps": 50000,
    },
    "pi05_r1_crave_seed1000": {
        "log_glob": REPO / "logs/r1/platform/crave_*_east.log",
        "expected_steps": 50000,
    },
    "pi05_r1_combined_seed1000": {
        "log_glob": REPO / "logs/r1/platform/combined_*_east.log",
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
    ("gf1", "pi05_mt1_oracle_seed1001"): "pi05_mt1_oracle_seed1001_train",
    ("gf1", "pi05_mt1_oracle_seed1002"): "pi05_mt1_oracle_seed1002_train",
    (
        "Robot-East-H20",
        "pi05_a2_abs_seed1000",
    ): "pi05_a2_abs_confirmatory_seed1000_train",
    (
        "Robot-East-H20",
        "pi05_a3_live_seed1000",
    ): "pi05_a3_live_confirmatory_seed1000_train",
    (
        "Robot-East-H20",
        "pi05_r1_crave_seed1000",
    ): "pi05_r1_crave_seed1000_train",
    (
        "Robot-East-H20",
        "pi05_r1_combined_seed1000",
    ): "pi05_r1_combined_seed1000_train",
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
WAITING_STATES = ("Creating", "Waiting", "Queueing")
SUBMITTED_JOB_STATES = (*ACTIVE_STATES, *WAITING_STATES)
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
R1_FROZEN_OVERLAY = REPO / "logs/frozen_source_overlays/pi05_r1_v1"
REPLICATION_FROZEN_OVERLAY = (
    REPO / "logs/frozen_source_overlays/pi05_replication_v1"
)


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
                    REPO / "lmvla/lawam/results/eval_runs/robotwin/"
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
                    "root": str(
                        REPO / "lmvla/lawam/results/eval_runs/robotwin" / label
                    ),
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
                    "root": str(
                        REPO / "lmvla/lawam/results/eval_runs/robotwin" / label
                    ),
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
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _submission_input_paths(
    task: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    """Collect concrete data and checkpoint paths used by one candidate."""
    key = (
        "ready_files_remote"
        if candidate.get("resource") == "Robot-North-H20"
        else "ready_files"
    )
    values: list[str] = [*task.get(key, []), *candidate.get(key, [])]
    values.extend(str(value) for value in candidate.get("env", {}).values())
    command = candidate.get("command")
    if command:
        for token in shlex.split(command):
            value = token.split("=", 1)[-1] if "=" in token else token
            values.append(value)
    return sorted({value for value in values if value.startswith("/")})


def capture_submission_recommendation(
    task: dict[str, Any],
    candidate: dict[str, Any],
    snapshot: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Run the shared router and persist its decision before launching work."""
    catalog = submission_router.load_json(submission_router.DEFAULT_CATALOG)
    input_paths = _submission_input_paths(task, candidate)
    prefixes = [
        prefix
        for spec in catalog.get("filesystems", {}).values()
        for prefix in spec.get("mount_prefixes", [])
    ]
    known_paths = [
        path
        for path in input_paths
        if any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in prefixes
        )
    ]
    locations = submission_router.infer_filesystems(known_paths, catalog)
    recommendations = submission_router.rank_targets(
        gpus=int(candidate["gpus"]),
        catalog=catalog,
        snapshot=snapshot,
        data_locations=locations,
    )
    if not recommendations:
        raise ValueError(f"submission router returned no target for {task['id']}")
    eligible_resources = {
        item["resource"]
        for item in task.get("candidates", [])
        if int(item.get("gpus", -1)) == int(candidate["gpus"])
    }
    eligible = [item for item in recommendations if item.resource in eligible_resources]
    selected = next(
        (item for item in recommendations if item.resource == candidate["resource"]),
        None,
    )
    if selected is None or not eligible:
        raise ValueError(
            f"submission router cannot audit {task['id']} on {candidate['resource']}"
        )
    matches = selected.resource == eligible[0].resource
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "task_id": task["id"],
        "requested_gpus": int(candidate["gpus"]),
        "input_paths": known_paths,
        "data_locations": sorted(locations),
        "snapshot_timestamp": snapshot.get("timestamp"),
        "eligible_resources": sorted(eligible_resources),
        "global_recommendation": recommendations[0].resource,
        "task_eligible_recommendation": eligible[0].resource,
        "selected_resource": candidate["resource"],
        "selected_matches_task_recommendation": matches,
        "selection_analysis": (
            "selected the highest-ranked eligible task target"
            if matches
            else "higher-ranked task targets were unavailable, exhausted, or cooling down"
        ),
        "recommendations": [submission_router.asdict(item) for item in recommendations],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = RECOMMENDATION_LOG_DIR / task["id"] / f"{stamp}.json"
    atomic_json(path, payload)
    return path, payload


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def add_pi05_shared_eval_attach_tasks(queue: dict[str, Any]) -> None:
    """Use idle robot-task cards to attach workers to shared formal evaluations."""
    specs = (
        {
            "label": "a0_s1001",
            "parent": "pi05_a0_public_exact_seed1001_eval",
            "result": "pi05_rt_a0_public_exact_seed1001",
            "config": "pi05_robotwin_a0_public_exact_bj",
            "run_group": "pi05_robotwin_a0_public_exact_bj__demo_clean",
            "groups": 5,
            "checkpoint": (
                f"{REPO}/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/"
                "pi05_robotwin_a0_public_exact_seed1001/49999"
            ),
            "extra_env": {},
            "attach_run_tag_prefix": "confirmatory-seed",
        },
        {
            "label": "a2_s1000",
            "parent": "pi05_a2_abs_confirmatory_seed1000_eval",
            "result": "pi05_rt_a2_abs_confirmatory_s1000",
            "config": "pi05_robotwin_a2_prefix_official_eval_bj",
            "run_group": "pi05_robotwin_a2_abs_confirmatory__demo_clean",
            "groups": 3,
            "checkpoint": (
                f"{REPO}/kai0/checkpoints/pi05_robotwin_a2_abs_confirmatory/"
                "pi05_robotwin_a2_abs_seed1000/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a2_abs_confirmatory_eval.json"
                ),
                "ROBOTWIN_HINT_ENCODER": "so400m",
                "OPENPI_SERVER_HINT_ENCODER": "so400m",
                "EVAL_HINT_RESIDUAL": "0",
            },
        },
        {
            "label": "a3_s1000",
            "parent": "pi05_a3_live_confirmatory_seed1000_eval",
            "result": "pi05_rt_a3_live_confirmatory_s1000",
            "config": "pi05_robotwin_a3_live_residual_prefix_official_eval",
            "run_group": "pi05_robotwin_a3_live_confirmatory__demo_clean",
            "groups": 3,
            "checkpoint": (
                f"{REPO}/kai0/checkpoints/pi05_robotwin_a3_live_confirmatory/"
                "pi05_robotwin_a3_live_seed1000/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json"
                ),
            },
        },
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    for spec in specs:
        result_root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / spec["result"]
        scheduler_alternatives = []
        for tag_template in (
            "local-unseen-a3-seed{seed}",
            "confirmatory-seed{seed}",
            "exact-a0-seed{seed}",
        ):
            scheduler_alternatives.append(
                {
                    "ready_files": [
                        str(
                            result_root
                            / f"seed{seed}"
                            / spec["run_group"]
                            / tag_template.format(seed=seed)
                            / ".task_scheduler.json"
                        )
                        for seed in range(4)
                    ]
                }
            )
        marker_prefix = f"pi05_{spec['label']}_eval_attach"
        for group in range(1, spec["groups"] + 1):
            task_id = f"{marker_prefix}_cnsh_g{group}"
            if task_id in existing:
                continue
            group_name = f"cnsh_g{group}"
            env = {
                "ATTACH_SEEDS": "0 1 2 3",
                "WORKER_INDEX_BASE": str(5000 + group * 4000),
                "ATTACH_GROUP_NAME": group_name,
                "ATTACH_MARKER_PREFIX": marker_prefix,
                "RESULT_NAME": spec["result"],
                "PI05_EVAL_CONFIG_NAME": spec["config"],
                "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
                "CKPT": spec["checkpoint"],
                "ATTACH_RUN_TAG_PREFIX": spec.get("attach_run_tag_prefix", ""),
                **spec["extra_env"],
            }
            attach_command = shlex.join(
                [
                    "env",
                    *(f"{key}={value}" for key, value in env.items()),
                    "bash",
                    str(
                        REPO / "train_scripts/kai/eval/"
                        "attach_pi05_a0_confirmatory_platform.sh"
                    ),
                ]
            )
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": 4,
                    "description": (
                        f"Attach four robot-task workers to {spec['label']} formal eval "
                        f"({group_name})"
                    ),
                    "completion_glob": str(
                        REPO
                        / "logs/resource_markers"
                        / f"{marker_prefix}_{group_name}.ok"
                    ),
                    "completion_min_count": 1,
                    "satisfied_by_task": spec["parent"],
                    "ready_files": [
                        f"{spec['checkpoint']}/params/_METADATA",
                        f"{spec['checkpoint']}/_CHECKPOINT_METADATA",
                    ],
                    "ready_any": scheduler_alternatives,
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": [4, 5, 6, 7],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO
                                / "logs/local_eval"
                                / f"gf1_{marker_prefix}_{group_name}_4g"
                            ),
                            "command": attach_command,
                        },
                        {
                            "kind": "platform",
                            "resource": "robot-task",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "min_dispatch_free": 4,
                            "queue_timeout_seconds": 120,
                            "retry_cooldown_seconds": 300,
                            "yaml": (
                                "train_scripts/kai/volc/"
                                "pi05_a0_confirmatory_attach_cnsh_4a100.yaml"
                            ),
                            "task_name": f"pi05-{spec['label']}-attach-{group_name}-4g",
                            "env": env,
                        },
                    ],
                }
            )
            existing.add(task_id)


def add_pi05_mt_eval_attach_tasks(queue: dict[str, Any]) -> None:
    """Attach idle last-priority workers to active shared MT1/MT2 evaluations."""
    artifact = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1"
    episodes = (
        "/vePFS/tim/workspace/VLANeXt-main/datasets/"
        "robotwin2.0_official_prompts_v21/meta/episodes.jsonl"
    )
    specs = (
        (
            "mt1_correct",
            "pi05_mt1_oracle_seed1000_correct_eval",
            "correct",
            "pi05_robotwin_mt1_oracle_exact",
            "pi05_mt1_oracle_seed1000_correct",
        ),
        (
            "mt1_null",
            "pi05_mt1_oracle_seed1000_null_eval",
            "null",
            "pi05_robotwin_mt1_oracle_exact",
            "pi05_mt1_oracle_seed1000_null",
        ),
        (
            "mt1_within",
            "pi05_mt1_oracle_seed1000_within_task_eval",
            "within-task",
            "pi05_robotwin_mt1_oracle_exact",
            "pi05_mt1_oracle_seed1000_within_task",
        ),
        (
            "mt1_cross",
            "pi05_mt1_oracle_seed1000_cross_task_eval",
            "cross-task",
            "pi05_robotwin_mt1_oracle_exact",
            "pi05_mt1_oracle_seed1000_cross_task",
        ),
        (
            "mt2_null",
            "pi05_mt2_null_seed1000_eval",
            "null",
            "pi05_robotwin_mt2_null_exact",
            "pi05_mt2_null_seed1000_eval",
        ),
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    for index, (label, parent, intervention, config, result_name) in enumerate(specs):
        checkpoint = (
            REPO
            / "kai0/checkpoints"
            / config
            / (
                "pi05_robotwin_mt2_null_seed1000"
                if label == "mt2_null"
                else "pi05_robotwin_mt1_oracle_seed1000"
            )
            / "49999"
        )
        run_group = f"{config}__demo_clean"
        result_root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / result_name
        scheduler_files = [
            str(
                result_root
                / f"seed{seed}"
                / run_group
                / f"local-unseen-a3-seed{seed}"
                / ".task_scheduler.json"
            )
            for seed in range(4)
        ]
        marker_prefix = f"pi05_{label}_eval_attach"
        common_env = {
            "ATTACH_SEEDS": "0 1 2 3",
            "ATTACH_MARKER_PREFIX": marker_prefix,
            "ATTACH_RUN_TAG_PREFIX": "local-unseen-a3-seed",
            "RESULT_NAME": result_name,
            "PI05_EVAL_CONFIG_NAME": config,
            "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
            "CKPT": str(checkpoint),
            "ROBOTWIN_RUN_GROUP": run_group,
            "ROBOTWIN_TRANSITION_ORACLE": "1",
            "ROBOTWIN_TRANSITION_INTERVENTION": intervention,
            "ROBOTWIN_TRANSITION_PAIRS": str(artifact / "pairs.npz"),
            "ROBOTWIN_TRANSITION_TASK_MAP": str(artifact / "eval_task_id.json"),
            "ROBOTWIN_TRANSITION_EPISODES": episodes,
        }
        east_env = {
            **common_env,
            "WORKER_INDEX_BASE": str(28000 + index * 2000),
            "ATTACH_GPU_COUNT": "4",
            "ATTACH_GROUP_NAME": "east4g",
        }
        gf1_env = {
            **common_env,
            "WORKER_INDEX_BASE": str(29000 + index * 2000),
            "ATTACH_GPU_COUNT": "4",
            "ATTACH_GROUP_NAME": "gf1g4",
        }
        local_env = {
            **common_env,
            "WORKER_INDEX_BASE": str(31000 + index * 2000),
            "ATTACH_GPU_COUNT": "2",
            "ATTACH_GROUP_NAME": "local2g",
            "ATTACH_LOG_DIR": str(
                REPO / "logs/local_eval" / f"pi05_{label}_eval_attach_local2g"
            ),
        }
        local_attach_command = shlex.join(
            [
                "env",
                *(f"{key}={value}" for key, value in local_env.items()),
                "bash",
                str(
                    REPO
                    / "train_scripts/kai/eval/attach_pi05_a0_confirmatory_platform.sh"
                ),
            ]
        )
        permission_command = shlex.join(
            [
                *GF1,
                f"chmod -R a+rwX {shlex.quote(str(result_root))}",
            ]
        )
        local_command = f"{permission_command} && {local_attach_command}"
        platform_env = {
            **common_env,
            "WORKER_INDEX_BASE": str(30000 + index * 2000),
            "ATTACH_GPU_COUNT": "4",
            "ATTACH_GROUP_NAME": "cnsh4g",
        }
        gf1_candidates = []
        for gpu_indices in ([0, 1, 2, 3], [4, 5, 6, 7]):
            gf1_command = shlex.join(
                [
                    "env",
                    f"ATTACH_GPU_INDEX_BASE={gpu_indices[0]}",
                    *(f"{key}={value}" for key, value in gf1_env.items()),
                    "bash",
                    str(
                        REPO
                        / "train_scripts/kai/eval/attach_pi05_a0_confirmatory_platform.sh"
                    ),
                ]
            )
            gf1_candidates.append(
                {
                    "kind": "ssh",
                    "resource": "gf1",
                    "gpus": 4,
                    "gpu_indices": gpu_indices,
                    "retry_cooldown_seconds": 300,
                    "status_dir": str(
                        REPO / "logs/local_eval" / f"pi05_{label}_eval_attach_gf1g4"
                    ),
                    "command": gf1_command,
                }
            )
        helpers = (
            (
                f"pi05_{label}_eval_attach_gf1g4",
                "gf1g4",
                3,
                gf1_candidates,
            ),
            (
                f"pi05_{label}_eval_attach_east4g",
                "east4g",
                4,
                [
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "min_dispatch_free": 4,
                        "queue_timeout_seconds": 120,
                        "retry_cooldown_seconds": 300,
                        "ready_files": [
                            str(
                                REPO / "logs/resource_markers/robotwin_renderer_east.ok"
                            )
                        ],
                        "yaml": "train_scripts/kai/volc/pi05_mt_transition_attach_east_4h20.yaml",
                        "task_name": f"pi05-{label}-attach-east4g",
                        "env": east_env,
                    }
                ],
            ),
            (
                f"pi05_{label}_eval_attach_cnsh4g",
                "cnsh4g",
                5,
                [
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "min_dispatch_free": 4,
                        "queue_timeout_seconds": 120,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_a0_confirmatory_attach_cnsh_4a100.yaml",
                        "task_name": f"pi05-{label}-attach-cnsh4g",
                        "env": platform_env,
                    }
                ],
            ),
            (
                f"pi05_{label}_eval_attach_local2g",
                "local2g",
                6,
                [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 2,
                        "gpu_indices": [0, 1],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO
                            / "logs/local_eval"
                            / f"pi05_{label}_eval_attach_local2g"
                        ),
                        "command": local_command,
                    }
                ],
            ),
        )
        for task_id, group_name, priority, candidates in helpers:
            if task_id in existing:
                continue
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": priority,
                    "description": (
                        f"Attach {group_name} workers to active {label} formal evaluation"
                    ),
                    "completion_glob": str(
                        REPO
                        / "logs/resource_markers"
                        / f"{marker_prefix}_{group_name}.ok"
                    ),
                    "completion_min_count": 1,
                    "satisfied_by_task": parent,
                    "ready_files": [
                        str(checkpoint / "params/_METADATA"),
                        str(checkpoint / "_CHECKPOINT_METADATA"),
                        *scheduler_files,
                    ],
                    "candidates": candidates,
                }
            )
            existing.add(task_id)


def add_pi05_mt1_replication_eval_attach_tasks(queue: dict[str, Any]) -> None:
    """Attach independent idle-resource workers to MT1 replication evaluations."""
    artifact = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1"
    episodes = (
        "/vePFS/tim/workspace/VLANeXt-main/datasets/"
        "robotwin2.0_official_prompts_v21/meta/episodes.jsonl"
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    for index, seed in enumerate((1001, 1002)):
        parent = f"pi05_mt1_oracle_seed{seed}_correct_eval"
        config = "pi05_robotwin_mt1_oracle_exact"
        result_name = f"pi05_mt1_oracle_seed{seed}_correct"
        checkpoint = (
            REPO
            / "kai0/checkpoints"
            / config
            / f"pi05_robotwin_mt1_oracle_seed{seed}"
            / "49999"
        )
        run_group = f"{config}__demo_clean"
        result_root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / result_name
        scheduler_files = [
            str(
                result_root
                / f"seed{eval_seed}"
                / run_group
                / f"local-unseen-a3-seed{eval_seed}"
                / ".task_scheduler.json"
            )
            for eval_seed in range(4)
        ]
        common_env = {
            "ATTACH_SEEDS": "0 1 2 3",
            "ATTACH_RUN_TAG_PREFIX": "local-unseen-a3-seed",
            "RESULT_NAME": result_name,
            "PI05_EVAL_CONFIG_NAME": config,
            "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
            "CKPT": str(checkpoint),
            "ROBOTWIN_RUN_GROUP": run_group,
            "ROBOTWIN_TRANSITION_ORACLE": "1",
            "ROBOTWIN_TRANSITION_INTERVENTION": "correct",
            "ROBOTWIN_TRANSITION_PAIRS": str(artifact / "pairs.npz"),
            "ROBOTWIN_TRANSITION_TASK_MAP": str(artifact / "eval_task_id.json"),
            "ROBOTWIN_TRANSITION_EPISODES": episodes,
        }
        small_robot_groups = (
            tuple(
                (
                    f"cnsh2g_{shard}",
                    "robot-task",
                    2,
                    5,
                    {
                        "kind": "platform",
                        "region": "cn-shanghai",
                        "min_dispatch_free": 2,
                        "queue_timeout_seconds": 900,
                        "yaml": (
                            "train_scripts/kai/volc/"
                            "pi05_a0_confirmatory_attach_cnsh_2a100.yaml"
                        ),
                    },
                )
                for shard in range(4)
            )
            if seed == 1001
            else ()
        )
        groups = (
            (
                "gf1g4",
                "gf1",
                4,
                3,
                {
                    "kind": "ssh",
                    "gpu_indices": list(range((1 - index) * 4, (1 - index) * 4 + 4)),
                },
            ),
            (
                "east4g",
                "Robot-East-H20",
                4,
                4,
                {
                    "kind": "platform",
                    "region": "cn-shanghai",
                    "min_dispatch_free": 4,
                    "queue_timeout_seconds": 900,
                    "ready_files": [
                        str(REPO / "logs/resource_markers/robotwin_renderer_east.ok")
                    ],
                    "yaml": "train_scripts/kai/volc/pi05_mt_transition_attach_east_4h20.yaml",
                },
            ),
            (
                "cnsh4g",
                "robot-task",
                4,
                5,
                {
                    "kind": "platform",
                    "region": "cn-shanghai",
                    "min_dispatch_free": 4,
                    "queue_timeout_seconds": 900,
                    "yaml": "train_scripts/kai/volc/pi05_a0_confirmatory_attach_cnsh_4a100.yaml",
                },
            ),
            *(
                (
                    f"cnsh4g_{shard}",
                    "robot-task",
                    4,
                    5,
                    {
                        "kind": "platform",
                        "region": "cn-shanghai",
                        "min_dispatch_free": 4,
                        "queue_timeout_seconds": 900,
                        "yaml": (
                            "train_scripts/kai/volc/"
                            "pi05_a0_confirmatory_attach_cnsh_4a100.yaml"
                        ),
                    },
                )
                for shard in range(1, 4)
            ),
            (
                "local2g",
                "local",
                2,
                6,
                {
                    "kind": "local",
                    "gpu_indices": [0, 1],
                },
            ),
            *small_robot_groups,
        )
        for group_index, (
            group,
            resource,
            gpus,
            priority,
            candidate_extra,
        ) in enumerate(groups):
            task_id = f"pi05_mt1_oracle_seed{seed}_correct_eval_attach_{group}"
            if task_id in existing:
                continue
            marker_prefix = task_id
            env = {
                **common_env,
                "WORKER_INDEX_BASE": str(20000 + index * 4000 + group_index * 400),
                "ATTACH_MARKER_PREFIX": marker_prefix,
                "ATTACH_GPU_COUNT": str(gpus),
                "ATTACH_GROUP_NAME": group,
            }
            if resource == "local":
                env["ATTACH_LOG_DIR"] = str(REPO / "logs/local_eval" / task_id)
            elif resource == "gf1":
                env["ATTACH_GPU_INDEX_BASE"] = str((1 - index) * 4)
            candidate = {
                **candidate_extra,
                "resource": resource,
                "gpus": gpus,
                "retry_cooldown_seconds": 300,
            }
            if candidate["kind"] == "platform":
                candidate.update(
                    {
                        "task_name": f"pi05-mt1-s{seed}-correct-attach-{group}",
                        "env": env,
                    }
                )
            else:
                attach_command = shlex.join(
                    [
                        "env",
                        *(f"{key}={value}" for key, value in env.items()),
                        "bash",
                        str(
                            REPO
                            / "train_scripts/kai/eval/attach_pi05_a0_confirmatory_platform.sh"
                        ),
                    ]
                )
                if resource == "gf1":
                    command = (
                        f"chmod -R a+rwX {shlex.quote(str(result_root))} && "
                        f"{attach_command}"
                    )
                else:
                    permission_command = shlex.join(
                        [*GF1, f"chmod -R a+rwX {shlex.quote(str(result_root))}"]
                    )
                    command = f"{permission_command} && {attach_command}"
                candidate.update(
                    {
                        "status_dir": str(REPO / "logs/local_eval" / task_id),
                        "command": command,
                    }
                )
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": priority,
                    "description": (
                        f"Attach {resource} workers to MT1 seed {seed} correct evaluation"
                    ),
                    "completion_glob": str(
                        REPO / "logs/resource_markers" / f"{marker_prefix}_*.ok"
                    ),
                    "completion_min_count": 1,
                    "satisfied_by_task": parent,
                    "ready_files": [
                        str(checkpoint / "params/_METADATA"),
                        str(checkpoint / "_CHECKPOINT_METADATA"),
                        *scheduler_files,
                    ],
                    "candidates": [candidate],
                }
            )
            existing.add(task_id)


def add_pi05_mt1_replication_north_overflow(queue: dict[str, Any]) -> None:
    """Stage seed 1002 to North while seed 1001 immediately occupies GF1."""
    seed = 1002
    config = "pi05_robotwin_mt1_oracle_exact"
    exp = f"pi05_robotwin_mt1_oracle_seed{seed}"
    result_name = f"pi05_mt1_oracle_seed{seed}_correct"
    parent_id = f"pi05_mt1_oracle_seed{seed}_correct_eval"
    local_checkpoint = REPO / "kai0/checkpoints" / config / exp / "49999"
    north_checkpoint = Path(NORTH_REPO) / "kai0/checkpoints" / config / exp / "49999"
    local_marker = REPO / "logs/resource_markers" / f"{result_name}.ok"
    north_marker = Path(NORTH_REPO) / "logs/resource_markers" / f"{result_name}.ok"
    stage_marker = (
        REPO / "logs/resource_markers" / f"pi05_mt1_seed{seed}_north_eval_checkpoint.ok"
    )
    decision_marker = (
        REPO / "logs/resource_markers" / f"pi05_mt1_seed{seed}_north_stage_decided.ok"
    )
    code_marker = REPO / "logs/resource_markers/pi05_mt_north_code_sync.ok"
    run_group = f"{config}__demo_clean"
    north_result_root = (
        Path(NORTH_REPO) / "lmvla/lawam/results/eval_runs/robotwin" / result_name
    )
    tasks = {task.get("id"): task for task in queue.get("tasks", [])}
    parent = tasks.get(parent_id)
    if parent is None:
        raise ValueError(f"missing MT1 replication parent: {parent_id}")

    if str(decision_marker) not in parent["ready_files"]:
        parent["ready_files"].append(str(decision_marker))
    parent["completion_locations"] = [
        {"label": "shared", "glob": str(local_marker), "remote": False},
        {"label": "north", "glob": str(north_marker), "remote": True},
    ]
    if not any(
        candidate.get("resource") == "Robot-North-H20"
        for candidate in parent["candidates"]
    ):
        parent["candidates"].append(
            {
                "kind": "platform",
                "resource": "Robot-North-H20",
                "region": "cn-beijing",
                "gpus": 4,
                "queue_timeout_seconds": 300,
                "retry_cooldown_seconds": 300,
                "ready_files": [str(code_marker), str(stage_marker)],
                "ready_files_remote": [
                    str(north_checkpoint / "params/_METADATA"),
                    str(north_checkpoint / "_CHECKPOINT_METADATA"),
                ],
                "yaml": "train_scripts/kai/volc/pi05_mt_transition_eval_north_4h20.yaml",
                "task_name": f"pi05-mt1-correct-s{seed}-eval-north4g",
                "env": {
                    "PI05_EVAL_CONFIG_NAME": config,
                    "CKPT": str(north_checkpoint),
                    "ROBOTWIN_TRANSITION_INTERVENTION": "correct",
                    "RESULT_NAME": result_name,
                    "MARKER": str(north_marker),
                    "PORT_BASE_OFFSET": "20600",
                },
            }
        )

    if f"pi05_mt1_seed{seed}_sync_north_eval_checkpoint" not in tasks:
        queue["tasks"].append(
            {
                "id": f"pi05_mt1_seed{seed}_sync_north_eval_checkpoint",
                "priority": 0,
                "description": (
                    f"Verified eval-only sync of final MT1 seed {seed} checkpoint to North"
                ),
                "completion_glob": str(decision_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(code_marker),
                    str(local_checkpoint / "params/_METADATA"),
                    str(local_checkpoint / "_CHECKPOINT_METADATA"),
                    str(
                        REPO
                        / "logs/resource_markers"
                        / f"pi05_mt1_oracle_seed{seed}_step49999_checkpoint_audit.ok"
                    ),
                ],
                "progress_logs": [
                    {
                        "label": "phase",
                        "glob": str(
                            REPO
                            / "logs/sync"
                            / f"pi05_mt1_seed{seed}_north_eval_checkpoint/launcher.log"
                        ),
                        "regex": r"phase=([a-z0-9-]+)",
                    },
                    {
                        "label": "transfer",
                        "glob": str(
                            REPO
                            / "logs/sync"
                            / f"pi05_mt1_seed{seed}_north_eval_checkpoint/launcher.log"
                        ),
                        "regex": r"([0-9]+(?:\.[0-9]+)?)%",
                    },
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(
                            REPO
                            / "logs/sync"
                            / f"pi05_mt1_seed{seed}_north_eval_checkpoint"
                        ),
                        "command": shlex.join(
                            [
                                "env",
                                f"SEED={seed}",
                                f"STAGE_MARKER={stage_marker}",
                                f"DECISION_MARKER={decision_marker}",
                                "bash",
                                str(
                                    REPO
                                    / "train_scripts/kai/stage_pi05_mt1_north_overflow.sh"
                                ),
                            ]
                        ),
                    }
                ],
            }
        )

    materialize_id = f"pi05_mt1_seed{seed}_correct_sync_from_north"
    if materialize_id not in tasks:
        queue["tasks"].append(
            {
                "id": materialize_id,
                "priority": 0,
                "description": (
                    f"Verify and materialize MT1 seed {seed} evaluation produced on North"
                ),
                "materialize_north_result_for": parent_id,
                "completion_glob": str(local_marker),
                "completion_min_count": 1,
                "ready_files_remote": [str(north_marker)],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/sync" / result_name),
                        "command": shlex.join(
                            [
                                "env",
                                f"RESULT_NAME={result_name}",
                                "INTERVENTION=correct",
                                f"CKPT={local_checkpoint}",
                                "bash",
                                str(
                                    REPO
                                    / "train_scripts/kai/sync_pi05_mt_eval_from_north.sh"
                                ),
                            ]
                        ),
                    }
                ],
            }
        )

    scheduler_files = [
        str(
            north_result_root
            / f"seed{eval_seed}"
            / run_group
            / f"local-unseen-a3-seed{eval_seed}"
            / ".task_scheduler.json"
        )
        for eval_seed in range(4)
    ]
    artifact = (
        Path(NORTH_REPO) / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1"
    )
    common_attach_env = {
        "ATTACH_SEEDS": "0 1 2 3",
        "ATTACH_RUN_TAG_PREFIX": "local-unseen-a3-seed",
        "RESULT_NAME": result_name,
        "PI05_EVAL_CONFIG_NAME": config,
        "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
        "CKPT": str(north_checkpoint),
        "ROBOTWIN_RUN_GROUP": run_group,
        "ROBOTWIN_TRANSITION_ORACLE": "1",
        "ROBOTWIN_TRANSITION_INTERVENTION": "correct",
        "ROBOTWIN_TRANSITION_PAIRS": str(artifact / "pairs.npz"),
        "ROBOTWIN_TRANSITION_TASK_MAP": str(artifact / "eval_task_id.json"),
        "ROBOTWIN_TRANSITION_EPISODES": (
            "/vePFS-North-E/vis_robot/huanqian/uniVP/data/robotwin2.0/"
            "robotwin2.0_official_prompts_v21/meta/episodes.jsonl"
        ),
    }

    attach_id = f"pi05_mt1_oracle_seed{seed}_correct_eval_attach_bj2g"
    if attach_id not in tasks:
        env = {
            **common_attach_env,
            "ATTACH_GPU_COUNT": "2",
            "WORKER_INDEX_BASE": "17000",
            "ATTACH_GROUP_NAME": "bj2g",
            "ATTACH_MARKER_PREFIX": attach_id,
        }
        queue["tasks"].append(
            {
                "id": attach_id,
                "priority": 4,
                "description": f"Attach two North workers to MT1 seed {seed} evaluation",
                "completion_glob": str(
                    Path(NORTH_REPO) / "logs/resource_markers" / f"{attach_id}_bj2g.ok"
                ),
                "completion_remote": True,
                "completion_min_count": 1,
                "satisfied_by_task": parent_id,
                "ready_files_remote": [
                    str(north_checkpoint / "params/_METADATA"),
                    str(north_checkpoint / "_CHECKPOINT_METADATA"),
                    *scheduler_files,
                ],
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-North-H20",
                        "region": "cn-beijing",
                        "gpus": 2,
                        "queue_timeout_seconds": 300,
                        "retry_cooldown_seconds": 300,
                        "max_failures": 6,
                        "yaml": "train_scripts/kai/volc/pi05_confirmatory_attach_bj_2h20.yaml",
                        "task_name": f"pi05-mt1-s{seed}-correct-attach-bj2g",
                        "env": env,
                    }
                ],
            }
        )

    for group_index in range(4):
        group = f"bj4g{group_index}"
        group_id = f"pi05_mt1_oracle_seed{seed}_correct_eval_attach_{group}"
        if group_id in tasks:
            continue
        env = {
            **common_attach_env,
            "ATTACH_GPU_COUNT": "4",
            "WORKER_INDEX_BASE": str(12000 + group_index * 800),
            "ATTACH_GROUP_NAME": group,
            "ATTACH_MARKER_PREFIX": group_id,
        }
        queue["tasks"].append(
            {
                "id": group_id,
                "priority": 4,
                "description": (
                    f"Attach North four-worker group {group_index} to MT1 seed {seed} evaluation"
                ),
                "completion_glob": str(
                    Path(NORTH_REPO)
                    / "logs/resource_markers"
                    / f"{group_id}_{group}.ok"
                ),
                "completion_remote": True,
                "completion_min_count": 1,
                "satisfied_by_task": parent_id,
                "ready_files_remote": [
                    str(north_checkpoint / "params/_METADATA"),
                    str(north_checkpoint / "_CHECKPOINT_METADATA"),
                    *scheduler_files,
                ],
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-North-H20",
                        "region": "cn-beijing",
                        "gpus": 4,
                        "queue_timeout_seconds": 300,
                        "retry_cooldown_seconds": 300,
                        "max_failures": 6,
                        "yaml": "train_scripts/kai/volc/pi05_confirmatory_attach_bj_4h20.yaml",
                        "task_name": f"pi05-mt1-s{seed}-correct-attach-{group}",
                        "env": env,
                    }
                ],
            }
        )


def add_pi05_mt3_eval_attach_tasks(queue: dict[str, Any]) -> None:
    """Stage conditional shared-worker acceleration for MT3/MT4 evaluations."""
    specs = [
        (1000, intervention)
        for intervention in ("predicted", "within_task", "null", "oracle")
    ]
    specs.extend(
        (seed, intervention)
        for seed in (1001, 1002)
        for intervention in ("predicted", "null", "within_task")
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
    wrapper = REPO / "train_scripts/kai/eval/attach_pi05_mt3_formal.sh"
    for seed, intervention in specs:
        task_id = f"pi05_mt3_seed{seed}_{intervention}_eval_attach"
        if task_id in existing:
            continue
        parent = f"pi05_mt3_learned_seed{seed}_{intervention}_eval"
        result_name = f"pi05_mt3_learned_seed{seed}_{intervention}"
        checkpoint = (
            REPO
            / "kai0/checkpoints/pi05_robotwin_mt3_learned_exact"
            / f"pi05_robotwin_mt3_learned_seed{seed}/49999"
        )
        result_root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / result_name
        scheduler_alternatives = []
        for candidate in ("current_frame", "history_proprio"):
            config = f"pi05_robotwin_mt3_learned_{candidate}_exact"
            scheduler_alternatives.append(
                {
                    "ready_files": [
                        str(
                            result_root
                            / f"seed{eval_seed}"
                            / f"{config}__demo_clean"
                            / f"local-unseen-a3-seed{eval_seed}"
                            / ".task_scheduler.json"
                        )
                        for eval_seed in range(4)
                    ]
                }
            )
        marker_prefix = f"pi05_mt3_seed{seed}_{intervention}_eval_attach"
        common_env = {
            "MT3_INTERVENTION": intervention,
            "MT3_SELECTION": str(selection),
            "CKPT": str(checkpoint),
            "RESULT_NAME": result_name,
            "EVAL_WORKERS_PER_GPU": "2",
            "ATTACH_SEEDS": "0 1 2 3",
            "ATTACH_MARKER_PREFIX": marker_prefix,
            "ATTACH_RUN_TAG_PREFIX": "local-unseen-a3-seed",
        }
        local_env = {
            **common_env,
            "ATTACH_GPU_COUNT": "2",
            "ATTACH_GROUP_NAME": "local2g",
            "WORKER_INDEX_BASE": "30000",
        }
        local_command = shlex.join(
            [
                "env",
                *(f"{key}={value}" for key, value in local_env.items()),
                "bash",
                str(wrapper),
            ]
        )
        platform_env = {
            **common_env,
            "ATTACH_GPU_COUNT": "4",
            "ATTACH_GROUP_NAME": "cnsh4g",
            "WORKER_INDEX_BASE": "28000",
        }
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 10,
                "description": (
                    f"Attach idle workers to MT3 seed {seed} {intervention} evaluation"
                ),
                "completion_glob": str(
                    REPO / "logs/resource_markers" / f"{marker_prefix}_*.ok"
                ),
                "completion_min_count": 1,
                "satisfied_by_task": parent,
                "ready_files": [
                    str(selected_marker),
                    str(selection),
                    str(checkpoint / "params/_METADATA"),
                    str(checkpoint / "_CHECKPOINT_METADATA"),
                ],
                "ready_any": scheduler_alternatives,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "min_dispatch_free": 4,
                        "queue_timeout_seconds": 120,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_mt3_attach_cnsh_4a100.yaml",
                        "task_name": f"pi05-mt3-s{seed}-{intervention}-attach-cnsh4g",
                        "env": platform_env,
                    },
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 2,
                        "gpu_indices": [0, 1],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO / "logs/local_eval" / f"{task_id}_local2g"
                        ),
                        "command": local_command,
                    },
                ],
            }
        )
        existing.add(task_id)


def add_pi05_mt1_8g_optimization_probes(queue: dict[str, Any]) -> None:
    """Benchmark 8-GPU input and hybrid-FSDP layouts at frozen global batch 16."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    specs = (
        ("w16-fsdp1", 16, 1),
        ("w16-fsdp2", 16, 2),
    )
    for label, workers, fsdp_devices in specs:
        task_id = f"pi05_mt1_b16_8g_opt_{label.replace('-', '_')}"
        if task_id in existing:
            continue
        probe_exp = f"pi05_mt1_b16_8g_opt_{label}"
        result = REPO / "logs/scaling" / f"{probe_exp}.json"
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 89,
                "description": (
                    "Non-claim-bearing 8xA100 MT1 throughput optimization "
                    f"workers={workers} fsdp_devices={fsdp_devices}"
                ),
                "completion_glob": str(result),
                "completion_min_count": 1,
                "ready_files": [
                    str(REPO / "kai0/checkpoints/pi05_base/params/_METADATA"),
                    str(
                        REPO
                        / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
                    ),
                    str(REPO / "train_scripts/kai/run_pi05_mt1_scaling_probe.sh"),
                ],
                "progress_logs": [
                    {
                        "label": "steps",
                        "glob": str(REPO / "logs/scaling" / f"{probe_exp}.log"),
                        "regex": "Step ([0-9]+)",
                        "total": 300,
                    }
                ],
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 8,
                        "min_dispatch_free": 8,
                        "queue_timeout_seconds": 900,
                        "retry_cooldown_seconds": 300,
                        "yaml": (
                            "train_scripts/kai/volc/"
                            "pi05_mt1_scaling_probe_cnsh_8a100.yaml"
                        ),
                        "task_name": f"pi05-mt1-b16-8g-opt-{label}",
                        "env": {
                            "PROBE_EXP": probe_exp,
                            "RESULT": str(result),
                            "WORKERS": str(workers),
                            "FSDP_DEVICES": str(fsdp_devices),
                        },
                    }
                ],
            }
        )
        existing.add(task_id)


def add_pi05_p1_north_failover_tasks(queue: dict[str, Any]) -> None:
    """Resume the source-frozen P1 pair from its verified North stage."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    parent_id = "pi05_p1_north_failover_pair"
    local_dir = REPO / "logs/pi05_p1_failover"
    remote_report = P1_NORTH_FAILOVER_STAGE / "pi05_p1_north_pair_training_report.json"
    local_report = local_dir / "north_pair_training_report.json"
    remote_final_metadata = str(
        P1_NORTH_FAILOVER_STAGE
        / "kai0/checkpoints/pi05_predictive_adapter_p1*"
        / "pi05_predictive_adapter_p1*_seed1000/49999/_CHECKPOINT_METADATA"
    )
    candidate_final = (
        P1_NORTH_FAILOVER_STAGE
        / "kai0/checkpoints/pi05_predictive_adapter_p1"
        / "pi05_predictive_adapter_p1_seed1000/49999"
    )

    if parent_id not in existing:
        queue["tasks"].append(
            {
                "id": parent_id,
                "priority": 0,
                "description": (
                    "Authorized P1 A0/candidate paired resume from frozen step-10000 "
                    "optimizer states"
                ),
                "ready_files": [
                    str(local_dir / "north_stage_report.json"),
                    str(local_dir / "north_runtime_preflight.json"),
                    str(local_dir / "north_authorization_audit.json"),
                    str(local_dir / "north_container_runtime_amendment.json"),
                    str(
                        REPO
                        / "train_scripts/kai/volc/"
                        "pi05_p1_north_failover_pair_8h20.yaml"
                    ),
                ],
                "ready_hashes": [
                    {
                        "path": str(
                            local_dir / "north_container_runtime_amendment.json"
                        ),
                        "sha256": (
                            "44f94287cedfccb6c1a02f1029199e0fdbee7e1efc4a1ad9"
                            "635ea971047408af"
                        ),
                    }
                ],
                "completion_glob": remote_final_metadata,
                "completion_min_count": 2,
                "completion_remote": True,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-North-H20",
                        "region": "cn-beijing",
                        "gpus": 8,
                        "max_failures": 6,
                        "queue_timeout_seconds": 300,
                        "retry_cooldown_seconds": 300,
                        "yaml": (
                            "train_scripts/kai/volc/"
                            "pi05_p1_north_failover_pair_8h20.yaml"
                        ),
                        "task_name": "pi05-p1-north-failover-pair",
                        "env": {
                            "PATH": (
                                f"{P1_NORTH_FAILOVER_STAGE}/runtime/bin:"
                                "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                                "/usr/bin:/sbin:/bin"
                            ),
                            "LD_LIBRARY_PATH": (
                                f"{P1_NORTH_FAILOVER_STAGE}/runtime/lib"
                            ),
                            "PYTHONPATH": (
                                f"{P1_NORTH_FAILOVER_STAGE}/kai0/src"
                            ),
                        },
                        "ready_files_remote": [
                            str(P1_NORTH_FAILOVER_STAGE / "north_stage_report.json"),
                            str(
                                P1_NORTH_FAILOVER_STAGE
                                / "pi05_p1_north_runtime_preflight.json"
                            ),
                            str(
                                P1_NORTH_FAILOVER_STAGE
                                / "pi05_p1_north_failover_authorization.json"
                            ),
                            str(
                                P1_NORTH_FAILOVER_STAGE
                                / "logs/pi05_p1_failover/authorization_audit.json"
                            ),
                            str(
                                P1_NORTH_FAILOVER_STAGE
                                / "north_container_runtime_amendment.json"
                            ),
                        ],
                    }
                ],
            }
        )
        existing.add(parent_id)

    # Once the candidate arm is complete, reserve only the four GPUs required
    # to resume A0. Candidate-specific remote readiness prevents this path from
    # being selected for a fresh two-arm recovery.
    parent = next(task for task in queue["tasks"] if task.get("id") == parent_id)
    parent["prefer_min_gpus_when_immediate"] = True
    recovery_yaml = "train_scripts/kai/volc/pi05_p1_north_failover_a0_resume_4h20.yaml"
    if not any(candidate.get("yaml") == recovery_yaml for candidate in parent["candidates"]):
        parent["candidates"].insert(
            0,
            {
                "kind": "platform",
                "resource": "Robot-North-H20",
                "region": "cn-beijing",
                "gpus": 4,
                "max_failures": 6,
                "queue_timeout_seconds": 300,
                "retry_cooldown_seconds": 300,
                "yaml": recovery_yaml,
                "task_name": "pi05-p1-north-failover-a0-resume",
                "ready_files_remote": [
                    str(candidate_final / "_CHECKPOINT_METADATA"),
                    str(candidate_final / "params/_METADATA"),
                ],
            },
        )

    materialize_id = "pi05_p1_north_failover_materialize"
    materialize_marker = (
        REPO / "logs/resource_markers/pi05_p1_north_failover_materialized.ok"
    )
    if materialize_id not in existing:
        queue["tasks"].append(
            {
                "id": materialize_id,
                "priority": 0,
                "description": "Atomically materialize the completed North P1 pair",
                "materialize_north_result_for": parent_id,
                "completion_glob": str(materialize_marker),
                "completion_min_count": 1,
                "produces_files": [str(local_report), str(materialize_marker)],
                "ready_files": [
                    str(REPO / "train_scripts/kai/sync_pi05_p1_north_failover_results.sh")
                ],
                "ready_files_remote": [str(remote_report)],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(local_dir / "north_result_sync"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec bash "
                            "train_scripts/kai/sync_pi05_p1_north_failover_results.sh"
                        ),
                    }
                ],
            }
        )


def add_pi05_r1_recurrence_aligned_tasks(queue: dict[str, Any]) -> None:
    """Stage the double-gated R1 screen and conditional three-seed replication."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    p0_gate = REPO / "logs/predictive/p0_eval/p0_gate.accepted"
    r0_gate = REPO / "logs/crave_r0/probe_gate/r0_gate.accepted"
    labels = REPO / "lmvla/lmwm/data/pi05_crave_r0_v1"
    protocol = REPO / "lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json"
    scene = REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    overlay_python = R1_FROZEN_OVERLAY / "kai0/src"

    checkpoint_roots = {
        "crave": REPO / "kai0/checkpoints/pi05_r1_crave/pi05_r1_crave_seed1000/49999",
        "combined": REPO
        / "kai0/checkpoints/pi05_r1_combined/pi05_r1_combined_seed1000/49999",
    }
    for arm, checkpoint in checkpoint_roots.items():
        task_id = f"pi05_r1_{arm}_seed1000_train"
        if task_id in existing:
            continue
        gpu_indices = [0, 1, 2, 3] if arm == "crave" else [4, 5, 6, 7]
        command = (
            f"cd {shlex.quote(str(REPO))} && export CUDA_VISIBLE_DEVICES="
            f"{','.join(map(str, gpu_indices))} && exec env R1_ARM={arm} "
            "SEED=1000 GPU_COUNT=4 bash train_scripts/kai/run_pi05_r1_train.sh"
        )
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 1,
                "description": f"P0+R0-gated R1 {arm} seed-1000 training",
                "completion_glob": str(checkpoint / "_CHECKPOINT_METADATA"),
                "completion_min_count": 1,
                "ready_files": [
                    str(p0_gate),
                    str(r0_gate),
                    str(labels / "READY_LABELS"),
                    str(labels / "probe_train.npz"),
                    str(labels / "labels_manifest.json"),
                    str(protocol),
                    str(REPO / "train_scripts/kai/run_pi05_r1_train.sh"),
                ],
                "candidates": [
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 4,
                        "gpu_indices": gpu_indices,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / f"logs/r1/train_{arm}_gf1"),
                        "command": command,
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r1_train_east_4h20.yaml",
                        "task_name": f"pi05-r1-{arm}-s1000-east4g",
                        "env": {"R1_ARM": arm, "SEED": "1000"},
                    },
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r1_train_cnsh_4a100.yaml",
                        "task_name": f"pi05-r1-{arm}-s1000-cnsh4g",
                        "env": {"R1_ARM": arm, "SEED": "1000"},
                    },
                ],
            }
        )
        existing.add(task_id)

    conditions = {
        "crave": ("crave", 0),
        "combined": ("combined", 4),
        "combined_zero_gate": ("combined", 0),
        "combined_shuffled": ("combined", 4),
    }
    for condition, (arm, gpu_offset) in conditions.items():
        task_id = f"pi05_r1_{condition}_seed1000_eval"
        if task_id in existing:
            continue
        result_name = f"pi05_r1_seed1000_{condition}"
        marker = REPO / "logs/resource_markers" / f"{result_name}.ok"
        checkpoint = checkpoint_roots[arm]
        port_base = 22800 + gpu_offset * 20
        local_command = (
            f"cd {shlex.quote(str(REPO))} && export PYTHONPATH="
            f"{shlex.quote(str(overlay_python))}:${{PYTHONPATH:-}} && exec env "
            f"R1_VERIFY_REPO={shlex.quote(str(R1_FROZEN_OVERLAY))} "
            f"ROBOTWIN_ATTACH_REQUEUE_FAILED=1 R1_CONDITION={condition} "
            "LOCAL_GPU_COUNT=2 MAX_PARALLEL_SEEDS=2 "
            f"GPU_INDEX_OFFSET=0 PORT_BASE_OFFSET={port_base} bash "
            "train_scripts/kai/eval/run_pi05_r1_formal.sh"
        )
        command = (
            f"cd {shlex.quote(str(REPO))} && export CUDA_VISIBLE_DEVICES="
            f"{','.join(map(str, range(gpu_offset, gpu_offset + 4)))} && "
            f"export PYTHONPATH={shlex.quote(str(overlay_python))}:${{PYTHONPATH:-}} && "
            f"exec env R1_VERIFY_REPO={shlex.quote(str(R1_FROZEN_OVERLAY))} "
            f"ROBOTWIN_ATTACH_REQUEUE_FAILED=1 R1_CONDITION={condition} "
            f"LOCAL_GPU_COUNT=4 GPU_INDEX_OFFSET={gpu_offset} "
            f"PORT_BASE_OFFSET={port_base} bash "
            "train_scripts/kai/eval/run_pi05_r1_formal.sh"
        )
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 1,
                "description": f"Frozen 24-cell R1 {condition} seed-1000 evaluation",
                "completion_glob": str(marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(checkpoint / "params/_METADATA"),
                    str(
                        checkpoint
                        / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
                    ),
                    str(scene),
                    str(protocol),
                    str(REPO / "train_scripts/kai/eval/run_pi05_r1_formal.sh"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 2,
                        "gpu_indices": [0, 1],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO / f"logs/r1/eval_{condition}_local"
                        ),
                        "command": local_command,
                    },
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 4,
                        "gpu_indices": list(range(gpu_offset, gpu_offset + 4)),
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / f"logs/r1/eval_{condition}_gf1"),
                        "command": command,
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 60,
                        "max_failures": 6,
                        "yaml": "train_scripts/kai/volc/pi05_r1_eval_east_4h20.yaml",
                        "task_name": f"pi05-r1-{condition}-s1000-east4g",
                        "env": {
                            "R1_CONDITION": condition,
                            "PORT_BASE_OFFSET": str(port_base),
                            "R1_VERIFY_REPO": str(R1_FROZEN_OVERLAY),
                            "PYTHONPATH": str(overlay_python),
                            "ROBOTWIN_ATTACH_REQUEUE_FAILED": "1",
                            "TORCH_CUDA_ARCH_LIST": "9.0",
                            "TORCH_EXTENSIONS_DIR": (
                                "/vePFS/tim/runtime/torch_extensions/"
                                "h20_sm90_py310"
                            ),
                        },
                    },
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r1_eval_cnsh_4a100.yaml",
                        "task_name": f"pi05-r1-{condition}-s1000-cnsh4g",
                        "env": {
                            "R1_CONDITION": condition,
                            "PORT_BASE_OFFSET": str(port_base),
                            "R1_VERIFY_REPO": str(R1_FROZEN_OVERLAY),
                            "PYTHONPATH": str(overlay_python),
                            "ROBOTWIN_ATTACH_REQUEUE_FAILED": "1",
                        },
                    },
                ],
            }
        )
        existing.add(task_id)

    gate_id = "pi05_r1_seed1000_gate"
    if gate_id not in existing:
        docs = REPO / "lmvla/lmwm/docs"
        queue["tasks"].append(
            {
                "id": gate_id,
                "priority": 1,
                "description": "Paired-bootstrap R1 combined-versus-three-arm/control gate",
                "completion_glob": str(REPO / "logs/r1/seed1000/r1_gate.*"),
                "completion_min_count": 1,
                "ready_files": [
                    str(docs / "pi05_predictive_adapter_p1_seed1000_a0.json"),
                    str(docs / "pi05_predictive_adapter_p1_seed1000_normal.json"),
                    str(docs / "pi05_r1_seed1000_crave.json"),
                    str(docs / "pi05_r1_seed1000_combined.json"),
                    str(docs / "pi05_r1_seed1000_combined_zero_gate.json"),
                    str(docs / "pi05_r1_seed1000_combined_shuffled.json"),
                    str(REPO / "lmvla/lmwm/scripts/analyze_pi05_r1_seed1000.py"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/r1/seed1000/gate"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec python3 "
                            "lmvla/lmwm/scripts/analyze_pi05_r1_seed1000.py "
                            "--combined lmvla/lmwm/docs/pi05_r1_seed1000_combined.json "
                            "--control a0=lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_a0.json "
                            "--control predictive=lmvla/lmwm/docs/pi05_predictive_adapter_p1_seed1000_normal.json "
                            "--control crave=lmvla/lmwm/docs/pi05_r1_seed1000_crave.json "
                            "--control zero_route=lmvla/lmwm/docs/pi05_r1_seed1000_combined_zero_gate.json "
                            "--control shuffled_action=lmvla/lmwm/docs/pi05_r1_seed1000_combined_shuffled.json "
                            "--output logs/r1/seed1000/report.json --gate-dir logs/r1/seed1000"
                        ),
                    }
                ],
            }
        )
        existing.add(gate_id)

    replication_gate = REPO / "logs/r1/seed1000/r1_gate.accepted"
    p1_audit = (
        REPO
        / "lmvla/paper_iclr_lmvla/manifests"
        / "pi05_predictive_adapter_p1_baseline_audit.json"
    )
    replication_checkpoints = {
        "a0": lambda seed: REPO
        / "kai0/checkpoints/pi05_predictive_adapter_p1_a0_exact"
        / f"pi05_predictive_adapter_p1_a0_seed{seed}/49999",
        "predictive": lambda seed: REPO
        / "kai0/checkpoints/pi05_predictive_adapter_p1"
        / f"pi05_predictive_adapter_p1_seed{seed}/49999",
        "crave": lambda seed: REPO
        / "kai0/checkpoints/pi05_r1_crave"
        / f"pi05_r1_crave_seed{seed}/49999",
        "combined": lambda seed: REPO
        / "kai0/checkpoints/pi05_r1_combined"
        / f"pi05_r1_combined_seed{seed}/49999",
    }
    for seed in (1001, 1002):
        for arm_index, arm in enumerate(("a0", "predictive", "crave", "combined")):
            task_id = f"pi05_r1_{arm}_seed{seed}_train"
            if task_id in existing:
                continue
            checkpoint = replication_checkpoints[arm](seed)
            gpu_offset = 0 if arm_index % 2 == 0 else 4
            gpu_indices = list(range(gpu_offset, gpu_offset + 4))
            if arm in ("a0", "predictive"):
                runner = "train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh"
                command_env = f"ARM={'candidate' if arm == 'predictive' else arm}"
                east_yaml = "train_scripts/kai/volc/pi05_predictive_adapter_p1_train_east_4h20.yaml"
                task_yaml = "train_scripts/kai/volc/pi05_predictive_adapter_p1_train_cnsh_4a100.yaml"
                env = {
                    "ARM": "candidate" if arm == "predictive" else arm,
                    "SEED": str(seed),
                }
                extra_ready = [str(p1_audit)]
            else:
                runner = "train_scripts/kai/run_pi05_r1_train.sh"
                command_env = f"R1_ARM={arm} GPU_COUNT=4"
                east_yaml = "train_scripts/kai/volc/pi05_r1_train_east_4h20.yaml"
                task_yaml = "train_scripts/kai/volc/pi05_r1_train_cnsh_4a100.yaml"
                env = {"R1_ARM": arm, "SEED": str(seed)}
                extra_ready = [
                    str(labels / "READY_LABELS"),
                    str(labels / "probe_train.npz"),
                ]
            command = (
                f"cd {shlex.quote(str(REPO))} && test -s "
                f"{shlex.quote(str(replication_gate))} && export CUDA_VISIBLE_DEVICES="
                f"{','.join(map(str, gpu_indices))} && exec env {command_env} "
                f"SEED={seed} WORKERS=8 bash {runner}"
            )
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": 1,
                    "description": f"R1-accepted matched {arm} seed-{seed} training",
                    "completion_glob": str(checkpoint / "_CHECKPOINT_METADATA"),
                    "completion_min_count": 1,
                    "hold_retry_while_running": (
                        [f"pi05_predictive_adapter_p2_candidate_seed{seed}_train"]
                        if arm == "predictive"
                        else []
                    ),
                    "ready_files": [
                        str(replication_gate),
                        str(protocol),
                        str(REPO / runner),
                        *extra_ready,
                    ],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": gpu_indices,
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / f"logs/r1/train_{arm}_s{seed}_gf1"
                            ),
                            "command": command,
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": east_yaml,
                            "task_name": f"pi05-r1-{arm}-s{seed}-east4g",
                            "env": env,
                        },
                        {
                            "kind": "platform",
                            "resource": "robot-task",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": task_yaml,
                            "task_name": f"pi05-r1-{arm}-s{seed}-cnsh4g",
                            "env": env,
                        },
                    ],
                }
            )
            existing.add(task_id)

        for arm_index, arm in enumerate(("a0", "predictive", "crave", "combined")):
            task_id = f"pi05_r1_{arm}_seed{seed}_eval"
            if task_id in existing:
                continue
            checkpoint = replication_checkpoints[arm](seed)
            result_name = f"pi05_r1_seed{seed}_{arm}"
            marker = REPO / "logs/resource_markers" / f"{result_name}.ok"
            gpu_offset = 0 if arm_index % 2 == 0 else 4
            gpu_indices = list(range(gpu_offset, gpu_offset + 4))
            port_base = 23200 + gpu_offset * 20 + (seed - 1001) * 200
            local_command = (
                f"cd {shlex.quote(str(REPO))} && export PYTHONPATH="
                f"{shlex.quote(str(overlay_python))}:${{PYTHONPATH:-}} && exec env "
                f"R1_VERIFY_REPO={shlex.quote(str(R1_FROZEN_OVERLAY))} "
                f"R1_CONDITION={arm} "
                f"SEED={seed} LOCAL_GPU_COUNT=2 MAX_PARALLEL_SEEDS=2 "
                f"GPU_INDEX_OFFSET=0 PORT_BASE_OFFSET={port_base} "
                "bash train_scripts/kai/eval/run_pi05_r1_formal.sh"
            )
            command = (
                f"cd {shlex.quote(str(REPO))} && export CUDA_VISIBLE_DEVICES="
                f"{','.join(map(str, gpu_indices))} && export PYTHONPATH="
                f"{shlex.quote(str(overlay_python))}:${{PYTHONPATH:-}} && exec env "
                f"R1_VERIFY_REPO={shlex.quote(str(R1_FROZEN_OVERLAY))} "
                f"R1_CONDITION={arm} "
                f"SEED={seed} LOCAL_GPU_COUNT=4 GPU_INDEX_OFFSET={gpu_offset} "
                f"PORT_BASE_OFFSET={port_base} "
                "bash train_scripts/kai/eval/run_pi05_r1_formal.sh"
            )
            eval_env = {
                "R1_CONDITION": arm,
                "SEED": str(seed),
                "PORT_BASE_OFFSET": str(port_base),
                "R1_VERIFY_REPO": str(R1_FROZEN_OVERLAY),
                "PYTHONPATH": str(overlay_python),
            }
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": 1,
                    "description": f"Frozen R1 {arm} seed-{seed} 24-cell evaluation",
                    "completion_glob": str(marker),
                    "completion_min_count": 1,
                    "ready_files": [
                        str(replication_gate),
                        str(checkpoint / "params/_METADATA"),
                        str(
                            checkpoint
                            / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
                        ),
                        str(scene),
                        str(protocol),
                        str(REPO / "train_scripts/kai/eval/run_pi05_r1_formal.sh"),
                    ],
                    "candidates": [
                        {
                            "kind": "local",
                            "resource": "local",
                            "gpus": 2,
                            "gpu_indices": [0, 1],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / f"logs/r1/eval_{arm}_s{seed}_local"
                            ),
                            "command": local_command,
                        },
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": gpu_indices,
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(REPO / f"logs/r1/eval_{arm}_s{seed}_gf1"),
                            "command": command,
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": "train_scripts/kai/volc/pi05_r1_eval_east_4h20.yaml",
                            "task_name": f"pi05-r1-{arm}-s{seed}-eval-east4g",
                            "env": eval_env,
                        },
                        {
                            "kind": "platform",
                            "resource": "robot-task",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": "train_scripts/kai/volc/pi05_r1_eval_cnsh_4a100.yaml",
                            "task_name": f"pi05-r1-{arm}-s{seed}-eval-cnsh4g",
                            "env": eval_env,
                        },
                    ],
                }
            )
            existing.add(task_id)

    final_gate_id = "pi05_r1_three_seed_gate"
    if final_gate_id not in existing:
        docs = REPO / "lmvla/lmwm/docs"
        report_paths = {
            (1000, "a0"): docs / "pi05_predictive_adapter_p1_seed1000_a0.json",
            (1000, "predictive"): docs
            / "pi05_predictive_adapter_p1_seed1000_normal.json",
            (1000, "crave"): docs / "pi05_r1_seed1000_crave.json",
            (1000, "combined"): docs / "pi05_r1_seed1000_combined.json",
            **{
                (seed, arm): docs / f"pi05_r1_seed{seed}_{arm}.json"
                for seed in (1001, 1002)
                for arm in ("a0", "predictive", "crave", "combined")
            },
        }
        report_args = " ".join(
            f"--report {seed}:{arm}={shlex.quote(str(path))}"
            for (seed, arm), path in sorted(report_paths.items())
        )
        queue["tasks"].append(
            {
                "id": final_gate_id,
                "priority": 1,
                "description": "Hierarchical paired three-seed R1 four-arm gate",
                "completion_glob": str(
                    REPO / "logs/r1/three_seed/r1_three_seed_gate.*"
                ),
                "completion_min_count": 1,
                "ready_files": [
                    str(replication_gate),
                    *[str(path) for path in report_paths.values()],
                    str(REPO / "lmvla/lmwm/scripts/analyze_pi05_r1_three_seed.py"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/r1/three_seed/gate"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec python3 "
                            f"lmvla/lmwm/scripts/analyze_pi05_r1_three_seed.py {report_args} "
                            "--output logs/r1/three_seed/report.json "
                            "--gate-dir logs/r1/three_seed"
                        ),
                    }
                ],
            }
        )

    replication_disabled_reason = (
        "R1 seed-1000 necessary comparison rejected: combined is significantly "
        "worse than CRAVE-only"
    )
    replication_ids = {
        f"pi05_r1_{arm}_seed{seed}_{phase}"
        for seed in (1001, 1002)
        for arm in ("a0", "predictive", "crave", "combined")
        for phase in ("train", "eval")
    }
    replication_ids.add(final_gate_id)
    for task in queue["tasks"]:
        if task.get("id") in replication_ids:
            task["enabled"] = False
            task["disabled_reason"] = replication_disabled_reason


def add_pi05_r4_outcome_collection_tasks(queue: dict[str, Any]) -> None:
    """Collect action-bearing outcomes without touching frozen evaluation sources."""
    managed_ids = {
        "pi05_r4_outcome_collection_smoke",
        "pi05_r4_outcome_collection_formal",
        "pi05_r4_beat_train_support_supplement",
        "pi05_r4_balanced_train_support_supplement",
        "pi05_r4_outcome_dataset_finalize",
        "pi05_r4_query_collection_smoke",
        "pi05_r4_query_collection_smoke_v2",
        "pi05_r4_query_collection_smoke_v3",
        "pi05_r4_query_base_train_collection",
        "pi05_r4_query_beat_support_collection",
        "pi05_r4_query_balanced_support_collection",
        "pi05_r4_query_dataset_finalize",
        "pi05_r4_training_chunks_build",
        "pi05_r4_lerobot_dataset_build",
        "pi05_r4_training_runtime_verify",
        "pi05_r4_outcome_free_manifest_build",
        "pi05_r4_crave_sidecar_build",
        "pi05_r4_matched_runtime_verify",
    }
    # Queue definitions are persisted separately from task state. Rebuild this
    # hash-pinned subgraph so an authorized amendment cannot leave stale hashes
    # in the persistent queue; running/completed state remains keyed by task ID.
    queue["tasks"] = [
        task for task in queue.get("tasks", []) if task.get("id") not in managed_ids
    ]
    existing = {task.get("id") for task in queue.get("tasks", [])}
    protocol_path = (
        REPO
        / "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_r4_outcome_collection_protocol_v1.json"
    )
    protocol = json.loads(protocol_path.read_text())
    ready_hashes = [
        {"path": str(REPO / relative), "sha256": expected}
        for relative, expected in sorted(protocol["file_sha256"].items())
    ]
    common_ready = [
        str(protocol_path),
        str(REPO / "logs/frozen_source_overlays/pi05_r4_collector_v1/lawam/COLLECTOR_READY"),
        str(REPO / "train_scripts/kai/eval/run_pi05_r4_outcome_collection.sh"),
        "/vePFS/tim/hf_models/SidneyXie_pi05_robotwin/model.safetensors",
        "/vePFS/tim/hf_models/paligemma_tokenizer/tokenizer.model",
        "/vePFS/tim/workspace/lerobot-pi05-server-venv/bin/python",
    ]
    smoke_marker = REPO / "logs/resource_markers/pi05_r4_outcome_collection_smoke.ok"
    smoke_id = "pi05_r4_outcome_collection_smoke"
    if smoke_id not in existing:
        queue["tasks"].append(
            {
                "id": smoke_id,
                "priority": 2,
                "description": "One-task/two-episode isolated R4 trajectory collector smoke",
                "completion_glob": str(smoke_marker),
                "completion_min_count": 1,
                "ready_files": [
                    *common_ready,
                    str(REPO / "lmvla/lmwm/data/pi05_r4_outcome_scene_seeds_smoke_v1.json"),
                ],
                "ready_hashes": ready_hashes,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 1,
                        "gpu_indices": [0],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/smoke_local"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec env "
                            "ROBOTWIN_TASKS=beat_block_hammer SEEDS=0 "
                            "ROBOTWIN_TEST_NUM=2 LOCAL_GPU_COUNT=1 "
                            "R4_FINALIZE_DATASET=0 "
                            "RESULT_NAME=pi05_r4_outcomes_smoke_v1 "
                            "RUN_TAG_PREFIX=r4-outcomes-smoke "
                            "PORT_BASE_OFFSET=26800 "
                            f"R4_SCENE_MANIFEST={shlex.quote(str(REPO / 'lmvla/lmwm/data/pi05_r4_outcome_scene_seeds_smoke_v1.json'))} "
                            f"MARKER={shlex.quote(str(smoke_marker))} "
                            "bash train_scripts/kai/eval/run_pi05_r4_outcome_collection.sh"
                        ),
                    }
                ],
            }
        )
        existing.add(smoke_id)

    formal_id = "pi05_r4_outcome_collection_formal"
    if formal_id not in existing:
        base_dataset_manifest = (
            REPO
            / "lmvla/lawam/results/eval_runs/robotwin/pi05_r4_outcomes_public_v1/"
            "dataset_manifest.json"
        )
        queue["tasks"].append(
            {
                "id": formal_id,
                "priority": 2,
                "description": "Frozen 24-cell pi0.5 R4 action-bearing outcome collection",
                "completion_glob": str(base_dataset_manifest),
                "completion_min_count": 1,
                "ready_files": [
                    *common_ready,
                    str(smoke_marker),
                    str(REPO / "lmvla/lmwm/data/pi05_r4_outcome_scene_seeds_v1.json"),
                ],
                "ready_hashes": ready_hashes,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r4_outcome_collection_east_4h20.yaml",
                        "task_name": "pi05-r4-outcomes-public-v1-east4g",
                        "env": {
                            "RESULT_NAME": "pi05_r4_outcomes_public_v1",
                            "PORT_BASE_OFFSET": "24800",
                            "TORCH_CUDA_ARCH_LIST": "9.0",
                            "TORCH_EXTENSIONS_DIR": (
                                "/vePFS/tim/runtime/torch_extensions/"
                                "h20_sm90_py310"
                            ),
                        },
                    }
                ],
            }
        )
        existing.add(formal_id)

    support_id = "pi05_r4_beat_train_support_supplement"
    support_manifest = (
        REPO / "lmvla/lmwm/data/pi05_r4_beat_train_support_supplement_v1.json"
    )
    support_builder = (
        REPO / "lmvla/lmwm/scripts/build_pi05_r4_support_scene_manifest.py"
    )
    support_amendment = (
        REPO
        / "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_r4_outcome_support_amendment_v1.json"
    )
    support_marker = (
        REPO / "logs/resource_markers/pi05_r4_beat_train_support_supplement.ok"
    )
    if support_id not in existing:
        support_result = "pi05_r4_beat_train_support_supplement_v1"
        first_cell_summaries = [
            str(
                REPO
                / "lmvla/lawam/results/eval_runs/robotwin/pi05_r4_outcomes_public_v1"
                / f"seed{seed}/SidneyXie_pi05_robotwin__demo_clean"
                / f"r4-outcomes-public-seed{seed}/tasks/beat_block_hammer/summary.json"
            )
            for seed in range(4)
        ]
        queue["tasks"].append(
            {
                "id": support_id,
                "priority": 1,
                "description": (
                    "Predeclared all-unused-scene train support supplement for "
                    "R4 beat_block_hammer"
                ),
                "completion_glob": str(support_marker),
                "completion_min_count": 1,
                "ready_files": [
                    *common_ready,
                    str(smoke_marker),
                    str(support_manifest),
                    str(support_builder),
                    str(support_amendment),
                    *first_cell_summaries,
                ],
                "ready_hashes": [
                    *ready_hashes,
                    {"path": str(support_manifest), "sha256": sha256_file(support_manifest)},
                    {"path": str(support_builder), "sha256": sha256_file(support_builder)},
                    {
                        "path": str(support_amendment),
                        "sha256": sha256_file(support_amendment),
                    },
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 2,
                        "gpu_indices": [0, 1],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/beat_support_local"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec env "
                            "ROBOTWIN_TASKS=beat_block_hammer SEEDS='0 1' "
                            "ROBOTWIN_TEST_NUM=40 LOCAL_GPU_COUNT=2 "
                            "R4_FINALIZE_DATASET=0 "
                            f"RESULT_NAME={support_result} "
                            "RUN_TAG_PREFIX=r4-beat-train-support "
                            "PORT_BASE_OFFSET=27000 "
                            f"R4_SCENE_MANIFEST={shlex.quote(str(support_manifest))} "
                            f"MARKER={shlex.quote(str(support_marker))} "
                            "bash train_scripts/kai/eval/run_pi05_r4_outcome_collection.sh"
                        ),
                    }
                ],
            }
        )

    balanced_support_id = "pi05_r4_balanced_train_support_supplement"
    balanced_support_a = REPO / "lmvla/lmwm/data/pi05_r4_balanced_train_support_a_v1.json"
    balanced_support_b = REPO / "lmvla/lmwm/data/pi05_r4_balanced_train_support_b_v1.json"
    balanced_amendment = (
        REPO / "lmvla/paper_iclr_lmvla/manifests/pi05_r4_balanced_support_amendment_v1.json"
    )
    balanced_protocol = json.loads(balanced_amendment.read_text())
    rejected_audit = REPO / "logs/r4/outcomes/dataset_audit_combined_v1.json"
    balanced_yaml = REPO / "train_scripts/kai/volc/pi05_r4_balanced_support_east_4h20.yaml"
    balanced_markers = str(REPO / "logs/resource_markers/pi05_r4_balanced_support_*.ok")
    if balanced_support_id not in existing:
        queue["tasks"].append(
            {
                "id": balanced_support_id,
                "priority": 1,
                "description": "Balance all six R4 train tasks with every unused predeclared scene",
                "completion_glob": balanced_markers,
                "completion_min_count": 2,
                "ready_files": [
                    *common_ready,
                    str(support_marker),
                    str(rejected_audit),
                    str(balanced_support_a),
                    str(balanced_support_b),
                    str(balanced_amendment),
                    str(balanced_yaml),
                ],
                "ready_hashes": [
                    *ready_hashes,
                    {"path": str(balanced_support_a), "sha256": sha256_file(balanced_support_a)},
                    {"path": str(balanced_support_b), "sha256": sha256_file(balanced_support_b)},
                    {"path": str(balanced_amendment), "sha256": sha256_file(balanced_amendment)},
                    {"path": str(balanced_yaml), "sha256": sha256_file(balanced_yaml)},
                    {
                        "path": str(rejected_audit),
                        "sha256": balanced_protocol["trigger"]["audit_sha256"],
                    },
                ],
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r4_balanced_support_east_4h20.yaml",
                        "task_name": "pi05-r4-balanced-support-east4g",
                    }
                ],
            }
        )

    finalize_id = "pi05_r4_outcome_dataset_finalize"
    if finalize_id not in existing:
        finalizer = REPO / "train_scripts/kai/analysis/finalize_pi05_r4_outcome_dataset.sh"
        merger = REPO / "lmvla/lmwm/scripts/merge_pi05_r4_outcome_manifests.py"
        merge_amendment = (
            REPO
            / "lmvla/paper_iclr_lmvla/manifests/"
            "pi05_r4_outcome_merge_amendment_v1.json"
        )
        queue["tasks"].append(
            {
                "id": finalize_id,
                "priority": 1,
                "description": "Merge base and predeclared support outcomes and run the original R4 audit",
                "completion_glob": str(
                    REPO / "logs/resource_markers/pi05_r4_outcome_collection.ok"
                ),
                "completion_min_count": 1,
                "rearm_after_ready_file": str(
                    REPO / "logs/resource_markers/pi05_r4_balanced_support_a.ok"
                ),
                "ready_files": [
                    str(
                        REPO
                        / "lmvla/lawam/results/eval_runs/robotwin/"
                        "pi05_r4_outcomes_public_v1/dataset_manifest.json"
                    ),
                    str(support_marker),
                    str(REPO / "logs/resource_markers/pi05_r4_balanced_support_a.ok"),
                    str(REPO / "logs/resource_markers/pi05_r4_balanced_support_b.ok"),
                    str(finalizer),
                    str(merger),
                    str(merge_amendment),
                    str(REPO / "lmvla/lmwm/scripts/build_pi05_r4_outcome_manifest.py"),
                    str(REPO / "lmvla/lmwm/scripts/audit_pi05_r4_outcome_dataset.py"),
                ],
                "ready_hashes": [
                    {"path": str(finalizer), "sha256": sha256_file(finalizer)},
                    {"path": str(merger), "sha256": sha256_file(merger)},
                    {
                        "path": str(merge_amendment),
                        "sha256": sha256_file(merge_amendment),
                    },
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/finalize"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec bash "
                            "train_scripts/kai/analysis/finalize_pi05_r4_outcome_dataset.sh"
                        ),
                    }
                ],
            }
        )

    query_amendment = (
        REPO
        / "lmvla/paper_iclr_lmvla/manifests/"
        "pi05_r4_query_observation_amendment_v1.json"
    )
    query_protocol = json.loads(query_amendment.read_text())
    query_hashes = [
        {"path": str(REPO / relative), "sha256": expected}
        for relative, expected in sorted(query_protocol["file_sha256"].items())
    ]
    query_wrapper = REPO / "train_scripts/kai/eval/run_pi05_r4_query_collection.sh"
    query_smoke_marker = REPO / "logs/resource_markers/pi05_r4_query_smoke.ok"
    query_smoke_id = "pi05_r4_query_collection_smoke_v3"
    if query_smoke_id not in existing:
        queue["tasks"].append(
            {
                "id": query_smoke_id,
                "priority": 1,
                "description": "Two-episode three-camera policy-query capture smoke",
                "completion_glob": str(query_smoke_marker),
                "completion_min_count": 1,
                "ready_files": [
                    *common_ready,
                    str(smoke_marker),
                    str(query_amendment),
                    str(REPO / "lmvla/lmwm/data/pi05_r4_query_smoke_scene_seeds_v1.json"),
                ],
                "ready_hashes": query_hashes,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 1,
                        "gpu_indices": [0],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/query_smoke_local"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec env "
                            "ROBOTWIN_TASKS=beat_block_hammer SEEDS=0 "
                            "ROBOTWIN_TEST_NUM=2 LOCAL_GPU_COUNT=1 "
                            "RESULT_NAME=pi05_r4_query_smoke_v1 "
                            "RUN_TAG_PREFIX=r4-query-smoke PORT_BASE_OFFSET=27400 "
                            f"R4_SCENE_MANIFEST={shlex.quote(str(REPO / 'lmvla/lmwm/data/pi05_r4_query_smoke_scene_seeds_v1.json'))} "
                            f"MARKER={shlex.quote(str(query_smoke_marker))} "
                            "bash train_scripts/kai/eval/run_pi05_r4_query_collection.sh"
                        ),
                    }
                ],
            }
        )
        existing.add(query_smoke_id)

    outcome_marker = REPO / "logs/resource_markers/pi05_r4_outcome_collection.ok"
    base_query_id = "pi05_r4_query_base_train_collection"
    base_query_markers = str(REPO / "logs/resource_markers/pi05_r4_query_base_train_*.ok")
    query_yaml = REPO / "train_scripts/kai/volc/pi05_r4_query_base_train_east_4h20.yaml"
    if base_query_id not in existing:
        queue["tasks"].append(
            {
                "id": base_query_id,
                "priority": 1,
                "description": "All 120 base train-scene three-camera policy queries",
                "completion_glob": base_query_markers,
                "completion_min_count": 2,
                "ready_files": [
                    str(outcome_marker),
                    str(query_smoke_marker),
                    str(query_amendment),
                    str(query_wrapper),
                    str(query_yaml),
                    str(REPO / "lmvla/lmwm/data/pi05_r4_outcome_scene_seeds_v1.json"),
                ],
                "ready_hashes": query_hashes,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r4_query_base_train_east_4h20.yaml",
                        "task_name": "pi05-r4-query-base-train-east4g",
                    }
                ],
            }
        )
        existing.add(base_query_id)

    support_query_marker = REPO / "logs/resource_markers/pi05_r4_query_beat_support.ok"
    support_query_id = "pi05_r4_query_beat_support_collection"
    if support_query_id not in existing:
        queue["tasks"].append(
            {
                "id": support_query_id,
                "priority": 1,
                "description": "All 80 supplemental train-scene three-camera policy queries",
                "completion_glob": str(support_query_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(outcome_marker),
                    str(query_smoke_marker),
                    str(query_amendment),
                    str(query_wrapper),
                    str(support_manifest),
                ],
                "ready_hashes": query_hashes,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 2,
                        "gpu_indices": [0, 1],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/query_support_local"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec env "
                            "ROBOTWIN_TASKS=beat_block_hammer SEEDS='0 1' "
                            "ROBOTWIN_TEST_NUM=40 LOCAL_GPU_COUNT=2 "
                            "RESULT_NAME=pi05_r4_query_beat_support_v1 "
                            "RUN_TAG_PREFIX=r4-query-beat-support PORT_BASE_OFFSET=28000 "
                            f"R4_SCENE_MANIFEST={shlex.quote(str(support_manifest))} "
                            f"MARKER={shlex.quote(str(support_query_marker))} "
                            "bash train_scripts/kai/eval/run_pi05_r4_query_collection.sh"
                        ),
                    }
                ],
            }
        )
        existing.add(support_query_id)

    balanced_query_id = "pi05_r4_query_balanced_support_collection"
    balanced_query_yaml = (
        REPO / "train_scripts/kai/volc/pi05_r4_query_balanced_support_east_4h20.yaml"
    )
    balanced_query_markers = str(
        REPO / "logs/resource_markers/pi05_r4_query_balanced_support_*.ok"
    )
    if balanced_query_id not in existing:
        queue["tasks"].append(
            {
                "id": balanced_query_id,
                "priority": 1,
                "description": "Three-camera queries for all 400 balanced support scenes",
                "completion_glob": balanced_query_markers,
                "completion_min_count": 2,
                "ready_files": [
                    str(outcome_marker),
                    str(query_smoke_marker),
                    str(query_amendment),
                    str(query_wrapper),
                    str(balanced_support_a),
                    str(balanced_support_b),
                    str(balanced_query_yaml),
                ],
                "ready_hashes": query_hashes,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r4_query_balanced_support_east_4h20.yaml",
                        "task_name": "pi05-r4-query-balanced-support-east4g",
                    }
                ],
            }
        )
        existing.add(balanced_query_id)

    query_finalize_id = "pi05_r4_query_dataset_finalize"
    query_dataset_marker = REPO / "logs/resource_markers/pi05_r4_query_dataset.ok"
    query_finalizer = REPO / "train_scripts/kai/analysis/finalize_pi05_r4_query_dataset.sh"
    if query_finalize_id not in existing:
        queue["tasks"].append(
            {
                "id": query_finalize_id,
                "priority": 1,
                "description": "Merge all query captures and enforce the R4 trainability gate",
                "completion_glob": str(query_dataset_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(REPO / "logs/resource_markers/pi05_r4_query_base_train_a.ok"),
                    str(REPO / "logs/resource_markers/pi05_r4_query_base_train_b.ok"),
                    str(support_query_marker),
                    str(REPO / "logs/resource_markers/pi05_r4_query_balanced_support_a.ok"),
                    str(REPO / "logs/resource_markers/pi05_r4_query_balanced_support_b.ok"),
                    str(query_finalizer),
                    str(query_amendment),
                ],
                "ready_hashes": query_hashes,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/outcomes/query_finalize"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec bash "
                            "train_scripts/kai/analysis/finalize_pi05_r4_query_dataset.sh"
                        ),
                    }
                ],
            }
        )

    training_chunks_id = "pi05_r4_training_chunks_build"
    training_chunks_marker = REPO / "logs/resource_markers/pi05_r4_training_chunks.ok"
    training_chunks_builder = REPO / "lmvla/lmwm/scripts/build_pi05_r4_training_chunks.py"
    query_auditor = REPO / "lmvla/lmwm/scripts/audit_pi05_r4_query_dataset.py"
    query_manifest = (
        REPO
        / "lmvla/lawam/results/eval_runs/robotwin/pi05_r4_query_train_v1.json"
    )
    outcome_manifest = REPO / "logs/r4/outcomes/query_outcome_manifest_combined_v1.json"
    training_chunks = REPO / "lmvla/lmwm/data/pi05_r4_training_v1/query_action_chunks.npz"
    training_chunks_report = REPO / "logs/r4/training/query_action_chunks_report.json"
    if training_chunks_id not in existing:
        queue["tasks"].append(
            {
                "id": training_chunks_id,
                "priority": 1,
                "description": (
                    "Build audited query-level action chunks for the matched R4 arms; "
                    "this does not authorize policy training"
                ),
                "completion_glob": str(training_chunks_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(outcome_marker),
                    str(query_dataset_marker),
                    str(query_manifest),
                    str(outcome_manifest),
                    str(training_chunks_builder),
                    str(query_auditor),
                ],
                "ready_hashes": [
                    {
                        "path": str(training_chunks_builder),
                        "sha256": sha256_file(training_chunks_builder),
                    },
                    {"path": str(query_auditor), "sha256": sha256_file(query_auditor)},
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/chunks_build"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && rm -f "
                            f"{shlex.quote(str(training_chunks_marker))} && "
                            f"python3 {shlex.quote(str(training_chunks_builder))} "
                            f"--query-manifest {shlex.quote(str(query_manifest))} "
                            f"--outcome-manifest {shlex.quote(str(outcome_manifest))} "
                            f"--output {shlex.quote(str(training_chunks))} "
                            f"--report {shlex.quote(str(training_chunks_report))} && "
                            f"printf 'completed=%s\\nchunks=%s\\nreport=%s\\n' "
                            f"\"$(date -u +%FT%TZ)\" {shlex.quote(str(training_chunks))} "
                            f"{shlex.quote(str(training_chunks_report))} > "
                            f"{shlex.quote(str(training_chunks_marker))}"
                        ),
                    }
                ],
            }
        )

    lerobot_id = "pi05_r4_lerobot_dataset_build"
    lerobot_marker = REPO / "logs/resource_markers/pi05_r4_lerobot_dataset.ok"
    lerobot_builder = REPO / "lmvla/lmwm/scripts/build_pi05_r4_lerobot_dataset.py"
    lerobot_python = Path("/vePFS/tim/workspace/lerobot-main/.venv/bin/python")
    lerobot_root = REPO / "lmvla/lmwm/data/pi05_r4_training_v1/lerobot_query_chunks"
    lerobot_report = REPO / "logs/r4/training/lerobot_query_chunks_report.json"
    if lerobot_id not in existing:
        queue["tasks"].append(
            {
                "id": lerobot_id,
                "priority": 1,
                "description": (
                    "Materialize audited R4 direct action chunks as LeRobot data; "
                    "this does not authorize policy training"
                ),
                "completion_glob": str(lerobot_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(training_chunks_marker),
                    str(training_chunks),
                    str(training_chunks_report),
                    str(lerobot_builder),
                    str(lerobot_python),
                ],
                "ready_hashes": [
                    {"path": str(lerobot_builder), "sha256": sha256_file(lerobot_builder)},
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/lerobot_build"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && rm -f "
                            f"{shlex.quote(str(lerobot_marker))} && "
                            f"{shlex.quote(str(lerobot_python))} "
                            f"{shlex.quote(str(lerobot_builder))} "
                            f"--chunks {shlex.quote(str(training_chunks))} "
                            f"--chunks-report {shlex.quote(str(training_chunks_report))} "
                            f"--output-root {shlex.quote(str(lerobot_root))} "
                            f"--report {shlex.quote(str(lerobot_report))} && "
                            f"printf 'completed=%s\\ndataset=%s\\nreport=%s\\n' "
                            f"\"$(date -u +%FT%TZ)\" {shlex.quote(str(lerobot_root))} "
                            f"{shlex.quote(str(lerobot_report))} > "
                            f"{shlex.quote(str(lerobot_marker))}"
                        ),
                    }
                ],
            }
        )
        existing.add(lerobot_id)

    runtime_verify_id = "pi05_r4_training_runtime_verify"
    runtime_marker = REPO / "logs/resource_markers/pi05_r4_training_runtime.ok"
    runtime_dir = REPO / "lmvla/lmwm/runtime/pi05_r4_training"
    runtime_verifier = runtime_dir / "verify_runtime.py"
    runtime_report = REPO / "logs/r4/training/runtime_preflight.json"
    public_model = Path("/vePFS/tim/hf_models/SidneyXie_pi05_robotwin")
    if runtime_verify_id not in existing:
        queue["tasks"].append(
            {
                "id": runtime_verify_id,
                "priority": 1,
                "description": (
                    "Strictly load the public pi0.5 checkpoint against R4 direct chunks; "
                    "this does not authorize policy training"
                ),
                "completion_glob": str(runtime_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(lerobot_marker),
                    str(lerobot_root / "meta/info.json"),
                    str(lerobot_report),
                    str(lerobot_python),
                    str(runtime_dir / "sitecustomize.py"),
                    str(runtime_verifier),
                    str(runtime_dir / "requirements.lock"),
                    str(public_model / "config.json"),
                    str(public_model / "model.safetensors"),
                    str(public_model / "policy_preprocessor.json"),
                    str(Path("/vePFS/tim/hf_models/paligemma_tokenizer/tokenizer.model")),
                ],
                "ready_hashes": [
                    {"path": str(runtime_verifier), "sha256": sha256_file(runtime_verifier)},
                    {
                        "path": str(runtime_dir / "sitecustomize.py"),
                        "sha256": sha256_file(runtime_dir / "sitecustomize.py"),
                    },
                    {
                        "path": str(runtime_dir / "requirements.lock"),
                        "sha256": sha256_file(runtime_dir / "requirements.lock"),
                    },
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/runtime_verify"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && rm -f "
                            f"{shlex.quote(str(runtime_marker))} && exec env "
                            "PI05_R4_TRAINING_RUNTIME=1 HF_HUB_OFFLINE=1 "
                            "TRANSFORMERS_OFFLINE=1 "
                            f"PYTHONPATH={shlex.quote(str(runtime_dir))} "
                            f"{shlex.quote(str(lerobot_python))} "
                            f"{shlex.quote(str(runtime_verifier))} "
                            f"--model {shlex.quote(str(public_model))} "
                            f"--dataset-root {shlex.quote(str(lerobot_root))} "
                            "--dataset-repo-id local/pi05-r4-query-train-v1 "
                            f"--load-policy --output {shlex.quote(str(runtime_report))} && "
                            f"printf 'completed=%s\\nreport=%s\\n' "
                            f"\"$(date -u +%FT%TZ)\" {shlex.quote(str(runtime_report))} > "
                            f"{shlex.quote(str(runtime_marker))}"
                        ),
                    }
                ],
            }
        )

    outcome_free_id = "pi05_r4_outcome_free_manifest_build"
    outcome_free_marker = REPO / "logs/resource_markers/pi05_r4_outcome_free_manifest.ok"
    outcome_free_manifest = REPO / "logs/r4/training/outcome_free_query_manifest.json"
    outcome_free_builder = REPO / "lmvla/lmwm/scripts/build_pi05_r4_outcome_free_manifest.py"
    matched_protocol = (
        REPO / "lmvla/paper_iclr_lmvla/manifests/pi05_r4_matched_weighting_protocol_v1.json"
    )
    matched_spec = json.loads(matched_protocol.read_text())
    matched_ready_hashes = [
        {"path": str(REPO / relative), "sha256": expected}
        for section in ("file_sha256", "external_artifact_sha256")
        for relative, expected in sorted(matched_spec[section].items())
    ]
    if outcome_free_id not in existing:
        queue["tasks"].append(
            {
                "id": outcome_free_id,
                "priority": 1,
                "description": "Remove all outcome-bearing fields before CRAVE label generation",
                "completion_glob": str(outcome_free_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(query_dataset_marker),
                    str(query_manifest),
                    str(outcome_free_builder),
                    str(matched_protocol),
                ],
                "ready_hashes": [
                    *matched_ready_hashes,
                    {"path": str(outcome_free_builder), "sha256": sha256_file(outcome_free_builder)},
                    {"path": str(matched_protocol), "sha256": sha256_file(matched_protocol)},
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/outcome_free_manifest"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && rm -f "
                            f"{shlex.quote(str(outcome_free_marker))} && python3 "
                            f"{shlex.quote(str(outcome_free_builder))} --query-manifest "
                            f"{shlex.quote(str(query_manifest))} --output "
                            f"{shlex.quote(str(outcome_free_manifest))} && printf "
                            f"'completed=%s\\nmanifest=%s\\n' \"$(date -u +%FT%TZ)\" "
                            f"{shlex.quote(str(outcome_free_manifest))} > "
                            f"{shlex.quote(str(outcome_free_marker))}"
                        ),
                    }
                ],
            }
        )
        existing.add(outcome_free_id)

    crave_sidecar_id = "pi05_r4_crave_sidecar_build"
    crave_sidecar_marker = REPO / "logs/resource_markers/pi05_r4_crave_sidecar.ok"
    crave_sidecar = REPO / "lmvla/lmwm/data/pi05_r4_training_v1/crave_weights.npz"
    crave_report = REPO / "logs/r4/training/crave_weights_report.json"
    crave_builder = REPO / "lmvla/lmwm/scripts/build_pi05_r4_crave_weight_sidecar.py"
    crave_yaml = REPO / "train_scripts/kai/volc/pi05_r4_crave_sidecar_east_1h20.yaml"
    crave_command = (
        f"cd {shlex.quote(str(REPO))} && rm -f {shlex.quote(str(crave_sidecar_marker))} && "
        f"{shlex.quote(str(REPO / 'kai0/.venv/bin/python'))} {shlex.quote(str(crave_builder))} "
        f"--manifest {shlex.quote(str(outcome_free_manifest))} "
        f"--selection {shlex.quote(str(REPO / 'lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json'))} "
        f"--labels-manifest {shlex.quote(str(REPO / 'lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json'))} "
        f"--reference-root {shlex.quote(str(REPO / 'lmvla/lmwm/data/robotwin_dinov3base'))} "
        f"--chunks {shlex.quote(str(training_chunks))} --output {shlex.quote(str(crave_sidecar))} "
        f"--report {shlex.quote(str(crave_report))} --temperature 1.0 --device cuda --batch-size 128 && "
        f"printf 'completed=%s\\nsidecar=%s\\nreport=%s\\n' \"$(date -u +%FT%TZ)\" "
        f"{shlex.quote(str(crave_sidecar))} {shlex.quote(str(crave_report))} > "
        f"{shlex.quote(str(crave_sidecar_marker))}"
    )
    if crave_sidecar_id not in existing:
        queue["tasks"].append(
            {
                "id": crave_sidecar_id,
                "priority": 1,
                "description": "Build index-aligned outcome-free CRAVE weights for R4",
                "completion_glob": str(crave_sidecar_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(outcome_free_marker),
                    str(outcome_free_manifest),
                    str(training_chunks_marker),
                    str(training_chunks),
                    str(REPO / "logs/resource_markers/pi05_crave_r0_features.ok"),
                    str(REPO / "lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json"),
                    str(REPO / "lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json"),
                    str(crave_builder),
                    str(crave_yaml),
                    str(matched_protocol),
                ],
                "ready_dirs": [str(REPO / "lmvla/lmwm/data/robotwin_dinov3base")],
                "ready_hashes": [
                    *matched_ready_hashes,
                    {"path": str(crave_builder), "sha256": sha256_file(crave_builder)},
                    {"path": str(crave_yaml), "sha256": sha256_file(crave_yaml)},
                    {"path": str(matched_protocol), "sha256": sha256_file(matched_protocol)},
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 1,
                        "gpu_indices": [0],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/crave_sidecar_local1"),
                        "command": f"export CUDA_VISIBLE_DEVICES=0 && {crave_command}",
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 1,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r4_crave_sidecar_east_1h20.yaml",
                        "task_name": "pi05-r4-crave-sidecar-east1g",
                    },
                ],
            }
        )

    matched_runtime_id = "pi05_r4_matched_runtime_verify"
    matched_runtime_marker = REPO / "logs/resource_markers/pi05_r4_matched_runtime.ok"
    matched_runtime_report = REPO / "logs/r4/training/matched_runtime_preflight.json"
    if matched_runtime_id not in existing:
        queue["tasks"].append(
            {
                "id": matched_runtime_id,
                "priority": 1,
                "description": (
                    "Load the exact public policy and verify the index-aligned CRAVE sidecar "
                    "against the identical R4 dataset; this still does not launch training"
                ),
                "completion_glob": str(matched_runtime_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(runtime_marker),
                    str(crave_sidecar_marker),
                    str(crave_sidecar),
                    str(lerobot_marker),
                    str(lerobot_root / "meta/info.json"),
                    str(runtime_dir / "sitecustomize.py"),
                    str(runtime_verifier),
                    str(public_model / "model.safetensors"),
                    str(matched_protocol),
                ],
                "ready_hashes": [
                    *matched_ready_hashes,
                    {"path": str(runtime_verifier), "sha256": sha256_file(runtime_verifier)},
                    {"path": str(runtime_dir / "sitecustomize.py"), "sha256": sha256_file(runtime_dir / "sitecustomize.py")},
                    {"path": str(matched_protocol), "sha256": sha256_file(matched_protocol)},
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r4/training/matched_runtime_verify"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && rm -f "
                            f"{shlex.quote(str(matched_runtime_marker))} && exec env "
                            "PI05_R4_TRAINING_RUNTIME=1 HF_HUB_OFFLINE=1 "
                            "TRANSFORMERS_OFFLINE=1 "
                            f"PYTHONPATH={shlex.quote(str(runtime_dir))} "
                            f"{shlex.quote(str(lerobot_python))} {shlex.quote(str(runtime_verifier))} "
                            f"--model {shlex.quote(str(public_model))} "
                            f"--dataset-root {shlex.quote(str(lerobot_root))} "
                            "--dataset-repo-id local/pi05-r4-query-train-v1 --load-policy "
                            f"--sidecar {shlex.quote(str(crave_sidecar))} --output "
                            f"{shlex.quote(str(matched_runtime_report))} && printf "
                            f"'completed=%s\\nreport=%s\\n' \"$(date -u +%FT%TZ)\" "
                            f"{shlex.quote(str(matched_runtime_report))} > "
                            f"{shlex.quote(str(matched_runtime_marker))}"
                        ),
                    }
                ],
            }
        )


def add_pi05_r2_adaptive_execution_tasks(queue: dict[str, Any]) -> None:
    """Stage the causal-readout and same-scene frozen-policy R2 screen."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    data = REPO / "lmvla/lmwm/data/pi05_crave_r0_v1"
    readout = REPO / "lmvla/lmwm/data/pi05_r2_causal_readout_v1"
    execution_enabled = not (readout / "r2_readout.rejected").is_file()
    disabled_reason = None if execution_enabled else "R2 causal readout gate rejected"
    protocol = (
        REPO
        / "lmvla/paper_iclr_lmvla/manifests/pi05_r2_adaptive_execution_protocol_v1.json"
    )
    scene_manifest = (
        REPO / "lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json"
    )
    north_sync_marker = REPO / "logs/resource_markers/pi05_r2_north_sync.ok"

    task_id = "pi05_r2_causal_readout"
    if task_id not in existing:
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 1,
                "description": "Build and validate the six-task causal CRAVE readout for R2",
                "completion_glob": str(readout / "r2_readout.*"),
                "completion_min_count": 1,
                "ready_files": [
                    str(data / "READY_LABELS"),
                    str(data / "selection_manifest.json"),
                    str(data / "labels_manifest.json"),
                    str(data / "labels.npz"),
                    str(data / "probe_train.npz"),
                    str(data / "reference_trajectories.npz"),
                    str(REPO / "lmvla/lmwm/scripts/build_pi05_r2_causal_readout.py"),
                    str(REPO / "train_scripts/kai/run_pi05_r2_causal_readout.sh"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 1,
                        "gpu_indices": [0],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO / "logs/r2_adaptive_screen/readout_local1"
                        ),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && export CUDA_VISIBLE_DEVICES=0 "
                            "&& exec bash train_scripts/kai/run_pi05_r2_causal_readout.sh"
                        ),
                    },
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 1,
                        "gpu_indices": [0],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r2_adaptive_screen/readout_gf1"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && export CUDA_VISIBLE_DEVICES=0 "
                            "&& exec bash train_scripts/kai/run_pi05_r2_causal_readout.sh"
                        ),
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 1,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r2_readout_east_1h20.yaml",
                        "task_name": "pi05-r2-causal-readout-east1g",
                    },
                    {
                        "kind": "platform",
                        "resource": "robot-task",
                        "region": "cn-shanghai",
                        "gpus": 1,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_r2_readout_cnsh_1a100.yaml",
                        "task_name": "pi05-r2-causal-readout-cnsh1g",
                    },
                ],
            }
        )
        existing.add(task_id)

    sync_id = "pi05_r2_north_sync"
    if sync_id not in existing:
        queue["tasks"].append(
            {
                "id": sync_id,
                "enabled": execution_enabled,
                "disabled_reason": disabled_reason,
                "priority": 1,
                "description": "Atomically sync and hash-verify accepted R2 runtime on North vePFS",
                "completion_glob": str(north_sync_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(readout / "r2_readout.accepted"),
                    str(readout / "readout.npz"),
                    str(readout / "readout_manifest.json"),
                    str(protocol),
                    str(REPO / "train_scripts/kai/sync_pi05_r2_to_north_verified.sh"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/r2_adaptive_screen/north_sync"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec bash "
                            "train_scripts/kai/sync_pi05_r2_to_north_verified.sh"
                        ),
                    }
                ],
            }
        )
        existing.add(sync_id)

    for condition in ("fixed4", "adaptive"):
        task_id = f"pi05_r2_{condition}_screen"
        result_name = f"pi05_r2_{condition}_screen_v1"
        shared_marker = REPO / "logs/resource_markers" / f"{result_name}.ok"
        north_marker = Path(NORTH_REPO) / "logs/resource_markers" / f"{result_name}.ok"
        command = (
            f"cd {shlex.quote(str(REPO))} && exec env R2_CONDITION={condition} "
            "LOCAL_GPU_COUNT=4 GPU_INDEX_OFFSET={offset} bash "
            "train_scripts/kai/eval/run_pi05_r2_adaptive_screen.sh"
        )
        if task_id not in existing:
            queue["tasks"].append(
                {
                    "id": task_id,
                    "enabled": execution_enabled,
                    "disabled_reason": disabled_reason,
                    "priority": 2,
                    "description": f"Public pi0.5 R2 {condition} same-scene execution screen",
                    "completion_glob": str(shared_marker),
                    "completion_min_count": 1,
                    "completion_locations": [
                        {
                            "label": "shared",
                            "glob": str(shared_marker),
                            "remote": False,
                        },
                        {"label": "north", "glob": str(north_marker), "remote": True},
                    ],
                    "ready_files": [
                        str(readout / "r2_readout.accepted"),
                        str(north_sync_marker),
                        str(protocol),
                        str(scene_manifest),
                        str(
                            REPO
                            / "train_scripts/kai/eval/run_pi05_r2_adaptive_screen.sh"
                        ),
                    ],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": [0, 1, 2, 3],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / f"logs/r2_adaptive_screen/{condition}_gf1_0_3"
                            ),
                            "command": command.format(offset=0),
                        },
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": [4, 5, 6, 7],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / f"logs/r2_adaptive_screen/{condition}_gf1_4_7"
                            ),
                            "command": command.format(offset=4),
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": f"train_scripts/kai/volc/pi05_r2_{condition}_east_4h20.yaml",
                            "task_name": f"pi05-r2-{condition}-east4g",
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-North-H20",
                            "region": "cn-beijing",
                            "gpus": 4,
                            "queue_timeout_seconds": 300,
                            "retry_cooldown_seconds": 300,
                            "yaml": f"train_scripts/kai/volc/pi05_r2_{condition}_north_4h20.yaml",
                            "task_name": f"pi05-r2-{condition}-north4g",
                        },
                        {
                            "kind": "platform",
                            "resource": "robot-task",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": f"train_scripts/kai/volc/pi05_r2_{condition}_cnsh_4a100.yaml",
                            "task_name": f"pi05-r2-{condition}-cnsh4g",
                        },
                    ],
                }
            )
            existing.add(task_id)

        materialize_id = f"pi05_r2_{condition}_materialize_north"
        if materialize_id not in existing:
            queue["tasks"].append(
                {
                    "id": materialize_id,
                    "enabled": execution_enabled,
                    "disabled_reason": disabled_reason,
                    "priority": 2,
                    "materialize_north_result_for": task_id,
                    "description": f"Materialize North R2 {condition} report on shared vePFS",
                    "completion_glob": str(shared_marker),
                    "completion_min_count": 1,
                    "ready_files": [
                        str(
                            REPO / "train_scripts/kai/sync_pi05_r2_report_from_north.sh"
                        )
                    ],
                    "ready_files_remote": [str(north_marker)],
                    "candidates": [
                        {
                            "kind": "local",
                            "resource": "local",
                            "gpus": 0,
                            "retry_cooldown_seconds": 60,
                            "status_dir": str(
                                REPO
                                / f"logs/r2_adaptive_screen/materialize_{condition}"
                            ),
                            "command": (
                                f"cd {shlex.quote(str(REPO))} && exec env R2_CONDITION={condition} "
                                "bash train_scripts/kai/sync_pi05_r2_report_from_north.sh"
                            ),
                        }
                    ],
                }
            )
            existing.add(materialize_id)

    gate_id = "pi05_r2_adaptive_screen_gate"
    if gate_id not in existing:
        fixed_report = REPO / "lmvla/lmwm/docs/pi05_r2_fixed4_screen_v1.json"
        adaptive_report = REPO / "lmvla/lmwm/docs/pi05_r2_adaptive_screen_v1.json"
        queue["tasks"].append(
            {
                "id": gate_id,
                "enabled": execution_enabled,
                "disabled_reason": disabled_reason,
                "priority": 2,
                "description": "Paired same-scene R2 success and policy-query efficiency gate",
                "completion_glob": str(REPO / "logs/r2_adaptive_screen/r2_gate.*"),
                "completion_min_count": 1,
                "ready_files": [
                    str(fixed_report),
                    str(adaptive_report),
                    str(REPO / "lmvla/lmwm/scripts/analyze_pi05_r2_adaptive_screen.py"),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/r2_adaptive_screen/gate"),
                        "command": (
                            f"cd {shlex.quote(str(REPO))} && exec kai0/.venv/bin/python "
                            "lmvla/lmwm/scripts/analyze_pi05_r2_adaptive_screen.py "
                            "--fixed-report lmvla/lmwm/docs/pi05_r2_fixed4_screen_v1.json "
                            "--adaptive-report lmvla/lmwm/docs/pi05_r2_adaptive_screen_v1.json "
                            "--output logs/r2_adaptive_screen/report.json "
                            "--gate-dir logs/r2_adaptive_screen"
                        ),
                    }
                ],
            }
        )


def add_pi05_north_eval_attach_tasks(queue: dict[str, Any]) -> None:
    """Use North overflow capacity to attach two workers to North formal evals."""
    specs = (
        {
            "label": "a0_s1002",
            "parent": "pi05_a0_public_exact_seed1002_eval",
            "result": "pi05_rt_a0_public_exact_seed1002",
            "config": "pi05_robotwin_a0_public_exact_bj",
            "run_group": "pi05_robotwin_a0_public_exact_bj__demo_clean",
            "checkpoint": (
                f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a0_public_exact_bj/"
                "pi05_robotwin_a0_public_exact_seed1002/49999"
            ),
            "extra_env": {},
        },
        {
            "label": "a2_s1001",
            "parent": "pi05_a2_abs_confirmatory_seed1001_eval",
            "result": "pi05_rt_a2_abs_confirmatory_s1001",
            "config": "pi05_robotwin_a2_prefix_official_eval_bj",
            "run_group": "pi05_robotwin_a2_abs_confirmatory__demo_clean",
            "checkpoint": (
                f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a2_abs_confirmatory/"
                "pi05_robotwin_a2_abs_seed1001/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a2_abs_confirmatory_eval.json"
                ),
                "ROBOTWIN_HINT_ENCODER": "so400m",
                "OPENPI_SERVER_HINT_ENCODER": "so400m",
                "EVAL_HINT_RESIDUAL": "0",
            },
        },
        {
            "label": "a2_s1002",
            "parent": "pi05_a2_abs_confirmatory_seed1002_eval",
            "result": "pi05_rt_a2_abs_confirmatory_s1002",
            "config": "pi05_robotwin_a2_prefix_official_eval_bj",
            "run_group": "pi05_robotwin_a2_abs_confirmatory__demo_clean",
            "checkpoint": (
                f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a2_abs_confirmatory/"
                "pi05_robotwin_a2_abs_seed1002/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a2_abs_confirmatory_eval.json"
                ),
                "ROBOTWIN_HINT_ENCODER": "so400m",
                "OPENPI_SERVER_HINT_ENCODER": "so400m",
                "EVAL_HINT_RESIDUAL": "0",
            },
        },
        {
            "label": "a3_s1001",
            "parent": "pi05_a3_live_confirmatory_seed1001_eval",
            "result": "pi05_rt_a3_live_confirmatory_s1001",
            "config": "pi05_robotwin_a3_live_residual_prefix_official_eval",
            "run_group": "pi05_robotwin_a3_live_confirmatory__demo_clean",
            "checkpoint": (
                f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a3_live_confirmatory/"
                "pi05_robotwin_a3_live_seed1001/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json"
                ),
            },
        },
        {
            "label": "a3_s1002",
            "parent": "pi05_a3_live_confirmatory_seed1002_eval",
            "result": "pi05_rt_a3_live_confirmatory_s1002",
            "config": "pi05_robotwin_a3_live_residual_prefix_official_eval",
            "run_group": "pi05_robotwin_a3_live_confirmatory__demo_clean",
            "checkpoint": (
                f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a3_live_confirmatory/"
                "pi05_robotwin_a3_live_seed1002/49999"
            ),
            "extra_env": {
                "OPENPI_EXTRA_CONFIG": (
                    f"{NORTH_REPO}/train_scripts/kai/volc/config_overrides/"
                    "pi05_a3_live_confirmatory_eval.json"
                ),
            },
        },
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    for ordinal, spec in enumerate(specs, start=1):
        task_id = f"pi05_{spec['label']}_eval_attach_bj2g"
        if task_id in existing:
            continue
        result_root = (
            Path(NORTH_REPO) / "lmvla/lawam/results/eval_runs/robotwin" / spec["result"]
        )
        scheduler_alternatives = []
        for tag_template in (
            "unseen-seed{seed}",
            "unseen-hint-seed{seed}",
            "confirmatory-seed{seed}",
            "exact-a0-seed{seed}",
        ):
            scheduler_alternatives.append(
                {
                    "ready_files_remote": [
                        str(
                            result_root
                            / f"seed{seed}"
                            / spec["run_group"]
                            / tag_template.format(seed=seed)
                            / ".task_scheduler.json"
                        )
                        for seed in range(4)
                    ]
                }
            )
        marker_prefix = f"pi05_{spec['label']}_eval_attach"
        env = {
            "ATTACH_SEEDS": "0 1 2 3",
            "ATTACH_GPU_COUNT": "2",
            "WORKER_INDEX_BASE": str(13000 + ordinal * 1000),
            "ATTACH_GROUP_NAME": "bj2g",
            "ATTACH_MARKER_PREFIX": marker_prefix,
            "RESULT_NAME": spec["result"],
            "PI05_EVAL_CONFIG_NAME": spec["config"],
            "PI05_ASSET_ID": "robotwin2.0_absolute_meanstd",
            "CKPT": spec["checkpoint"],
            **spec["extra_env"],
        }
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 4,
                "description": (
                    f"Attach two North workers to {spec['label']} formal eval"
                ),
                "completion_glob": (
                    f"{NORTH_REPO}/logs/resource_markers/{marker_prefix}_bj2g.ok"
                ),
                "completion_remote": True,
                "completion_min_count": 1,
                "satisfied_by_task": spec["parent"],
                "ready_files_remote": [
                    f"{spec['checkpoint']}/params/_METADATA",
                    f"{spec['checkpoint']}/_CHECKPOINT_METADATA",
                ],
                "ready_any": scheduler_alternatives,
                "candidates": [
                    {
                        "kind": "platform",
                        "resource": "Robot-North-H20",
                        "region": "cn-beijing",
                        "gpus": 2,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "max_failures": 6,
                        "yaml": (
                            "train_scripts/kai/volc/"
                            "pi05_confirmatory_attach_bj_2h20.yaml"
                        ),
                        "task_name": f"pi05-{spec['label']}-attach-bj2g",
                        "env": env,
                    }
                ],
            }
        )
        existing.add(task_id)


def add_pi05_step40000_safety_probes(queue: dict[str, Any]) -> None:
    """Use otherwise-idle gf1 cards for non-gating final-training safety probes."""
    specs = (
        (
            "a2_abs",
            "pi05_a2_abs_confirmatory_seed1000_eval",
            "pi05_robotwin_a2_abs_confirmatory/pi05_robotwin_a2_abs_seed1000",
        ),
        (
            "a3_live",
            "pi05_a3_live_confirmatory_seed1000_eval",
            "pi05_robotwin_a3_live_confirmatory/pi05_robotwin_a3_live_seed1000",
        ),
    )
    existing = {task.get("id") for task in queue.get("tasks", [])}
    gpu_index = 0
    for arm, parent, checkpoint_rel in specs:
        for task_name in ("beat_block_hammer", "stack_blocks_three"):
            task_id = f"pi05_{arm}_step40000_{task_name}_safety_gf1"
            if task_id in existing:
                gpu_index += 1
                continue
            checkpoint = REPO / "kai0/checkpoints" / checkpoint_rel / "40000"
            result_name = f"pi05_{arm}_seed1000_step40000_{task_name}_probe_safety40k"
            status_dir = REPO / "logs/resource_scheduler_gf1" / task_id
            command = shlex.join(
                [
                    "env",
                    f"ARM={arm}",
                    f"GPU_INDEX={gpu_index}",
                    f"TASK_NAME={task_name}",
                    "STEP=40000",
                    "TEST_NUM=50",
                    "INTERVENTION=correct",
                    "RESULT_SUFFIX=_safety40k",
                    f"PORT_BASE_OFFSET={24400 + gpu_index * 100}",
                    "SEED_STAGGER_SECONDS=0",
                    "bash",
                    str(
                        REPO
                        / "train_scripts/kai/eval/run_pi05_confirmatory_midpoint_probe.sh"
                    ),
                ]
            )
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": 8,
                    "description": (
                        f"Non-gating step-40k {arm} {task_name} frozen-scene safety probe"
                    ),
                    "completion_glob": str(
                        REPO / "logs/resource_markers" / f"{result_name}.ok"
                    ),
                    "completion_min_count": 1,
                    "satisfied_by_task": parent,
                    "progress_logs": [
                        {
                            "label": "episodes",
                            "glob": str(
                                REPO
                                / "lmvla/lawam/results/eval_runs/robotwin"
                                / result_name
                                / "**/run.log"
                            ),
                            "regex": r"progress:.*?([0-9]+)/([0-9]+)",
                            "aggregate": True,
                            "total": 50,
                        }
                    ],
                    "ready_files": [
                        str(checkpoint / "params/_METADATA"),
                        str(checkpoint / "_CHECKPOINT_METADATA"),
                        str(
                            checkpoint
                            / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
                        ),
                    ],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 1,
                            "gpu_indices": [gpu_index],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(status_dir),
                            "command": command,
                        }
                    ],
                }
            )
            existing.add(task_id)
            gpu_index += 1


def _add_pi05_mt3_feature_tasks(queue: dict[str, Any]) -> None:
    """Stage the sharded MT3 feature materialization tasks."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    marker = REPO / "logs/resource_markers/pi05_mt3_protocol.ok"
    feature_root = REPO / "logs/mt_stage_tracker/features_raw_pi05_base"
    split = REPO / "lmvla/lmwm/data/robotwin_mt_stage_tracker_split_v1.json"
    pairs = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
    extractor = REPO / "lmvla/lmwm/scripts/extract_mt3_tracker_features.py"
    manifests = [
        feature_root / f"shard-{index:02d}-of-08/manifest.json" for index in range(8)
    ]
    for index, manifest in enumerate(manifests):
        task_id = f"pi05_mt3_feature_shard{index}_of8"
        if task_id in existing:
            continue
        command = shlex.join(
            [
                "env",
                f"CUDA_VISIBLE_DEVICES={index}",
                f"PYTHONPATH={REPO / 'kai0/src'}",
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.75",
                str(REPO / "kai0/.venv/bin/python"),
                str(extractor),
                "--repo",
                str(REPO),
                "--checkpoint",
                str(REPO / "kai0/checkpoints/pi05_base"),
                "--data-repo",
                "/vePFS/tim/workspace/VLANeXt-main/datasets/robotwin2.0_official_prompts_v21",
                "--pairs",
                str(pairs),
                "--split",
                str(split),
                "--output",
                str(feature_root),
                "--shard-index",
                str(index),
                "--num-shards",
                "8",
                "--batch-size",
                "16",
                "--rows-per-file",
                "4096",
            ]
        )
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 0,
                "description": f"Frozen raw-pi0.5 MT3 feature extraction shard {index}/8",
                "completion_glob": str(manifest),
                "completion_min_count": 1,
                "ready_files": [str(marker), str(extractor), str(pairs), str(split)],
                "candidates": [
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 1,
                        "gpu_indices": [index],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/local_train" / f"gf1_{task_id}"),
                        "command": command,
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 1,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_mt3_feature_extract_east_1h20.yaml",
                        "task_name": f"pi05-mt3-feature-s{index}-east1g",
                        "env": {"SHARD_INDEX": str(index)},
                    },
                ],
            }
        )
        existing.add(task_id)


def add_pi05_mt4_replication_tasks(queue: dict[str, Any]) -> None:
    """Stage MT3 pilot analysis and conditional MT4 replication."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    docs = REPO / "lmvla/lmwm/docs"
    eval_reports = REPO / "logs/eval_reports"
    paper = REPO / "lmvla/paper_iclr_lmvla"
    markers = REPO / "logs/resource_markers"
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    selected_marker = markers / "pi05_mt3_tracker_selected.ok"
    pilot_output = paper / "RESULTS_pi05_mt3_seed1000_controls.json"
    pilot_decision = paper / "RESULTS_pi05_mt3_seed1000_gate.json"
    pilot_marker = markers / "pi05_mt3_seed1000_replication_gate.ok"

    pilot_task = "pi05_mt3_seed1000_control_analysis"
    if pilot_task not in existing:
        predicted = docs / "pi05_mt3_learned_seed1000_predicted.json"
        null = docs / "pi05_mt3_learned_seed1000_null.json"
        within = docs / "pi05_mt3_learned_seed1000_within_task.json"
        oracle = docs / "pi05_mt3_learned_seed1000_oracle.json"
        a0 = eval_reports / "pi05_rt_a0_public_exact_seed1000.json"
        command = (
            shlex.join(
                [
                    str(REPO / "kai0/.venv/bin/python"),
                    str(REPO / "lmvla/lmwm/scripts/analyze_mt_transition_controls.py"),
                    "--correct",
                    str(predicted),
                    "--control",
                    f"a0={a0}",
                    "--control",
                    f"null={null}",
                    "--control",
                    f"within_task={within}",
                    "--output",
                    str(pilot_output),
                ]
            )
            + " && "
            + shlex.join(
                [
                    str(REPO / "kai0/.venv/bin/python"),
                    str(REPO / "lmvla/lmwm/scripts/decide_mt3_seed1000_gate.py"),
                    "--analysis",
                    str(pilot_output),
                    "--output",
                    str(pilot_decision),
                    "--accepted-marker",
                    str(pilot_marker),
                ]
            )
        )
        queue["tasks"].append(
            {
                "id": pilot_task,
                "priority": 6,
                "description": "Frozen paired MT3 seed-1000 control analysis and replication gate",
                "completion_glob": str(pilot_decision),
                "completion_min_count": 1,
                "ready_files": [
                    str(predicted),
                    str(null),
                    str(within),
                    str(oracle),
                    str(a0),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / pilot_task),
                        "command": command,
                    }
                ],
            }
        )
        existing.add(pilot_task)

    policy_config = "pi05_robotwin_mt3_learned_exact"
    tracker_root = REPO / "logs/mt_stage_tracker"
    for index, seed in enumerate((1001, 1002)):
        exp = f"pi05_robotwin_mt3_learned_seed{seed}"
        checkpoint = REPO / "kai0/checkpoints" / policy_config / exp / "49999"
        train_task = f"pi05_mt3_learned_seed{seed}_train"
        if train_task not in existing:
            gpu_start = index * 4
            command = f"""cd {shlex.quote(str(REPO))} && \
candidate=$({shlex.quote(str(REPO / "kai0/.venv/bin/python"))} -c 'import json; print(json.load(open(\"{selection}\"))[\"selected\"])') && \
tracker={shlex.quote(str(tracker_root))}/$candidate/tracker.pt && test -f \"$tracker\" && \
export CUDA_VISIBLE_DEVICES={",".join(str(gpu_start + value) for value in range(4))} \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OPENPI_DATA_HOME={shlex.quote(str(REPO / "openpi_cache"))} \
JAX_COMPILATION_CACHE_DIR={shlex.quote(str(REPO / ".cache/jax-mt-transition"))} && \
mkdir -p \"$JAX_COMPILATION_CACHE_DIR\" && \
exec env ARM=mt3_learned SEED={seed} STEPS=50000 WORKERS=8 SAVE_INTERVAL=5000 \
CONFIG={policy_config} EXP={exp} TRACKER_CANDIDATE=\"$candidate\" TRACKER_CHECKPOINT=\"$tracker\" \
bash train_scripts/kai/run_pi05_mt_transition_train.sh"""
            queue["tasks"].append(
                {
                    "id": train_task,
                    "priority": 7,
                    "description": f"Conditional MT4 selected learned policy training seed {seed}",
                    "completion_glob": str(checkpoint / "_CHECKPOINT_METADATA"),
                    "completion_min_count": 1,
                    "ready_files": [
                        str(pilot_marker),
                        str(pilot_decision),
                        str(selected_marker),
                        str(selection),
                    ],
                    "progress_logs": [
                        {
                            "label": "step",
                            "glob": str(
                                REPO
                                / "logs/local_train"
                                / f"gf1_{train_task}"
                                / "launcher.log"
                            ),
                            "regex": "Step ([0-9]+):",
                        }
                    ],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": list(range(gpu_start, gpu_start + 4)),
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / "logs/local_train" / f"gf1_{train_task}"
                            ),
                            "command": command,
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": "train_scripts/kai/volc/pi05_mt3_policy_train_east_4h20.yaml",
                            "task_name": f"pi05-mt3-learned-s{seed}-east4g",
                            "env": {
                                "SEED": str(seed),
                                "CONFIG": policy_config,
                                "EXP": exp,
                                "PILOT_GATE": str(pilot_marker),
                            },
                        },
                    ],
                }
            )
            existing.add(train_task)

        for intervention_index, intervention in enumerate(
            ("predicted", "null", "within_task")
        ):
            eval_task = f"pi05_mt3_learned_seed{seed}_{intervention}_eval"
            if eval_task in existing:
                continue
            result_name = f"pi05_mt3_learned_seed{seed}_{intervention}"
            result_marker = markers / f"{result_name}.ok"
            gpu_start = index * 4
            port_offset = 20000 + index * 800 + intervention_index * 200
            command = shlex.join(
                [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={','.join(str(gpu_start + value) for value in range(4))}",
                    f"MT3_INTERVENTION={intervention}",
                    f"CKPT={checkpoint}",
                    f"RESULT_NAME={result_name}",
                    f"MARKER={result_marker}",
                    f"PORT_BASE_OFFSET={port_offset}",
                    "bash",
                    str(REPO / "train_scripts/kai/eval/run_pi05_mt3_formal.sh"),
                ]
            )
            queue["tasks"].append(
                {
                    "id": eval_task,
                    "priority": 8,
                    "description": (
                        f"Frozen {intervention}-condition MT4 evaluation seed {seed}"
                    ),
                    "completion_glob": str(result_marker),
                    "completion_min_count": 1,
                    "produces_files": [str(docs / f"{result_name}.json")],
                    "ready_files": [
                        str(pilot_marker),
                        str(pilot_decision),
                        str(checkpoint / "params/_METADATA"),
                        str(checkpoint / "_CHECKPOINT_METADATA"),
                        str(selection),
                    ],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 4,
                            "gpu_indices": list(range(gpu_start, gpu_start + 4)),
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / "logs/local_eval" / f"gf1_{eval_task}"
                            ),
                            "command": command,
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 4,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": "train_scripts/kai/volc/pi05_mt3_policy_eval_east_4h20.yaml",
                            "task_name": f"pi05-mt3-{intervention}-s{seed}-east4g",
                            "env": {
                                "MT3_INTERVENTION": intervention,
                                "CKPT": str(checkpoint),
                                "RESULT_NAME": result_name,
                                "MARKER": str(result_marker),
                                "PORT_BASE_OFFSET": str(port_offset),
                            },
                        },
                    ],
                }
            )
            existing.add(eval_task)

    final_task = "pi05_mt3_three_seed_analysis"
    final_output = paper / "RESULTS_pi05_mt3_three_seed.json"
    if final_task not in existing:
        output = final_output
        accepted = markers / "pi05_mt3_three_seed.ok"
        manifest = (
            REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
        )
        command_parts = [
            str(REPO / "kai0/.venv/bin/python"),
            str(REPO / "lmvla/lmwm/scripts/analyze_mt1_three_seed.py"),
        ]
        for seed in (1000, 1001, 1002):
            command_parts.extend(
                [
                    "--candidate",
                    f"{seed}={docs / f'pi05_mt3_learned_seed{seed}_predicted.json'}",
                ]
            )
        for seed in (1000, 1001, 1002):
            command_parts.extend(
                [
                    "--baseline",
                    f"{seed}={eval_reports / f'pi05_rt_a0_public_exact_seed{seed}.json'}",
                ]
            )
        command_parts.extend(
            [
                "--manifest",
                str(manifest),
                "--pilot-gate",
                str(pilot_marker),
                "--output",
                str(output),
                "--accepted-marker",
                str(accepted),
            ]
        )
        ready = [str(pilot_marker), str(pilot_decision), str(manifest)]
        ready.extend(
            str(docs / f"pi05_mt3_learned_seed{seed}_predicted.json")
            for seed in (1000, 1001, 1002)
        )
        ready.extend(
            str(eval_reports / f"pi05_rt_a0_public_exact_seed{seed}.json")
            for seed in (1000, 1001, 1002)
        )
        queue["tasks"].append(
            {
                "id": final_task,
                "priority": 9,
                "description": "Frozen hierarchical three-seed MT3 versus A0 analysis",
                "completion_glob": str(output),
                "completion_min_count": 1,
                "ready_files": ready,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / final_task),
                        "command": shlex.join(command_parts),
                    }
                ],
            }
        )

    comparison_markers = [markers / "pi05_mt3_three_seed.ok"]
    comparison_results = [final_output]
    manifest = REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    for control in ("null", "within_task"):
        task_id = f"pi05_mt3_three_seed_vs_{control}_analysis"
        output = paper / f"RESULTS_pi05_mt3_three_seed_vs_{control}.json"
        accepted = markers / f"pi05_mt3_three_seed_beats_{control}.ok"
        comparison_markers.append(accepted)
        comparison_results.append(output)
        if task_id in existing:
            continue
        command_parts = [
            str(REPO / "kai0/.venv/bin/python"),
            str(REPO / "lmvla/lmwm/scripts/analyze_mt1_three_seed.py"),
        ]
        for seed in (1000, 1001, 1002):
            command_parts.extend(
                [
                    "--candidate",
                    f"{seed}={docs / f'pi05_mt3_learned_seed{seed}_predicted.json'}",
                ]
            )
        for seed in (1000, 1001, 1002):
            command_parts.extend(
                [
                    "--baseline",
                    f"{seed}={docs / f'pi05_mt3_learned_seed{seed}_{control}.json'}",
                ]
            )
        command_parts.extend(
            [
                "--manifest",
                str(manifest),
                "--pilot-gate",
                str(pilot_marker),
                "--output",
                str(output),
                "--accepted-marker",
                str(accepted),
            ]
        )
        ready = [str(pilot_marker), str(pilot_decision), str(manifest)]
        ready.extend(
            str(docs / f"pi05_mt3_learned_seed{seed}_{intervention}.json")
            for seed in (1000, 1001, 1002)
            for intervention in ("predicted", control)
        )
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 9,
                "description": f"Frozen three-seed MT3 predicted versus {control} analysis",
                "completion_glob": str(output),
                "completion_min_count": 1,
                "ready_files": ready,
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / task_id),
                        "command": shlex.join(command_parts),
                    }
                ],
            }
        )
        existing.add(task_id)

    complete_task = "pi05_mt4_three_seed_content_gate"
    if complete_task not in existing:
        complete_marker = markers / "pi05_mt4_three_seed_content.ok"
        complete_decision = paper / "RESULTS_pi05_mt4_content_gate.json"
        comparisons = ",".join(str(path) for path in comparison_markers)
        command = (
            "mkdir -p "
            + shlex.quote(str(complete_marker.parent))
            + " "
            + shlex.quote(str(complete_decision.parent))
            + ' && printf \'{\\n  "accepted": true,\\n  "comparisons": "%s"\\n}\\n\' '
            + shlex.quote(comparisons)
            + " > "
            + shlex.quote(str(complete_decision) + ".tmp")
            + " && mv "
            + shlex.quote(str(complete_decision) + ".tmp")
            + " "
            + shlex.quote(str(complete_decision))
            + " && printf 'accepted=true\\ndecision=%s\\n' "
            + shlex.quote(str(complete_decision))
            + " > "
            + shlex.quote(str(complete_marker))
        )
        queue["tasks"].append(
            {
                "id": complete_task,
                "priority": 10,
                "description": "Publish MT4 only after A0, null, and shuffled three-seed gates pass",
                "completion_glob": str(complete_decision),
                "completion_min_count": 1,
                "produces_files": [str(complete_marker)],
                "ready_files": [
                    str(path) for path in comparison_markers + comparison_results
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / complete_task),
                        "command": command,
                    }
                ],
            }
        )


def add_pi05_mt6_scope_task(queue: dict[str, Any]) -> None:
    """Stage the predeclared scope interaction behind confirmed MT4 content."""
    task_id = "pi05_mt6_scope_analysis"
    if any(task.get("id") == task_id for task in queue.get("tasks", [])):
        return
    paper = REPO / "lmvla/paper_iclr_lmvla"
    docs = REPO / "lmvla/lmwm/docs"
    reports = REPO / "logs/eval_reports"
    manifest = REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    scope = paper / "manifests/robotwin_mt6_scope_v1.json"
    output = paper / "RESULTS_pi05_mt6_scope.json"
    content_marker = REPO / "logs/resource_markers/pi05_mt4_three_seed_content.ok"
    content_decision = paper / "RESULTS_pi05_mt4_content_gate.json"
    comparison_results = [
        paper / "RESULTS_pi05_mt3_three_seed.json",
        paper / "RESULTS_pi05_mt3_three_seed_vs_null.json",
        paper / "RESULTS_pi05_mt3_three_seed_vs_within_task.json",
    ]
    command = [
        str(REPO / "kai0/.venv/bin/python"),
        str(REPO / "lmvla/lmwm/scripts/analyze_mt6_scope.py"),
    ]
    for seed in (1000, 1001, 1002):
        command.extend(
            [
                "--candidate",
                f"{seed}={docs / f'pi05_mt3_learned_seed{seed}_predicted.json'}",
            ]
        )
    for seed in (1000, 1001, 1002):
        command.extend(
            [
                "--baseline",
                f"{seed}={reports / f'pi05_rt_a0_public_exact_seed{seed}.json'}",
            ]
        )
    command.extend(
        ["--manifest", str(manifest), "--scope", str(scope), "--output", str(output)]
    )
    ready_files = [
        str(content_marker),
        str(content_decision),
        str(manifest),
        str(scope),
        *(str(path) for path in comparison_results),
    ]
    ready_files.extend(
        str(docs / f"pi05_mt3_learned_seed{seed}_predicted.json")
        for seed in (1000, 1001, 1002)
    )
    ready_files.extend(
        str(reports / f"pi05_rt_a0_public_exact_seed{seed}.json")
        for seed in (1000, 1001, 1002)
    )
    queue["tasks"].append(
        {
            "id": task_id,
            "priority": 11,
            "description": "Frozen three-seed multistage-versus-control scope interaction",
            "completion_glob": str(output),
            "completion_min_count": 1,
            "ready_files": ready_files,
            "candidates": [
                {
                    "kind": "local",
                    "resource": "local",
                    "gpus": 0,
                    "retry_cooldown_seconds": 60,
                    "status_dir": str(REPO / "logs/analysis" / task_id),
                    "command": shlex.join(command),
                }
            ],
        }
    )


def add_pi05_mt5_tasks(queue: dict[str, Any]) -> None:
    """Stage the frozen local-dynamics x milestone-transition 2x2."""
    existing = {task.get("id") for task in queue.get("tasks", [])}
    paper = REPO / "lmvla/paper_iclr_lmvla"
    docs = REPO / "lmvla/lmwm/docs"
    reports = REPO / "logs/eval_reports"
    content_marker = REPO / "logs/resource_markers/pi05_mt4_three_seed_content.ok"
    content_decision = paper / "RESULTS_pi05_mt4_content_gate.json"
    selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    tracker_root = REPO / "logs/mt_stage_tracker"
    protocol = paper / "manifests/robotwin_mt5_protocol_v1.json"
    data_audit = paper / "manifests/robotwin_mt5_fixed_horizon_data_v1.json"
    fixed_pairs = REPO / "lmvla/lmwm/data/robotwin_fixed_horizon_1s_v1/pairs.npz"
    common_ready = [
        str(content_marker),
        str(content_decision),
        str(selected_marker),
        str(selection),
        str(protocol),
        str(data_audit),
        str(fixed_pairs),
        str(REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"),
        str(REPO / "kai0/checkpoints/pi05_base/params/_METADATA"),
        str(REPO / "train_scripts/kai/run_pi05_mt5_train.sh"),
    ]
    tracker_alternatives = [
        {"ready_files": [str(tracker_root / "current_frame/tracker.pt")]},
        {"ready_files": [str(tracker_root / "history_proprio/tracker.pt")]},
    ]
    for arm in ("local", "combined"):
        config = f"pi05_robotwin_mt5_{arm}_exact"
        gpu_indices = [0, 1, 2, 3] if arm == "local" else [4, 5, 6, 7]
        for seed in (1000, 1001, 1002):
            exp = f"pi05_robotwin_mt5_{arm}_seed{seed}"
            checkpoint = REPO / "kai0/checkpoints" / config / exp / "49999"
            task_id = f"pi05_mt5_{arm}_seed{seed}_train"
            command = shlex.join(
                [
                    "env",
                    "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpu_indices)),
                    "HF_HUB_OFFLINE=1",
                    "TRANSFORMERS_OFFLINE=1",
                    "TOKENIZERS_PARALLELISM=false",
                    f"OPENPI_DATA_HOME={REPO / 'openpi_cache'}",
                    f"JAX_COMPILATION_CACHE_DIR=/tmp/jax-mt5-{arm}",
                    f"ARM=mt5_{arm}",
                    f"SEED={seed}",
                    "STEPS=50000",
                    "WORKERS=8",
                    "SAVE_INTERVAL=5000",
                    f"CONFIG={config}",
                    f"EXP={exp}",
                    "bash",
                    str(REPO / "train_scripts/kai/run_pi05_mt5_train.sh"),
                ]
            )
            if task_id not in existing:
                queue["tasks"].append(
                    {
                        "id": task_id,
                        "priority": 11,
                        "description": f"Frozen MT5 {arm} policy training seed {seed}",
                        "completion_glob": str(checkpoint / "_CHECKPOINT_METADATA"),
                        "completion_min_count": 1,
                        "ready_files": common_ready,
                        "ready_any": tracker_alternatives,
                        "progress_logs": [
                            {
                                "label": "step",
                                "glob": str(
                                    REPO
                                    / "logs/local_train"
                                    / f"gf1_{task_id}_4g/launcher.log"
                                ),
                                "regex": "Step ([0-9]+):",
                            }
                        ],
                        "candidates": [
                            {
                                "kind": "ssh",
                                "resource": "gf1",
                                "gpus": 4,
                                "gpu_indices": gpu_indices,
                                "retry_cooldown_seconds": 300,
                                "status_dir": str(
                                    REPO / "logs/local_train" / f"gf1_{task_id}_4g"
                                ),
                                "command": command,
                            },
                            {
                                "kind": "platform",
                                "resource": "Robot-East-H20",
                                "region": "cn-shanghai",
                                "gpus": 4,
                                "queue_timeout_seconds": 180,
                                "retry_cooldown_seconds": 300,
                                "yaml": "train_scripts/kai/volc/pi05_mt5_train_east_4h20.yaml",
                                "task_name": f"pi05-mt5-{arm}-s{seed}-east4g",
                                "env": {
                                    "ARM": f"mt5_{arm}",
                                    "SEED": str(seed),
                                    "CONFIG": config,
                                    "EXP": exp,
                                },
                            },
                            {
                                "kind": "platform",
                                "resource": "robot-task",
                                "region": "cn-shanghai",
                                "gpus": 4,
                                "min_dispatch_free": 4,
                                "queue_timeout_seconds": 180,
                                "retry_cooldown_seconds": 300,
                                "yaml": "train_scripts/kai/volc/pi05_mt5_train_cnsh_4a100.yaml",
                                "task_name": f"pi05-mt5-{arm}-s{seed}-cnsh4g",
                                "env": {
                                    "ARM": f"mt5_{arm}",
                                    "SEED": str(seed),
                                    "CONFIG": config,
                                    "EXP": exp,
                                },
                            },
                        ],
                    }
                )
                existing.add(task_id)

            eval_task = f"pi05_mt5_{arm}_seed{seed}_eval"
            result = docs / f"pi05_mt5_{arm}_seed{seed}.json"
            marker = REPO / "logs/resource_markers" / f"pi05_mt5_{arm}_seed{seed}.ok"
            eval_command = shlex.join(
                [
                    "env",
                    "CUDA_VISIBLE_DEVICES=" + ",".join(map(str, gpu_indices)),
                    f"MT5_ARM={arm}",
                    f"SEED={seed}",
                    "bash",
                    str(REPO / "train_scripts/kai/eval/run_pi05_mt5_formal.sh"),
                ]
            )
            if eval_task not in existing:
                queue["tasks"].append(
                    {
                        "id": eval_task,
                        "priority": 11,
                        "description": f"Frozen 24-cell MT5 {arm} evaluation seed {seed}",
                        "completion_glob": str(result),
                        "completion_min_count": 1,
                        "produces_files": [str(marker)],
                        "ready_files": [
                            str(content_marker),
                            str(content_decision),
                            str(selected_marker),
                            str(selection),
                            str(checkpoint / "params/_METADATA"),
                            str(checkpoint / "_CHECKPOINT_METADATA"),
                            str(REPO / "train_scripts/kai/eval/run_pi05_mt5_formal.sh"),
                        ],
                        "ready_any": tracker_alternatives,
                        "candidates": [
                            {
                                "kind": "ssh",
                                "resource": "gf1",
                                "gpus": 4,
                                "gpu_indices": gpu_indices,
                                "retry_cooldown_seconds": 300,
                                "status_dir": str(
                                    REPO / "logs/local_eval" / f"gf1_{eval_task}_4g"
                                ),
                                "command": eval_command,
                            },
                            {
                                "kind": "platform",
                                "resource": "Robot-East-H20",
                                "region": "cn-shanghai",
                                "gpus": 4,
                                "queue_timeout_seconds": 180,
                                "retry_cooldown_seconds": 300,
                                "yaml": "train_scripts/kai/volc/pi05_mt5_eval_east_4h20.yaml",
                                "task_name": f"pi05-mt5-{arm}-s{seed}-eval-east4g",
                                "env": {"MT5_ARM": arm, "SEED": str(seed)},
                            },
                            {
                                "kind": "platform",
                                "resource": "robot-task",
                                "region": "cn-shanghai",
                                "gpus": 4,
                                "min_dispatch_free": 4,
                                "queue_timeout_seconds": 180,
                                "retry_cooldown_seconds": 300,
                                "yaml": "train_scripts/kai/volc/pi05_mt5_eval_cnsh_4a100.yaml",
                                "task_name": f"pi05-mt5-{arm}-s{seed}-eval-cnsh4g",
                                "env": {"MT5_ARM": arm, "SEED": str(seed)},
                            },
                        ],
                    }
                )
                existing.add(eval_task)

    analysis_task = "pi05_mt5_complementarity_analysis"
    if analysis_task in existing:
        return
    output = paper / "RESULTS_pi05_mt5_three_seed.json"
    accepted_marker = REPO / "logs/resource_markers/pi05_mt5_complementarity.ok"
    command = [
        str(REPO / "kai0/.venv/bin/python"),
        str(REPO / "lmvla/lmwm/scripts/analyze_mt5_complementarity.py"),
    ]
    ready_files = [
        str(content_marker),
        str(content_decision),
        str(protocol),
        str(REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"),
    ]
    for seed in (1000, 1001, 1002):
        paths = {
            "a0": reports / f"pi05_rt_a0_public_exact_seed{seed}.json",
            "local": docs / f"pi05_mt5_local_seed{seed}.json",
            "transition": docs / f"pi05_mt3_learned_seed{seed}_predicted.json",
            "combined": docs / f"pi05_mt5_combined_seed{seed}.json",
        }
        for method, path in paths.items():
            command.extend([f"--{method}", f"{seed}={path}"])
            ready_files.append(str(path))
    command.extend(
        [
            "--manifest",
            str(
                REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
            ),
            "--protocol",
            str(protocol),
            "--output",
            str(output),
            "--accepted-marker",
            str(accepted_marker),
        ]
    )
    queue["tasks"].append(
        {
            "id": analysis_task,
            "priority": 11,
            "description": "Frozen three-seed MT5 temporal-scale complementarity analysis",
            "completion_glob": str(output),
            "completion_min_count": 1,
            "produces_files": [str(accepted_marker)],
            "ready_files": ready_files,
            "candidates": [
                {
                    "kind": "local",
                    "resource": "local",
                    "gpus": 0,
                    "retry_cooldown_seconds": 60,
                    "status_dir": str(REPO / "logs/analysis" / analysis_task),
                    "command": shlex.join(command),
                }
            ],
        }
    )


def add_pi05_mt6_efficiency_task(queue: dict[str, Any]) -> None:
    """Stage selected MT3 efficiency evidence behind confirmed content utility."""
    task_id = "pi05_mt6_selected_efficiency"
    if any(task.get("id") == task_id for task in queue.get("tasks", [])):
        return
    content_marker = REPO / "logs/resource_markers/pi05_mt4_three_seed_content.ok"
    content_decision = (
        REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt4_content_gate.json"
    )
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
    a0_checkpoint = (
        REPO
        / "kai0/checkpoints/pi05_robotwin_a0_public_exact_bj"
        / "pi05_robotwin_a0_public_exact_seed1000/49999"
    )
    mt3_checkpoint = (
        REPO
        / "kai0/checkpoints/pi05_robotwin_mt3_learned_exact"
        / "pi05_robotwin_mt3_learned_seed1000/49999"
    )
    script = REPO / "train_scripts/kai/eval/local_pi05_mt6_efficiency_1gpu.sh"
    output = REPO / "logs/efficiency/pi05_mt6_selected.json"
    command = f"CUDA_VISIBLE_DEVICES=0 bash {shlex.quote(str(script))}"
    queue["tasks"].append(
        {
            "id": task_id,
            "priority": 12,
            "description": "Matched selected-MT3 versus clean-pi0.5 efficiency benchmark",
            "completion_glob": str(output),
            "completion_min_count": 1,
            "produces_files": [
                str(REPO / "logs/efficiency/pi05_mt6_selected/a0.json"),
                str(REPO / "logs/efficiency/pi05_mt6_selected/mt3_current_frame.json"),
                str(
                    REPO / "logs/efficiency/pi05_mt6_selected/mt3_history_proprio.json"
                ),
            ],
            "ready_files": [
                str(content_marker),
                str(content_decision),
                str(selection),
                str(selected_marker),
                str(a0_checkpoint / "params/_METADATA"),
                str(a0_checkpoint / "_CHECKPOINT_METADATA"),
                str(mt3_checkpoint / "params/_METADATA"),
                str(mt3_checkpoint / "_CHECKPOINT_METADATA"),
                str(script),
                str(
                    REPO / "train_scripts/kai/analysis/benchmark_pi05_policy_latency.py"
                ),
            ],
            "candidates": [
                {
                    "kind": "ssh",
                    "resource": "gf1",
                    "gpus": 1,
                    "gpu_indices": [0],
                    "retry_cooldown_seconds": 300,
                    "status_dir": str(REPO / "logs/efficiency/pi05_mt6_selected_gf1"),
                    "command": command,
                },
                {
                    "kind": "platform",
                    "resource": "Robot-East-H20",
                    "region": "cn-shanghai",
                    "gpus": 1,
                    "queue_timeout_seconds": 180,
                    "retry_cooldown_seconds": 300,
                    "yaml": "train_scripts/kai/volc/pi05_mt6_efficiency_east_1h20.yaml",
                    "task_name": "pi05-mt6-efficiency-east1g",
                },
                {
                    "kind": "local",
                    "resource": "local",
                    "gpus": 1,
                    "gpu_indices": [0],
                    "retry_cooldown_seconds": 60,
                    "status_dir": str(REPO / "logs/efficiency/pi05_mt6_selected_local"),
                    "command": command,
                },
            ],
        }
    )


def add_pi05_mt6_train_memory_task(queue: dict[str, Any]) -> None:
    """Measure selected MT3 training memory with the matched four-A100 protocol."""
    task_id = "pi05_mt6_selected_train_memory"
    if any(task.get("id") == task_id for task in queue.get("tasks", [])):
        return
    paper = REPO / "lmvla/paper_iclr_lmvla"
    content_marker = REPO / "logs/resource_markers/pi05_mt4_three_seed_content.ok"
    content_decision = paper / "RESULTS_pi05_mt4_content_gate.json"
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
    tracker_root = REPO / "logs/mt_stage_tracker"
    script = REPO / "train_scripts/kai/eval/pi05_mt6_selected_train_memory_4a100.sh"
    output = REPO / "logs/efficiency/pi05_mt6_train_memory_selected.json"
    command = f"CUDA_VISIBLE_DEVICES=0,1,2,3 bash {shlex.quote(str(script))}"
    queue["tasks"].append(
        {
            "id": task_id,
            "priority": 12,
            "description": "Matched selected-MT3 versus clean-pi0.5 training peak memory",
            "completion_glob": str(output),
            "completion_min_count": 1,
            "produces_files": [
                str(REPO / "logs/efficiency/pi05_mt6_train_memory_selected.csv"),
                str(REPO / "logs/efficiency/pi05_mt6_train_memory_selected.log"),
            ],
            "ready_files": [
                str(content_marker),
                str(content_decision),
                str(selection),
                str(selected_marker),
                str(REPO / "logs/efficiency/pi05_train_memory_a0.json"),
                str(REPO / "kai0/checkpoints/pi05_base/params/_METADATA"),
                str(
                    REPO
                    / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
                ),
                str(script),
            ],
            "ready_any": [
                {"ready_files": [str(tracker_root / "current_frame/tracker.pt")]},
                {"ready_files": [str(tracker_root / "history_proprio/tracker.pt")]},
            ],
            "candidates": [
                {
                    "kind": "ssh",
                    "resource": "gf1",
                    "gpus": 4,
                    "gpu_indices": [0, 1, 2, 3],
                    "retry_cooldown_seconds": 300,
                    "status_dir": str(
                        REPO / "logs/efficiency/pi05_mt6_train_memory_gf1"
                    ),
                    "command": command,
                },
                {
                    "kind": "platform",
                    "resource": "robot-task",
                    "region": "cn-shanghai",
                    "gpus": 4,
                    "min_dispatch_free": 4,
                    "queue_timeout_seconds": 180,
                    "retry_cooldown_seconds": 300,
                    "yaml": "train_scripts/kai/volc/pi05_mt6_train_memory_cnsh_4a100.yaml",
                    "task_name": "pi05-mt6-train-memory-cnsh4g",
                },
            ],
        }
    )


def add_pi05_mt3_tracker_tasks(queue: dict[str, Any]) -> None:
    """Stage the frozen MT3 feature/tracker pipeline behind the three-seed gate."""
    _add_pi05_mt3_feature_tasks(queue)
    existing = {task.get("id") for task in queue.get("tasks", [])}
    protocol_task = "pi05_mt3_protocol_validate"
    if protocol_task not in existing:
        protocol = (
            REPO / "lmvla/paper_iclr_lmvla/manifests/robotwin_mt3_protocol_v1.json"
        )
        validator = REPO / "lmvla/lmwm/scripts/validate_mt3_protocol.py"
        three_seed_gate = REPO / "logs/resource_markers/pi05_mt1_three_seed_gate.ok"
        three_seed_decision = (
            REPO / "lmvla/paper_iclr_lmvla/RESULTS_pi05_mt1_three_seed.json"
        )
        protocol_marker = REPO / "logs/resource_markers/pi05_mt3_protocol.ok"
        command = (
            shlex.join(
                [
                    str(REPO / "kai0/.venv/bin/python"),
                    str(validator),
                    "--protocol",
                    str(protocol),
                    "--repo",
                    str(REPO),
                ]
            )
            + " && mkdir -p "
            + shlex.quote(str(protocol_marker.parent))
            + " && tmp="
            + shlex.quote(str(protocol_marker) + ".tmp.$$")
            + " && printf 'validated=%s\\nprotocol=%s\\n' \"$(date -u +%FT%TZ)\" "
            + shlex.quote(str(protocol))
            + ' > "$tmp" && mv "$tmp" '
            + shlex.quote(str(protocol_marker))
        )
        queue["tasks"].append(
            {
                "id": protocol_task,
                "priority": 0,
                "description": "Validate frozen MT3 protocol after the MT1 three-seed gate",
                "completion_glob": str(protocol_marker),
                "completion_min_count": 1,
                "ready_files": [
                    str(three_seed_gate),
                    str(three_seed_decision),
                    str(protocol),
                    str(validator),
                ],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / protocol_task),
                        "command": command,
                    }
                ],
            }
        )
        existing.add(protocol_task)
    feature_root = REPO / "logs/mt_stage_tracker/features_raw_pi05_base"
    split = REPO / "lmvla/lmwm/data/robotwin_mt_stage_tracker_split_v1.json"
    manifests = [
        feature_root / f"shard-{index:02d}-of-08/manifest.json" for index in range(8)
    ]

    policy_task = "pi05_mt3_learned_seed1000_train"
    if policy_task not in existing:
        policy_config = "pi05_robotwin_mt3_learned_exact"
        policy_exp = "pi05_robotwin_mt3_learned_seed1000"
        policy_checkpoint = (
            REPO
            / "kai0/checkpoints"
            / policy_config
            / policy_exp
            / "49999/_CHECKPOINT_METADATA"
        )
        selection = REPO / "logs/mt_stage_tracker/selection.json"
        selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"

        def policy_command(cuda_devices: str, workers: int) -> str:
            return f"""cd {shlex.quote(str(REPO))} && \
candidate=$({shlex.quote(str(REPO / "kai0/.venv/bin/python"))} -c 'import json; print(json.load(open(\"{selection}\"))[\"selected\"])') && \
tracker={shlex.quote(str(REPO / "logs/mt_stage_tracker"))}/$candidate/tracker.pt && \
test -f \"$tracker\" && \
export CUDA_VISIBLE_DEVICES={cuda_devices} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OPENPI_DATA_HOME={shlex.quote(str(REPO / "openpi_cache"))} \
JAX_COMPILATION_CACHE_DIR={shlex.quote(str(REPO / ".cache/jax-mt-transition"))} && \
mkdir -p \"$JAX_COMPILATION_CACHE_DIR\" && \
exec env ARM=mt3_learned SEED=1000 STEPS=50000 WORKERS={workers} SAVE_INTERVAL=5000 \
CONFIG={policy_config} EXP={policy_exp} TRACKER_CANDIDATE=\"$candidate\" \
TRACKER_CHECKPOINT=\"$tracker\" bash train_scripts/kai/run_pi05_mt_transition_train.sh"""

        queue["tasks"].append(
            {
                "id": policy_task,
                "priority": 4,
                "description": "Gate-controlled selected MT3 learned transition policy seed 1000",
                "completion_glob": str(policy_checkpoint),
                "completion_min_count": 1,
                "ready_files": [str(selected_marker), str(selection)],
                "progress_logs": [
                    {
                        "label": "step-8g",
                        "glob": str(
                            REPO
                            / "logs/local_train/gf1_pi05_mt3_learned_seed1000_8g/launcher.log"
                        ),
                        "regex": "Step ([0-9]+):",
                    },
                    {
                        "label": "step-4g",
                        "glob": str(
                            REPO
                            / "logs/local_train/gf1_pi05_mt3_learned_seed1000_4g/launcher.log"
                        ),
                        "regex": "Step ([0-9]+):",
                    },
                ],
                "candidates": [
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 8,
                        "gpu_indices": list(range(8)),
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO / "logs/local_train/gf1_pi05_mt3_learned_seed1000_8g"
                        ),
                        "command": policy_command("0,1,2,3,4,5,6,7", 16),
                    },
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 4,
                        "gpu_indices": [0, 1, 2, 3],
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(
                            REPO / "logs/local_train/gf1_pi05_mt3_learned_seed1000_4g"
                        ),
                        "command": policy_command("0,1,2,3", 8),
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_mt3_policy_train_east_4h20.yaml",
                        "task_name": "pi05-mt3-learned-s1000-east4g",
                        "env": {
                            "SEED": "1000",
                            "CONFIG": policy_config,
                            "EXP": policy_exp,
                        },
                    },
                ],
            }
        )
        existing.add(policy_task)

    tracker_script = REPO / "lmvla/lmwm/scripts/train_mt3_stage_tracker.py"
    for index, candidate_name in enumerate(("current_frame", "history_proprio")):
        output = REPO / "logs/mt_stage_tracker" / candidate_name
        task_id = f"pi05_mt3_tracker_{candidate_name}_train"
        if task_id not in existing:
            command = shlex.join(
                [
                    "env",
                    f"CUDA_VISIBLE_DEVICES={index}",
                    str(REPO / "kai0/.venv/bin/python"),
                    str(tracker_script),
                    "--candidate",
                    candidate_name,
                    "--features",
                    str(feature_root),
                    "--output",
                    str(output),
                    "--updates",
                    "10000",
                    "--batch-size",
                    "64",
                    "--learning-rate",
                    "0.0003",
                    "--weight-decay",
                    "0.01",
                    "--seed",
                    "1000",
                    "--device",
                    "cuda",
                ]
            )
            queue["tasks"].append(
                {
                    "id": task_id,
                    "priority": 1,
                    "description": f"Frozen-protocol MT3 {candidate_name} tracker training",
                    "completion_glob": str(output / "train_report.json"),
                    "completion_min_count": 1,
                    "produces_files": [
                        str(output / "tracker.pt"),
                        str(output / "validation_predictions.npz"),
                    ],
                    "ready_files": [str(path) for path in manifests],
                    "candidates": [
                        {
                            "kind": "ssh",
                            "resource": "gf1",
                            "gpus": 1,
                            "gpu_indices": [index],
                            "retry_cooldown_seconds": 300,
                            "status_dir": str(
                                REPO / "logs/local_train" / f"gf1_{task_id}"
                            ),
                            "command": command,
                        },
                        {
                            "kind": "platform",
                            "resource": "Robot-East-H20",
                            "region": "cn-shanghai",
                            "gpus": 1,
                            "queue_timeout_seconds": 180,
                            "retry_cooldown_seconds": 300,
                            "yaml": "train_scripts/kai/volc/pi05_mt3_tracker_train_east_1h20.yaml",
                            "task_name": f"pi05-mt3-{candidate_name}-east1g",
                            "env": {"CANDIDATE": candidate_name},
                        },
                    ],
                }
            )
            existing.add(task_id)

        metric_task = f"pi05_mt3_tracker_{candidate_name}_metrics"
        if metric_task not in existing:
            predictions = output / "validation_predictions.npz"
            metrics = output / "metrics.json"
            queue["tasks"].append(
                {
                    "id": metric_task,
                    "priority": 2,
                    "description": f"Frozen held-out metrics for MT3 {candidate_name}",
                    "completion_glob": str(metrics),
                    "completion_min_count": 1,
                    "ready_files": [
                        str(output / "train_report.json"),
                        str(predictions),
                    ],
                    "candidates": [
                        {
                            "kind": "local",
                            "resource": "local",
                            "gpus": 0,
                            "retry_cooldown_seconds": 60,
                            "status_dir": str(REPO / "logs/analysis" / metric_task),
                            "command": shlex.join(
                                [
                                    str(REPO / "kai0/.venv/bin/python"),
                                    str(
                                        REPO
                                        / "lmvla/lmwm/scripts/evaluate_mt_stage_tracker.py"
                                    ),
                                    "--predictions",
                                    str(predictions),
                                    "--split-manifest",
                                    str(split),
                                    "--output",
                                    str(metrics),
                                ]
                            ),
                        }
                    ],
                }
            )
            existing.add(metric_task)

    select_task = "pi05_mt3_tracker_select"
    if select_task not in existing:
        current_metrics = REPO / "logs/mt_stage_tracker/current_frame/metrics.json"
        history_metrics = REPO / "logs/mt_stage_tracker/history_proprio/metrics.json"
        selection = REPO / "logs/mt_stage_tracker/selection.json"
        selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
        command = (
            shlex.join(
                [
                    str(REPO / "kai0/.venv/bin/python"),
                    str(REPO / "lmvla/lmwm/scripts/select_mt_stage_tracker.py"),
                    "--current-frame",
                    str(current_metrics),
                    "--history-proprio",
                    str(history_metrics),
                    "--output",
                    str(selection),
                ]
            )
            + " && mkdir -p "
            + shlex.quote(str(selected_marker.parent))
            + " && printf 'validated=%s\\nselection=%s\\n' \"$(date -u +%FT%TZ)\" "
            + shlex.quote(str(selection))
            + " > "
            + shlex.quote(str(selected_marker))
        )
        queue["tasks"].append(
            {
                "id": select_task,
                "priority": 3,
                "description": "Select MT3 tracker using only frozen held-out metrics",
                "completion_glob": str(selected_marker),
                "completion_min_count": 1,
                "produces_files": [str(selection)],
                "ready_files": [str(current_metrics), str(history_metrics)],
                "candidates": [
                    {
                        "kind": "local",
                        "resource": "local",
                        "gpus": 0,
                        "retry_cooldown_seconds": 60,
                        "status_dir": str(REPO / "logs/analysis" / select_task),
                        "command": command,
                    }
                ],
            }
        )

    policy_checkpoint = (
        REPO
        / "kai0/checkpoints/pi05_robotwin_mt3_learned_exact"
        / "pi05_robotwin_mt3_learned_seed1000/49999"
    )
    selection = REPO / "logs/mt_stage_tracker/selection.json"
    selected_marker = REPO / "logs/resource_markers/pi05_mt3_tracker_selected.ok"
    interventions = ("predicted", "within_task", "null", "oracle")
    for index, intervention in enumerate(interventions):
        task_id = f"pi05_mt3_learned_seed1000_{intervention}_eval"
        if task_id in existing:
            continue
        result_name = f"pi05_mt3_learned_seed1000_{intervention}"
        marker_path = REPO / "logs/resource_markers" / f"{result_name}.ok"
        gpu_start = 0 if index % 2 == 0 else 4
        command = shlex.join(
            [
                "env",
                f"CUDA_VISIBLE_DEVICES={','.join(str(gpu_start + value) for value in range(4))}",
                f"MT3_INTERVENTION={intervention}",
                f"RESULT_NAME={result_name}",
                f"MARKER={marker_path}",
                f"PORT_BASE_OFFSET={19000 + index * 200}",
                "bash",
                str(REPO / "train_scripts/kai/eval/run_pi05_mt3_formal.sh"),
            ]
        )
        queue["tasks"].append(
            {
                "id": task_id,
                "priority": 5,
                "description": f"Frozen 24-cell MT3 {intervention} intervention evaluation",
                "completion_glob": str(marker_path),
                "completion_min_count": 1,
                "produces_files": [
                    str(REPO / "lmvla/lmwm/docs" / f"{result_name}.json")
                ],
                "ready_files": [
                    str(selected_marker),
                    str(selection),
                    str(policy_checkpoint / "params/_METADATA"),
                    str(policy_checkpoint / "_CHECKPOINT_METADATA"),
                ],
                "candidates": [
                    {
                        "kind": "ssh",
                        "resource": "gf1",
                        "gpus": 4,
                        "gpu_indices": list(range(gpu_start, gpu_start + 4)),
                        "retry_cooldown_seconds": 300,
                        "status_dir": str(REPO / "logs/local_eval" / f"gf1_{task_id}"),
                        "command": command,
                    },
                    {
                        "kind": "platform",
                        "resource": "Robot-East-H20",
                        "region": "cn-shanghai",
                        "gpus": 4,
                        "queue_timeout_seconds": 180,
                        "retry_cooldown_seconds": 300,
                        "yaml": "train_scripts/kai/volc/pi05_mt3_policy_eval_east_4h20.yaml",
                        "task_name": f"pi05-mt3-{intervention}-eval-east4g",
                        "env": {
                            "MT3_INTERVENTION": intervention,
                            "RESULT_NAME": result_name,
                            "MARKER": str(marker_path),
                            "PORT_BASE_OFFSET": str(19000 + index * 200),
                        },
                    },
                ],
            }
        )
        existing.add(task_id)


def validate_queue(queue: dict[str, Any]) -> None:
    """Reject queue edits that would silently invalidate confirmatory evidence."""
    tasks = queue.get("tasks", [])
    task_ids = [task.get("id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("resource queue contains duplicate task ids")

    tasks_by_id = {task["id"]: task for task in tasks}
    router_catalog = submission_router.load_json(submission_router.DEFAULT_CATALOG)
    router_resources = set(router_catalog.get("resources", {}))
    for task in tasks:
        if not task["id"].startswith("pi05_mt"):
            continue
        for candidate in task.get("candidates", []):
            if (
                int(candidate.get("gpus", 0)) > 0
                and candidate.get("resource") not in router_resources
            ):
                raise ValueError(
                    f"{task['id']} uses resource absent from submission router: "
                    f"{candidate.get('resource')}"
                )
            worker_base = candidate_env_value(candidate, "WORKER_INDEX_BASE")
            attach_count = candidate_env_value(candidate, "ATTACH_GPU_COUNT")
            if worker_base is None or attach_count is None:
                continue
            # attach_pi05_a0_confirmatory_platform.sh fixes PORT_BASE_OFFSET at
            # 22200 and can add seed=3 (120) plus seed_index=3 (300).
            max_requested_port = 22200 + int(worker_base) + 120 + 300
            if max_requested_port >= 65536:
                raise ValueError(
                    f"{task['id']} attach worker port exceeds TCP range: "
                    f"base={worker_base} max_port={max_requested_port}"
                )
    gate_decisions = {
        marker: str(decision_path)
        for marker, (decision_path, _keys) in GATE_DECISION_SPECS.items()
    }
    for task in tasks:
        readiness_specs = [
            task,
            *task.get("ready_any", []),
            *task.get("candidates", []),
        ]
        readiness_paths = {
            path
            for spec in readiness_specs
            for key in ("ready_files", "ready_files_remote")
            for path in spec.get(key, [])
        }
        for marker, decision in gate_decisions.items():
            if marker in readiness_paths and decision not in readiness_paths:
                raise ValueError(
                    f"{task['id']} consumes {marker} without gate decision {decision}"
                )
    for task in tasks:
        gate_outputs = []
        is_gate_producer = False
        for candidate in task.get("candidates", []):
            command = candidate.get("command")
            if not command:
                continue
            tokens = shlex.split(command)
            if "--accepted-marker" not in tokens:
                continue
            is_gate_producer = True
            gate_outputs.extend(
                tokens[index + 1]
                for index, token in enumerate(tokens[:-1])
                if token == "--output"
            )
        if is_gate_producer:
            if not gate_outputs:
                raise ValueError(f"{task['id']} gate producer has no decision output")
            if task.get("completion_glob") != gate_outputs[-1]:
                raise ValueError(
                    f"{task['id']} gate producer must complete on final decision output "
                    f"{gate_outputs[-1]}"
                )
    # Orbax writes params metadata before atomically committing a saved training
    # step. Legacy parameter-only initialization bundles such as ``pi05_base``
    # intentionally have no root checkpoint sentinel, so only apply this gate
    # to numeric step directories.
    for task in tasks:
        mt12_eval = re.fullmatch(
            r"pi05_(?:mt1_oracle_seed100[012]_(?:correct|null|within_task|cross_task)|mt2_null_seed1000)_eval",
            task["id"],
        )
        readiness_specs = [
            task,
            *task.get("ready_any", []),
            *task.get("candidates", []),
        ]
        for readiness in readiness_specs:
            for key in ("ready_files", "ready_files_remote"):
                paths = readiness.get(key, [])
                checkpoint_roots = [
                    path[: -len("/params/_METADATA")]
                    for path in paths
                    if path.endswith("/params/_METADATA")
                ]
                sentinels = [
                    f"{root}/_CHECKPOINT_METADATA"
                    for root in checkpoint_roots
                    if Path(root).name.isdigit()
                ]
                if mt12_eval:
                    sentinels.extend(
                        sentinel
                        for root in checkpoint_roots
                        for sentinel in (
                            *(
                                (f"{root}/train_state/_METADATA",)
                                if key == "ready_files"
                                else ()
                            ),
                            f"{root}/assets/robotwin2.0_absolute_meanstd/norm_stats.json",
                        )
                    )
                for sentinel in sentinels:
                    if sentinel not in paths:
                        paths.append(sentinel)
    mt1_control = tasks_by_id.get("pi05_mt1_seed1000_control_analysis")
    if mt1_control is not None:
        authoritative_a0 = str(
            REPO / "logs/eval_reports/pi05_rt_a0_public_exact_seed1000.json"
        )
        if authoritative_a0 not in mt1_control.get("ready_files", []):
            raise ValueError(
                "MT1 control analysis must use the authoritative A0 report"
            )
        command = mt1_control.get("candidates", [{}])[0].get("command", "")
        if f"--control a0={authoritative_a0}" not in command:
            raise ValueError("MT1 control command must use the authoritative A0 report")
        if "lmvla/lmwm/docs/pi05_rt_a0_public_exact" in command:
            raise ValueError("MT1 control command may not use an A0 report copy")
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
        if spec.get("override_use_delta_joint_actions") is not False:
            raise ValueError(f"{filename} must preserve absolute-action inference")
        if spec.get("override_use_quantile_norm") is not False:
            raise ValueError(f"{filename} must preserve mean/std inference")
    for task in tasks:
        if not PI05_CONFIRMATORY_EVAL_RE.fullmatch(task["id"]):
            continue
        alternatives = task.get("ready_any", [])
        for candidate in task.get("candidates", []):
            key = (
                "ready_files_remote"
                if candidate.get("resource") == "Robot-North-H20"
                else "ready_files"
            )
            matching = [spec for spec in alternatives if spec.get(key)]
            task_paths = task.get(key, [])
            if alternatives and not matching and not task_paths:
                raise ValueError(
                    f"{task['id']} {candidate.get('resource')} has no matching "
                    "checkpoint readiness alternative"
                )
            candidate_paths = candidate.setdefault(key, [])
            for paths in (task_paths, *(spec[key] for spec in matching)):
                for path in paths:
                    if path not in candidate_paths:
                        candidate_paths.append(path)
        progress_locations = task.get("completion_locations") or [
            {
                "glob": task["completion_glob"],
                "remote": bool(task.get("completion_remote")),
            }
        ]
        progress_location = next(
            (location for location in progress_locations if not location.get("remote")),
            progress_locations[0],
        )
        progress_glob = re.sub(
            r"/summary\.json$", "/run.log", progress_location["glob"]
        )
        if not any(
            item.get("label") == "episodes" for item in task.get("progress_logs", [])
        ):
            task.setdefault("progress_logs", []).append(
                {
                    "label": "episodes",
                    "glob": progress_glob,
                    "regex": r"progress:.*?([0-9]+)/([0-9]+)",
                    "aggregate": True,
                    "total": 1200,
                    "remote": bool(progress_location.get("remote")),
                }
            )
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
                    f"{key}={shlex.quote(value)}"
                    for key, value in fixed_seed_env.items()
                )
                if "ROBOTWIN_EPISODE_SEED_MANIFEST=" not in candidate["command"]:
                    candidate["command"] = f"env {prefix} {candidate['command']}"
            expected = {
                "ROBOTWIN_EPISODE_SEED_MANIFEST": manifest,
                "ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS": "500",
            }
            if any(
                candidate["env"].get(key) != value for key, value in expected.items()
            ):
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
                    raise ValueError(
                        f"{task_id} {candidate['resource']} A3 sidecar mismatch"
                    )
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


def apply_permanent_resource_policy(queue: dict[str, Any]) -> None:
    """Remove retired resources without rewriting historical queue provenance."""
    for task in queue.get("tasks", []):
        candidates = task.get("candidates", [])
        retired = [
            candidate
            for candidate in candidates
            if candidate.get("resource") in PERMANENTLY_DISABLED_RESOURCES
        ]
        if not retired:
            continue
        task["candidates"] = [
            candidate
            for candidate in candidates
            if candidate.get("resource") not in PERMANENTLY_DISABLED_RESOURCES
        ]
        task["retired_resource_candidates"] = sorted(
            {candidate["resource"] for candidate in retired}
        )
        if task["candidates"] or not task.get("enabled", True):
            continue
        task["enabled"] = False
        task["disabled_by_retired_resource"] = True
        task["disabled_reason"] = (
            "all execution candidates retired: "
            + "; ".join(
                PERMANENTLY_DISABLED_RESOURCES[resource]
                for resource in task["retired_resource_candidates"]
            )
        )


def _nested_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"gate decision is missing {'.'.join(keys)}")
        value = value[key]
    return value


def gate_rejection_closure(queue: dict[str, Any]) -> dict[str, str]:
    """Return tasks made impossible by an observed negative scientific gate."""
    rejected_markers: dict[str, str] = {}
    for marker, (decision_path, keys) in GATE_DECISION_SPECS.items():
        if not decision_path.is_file():
            continue
        payload = json.loads(decision_path.read_text())
        accepted = _nested_value(payload, keys)
        if not isinstance(accepted, bool):
            raise ValueError(f"gate decision must be boolean: {decision_path}")
        if not accepted:
            rejected_markers[marker] = str(decision_path)

    closed: dict[str, str] = {}
    blocked_paths = dict(rejected_markers)
    changed = True
    while changed:
        changed = False
        for task in queue.get("tasks", []):
            task_id = task["id"]
            if task_id in closed or not task.get("enabled", True):
                continue
            required_block = next(
                (
                    blocked_paths[path]
                    for path in task.get("ready_files", [])
                    if path in blocked_paths
                ),
                None,
            )
            alternatives = task.get("ready_any", [])
            alternative_blocks = [
                next(
                    (
                        blocked_paths[path]
                        for path in alternative.get("ready_files", [])
                        if path in blocked_paths
                    ),
                    None,
                )
                for alternative in alternatives
            ]
            all_alternatives_blocked = bool(alternatives) and all(alternative_blocks)
            reason = required_block or (
                alternative_blocks[0] if all_alternatives_blocked else None
            )
            if reason is None:
                continue
            closed[task_id] = reason
            completion = task.get("completion_glob", "")
            if completion and not glob.has_magic(completion):
                blocked_paths[completion] = reason
            for output in task.get("produces_files", []):
                if not glob.has_magic(output):
                    blocked_paths[output] = reason
            changed = True
    return closed


def load_state(queue: dict[str, Any]) -> dict[str, Any]:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
    else:
        state = {"tasks": {}, "created_at": utc_now()}
    queue_task_ids = {task["id"] for task in queue["tasks"]}
    for retired_task_id in set(state["tasks"]) - queue_task_ids:
        state["tasks"].pop(retired_task_id)
    gate_closed = gate_rejection_closure(queue)
    for task in queue["tasks"]:
        task_state = state["tasks"].setdefault(
            task["id"], {"status": "pending", "attempts": []}
        )
        if (
            task["id"] == "pi05_p1_north_failover_pair"
            and task_state.get("status") == "completed"
            and not completion_evidence(task)[0]
        ):
            task_state["status"] = "pending"
            task_state["artifacts_complete"] = False
            task_state.pop("completed_at", None)
            task_state["waiting_reason"] = (
                "prior platform failure was misclassified; remote 49999 outputs missing"
            )
            if task_state.get("attempts"):
                task_state["attempts"][-1]["completion_misclassification_repaired"] = (
                    utc_now()
                )
        if (
            task["id"] == "pi05_p1_north_failover_materialize"
            and task_state.get("status") == "completed"
            and not completion_evidence(task)[0]
        ):
            task_state["status"] = "pending"
            task_state["artifacts_complete"] = False
            task_state.pop("completed_at", None)
            task_state.pop("satisfied_by_task", None)
            task_state["waiting_reason"] = (
                "legacy report-only completion repaired; checkpoint marker missing"
            )
            if task_state.get("attempts"):
                task_state["attempts"][-1]["completion_misclassification_repaired"] = (
                    utc_now()
                )
        if not task.get("enabled", True):
            has_completion_spec = bool(
                task.get("completion_glob") or task.get("completion_locations")
            )
            complete = (
                bool(task.get("disabled_by_retired_resource"))
                and has_completion_spec
                and completion_evidence(task)[0]
            )
            if task_state.get("status") == "completed" or complete:
                task_state["status"] = "completed"
                task_state.pop("disabled_reason", None)
            else:
                task_state["status"] = "disabled"
                task_state["disabled_reason"] = task.get(
                    "disabled_reason", "disabled by queue configuration"
                )
        elif task["id"] in gate_closed and task_state.get("status") != "completed":
            task_state["status"] = "disabled"
            task_state["disabled_reason"] = (
                "scientific gate rejected: " + gate_closed[task["id"]]
            )
        elif task_state.get("status") == "disabled":
            task_state["status"] = "pending"
            task_state.pop("disabled_reason", None)
        if task_state.get("status") != "running" or not task_state.get("attempts"):
            continue
        attempt = task_state["attempts"][-1]
        resource = attempt.get("resource")
        if resource not in PERMANENTLY_DISABLED_RESOURCES:
            continue
        attempt["finished_at"] = utc_now()
        attempt["failure"] = PERMANENTLY_DISABLED_RESOURCES[resource]
        attempt["terminal_reason"] = "resource permanently retired by operator"
        attempt.pop("monitor_error", None)
        if task.get("enabled", True):
            task_state["status"] = "pending"
            task_state["waiting_reason"] = (
                f"{resource} retired; waiting for an eligible replacement resource"
            )
        else:
            task_state["status"] = "disabled"
            task_state["disabled_reason"] = task.get(
                "disabled_reason", PERMANENTLY_DISABLED_RESOURCES[resource]
            )
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
            "ListJobs": ApiInfo(
                "POST", "/", {"Action": "ListJobs", "Version": "2024-07-01"}, {}, {}
            ),
            "GetJob": ApiInfo(
                "POST", "/", {"Action": "GetJob", "Version": "2024-07-01"}, {}, {}
            ),
            "StopJob": ApiInfo(
                "POST", "/", {"Action": "StopJob", "Version": "2024-07-01"}, {}, {}
            ),
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


def list_jobs(
    region: str, queue_id: str, profile: str = "primary"
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for state in SUBMITTED_JOB_STATES:
        body = {"ResourceQueueId": queue_id, "PageSize": 100, "State": state}
        raw = service(region, profile).json("ListJobs", {}, json.dumps(body).encode())
        result = json.loads(raw).get("Result", {})
        for job in result.get("Items", result.get("List", [])):
            job = dict(job)
            job["_state"] = state
            job["_gpus"] = job_gpus(job)
            key = str(job.get("Id") or f"{state}:{len(rows)}")
            rows[key] = job
    return list(rows.values())


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
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
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
    output = (
        run(query, timeout=20)
        if command is None
        else ssh(command, " ".join(query), timeout=20)
    )
    rows = []
    for line in output.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        rows.append(
            {
                "index": values[0],
                "memory_used_mib": values[1],
                "memory_total_mib": values[2],
                "utilization": values[3],
            }
        )
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
            health_hits = TRAIN_HEALTH_PATTERN.findall(log_tail)
            health = "ALERT" if health_hits else "OK"
            log_tail = re.sub(
                r"([0-9.]+)([kM])it",
                lambda match: str(
                    int(float(match[1]) * {"k": 1_000, "M": 1_000_000}[match[2]])
                ),
                log_tail,
            )
            matches = list(progress_pattern.finditer(log_tail))
            if not matches:
                statuses[label] = {
                    "status": remote_status,
                    "progress": "INITIALIZING",
                    "health": health,
                    "health_hits": len(health_hits),
                }
                continue
            match = matches[-1]
            step = int(match.group(1))
            expected = int(match.group(2))
            recent_seconds_per_step = [
                (
                    1.0 / float(recent.group(3))
                    if recent.group(4) == "it/s"
                    else float(recent.group(3))
                )
                for recent in matches[-20:]
            ]
            seconds_per_step = statistics.median(recent_seconds_per_step)
            rate_samples = len(recent_seconds_per_step)
            statuses[label] = {
                "status": remote_status,
                "step": step,
                "expected_steps": expected,
                "seconds_per_step": seconds_per_step,
                "rate_samples": rate_samples,
                "eta_hours": round(
                    max(0, expected - step) * seconds_per_step / 3600, 2
                ),
                "health": health,
                "health_hits": len(health_hits),
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
        matches = list(
            re.finditer(r"step (\d+)/(\d+).*?rate=([0-9.]+)it/s.*?eta=([^\s]+)", text)
        )
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
            "log_mtime": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return statuses


def staged_failover_statuses() -> dict[str, Any]:
    if not P1_NORTH_FAILOVER_PROGRESS_PATH.is_file():
        return {}
    try:
        status = json.loads(P1_NORTH_FAILOVER_PROGRESS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "pi05_p1_north": {
                "status": "MONITOR_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
        }
    if P1_NORTH_FAILOVER_AUTH_AUDIT_PATH.is_file():
        try:
            authorization = json.loads(
                P1_NORTH_FAILOVER_AUTH_AUDIT_PATH.read_text()
            )
            status["launch_authorized"] = bool(
                authorization.get("launch_authorized")
            )
            status["authorization_audit"] = str(
                P1_NORTH_FAILOVER_AUTH_AUDIT_PATH
            )
        except (OSError, json.JSONDecodeError):
            pass
    return {"pi05_p1_north": status}


def platform_training_statuses(
    watch_tasks: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    progress_pattern = re.compile(r"([0-9.]+)(?:it)?/([0-9.]+).*?([0-9.]+)(it/s|s/it)")
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
            text_tail = (
                stream.read().decode("utf-8", errors="replace").replace("\r", "\n")
            )
        health_hits = TRAIN_HEALTH_PATTERN.findall(text_tail)
        health = "ALERT" if health_hits else "OK"
        text_tail = re.sub(
            r"([0-9.]+)([kM])it",
            lambda match: str(
                int(float(match[1]) * {"k": 1_000, "M": 1_000_000}[match[2]])
            ),
            text_tail,
        )
        matches = list(progress_pattern.finditer(text_tail))
        if not matches:
            statuses[label] = {
                "status": "INITIALIZING",
                "log_path": str(path),
                "health": health,
                "health_hits": len(health_hits),
            }
            continue
        match = matches[-1]
        step = int(float(match.group(1)))
        expected = int(float(match.group(2)))
        rate_value = float(match.group(3))
        seconds_per_step = 1.0 / rate_value if match.group(4) == "it/s" else rate_value
        age_seconds = max(
            0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        )
        statuses[label] = {
            "status": "RUNNING" if age_seconds < 300 else "STALE_LOG",
            "step": step,
            "expected_steps": expected,
            "seconds_per_step": seconds_per_step,
            "eta_hours": round(max(0, expected - step) * seconds_per_step / 3600, 2),
            "log_mtime": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "health": health,
            "health_hits": len(health_hits),
        }
    return statuses


def north_training_statuses() -> dict[str, Any]:
    program = f"""
import calendar, glob, json, os, re, time
tasks = {NORTH_TRAIN_WATCH_TASKS!r}
pattern = re.compile(
    r'([0-9.]+)(?:it)?/([0-9.]+).*?(?:rate:)?([0-9.]+)(it/s|s/it)'
)
health_pattern = re.compile({TRAIN_HEALTH_PATTERN.pattern!r}, re.IGNORECASE)
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
    health_hits = health_pattern.findall(tail)
    health = 'ALERT' if health_hits else 'OK'
    tail = re.sub(
        r'([0-9.]+)([kM])it',
        lambda match: str(int(float(match[1]) * {{'k': 1_000, 'M': 1_000_000}}[match[2]])),
        tail,
    )
    matches = list(pattern.finditer(tail))
    if not matches:
        simple_matches = list(simple_pattern.finditer(tail))
        if not simple_matches:
            result[label] = {{
                'status': 'INITIALIZING',
                'log_path': path,
                'health': health,
                'health_hits': len(health_hits),
            }}
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
        'health': health,
        'health_hits': len(health_hits),
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
        for path_text in glob.glob(
            str(root / "**/.task_scheduler.json"), recursive=True
        ):
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
    if any(not Path(path).is_dir() for path in spec.get("ready_dirs", [])):
        return False
    if any(not glob.glob(pattern) for pattern in spec.get("ready_globs", [])):
        return False
    if readiness_hash_failures(spec):
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


def readiness_hash_failures(spec: dict[str, Any]) -> list[str]:
    failures = []
    for item in spec.get("ready_hashes", []):
        path = Path(item["path"])
        expected = item["sha256"]
        if not path.is_file():
            failures.append(f"{path}:missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"{path}:{actual[:12]}!={expected[:12]}")
    return failures


def ready(task: dict[str, Any]) -> bool:
    if not readiness_spec_satisfied(task):
        return False
    alternatives = task.get("ready_any", [])
    return not alternatives or any(
        readiness_spec_satisfied(spec) for spec in alternatives
    )


def apply_frozen_source_readiness(queue: dict[str, Any]) -> None:
    """Gate frozen P1/P2/R1 jobs before resource recommendation and launch."""
    manifests = REPO / "lmvla/paper_iclr_lmvla/manifests"
    p1_audit = json.loads(
        (manifests / "pi05_predictive_adapter_p1_baseline_audit.json").read_text()
    )
    p1_paths = {
        "config.py": "kai0/src/openpi/training/config.py",
        "pi0.py": "kai0/src/openpi/models/pi0.py",
        "weight_loaders.py": "kai0/src/openpi/training/weight_loaders.py",
        "train_pi05_robotwin_confirmatory.py": (
            "kai0/scripts/train_pi05_robotwin_confirmatory.py"
        ),
    }
    p1_hashes = [
        {
            "path": str(REPO / relative),
            "sha256": p1_audit["source_identity"]["current"][name],
        }
        for name, relative in p1_paths.items()
    ]
    p1_amendment = json.loads(
        (manifests / "pi05_p1_frozen_overlay_amendment_v1.json").read_text()
    )
    p1_eval_authorized = set(p1_amendment["authorization_scope"])
    p1_overlay_hashes = [
        {
            "path": str(R1_FROZEN_OVERLAY / relative),
            "sha256": p1_audit["source_identity"]["current"][name],
        }
        for name, relative in p1_paths.items()
    ]
    for key in (
        "overlay_preflight",
        "preflight_script",
        "source_verifier",
        "canonical_eval_launcher",
        "runtime_wrapper",
        "east_eval_yaml",
        "cnsh_eval_yaml",
    ):
        p1_overlay_hashes.append(
            {
                "path": str(REPO / p1_amendment[key]),
                "sha256": p1_amendment[f"{key}_sha256"],
            }
        )

    r1_protocol = json.loads((manifests / "pi05_r1_protocol_v1.json").read_text())
    r1_amendment = json.loads(
        (manifests / "pi05_r1_frozen_overlay_amendment_v1.json").read_text()
    )
    r1_hashes = [
        {"path": str(REPO / relative), "sha256": expected}
        for relative, expected in r1_protocol["source_sha256"].items()
    ]
    r1_overlay_hashes = [
        {"path": str(R1_FROZEN_OVERLAY / relative), "sha256": expected}
        for relative, expected in r1_protocol["source_sha256"].items()
    ]
    r1_overlay_hashes.extend(
        (
            {
                "path": str(REPO / "train_scripts/kai/eval/run_pi05_r1_formal.sh"),
                "sha256": r1_amendment["runtime_eval_launcher_sha256"],
            },
            {
                "path": str(REPO / r1_amendment["cpu_preflight"]),
                "sha256": r1_amendment["cpu_preflight_sha256"],
            },
        )
    )
    r1_eval_authorized = set(r1_amendment["authorization_scope"])

    replication_amendment_path = (
        manifests / "pi05_replication_frozen_training_amendment_v1.json"
    )
    replication_amendment = json.loads(replication_amendment_path.read_text())
    p2_protocol = json.loads(
        (manifests / "pi05_predictive_adapter_p2_protocol.json").read_text()
    )
    replication_sources = dict(r1_protocol["source_sha256"])
    replication_sources.update(p2_protocol["file_sha256"])
    replication_overlay_hashes = [
        {
            "path": str(REPLICATION_FROZEN_OVERLAY / relative),
            "sha256": expected,
        }
        for relative, expected in sorted(replication_sources.items())
    ]
    replication_overlay_hashes.extend(
        {
            "path": str(REPO / relative),
            "sha256": expected,
        }
        for relative, expected in sorted(
            replication_amendment["runtime_launchers"].items()
        )
    )
    replication_overlay_hashes.extend(
        (
            {
                "path": str(REPO / replication_amendment["materializer"]),
                "sha256": replication_amendment["materializer_sha256"],
            },
            {
                "path": str(REPO / replication_amendment["frozen_p2_launcher"]),
                "sha256": replication_amendment["frozen_p2_launcher_sha256"],
            },
            {
                "path": str(REPO / replication_amendment["overlay_ready"]),
                "sha256": replication_amendment["overlay_ready_sha256"],
            },
        )
    )
    p2_train_authorized = set(
        replication_amendment["authorization"]["p2_train_tasks"]
    )
    r1_train_authorized = set(
        replication_amendment["authorization"]["r1_train_tasks"]
    )
    replication_train_authorized = p2_train_authorized | r1_train_authorized
    replication_eval_amendment_path = (
        manifests / "pi05_replication_frozen_evaluation_amendment_v1.json"
    )
    replication_eval_amendment = json.loads(
        replication_eval_amendment_path.read_text()
    )
    p2_eval_authorized = set(
        replication_eval_amendment["authorization"]["p2_eval_tasks"]
    )
    frozen_p2_eval_launcher = REPO / replication_eval_amendment[
        "frozen_eval_launcher"
    ]
    replication_eval_hashes = list(replication_overlay_hashes)
    replication_eval_hashes.append(
        {
            "path": str(frozen_p2_eval_launcher),
            "sha256": replication_eval_amendment["frozen_eval_launcher_sha256"],
        }
    )
    replication_eval_hashes.extend(
        {
            "path": str(REPO / relative),
            "sha256": expected,
        }
        for relative, expected in sorted(
            replication_eval_amendment["platform_yamls"].items()
        )
    )

    for task in queue.get("tasks", []):
        task_id = task.get("id", "")
        if not task_id.endswith(("_train", "_eval")):
            continue
        if task_id in p1_eval_authorized:
            task["ready_hashes"] = p1_overlay_hashes
            for path in (
                R1_FROZEN_OVERLAY / "READY",
                REPO / p1_amendment["overlay_preflight"],
                REPO / p1_amendment["runtime_wrapper"],
            ):
                path_text = str(path)
                if path_text not in task.setdefault("ready_files", []):
                    task["ready_files"].append(path_text)
            for candidate in task.get("candidates", []):
                if candidate.get("kind") in {"local", "ssh"}:
                    old = (
                        "bash train_scripts/kai/eval/"
                        "run_pi05_predictive_adapter_p1_formal.sh"
                    )
                    new = (
                        f"P1_VERIFY_REPO={shlex.quote(str(R1_FROZEN_OVERLAY))} "
                        "ROBOTWIN_ATTACH_REQUEUE_FAILED=1 bash "
                        "train_scripts/kai/eval/"
                        "run_pi05_predictive_adapter_p1_frozen.sh"
                    )
                    candidate["command"] = candidate["command"].replace(old, new)
                elif candidate.get("kind") == "platform":
                    candidate.setdefault("env", {}).update(
                        {
                            "P1_VERIFY_REPO": str(R1_FROZEN_OVERLAY),
                            "ROBOTWIN_ATTACH_REQUEUE_FAILED": "1",
                        }
                    )
                    if candidate.get("resource") == "Robot-East-H20":
                        candidate["env"].update(
                            {
                                "TORCH_CUDA_ARCH_LIST": "9.0",
                                "TORCH_EXTENSIONS_DIR": (
                                    "/vePFS/tim/runtime/torch_extensions/"
                                    "h20_sm90_py310"
                                ),
                            }
                        )
        elif task_id in p2_eval_authorized:
            task["ready_hashes"] = replication_eval_hashes
            for path in (
                REPLICATION_FROZEN_OVERLAY / "REPLICATION_READY",
                replication_eval_amendment_path,
                frozen_p2_eval_launcher,
            ):
                path_text = str(path)
                if path_text not in task.setdefault("ready_files", []):
                    task["ready_files"].append(path_text)

            eval_env = {
                "P2_VERIFY_REPO": str(REPLICATION_FROZEN_OVERLAY),
                "PYTHONPATH": str(REPLICATION_FROZEN_OVERLAY / "kai0/src"),
                "P2_EVAL_LAUNCHER": str(frozen_p2_eval_launcher),
            }
            assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in eval_env.items()
            )
            for candidate in task.get("candidates", []):
                if candidate.get("kind") == "platform":
                    candidate.setdefault("env", {}).update(eval_env)
                elif candidate.get("kind") in {"local", "ssh"}:
                    candidate["command"] = candidate["command"].replace(
                        "exec env ", f"exec env {assignments} ", 1
                    ).replace(
                        "bash train_scripts/kai/eval/"
                        "run_pi05_predictive_adapter_p2_formal.sh",
                        f"bash {shlex.quote(str(frozen_p2_eval_launcher))}",
                        1,
                    )
        elif task_id in replication_train_authorized:
            task["ready_hashes"] = replication_overlay_hashes
            for path in (
                REPLICATION_FROZEN_OVERLAY / "REPLICATION_READY",
                replication_amendment_path,
            ):
                path_text = str(path)
                if path_text not in task.setdefault("ready_files", []):
                    task["ready_files"].append(path_text)

            env = {
                "TRAIN_SOURCE_REPO": str(REPLICATION_FROZEN_OVERLAY),
                "PYTHONPATH": str(REPLICATION_FROZEN_OVERLAY / "kai0/src"),
            }
            if task_id in p2_train_authorized:
                env["P2_VERIFY_REPO"] = str(REPLICATION_FROZEN_OVERLAY)
            elif "_a0_" in task_id or "_predictive_" in task_id:
                env["TRAIN_VERIFY_REPO"] = str(REPLICATION_FROZEN_OVERLAY)
            else:
                env["R1_VERIFY_REPO"] = str(REPLICATION_FROZEN_OVERLAY)

            assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env.items()
            )
            for candidate in task.get("candidates", []):
                if candidate.get("kind") == "platform":
                    candidate.setdefault("env", {}).update(env)
                elif candidate.get("kind") in {"local", "ssh"}:
                    candidate["command"] = candidate["command"].replace(
                        "exec env ", f"exec env {assignments} ", 1
                    )
        elif task_id.startswith(("pi05_predictive_adapter_p1_", "pi05_predictive_adapter_p2_")):
            task["ready_hashes"] = p1_hashes
        elif task_id in r1_eval_authorized:
            task["ready_hashes"] = r1_overlay_hashes
            overlay_ready = str(R1_FROZEN_OVERLAY / "READY")
            ready_files = task.setdefault("ready_files", [])
            if overlay_ready not in ready_files:
                ready_files.append(overlay_ready)
            cpu_preflight = str(R1_FROZEN_OVERLAY / "CPU_PREFLIGHT")
            if cpu_preflight not in ready_files:
                ready_files.append(cpu_preflight)
        elif task_id.startswith("pi05_r1_"):
            task["ready_hashes"] = r1_hashes


def completion_root_from_glob(pattern: str) -> str:
    """Return the evaluation root for broad and run-tag-specific globs."""
    if "/seed" in pattern:
        return pattern.split("/seed", 1)[0]
    return pattern.split("/**/", 1)[0]


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
            root = completion_root_from_glob(location_pattern)
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


def shared_eval_cell_state(root: Path) -> dict[str, int]:
    """Summarize the four frozen per-seed task schedulers."""
    schedulers = list(root.glob("**/.task_scheduler.json"))
    totals = {
        "schedulers": len(schedulers),
        "completed": 0,
        "in_progress": 0,
        "pending": 0,
        "failed": 0,
    }
    for path in schedulers:
        try:
            payload = json.loads(path.read_text())
        except Exception:
            totals["failed"] += 1
            continue
        for key in ("completed", "in_progress", "pending", "failed"):
            totals[key] += len(payload.get(key, {}))
    return totals


def shared_mt_eval_has_active_work(task_id: str) -> bool:
    spec = PI05_MT12_SHARED_FINALIZERS.get(task_id)
    if spec is None:
        return False
    root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / spec[0]
    state = shared_eval_cell_state(root)
    return (
        state["schedulers"] == 4
        and state["failed"] == 0
        and state["completed"] < 24
        and state["in_progress"] > 0
    )


def refresh_pi05_mt12_shared_finalizers() -> None:
    """Publish an authoritative marker after shared attach workers finish 24 cells."""
    manifest = REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    verifier = REPO / "lmvla/lmwm/scripts/verify_robotwin_fixed_seed_eval.py"
    summarizer = REPO / "lmvla/lmwm/scripts/summarize_robotwin_eval.py"
    for task_id, (result_name, intervention) in PI05_MT12_SHARED_FINALIZERS.items():
        marker = REPO / "logs/resource_markers" / f"{result_name}.ok"
        if marker.is_file():
            continue
        root = REPO / "lmvla/lawam/results/eval_runs/robotwin" / result_name
        state = shared_eval_cell_state(root)
        if state != {
            "schedulers": 4,
            "completed": 24,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
        }:
            continue
        report = REPO / "lmvla/lmwm/docs" / f"{result_name}.json"
        try:
            run(
                [
                    "python3",
                    str(verifier),
                    "--manifest",
                    str(manifest),
                    "--root",
                    str(root),
                ],
                timeout=180,
            )
            report_text = run(
                ["python3", str(summarizer), str(root), "--expected-cells", "24"],
                timeout=180,
            )
            atomic_text(report, report_text)
            atomic_text(
                marker,
                (
                    f"validated={utc_now()}\n"
                    f"intervention={intervention}\n"
                    f"report={report}\n"
                    "finalized_by=resource_aware_scheduler\n"
                ),
            )
            log(f"finalized shared 24-cell result for {task_id}")
        except Exception as exc:
            log(f"shared finalizer failed {task_id}: {type(exc).__name__}: {exc}")


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
                    "import glob; print(len(glob.glob("
                    + repr(remote_root + "/**/summary.json")
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
        REPO / "lmvla/lmwm/data/robotwin_l2_seed_manifests/"
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
        REPO / "lmvla/lawam/results/eval_runs/robotwin/rt_all6_v2_combo_seed2026_unseen"
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
        log(
            f"A3 causal report generation failed {output.name}: {type(exc).__name__}: {exc}"
        )
        return
    atomic_json(output, report)
    log(f"materialized local causal report: {output}")


def run_pi05_a3_causal_report(output: Path, controls: dict[str, Path]) -> None:
    correct = (
        REPO / "lmvla/lawam/results/eval_runs/robotwin/"
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
        REPO / "lmvla/lawam/results/eval_runs/robotwin/"
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
            ssh(
                GSY,
                f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}",
                timeout=180,
            )
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
            "pi05_rt_a0_public_exact_seed1000_shared_final"
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
            ssh(
                GSY,
                f"cd {shlex.quote(NORTH_REPO)} && {shlex.join(command)}",
                timeout=180,
            )
        )
    except Exception as exc:
        log(f"exact A0 gate audit failed: {type(exc).__name__}: {exc}")
        return
    atomic_json(output, result)
    log(
        f"materialized exact A0 gate accepted={result.get('accepted')} "
        f"macro={result.get('macro_success_rate')}: {output}"
    )


def refresh_pi05_actionfix_intervention_gate() -> None:
    """Unlock midpoint interventions only after A3 shows base control."""
    output = REPO / "logs/pi05_actionfix_a3_stack2_intervention_gate.json"
    if output.is_file():
        return
    pattern = (
        REPO
        / "lmvla/lawam/results/eval_runs/robotwin"
        / "pi05_a3_live_seed1000_step20000_stack_blocks_two_probe_actionfix"
        / "**/run.log"
    )
    progress_re = re.compile(r"Success rate:\s*(\d+)/(\d+).*?progress:\s*(\d+)/(\d+)")
    latest: tuple[int, int, int] | None = None
    for path in glob.glob(str(pattern), recursive=True):
        for match in progress_re.finditer(Path(path).read_text(errors="replace")):
            successes, attempts, progress, _ = map(int, match.groups())
            if latest is None or progress > latest[2]:
                latest = (successes, attempts, progress)
    if latest is None:
        return
    successes, attempts, progress = latest
    if progress < 10 or successes < 1:
        return
    atomic_json(
        output,
        {
            "accepted": True,
            "successes": successes,
            "attempts": attempts,
            "progress": progress,
            "criterion": "progress >= 10 and successes >= 1",
            "created_at": utc_now(),
        },
    )
    log(f"unlocked action-fix Stack-2 interventions at {successes}/{attempts}")


def refresh_pi05_actionfix_midpoint_report() -> None:
    script = REPO / "train_scripts/kai/analysis/summarize_pi05_actionfix_midpoint.py"
    root = REPO / "lmvla/lawam/results/eval_runs/robotwin"
    output = REPO / "logs/diagnostics/pi05_actionfix_midpoint_step20000.json"
    try:
        report = json.loads(run(["python3", str(script), str(root)], timeout=30))
    except Exception:
        return
    if output.is_file():
        try:
            if json.loads(output.read_text()) == report:
                return
        except Exception:
            pass
    atomic_json(output, report)


def refresh_pi05_step40000_safety_report() -> None:
    script = REPO / "train_scripts/kai/analysis/summarize_pi05_step40000_safety.py"
    output = REPO / "logs/diagnostics/pi05_step40000_safety.json"
    try:
        report = json.loads(
            run(["python3", str(script), "--repo", str(REPO)], timeout=30)
        )
    except Exception:
        return
    if output.is_file():
        try:
            if json.loads(output.read_text()) == report:
                return
        except Exception:
            pass
    atomic_json(output, report)
    if report.get("complete"):
        log(f"materialized complete step-40k safety report: {output}")


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


def submit_platform(
    candidate: dict[str, Any], credential_profile: str = "primary"
) -> str:
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
    task: dict[str, Any],
    candidate: dict[str, Any],
    job_id: str,
    *,
    backfill: bool = False,
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
    manifest_path = (
        manifest_dir / f"pi05_confirmatory_{arm}_seed{seed_text}_launch.json"
    )
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
        runtime_env.get(
            "PI05_CONFIRM_EXP", f"pi05_robotwin_a0_public_exact_seed{seed_text}"
        ),
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
            source_paths.append(
                f"{source_root}/kai0/scripts/train_pi05_robotwin_confirmatory.py"
            )
        source_hashes = remote_sha256(source_paths)
        source_hashes = {
            Path(path).name: digest for path, digest in source_hashes.items()
        }
        dataset_raw_sha = remote_sha256([dataset_manifest_path])[dataset_manifest_path]
    else:
        dataset_payload = json.loads(shared_dataset_manifest.read_text())
        source_files = [
            REPO / "kai0/src/openpi/training/config.py",
            REPO / "kai0/src/openpi/models/pi0.py",
        ]
        if arm != "a0":
            source_files.append(
                REPO / "kai0/scripts/train_pi05_robotwin_confirmatory.py"
            )
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
            "asset_id": runtime_env.get(
                "PI05_ASSET_ID", "robotwin2.0_absolute_meanstd"
            ),
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
    training_manifest = (
        manifest_dir / f"pi05_confirmatory_{arm}_seed{seed_text}_launch.json"
    )
    runtime_env = {
        str(key): str(value) for key, value in candidate.get("env", {}).items()
    }
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


def capture_pi05_mt3_eval_launch(
    task: dict[str, Any], candidate: dict[str, Any], job_id: str
) -> None:
    """Freeze MT3 checkpoint, protocol, condition, source, and scene provenance."""
    match = re.fullmatch(
        r"pi05_mt3_learned_seed(100[012])_(predicted|null|within_task|oracle)_eval",
        task["id"],
    )
    if match is None:
        return
    seed_text, intervention = match.groups()
    manifest_dir = REPO / "lmvla/paper_iclr_lmvla/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / f"pi05_mt3_seed{seed_text}_{intervention}_eval_launch.json"
    if output.is_file():
        return

    metadata_path = next(
        (
            Path(path)
            for path in [
                *task.get("ready_files", []),
                *candidate.get("ready_files", []),
            ]
            if path.endswith("/49999/params/_METADATA")
        ),
        None,
    )
    if metadata_path is None:
        raise ValueError(
            f"cannot resolve MT3 final checkpoint metadata for {task['id']}"
        )
    selection_path = REPO / "logs/mt_stage_tracker/selection.json"
    selection = json.loads(selection_path.read_text())
    tracker_candidate = selection["selected"]
    if tracker_candidate not in {"current_frame", "history_proprio"}:
        raise ValueError(f"unsupported selected MT3 tracker: {tracker_candidate}")

    protocol_path = manifest_dir / "robotwin_mt3_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text())
    scene_manifest = (
        REPO / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    )
    checkpoint = metadata_path.parent.parent
    norm_stats = checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    pairs = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
    task_map = (
        REPO
        / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/eval_task_id.json"
    )
    source_files = [
        REPO / "train_scripts/kai/eval/run_pi05_mt3_formal.sh",
        REPO / "train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh",
        REPO / "kai0/scripts/serve_policy.py",
        REPO / "kai0/src/openpi/models/pi0.py",
        REPO / "kai0/src/openpi/training/config.py",
        REPO / "lmvla/lawam/examples/Robotwin/eval_files/robotwin_batch_bridge.py",
        REPO / "lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi.py",
    ]
    runtime_env = {
        str(key): str(value) for key, value in candidate.get("env", {}).items()
    }
    manifest = {
        "capture_source": "resource_aware_scheduler MT3 evaluator dispatch",
        "captured_at": utc_now(),
        "job_id": job_id,
        "job_name": candidate.get("task_name", task["id"]),
        "training_seed": int(seed_text),
        "intervention": intervention,
        "selected_tracker": tracker_candidate,
        "inference_config": f"pi05_robotwin_mt3_learned_{tracker_candidate}_exact",
        "checkpoint": str(checkpoint),
        "protocol": {
            "action_representation": protocol["joint_policy_training"][
                "action_representation"
            ],
            "normalization": protocol["joint_policy_training"]["normalization"],
            "policy_updates": protocol["joint_policy_training"]["policy_updates"],
            "policy_batch_size": protocol["joint_policy_training"]["policy_batch_size"],
            "evaluation_cells": 24,
            "evaluation_episodes_per_cell": 50,
            "history_enabled": tracker_candidate == "history_proprio",
            "oracle_enabled": intervention == "oracle",
        },
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
        "sha256": {
            "checkpoint_metadata": sha256_file(metadata_path),
            "selection": sha256_file(selection_path),
            "frozen_protocol": sha256_file(protocol_path),
            "scene_manifest": sha256_file(scene_manifest),
            "normalization": sha256_file(norm_stats),
            "transition_pairs": sha256_file(pairs),
            "transition_task_map": sha256_file(task_map),
            "execution_sources": {
                str(path.relative_to(REPO)): sha256_file(path) for path in source_files
            },
        },
    }
    if candidate["kind"] == "platform":
        manifest["sha256"]["submitted_yaml"] = sha256_file(REPO / candidate["yaml"])
    else:
        manifest["sha256"]["launch_command"] = hashlib.sha256(
            candidate["command"].encode()
        ).hexdigest()
    atomic_json(output, manifest)
    log(
        f"captured MT3 seed{seed_text} {intervention} evaluator provenance "
        f"job_id={job_id}"
    )


def capture_pi05_mt12_training_launch(
    task: dict[str, Any],
    candidate: dict[str, Any],
    attempt: dict[str, Any],
    *,
    backfill: bool = False,
) -> None:
    """Freeze actual MT1/MT2 training provenance, including pre-capture jobs."""
    match = re.fullmatch(
        r"pi05_(mt1_oracle|mt2_null)_seed(100[012])_train",
        task["id"],
    )
    if match is None:
        return
    arm, seed_text = match.groups()
    manifest_dir = REPO / "lmvla/paper_iclr_lmvla/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = manifest_dir / f"pi05_{arm}_seed{seed_text}_train_launch.json"
    if output.is_file():
        return

    dataset_manifest = Path(
        "/vePFS/tim/workspace/VLANeXt-main/datasets/"
        "robotwin2.0_official_prompts_v21/meta/official_prompt_repair_manifest.json"
    )
    base_metadata = REPO / "kai0/checkpoints/pi05_base/params/_METADATA"
    norm_stats = (
        REPO / "kai0/assets/pi05_robotwin_a0_public_exact_bj/"
        "robotwin2.0_absolute_meanstd/norm_stats.json"
    )
    pairs = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
    source_files = [
        REPO / "train_scripts/kai/run_pi05_mt_transition_train.sh",
        REPO / "kai0/scripts/train_pi05_robotwin_confirmatory.py",
        REPO / "kai0/src/openpi/models/pi0.py",
        REPO / "kai0/src/openpi/training/config.py",
        REPO / "kai0/src/openpi/training/data_loader.py",
        REPO / "kai0/src/openpi/policies/aloha_policy.py",
    ]
    config = (
        "pi05_robotwin_mt1_oracle_exact"
        if arm == "mt1_oracle"
        else "pi05_robotwin_mt2_null_exact"
    )
    experiment = f"pi05_robotwin_{arm}_seed{seed_text}"
    manifest = {
        "capture_source": (
            "resource_aware_scheduler active-state backfill"
            if backfill
            else "resource_aware_scheduler dispatch"
        ),
        "captured_at": utc_now(),
        "task_id": task["id"],
        "arm": arm,
        "training_seed": int(seed_text),
        "config_name": config,
        "experiment_name": experiment,
        "attempt": {
            "started_at": attempt.get("started_at"),
            "pid": attempt.get("pid"),
            "job_id": attempt.get("job_id"),
            "resource": attempt.get("resource", candidate["resource"]),
            "backend": attempt.get("kind", candidate["kind"]),
            "gpus": int(attempt.get("gpus", candidate["gpus"])),
            "gpu_indices": attempt.get("gpu_indices", candidate.get("gpu_indices")),
        },
        "protocol": {
            "initialization": "raw pi05_base",
            "dataset_episodes": 27500,
            "action_representation": "absolute joint actions",
            "normalization": "mean/std",
            "image_augmentation": "none",
            "batch_size": 16,
            "num_workers": 8,
            "updates": 50000,
            "checkpoint_interval": 5000,
            "transition_input": "oracle"
            if arm == "mt1_oracle"
            else "parameter-matched null",
        },
        "sha256": {
            "dataset_manifest_raw": sha256_file(dataset_manifest),
            "dataset_manifest_semantic": canonical_manifest_sha256(
                json.loads(dataset_manifest.read_text())
            ),
            "base_checkpoint_metadata": sha256_file(base_metadata),
            "normalization": sha256_file(norm_stats),
            "transition_pairs": sha256_file(pairs),
            "execution_sources": {
                str(path.relative_to(REPO)): sha256_file(path) for path in source_files
            },
        },
    }
    if candidate["kind"] == "platform":
        manifest["sha256"]["submitted_yaml"] = sha256_file(REPO / candidate["yaml"])
    else:
        manifest["sha256"]["launch_command"] = hashlib.sha256(
            candidate["command"].encode()
        ).hexdigest()
    atomic_json(output, manifest)
    log(f"captured {arm} seed{seed_text} training provenance")


def capture_pi05_mt12_eval_launch(
    task: dict[str, Any], candidate: dict[str, Any], job_id: str
) -> None:
    """Freeze P0 control-evaluation provenance on shared or North storage."""
    mt1 = re.fullmatch(
        r"pi05_mt1_oracle_seed(100[012])_(correct|null|within_task|cross_task)_eval",
        task["id"],
    )
    mt2 = re.fullmatch(r"pi05_mt2_null_seed1000_eval", task["id"])
    if mt1 is None and mt2 is None:
        return
    arm = "mt1_oracle" if mt1 is not None else "mt2_null"
    seed_text = mt1.group(1) if mt1 is not None else "1000"
    intervention = mt1.group(2) if mt1 is not None else "null"
    manifest_dir = REPO / "lmvla/paper_iclr_lmvla/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output = (
        manifest_dir / f"pi05_{arm}_seed{seed_text}_{intervention}_eval_launch.json"
    )
    if output.is_file():
        return

    is_north = candidate["resource"] == "Robot-North-H20"
    key = "ready_files_remote" if is_north else "ready_files"
    paths = [*task.get(key, []), *candidate.get(key, [])]
    metadata_path = next(
        (path for path in paths if path.endswith("/49999/params/_METADATA")),
        None,
    )
    if metadata_path is None:
        raise ValueError(f"cannot resolve P0 eval checkpoint metadata for {task['id']}")
    checkpoint = str(Path(metadata_path).parent.parent)
    root = NORTH_REPO if is_north else str(REPO)
    scene_manifest = (
        PI05_CONFIRMATORY_SCENE_MANIFEST_NORTH
        if is_north
        else PI05_CONFIRMATORY_SCENE_MANIFEST_SHARED
    )
    norm_stats = f"{checkpoint}/assets/robotwin2.0_absolute_meanstd/norm_stats.json"
    pairs = f"{root}/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/pairs.npz"
    task_map = f"{root}/lmvla/lmwm/data/robotwin_milestone_all6_confirmatory_v1/eval_task_id.json"
    source_relpaths = (
        "train_scripts/kai/eval/run_pi05_mt_transition_formal.sh",
        "train_scripts/kai/eval/local_robotwin_a3_official_2gpu.sh",
        "kai0/scripts/serve_policy.py",
        "kai0/src/openpi/models/pi0.py",
        "lmvla/lawam/examples/Robotwin/eval_files/robotwin_batch_bridge.py",
        "lmvla/lawam/examples/Robotwin/eval_files/model2robotwin_openpi.py",
    )
    source_paths = [f"{root}/{relative}" for relative in source_relpaths]
    hash_paths = [
        metadata_path,
        norm_stats,
        scene_manifest,
        pairs,
        task_map,
        *source_paths,
    ]
    if is_north:
        path_hashes = remote_sha256(hash_paths)
    else:
        path_hashes = {path: sha256_file(Path(path)) for path in hash_paths}
    training_manifest = manifest_dir / f"pi05_{arm}_seed{seed_text}_train_launch.json"
    runtime_env = {
        str(key): str(value) for key, value in candidate.get("env", {}).items()
    }
    manifest = {
        "capture_source": "resource_aware_scheduler P0 evaluator dispatch",
        "captured_at": utc_now(),
        "job_id": job_id,
        "job_name": candidate.get("task_name", task["id"]),
        "arm": arm,
        "training_seed": int(seed_text),
        "intervention": intervention,
        "checkpoint": checkpoint,
        "protocol": {
            "action_representation": "absolute joint actions",
            "normalization": "mean/std",
            "scene_cells": 24,
            "episodes_per_cell": 50,
            "fixed_seed_max_attempts": 500,
            "oracle_trajectory_enabled": arm == "mt1_oracle",
        },
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
        "sha256": {
            "checkpoint_metadata": path_hashes[metadata_path],
            "training_launch_manifest": sha256_file(training_manifest),
            "normalization": path_hashes[norm_stats],
            "scene_manifest": path_hashes[scene_manifest],
            "transition_pairs": path_hashes[pairs],
            "transition_task_map": path_hashes[task_map],
            "execution_sources": {
                relative: path_hashes[path]
                for relative, path in zip(source_relpaths, source_paths, strict=True)
            },
        },
    }
    if candidate["kind"] == "platform":
        manifest["sha256"]["submitted_yaml"] = sha256_file(REPO / candidate["yaml"])
    else:
        manifest["sha256"]["launch_command"] = hashlib.sha256(
            candidate["command"].encode()
        ).hexdigest()
    atomic_json(output, manifest)
    log(
        f"captured {arm} seed{seed_text} {intervention} evaluator provenance "
        f"resource={candidate['resource']} job_id={job_id}"
    )


def refresh_pi05_mt12_training_provenance(
    queue: dict[str, Any], state: dict[str, Any]
) -> None:
    """Backfill MT1/MT2 manifests for active jobs launched before capture existed."""
    tasks_by_id = {task["id"]: task for task in queue["tasks"]}
    for task_id, task_state in state.get("tasks", {}).items():
        if not re.fullmatch(r"pi05_(mt1_oracle|mt2_null)_seed100[012]_train", task_id):
            continue
        attempts = task_state.get("attempts", [])
        if not attempts:
            continue
        task = tasks_by_id.get(task_id)
        if task is None:
            continue
        attempt = attempts[-1]
        candidate = next(
            (
                item
                for item in task.get("candidates", [])
                if item.get("resource") == attempt.get("resource")
                and item.get("kind") == attempt.get("kind")
            ),
            None,
        )
        if candidate is None:
            continue
        try:
            capture_pi05_mt12_training_launch(task, candidate, attempt, backfill=True)
        except Exception as exc:
            log(
                f"MT1/MT2 provenance backfill failed {task_id}: "
                f"{type(exc).__name__}: {exc}"
            )


def audit_pi05_mt1_replication_checkpoint(seed: int, step: int) -> Path | None:
    checkpoint = (
        REPO
        / "kai0/checkpoints/pi05_robotwin_mt1_oracle_exact"
        / f"pi05_robotwin_mt1_oracle_seed{seed}/{step}"
    )
    root_metadata = checkpoint / "_CHECKPOINT_METADATA"
    if not root_metadata.is_file():
        return None
    required = {
        "root_metadata": root_metadata,
        "params_metadata": checkpoint / "params/_METADATA",
        "train_state_metadata": checkpoint / "train_state/_METADATA",
        "norm_stats": (
            checkpoint / "assets/robotwin2.0_absolute_meanstd/norm_stats.json"
        ),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    empty = [
        str(path)
        for path in required.values()
        if path.is_file() and path.stat().st_size <= 0
    ]
    if missing or empty:
        log(
            f"MT1 checkpoint audit waiting seed={seed} step={step} "
            f"missing={missing} empty={empty}"
        )
        return None
    try:
        norm_payload = json.loads(required["norm_stats"].read_text())
    except Exception as exc:
        log(
            f"MT1 checkpoint audit invalid norm seed={seed} step={step}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    if not isinstance(norm_payload, dict) or not norm_payload:
        log(f"MT1 checkpoint audit empty norm payload seed={seed} step={step}")
        return None

    audit_dir = REPO / "logs/checkpoint_audits"
    audit_path = audit_dir / f"pi05_mt1_oracle_seed{seed}_step{step}.json"
    marker = (
        REPO
        / "logs/resource_markers"
        / f"pi05_mt1_oracle_seed{seed}_step{step}_checkpoint_audit.ok"
    )
    if audit_path.is_file() and marker.is_file():
        return marker
    file_sizes = {name: path.stat().st_size for name, path in required.items()}
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    )
    audit = {
        "schema_version": 1,
        "audited_at": utc_now(),
        "accepted": True,
        "training_seed": seed,
        "step": step,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes,
        "required_files": {name: str(path) for name, path in required.items()},
        "file_sizes": file_sizes,
        "sha256": {name: sha256_file(path) for name, path in required.items()},
        "norm_asset_ids": sorted(norm_payload),
    }
    atomic_json(audit_path, audit)
    atomic_text(
        marker,
        (
            f"audited={audit['audited_at']}\n"
            f"checkpoint={checkpoint}\n"
            f"audit={audit_path}\n"
        ),
    )
    log(f"audited MT1 checkpoint seed={seed} step={step} bytes={checkpoint_bytes}")
    return marker


def refresh_pi05_mt1_replication_checkpoint_audits() -> None:
    for seed in (1001, 1002):
        for step in (*range(5000, 50000, 5000), 49999):
            audit_pi05_mt1_replication_checkpoint(seed, step)


def refresh_pi05_launch_provenance(
    queue: dict[str, Any], state: dict[str, Any]
) -> None:
    """Backfill manifests for jobs that were dispatched before unified capture existed."""
    for task in queue["tasks"]:
        if not re.fullmatch(
            r"pi05_(a0_public_exact|a2_abs_confirmatory|a3_live_confirmatory)_seed100[012]_train",
            task["id"],
        ):
            continue
        task_state = state["tasks"].get(task["id"], {})
        attempts = [
            attempt
            for attempt in task_state.get("attempts", [])
            if attempt.get("job_id") or attempt.get("pid")
        ]
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
    del candidate
    raise RuntimeError(PERMANENTLY_DISABLED_RESOURCES["gf1"])


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
        f'echo "RUNNING start=$start host=$(hostname)" > {shlex.quote(str(status_path))}; '
        f"bash -lc {shlex.quote(command)}; rc=$?; end=$(date -u +%FT%TZ); "
        f'echo "FINISHED rc=$rc start=$start end=$end host=$(hostname)" > {shlex.quote(str(status_path))}; exit $rc'
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
    task_state.pop("waiting_reason", None)
    attempt = task_state["attempts"][-1]
    if task.get("completion_locations") and (
        task_state.get("artifacts_complete")
        or any(
            glob.glob(location["glob"], recursive=True)
            for location in task["completion_locations"]
            if not location.get("remote")
        )
    ):
        complete, evidence = completion_evidence(task)
        record_artifact_progress(task_state, complete, evidence)
        if complete:
            if attempt.get("last_state") not in TERMINAL_STATES:
                stop_managed_attempt(attempt)
                attempt["stopped_after_completion_artifact"] = utc_now()
            mark_task_completed(task, task_state)
            return
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
                attempt["failure"] = (
                    f"terminal state without complete outputs: {evidence}"
                )
                log(f"retrying {task['id']}: {evidence}")
        elif info["state"] in {"Failed", "Stopped"}:
            complete, evidence = completion_evidence(task)
            attempt["completion_evidence"] = evidence
            record_artifact_progress(task_state, complete, evidence)
            if complete:
                mark_task_completed(task, task_state)
            elif shared_mt_eval_has_active_work(task["id"]):
                attempt["passive_shared_wait"] = True
                attempt["failure"] = info["message"]
                task_state["waiting_reason"] = (
                    "platform parent exited; waiting for healthy shared attach workers"
                )
            else:
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                attempt["failure"] = info["message"]
        elif info["state"] == "Queueing":
            queue_timeout = int(attempt.get("queue_timeout_seconds", 300))
            if queued_attempt_timed_out(attempt):
                service(attempt["region"], credential_profile).json(
                    "StopJob", {}, json.dumps({"Id": attempt["job_id"]}).encode()
                )
                task_state["status"] = "pending"
                attempt["finished_at"] = utc_now()
                if attempt.get("resource") == "robot-task":
                    attempt["failure"] = (
                        "reclaimed after queueing because Shanghai queueing is disabled"
                    )
                else:
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
        attempt.pop("monitor_error", None)
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
                attempt["failure"] = (
                    f"successful process without complete outputs: {evidence}"
                )
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
                probe = ssh(
                    GF1,
                    (
                        f"if kill -0 {int(attempt['pid'])} 2>/dev/null; "
                        "then echo ALIVE; else echo DEAD; fi"
                    ),
                    timeout=20,
                ).strip()
            except Exception as exc:
                attempt["monitor_error"] = (
                    f"launcher PID probe unavailable: {type(exc).__name__}: {exc}"
                )
                attempt["last_checked_at"] = utc_now()
                return
            if probe == "ALIVE":
                attempt.pop("launcher_dead_confirmations", None)
                attempt.pop("monitor_status", None)
            elif probe == "DEAD":
                confirmations = int(attempt.get("launcher_dead_confirmations", 0)) + 1
                attempt["launcher_dead_confirmations"] = confirmations
                attempt["monitor_status"] = (
                    "launcher explicitly dead "
                    f"({confirmations}/{REMOTE_LAUNCHER_DEAD_CONFIRMATIONS})"
                )
                if confirmations >= REMOTE_LAUNCHER_DEAD_CONFIRMATIONS:
                    task_state["status"] = "pending"
                    attempt["finished_at"] = utc_now()
                    attempt["failure"] = (
                        "launcher explicitly dead for three consecutive polls "
                        "while status remained RUNNING"
                    )
                    log(f"reclaimed orphaned gf1 launcher for {task['id']}")
            else:
                attempt["monitor_error"] = f"unexpected launcher PID probe: {probe!r}"
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
                attempt["monitor_error"] = (
                    f"cannot inspect launcher pid={attempt['pid']}"
                )
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
                attempt["failure"] = (
                    f"successful process without complete outputs: {evidence}"
                )
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
                attempt["failure"] = (
                    "launcher disappeared while status remained RUNNING"
                )
                log(f"reclaimed orphaned local launcher for {task['id']}")
            except PermissionError:
                attempt["monitor_error"] = (
                    f"cannot inspect launcher pid={attempt['pid']}"
                )


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
    gf1 = {
        "available": False,
        "submission_enabled": False,
        "retired": True,
        "retired_reason": PERMANENTLY_DISABLED_RESOURCES["gf1"],
        "count": 0,
        "free_count": 0,
        "gpus": [],
        "watched_tasks": {},
    }
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
        "managed_queued_gpus": 0,
        "managed_queueing": [],
        "personal_limit": NORTH_BACKUP_PERSONAL_LIMIT,
        "managed_submitted_jobs": 0,
        "max_submitted_jobs": NORTH_BACKUP_MAX_JOBS,
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
                    "managed_queued_gpus": sum(
                        job["_gpus"]
                        for job in managed_jobs
                        if job["_state"] in WAITING_STATES
                    ),
                    "managed_queueing": [
                        job.get("Id")
                        for job in managed_jobs
                        if job["_state"] in WAITING_STATES
                    ],
                    "managed_submitted_jobs": len(managed_jobs),
                }
            )
        except Exception as exc:
            backup_north["error_type"] = type(exc).__name__
    tracked = {}
    for job_id, (region, label) in TRACKED_JOBS.items():
        try:
            tracked[job_id] = {
                "label": label,
                "region": region,
                **get_job(region, job_id),
            }
        except Exception as exc:
            tracked[job_id] = {"label": label, "region": region, "error": str(exc)}
    return {
        "timestamp": utc_now(),
        "staged_failovers": staged_failover_statuses(),
        "resources": {
            "beijing": {
                "available": "beijing" not in queue_errors,
                "owned_active_gpus": sum(
                    job["_gpus"]
                    for job in north_owned
                    if job["_state"] in ACTIVE_STATES
                ),
                "owned_queued_gpus": sum(
                    job["_gpus"]
                    for job in north_owned
                    if job["_state"] in WAITING_STATES
                ),
                "owned_queueing": [
                    job.get("Id")
                    for job in north_owned
                    if job["_state"] in WAITING_STATES
                ],
                "owned_submitted_jobs": len(north_owned),
                "active_gpus_all_users": sum(
                    job["_gpus"] for job in north if job["_state"] in ACTIVE_STATES
                ),
                "queueing_all_users": [
                    job.get("Id") for job in north if job["_state"] in WAITING_STATES
                ],
                "capacity": NORTH_CAPACITY,
                "personal_limit": NORTH_PERSONAL_LIMIT,
                "max_submitted_jobs": NORTH_PRIMARY_MAX_JOBS,
                "backup": backup_north,
                "watched_tasks": north_watched,
            },
            "robot-task": {
                "available": "robot-task" not in queue_errors,
                "submission_enabled": not ROBOT_TASK_DISABLE_MARKER.exists(),
                "active_gpus_all_users": sum(
                    job["_gpus"] for job in shanghai if job["_state"] in ACTIVE_STATES
                ),
                "owned_active_gpus": sum(
                    job["_gpus"]
                    for job in shanghai_owned
                    if job["_state"] in ACTIVE_STATES
                ),
                "queueing_all_users": [
                    job.get("Id") for job in shanghai if job["_state"] in WAITING_STATES
                ],
                "owned_queueing": [
                    job.get("Id")
                    for job in shanghai_owned
                    if job["_state"] in WAITING_STATES
                ],
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
                    job.get("Id") for job in east if job["_state"] in WAITING_STATES
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
        (
            "Beijing primary owned",
            resources["beijing"]["owned_active_gpus"],
            NORTH_PERSONAL_LIMIT,
            len(resources["beijing"]["owned_queueing"]),
        ),
        (
            "Beijing all users",
            resources["beijing"]["active_gpus_all_users"],
            NORTH_CAPACITY,
            len(resources["beijing"]["queueing_all_users"]),
        ),
        (
            "robot-task owned",
            resources["robot-task"]["owned_active_gpus"],
            SH_PERSONAL_LIMIT,
            len(resources["robot-task"]["owned_queueing"]),
        ),
        (
            "robot-task all users",
            resources["robot-task"]["active_gpus_all_users"],
            SH_CAPACITY,
            len(resources["robot-task"]["queueing_all_users"]),
        ),
        (
            "Robot-East-H20 all users",
            resources["Robot-East-H20"]["active_gpus_all_users"],
            8,
            len(resources["Robot-East-H20"]["queueing_all_users"]),
        ),
        (
            "gf1",
            resources["gf1"]["count"] - resources["gf1"]["free_count"],
            resources["gf1"]["count"],
            0,
        ),
        (
            "local",
            resources["local"]["count"] - resources["local"]["free_count"],
            resources["local"]["count"],
            0,
        ),
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
        "Dispatch priority: `Robot-East-H20 > Robot-North-H20 > robot-task`; gf1 is permanently retired.",
        f"robot-task new submissions: `{'enabled' if resources['robot-task'].get('submission_enabled', True) else 'disabled'}`.",
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
            "## Beijing Submission Quotas",
            "",
            "| Credential | Submitted jobs | Job limit | Active GPUs | GPU limit |",
            "|---|---:|---:|---:|---:|",
            (
                "| primary | "
                f"{resources['beijing'].get('owned_submitted_jobs', 0)} | "
                f"{resources['beijing'].get('max_submitted_jobs', NORTH_PRIMARY_MAX_JOBS)} | "
                f"{resources['beijing']['owned_active_gpus']} | {NORTH_PERSONAL_LIMIT} |"
            ),
        ]
    )
    if backup.get("enabled"):
        lines.append(
            "| backup | "
            f"{backup.get('managed_submitted_jobs', 0)} | "
            f"{backup.get('max_submitted_jobs', NORTH_BACKUP_MAX_JOBS)} | "
            f"{backup.get('managed_active_gpus', 0)} | "
            f"{backup.get('personal_limit', NORTH_BACKUP_PERSONAL_LIMIT)} |"
        )
    scheduler_states = list(snapshot.get("scheduler_tasks", {}).values())
    running_count = sum(state.get("status") == "running" for state in scheduler_states)
    resource_wait_count = sum(
        state.get("status") == "pending"
        and state.get("waiting_reason") == "waiting for an eligible resource"
        for state in scheduler_states
    )
    input_blocked_count = sum(
        state.get("status") == "pending"
        and state.get("waiting_reason")
        == "blocked by required input, checkpoint, or gate"
        for state in scheduler_states
    )
    lines.extend(
        [
            "",
            "## Dispatch Readiness",
            "",
            f"- Running: `{running_count}`",
            f"- Ready but waiting for a resource: `{resource_wait_count}`",
            f"- Waiting for checkpoint, input, or gate: `{input_blocked_count}`",
        ]
    )
    inventory = snapshot.get("queue_inventory", {})
    if inventory:
        lines.append(
            "- Queue inventory: "
            + ", ".join(
                f"{label}=`{inventory.get(label, 0)}`"
                for label in ("completed", "running", "pending", "disabled", "total")
            )
        )
    staged_failovers = snapshot.get("staged_failovers", {})
    if staged_failovers:
        lines.extend(
            [
                "",
                "## Staged Failover (Not Execution)",
                "",
                "| Stage | Status | Global progress | Rate | ETA | Verified | Launch authorized | Updated |",
                "|---|---|---:|---:|---:|---|---|---|",
            ]
        )
        for label, status in sorted(staged_failovers.items()):
            progress = status.get("progress_fraction", 0.0)
            progress_text = (
                f"{100.0 * progress:.2f}%"
                if isinstance(progress, (int, float))
                else ""
            )
            rate = status.get("rate_bytes_per_second")
            rate_text = (
                f"{rate / 1_000_000:.2f} MB/s"
                if isinstance(rate, (int, float)) and rate > 0
                else ""
            )
            eta_seconds = status.get("eta_seconds")
            eta_text = (
                f"{eta_seconds / 3600:.2f} h"
                if isinstance(eta_seconds, (int, float)) and eta_seconds >= 0
                else ""
            )
            lines.append(
                f"| `{label}` | {status.get('status', 'unknown')} | {progress_text} | "
                f"{rate_text} | {eta_text} | "
                f"{status.get('stage_verified', False)} | "
                f"{status.get('launch_authorized', False)} | "
                f"{status.get('timestamp', '')} |"
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
            progress = "; ".join(
                value
                for value in (
                    state.get("runtime_progress"),
                    state.get("artifact_progress"),
                )
                if value
            )
        lines.append(f"| `{task_id}` | {state.get('status', 'unknown')} | {progress} |")
    lines.extend(
        [
            "",
            "## Training Heartbeats",
            "",
            "| Resource | Task | Step | Rate | ETA | Health | Status |",
            "|---|---|---:|---|---|---|---|",
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
            rate_text = (
                f"{rate:.2f} s/step"
                if isinstance(rate, (int, float))
                else str(status.get("rate_it_s", ""))
            )
            eta = status.get("eta_hours", status.get("eta", ""))
            eta_text = f"{eta:.2f} h" if isinstance(eta, (int, float)) else str(eta)
            lines.append(
                f"| {resource} | `{task}` | {status.get('step', '')} | {rate_text} | {eta_text} | {status.get('health', '')} | {status.get('status', '')} |"
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
        if not state.get("submission_enabled", True):
            return False
        required_indices = candidate.get("gpu_indices")
        if required_indices is not None:
            reserved_indices = set(state.get("managed_reserved_indices", []))
            free_indices = {
                row["index"]
                for row in state.get("gpus", [])
                if row["memory_used_mib"] < 1024
                and row["index"] not in reserved_indices
            }
            return (
                state.get("available", True) and set(required_indices) <= free_indices
            )
        return state.get("available", True) and state["free_count"] >= gpus
    if resource == "local":
        state = resources["local"]
        required_indices = candidate.get("gpu_indices")
        if required_indices is not None:
            reserved_indices = set(state.get("managed_reserved_indices", []))
            free_indices = {
                row["index"]
                for row in state.get("gpus", [])
                if row["memory_used_mib"] < 1024
                and row["index"] not in reserved_indices
            }
            return (
                state.get("available", True) and set(required_indices) <= free_indices
            )
        return state.get("available", True) and state["free_count"] >= gpus
    if resource == "robot-task":
        state = resources["robot-task"]
        if not state.get("submission_enabled", True):
            return False
        # Nominal free cards can be split across nodes. Try once when the
        # nominal count fits, then let queue-timeout recovery hold subsequent
        # retries until active usage actually drops.
        free = state["capacity"] - state["active_gpus_all_users"]
        min_dispatch_free = int(
            candidate.get("min_dispatch_free", SH_MIN_DISPATCH_FREE)
        )
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
        physical_free = state["capacity"] - state["active_gpus_all_users"]
        min_dispatch_free = int(candidate.get("min_dispatch_free", gpus))
        if credential_profile == "primary":
            return (
                state.get("available", True)
                and not state["owned_queueing"]
                and not state.get("queueing_all_users")
                and state.get("owned_submitted_jobs", 0) + 1
                <= state.get("max_submitted_jobs", NORTH_PRIMARY_MAX_JOBS)
                and state["owned_active_gpus"]
                + state.get("owned_queued_gpus", 0)
                + gpus
                <= state["personal_limit"]
                and physical_free >= max(gpus, min_dispatch_free)
            )
        backup = state.get("backup", {})
        return (
            credential_profile == "backup"
            # Spill to the backup identity once the next task no longer fits
            # under the primary identity's personal limit. Requiring the
            # primary usage to equal the limit leaves unusable 1-3 GPU gaps.
            and state.get("available", True)
            and (
                state["owned_active_gpus"] + state.get("owned_queued_gpus", 0) + gpus
                > state["personal_limit"]
                or state.get("owned_submitted_jobs", 0) + 1
                > state.get("max_submitted_jobs", NORTH_PRIMARY_MAX_JOBS)
            )
            and backup.get("enabled")
            and backup.get("available")
            and not backup.get("managed_queueing")
            and not state.get("queueing_all_users")
            and backup.get("managed_submitted_jobs", 0) + 1
            <= backup.get("max_submitted_jobs", NORTH_BACKUP_MAX_JOBS)
            and backup.get("managed_active_gpus", 0)
            + backup.get("managed_queued_gpus", 0)
            + gpus
            <= backup.get("personal_limit", NORTH_BACKUP_PERSONAL_LIMIT)
            and physical_free >= max(gpus, min_dispatch_free)
        )
    return False


def reserve_dispatched_candidate(
    snapshot: dict[str, Any],
    candidate: dict[str, Any],
    credential_profile: str = "primary",
) -> None:
    """Conservatively reserve a just-dispatched task in the current snapshot."""
    resource = candidate["resource"]
    gpus = int(candidate["gpus"])
    resources = snapshot["resources"]
    if resource in {"gf1", "local"}:
        state = resources[resource]
        state["managed_reserved_gpus"] = (
            int(state.get("managed_reserved_gpus", 0)) + gpus
        )
        state["free_count"] = max(0, int(state.get("free_count", 0)) - gpus)
        indices = set(state.get("managed_reserved_indices", []))
        indices.update(int(index) for index in candidate.get("gpu_indices", []))
        state["managed_reserved_indices"] = sorted(indices)
        return
    if resource == "robot-task":
        state = resources[resource]
        state["active_gpus_all_users"] += gpus
        state["owned_active_gpus"] += gpus
        return
    if resource == "Robot-East-H20":
        resources[resource]["active_gpus_all_users"] += gpus
        return
    if resource == "Robot-North-H20":
        state = resources["beijing"]
        state["active_gpus_all_users"] += gpus
        if credential_profile == "backup":
            state["backup"]["managed_active_gpus"] += gpus
            state["backup"]["managed_submitted_jobs"] = (
                state["backup"].get("managed_submitted_jobs", 0) + 1
            )
        else:
            state["owned_active_gpus"] += gpus
            state["owned_submitted_jobs"] = state.get("owned_submitted_jobs", 0) + 1


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


def ordered_dispatch_candidates(
    task: dict[str, Any], snapshot: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Use the full submission-router score for actual dispatch order."""
    catalog = submission_router.load_json(submission_router.DEFAULT_CATALOG)
    orders: dict[int, dict[str, int]] = {}
    indexed = list(enumerate(task.get("candidates", [])))

    def preference_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, candidate = item
        gpus = int(candidate.get("gpus", 0))
        if gpus not in orders:
            orders[gpus] = {
                resource: rank
                for rank, resource in enumerate(
                    submission_router.preference_order(gpus, catalog)
                )
            }
        return (
            orders[gpus].get(candidate.get("resource", ""), len(orders[gpus])),
            index,
        )

    if snapshot is None:
        return [candidate for _, candidate in sorted(indexed, key=preference_key)]

    prefixes = [
        prefix
        for spec in catalog.get("filesystems", {}).values()
        for prefix in spec.get("mount_prefixes", [])
    ]

    prefer_max_gpus = bool(task.get("prefer_max_gpus_when_immediate"))
    prefer_min_gpus = bool(task.get("prefer_min_gpus_when_immediate"))
    if prefer_max_gpus and prefer_min_gpus:
        raise ValueError("task cannot prefer both minimum and maximum GPU shapes")

    def router_key(
        item: tuple[int, dict[str, Any]],
    ) -> tuple[bool, int, int, int, int]:
        index, candidate = item
        gpus = int(candidate.get("gpus", 0))
        if gpus <= 0:
            preference_rank, _ = preference_key(item)
            return (False, 0, preference_rank, preference_rank, index)
        input_paths = _submission_input_paths(task, candidate)
        known_paths = [
            path
            for path in input_paths
            if any(
                path == prefix or path.startswith(prefix.rstrip("/") + "/")
                for prefix in prefixes
            )
        ]
        locations = submission_router.infer_filesystems(known_paths, catalog)
        recommendations = submission_router.rank_targets(
            gpus=gpus,
            catalog=catalog,
            snapshot=snapshot,
            data_locations=locations,
        )
        selected = next(
            (
                recommendation
                for recommendation in recommendations
                if recommendation.resource == candidate.get("resource")
            ),
            None,
        )
        if selected is None:
            return (True, 0, 10**9, 10**9, index)
        return (
            not selected.immediately_runnable,
            (
                -gpus
                if prefer_max_gpus and selected.immediately_runnable
                else gpus
                if prefer_min_gpus and selected.immediately_runnable
                else 0
            ),
            selected.score,
            selected.rank,
            index,
        )

    return [candidate for _, candidate in sorted(indexed, key=router_key)]


def candidate_env_value(candidate: dict[str, Any], key: str) -> str | None:
    value = candidate.get("env", {}).get(key)
    if value is not None:
        return str(value)
    command = candidate.get("command")
    if not command:
        return None
    prefix = f"{key}="
    for token in shlex.split(command):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def north_queue_credential_profile(
    candidate: dict[str, Any], snapshot: dict[str, Any]
) -> str | None:
    """Choose a North identity when this task must enter the designated queue."""
    if (
        candidate.get("kind") != "platform"
        or candidate.get("resource") != "Robot-North-H20"
        or not readiness_spec_satisfied(candidate)
    ):
        return None
    state = snapshot["resources"]["beijing"]
    if not state.get("available", True):
        return None
    gpus = int(candidate["gpus"])
    primary_jobs_fit = state.get("owned_submitted_jobs", 0) + 1 <= state.get(
        "max_submitted_jobs", NORTH_PRIMARY_MAX_JOBS
    )
    primary_gpus_fit = (
        state.get("owned_active_gpus", 0) + state.get("owned_queued_gpus", 0) + gpus
        <= state["personal_limit"]
    )
    backup = state.get("backup", {})
    backup_jobs_fit = backup.get("managed_submitted_jobs", 0) + 1 <= backup.get(
        "max_submitted_jobs", NORTH_BACKUP_MAX_JOBS
    )
    backup_gpus_fit = backup.get("managed_active_gpus", 0) + backup.get(
        "managed_queued_gpus", 0
    ) + gpus <= backup.get("personal_limit", NORTH_BACKUP_PERSONAL_LIMIT)
    backup_usable = (
        backup.get("enabled")
        and backup.get("available")
        and backup_jobs_fit
        and backup_gpus_fit
    )
    if primary_jobs_fit and primary_gpus_fit:
        return "primary"
    if backup_usable:
        return "backup"
    if primary_jobs_fit:
        return "primary"
    return None


def reserve_queued_north_candidate(
    snapshot: dict[str, Any], candidate: dict[str, Any], credential_profile: str
) -> None:
    """Project identity quota without claiming physical GPUs for a queued job."""
    state = snapshot["resources"]["beijing"]
    gpus = int(candidate["gpus"])
    if credential_profile == "backup":
        backup = state["backup"]
        backup["managed_submitted_jobs"] = backup.get("managed_submitted_jobs", 0) + 1
        backup["managed_queued_gpus"] = backup.get("managed_queued_gpus", 0) + gpus
    else:
        state["owned_submitted_jobs"] = state.get("owned_submitted_jobs", 0) + 1
        state["owned_queued_gpus"] = state.get("owned_queued_gpus", 0) + gpus


def queued_attempt_timed_out(attempt: dict[str, Any]) -> bool:
    if attempt.get("persistent_north_queue_sink"):
        return False
    if attempt.get("resource") == "robot-task":
        return True
    started = datetime.fromisoformat(attempt["started_at"].replace("Z", "+00:00"))
    queue_timeout = int(attempt.get("queue_timeout_seconds", 300))
    return (datetime.now(timezone.utc) - started).total_seconds() > queue_timeout


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
            current_active = snapshot["resources"]["robot-task"][
                "active_gpus_all_users"
            ]
            if current_active < int(attempt["active_gpus_at_dispatch"]):
                return False
        failed_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return (now - failed_at).total_seconds() < int(
            candidate.get("retry_cooldown_seconds", RETRY_COOLDOWN_SECONDS)
        )
    return False


def robot_task_fragmentation_blocked(
    candidate: dict[str, Any],
    state: dict[str, Any],
    snapshot: dict[str, Any],
) -> bool:
    """Share a failed Shanghai shape probe across equivalent queued tasks."""
    if candidate.get("resource") != "robot-task":
        return False
    requested = int(candidate.get("gpus", 0))
    current_active = int(snapshot["resources"]["robot-task"]["active_gpus_all_users"])
    transient_markers = ("reclaimed after queueing", "剩余配额不足")
    for task_state in state.get("tasks", {}).values():
        for attempt in reversed(task_state.get("attempts", [])):
            if (
                attempt.get("resource") != "robot-task"
                or attempt.get("active_gpus_at_dispatch") is None
                or not any(
                    marker in str(attempt.get("failure", ""))
                    for marker in transient_markers
                )
            ):
                continue
            failed_shape = int(attempt.get("gpus") or 0)
            failed_active = int(attempt["active_gpus_at_dispatch"])
            if failed_shape <= 0:
                continue
            if requested >= failed_shape and current_active >= failed_active:
                return True
    return False


def candidate_failure_count(
    task_state: dict[str, Any],
    candidate: dict[str, Any],
    credential_profile: str = "primary",
) -> int:
    """Count runtime/template failures, excluding transient capacity failures."""
    transient_markers = ("reclaimed after queueing", "剩余配额不足")
    ignore_before = task_state.get("ignore_failures_before")

    def is_current_failure(attempt: dict[str, Any]) -> bool:
        if not ignore_before:
            return True
        finished_at = attempt.get("finished_at")
        if not finished_at:
            return True
        return datetime.fromisoformat(finished_at.replace("Z", "+00:00")) > datetime.fromisoformat(
            ignore_before.replace("Z", "+00:00")
        )

    return sum(
        1
        for attempt in task_state.get("attempts", [])
        if attempt.get("resource") == candidate["resource"]
        and attempt.get("credential_profile", "primary") == credential_profile
        and attempt.get("failure")
        and is_current_failure(attempt)
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
        resource
        if credential_profile == "primary"
        else f"{resource}@{credential_profile}"
    )
    if exhausted.get(resource_key, {}).get("failures") != failures:
        exhausted[resource_key] = {
            "failures": failures,
            "limit": limit,
            "updated_at": utc_now(),
        }
        if task_id:
            log(f"exhausted {resource_key} for {task_id} after {failures} failures")
    return True


def stop_managed_attempt(attempt: dict[str, Any]) -> None:
    """Stop an active helper, including its evaluator/server child processes."""
    if attempt.get("kind") == "platform" and attempt.get("job_id"):
        profile = attempt.get("credential_profile", "primary")
        service(attempt["region"], profile).json(
            "StopJob", {}, json.dumps({"Id": attempt["job_id"]}).encode()
        )
        return
    pid = int(attempt.get("pid", 0))
    if pid <= 0:
        return
    if attempt.get("kind") == "ssh":
        if attempt.get("resource") in PERMANENTLY_DISABLED_RESOURCES:
            return
        ssh(
            GF1,
            f"kill -TERM -- -{pid} 2>/dev/null || kill -TERM {pid} 2>/dev/null || true",
        )
    elif attempt.get("kind") == "local":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def north_materialization_required(parent_state: dict[str, Any]) -> bool | None:
    """Return whether a completed parent needs its North-only result copied back."""
    if parent_state.get("status") != "completed" or not parent_state.get("attempts"):
        return None
    return parent_state["attempts"][-1].get("resource") == "Robot-North-H20"


def dispatch(
    queue: dict[str, Any], state: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    tasks = sorted(queue["tasks"], key=lambda item: (item["priority"], item["id"]))
    dispatched = 0

    for task in tasks:
        ready_file = task.get("rearm_after_ready_file")
        if not ready_file:
            continue
        path = Path(ready_file)
        if not path.is_file():
            continue
        task_state = state["tasks"][task["id"]]
        marker_mtime_ns = path.stat().st_mtime_ns
        if task_state.get("rearmed_ready_file_mtime_ns") == marker_mtime_ns:
            continue
        marker_time = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        task_state["ignore_failures_before"] = marker_time.isoformat().replace(
            "+00:00", "Z"
        )
        task_state["rearmed_ready_file_mtime_ns"] = marker_mtime_ns
        task_state.pop("exhausted_resources", None)
        log(f"rearmed {task['id']} after ready file {path}")

    # Helpers are no longer useful once their authoritative parent task has
    # completed. Close active platform helpers and suppress marker-only retries.
    for task in tasks:
        parent_id = task.get("materialize_north_result_for")
        if not parent_id:
            continue
        required = north_materialization_required(state["tasks"].get(parent_id, {}))
        parent_state = state["tasks"].get(parent_id, {})
        task_state = state["tasks"][task["id"]]
        if required is None and task_state.get("status") == "pending":
            task_state["waiting_reason"] = (
                f"waiting for North parent task to complete: {parent_id}"
            )
        parent_completed_at = parent_state.get("completed_at")
        if (
            required is True
            and parent_completed_at
            and task_state.get("parent_completion_failure_epoch") != parent_completed_at
        ):
            exhausted = task_state.pop("exhausted_resources", None)
            task_state["rearmed_after_parent_completion"] = utc_now()
            task_state["parent_completion_failure_epoch"] = parent_completed_at
            task_state["ignore_failures_before"] = parent_completed_at
            if exhausted:
                log(
                    f"rearmed {task['id']} after {parent_id} completed on North; "
                    "discarded only pre-completion failure exhaustion"
                )
        if required is False and task_state.get("status") != "completed":
            mark_task_completed(task, task_state)
            task_state["satisfied_by_task"] = parent_id
            task_state.pop("waiting_reason", None)
            log(f"completed {task['id']}; {parent_id} did not run on North")

    for task in tasks:
        satisfied_by = task.get("satisfied_by_task")
        if not satisfied_by:
            continue
        parent_state = state["tasks"].get(satisfied_by, {})
        task_state = state["tasks"][task["id"]]
        if (
            parent_state.get("status") != "completed"
            or task_state.get("status") == "completed"
        ):
            continue
        if task_state.get("status") == "running" and task_state.get("attempts"):
            attempt = task_state["attempts"][-1]
            try:
                stop_managed_attempt(attempt)
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
        parent_id = task.get("materialize_north_result_for")
        if parent_id and north_materialization_required(
            state["tasks"].get(parent_id, {})
        ) is None:
            task_state["waiting_reason"] = (
                f"waiting for North parent task to complete: {parent_id}"
            )
            continue
        # Reconcile durable artifacts even when state was reconstructed without
        # attempt history (for example after importing an older scheduler state).
        if task.get("completion_glob"):
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
            hash_failures = readiness_hash_failures(task)
            if hash_failures:
                task_state["waiting_reason"] = (
                    "blocked by frozen source hash mismatch: "
                    + ", ".join(hash_failures)
                )
            else:
                task_state["waiting_reason"] = (
                    "blocked by required input, checkpoint, or gate"
                )
            continue
        task_state["waiting_reason"] = "waiting for an eligible resource"
        candidates = ordered_dispatch_candidates(task, snapshot)
        launch_options: list[tuple[dict[str, Any], str, bool]] = []
        for candidate in candidates:
            if robot_task_fragmentation_blocked(candidate, state, snapshot):
                continue
            if candidate["kind"] == "platform":
                credential_profile = candidate_credential_profile(candidate, snapshot)
                if credential_profile is None:
                    continue
            else:
                credential_profile = "primary"
                if not candidate_available(candidate, snapshot):
                    continue
            launch_options.append((candidate, credential_profile, False))
        if not launch_options:
            for candidate in candidates:
                credential_profile = north_queue_credential_profile(candidate, snapshot)
                if credential_profile is not None:
                    launch_options.append((candidate, credential_profile, True))
                    break
        for (
            candidate,
            credential_profile,
            persistent_north_queue_sink,
        ) in launch_options:
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
                "gpu_indices": candidate.get("gpu_indices"),
                "started_at": utc_now(),
            }
            if persistent_north_queue_sink:
                attempt["persistent_north_queue_sink"] = True
            try:
                if int(candidate["gpus"]) > 0:
                    recommendation_path, recommendation = (
                        capture_submission_recommendation(task, candidate, snapshot)
                    )
                    attempt["recommendation_audit"] = str(recommendation_path)
                    attempt["recommendation"] = {
                        "global": recommendation["global_recommendation"],
                        "task_eligible": recommendation["task_eligible_recommendation"],
                        "selected": recommendation["selected_resource"],
                        "analysis": recommendation["selection_analysis"],
                    }
                    log(
                        f"submission recommendation {task['id']} "
                        f"global={recommendation['global_recommendation']} "
                        f"eligible={recommendation['task_eligible_recommendation']} "
                        f"selected={candidate['resource']} audit={recommendation_path}"
                    )
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
                            "queue_timeout_seconds": int(
                                candidate.get("queue_timeout_seconds", 300)
                            ),
                        }
                    )
                    try:
                        capture_pi05_confirmatory_launch(task, candidate, job_id)
                        capture_pi05_confirmatory_eval_launch(task, candidate, job_id)
                        capture_pi05_mt3_eval_launch(task, candidate, job_id)
                        capture_pi05_mt12_training_launch(task, candidate, attempt)
                        capture_pi05_mt12_eval_launch(task, candidate, job_id)
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
                        capture_pi05_confirmatory_eval_launch(
                            task, candidate, f"gf1-{pid}"
                        )
                        capture_pi05_mt3_eval_launch(task, candidate, f"gf1-{pid}")
                        capture_pi05_mt12_training_launch(task, candidate, attempt)
                        capture_pi05_mt12_eval_launch(task, candidate, f"gf1-{pid}")
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
                        capture_pi05_confirmatory_eval_launch(
                            task, candidate, f"local-{pid}"
                        )
                        capture_pi05_mt3_eval_launch(task, candidate, f"local-{pid}")
                        capture_pi05_mt12_training_launch(task, candidate, attempt)
                        capture_pi05_mt12_eval_launch(task, candidate, f"local-{pid}")
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
            if persistent_north_queue_sink:
                reserve_queued_north_candidate(snapshot, candidate, credential_profile)
            else:
                reserve_dispatched_candidate(snapshot, candidate, credential_profile)
            dispatched += 1
            if dispatched >= MAX_DISPATCHES_PER_POLL:
                return
            break


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
            if item.get("aggregate") and item.get("remote"):
                program = f"""
import glob, json, os, re
pattern = {item["glob"]!r}
regex = re.compile({item["regex"]!r})
numerator = 0
denominator = 0
for path in glob.glob(pattern, recursive=True):
    with open(path, 'rb') as stream:
        stream.seek(max(0, os.path.getsize(path) - 512 * 1024))
        tail = stream.read().decode('utf-8', errors='replace').replace('\\r', '\\n')
    matches = regex.findall(tail)
    if not matches:
        continue
    match = matches[-1]
    numerator += int(match[0])
    denominator += int(match[1])
print(json.dumps({{'numerator': numerator, 'denominator': denominator}}))
"""
                try:
                    progress = json.loads(
                        ssh(GSY, f"python3 -c {shlex.quote(program)}", timeout=30)
                    )
                except Exception:
                    continue
                if progress["denominator"]:
                    total = int(item.get("total", progress["denominator"]))
                    progress_items.append(
                        f"{item['label']}={progress['numerator']}/{total}"
                    )
                continue
            files = [Path(path) for path in glob.glob(item["glob"], recursive=True)]
            if not files:
                continue
            if item.get("aggregate"):
                numerator = 0
                denominator = 0
                for path in files:
                    matches = re.findall(
                        item["regex"],
                        path.read_text(errors="replace").replace("\r", "\n"),
                    )
                    if not matches:
                        continue
                    match = matches[-1]
                    numerator += int(match[0])
                    denominator += int(match[1])
                if denominator:
                    total = int(item.get("total", denominator))
                    progress_items.append(f"{item['label']}={numerator}/{total}")
                continue
            latest = max(files, key=lambda path: path.stat().st_mtime)
            matches = re.findall(item["regex"], latest.read_text(errors="replace"))
            if matches:
                match = matches[-1]
                value = "/".join(match) if isinstance(match, tuple) else str(match)
                progress_items.append(f"{item['label']}={value}")
        if progress_items:
            task_state["runtime_progress"] = ", ".join(progress_items)
        else:
            task_state.pop("runtime_progress", None)
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
            log(
                f"stale progress warning {task['id']}: {evidence} unchanged for {stale_seconds}s"
            )
            task_state["artifact_stale_warning_at"] = utc_now()


def apply_managed_gpu_reservations(
    queue: dict[str, Any], state: dict[str, Any], snapshot: dict[str, Any]
) -> None:
    """Keep local/SSH cards reserved across transient model reload gaps."""
    reserved = {"local": 0, "gf1": 0}
    reserved_indices: dict[str, set[int]] = {"local": set(), "gf1": set()}
    tasks_by_id = {task["id"]: task for task in queue["tasks"]}
    for task_id, task_state in state["tasks"].items():
        if task_state.get("status") != "running" or not task_state.get("attempts"):
            continue
        attempt = task_state["attempts"][-1]
        resource = attempt.get("resource")
        if resource not in reserved:
            continue
        task = tasks_by_id.get(task_id, {})
        matching = [
            candidate
            for candidate in task.get("candidates", [])
            if candidate.get("resource") == resource
            and candidate.get("kind") == attempt.get("kind")
        ]
        gpus = attempt.get("gpus")
        if gpus is None:
            if matching:
                gpus = matching[0].get("gpus", 0)
        reserved[resource] += int(gpus or 0)
        gpu_indices = attempt.get("gpu_indices")
        if gpu_indices is None and matching:
            gpu_indices = matching[0].get("gpu_indices")
        if gpu_indices is not None:
            reserved_indices[resource].update(int(index) for index in gpu_indices)

    for resource, managed_gpus in reserved.items():
        resource_state = snapshot["resources"][resource]
        count = int(resource_state["count"])
        observed_free = int(resource_state["free_count"])
        observed_busy = count - observed_free
        effective_busy = min(count, max(observed_busy, managed_gpus))
        resource_state["observed_free_count"] = observed_free
        resource_state["managed_reserved_gpus"] = managed_gpus
        resource_state["managed_reserved_indices"] = sorted(reserved_indices[resource])
        resource_state["free_count"] = count - effective_busy


def poll_once(queue: dict[str, Any], state: dict[str, Any]) -> None:
    refresh_pi05_actionfix_intervention_gate()
    refresh_pi05_actionfix_midpoint_report()
    refresh_pi05_step40000_safety_report()
    refresh_pi05_mt12_shared_finalizers()
    refresh_pi05_mt1_replication_checkpoint_audits()
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
    refresh_pi05_mt12_training_provenance(queue, state)
    refresh_pi05_corrected_a0_gate()
    refresh_pi05_exact_a0_gate()
    refresh_method_matrix()
    snapshot["scheduler_tasks"] = state["tasks"]
    inventory = {"completed": 0, "running": 0, "pending": 0, "disabled": 0}
    for task in queue["tasks"]:
        state_status = state["tasks"][task["id"]].get("status", "pending")
        status = (
            "disabled"
            if not task.get("enabled", True)
            and not task.get("disabled_by_retired_resource")
            else state_status
        )
        inventory[status] = inventory.get(status, 0) + 1
    inventory["total"] = len(queue["tasks"])
    snapshot["queue_inventory"] = inventory
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
        f"sh_submit={'enabled' if resources['robot-task'].get('submission_enabled', True) else 'disabled'} "
        f"east={resources['Robot-East-H20']['active_gpus_all_users']}/8 queued={len(resources['Robot-East-H20']['queueing_all_users'])} "
        "gf1=retired "
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
        raise SystemExit(
            "another resource-aware scheduler instance already holds the lock"
        )
    lock_stream.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_stream.flush()
    if not os.environ.get("VOLC_AK") or not os.environ.get("VOLC_SK"):
        raise SystemExit("VOLC_AK/VOLC_SK are required")
    queue = json.loads(QUEUE_PATH.read_text())
    add_pi05_shared_eval_attach_tasks(queue)
    add_pi05_mt_eval_attach_tasks(queue)
    add_pi05_mt1_replication_eval_attach_tasks(queue)
    add_pi05_mt1_replication_north_overflow(queue)
    add_pi05_mt1_8g_optimization_probes(queue)
    add_pi05_p1_north_failover_tasks(queue)
    add_pi05_r1_recurrence_aligned_tasks(queue)
    add_pi05_r4_outcome_collection_tasks(queue)
    add_pi05_r2_adaptive_execution_tasks(queue)
    add_pi05_north_eval_attach_tasks(queue)
    add_pi05_step40000_safety_probes(queue)
    add_pi05_mt3_tracker_tasks(queue)
    add_pi05_mt4_replication_tasks(queue)
    add_pi05_mt5_tasks(queue)
    add_pi05_mt6_scope_task(queue)
    add_pi05_mt6_efficiency_task(queue)
    add_pi05_mt6_train_memory_task(queue)
    add_pi05_mt3_eval_attach_tasks(queue)
    apply_frozen_source_readiness(queue)
    validate_queue(queue)
    apply_permanent_resource_policy(queue)
    state = load_state(queue)
    log(f"scheduler start interval={args.interval}s once={args.once}")
    while True:
        try:
            queue = json.loads(QUEUE_PATH.read_text())
            add_pi05_shared_eval_attach_tasks(queue)
            add_pi05_mt_eval_attach_tasks(queue)
            add_pi05_mt1_replication_eval_attach_tasks(queue)
            add_pi05_mt1_replication_north_overflow(queue)
            add_pi05_mt1_8g_optimization_probes(queue)
            add_pi05_p1_north_failover_tasks(queue)
            add_pi05_r1_recurrence_aligned_tasks(queue)
            add_pi05_r4_outcome_collection_tasks(queue)
            add_pi05_r2_adaptive_execution_tasks(queue)
            add_pi05_north_eval_attach_tasks(queue)
            add_pi05_step40000_safety_probes(queue)
            add_pi05_mt3_tracker_tasks(queue)
            add_pi05_mt4_replication_tasks(queue)
            add_pi05_mt5_tasks(queue)
            add_pi05_mt6_scope_task(queue)
            add_pi05_mt6_efficiency_task(queue)
            add_pi05_mt6_train_memory_task(queue)
            add_pi05_mt3_eval_attach_tasks(queue)
            apply_frozen_source_readiness(queue)
            validate_queue(queue)
            apply_permanent_resource_policy(queue)
            state = load_state(queue)
            poll_once(queue, state)
        except Exception as exc:
            log(f"poll error {type(exc).__name__}: {exc}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
