#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


REPO = Path("/vePFS/tim/workspace/deepdive_kai0")
QUEUE_ID = "q-20260516104642-khch9"
OWNER = "trn:iam::2113249311:user/suiyang.guo"
REMOTE_RUN = (
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lawam/"
    "results/Checkpoints/robotwin/20260730_152020+robotwin_all6_v2_combo_seed2026"
)
YAML = REPO / "train_scripts/kai/volc/robotwin_all6_v2_eval_bj_8h20.yaml"
LOG = REPO / "lmvla/lawam/logs/local_rteval/all6_combo_s2026_bj_quota_watcher.log"
GPU_BY_FLAVOR = {
    "ml.hpcpni3ln.45xlarge": 8,
    "ml.pni3ln.17xlarge": 4,
    "ml.pni3ln.5xlarge": 1,
}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"[{stamp}] {message}\n")


def service() -> Service:
    info = ServiceInfo(
        "open.volcengineapi.com",
        {"Accept": "application/json"},
        Credentials(os.environ["VOLC_AK"], os.environ["VOLC_SK"], "ml_platform", "cn-beijing"),
        10,
        10,
    )
    return Service(
        info,
        {"ListJobs": ApiInfo("POST", "/", {"Action": "ListJobs", "Version": "2024-07-01"}, {}, {})},
    )


def active_owned_gpus(svc: Service) -> tuple[int, list[str]]:
    total = 0
    rows: list[str] = []
    for state in ("Running", "Deploying", "Queueing"):
        body = {"ResourceQueueId": QUEUE_ID, "PageSize": 100, "State": state}
        payload = json.loads(svc.json("ListJobs", {}, json.dumps(body).encode()))["Result"]
        for job in payload.get("Items", payload.get("List", [])):
            if job.get("CreatedBy") != OWNER:
                continue
            job_gpus = 0
            for role in job.get("ResourceConfig", {}).get("Roles", []):
                flavor = role.get("Resource", {}).get("InstanceTypeId", "")
                replicas = int(role.get("Replicas", 1))
                if flavor not in GPU_BY_FLAVOR:
                    raise RuntimeError(f"Unknown Beijing flavor {flavor!r} for {job.get('Id')}")
                job_gpus += GPU_BY_FLAVOR[flavor] * replicas
            total += job_gpus
            rows.append(f"{job.get('Id')}:{job_gpus}:{state}:{job.get('Name')}")
    return total, rows


def remote_ready() -> bool:
    command = (
        f"test -s '{REMOTE_RUN}/final_model/pytorch_model.pt' "
        f"&& test -s '{REMOTE_RUN}/config.yaml' "
        f"&& test -s '{REMOTE_RUN}/dataset_statistics.json'"
    )
    return subprocess.run(
        ["ssh", "-p", "16370", "-o", "BatchMode=yes", "root@124.174.16.237", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def submit() -> str:
    command = [
        str(REPO / "kai0/.venv/bin/python"),
        str(REPO / "train_scripts/kai/volc/submit_yaml.py"),
        str(YAML),
        "--task-name",
        "robotwin-all6-v2-combo-s2026-eval-bj",
        "--set-env",
        "ALL6_EVAL_VARIANT=combo",
        "--set-env",
        "ALL6_EVAL_RUN_BASENAME=20260730_152020+robotwin_all6_v2_combo_seed2026",
    ]
    env = os.environ.copy()
    env["VOLC_REGION"] = "cn-beijing"
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, env=env)
    for line in output.splitlines():
        if "SUCCESS task_id=" in line:
            return line.rsplit("=", 1)[-1].strip()
    raise RuntimeError(f"Could not parse task id:\n{output[-2000:]}")


def main() -> None:
    svc = service()
    while True:
        if not remote_ready():
            log("waiting for combo checkpoint sync")
            time.sleep(120)
            continue
        used, rows = active_owned_gpus(svc)
        log(f"owned_active_gpus={used}; " + " | ".join(rows))
        if used <= 12:
            task_id = submit()
            log(f"submitted combo eval task_id={task_id}; projected_gpus={used + 8}")
            return
        time.sleep(180)


if __name__ == "__main__":
    main()
