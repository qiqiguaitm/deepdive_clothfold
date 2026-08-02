#!/usr/bin/env python3
"""Poll Volc ML Platform jobs and advance the current RoboTwin/LIBERO queue.

This is intentionally stateful and idempotent: every submitted follow-up task is
recorded in logs/volc_auto_progress_state.json before the monitor continues.
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


REPO = Path(__file__).resolve().parents[3]
LOG_DIR = REPO / "logs"
STATE_PATH = LOG_DIR / "volc_auto_progress_state.json"
LOG_PATH = LOG_DIR / "volc_auto_progress.log"
SSH = ["ssh", "-p", "16370", "root@124.174.16.237"]
NORTH_REPO = "/vePFS-North-E/vis_robot/workspace/deepdive_kai0"


@dataclass(frozen=True)
class Job:
    id: str
    region: str
    kind: str
    group: str
    result_root: str | None = None


INITIAL_JOBS = [
    Job("t-20260728223558-88mlc", "cn-shanghai", "libero_eval", "n2_seed2"),
    Job("t-20260728223603-4tph8", "cn-shanghai", "libero_eval", "n2_seed2"),
    Job("t-20260728223608-tx695", "cn-shanghai", "libero_eval", "n2_seed2"),
    Job("t-20260728223613-cm6wd", "cn-shanghai", "libero_eval", "n2_seed2"),
    Job("t-20260728223619-pvjnv", "cn-beijing", "libero_eval", "resid_noTsS4"),
    Job("t-20260728223624-9vqrf", "cn-beijing", "libero_eval", "resid_noTsS4"),
    Job("t-20260728223631-bvcjt", "cn-beijing", "libero_eval", "resid_noTsS4"),
    Job("t-20260728223637-gmnj2", "cn-beijing", "libero_eval", "resid_noTsS4"),
    Job(
        "t-20260728224221-5q54w",
        "cn-beijing",
        "robotwin_eval",
        "gate_balanced",
        "lmvla/lawam/results/eval_runs/robotwin/rt_lmwm_dual2q_balanced_gate",
    ),
    Job(
        "t-20260728225316-mchrf",
        "cn-beijing",
        "robotwin_eval",
        "a0_replan1_probe",
        "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a0_replan1_probe",
    ),
    Job("t-20260729074059-22gq7", "cn-beijing", "train", "a0_official"),
]

THROUGHPUT_PROBE_JOBS = [
    Job("t-20260730131155-62dqf", "cn-beijing", "throughput_probe", "all6_b32a1_8gpu"),
    Job("t-20260730131618-cgntr", "cn-shanghai", "throughput_probe", "a3_diagopt"),
    Job("t-20260730131623-gwv4n", "cn-shanghai", "throughput_probe", "a3_fusedvision"),
    Job("t-20260730133037-4cg54", "cn-beijing", "throughput_probe", "all6_fusedadam_1gpu"),
]

VALIDATION_JOBS = [
    Job("t-20260730140251-fqjjl", "cn-beijing", "validation_probe", "so400m_online_hint"),
    Job("t-20260730140537-pr7bc", "cn-beijing", "validation_probe", "so400m_online_hint_diag"),
    Job("t-20260730140753-z2d7d", "cn-beijing", "validation_probe", "so400m_online_hint_batchdiag"),
    Job("t-20260730141040-v7m8p", "cn-beijing", "validation_probe", "so400m_online_hint_final"),
    Job("t-20260730151342-6tvjv", "cn-beijing", "prep", "a2_residual_recovery"),
]

THROUGHPUT_LOG_GLOBS = {
    "all6_b32a1_8gpu": "lmvla/lawam/logs/volc_robotwin/all6_v2_b32a1_probe_*.log",
    "a3_diagopt": "lmvla/lawam/logs/volc_robotwin/pi05_a3_diagopt_probe_*.log",
    "a3_fusedvision": "lmvla/lawam/logs/volc_robotwin/pi05_a3_fusedvision_probe_*.log",
    "all6_fusedadam_1gpu": "lmvla/lawam/logs/volc_robotwin/all6_v2_fusedadam_smoke_*.log",
}

GROUP_EXPECTED = {
    "n2_seed2": 4,
    "resid_noTsS4": 4,
}

FOLLOWUP_YAMLS = {
    "a1_official_train": "train_scripts/kai/volc/pi05_robotwin_a1_prefix_official_bj.yaml",
    "a2_so400m_hint_prep": "train_scripts/kai/volc/robotwin_so400m_hint_prep_bj_4h20.yaml",
    "a2_official_train": "train_scripts/kai/volc/pi05_robotwin_a2_prefix_official_bj.yaml",
    "a2_residual_official_train": "train_scripts/kai/volc/pi05_robotwin_a2_residual_prefix_official_bj.yaml",
    "a3_live_residual_official_train": "train_scripts/kai/volc/pi05_robotwin_a3_live_residual_prefix_official_bj.yaml",
    "a0_official_eval": "train_scripts/kai/volc/pi05_robotwin_eval_a0_official_x4_bj.yaml",
    "a1_official_eval": "train_scripts/kai/volc/pi05_robotwin_eval_a1_prefix_official_x4_bj.yaml",
    "hint_official_eval": "train_scripts/kai/volc/pi05_robotwin_eval_hint_official_x4_bj.yaml",
    "all6_v2_prep": "train_scripts/kai/volc/robotwin_all6_v2_prep_bj_1h20.yaml",
}

ALL6_MATRIX = [
    # Shanghai has one dedicated 8-H20 node. Start the final method there first.
    ("all6_train_combo_seed2026", "combo", 2026, "cn-shanghai"),
    ("all6_train_nowm_seed2026", "nowm", 2026, "cn-beijing"),
    ("all6_train_local_seed2026", "local", 2026, "cn-beijing"),
    ("all6_train_absolute_seed2026", "absolute", 2026, "cn-beijing"),
    ("all6_train_residual_seed2026", "residual", 2026, "cn-beijing"),
    ("all6_train_isolation_seed2026", "isolation", 2026, "cn-beijing"),
    # Training-seed replication for the direct baseline and final method.
    ("all6_train_local_seed2027", "local", 2027, "cn-beijing"),
    ("all6_train_combo_seed2027", "combo", 2027, "cn-beijing"),
]

TERMINAL_STATES = {"Completed", "Success", "Failed", "Stopped"}
SUCCESS_STATES = {"Completed", "Success"}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"jobs": {}, "actions": {}, "summaries": {}}


def save_state(state: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True))
    tmp.replace(STATE_PATH)


def reconcile_archived_attempts(state: dict[str, Any]) -> None:
    """Keep superseded attempts from appearing active in the state snapshot."""
    for attempts in state.get("failed_attempts", {}).values():
        for attempt in attempts:
            task_id = attempt.get("task_id")
            terminal_state = attempt.get("state")
            if not task_id or terminal_state not in TERMINAL_STATES:
                continue
            job = state.get("jobs", {}).get(task_id)
            if job is not None:
                job["state"] = terminal_state
                job["archived"] = True


def service(region: str) -> Service:
    si = ServiceInfo(
        "open.volcengineapi.com",
        {"Accept": "application/json"},
        Credentials(os.environ["VOLC_AK"], os.environ["VOLC_SK"], "ml_platform", region),
        10,
        10,
    )
    return Service(si, {"GetJob": ApiInfo("POST", "/", {"Action": "GetJob", "Version": "2024-07-01"}, {}, {})})


_SVC: dict[str, Service] = {}


def get_job(job: Job) -> dict[str, Any]:
    svc = _SVC.setdefault(job.region, service(job.region))
    raw = svc.json("GetJob", {}, json.dumps({"Id": job.id}).encode())
    result = json.loads(raw).get("Result", {})
    status = result.get("Status") or {}
    return {
        "id": job.id,
        "region": job.region,
        "name": result.get("Name"),
        "state": status.get("State") or result.get("State"),
        "message": status.get("Message") or "",
        "create_time": result.get("CreateTime"),
        "update_time": result.get("UpdateTime"),
    }


def run(cmd: list[str], *, timeout: int = 300) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def remote(cmd: str, *, timeout: int = 300) -> str:
    return run([*SSH, cmd], timeout=timeout)


def remote_exists(path: str) -> bool:
    try:
        out = remote(f"test -e {sh_quote(path)} && echo yes || echo no", timeout=30)
        return out.strip() == "yes"
    except Exception:
        return False


def sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def submit_yaml(
    yaml_path: str,
    *,
    task_name: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> str:
    cmd = [
        sys.executable,
        str(REPO / "train_scripts/kai/volc/submit_yaml.py"),
        str(REPO / yaml_path),
    ]
    if task_name:
        cmd.extend(["--task-name", task_name])
    for name, value in (env_overrides or {}).items():
        cmd.extend(["--set-env", f"{name}={value}"])
    env = os.environ.copy()
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(key, None)
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env, timeout=120)
    match = re.search(r"task_id=(t-[0-9a-z-]+)", out)
    if not match:
        raise RuntimeError(f"Could not parse task id from submit output:\n{out[-2000:]}")
    return match.group(1)


def parse_robotwin(root: str) -> dict[str, Any]:
    code = r"""
import glob,json,os,statistics as st,sys
root=sys.argv[1]
per_seed={}
for f in sorted(glob.glob(root + "/seed*/**/tasks/*/summary.json", recursive=True)):
    try: d=json.load(open(f))
    except Exception: continue
    if not isinstance(d, dict) or "success_rate" not in d: continue
    seed=f.split("/seed")[1].split("/")[0]
    per_seed.setdefault(seed,{})[d.get("task_name", os.path.basename(os.path.dirname(f)))] = float(d["success_rate"])
tasks=sorted({t for row in per_seed.values() for t in row})
task_mean={t: st.mean([per_seed[s][t] for s in per_seed if t in per_seed[s]]) for t in tasks}
allv=[per_seed[s][t] for s in per_seed for t in per_seed[s]]
print(json.dumps({"root": root, "seeds": per_seed, "task_mean": task_mean, "aggregate": (st.mean(allv) if allv else None), "n": len(allv)}, ensure_ascii=False))
"""
    out = remote(f"cd {sh_quote(NORTH_REPO)} && python -c {sh_quote(code)} {sh_quote(root)}", timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def parse_libero(root: str) -> dict[str, Any]:
    code = r"""
import glob,json,os,statistics as st,sys
root=sys.argv[1]
rows=[]
for f in sorted(glob.glob(root + "/**/summary.json", recursive=True)):
    try: d=json.load(open(f))
    except Exception: continue
    sr=d.get("total_success_rate", d.get("success_rate", d.get("aggregate_SR")))
    if sr is None: continue
    rows.append({"file": f, "suite": d.get("suite") or d.get("task_suite_name") or d.get("benchmark_name") or os.path.basename(os.path.dirname(f)), "sr": float(sr)})
by={}
for r in rows: by.setdefault(r["suite"], []).append(r["sr"])
print(json.dumps({"root": root, "suite_mean": {k: st.mean(v) for k,v in by.items()}, "aggregate": (st.mean([r["sr"] for r in rows]) if rows else None), "n": len(rows)}, ensure_ascii=False))
"""
    out = remote(f"cd {sh_quote(NORTH_REPO)} && python -c {sh_quote(code)} {sh_quote(root)}", timeout=120)
    return json.loads(out.strip().splitlines()[-1])


def parse_libero_local(root: str) -> dict[str, Any]:
    import glob
    import statistics as _st

    rows = []
    for f in sorted(glob.glob(str(REPO / root) + "/**/summary.json", recursive=True)):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        sr = d.get("total_success_rate", d.get("success_rate", d.get("aggregate_SR")))
        if sr is None:
            continue
        suite = d.get("suite") or d.get("task_suite_name") or d.get("benchmark_name") or Path(f).parent.name
        rows.append({"file": f, "suite": suite, "sr": float(sr)})
    by: dict[str, list[float]] = {}
    for row in rows:
        by.setdefault(row["suite"], []).append(row["sr"])
    return {
        "root": str(REPO / root),
        "suite_mean": {k: _st.mean(v) for k, v in by.items()},
        "aggregate": (_st.mean([r["sr"] for r in rows]) if rows else None),
        "n": len(rows),
        "note": "Parsed locally from Shanghai /vePFS workspace.",
    }


def parse_throughput_probe(job: Job) -> dict[str, Any]:
    log_glob = THROUGHPUT_LOG_GLOBS[job.group]
    if job.region == "cn-beijing":
        text = remote(
            f"cd {sh_quote(NORTH_REPO)} && "
            f"f=$(ls -1t {log_glob} 2>/dev/null | head -1); "
            f"test -n \"$f\" && tr '\\r' '\\n' < \"$f\"",
            timeout=120,
        )
    else:
        paths = sorted(REPO.glob(log_glob), key=lambda path: path.stat().st_mtime, reverse=True)
        if not paths:
            raise FileNotFoundError(f"No throughput log matching {log_glob}")
        text = paths[0].read_text(errors="replace").replace("\r", "\n")

    cuda_times = [float(value) for value in re.findall(r"model_times=([0-9.]+)", text)]
    tqdm_times = [float(value) for value in re.findall(r"([0-9.]+)s/it", text)]
    values = cuda_times[-5:] if cuda_times else tqdm_times[-20:]
    if not values:
        raise RuntimeError(f"No throughput samples found for {job.group}")
    return {
        "group": job.group,
        "task_id": job.id,
        "stable_seconds_per_step": st.median(values),
        "sample_count": len(values),
        "metric": "cuda_event" if cuda_times else "tqdm_wall",
    }


def maybe_group_complete(state: dict[str, Any], jobs: list[Job], group: str) -> bool:
    ids = [j.id for j in jobs if j.group == group]
    return len(ids) >= GROUP_EXPECTED[group] and all(state["jobs"].get(jid, {}).get("state") in SUCCESS_STATES for jid in ids)


def all_jobs(state: dict[str, Any]) -> list[Job]:
    jobs = [*INITIAL_JOBS, *THROUGHPUT_PROBE_JOBS, *VALIDATION_JOBS]
    for key, info in state.get("actions", {}).items():
        jid = info.get("task_id")
        if not jid:
            continue
        if key == "a0_official_train":
            jobs.append(Job(jid, "cn-beijing", "train", "a0_official"))
        elif key == "a1_official_train":
            jobs.append(Job(jid, "cn-beijing", "train", "a1_official"))
        elif key == "a2_so400m_hint_prep":
            jobs.append(Job(jid, "cn-beijing", "prep", "a2_so400m_hint_prep"))
        elif key == "a2_official_train":
            jobs.append(Job(jid, "cn-beijing", "train", "a2_official"))
        elif key == "a2_residual_official_train":
            jobs.append(Job(jid, "cn-beijing", "train", "a2_residual_official"))
        elif key == "a3_live_residual_official_train":
            jobs.append(Job(jid, info.get("region", "cn-beijing"), "train", "a3_live_residual_official"))
        elif key == "a3_live_residual_official_eval":
            jobs.append(
                Job(
                    jid,
                    info.get("region", "cn-shanghai"),
                    "robotwin_eval",
                    "a3_live_residual_official_eval",
                    "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a3_live_residual_official",
                )
            )
        elif key == "a0_official_eval":
            jobs.append(Job(jid, "cn-beijing", "robotwin_eval", "a0_official_eval", "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a0_official"))
        elif key == "a1_official_eval":
            jobs.append(Job(jid, "cn-beijing", "robotwin_eval", "a1_official_eval", "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a1_prefix_official"))
        elif key == "a2_official_eval":
            jobs.append(Job(jid, "cn-beijing", "robotwin_eval", "a2_official_eval", "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_prefix_official"))
        elif key == "a2_residual_official_eval":
            jobs.append(Job(jid, "cn-beijing", "robotwin_eval", "a2_residual_official_eval", "lmvla/lawam/results/eval_runs/robotwin/pi05_rt_a2_residual_prefix_official"))
        elif key == "all6_v2_prep":
            jobs.append(Job(jid, "cn-beijing", "prep", "all6_v2_prep"))
        elif key.startswith("all6_train_"):
            jobs.append(Job(jid, info.get("region", "cn-beijing"), "train", key))
    return jobs


def ensure_all6_local_assets() -> None:
    local_ms = REPO / "lmvla/lmwm/data/robotwin_milestone_all6_v2"
    local_ds = REPO / "lmvla/lawam/dataset/robotwin2_lmwm_all6_v2_v30"
    required = [
        local_ms / "READY",
        local_ms / "pairs.npz",
        local_ms / "target_compact.npz",
        local_ms / "lmwm.pt",
        local_ds / "meta/info.json",
        local_ds / "meta/episodes/chunk-000/file-000.parquet",
    ]
    if all(path.exists() for path in required):
        return

    local_ms.mkdir(parents=True, exist_ok=True)
    remote_ms = (
        f"root@124.174.16.237:{NORTH_REPO}/"
        "lmvla/lmwm/data/robotwin_milestone_all6_v2"
    )
    for name in [
        "READY",
        "pairs.npz",
        "target_compact.npz",
        "lmwm.pt",
        "canon_id.json",
        "selection_manifest.json",
    ]:
        run(
            [
                "scp",
                "-P",
                "16370",
                f"{remote_ms}/{name}",
                str(local_ms / name),
            ],
            timeout=7200,
        )
    run(
        [
            sys.executable,
            str(REPO / "lmvla/lmwm/scripts/build_robotwin_v30_subset.py"),
            "--src",
            str(REPO / "lmvla/lawam/dataset/robotwin2.0"),
            "--pairs",
            str(local_ms / "pairs.npz"),
            "--out",
            str(local_ds),
        ],
        timeout=1800,
    )
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise RuntimeError(f"all6 local asset sync incomplete: {missing}")
    log("all6_v2 assets synced to Shanghai and local v3 dataset built")


def poll_once(state: dict[str, Any]) -> None:
    jobs = all_jobs(state)
    for job in jobs:
        cached = state["jobs"].get(job.id, {})
        prev = cached.get("state")
        if prev in TERMINAL_STATES:
            info = cached
        else:
            try:
                info = get_job(job)
            except Exception as exc:
                log(f"status error {job.id}: {type(exc).__name__}: {exc}")
                continue
            state["jobs"][job.id] = {**info, "kind": job.kind, "group": job.group}
            if info["state"] != prev:
                log(f"{job.id} {info['name']} {prev} -> {info['state']} {info['message']}")

        if info["state"] in SUCCESS_STATES and job.kind == "robotwin_eval" and job.result_root:
            key = f"summary:{job.group}"
            if key not in state["summaries"]:
                summary = parse_robotwin(f"{NORTH_REPO}/{job.result_root}")
                state["summaries"][key] = summary
                log(f"{job.group} robotwin aggregate={summary.get('aggregate')} n={summary.get('n')}")
        if info["state"] in SUCCESS_STATES and job.kind == "throughput_probe":
            key = f"throughput:{job.group}"
            if key not in state["summaries"]:
                summary = parse_throughput_probe(job)
                state["summaries"][key] = summary
                log(
                    f"{job.group} throughput={summary['stable_seconds_per_step']:.4f}s "
                    f"metric={summary['metric']} n={summary['sample_count']}"
                )

    for group, root in {
        "n2_seed2": "lmvla/lawam/results/eval_runs/libero/n2_seed2_resid_noTs",
        "resid_noTsS4": "lmvla/lawam/results/eval_runs/libero/resid_noTsS4",
    }.items():
        key = f"summary:{group}"
        existing_summary = state["summaries"].get(key)
        summary_is_empty = existing_summary is not None and not existing_summary.get("n")
        if (key not in state["summaries"] or summary_is_empty) and maybe_group_complete(state, jobs, group):
            if group == "n2_seed2":
                summary = parse_libero_local(root)
            else:
                summary = parse_libero(f"{NORTH_REPO}/{root}")
            state["summaries"][key] = summary
            log(f"{group} libero aggregate={summary.get('aggregate')} n={summary.get('n')}")

    a0_norm = f"{NORTH_REPO}/kai0/assets/pi05_robotwin_a0_official_bj/robotwin2.0/norm_stats.json"
    if "a1_official_train" not in state["actions"] and remote_exists(a0_norm):
        jid = submit_yaml(FOLLOWUP_YAMLS["a1_official_train"])
        state["actions"]["a1_official_train"] = {"task_id": jid, "submitted_at": now(), "reason": "official A0 norm exists"}
        log(f"submitted a1_official_train {jid}")

    a2_hint = f"{NORTH_REPO}/lmvla/lmwm/data/pi05_hint/robotwin_so400m/hint.npz"
    a2_res_hint = f"{NORTH_REPO}/lmvla/lmwm/data/pi05_hint/robotwin_so400m_residual/hint.npz"
    if "a2_official_train" not in state["actions"] and remote_exists(a2_hint):
        jid = submit_yaml(FOLLOWUP_YAMLS["a2_official_train"])
        state["actions"]["a2_official_train"] = {
            "task_id": jid,
            "submitted_at": now(),
            "reason": "RoboTwin So400m A2 hint exists",
        }
        log(f"submitted a2_official_train {jid}")
    if "a2_residual_official_train" not in state["actions"] and remote_exists(a2_res_hint):
        jid = submit_yaml(FOLLOWUP_YAMLS["a2_residual_official_train"])
        state["actions"]["a2_residual_official_train"] = {
            "task_id": jid,
            "submitted_at": now(),
            "reason": "RoboTwin So400m residual hint exists",
        }
        log(f"submitted a2_residual_official_train {jid}")

    all6_ready = f"{NORTH_REPO}/lmvla/lmwm/data/robotwin_milestone_all6_v2/READY"
    prep_action = state["actions"].get("all6_v2_prep")
    prep_state = (
        state["jobs"].get(prep_action["task_id"], {}).get("state")
        if prep_action
        else None
    )
    should_submit_prep = (
        not remote_exists(all6_ready)
        and (prep_action is None or prep_state in {"Failed", "Stopped"})
    )
    if should_submit_prep:
        if prep_action is not None:
            state.setdefault("failed_attempts", {}).setdefault("all6_v2_prep", []).append(
                {**prep_action, "state": prep_state}
            )
        jid = submit_yaml(FOLLOWUP_YAMLS["all6_v2_prep"])
        state["actions"]["all6_v2_prep"] = {
            "task_id": jid,
            "submitted_at": now(),
            "reason": "build exact official six-task pairs, predictor, compact targets, and v3 dataset",
        }
        save_state(state)
        log(f"submitted all6_v2_prep {jid}")

    if remote_exists(all6_ready):
        ensure_all6_local_assets()
        for action, variant, seed, region in ALL6_MATRIX:
            if action in state["actions"]:
                continue
            template = (
                "train_scripts/kai/volc/robotwin_all6_v2_train_east_8h20.yaml"
                if region == "cn-shanghai"
                else "train_scripts/kai/volc/robotwin_all6_v2_train_bj_8h20.yaml"
            )
            task_name = f"robotwin-all6-v2-{variant}-s{seed}-{'east' if region == 'cn-shanghai' else 'bj'}"
            jid = submit_yaml(
                template,
                task_name=task_name,
                env_overrides={
                    "ROBOTWIN_ALL6_VARIANT": variant,
                    "ROBOTWIN_ALL6_SEED": str(seed),
                },
            )
            state["actions"][action] = {
                "task_id": jid,
                "submitted_at": now(),
                "reason": "exact all6 v2 method matrix",
                "variant": variant,
                "seed": seed,
                "region": region,
            }
            save_state(state)
            log(f"submitted {action} {jid} region={region}")

    a0_info = state["actions"].get("a0_official_train")
    a0_train = (
        Job(a0_info["task_id"], "cn-beijing", "train", "a0_official")
        if a0_info
        else next(j for j in jobs if j.group == "a0_official" and j.kind == "train")
    )
    if (
        state["jobs"].get(a0_train.id, {}).get("state") in SUCCESS_STATES
        and "a0_official_eval" not in state["actions"]
    ):
        jid = submit_yaml(FOLLOWUP_YAMLS["a0_official_eval"])
        state["actions"]["a0_official_eval"] = {"task_id": jid, "submitted_at": now(), "reason": "official A0 train completed"}
        log(f"submitted a0_official_eval {jid}")

    a1_info = state["actions"].get("a1_official_train")
    if a1_info:
        a1_state = state["jobs"].get(a1_info["task_id"], {}).get("state")
        if a1_state in SUCCESS_STATES and "a1_official_eval" not in state["actions"]:
            jid = submit_yaml(FOLLOWUP_YAMLS["a1_official_eval"])
            state["actions"]["a1_official_eval"] = {"task_id": jid, "submitted_at": now(), "reason": "official A1 train completed"}
            log(f"submitted a1_official_eval {jid}")

    hint_eval_specs = [
        (
            "a2_official_train",
            "a2_official_eval",
            "pi05-robotwin-a2-prefix-official-eval-x4",
            "pi05_robotwin_a2_prefix_official_eval_bj",
            f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a2_prefix_official_bj/pi05_robotwin_a2_prefix_official",
            "pi05_rt_a2_prefix_official",
            "0",
            "7900",
        ),
        (
            "a2_residual_official_train",
            "a2_residual_official_eval",
            "pi05-robotwin-a2-residual-official-eval-x4",
            "pi05_robotwin_a2_residual_prefix_official_eval_bj",
            f"{NORTH_REPO}/kai0/checkpoints/pi05_robotwin_a2_residual_prefix_official_bj/pi05_robotwin_a2_residual_prefix_official",
            "pi05_rt_a2_residual_prefix_official",
            "1",
            "8100",
        ),
    ]
    for train_key, eval_key, task_name, config, ckpt_root, result_name, residual, port_base in hint_eval_specs:
        train_info = state["actions"].get(train_key)
        if train_info is None or eval_key in state["actions"]:
            continue
        train_state = state["jobs"].get(train_info["task_id"], {}).get("state")
        if train_state not in SUCCESS_STATES:
            continue
        jid = submit_yaml(
            FOLLOWUP_YAMLS["hint_official_eval"],
            task_name=task_name,
            env_overrides={
                "ROBOTWIN_EVAL_CONFIG": config,
                "ROBOTWIN_EVAL_CKPT_ROOT": ckpt_root,
                "ROBOTWIN_EVAL_RESULT_NAME": result_name,
                "ROBOTWIN_HINT_ENCODER": "so400m",
                "EVAL_HINT_RESIDUAL": residual,
                "ROBOTWIN_EVAL_PORT_BASE": port_base,
            },
        )
        state["actions"][eval_key] = {
            "task_id": jid,
            "submitted_at": now(),
            "reason": f"{train_key} completed",
        }
        save_state(state)
        log(f"submitted {eval_key} {jid}")


def main(interval_sec: int = 180, *, once: bool = False) -> None:
    if "VOLC_AK" not in os.environ or "VOLC_SK" not in os.environ:
        raise SystemExit("VOLC_AK/VOLC_SK must be set")
    state = load_state()
    reconcile_archived_attempts(state)
    save_state(state)
    log(f"monitor start interval={interval_sec}s once={once} state={STATE_PATH}")
    while True:
        try:
            poll_once(state)
            save_state(state)
        except Exception as exc:
            log(f"poll fatal-but-continuing {type(exc).__name__}: {exc}")
            save_state(state)
        if once:
            return
        time.sleep(interval_sec)


if __name__ == "__main__":
    once = "--once" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--once"]
    interval = int(args[0]) if args else int(os.environ.get("VOLC_MONITOR_INTERVAL_SEC", "180"))
    main(interval, once=once)
