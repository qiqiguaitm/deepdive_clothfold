#!/usr/bin/env python3
"""Benchmark one pi0.5 policy in-process and through its WebSocket server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def summarize(values: list[float], seed: int) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(10_000, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    return {
        "n": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_95ci_ms": [float(x) for x in np.percentile(bootstrap_means, [2.5, 97.5])],
    }


def gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_observation(
    hint_dim: int, *, transition_task_id: int | None = None, history_steps: int = 0
) -> dict:
    # RoboTwin's OpenPI bridge sends client images in CHW layout.
    image = np.zeros((3, 480, 640), dtype=np.uint8)
    observation = {
        "images": {
            "cam_high": image,
            "cam_left_wrist": image.copy(),
            "cam_right_wrist": image.copy(),
        },
        "state": np.zeros(14, dtype=np.float32),
        "prompt": "stack two blocks",
    }
    if hint_dim:
        observation["lmwm_hint"] = np.zeros((1, hint_dim), dtype=np.float32)
    if transition_task_id is not None:
        observation["lmwm_transition_task"] = np.asarray(
            transition_task_id, dtype=np.int32
        )
        observation["lmwm_transition_mask"] = np.asarray(True)
    if history_steps:
        observation["lmwm_transition_history_images"] = np.zeros(
            (history_steps, 3, 480, 640), dtype=np.uint8
        )
        observation["lmwm_transition_history_state"] = np.zeros(
            (history_steps, 14), dtype=np.float32
        )
    return observation


def parameter_count(model) -> int:
    state = nnx.state(model, nnx.Param)
    return int(sum(np.prod(value.shape) for value in jax.tree.leaves(state)))


def lowered_cost_analysis(policy, observation: dict) -> dict[str, float]:
    inputs = policy._input_transform(jax.tree.map(lambda value: value, observation))
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[np.newaxis, ...], inputs)
    from openpi.models import model as model_module
    from openpi.shared import array_typing

    model_observation = model_module.Observation.from_dict(inputs)
    with array_typing.disable_typechecking():
        executable = policy._sample_actions.lower(
            jax.random.key(0), model_observation, **dict(policy._sample_kwargs)
        ).compile()
    raw = executable.cost_analysis()
    if isinstance(raw, list):
        raw = raw[0]
    reported_costs = {"flops", "transcendentals", "bytes accessed"}
    result = {
        str(key): float(value)
        for key, value in raw.items()
        if key in reported_costs and isinstance(value, (int, float))
    }
    if not result.get("flops", 0.0) > 0.0:
        raise RuntimeError(f"XLA cost analysis did not report positive FLOPs: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hint-dim", type=int, default=0)
    parser.add_argument("--transition-task-id", type=int)
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as train_config
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")

    memory_before = gpu_memory_mib()
    load_started = time.perf_counter()
    policy = policy_config.create_trained_policy(
        train_config.get_config(args.config), args.checkpoint
    )
    load_seconds = time.perf_counter() - load_started
    memory_loaded = gpu_memory_mib()
    observation = make_observation(
        args.hint_dim,
        transition_task_id=args.transition_task_id,
        history_steps=args.history_steps,
    )

    for _ in range(args.warmup):
        policy.infer(observation)
    memory_warm = gpu_memory_mib()
    model_parameter_count = parameter_count(policy._model)
    xla_cost = lowered_cost_analysis(policy, observation)

    direct_wall_ms: list[float] = []
    direct_model_ms: list[float] = []
    for _ in range(args.trials):
        started = time.perf_counter()
        result = policy.infer(observation)
        direct_wall_ms.append((time.perf_counter() - started) * 1000)
        direct_model_ms.append(float(result["policy_timing"]["infer_ms"]))

    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port, metadata={})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    for _ in range(args.warmup):
        client.infer(observation)

    websocket_wall_ms: list[float] = []
    websocket_server_ms: list[float] = []
    for _ in range(args.trials):
        started = time.perf_counter()
        result = client.infer(observation)
        websocket_wall_ms.append((time.perf_counter() - started) * 1000)
        websocket_server_ms.append(float(result["server_timing"]["infer_ms"]))

    websocket_summary = summarize(websocket_wall_ms, seed=23)
    report = {
        "arm": args.arm,
        "config": args.config,
        "checkpoint": str(args.checkpoint.resolve()),
        "protocol": {
            "device": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "client_image_shape": [3, 480, 640],
            "camera_count": 3,
            "state_dim": 14,
            "hint_dim": args.hint_dim,
            "transition_task_id": args.transition_task_id,
            "history_steps": args.history_steps,
            "warmup": args.warmup,
            "trials": args.trials,
            "input": "deterministic synthetic uint8 images and zero state/hint",
        },
        "load_seconds": load_seconds,
        "model_parameter_count": model_parameter_count,
        "xla_cost_analysis": xla_cost,
        "gpu_memory_mib": {
            "before_load": memory_before,
            "after_load": memory_loaded,
            "after_warmup": memory_warm,
        },
        "direct_wall": summarize(direct_wall_ms, seed=17),
        "direct_model": summarize(direct_model_ms, seed=19),
        "websocket_roundtrip": websocket_summary,
        "websocket_server": summarize(websocket_server_ms, seed=29),
        "websocket_throughput_requests_per_second": 1000.0 / float(websocket_summary["mean_ms"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
