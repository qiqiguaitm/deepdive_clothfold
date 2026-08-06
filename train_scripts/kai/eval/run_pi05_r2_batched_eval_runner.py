#!/usr/bin/env python3
"""Run an ephemeral R2 bridge while preserving the frozen base evaluator files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile


BASE_BRIDGE_SHA256 = "8ae453cce53c5e2516dd8e4d0f9204ce789c61ba26704a27869fd0165fc8a167"
BASE_RUNNER_SHA256 = "1bd7cc518588aff27e3c0afd3368be99cf162b09f61f62b3ec028a5faffe3940"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"R2 patch {label!r} expected one match, found {count}")
    return source.replace(old, new, 1)


def render_r2_bridge(source: str) -> str:
    source = replace_once(
        source,
        'SCRIPT_DIR = Path(__file__).resolve().parent',
        'SCRIPT_DIR = Path(os.environ["ROBOTWIN_R2_BASE_EVAL_DIR"]).resolve()',
        label="stable import root",
    )
    source = replace_once(
        source,
        '''if _MODEL_INTERFACE == "openpi":
    from examples.Robotwin.eval_files.model2robotwin_openpi import (  # noqa: E402
        get_model,
    )
else:''',
        '''if _MODEL_INTERFACE == "openpi_r2":
    from examples.Robotwin.eval_files.model2robotwin_openpi_r2 import (  # noqa: E402
        get_model,
    )
elif _MODEL_INTERFACE == "openpi":
    from examples.Robotwin.eval_files.model2robotwin_openpi import (  # noqa: E402
        get_model,
    )
else:''',
        label="R2 model interface",
    )
    source = replace_once(
        source,
        '''                    observation = message["observation"]
                    requires_query = _observation_requires_model_query(
                        model=model,
                        slot_id=slot_id,
                        instruction=instruction,
                        must_query=observation_request.must_query,
                    )
                    if requires_query:
                        pending_examples[slot_id] = {
                            "slot_id": slot_id,
                            "seed": int(message["seed"]),
                            "episode_id": int(message["episode_id"]),
                            "example": model.build_example(instruction, observation),
                        }''',
        '''                    observation = message["observation"]
                    example = model.build_example(instruction, observation)
                    observe = getattr(model, "observe", None)
                    if observe is not None:
                        observe(example, slot_id=slot_id, step=int(message["step"]))
                    requires_query = _observation_requires_model_query(
                        model=model,
                        slot_id=slot_id,
                        instruction=instruction,
                        must_query=observation_request.must_query,
                    )
                    if requires_query:
                        pending_examples[slot_id] = {
                            "slot_id": slot_id,
                            "seed": int(message["seed"]),
                            "episode_id": int(message["episode_id"]),
                            "example": example,
                        }''',
        label="causal observation hook",
    )
    return source


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        raise SystemExit(f"usage: {sys.argv[0]} BASE_RUNNER -- [runner arguments ...]")
    runner_path = Path(sys.argv[1]).resolve()
    runner_bytes = runner_path.read_bytes()
    if sha256_bytes(runner_bytes) != BASE_RUNNER_SHA256:
        raise RuntimeError(f"frozen batched evaluator hash mismatch: {runner_path}")
    eval_dir = runner_path.parent
    bridge_path = eval_dir / "robotwin_batch_bridge.py"
    bridge_bytes = bridge_path.read_bytes()
    if sha256_bytes(bridge_bytes) != BASE_BRIDGE_SHA256:
        raise RuntimeError(f"frozen RoboTwin bridge hash mismatch: {bridge_path}")
    rendered = render_r2_bridge(bridge_bytes.decode("utf-8"))

    descriptor, temporary_name = tempfile.mkstemp(prefix="pi05-r2-bridge-", suffix=".py")
    temporary_bridge = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
        os.environ["ROBOTWIN_R2_BASE_EVAL_DIR"] = str(eval_dir)
        os.environ["ROBOTWIN_R2_BRIDGE_PATH"] = str(temporary_bridge)
        runner_source = runner_bytes.decode("utf-8")
        runner_source = replace_once(
            runner_source,
            'str(SCRIPT_DIR / "robotwin_batch_bridge.py")',
            'os.environ["ROBOTWIN_R2_BRIDGE_PATH"]',
            label="ephemeral bridge command",
        )
        sys.argv = [str(runner_path), *sys.argv[3:]]
        globals_dict = {
            "__name__": "__main__",
            "__file__": str(runner_path),
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(runner_source, str(runner_path), "exec"), globals_dict)
    finally:
        temporary_bridge.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
