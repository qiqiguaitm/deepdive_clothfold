"""Safety-gated operation mode and execute controller."""
from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any


class OperationMode(str, Enum):
    OBSERVE = "observe"
    DEPLOY = "deploy"
    DAGGER = "dagger"


class DeploymentController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = OperationMode.OBSERVE
        self._last_preflight: dict[str, Any] | None = None

    def mode(self) -> str:
        with self._lock:
            return self._mode.value

    def set_mode(self, mode: OperationMode, bridge, session) -> dict[str, Any]:
        # Changing operating semantics while commands are live is forbidden.
        if bridge.snapshot().get("policy_execute"):
            raise RuntimeError("disable execution before changing operation mode")
        if mode != OperationMode.OBSERVE and not session.is_running():
            raise RuntimeError("load a policy session before selecting deploy/dagger mode")
        with self._lock:
            self._mode = mode
        return {"mode": mode.value}

    def preflight(self, bridge, session) -> dict[str, Any]:
        ros = bridge.snapshot()
        checks = {
            "policy_session": session.is_running(),
            "policy_node": bool(ros.get("policy_node_ready")),
            "ros_bridge": bool(ros.get("ros_alive")),
            "dagger_recorder": ros.get("state") is not None,
        }
        cameras = ros.get("cameras", {})
        for name in ("top_head", "hand_left", "hand_right"):
            checks[f"camera_{name}"] = float(cameras.get(name, {}).get("fps", 0.0)) >= 5.0
        failures = [name for name, ok in checks.items() if not ok]
        result = {
            "ok": not failures,
            "checks": checks,
            "failures": failures,
            "mode": self.mode(),
            "ts": time.time(),
        }
        with self._lock:
            self._last_preflight = result
        return result

    def execute(self, enable: bool, bridge, session) -> dict[str, Any]:
        if enable:
            if self.mode() == OperationMode.OBSERVE.value:
                raise RuntimeError("observe mode cannot enable real-arm execution")
            result = self.preflight(bridge, session)
            if not result["ok"]:
                raise RuntimeError("preflight failed: " + ", ".join(result["failures"]))
        if not bridge.publish_execute(enable):
            raise RuntimeError("ROS bridge not alive")
        # Require topic readback rather than treating publish() as success.
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            if bridge.snapshot().get("policy_execute") == bool(enable):
                return {"ok": True, "enable": enable, "mode": self.mode()}
            time.sleep(0.05)
        raise RuntimeError("/policy/execute readback timeout")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"operation_mode": self._mode.value,
                    "preflight": self._last_preflight}


controller = DeploymentController()
