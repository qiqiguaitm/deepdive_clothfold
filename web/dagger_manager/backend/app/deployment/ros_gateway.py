"""Single boundary between deployment-domain configuration and ROS params."""
from __future__ import annotations

import subprocess
from typing import Any

from .control_policy import ControlPolicyConfig, build_update_plan


DOMAIN_TO_ROS = {
    "timing.inference_rate_hz": "inference_rate",
    "timing.publish_rate_hz": "publish_rate",
    "timing.speed_factor": "speed_factor",
    "timing.speed_factor_max": "speed_factor_max",
    "rtc.enabled": "enable_rtc",
    "rtc.execute_horizon": "rtc_execute_horizon",
    "rtc.max_guidance_weight": "rtc_max_guidance_weight",
    "rtc.latency_steps": "latency_k",
    "chunk_blend.method": "rtc_smooth_method",
    "chunk_blend.min_steps": "min_smooth_steps",
    "chunk_blend.max_steps": "max_smooth_steps",
    "chunk_blend.decay_alpha": "decay_alpha",
    "publish_filter.alpha": "publish_smooth_alpha",
    "observation_filter.state_lowpass_alpha": "obs_state_lowpass_alpha",
}


class RosPolicyGateway:
    def __init__(self, node: str = "/policy_inference") -> None:
        self.node = node

    @staticmethod
    def launch_args(config: ControlPolicyConfig) -> list[str]:
        flat = _flat(config)
        return [f"{ros}:={_launch_value(flat[field])}"
                for field, ros in DOMAIN_TO_ROS.items()]

    def apply_hot(self, current: ControlPolicyConfig,
                  desired: ControlPolicyConfig) -> dict[str, Any]:
        plan = build_update_plan(current, desired)
        if plan["requires_restart"]:
            raise RuntimeError("control update contains restart-required fields")
        if plan["requires_safe_idle"]:
            raise RuntimeError("control update requires execute=false and buffer flush")
        applied: list[str] = []
        try:
            for change in plan["changes"]:
                field = change["field"]
                ros_name = DOMAIN_TO_ROS[field]
                self._set(ros_name, change["new"])
                applied.append(field)
        except Exception:
            # Best-effort rollback of already acknowledged changes. Keeping
            # this boundary transactional prevents the UI config from drifting
            # away from a partially updated ROS node.
            by_field = {c["field"]: c for c in plan["changes"]}
            for field in reversed(applied):
                try:
                    self._set(DOMAIN_TO_ROS[field], by_field[field]["old"])
                except Exception:
                    pass
            raise
        return {"applied": applied, "plan": plan}

    def _set(self, name: str, value: Any) -> None:
        cmd = ["ros2", "param", "set", self.node, name, _cli_value(value)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0 or "Successful" not in output:
            raise RuntimeError(f"failed to set {name}: {output or proc.returncode}")


def _flat(config: ControlPolicyConfig) -> dict[str, Any]:
    raw = config.model_dump()
    return {f"{section}.{key}": value
            for section, values in raw.items()
            for key, value in values.items()}


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _launch_value(value: Any) -> str:
    return _cli_value(value)


gateway = RosPolicyGateway()
