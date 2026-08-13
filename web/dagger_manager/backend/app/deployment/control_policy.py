"""Typed control-policy configuration and update planning.

This module deliberately uses domain names (``rtc.execute_horizon``) rather
than ROS parameter names.  The ROS mapping lives in ``ros_gateway.py`` so the
web API and model adapters do not depend on policy_inference_node internals.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TimingConfig(BaseModel):
    inference_rate_hz: float = Field(3.0, ge=0.1, le=60.0)
    publish_rate_hz: int = Field(30, ge=1, le=200)
    speed_factor: float = Field(1.0, ge=0.5, le=2.0)
    speed_factor_max: float = Field(2.0, ge=0.5, le=3.0)


class RtcConfig(BaseModel):
    enabled: bool = True
    execute_horizon: int = Field(16, ge=1, le=50)
    max_guidance_weight: float = Field(0.5, ge=0.0, le=5.0)
    latency_steps: int = Field(8, ge=0, le=49)


class ChunkBlendConfig(BaseModel):
    method: Literal["min_jerk", "linear"] = "min_jerk"
    min_steps: int = Field(8, ge=0, le=50)
    max_steps: int = Field(12, ge=0, le=50)
    decay_alpha: float = Field(0.25, gt=0.0, le=1.0)


class PublishFilterConfig(BaseModel):
    type: Literal["ema"] = "ema"
    alpha: float = Field(0.5, gt=0.0, le=1.0)
    exclude_gripper: bool = True


class ObservationFilterConfig(BaseModel):
    state_lowpass_alpha: float = Field(1.0, gt=0.0, le=1.0)


class ControlPolicyConfig(BaseModel):
    timing: TimingConfig = TimingConfig()
    rtc: RtcConfig = RtcConfig()
    chunk_blend: ChunkBlendConfig = ChunkBlendConfig()
    publish_filter: PublishFilterConfig = PublishFilterConfig()
    observation_filter: ObservationFilterConfig = ObservationFilterConfig()

    @model_validator(mode="after")
    def validate_combinations(self) -> "ControlPolicyConfig":
        if self.timing.speed_factor > self.timing.speed_factor_max:
            raise ValueError("speed_factor must be <= speed_factor_max")
        if self.rtc.enabled and self.rtc.execute_horizon <= self.rtc.latency_steps:
            raise ValueError("RTC execute_horizon must be greater than latency_steps")
        if self.chunk_blend.max_steps and self.chunk_blend.max_steps < self.chunk_blend.min_steps:
            raise ValueError("chunk_blend.max_steps must be 0 (unlimited) or >= min_steps")
        return self

    def warnings(self) -> list[str]:
        out: list[str] = []
        max_steps = self.chunk_blend.max_steps
        if max_steps and max_steps / self.timing.publish_rate_hz > 0.5:
            out.append("chunk blend window introduces more than 500 ms of command lag")
        if self.timing.speed_factor > 1.0 and self.publish_filter.alpha < 0.3:
            out.append("high speed with strong EMA can cause visible command lag")
        if self.rtc.max_guidance_weight > 0.5:
            out.append("RTC guidance weight above the validated 0.5 baseline is experimental")
        return out


class ControlPolicyPatch(BaseModel):
    """A full desired config plus apply semantics.

    Full configs make validation/reproducibility deterministic.  The backend
    computes the field-level delta against the live ROS/readback config.
    """

    config: ControlPolicyConfig
    dry_run: bool = False


class UpdateClass(str, Enum):
    HOT = "hot"
    SAFE_IDLE = "safe_idle"
    RESTART = "restart"


# Restart means the node/model/timer is constructed from this value. Safe-idle
# means the value can be set live only after execution is disabled and buffers
# are flushed. Everything else listed is supported as a true hot update.
FIELD_UPDATE_CLASS: dict[str, UpdateClass] = {
    "timing.publish_rate_hz": UpdateClass.RESTART,
    "rtc.enabled": UpdateClass.RESTART,
    "chunk_blend.method": UpdateClass.SAFE_IDLE,
    "timing.inference_rate_hz": UpdateClass.HOT,
    "timing.speed_factor": UpdateClass.HOT,
    "timing.speed_factor_max": UpdateClass.RESTART,
    "rtc.execute_horizon": UpdateClass.SAFE_IDLE,
    "rtc.max_guidance_weight": UpdateClass.SAFE_IDLE,
    "rtc.latency_steps": UpdateClass.SAFE_IDLE,
    "chunk_blend.min_steps": UpdateClass.SAFE_IDLE,
    "chunk_blend.max_steps": UpdateClass.SAFE_IDLE,
    "chunk_blend.decay_alpha": UpdateClass.HOT,
    "publish_filter.alpha": UpdateClass.HOT,
    "observation_filter.state_lowpass_alpha": UpdateClass.HOT,
}


def _flatten(value: BaseModel, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.model_dump().items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            for sub_key, sub_item in item.items():
                out[f"{path}.{sub_key}"] = sub_item
        else:
            out[path] = item
    return out


def build_update_plan(current: ControlPolicyConfig, desired: ControlPolicyConfig) -> dict[str, Any]:
    before, after = _flatten(current), _flatten(desired)
    changes: list[dict[str, Any]] = []
    for field, new_value in after.items():
        old_value = before.get(field)
        if old_value == new_value:
            continue
        classification = FIELD_UPDATE_CLASS.get(field)
        # exclude_gripper/type are fixed contracts rather than mutable knobs.
        if classification is None:
            continue
        changes.append({
            "field": field,
            "old": old_value,
            "new": new_value,
            "classification": classification.value,
        })
    return {
        "changes": changes,
        "requires_safe_idle": any(c["classification"] == UpdateClass.SAFE_IDLE for c in changes),
        "requires_restart": any(c["classification"] == UpdateClass.RESTART for c in changes),
        "warnings": desired.warnings(),
    }


def preset_config(name: str, variant: Optional[str] = None) -> ControlPolicyConfig:
    if name == "safe_observe":
        return ControlPolicyConfig(
            timing={"inference_rate_hz": 3.0, "publish_rate_hz": 30,
                    "speed_factor": 0.5, "speed_factor_max": 2.0},
            rtc={"enabled": True, "execute_horizon": 16,
                 "max_guidance_weight": 0.5, "latency_steps": 8},
        )
    if name == "raw_ablation":
        return ControlPolicyConfig(
            rtc={"enabled": False, "execute_horizon": 16,
                 "max_guidance_weight": 0.0, "latency_steps": 8},
            chunk_blend={"method": "linear", "min_steps": 0,
                         "max_steps": 0, "decay_alpha": 0.25},
            publish_filter={"type": "ema", "alpha": 1.0,
                            "exclude_gripper": True},
        )
    if name != "production_default":
        raise KeyError(f"unknown control preset: {name}")
    if variant == "v1":
        return ControlPolicyConfig(
            timing={"inference_rate_hz": 20.0, "publish_rate_hz": 30,
                    "speed_factor": 1.0, "speed_factor_max": 2.0},
            rtc={"enabled": True, "execute_horizon": 12,
                 "max_guidance_weight": 0.5, "latency_steps": 6},
            chunk_blend={"method": "min_jerk", "min_steps": 8,
                         "max_steps": 12, "decay_alpha": 0.25},
            # Historical V1 real-arm deployment uses mild publish EMA (0.7).
            publish_filter={"type": "ema", "alpha": 0.7,
                            "exclude_gripper": True},
            observation_filter={"state_lowpass_alpha": 1.0},
        )
    # Historical start_autonomy_from_ckpt.sh / JAX deployment defaults:
    # 3 Hz replan, k=8, RTC horizon=16, uncapped overlap, EMA=0.5.
    return ControlPolicyConfig(
        timing={"inference_rate_hz": 3.0, "publish_rate_hz": 30,
                "speed_factor": 1.0, "speed_factor_max": 2.0},
        rtc={"enabled": True, "execute_horizon": 16,
             "max_guidance_weight": 0.5, "latency_steps": 8},
        chunk_blend={"method": "min_jerk", "min_steps": 8,
                     "max_steps": 0, "decay_alpha": 0.25},
        publish_filter={"type": "ema", "alpha": 0.5,
                        "exclude_gripper": True},
        observation_filter={"state_lowpass_alpha": 1.0},
    )
