from __future__ import annotations

import unittest

from app.deployment.control_policy import (
    ControlPolicyConfig,
    build_update_plan,
    preset_config,
)
from app.deployment.ros_gateway import RosPolicyGateway
from app.deployment.controller import DeploymentController, OperationMode


class ControlPolicyTests(unittest.TestCase):
    def test_v1_production_preset(self) -> None:
        cfg = preset_config("production_default", "v1")
        self.assertEqual(cfg.timing.inference_rate_hz, 20.0)
        self.assertEqual(cfg.timing.publish_rate_hz, 30)
        self.assertEqual(cfg.rtc.latency_steps, 6)
        self.assertEqual(cfg.rtc.execute_horizon, 12)
        self.assertEqual(cfg.chunk_blend.min_steps, 8)
        self.assertEqual(cfg.chunk_blend.max_steps, 12)
        self.assertEqual(cfg.publish_filter.alpha, 0.7)

    def test_v0_production_preset_matches_historical_deploy(self) -> None:
        cfg = preset_config("production_default", "v0")
        self.assertEqual(cfg.timing.inference_rate_hz, 3.0)
        self.assertEqual(cfg.timing.publish_rate_hz, 30)
        self.assertEqual(cfg.rtc.latency_steps, 8)
        self.assertEqual(cfg.rtc.execute_horizon, 16)
        self.assertEqual(cfg.chunk_blend.min_steps, 8)
        self.assertEqual(cfg.chunk_blend.max_steps, 0)
        self.assertEqual(cfg.publish_filter.alpha, 0.5)

    def test_classifies_hot_safe_idle_and_restart(self) -> None:
        current = preset_config("production_default", "v0")
        desired = current.model_copy(deep=True)
        desired.publish_filter.alpha = 0.7
        desired.rtc.execute_horizon = 20
        desired.timing.publish_rate_hz = 20
        plan = build_update_plan(current, desired)
        classes = {c["field"]: c["classification"] for c in plan["changes"]}
        self.assertEqual(classes["publish_filter.alpha"], "hot")
        self.assertEqual(classes["rtc.execute_horizon"], "safe_idle")
        self.assertEqual(classes["timing.publish_rate_hz"], "restart")
        self.assertTrue(plan["requires_safe_idle"])
        self.assertTrue(plan["requires_restart"])

    def test_rejects_invalid_rtc_window(self) -> None:
        with self.assertRaises(ValueError):
            ControlPolicyConfig(rtc={"enabled": True, "execute_horizon": 6,
                                     "latency_steps": 6,
                                     "max_guidance_weight": 0.5})

    def test_launch_args_are_ros_boundary_only(self) -> None:
        args = RosPolicyGateway.launch_args(preset_config("production_default", "v1"))
        self.assertIn("enable_rtc:=true", args)
        self.assertIn("publish_smooth_alpha:=0.7", args)
        self.assertIn("publish_rate:=30", args)


class _FakeSession:
    def __init__(self, running: bool = True) -> None:
        self.running = running

    def is_running(self) -> bool:
        return self.running


class _FakeBridge:
    def __init__(self) -> None:
        self.execute = False
        self.snap = {
            "policy_execute": False, "policy_node_ready": True,
            "ros_alive": True, "state": "POLICY_RUN",
            "cameras": {name: {"fps": 30.0} for name in
                        ("top_head", "hand_left", "hand_right")},
        }

    def snapshot(self):
        return {**self.snap, "policy_execute": self.execute}

    def publish_execute(self, enable: bool) -> bool:
        self.execute = enable
        return True


class DeploymentControllerTests(unittest.TestCase):
    def test_observe_mode_blocks_execution(self) -> None:
        ctl = DeploymentController()
        with self.assertRaises(RuntimeError):
            ctl.execute(True, _FakeBridge(), _FakeSession())

    def test_deploy_requires_preflight_and_readback(self) -> None:
        ctl, bridge, session = DeploymentController(), _FakeBridge(), _FakeSession()
        ctl.set_mode(OperationMode.DEPLOY, bridge, session)
        result = ctl.execute(True, bridge, session)
        self.assertTrue(result["ok"])
        self.assertTrue(bridge.execute)

    def test_failed_camera_blocks_execution(self) -> None:
        ctl, bridge, session = DeploymentController(), _FakeBridge(), _FakeSession()
        ctl.set_mode(OperationMode.DAGGER, bridge, session)
        bridge.snap["cameras"]["hand_left"]["fps"] = 0.0
        with self.assertRaises(RuntimeError):
            ctl.execute(True, bridge, session)


if __name__ == "__main__":
    unittest.main()
