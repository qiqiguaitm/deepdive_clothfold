#!/usr/bin/env python3
"""Acknowledged execute/stop control for the isolated FastWAM deployment.

Unlike a one-shot ``ros2 topic pub``, this helper waits until the
``policy_inference`` subscriber is discovered, publishes a short idempotent
burst, and verifies the result through the ``/policy/actions`` stream.
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class FastwamIsolatedControl(Node):
    def __init__(
        self,
        execute_topic: str,
        actions_topic: str,
        policy_node: str,
    ) -> None:
        super().__init__("fastwam_isolated_control")
        self.execute_topic = execute_topic
        self.actions_topic = actions_topic
        self.policy_node = policy_node.lstrip("/")
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.execute_pub = self.create_publisher(Bool, self.execute_topic, reliable)
        self.last_action_time: float | None = None
        self.action_count = 0
        self.create_subscription(
            JointState, self.actions_topic, self._on_action, reliable
        )

    def _on_action(self, _msg: JointState) -> None:
        self.last_action_time = time.monotonic()
        self.action_count += 1

    def spin_for(self, duration: float) -> None:
        deadline = time.monotonic() + max(duration, 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.05, deadline - time.monotonic()))

    def wait_for_policy(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            endpoints = self.get_subscriptions_info_by_topic(self.execute_topic)
            if any(
                endpoint.node_name.lstrip("/") == self.policy_node
                for endpoint in endpoints
            ):
                return True
        return False

    def publish_burst(self, enabled: bool, duration: float, rate: float = 20.0) -> None:
        msg = Bool(data=enabled)
        period = 1.0 / max(rate, 1.0)
        deadline = time.monotonic() + duration
        while rclpy.ok() and time.monotonic() < deadline:
            self.execute_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=period)

    def wait_for_actions(self, timeout: float) -> bool:
        baseline = self.action_count
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.action_count > baseline:
                return True
        return False

    def wait_for_quiet(self, quiet_window: float, timeout: float) -> bool:
        started = time.monotonic()
        deadline = started + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            last = self.last_action_time
            if last is None:
                if time.monotonic() - started >= quiet_window:
                    return True
            elif time.monotonic() - last >= quiet_window:
                return True
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("on", "off"))
    parser.add_argument("--discover-timeout", type=float, default=5.0)
    parser.add_argument("--ack-timeout", type=float, default=3.0)
    parser.add_argument("--burst-seconds", type=float, default=0.8)
    parser.add_argument("--quiet-seconds", type=float, default=0.8)
    parser.add_argument("--execute-topic", default="/policy/execute")
    parser.add_argument("--actions-topic", default="/policy/actions")
    parser.add_argument("--policy-node", default="policy_inference")
    args = parser.parse_args()

    rclpy.init()
    node = FastwamIsolatedControl(
        execute_topic=args.execute_topic,
        actions_topic=args.actions_topic,
        policy_node=args.policy_node,
    )
    try:
        if not node.wait_for_policy(args.discover_timeout):
            print(
                f"ERROR: {args.execute_topic} 上没有发现 {args.policy_node} 订阅者；"
                "拒绝发送未确认命令。",
                file=sys.stderr,
            )
            return 2

        if args.command == "on":
            print(f"执行请求：已确认 {args.policy_node} 订阅者，发送重复 true。")
            node.publish_burst(True, args.burst_seconds)
            if node.wait_for_actions(args.ack_timeout):
                print(f"ACK: {args.actions_topic} 已开始输出，FastWAM 执行已开启。")
                return 0
            print(
                "ERROR: true 已发送但未检测到动作流；立即回退为 false。",
                file=sys.stderr,
            )
            node.publish_burst(False, max(args.burst_seconds, 1.0))
            return 3

        for attempt in range(1, 4):
            node.publish_burst(False, args.burst_seconds)
            if node.wait_for_quiet(args.quiet_seconds, args.ack_timeout):
                print(
                    f"ACK: {args.actions_topic} 已静默 {args.quiet_seconds:.1f}s，"
                    f"FastWAM 执行已停止（attempt {attempt}）。"
                )
                return 0
            print(f"WARN: stop attempt {attempt} 后动作流仍存在，重试。")
        print(
            "ERROR: 连续 3 次 false 后动作流仍未停止；请使用硬件急停。",
            file=sys.stderr,
        )
        return 4
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
