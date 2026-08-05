#!/usr/bin/env python3
# -*-coding:utf8-*-
"""VLA 部署前置体检 —— 只读, 不驱动任何硬件。

起因 (2026-07-28 FastWAM 真机复盘): 一次真机跑表现为"抖动到衣物附近、有抓的趋势但抓不成功"。
离线复盘定位到两类可在启动前拦住的问题:

  1. **相机分辨率失配 → letterbox**。训练数据是原生 4:3 640x480; 若相机以 16:9 出流
     (848x480 / 1280x720), node 的 resize_with_pad 会把内容压到 75% 高度 + 上下黑边。
     离线实测该形变使 chunk 内运动量降到 0.82x、free_MAE(相邻 chunk 分歧) +28%。
     历史 artifact: /data2/gwp_eval/out/*_dump/ref_*.png 里 6 次真机跑有 5 次是 letterbox 的。

  2. **夹爪 proprio 闩锁**。训练数据 action[t] ≡ state[t] (relabel 约定), 模型的夹爪输出
     基本是夹爪 proprio 的回读: 离线把夹爪 proprio 冻结在 1.5mm(闭) → 输出张开率 52.6%→19.1%;
     冻结在 79mm(开) → 90.6%。闭环下这是自锁: 夹爪停在哪就继续命令哪 → 永远抓不到。
     启动时若两臂夹爪都读到近全闭, 提示确认夹爪确实能动作 (使能/标定/CAN)。

  3. **起始位偏离 demo 起始分布**。影响比 1/2 温和 (离线把 proprio 偏到实测真机位姿,
     运动量与夹爪几乎不退化, 只有 free_MAE +34%), 但仍会加剧抖动 → 作为 WARN。

用法 (需先 source ROS2 + ros2_ws):
    python3 fastwam/deploy/preflight.py
    python3 fastwam/deploy/preflight.py --ref <ref.json> --timeout 15
    python3 fastwam/deploy/preflight.py --warn-only      # 只报告, 永远 exit 0

退出码: 0 = 全通过 / 仅 WARN;  1 = 有 FAIL;  2 = 传感数据没收齐 (topic 没起来)。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_REF = _REPO / "fastwam" / "deploy" / "refs" / "visrobot01_v3.json"

# 运行时真正的 color topic 由 multi_camera_node 转发 (单层 'camera'), 与
# autonomy_launch.py:384 和 policy_inference_node 的默认参数一致。
# ⚠️ config/cameras.yml 里写的是双层 (/camera_f/camera_f/...) —— 那是 realsense2_camera
# 的原始 topic, 被 multi_camera_node 转发之前的名字, 不是 policy node 订阅的。
# 两种都订上, 谁先来消息用谁 (免受该不一致影响)。
IMG_TOPICS = {
    "top_head":   ["/camera_f/camera/color/image_raw", "/camera_f/camera_f/color/image_raw"],
    "hand_left":  ["/camera_l/camera/color/image_raw", "/camera_l/camera_l/color/image_raw"],
    "hand_right": ["/camera_r/camera/color/image_raw", "/camera_r/camera_r/color/image_raw"],
}
JOINT_TOPICS = {"left": ["/puppet/joint_left"], "right": ["/puppet/joint_right"]}

GRIP_LATCH_THR = 0.003      # 两臂夹爪都 < 3mm → 闩锁风险 WARN
START_SIGMA_WARN = 2.5      # 单关节 σ 偏离超此值 → WARN
START_L2_WARN = 1.0         # 到最近 demo 起始位的 L2 (rad) 超此值 → WARN

OK, WARN, FAIL = "\033[32m✅\033[0m", "\033[33m⚠️ \033[0m", "\033[31m🔴\033[0m"


def collect(timeout: float):
    """订阅一次拿到每个 topic 的最新一条消息。返回 (imgs, joints)。"""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSPresetProfiles
    from sensor_msgs.msg import Image, JointState

    got_img, got_joint = {}, {}

    class Probe(Node):
        def __init__(self):
            super().__init__("deploy_preflight")
            qos = QoSPresetProfiles.SENSOR_DATA.value
            for name, topics in IMG_TOPICS.items():
                for topic in topics:
                    self.create_subscription(Image, topic,
                                             lambda m, n=name: got_img.setdefault(n, m), qos)
            for name, topics in JOINT_TOPICS.items():
                for topic in topics:
                    self.create_subscription(JointState, topic,
                                             lambda m, n=name: got_joint.setdefault(n, m), 10)

    rclpy.init()
    node = Probe()
    try:
        deadline = node.get_clock().now().nanoseconds + int(timeout * 1e9)
        while node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(got_img) == len(IMG_TOPICS) and len(got_joint) == len(JOINT_TOPICS):
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return got_img, got_joint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--warn-only", action="store_true", help="只报告, 永远 exit 0")
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    exp_h, exp_w = ref["expected_image_hw"]
    print(f"\n=== 部署前置体检 ===  参考: {Path(args.ref).name} "
          f"({ref['n_episodes']} 集)\n{ref.get('_note', '')}\n")

    imgs, joints = collect(args.timeout)
    missing = [t for t in IMG_TOPICS if t not in imgs] + [t for t in JOINT_TOPICS if t not in joints]
    if missing:
        print(f"{FAIL} 以下 topic 在 {args.timeout:.0f}s 内没收到消息: {missing}")
        print("   相机/臂 stack 没起来? 先跑 start_autonomy 的相机部分或 piper_tools/test_cameras.py")
        return 2

    fails, warns = [], []

    # ── 1. 相机分辨率 / 编码 ────────────────────────────────────────────────
    print("── 1. 相机 ──")
    for name in IMG_TOPICS:
        m = imgs[name]
        shape_ok = (m.height, m.width) == (exp_h, exp_w)
        aspect = m.width / max(m.height, 1)
        tag = OK if shape_ok else FAIL
        print(f"  {tag} {name:<11} {m.width}x{m.height} (aspect {aspect:.3f}) encoding={m.encoding}")
        if not shape_ok:
            pad = 1.0 - (exp_w / max(m.width, 1) * m.height) / exp_h if aspect > exp_w / exp_h else 0.0
            fails.append(f"{name} 分辨率 {m.width}x{m.height} ≠ 训练 {exp_w}x{exp_h}"
                         + (f" → resize_with_pad 会 letterbox 掉 ~{pad*100:.0f}% 高度" if pad > 0.01 else ""))
        if m.encoding != "rgb8":
            warns.append(f"{name} encoding={m.encoding} ≠ rgb8 — "
                         f"fast_obs_pipeline:=true 会跳过 BGR→RGB 转换 → 通道错位")

    # ── 2. 夹爪闩锁风险 ────────────────────────────────────────────────────
    print("\n── 2. 夹爪 ──")
    gl = float(joints["left"].position[6]) if len(joints["left"].position) > 6 else float("nan")
    gr = float(joints["right"].position[6]) if len(joints["right"].position) > 6 else float("nan")
    latch = (gl < GRIP_LATCH_THR) and (gr < GRIP_LATCH_THR)
    print(f"  {WARN if latch else OK} 左={gl*1000:.1f}mm  右={gr*1000:.1f}mm")
    if latch:
        warns.append(f"两臂夹爪都 <{GRIP_LATCH_THR*1000:.0f}mm(近全闭) → 闩锁风险。"
                     "启动后请确认夹爪真的会动作 (使能/标定/CAN); "
                     "夹爪不动作时模型会一直回读闭合状态, 永远抓不到")

    # ── 3. 起始位 ──────────────────────────────────────────────────────────
    print("\n── 3. 起始位 vs demo 起始分布 ──")
    q = list(joints["left"].position[:7]) + list(joints["right"].position[:7])
    if len(q) < 14:
        fails.append(f"关节维度不足 14 (拿到 {len(q)})")
        q += [0.0] * (14 - len(q))
    mu, sd, med = ref["start_mean"], ref["start_std"], ref["start_median"]
    names = ref["dim_names"]
    worst = 0.0
    print(f"  {'维':<8} {'当前':>9} {'demo均值':>9} {'σ':>7} {'σ偏离':>8}")
    for i in range(14):
        if names[i].endswith("grip"):
            continue
        z = (q[i] - mu[i]) / max(sd[i], 1e-6)
        worst = max(worst, abs(z))
        flag = "  <<<" if abs(z) > START_SIGMA_WARN else ""
        print(f"  {names[i]:<8} {q[i]:>9.3f} {mu[i]:>9.3f} {sd[i]:>7.3f} {z:>+7.1f}σ{flag}")
    l2 = math.sqrt(sum((q[i] - med[i]) ** 2 for i in range(14) if not names[i].endswith("grip")))
    print(f"  到 demo 起始位中位数的 L2 (12 关节) = {l2:.3f} rad")
    if worst > START_SIGMA_WARN or l2 > START_L2_WARN:
        warns.append(f"起始位偏离 demo 分布 (最大 {worst:.1f}σ, L2 {l2:.2f} rad) → 建议先归位")
        deg = [round(math.degrees(med[i]), 2) for i in range(14)]
        print("\n  归位命令 (会让真机运动, 需人工监护; 先确保 execute=false):")
        print(f"    python3 piper_tools/go_to_pose.py \\\n"
              f"      --left  {' '.join(str(x) for x in deg[0:6])} \\\n"
              f"      --right {' '.join(str(x) for x in deg[7:13])} \\\n"
              f"      --grip 0.0 --duration 5")

    # ── 汇总 ───────────────────────────────────────────────────────────────
    print("\n=== 汇总 ===")
    for w in warns:
        print(f"{WARN} {w}")
    for f in fails:
        print(f"{FAIL} {f}")
    if not warns and not fails:
        print(f"{OK} 全部通过")
    if fails and not args.warn_only:
        print("\n有 FAIL — 建议先修再上真机 (要强行跳过: --warn-only, 或给启动脚本传 --skip-preflight)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
