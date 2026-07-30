#!/usr/bin/env python3
"""给两条机械臂下发【同一组关节角】, 到位后读回各自真实反馈, 对比两臂是否一致。

用来验证: 命令同样的 q, 两条臂(默认两条从臂)实际到达的位姿是否一致 ——
差异即两臂的机械/标定/零位偏差 (per-joint offset)。

⚠️ 这是【运动指令】: 会使能电机并让两条臂同步运动。运行前清空臂周围、手放急停旁。
   默认 dry-run 只打印计划; 必须加 --execute 且交互确认 yes 才真正运动。
   不通信/使能超时的臂自动跳过, 不会动。

接口与 go_zero_all.py 一致:
  * JointCtrl 单位 = 0.001° (度*1000); 反馈 joint_state.joint_i / 1000 = 度。
  * MotionCtrl_2(0x01, 0x01, speed, 0x00) = CAN控制 + 关节模式 + 速度%。

用法:
  # dry-run, 只打印将下发的目标 (零位)
  python3 compare_dual_arm.py

  # 真正运动到零位并对比两臂反馈
  python3 compare_dual_arm.py --execute

  # 指定一组关节角 (度, 6 个), 慢速 15%
  python3 compare_dual_arm.py --execute --joints 0,20,-30,0,40,0 --speed 15

  # 扫过一串内置测试位姿, 每个都对比 (逐关节隔离)
  python3 compare_dual_arm.py --execute --sweep

  # 换成对比两条主臂
  python3 compare_dual_arm.py --execute --cans can_left_mas,can_right_mas
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2

# 内置测试位姿 (度): 零位 + 逐关节小幅隔离动作, 便于定位是哪个关节两臂不一致。
SWEEP_DEG = [
    ("zero",   [0,   0,   0,   0,   0,   0]),
    ("j1+30",  [30,  0,   0,   0,   0,   0]),
    ("j2+30",  [0,   30,  0,   0,   0,   0]),
    ("j3-30",  [0,   0,  -30,  0,   0,   0]),
    ("j4+45",  [0,   0,   0,   45,  0,   0]),
    ("j5+45",  [0,   0,   0,   0,   45,  0]),
    ("j6+45",  [0,   0,   0,   0,   0,   45]),
    ("zero2",  [0,   0,   0,   0,   0,   0]),
]

DEG2MILLI = 1000.0  # JointCtrl 单位 = 0.001°


def jdeg(p):
    """读当前 6 关节反馈, 单位度。"""
    js = p.GetArmJointMsgs().joint_state
    return [round(getattr(js, f"joint_{i}") / 1000.0, 2) for i in range(1, 7)]


def enable(p, timeout=5.0):
    t = 0.0
    while not p.EnablePiper():
        time.sleep(0.05); t += 0.05
        if t >= timeout:
            return False
    return True


def send_pose(p, deg6, speed):
    """下发一次目标关节角 (度 → 0.001°)。"""
    p.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    milli = [round(d * DEG2MILLI) for d in deg6]
    p.JointCtrl(*milli)


def goto_and_read(arms, deg6, speed, hold):
    """两臂同步下发同一目标, 持续 hold 秒, 返回 {can: 实测度}。"""
    t = 0.0
    while t < hold:
        for _c, p in arms:
            send_pose(p, deg6, speed)
        time.sleep(0.1); t += 0.1
    time.sleep(0.15)
    return {c: jdeg(p) for c, p in arms}


def print_compare(name, cmd, fb, cans):
    """打印一个位姿的对比表: 命令 / 各臂实测 / 命令误差 / 两臂差。"""
    a, b = cans[0], cans[1] if len(cans) > 1 else None
    print(f"\n── 位姿 {name}  命令(度) = {cmd}")
    hdr = f"  {'joint':6}{'cmd':>9}"
    for c in cans:
        hdr += f"{c.replace('can_',''):>16}"
    if b:
        hdr += f"{'|A-B|':>9}"
    print(hdr)
    max_lr = 0.0
    for i in range(6):
        row = f"  J{i+1:<5}{cmd[i]:>9.2f}"
        for c in cans:
            row += f"{fb[c][i]:>16.2f}"
        if b:
            d = abs(fb[a][i] - fb[b][i])
            max_lr = max(max_lr, d)
            row += f"{d:>9.2f}"
        print(row)
    if b:
        print(f"  → 两臂最大关节差 |A-B| = {max_lr:.2f}°")
    return max_lr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cans", default="can_left_slave,can_right_slave",
                    help="要对比的两条臂 (默认两条从臂)")
    ap.add_argument("--joints", default=None,
                    help="单个目标位姿, 6 个逗号分隔的度数, e.g. 0,20,-30,0,40,0")
    ap.add_argument("--sweep", action="store_true", help="扫过内置一串测试位姿")
    ap.add_argument("--speed", type=int, default=20, help="速度百分比 (默认 20, 安全慢速)")
    ap.add_argument("--hold", type=float, default=3.0, help="每个位姿持续下发的秒数")
    ap.add_argument("--execute", action="store_true", help="真正运动(否则只 dry-run)")
    args = ap.parse_args()

    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    if len(cans) != 2:
        print(f"⚠️ 需要正好两条臂做对比, 得到: {cans}")
        return
    sp = max(1, min(100, args.speed))

    # 目标位姿列表
    if args.sweep:
        poses = SWEEP_DEG
    elif args.joints:
        vals = [float(x) for x in args.joints.split(",")]
        if len(vals) != 6:
            print(f"⚠️ --joints 需要 6 个值, 得到 {len(vals)} 个"); return
        poses = [("custom", vals)]
    else:
        poses = [("zero", [0, 0, 0, 0, 0, 0])]

    # 连接读当前位姿
    arms = []
    for c in cans:
        try:
            p = C_PiperInterface_V2(c); p.ConnectPort(); time.sleep(0.4)
            arms.append((c, p))
            print(f"  {c:18} 当前 J={jdeg(p)}")
        except Exception as e:
            print(f"  {c:18} 连接失败: {e}")
    if len(arms) != 2:
        print("⚠️ 两条臂未全部连上, 无法对比。"); return

    if not args.execute:
        print(f"\n[dry-run] 将把 {[c for c,_ in arms]} 依次移动到以下位姿并对比 (速度 {sp}%):")
        for name, deg in poses:
            print(f"    {name:8} {deg}")
        print("确认无误后加 --execute 重跑。")
        return

    print(f"\n⚠️ 即将让 {[c for c,_ in arms]} 以 {sp}% 速度同步运动 {len(poses)} 个位姿。"
          f"请确认臂周围无障碍、手放急停旁。")
    if input("确认运动? [yes/no]: ") != "yes":
        print("已取消, 未运动。"); return

    # 使能两臂
    ready = []
    for c, p in arms:
        if enable(p, timeout=5.0):
            ready.append((c, p)); print(f"  {c:18} 已使能")
        else:
            print(f"  {c:18} ⚠️ 使能超时")
    if len(ready) != 2:
        print("⚠️ 两条臂未全部使能, 中止(不运动)。"); return

    # 逐位姿: 下发同一目标 → 到位 → 读回对比
    worst = 0.0
    for name, deg in poses:
        fb = goto_and_read(ready, deg, sp, args.hold)
        m = print_compare(name, deg, fb, [c for c, _ in ready])
        worst = max(worst, m)

    print(f"\n================ 汇总 ================")
    print(f"全程两臂最大单关节差异 = {worst:.2f}°")
    if worst < 1.0:
        print("→ 两臂一致性良好 (<1°)。")
    elif worst < 3.0:
        print("→ 存在轻微偏差 (1~3°), 可能是标定/零位差, 建议核对。")
    else:
        print("→ ⚠️ 偏差较大 (>3°), 两臂零位或机械标定不一致, 需重新标定。")
    print("(结束后卸力: python3 piper_tools/go_zero_all.py 或用你的 disable 脚本)")


if __name__ == "__main__":
    main()
