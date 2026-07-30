#!/usr/bin/env python3
"""一键刷新 4 个机械臂的固件主从角色 (MasterSlaveConfig)。

支持两种模式:
  all_slave   — 4 臂全部设为从臂 (0xFC), 用于 v2 DAgger / autonomy / arm_master_servo
  master_slave — 主臂→0xFA (leader), 从臂→0xFC (follower), 用于 v0 官方主从遥操

四臂 CAN 端口 (来自 config/pipers.yml):
  can_left_mas   — 左主臂 (示教手柄)
  can_left_slave — 左从臂 (执行器)
  can_right_mas  — 右主臂 (示教手柄)
  can_right_slave— 右从臂 (执行器)

⚠️ 改完角色【必须把臂断电重启】才生效 (SDK 规定: MasterSlaveConfig 写入固件 NVRAM,
   上电时读取, 运行时修改不立即切换)。

用法:
  python3 flash_all_arm_roles.py all_slave       # 4 臂全从臂 → v2/DAgger
  python3 flash_all_arm_roles.py master_slave    # 2 主 2 从 → v0 官方遥操
  python3 flash_all_arm_roles.py --dry-run all_slave   # 仅预览, 不写入
"""

import argparse
import sys
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    raise RuntimeError("piper_sdk 未安装 (pip install piper_sdk)")

# ── 端口定义 (与 config/pipers.yml 对齐) ──
MASTER_CANS: tuple[str, ...] = ("can_left_mas", "can_right_mas")
SLAVE_CANS: tuple[str, ...]  = ("can_left_slave", "can_right_slave")
ALL_CANS: tuple[str, ...]    = MASTER_CANS + SLAVE_CANS

# ── 角色定义 ──
ROLE_MASTER = 0xFA  # leader — 拖拽示教, 固件级主从联动
ROLE_SLAVE  = 0xFC  # follower — 接收 JointCtrl, CAN_CTRL 可控
N_WRITES    = 5      # 固件 NVRAM 写入重试次数 (与 flash_master_to_follower.py 一致)


def flash_one(port: str, role: int) -> bool:
    """写入单个臂的 MasterSlaveConfig, 返回是否成功。"""
    role_name = "主臂 0xFA" if role == ROLE_MASTER else "从臂 0xFC"
    try:
        arm = C_PiperInterface_V2(can_name=port, judge_flag=False)
        arm.ConnectPort()
        time.sleep(0.4)

        for i in range(N_WRITES):
            arm.MasterSlaveConfig(role, 0x00, 0x00, 0x00)
            time.sleep(0.25)

        print(f"  ✓ {port:18} → {role_name}  ({N_WRITES} 次写入完成)")
        return True
    except Exception as e:
        print(f"  ✗ {port:18} 失败: {e}")
        return False


def show_summary(mode: str, dry: bool) -> None:
    """打印计划操作摘要。"""
    if mode == "all_slave":
        roles = [(c, ROLE_SLAVE) for c in ALL_CANS]
        title = "模式: all_slave — 4 臂全部设为从臂 (0xFC)"
        desc = "适用: v2 DAgger / autonomy / arm_master_servo"
    elif mode == "master_slave":
        roles = [(c, ROLE_MASTER) for c in MASTER_CANS] + \
                [(c, ROLE_SLAVE)  for c in SLAVE_CANS]
        title = "模式: master_slave — 2 主臂 0xFA + 2 从臂 0xFC"
        desc = "适用: v0 官方主从遥操 (start_teleop_v0.sh / start_data_collect_v0.sh)"
    else:
        raise ValueError(f"未知模式: {mode}")

    print("=" * 60)
    print(f"  {title}")
    print(f"  {desc}")
    print("=" * 60)
    print()
    if dry:
        print(">>> DRY RUN (仅预览, 不写入) <<<")
        print()

    for port, role in roles:
        tag = "主臂 0xFA" if role == ROLE_MASTER else "从臂 0xFC"
        print(f"  {port:18}  →  {tag}")
    print()

    return roles


def print_post_instructions(mode: str) -> None:
    """打印断电重启后的验证/启动指令。"""
    print()
    print("=" * 60)
    print("  ⚠️ 下一步 (物理操作):")
    print("=" * 60)
    print("  1. 拔掉四臂 24V 电源 (从臂端拔, 不要拔适配器端)")
    print("  2. 等待 10 秒")
    print("  3. 重新插上 24V 电源")
    print("  4. 等待 5 秒让固件启动")
    print()
    if mode == "all_slave":
        print("  验证:")
        print("    所有臂均可接受 JointCtrl → v2 遥操 / autonomy")
        print("    启动: ./start_scripts/start_data_collect.sh")
    else:
        print("  验证:")
        print("    主臂 0xFA 可拖拽示教 → 从臂 0xFC 跟随")
        print("    启动: ./start_scripts/start_data_collect_v0.sh")
        print("    或独立测试遥操: ./start_scripts/kai/start_teleop_v0.sh")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="一键刷新全部 4 臂主从角色",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  %(prog)s all_slave         # → 4 臂全从臂 (v2/DAgger)
  %(prog)s master_slave      # → 2 主 2 从 (v0 官方遥操)
  %(prog)s --dry-run all_slave  # 预览不写入
        """,
    )
    ap.add_argument(
        "mode", choices=["all_slave", "master_slave"],
        help="all_slave: 四臂全从臂; master_slave: 2主2从",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="仅打印将要写入的角色, 不实际连接臂",
    )
    args = ap.parse_args()

    # 1. 打印计划
    roles = show_summary(args.mode, args.dry_run)

    if args.dry_run:
        print("Dry run 完成 — 未写入任何臂。")
        print_post_instructions(args.mode)
        return 0

    # 2. 确认
    try:
        resp = input("确认写入以上角色? [yes/no]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return 1
    if resp != "yes":
        print("已取消, 未写入。")
        return 0

    print()

    # 3. 逐臂写入
    ok, fail = 0, 0
    for port, role in roles:
        if flash_one(port, role):
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)  # CAN 总线间隔

    # 4. 结果
    print()
    print(f"结果: {ok} 成功, {fail} 失败")
    if fail > 0:
        print("⚠️  有失败项, 请检查 CAN 连接后重试")

    print_post_instructions(args.mode)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
