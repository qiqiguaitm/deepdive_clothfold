#!/usr/bin/env python3
"""从臂健康监控 + 问题状态记录器.

只读连接 CAN (与 arm_teleop_node 并存, socketcan RX 共享, 不发任何控制帧, 不干扰遥操).
高频轮询每个关节 指令(GetArmJointCtrl) vs 反馈(GetArmJointMsgs) / 温度 / 使能 / 过温过流
/ err_code / CAN isOk, 检测到异常时把详细快照 + 前若干秒滚动上下文写入日志文件.

判据 (针对本次排查的"跟随但缓慢下滑"设计):
  - 下滑/欠力: 指令基本不动(holding) 但 |指令-反馈| 超阈值 → 守不住位
               (遥操移动时指令在变→holding=False, 不会误报正常跟踪滞后)
  - J2式假象规避: 只在 holding 且背离大时报, 关节合法停在 0° 不会触发
  - 电机过温 / 过流 / 掉使能 (读固件 foc_status 位)
  - 温度超阈值 (motor_temp)
  - CAN 掉线 (isOk=False / Hz=0) / err_code!=0

用法:
  python3 piper_tools/arm_monitor.py                          # 默认 can_left_slave
  python3 piper_tools/arm_monitor.py --cans can_left_slave,can_right_slave
  python3 piper_tools/arm_monitor.py --hz 20 --hold-delta-deg 0.5 --temp-alarm 75

日志: piper_tools/logs/arm_monitor.log  (出问题后把这个文件发我)
Ctrl+C 停止.
"""
import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    sys.exit("piper_sdk 未安装 (pip install piper_sdk): %s" % e)

MDEG = 1000.0  # 原始单位 0.001deg → 1° = 1000


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class ArmMonitor:
    def __init__(self, can: str, args, log):
        self.can = can
        self.args = args
        self.log = log
        self.p = C_PiperInterface_V2(can)
        self.p.ConnectPort()
        time.sleep(0.4)
        self.prev_cmd = [None] * 6
        # 滚动上下文: 保存最近 context_s 秒的每帧摘要, 出事时回放看下滑起点
        self.hist = deque(maxlen=max(1, int(args.hz * args.context_s)))
        self.in_problem = False
        self.problem_started = 0.0
        self.last_hb = 0.0
        self.frames = 0
        self.cand_since = None  # 下滑类异常连续起始时间 (用于 sustain 确认)

    def sample(self):
        """读一帧, 返回结构化状态 (读失败抛异常由 caller 兜)."""
        js = self.p.GetArmJointMsgs()
        j = js.joint_state
        ctl = self.p.GetArmJointCtrl().joint_ctrl
        st = self.p.GetArmStatus().arm_status
        low = self.p.GetArmLowSpdInfoMsgs()
        try:
            ok = self.p.isOk()
        except Exception:
            ok = None
        rows = []
        for i in range(1, 7):
            m = getattr(low, f"motor_{i}")
            f = m.foc_status
            rows.append({
                "i": i,
                "cmd": getattr(ctl, f"joint_{i}"),
                "fb": getattr(j, f"joint_{i}"),
                "temp": m.motor_temp,
                "en": bool(f.driver_enable_status),
                "ot": bool(f.motor_overheating),
                "oc": bool(f.driver_overcurrent),
            })
        return {
            "ok": ok,
            "hz": getattr(js, "Hz", None),
            "err": int(st.err_code),
            "ctrl_mode": int(st.ctrl_mode),
            "rows": rows,
        }

    def detect(self, s):
        """返回 (imm, sus): imm=硬故障立即报, sus=下滑类需持续确认."""
        imm: list[str] = []
        sus: list[str] = []
        if s["ok"] is False:
            imm.append("CAN isOk=False (总线掉线/bus-off)")
        if s["hz"] == 0:
            imm.append("Hz=0 (无帧)")
        if s["err"]:
            imm.append(f"err_code=0x{s['err']:04x}")
        hold_d = self.args.hold_delta_deg * MDEG   # 保持时可容忍的最大偏差
        move_th = self.args.move_thresh_deg * MDEG  # 指令变化超此值视为"在动"
        for r in s["rows"]:
            i = r["i"]
            pc = self.prev_cmd[i - 1]
            holding = pc is not None and abs(r["cmd"] - pc) < move_th
            d = r["cmd"] - r["fb"]
            # 下滑/欠力: 指令没在动, 却守不住位 → 需持续确认(滤掉甩腕瞬时滞后)
            if holding and abs(d) > hold_d:
                sus.append(
                    f"J{i} 保持偏差 {d/MDEG:+.2f}° "
                    f"(指令{r['cmd']/MDEG:.1f}° 反馈{r['fb']/MDEG:.1f}°)"
                )
            if r["ot"]:
                imm.append(f"J{i} 电机过温(固件位)")
            if r["oc"]:
                imm.append(f"J{i} 过流(固件位)")
            if not r["en"]:
                imm.append(f"J{i} 失能")
            if r["temp"] >= self.args.temp_alarm:
                imm.append(f"J{i} 温度 {r['temp']}°C ≥{self.args.temp_alarm}")
            self.prev_cmd[i - 1] = r["cmd"]
        return imm, sus

    def fmt_snapshot(self, s) -> str:
        head = (f"    ok={s['ok']} Hz={s['hz']} ctrl_mode=0x{s['ctrl_mode']:02x} "
                f"err=0x{s['err']:04x}")
        lines = [head]
        for r in s["rows"]:
            d = r["cmd"] - r["fb"]
            lines.append(
                f"    J{r['i']}: 指令{r['cmd']:>8} 反馈{r['fb']:>8} Δ{d:>6} "
                f"温度{r['temp']:>3} 使能{'Y' if r['en'] else 'N'} "
                f"过温{'Y' if r['ot'] else '-'} 过流{'Y' if r['oc'] else '-'}"
            )
        return "\n".join(lines)

    def emit(self, text: str):
        line = f"[{ts()}][{self.can}] {text}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def step(self):
        try:
            s = self.sample()
        except Exception as e:  # noqa: BLE001
            self.emit(f"读取异常: {e}")
            return
        self.frames += 1
        # 存滚动上下文 (紧凑: 每关节 Δ + 温度)
        digest = {"t": ts(), "ok": s["ok"], "hz": s["hz"],
                  "d": [r["cmd"] - r["fb"] for r in s["rows"]],
                  "temp": [r["temp"] for r in s["rows"]]}
        self.hist.append(digest)

        imm, sus = self.detect(s)
        nowt = time.time()
        # 下滑类需连续保持 sustain_s 秒才确认 (滤掉甩腕瞬时滞后)
        if sus:
            if self.cand_since is None:
                self.cand_since = nowt
            sus_ok = (nowt - self.cand_since) >= self.args.sustain_s
        else:
            self.cand_since = None
            sus_ok = False
        probs = list(imm) + (sus if sus_ok else [])

        if probs and not self.in_problem:
            # 问题起始: 打上下文回放 + 完整快照
            self.in_problem = True
            self.problem_started = nowt
            self.emit("★★★ 问题出现 ★★★ " + "; ".join(probs))
            self.emit(f"  --- 前 {self.args.context_s}s 上下文 (Δ单位0.001°, [J1..J6]) ---")
            for h in list(self.hist)[:-1]:
                self.emit(f"    {h['t']} ok={h['ok']} Hz={h['hz']} Δ={h['d']} 温度={h['temp']}")
            self.emit("  --- 当前完整快照 ---")
            self.emit(self.fmt_snapshot(s))
        elif probs and self.in_problem:
            # 问题持续: 降频记录 (每 1s)
            if nowt - self.last_hb >= 1.0:
                self.last_hb = nowt
                self.emit("… 问题持续: " + "; ".join(probs))
        elif not probs and self.in_problem:
            # 恢复
            dur = nowt - self.problem_started
            self.in_problem = False
            self.emit(f"✓ 恢复正常 (问题持续 {dur:.1f}s)")
        else:
            # 健康: 周期心跳
            if nowt - self.last_hb >= self.args.heartbeat_s:
                self.last_hb = nowt
                temps = [r["temp"] for r in s["rows"]]
                maxd = max(abs(r["cmd"] - r["fb"]) for r in s["rows"])
                self.emit(f"心跳 健康 ok={s['ok']} Hz={s['hz']} 温度={temps} 最大Δ={maxd}")


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="从臂健康监控 + 问题状态记录 (只读, 不干扰遥操)")
    ap.add_argument("--cans", default="can_left_slave",
                    help="逗号分隔的 CAN 口 (默认 can_left_slave)")
    ap.add_argument("--hz", type=float, default=20.0, help="轮询频率 (默认 20)")
    ap.add_argument("--hold-delta-deg", type=float, default=3.0,
                    help="保持状态下判为下滑的偏差阈值 ° (默认 3.0). "
                         "注: J2 停在 0° 附近时反馈读精确 0 是已知无害现象(顶多~2°偏差), "
                         "阈值 3° 可避开它, 只抓肉眼可见的真实下滑(5~30°)")
    ap.add_argument("--move-thresh-deg", type=float, default=0.3,
                    help="指令变化超此值视为在动, 不判下滑 ° (默认 0.3)")
    ap.add_argument("--sustain-s", type=float, default=1.0,
                    help="下滑偏差需连续保持多少秒才确认为问题 (默认 1.0, "
                         "滤掉甩腕的瞬时跟踪滞后; 硬故障如过温/掉线立即报不受此限)")
    ap.add_argument("--temp-alarm", type=int, default=75,
                    help="电机温度告警阈值 °C (默认 75; 固件过温位优先)")
    ap.add_argument("--context-s", type=float, default=4.0,
                    help="问题出现时回放的前置上下文秒数 (默认 4)")
    ap.add_argument("--heartbeat-s", type=float, default=15.0,
                    help="健康时心跳间隔秒 (默认 15)")
    ap.add_argument("--log", default=None, help="日志路径 (默认 piper_tools/logs/arm_monitor.log)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    log_path = args.log or os.path.join(here, "logs", "arm_monitor.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a", buffering=1)

    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    header = (f"\n===== 监控启动 {ts()} =====\n"
              f"  CAN: {cans}  hz={args.hz}  下滑阈值={args.hold_delta_deg}° "
              f"温度告警={args.temp_alarm}°C  日志={log_path}")
    print(header, flush=True)
    log.write(header + "\n")
    log.flush()

    mons = []
    for c in cans:
        try:
            mons.append(ArmMonitor(c, args, log))
            print(f"  连接 {c} 成功", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  连接 {c} 失败: {e}", flush=True)
    if not mons:
        return 1

    period = 1.0 / args.hz
    try:
        while True:
            t0 = time.time()
            for m in mons:
                m.step()
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        for m in mons:
            m.emit("监控停止 (Ctrl+C)")
        print("\n已停止, 日志:", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
