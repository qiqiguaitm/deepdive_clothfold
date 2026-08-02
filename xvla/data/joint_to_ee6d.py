"""Convert kai0/vis LeRobot parquets: 14D joint → 20D EE6D action+state.

EE6D layout per arm (10D): xyz (3, m) + Rot6D (6, first two cols of R 3x3) + gripper (1).
Two arms → 20D total.

Reads PiperFK with 2° j2/j3 offset (0x01 DH mode) for accurate piper kinematics.
Output preserves dataset structure (videos symlinks reused, parquets rewritten,
info.json features updated to 20D).
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

# Per-process FK instance (init lazily)
_FK = None
# Global flag for continuous gripper mode (set before convert_parquet)
_CONTINUOUS_GRIPPER = False
# Continuous-gripper alpha normalization bounds (m). alpha = clip((open-g)/(open-close),0,1).
#   v3 default (0.08, 0.04): 只用 [0.04,0.08] 半带 → 抓取区 [0,0.04] 全饱和到 alpha=1
#     (丢挤压力度 + 部署无法还原力), 且中段占多数 → MSE 塌成均值。
#   v4 (0.08, 0.0): 数据驱动满行程 — open=state p99.9(物理开度上限 ~80mm),
#     close=state p0(物理全闭 0mm, 兼容 action 挤压负值经 clip)。抓取区 [0,0.04] 展到
#     alpha [0.5,1.0] 有分辨率; 部署逆映射 g=0.08*(1-alpha) 忠实 (alpha=1→0 力闭合,
#     alpha=0.5→0.04 物宽), train==deploy 无偏移。见 2026-07-04 Task_P 夹爪分位数分析。
_G_OPEN_BOUND = 0.08
_G_CLOSE_BOUND = 0.04


def get_fk():
    global _FK
    if _FK is None:
        _FK = C_PiperForwardKinematics(0x01)  # 2° j2/j3 offset
    return _FK


def joint_to_ee6d_row(q14: np.ndarray) -> np.ndarray:
    """14D (7+7: 6 joints + 1 gripper per arm) → 20D EE6D."""
    fk = get_fk()
    out = np.zeros(20, dtype=np.float32)
    for arm in range(2):
        q6 = q14[arm*7 : arm*7+6]  # 6 joints (rad)
        gripper = q14[arm*7 + 6]   # gripper (m or rad)
        result = fk.CalFK(list(q6.astype(np.float64)))
        ee = result[-1]  # last link = end effector
        xyz_m = np.array(ee[:3], dtype=np.float32) / 1000.0  # mm → m
        rpy_deg = np.array(ee[3:], dtype=np.float32)
        R = Rotation.from_euler("xyz", np.radians(rpy_deg)).as_matrix()
        # Rot6D = first two columns, row-major [r00,r01,r10,r11,r20,r21] — matches
        # X-VLA upstream quat_to_rotate6d / deploy rotation.py. (was .T.flatten() = block, a bug)
        rot6d = R[:, :2].flatten().astype(np.float32)
        out[arm*10 : arm*10+3] = xyz_m
        out[arm*10+3 : arm*10+9] = rot6d
        if _CONTINUOUS_GRIPPER:
            # Map physical gripper (m) to normalized alpha ∈ [0, 1] via data-driven bounds:
            #   alpha=0 → _G_OPEN_BOUND  (fully open)
            #   alpha=1 → _G_CLOSE_BOUND (full close / grasp)
            # Formula: alpha = clip((open - g) / (open - close), 0, 1). Bounds set by
            # convert_dataset (v4 = 0.08/0.0 满行程无饱和; 见模块顶部注释)。
            alpha = np.float32(np.clip((_G_OPEN_BOUND - gripper) / (_G_OPEN_BOUND - _G_CLOSE_BOUND),
                                       0.0, 1.0))
            out[arm*10+9] = alpha
        else:
            # Binarize gripper to {0,1} — action_hub uses BCEWithLogitsLoss on this channel.
            # Matches upstream AIRAgilex (real_world.py): raw*50<1.0 → 1 (closed). gripper in meters.
            out[arm*10+9] = np.float32(gripper * 50.0 < 1.0)
    return out

def convert_parquet(in_path: Path, out_path: Path) -> int:
    """Convert one parquet, return # rows."""
    if out_path.exists():
        return -1  # skip if already exists
    t = pq.read_table(in_path)
    df = t.to_pandas()
    n = len(df)
    state_arr = np.stack([np.array(s, dtype=np.float32) for s in df["observation.state"]])
    action_arr = np.stack([np.array(a, dtype=np.float32) for a in df["action"]])
    state_ee6d = np.stack([joint_to_ee6d_row(s) for s in state_arr])
    action_ee6d = np.stack([joint_to_ee6d_row(a) for a in action_arr])
    df["observation.state"] = list(state_ee6d)
    df["action"] = list(action_ee6d)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), out_path)
    return n

def convert_dataset(in_root: Path, out_root: Path, num_workers: int = 16,
                    continuous_gripper: bool = False,
                    g_open_bound: float = 0.08, g_close_bound: float = 0.04):
    """Convert all parquets in in_root → out_root, mirror video/meta structure.
    Globals set BEFORE the ProcessPoolExecutor so forked workers inherit them (Linux fork)."""
    global _CONTINUOUS_GRIPPER, _G_OPEN_BOUND, _G_CLOSE_BOUND
    _CONTINUOUS_GRIPPER = continuous_gripper
    _G_OPEN_BOUND, _G_CLOSE_BOUND = float(g_open_bound), float(g_close_bound)
    # Resolve before creating the videos symlink. A relative in_root embedded verbatim in a
    # symlink is interpreted relative to out_root and silently produces all-black training data.
    in_root, out_root = Path(in_root).resolve(), Path(out_root).resolve()
    parquets = sorted(in_root.rglob("data/**/*.parquet"))
    mode = (f"continuous alpha bounds open={g_open_bound} close={g_close_bound}"
            if continuous_gripper else "binary {0,1}")
    print(f"[{in_root.name}] {len(parquets)} parquets to convert (gripper={mode})")

    # Mirror meta + symlink videos
    if (in_root / "meta").exists() and not (out_root / "meta").exists():
        shutil.copytree(in_root / "meta", out_root / "meta")
        # Update info.json to mark 20D state/action
        info_path = out_root / "meta" / "info.json"
        info = json.load(open(info_path))
        if "features" in info:
            for k in ("observation.state", "action"):
                if k in info["features"]:
                    info["features"][k]["shape"] = [20]
                    if info["features"][k].get("names"):
                        info["features"][k]["names"] = [
                            f"{side}_{x}" for side in ("left", "right")
                            for x in ("x", "y", "z", "r00", "r01", "r10", "r11", "r20", "r21", "grip")
                        ]
        # Record gripper alpha bounds so downstream (deploy sidecar) reads the EXACT
        # inverse map — single source of truth, prevents train/deploy mismatch.
        if continuous_gripper:
            info["gripper_alpha_bounds"] = {"open_m": _G_OPEN_BOUND, "close_m": _G_CLOSE_BOUND,
                                            "formula": "alpha=clip((open-g)/(open-close),0,1)"}
        json.dump(info, open(info_path, "w"), indent=2)
    if (in_root / "videos").exists() and not (out_root / "videos").exists():
        os.symlink(in_root / "videos", out_root / "videos")
    if (in_root / "norm_stats.json").exists():
        # Skip norm_stats (different scale for EE6D - recompute later)
        pass

    # Parallel parquet conversion
    done, total_rows = 0, 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(convert_parquet, p, out_root / p.relative_to(in_root)): p
            for p in parquets
        }
        for f in as_completed(futures):
            n = f.result()
            done += 1
            if n > 0:
                total_rows += n
            if done % 100 == 0:
                print(f"  [{in_root.name}] {done}/{len(parquets)} done, {total_rows} rows")
    print(f"[{in_root.name}] DONE: {done} parquets, {total_rows} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--continuous", action="store_true",
                    help="Preserve continuous gripper as normalized alpha instead of binarizing to {0,1}. "
                         "Use with ee6d_alpha action space (MSE loss) for pick-and-place tasks.")
    ap.add_argument("--g_open_bound", type=float, default=0.08,
                    help="连续夹爪 alpha=0 对应的物理开度 (m)。默认 0.08 (Task_P state p99.9 物理开度上限)。")
    ap.add_argument("--g_close_bound", type=float, default=0.04,
                    help="连续夹爪 alpha=1 对应的物理闭合 (m)。v3=0.04(半带,已废); "
                         "v4=0.0(满行程, state p0 全闭, 抓取区有分辨率 + 部署力闭合)。")
    args = ap.parse_args()
    convert_dataset(Path(args.in_dir), Path(args.out_dir), args.workers,
                    continuous_gripper=args.continuous,
                    g_open_bound=args.g_open_bound, g_close_bound=args.g_close_bound)
