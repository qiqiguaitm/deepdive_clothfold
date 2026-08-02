#!/usr/bin/env python
"""P0-D 跨-episode 换 hint 因果探针 (LaWAM dual2q).

背景/关键发现 (先读 docs/RESULTS_swap_hint_probe_2026-07-28.md):
  * dual2q 推理期 milestone hint = 模型**自预测** h_ms_pred(由当前 obs 经 VLM→_decode_ms_future
    生成), 经 CFG 进入 flow 动作生成 (lawam.py predict_action 行~1050, h_ms_star=h_ms_pred)。
  * provider(MilestoneTargetProvider / LMWM_MILESTONE_TARGET / h_ms_gt)与 LMWM_MS_RESIDUAL 只在
    **训练期** 生效; eval yaml 里 `unset LMWM_MILESTONE_TARGET`。
    → 因此"让 provider 查表返回别的 episode"在推理期是 **no-op**, 无法换 hint。
  * 唯一能"换 hint"的注入点 = 覆盖 predict_action 里的 h_ms_star。已在 lawam.py 加 env 门控:
      LMWM_SWAP_HINT=<npy>   : h_ms_star ← load(npy) 广播 (npy 已备成 ckpt 原生形态)
      LMWM_SWAP_HINT_ZERO=1  : h_ms_star ← 0 (无 hint 对照)
  * 残差 vs 绝对是 **ckpt 属性**(训练形态), 不是 eval 开关。故需两个 ckpt:
      绝对 dual2q (如 20260718_111535+lmwm_dual_2q / dual2q_shapeB), 注入 **绝对** milestone;
      残差 dual2q (rl4jj_2q_resid_noTs, 训练用 LMWM_MS_RESIDUAL=1), 注入 **残差** milestone。

本脚本两种模式:
  prep     : 从 target_compact.npz 备好各条件的注入特征 (正确/错误-ep 的 abs 与 resid 形态),
             并打印闭环 SR eval 的 launch 命令 (真正的裁决实验)。
  forward  : in-process 单帧前向敏感性诊断 (证明注入 hook 因果 live; 非 SR 裁决)。

用法:
  python pr_swap_hint_probe.py prep --out_dir <dir> [--wrong_ep N] [--pairs <pairs.npz>] [--compact <target_compact.npz>]
  CKPT=... FORM={abs,resid} [LMWM_CFG_GUIDANCE=5] python pr_swap_hint_probe.py forward
"""
from __future__ import annotations
import argparse, os, sys, zipfile
import numpy as np

REPO = os.environ.get("REPO", "/vePFS/tim/workspace/deepdive_kai0")
DEF_COMPACT = f"{REPO}/lmvla/lmwm/data/libero_rvalley/target_compact.npz"
DEF_PAIRS = f"{REPO}/lmvla/lmwm/data/libero_rvalley/pairs.npz"


def _load_npz_rows(path, key, indices):
    """Read selected rows from a stored NPZ member without loading multi-GB arrays."""
    with zipfile.ZipFile(path) as archive:
        member = archive.open(f"{key}.npy")
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
        if fortran:
            raise ValueError(f"{path}:{key} uses Fortran order; row seeking is unsupported")
        data_offset = member.tell()
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape)) * dtype.itemsize
        rows = []
        for index in indices:
            member.seek(data_offset + int(index) * row_bytes)
            raw = member.read(row_bytes)
            if len(raw) != row_bytes:
                raise EOFError(f"short read for {path}:{key}[{index}]")
            rows.append(np.frombuffer(raw, dtype=dtype).reshape(row_shape).copy())
    return rows


def cmd_prep(args):
    C = np.load(args.compact)
    eps, tfi = C["ep"], C["tgt_fi"]
    os.makedirs(args.out_dir, exist_ok=True)

    # 错误-episode(默认取 ep 索引最大者, 与任一评测任务无关)。也可 --wrong_ep 指定。
    if args.wrong_ep is not None:
        rows = np.where(eps == args.wrong_ep)[0]
        if len(rows) == 0:
            sys.exit(f"[prep] wrong_ep={args.wrong_ep} 无 milestone 行")
    else:
        rows = np.where(eps == int(eps.max()))[0]
    j = int(rows[0])
    wrong_ep = int(eps[j])

    # 身份无关的错误残差 = 同一错误 ep 的两个 milestone 帧之差(纯"变化方向", 场景身份相消)
    k = int(rows[1]) if len(rows) > 1 else int(np.where(eps == int(eps.min()))[0][0])
    feat_j, feat_k = _load_npz_rows(args.compact, "feat", [j, k])
    foreign_abs = feat_j.astype(np.float32)                       # [256,768] 绝对 milestone(错误 ep)
    foreign_resid = feat_j.astype(np.float32) - feat_k.astype(np.float32)

    # 固定 token permutation 保留每个 token 的完整通道向量、总体范数和边际分布，
    # 但破坏 16x16 patch 的空间对应，用作 shuffled-content 控制。
    permutation = np.random.default_rng(args.shuffle_seed).permutation(foreign_abs.shape[0])
    shuffled_abs = foreign_abs[permutation].copy()
    shuffled_resid = foreign_resid[permutation].copy()

    np.save(os.path.join(args.out_dir, "other_abs.npy"), foreign_abs)
    np.save(os.path.join(args.out_dir, "other_resid.npy"), foreign_resid)
    np.save(os.path.join(args.out_dir, "shuffled_abs.npy"), shuffled_abs)
    np.save(os.path.join(args.out_dir, "shuffled_resid.npy"), shuffled_resid)
    # Backward-compatible aliases for existing launch recipes.
    np.save(os.path.join(args.out_dir, "wrong_abs.npy"), foreign_abs)
    np.save(os.path.join(args.out_dir, "wrong_resid.npy"), foreign_resid)
    zero_flag = os.path.join(args.out_dir, "USE_LMWM_SWAP_HINT_ZERO=1")
    open(zero_flag, "w").close()

    print(f"[prep] wrong_ep={wrong_ep} rows={rows.tolist()[:4]}...")
    print(f"[prep] wrong_abs.npy   ‖·‖={np.linalg.norm(foreign_abs):.1f}  (绝对 ckpt 的错误 hint)")
    print(f"[prep] wrong_resid.npy ‖·‖={np.linalg.norm(foreign_resid):.1f}  (残差 ckpt 的错误 hint)")
    print(f"[prep] shuffled token permutation seed={args.shuffle_seed}")
    print()
    print("=" * 78)
    print("闭环 SR eval launch (真正裁决实验) —— 每 ckpt × {正确=native, 错误注入, 无hint}:")
    print("  绝对 ckpt (无 LMWM_MS_RESIDUAL 训练):")
    print("    ① 对照(正确/自预测):  (不设 SWAP env) run_libero_benchmark.sh $ABS_CKPT")
    print(f"    ② 错误绝对 hint:      LMWM_SWAP_HINT={args.out_dir}/wrong_abs.npy   run_...  $ABS_CKPT")
    print("    ④ 无 hint:            LMWM_SWAP_HINT_ZERO=1                         run_...  $ABS_CKPT")
    print("  残差 ckpt (LMWM_MS_RESIDUAL=1 训练):")
    print("    ① 对照:               (不设 SWAP env) run_libero_benchmark.sh $RESID_CKPT")
    print(f"    ③ 错误残差 hint:      LMWM_SWAP_HINT={args.out_dir}/wrong_resid.npy run_...  $RESID_CKPT")
    print("    ④ 无 hint:            LMWM_SWAP_HINT_ZERO=1                         run_...  $RESID_CKPT")
    print("  公共 env: LMWM_DUAL=1 LMWM_DUAL_2Q=1 LMWM_SWAP_TEACHER=1 LMWM_CKPT=<lmwm.pt>")
    print("            LMWM_ADAPTER_DIR=$REPO/lmvla/lmwam/adapter  (unset LMWM_MILESTONE_TARGET)")
    print("            SUITES=<套件> NUM_TRIALS_PER_TASK=20 MAX_TASKS=<t5/t6/饱和> MUJOCO_GL=egl")
    print("=" * 78)


def cmd_forward(args):
    import torch
    os.environ.setdefault("LMWM_DUAL", "1"); os.environ.setdefault("LMWM_DUAL_2Q", "1")
    os.environ.setdefault("LMWM_SWAP_TEACHER", "1")
    os.environ.setdefault("LMWM_CKPT", f"{REPO}/lmvla/lmwm/checkpoints/lmwm_libero_rvalley/lmwm.pt")
    os.environ.setdefault("LMWM_ADAPTER_DIR", f"{REPO}/lmvla/lmwam/adapter")
    os.environ.pop("LMWM_MILESTONE_TARGET", None); os.environ.pop("LMWM_TARGET_COMPACT", None)
    sys.path.insert(0, f"{REPO}/lmvla/lawam")
    from starVLA.model.framework.base_framework import baseframework

    ckpt = os.environ["CKPT"]; form = os.environ.get("FORM", "abs")
    model = baseframework.from_pretrained(ckpt).to("cuda:0").to(torch.bfloat16).eval()

    rng = np.random.RandomState(0)
    ex = {"primary_image": [rng.randint(0, 255, (256, 256, 3), np.uint8)],
          "wrist_image": [rng.randint(0, 255, (256, 256, 3), np.uint8)],
          "lang": args.prompt, "state": np.zeros((7,), np.float32),
          "embodiment_id": 25, "action_hz": 20.0}

    def run(seed=1234):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = model.predict_action([ex])
        return np.asarray(out["normalized_actions"], np.float32).reshape(-1)

    C = np.load(args.compact); feat, eps = C["feat"], C["ep"]
    j = int(np.where(eps == int(eps.max()))[0][0])
    tmp = args.out_dir; os.makedirs(tmp, exist_ok=True)
    np.save(f"{tmp}/f_abs.npy", feat[j].astype(np.float32))
    k = int(np.where(eps == int(eps.min()))[0][0])
    np.save(f"{tmp}/f_resid.npy", (feat[j].astype(np.float32) - feat[k].astype(np.float32)))

    os.environ.pop("LMWM_SWAP_HINT", None); os.environ.pop("LMWM_SWAP_HINT_ZERO", None)
    a0 = run(); noise = np.linalg.norm(a0 - run())
    os.environ["LMWM_SWAP_HINT"] = f"{tmp}/f_abs.npy" if form == "abs" else f"{tmp}/f_resid.npy"
    d_f = np.linalg.norm(run() - a0)
    os.environ["LMWM_SWAP_HINT"] = f"{tmp}/f_resid.npy" if form == "abs" else f"{tmp}/f_abs.npy"
    d_fo = np.linalg.norm(run() - a0)
    os.environ.pop("LMWM_SWAP_HINT", None); os.environ["LMWM_SWAP_HINT_ZERO"] = "1"
    d_z = np.linalg.norm(run() - a0)
    b = np.linalg.norm(a0) + 1e-6
    print(f"\nckpt={os.path.basename(os.path.dirname(os.path.dirname(ckpt)))} form={form} "
          f"cfg={os.environ.get('LMWM_CFG_GUIDANCE','1.0')}")
    print(f"  ||a||={b:.3f} noise={noise:.3f}({100*noise/b:.1f}%) "
          f"Δforeign({form})={d_f:.3f}({100*d_f/b:.1f}%) "
          f"Δforeign(other)={d_fo:.3f}({100*d_fo/b:.1f}%) Δzero={d_z:.3f}({100*d_z/b:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prep"); p.add_argument("--out_dir", required=True)
    p.add_argument("--wrong_ep", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=2027)
    p.add_argument("--compact", default=DEF_COMPACT); p.add_argument("--pairs", default=DEF_PAIRS)
    p.set_defaults(func=cmd_prep)
    f = sub.add_parser("forward"); f.add_argument("--out_dir", default="/tmp/swap_probe")
    f.add_argument("--compact", default=DEF_COMPACT)
    f.add_argument("--prompt", default="put both the alphabet soup and the tomato sauce in the basket")
    f.set_defaults(func=cmd_forward)
    a = ap.parse_args(); a.func(a)
