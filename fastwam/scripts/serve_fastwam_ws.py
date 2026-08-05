"""FastWAM 推理服务 —— openpi WebSocket 协议兼容版 (部署用)。

把 FastWAM 包成 openpi policy.infer(obs)->{"actions":[48,14]},用 openpi WebsocketPolicyServer 起服务。
于是**现有 kai0 policy_inference_node + start_autonomy 链原封不动**就能连(--mode websocket --ws-port),
与 gwp 同栈同参 —— 在线对比 apples-to-apples。结构镜像 giga_world_policy/scripts/serve_gwp_ws.py。

FastWAM 特性(与 gwp 不同):
  - **无 test-time 视频想象**:action expert 只读首帧 KV + 文本 + 因果自注意(infer_action),
    天然回避 gwp_ans 的闭环视频塌缩。
  - 归一化:**z-score**(dataset_stats.json 的 global_mean/std),非 gwp 的 q01/q99。
  - 文本:**预算 T5 context 缓存**(data/text_embeds_cache/visrobot01_fold/*.pt)。
  - 图像:3 相机拼 **[3,384,320]**(top 256x320 + 双腕 128x160),非 gwp 的 768x192。
  - 优化:opt_infer_action(ActionStepRunner,torch.compile+CUDA-graph,fp8 可选)~75ms@nfe4。

依赖:gwp_eval_env(torch 2.11 + hydra/modelscope/boto3 + openpi_client + websockets==15.0.1)。

用法(gwp_eval_env):
  cd fastwam && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src:scripts \
    /home/tim/gwp_eval_env/venv/bin/python scripts/serve_fastwam_ws.py \
      --weights runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v3/checkpoints/weights/step_025510.pt \
      --stats data/visrobot01_fold/dataset_stats.json --nfe 4 --opt_tier exact --port 8004
"""
import argparse, asyncio, http, glob, logging, pathlib, sys, time, traceback
import numpy as np
import torch
import torchvision.transforms.functional as TF

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

from eval_offline_fold import build_model, prep_image   # 复用同一套 load + 像素链
from opt_infer_action import ActionStepRunner, opt_infer_action

# 夹爪 frame 重映射 (旧 0.08m frame ckpt → 官方 0-70mm 真机)。openpi 侧的同一份实现,
# 依赖只有 numpy/os/logging, 在 gwp_eval_env 里可直接 import。KAI0_GRIPPER_DEPLOY_REMAP=0
# (默认) 时逐比特 no-op。见 docs/deployment/data_collection/gripper_calibration.md。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "kai0" / "src"))
from openpi.shared.gripper_remap import remap_gripper_raw  # noqa: E402

# node 发 bare 键 -> FastWAM 相机名
KMAP = {"top_head": "cam_high", "hand_left": "cam_left_wrist", "hand_right": "cam_right_wrist"}


def _to_hwc_u8(v):
    a = np.asarray(v)
    if a.ndim == 3 and a.shape[0] in (1, 3):     # CHW -> HWC (node 发 CHW)
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255.0).astype(np.uint8) if a.max() <= 1.5 else a.astype(np.uint8)
    return a[:, :, :3]


class FastwamPolicy:
    def __init__(self, args):
        self.args = args
        self.model = build_model(args.weights)
        dev, dt = self.model.device, self.model.torch_dtype
        # z-score stats。夹爪维走 openpi 同一份仿射重映射: 按 ckpt 自身训练量程
        # [q01,q99] 映到真机 [0,0.07]。action(反归一化) 与 state(归一化) 用同一组
        # 系数改写 mean/std → 出入两侧一致。KAI0_GRIPPER_DEPLOY_REMAP 未开时为 no-op。
        import json
        st = json.load(open(args.stats))
        _d = {"actions": st["action"]["default"], "state": st["state"]["default"]}
        _norm = {k: {"mean": np.array(v["global_mean"], np.float64),
                     "std": np.array(v["global_std"], np.float64),
                     "q01": np.array(v["global_q01"], np.float64),
                     "q99": np.array(v["global_q99"], np.float64)} for k, v in _d.items()}
        _before = {k: (_norm[k]["mean"][[6, 13]].copy(), _norm[k]["std"][[6, 13]].copy()) for k in _norm}
        remap_gripper_raw(_norm)
        for k in ("actions", "state"):
            m0, s0 = _before[k]
            m1, s1 = _norm[k]["mean"][[6, 13]], _norm[k]["std"][[6, 13]]
            if np.allclose(m0, m1) and np.allclose(s0, s1):
                print(f"[gripper-remap] {k}: OFF (no-op) — 夹爪维保持训练 frame "
                      f"q99={_norm[k]['q99'][[6, 13]].round(5).tolist()}", flush=True)
            else:
                print(f"[gripper-remap] {k}: ON  mean {m0.round(5).tolist()}→{m1.round(5).tolist()} "
                      f"std {s0.round(5).tolist()}→{s1.round(5).tolist()}", flush=True)
        self.a_mean = _norm["actions"]["mean"].astype(np.float32)
        self.a_std = _norm["actions"]["std"].astype(np.float32)
        self.s_mean = _norm["state"]["mean"].astype(np.float32)
        self.s_std = _norm["state"]["std"].astype(np.float32)
        # cached T5 context (single fold prompt)
        t5 = torch.load(glob.glob(args.t5_cache)[0], map_location="cpu", weights_only=False)
        ctx = t5["context"]; cmask = t5["mask"].bool()
        ctx = ctx.clone(); ctx[~cmask] = 0.0; cmask = torch.ones_like(cmask)
        if ctx.ndim == 2: ctx = ctx.unsqueeze(0)
        if cmask.ndim == 1: cmask = cmask.unsqueeze(0)
        self.ctx = ctx.to(dev, dt); self.cmask = cmask.to(dev)
        # opt engine
        if args.opt_tier == "fp8":
            from opt_infer_action import _swap_fp8
            n, mode = _swap_fp8(self.model.action_expert.blocks)
            _swap_fp8(self.model.action_expert.text_embedding); _swap_fp8(self.model.action_expert.time_embedding)
            _swap_fp8(self.model.action_expert.time_projection)
            print(f"[serve_fastwam] fp8/{mode} blocks={n}", flush=True)
        self.runner = ActionStepRunner(self.model)
        if args.opt_tier in ("exact", "fp8"):
            self.runner.compile_step("reduce-overhead")
        self.dev, self.dt = dev, dt
        print(f"[serve_fastwam] tier={args.opt_tier} nfe={args.nfe} (infer_action, no video imagination)", flush=True)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        a = self.args
        state = np.asarray(obs["state"], np.float32).reshape(-1)[:14]
        # 夹爪闩锁缓解 (2026-07-28 复盘, 默认关 = 逐比特回退)。
        # 训练数据 action[t] ≡ state[t] (relabel 约定) → 模型的夹爪输出基本是夹爪 proprio
        # 的回读。离线实测 (val, 240 chunk): 把夹爪 proprio 冻结在 1.5mm(闭) 时输出张开率
        # 52.6%→12.0%、与真值二值一致率 91.3%→65.5%(≈猜多数类); 冻结在 79mm(开) 时 75.6%。
        # 闭环下这是自锁: 夹爪停在哪就继续命令哪 → 永远抓不到。
        # 把闩锁读数换成中性常量可部分恢复视觉判别: 40mm → 张开率 50.3% / 一致率 77.3%
        # (26mm → 26.6% / 77.0%; 0mm 无效 11.6% / 65.1%)。仍低于真实 proprio 的 91.3%,
        # 故这是缓解而非根治 —— 根治要么让夹爪硬件真的动作(proprio 恢复真实), 要么重训。
        if a.gripper_proprio_neutral is not None:
            state = state.copy()
            state[6] = state[13] = np.float32(a.gripper_proprio_neutral)
        frames = {KMAP.get(k, k): _to_hwc_u8(v) for k, v in obs["images"].items()}
        img = prep_image(frames)                              # [3,384,320] in [-1,1]
        prop = torch.from_numpy((state - self.s_mean) / (self.s_std + 1e-8)).float()
        out = opt_infer_action(self.model, self.runner, context=self.ctx, context_mask=self.cmask,
                               image=img, proprio=prop, action_horizon=48,
                               num_inference_steps=a.nfe, seed=0)
        pa = out["action"].float().cpu().numpy() * (self.a_std + 1e-8) + self.a_mean   # [48,14] abs joints
        pa = pa.astype(np.float32)
        self._n = getattr(self, "_n", 0) + 1
        motion = float(np.abs(np.diff(pa, axis=0)).mean())
        if self._n <= 5 or self._n % 20 == 0:
            print(f"[infer #{self._n}] motion={motion:.4f} act[0,:7]={pa[0,:7].round(3).tolist()}", flush=True)
        if a.debug_dump_dir and self._n <= a.debug_dump_n:
            import os
            from PIL import Image
            os.makedirs(a.debug_dump_dir, exist_ok=True)
            ref = ((img.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()  # [384,320,3]
            Image.fromarray(ref).save(os.path.join(a.debug_dump_dir, f"ref_{self._n:03d}.png"))
            np.savez(os.path.join(a.debug_dump_dir, f"io_{self._n:03d}.npz"), state=state, action=pa)
        return {"actions": pa}


# --- openpi WebsocketPolicyServer 最小复制 (无 JAX) ---
class WebsocketPolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=None, metadata=None):
        self._policy, self._host, self._port, self._metadata = policy, host, port, (metadata or {})

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(self._handler, self._host, self._port, compression=None, max_size=None,
                                  ping_timeout=300, close_timeout=300, process_request=_health) as s:
            print(f"[serve_fastwam] ready, listening ws://{self._host}:{self._port}", flush=True)
            await s.serve_forever()

    async def _handler(self, ws):
        # A persistent inference server may be reused across real-robot trials.
        # Reset only session-scoped policy state when a new controller connects;
        # model weights and compiled graphs remain resident on the GPU.
        reset = getattr(self._policy, "reset_session", None)
        if callable(reset):
            reset()
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack(self._metadata))
        prev = None
        while True:
            try:
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(await ws.recv())
                ti = time.monotonic(); action = self._policy.infer(obs)
                action["server_timing"] = {"infer_ms": (time.monotonic() - ti) * 1000}
                if prev is not None: action["server_timing"]["prev_total_ms"] = prev * 1000
                await ws.send(packer.pack(action)); prev = time.monotonic() - t0
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="server error"); raise


def _health(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stats", default="data/visrobot01_fold/dataset_stats.json")
    ap.add_argument("--t5_cache", default="data/text_embeds_cache/visrobot01_fold/*.pt")
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--opt_tier", default="exact", choices=["eager", "exact", "fp8"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gripper_proprio_neutral", type=float, default=None,
                    help="把喂给模型的夹爪 proprio (state[6],state[13]) 覆盖为该常量 (米), "
                         "用于打破夹爪闩锁; 离线最优 0.04。缺省=不覆盖 (逐比特回退)。"
                         "仅当夹爪 proprio 卡死(硬件不动作)时才需要 — 见 infer() 内注释。")
    ap.add_argument("--debug_dump_dir", default="")
    ap.add_argument("--debug_dump_n", type=int, default=15)
    args = ap.parse_args()
    policy = FastwamPolicy(args)
    if args.warmup:
        dummy = {"state": np.zeros(14, np.float32),
                 "images": {k: np.zeros((3, 240, 320), np.uint8) for k in ("top_head", "hand_left", "hand_right")}}
        for i in range(int(args.warmup)):
            t = time.monotonic(); r = policy.infer(dummy)
            print(f"[serve_fastwam] warmup {i}: {r['actions'].shape} {(time.monotonic()-t)*1e3:.0f}ms", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port,
                          metadata={"model": "fastwam", "action_dim": 14, "action_horizon": 48}).serve_forever()


if __name__ == "__main__":
    main()
