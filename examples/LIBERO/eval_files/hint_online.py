#!/usr/bin/env python
"""在线 LMWM hint 计算 (A1/A2 eval): live 图像 → 编码器(DINOv3/So400m) → grid → LMWM grid_to_hint → hint[DIN].
复刻 extract_v2p1_grid.py(编码器) + export_pi05_hint.py(grid_to_hint), 使在线 hint 与训练离线 hint 同分布.

校验: python hint_online.py --encoder dinov3-base  (对训练帧在线算 hint, 比对离线 hint.npz)
"""
import os, sys
import numpy as np
import torch

# REPO 从本文件位置自动推导 (REPO/lmvla/lawam/examples/LIBERO/eval_files/hint_online.py), 跨集群通用
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
LMWM_CKPT = {  # 与 pi05_hint/*/_env.json 一致
    "dinov3-base": f"{REPO}/lmvla/lmwm/checkpoints/lmwm_libero_rvalley/lmwm.pt",
    "so400m":      f"{REPO}/lmvla/lmwm/checkpoints/lmwm_libero_so400m/lmwm.pt",
}
PGRID = 16


def _load_lmwm(ckpt_path, device):
    sys.path.insert(0, f"{REPO}/lmvla/lmwm/scripts")
    from p1_train_lmwm_libero import MilestoneGenerator, MilestonePredictorGrid
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    din, cd = int(ck["din"]), int(ck["code_dim"])
    Kp = ck["prd"]["pi.weight"].shape[0]
    gen = MilestoneGenerator(din, cd).to(device).eval(); gen.load_state_dict(ck["gen"])
    prd = MilestonePredictorGrid(din, cd, Kp).to(device).eval(); prd.load_state_dict(ck["prd"])
    return gen, prd, din, cd


class HintComputer:
    """encoder + LMWM. compute(image uint8 HxWx3, flip 后=正立训练朝向) → hint[DIN] fp32."""
    def __init__(self, encoder="dinov3-base", device="cuda"):
        self.device = device
        self.encoder_name = encoder
        self.gen, self.prd, self.din, self.cd = _load_lmwm(LMWM_CKPT[encoder], device)
        if encoder == "so400m":
            from PIL import Image
            from transformers import AutoModel, AutoProcessor
            so = next(p for p in [f"{REPO}/lmvla/lmwm/data/hf_so400m"] if os.path.isdir(p))
            self.proc = AutoProcessor.from_pretrained(so)
            self.mdl = AutoModel.from_pretrained(so, dtype=torch.bfloat16).to(device).eval()
            self._Image = Image
        else:
            os.environ["CRAVE_REPO"] = REPO  # crave DINOv3 权重路径 REPO/temp/dinov3_vitb16, 跨集群自适应
            sys.path.insert(0, f"{REPO}/lmvla/crave/src")
            from crave.encoders import load_encoder
            self.enc = load_encoder(encoder, dtype="bf16")
        print(f"[hint] encoder={encoder} din={self.din} code_dim={self.cd} loaded", flush=True)

    @torch.no_grad()
    def _encode_grid(self, imgs):
        """imgs: list[HxWx3 uint8] → grid [N, 256, DIN]."""
        if self.encoder_name == "so400m":
            chunk = [self._Image.fromarray(f) for f in imgs]
            px = self.proc(images=chunk, return_tensors="pt")["pixel_values"].to(self.device).to(torch.bfloat16)
            h = self.mdl.vision_model(pixel_values=px).last_hidden_state  # [N,256,1152]
            return h.float().cpu().numpy()
        g = self.enc.encode_grid(imgs)  # [N,D,P,P]
        g = g.detach().cpu().float().numpy() if hasattr(g, "detach") else np.asarray(g)
        N, D, P, _ = g.shape
        return g.transpose(0, 2, 3, 1).reshape(N, P * P, D)

    @torch.no_grad()
    def compute(self, image_uint8):
        """单帧 → hint[DIN] fp32. image_uint8 = 正立(flip 后) agentview HxWx3 uint8.
        EVAL_HINT_RESIDUAL=1 → 返回残差 ĝ_next − g_current(池化), 匹配残差训练(剥离冗余当前态)。"""
        grid = self._encode_grid([np.ascontiguousarray(image_uint8)])  # [1,256,DIN]
        G = torch.from_numpy(grid.reshape(1, PGRID, PGRID, self.din).transpose(0, 3, 1, 2).astype(np.float32)).to(self.device)
        logit, mu, ls = self.prd(G)
        code = mu[torch.arange(len(G)), logit.argmax(1)]
        gnext = self.gen(G, code).mean((2, 3))[0].float().cpu().numpy()  # [DIN] 绝对 ĝ_next
        if os.environ.get("EVAL_HINT_RESIDUAL", "0") == "1":
            gcur = grid[0].mean(0).astype(np.float32)  # [DIN] 池化当前 grid
            return gnext - gcur                          # 残差 hint
        return gnext


def _validate(encoder="dinov3-base"):
    """训练帧在线算 hint, 比对离线 hint.npz + 缓存 grid, 验证管线一致."""
    import av, pandas as pd  # noqa
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hc = HintComputer(encoder, dev)
    din_dir = "libero_dino" if encoder == "dinov3-base" else "libero_so400m"
    feat_dir = "libero_v2p1_dinov3base" if encoder == "dinov3-base" else "libero_v2p1_so400m_grid"
    # 离线 hint (libero_10, ep188, frame0)
    z = np.load(f"{REPO}/lmvla/lmwm/data/pi05_hint/{din_dir}/hint.npz", allow_pickle=True)
    m = (z["suite"] == "libero_10") & (z["episode_index"] == 188) & (z["frame_index"] == 0)
    off_hint = z["hint"][m][0].astype(np.float32)
    # 缓存 grid → grid_to_hint (验证 LMWM 部分)
    gz = np.load(f"{REPO}/lmvla/lmwm/data/pi05_feat/{feat_dir}/libero_10/ep188.npz")
    g0 = gz["grid"][0].astype(np.float32)  # [256,DIN]
    G = torch.from_numpy(g0.reshape(1, PGRID, PGRID, hc.din).transpose(0, 3, 1, 2)).to(dev)
    logit, mu, ls = hc.prd(G); code = mu[torch.arange(1), logit.argmax(1)]
    h_from_cache = hc.gen(G, code).mean((2, 3))[0].float().cpu().numpy()
    print(f"[val] LMWM: cache-grid→hint vs 离线 hint  MAE={np.abs(h_from_cache-off_hint).mean():.5f}  cos={np.dot(h_from_cache,off_hint)/(np.linalg.norm(h_from_cache)*np.linalg.norm(off_hint)):.4f}")
    # 训练帧 mp4 → encode → 比对缓存 grid (验证编码器)
    D = "/vePFS/tim/workspace/LIBERO_fastwam/libero_10_no_noops_lerobot"
    frame0 = next(av.open(D + "/videos/chunk-000/observation.images.image/episode_000188.mp4").decode(video=0)).to_ndarray(format="rgb24")
    my_grid = hc._encode_grid([frame0])[0]  # [256,DIN]
    print(f"[val] 编码器: 我的grid vs 缓存grid  MAE={np.abs(my_grid-g0).mean():.5f}  cos={np.dot(my_grid.ravel(),g0.ravel())/(np.linalg.norm(my_grid)*np.linalg.norm(g0)):.4f}")
    # 全链路: 训练帧 → hint
    my_hint = hc.compute(frame0)
    print(f"[val] 全链路: 训练帧→hint vs 离线 hint  MAE={np.abs(my_hint-off_hint).mean():.5f}  cos={np.dot(my_hint,off_hint)/(np.linalg.norm(my_hint)*np.linalg.norm(off_hint)):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", default="dinov3-base"); a = ap.parse_args()
    _validate(a.encoder)
