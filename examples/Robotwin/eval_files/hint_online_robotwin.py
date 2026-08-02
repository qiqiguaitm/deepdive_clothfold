#!/usr/bin/env python
"""在线 LMWM hint 计算 (RoboTwin A1 eval): live head_camera → DINOv3-base → grid → LMWM → hint[768]。

泛化自 examples/LIBERO/eval_files/hint_online.py 的 HintComputer:
  - ckpt 换 lmwm_robotwin_rvalley/lmwm.pt (din=768, code_dim=32, K=4)
  - 模型类换 p1_train_lmwm_robotwin (MilestoneGenerator / MilestonePredictorGrid, forward 与 libero 版一致)
  - encoder = dinov3-base (crave, 同 robotwin grid 抽取)
  - **图像预处理匹配训练**: robotwin grid 由 cam_high 帧 (frame_cache_jpeg256: 640×480→256×256, RGB, **无 flip**)
    抽取 (robotwin_dinov3base_grid_extract.py)。RoboTwin head_camera 正立, 故不 flip; resize 256×256 匹配缓存。

用法(校验, 对训练缓存 grid 比对): python hint_online_robotwin.py --check
"""
import os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
LMWM_CKPT = {
    "dinov3-base": f"{REPO}/lmvla/lmwm/checkpoints/lmwm_robotwin_rvalley/lmwm.pt",
    "so400m": f"{REPO}/lmvla/lmwm/checkpoints/lmwm_robotwin_so400m/lmwm.pt",
}
PGRID = 16
_FRAME_CACHE_SIZE = 256  # robotwin_frame_cache_build.py: cam_high resize 640×480 → 256×256


def _load_lmwm(ckpt_path, device):
    sys.path.insert(0, f"{REPO}/lmvla/lmwm/scripts")
    from p1_train_lmwm_robotwin import MilestoneGenerator, MilestonePredictorGrid
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    din, cd = int(ck["din"]), int(ck["code_dim"])
    Kp = ck["prd"]["pi.weight"].shape[0]
    gen = MilestoneGenerator(din, cd).to(device).eval(); gen.load_state_dict(ck["gen"])
    prd = MilestonePredictorGrid(din, cd, Kp).to(device).eval(); prd.load_state_dict(ck["prd"])
    return gen, prd, din, cd


class RobotwinHintComputer:
    """encoder + LMWM. compute(head_camera uint8 HxWx3 RGB) → hint[768] fp32。

    与训练一致: cam_high 正立 (无 flip), resize 256×256, dinov3-base encode_grid。"""

    def __init__(self, encoder="dinov3-base", device="cuda"):
        if encoder not in LMWM_CKPT:
            raise ValueError(f"Unsupported RoboTwin hint encoder: {encoder!r}")
        self.device = device
        self.encoder_name = encoder
        self.gen, self.prd, self.din, self.cd = _load_lmwm(LMWM_CKPT[encoder], device)
        if encoder == "so400m":
            from PIL import Image
            from transformers import AutoModel, AutoProcessor

            candidates = [
                f"{REPO}/lmvla/lmwm/data/hf_so400m",
                "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/hf_so400m",
                "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lmwm/data/hf_so400m",
            ]
            so400m = next((path for path in candidates if os.path.isdir(path)), None)
            if so400m is None:
                raise FileNotFoundError(f"So400m HF directory not found: {candidates}")
            self.proc = AutoProcessor.from_pretrained(so400m)
            self.mdl = AutoModel.from_pretrained(
                so400m, torch_dtype=torch.bfloat16
            ).to(device).eval()
            self._Image = Image
        else:
            os.environ.setdefault("CRAVE_REPO", REPO)  # crave DINOv3 weights
            sys.path.insert(0, f"{REPO}/lmvla/crave/src")
            from crave.encoders import load_encoder

            self.enc = load_encoder(encoder, dtype="bf16")
        import cv2
        self._cv2 = cv2
        self.intervention = os.environ.get("ROBOTWIN_HINT_INTERVENTION", "correct").strip().lower()
        if self.intervention not in {"correct", "current", "zero", "shuffle"}:
            raise ValueError(f"Unsupported ROBOTWIN_HINT_INTERVENTION={self.intervention!r}")
        self.shuffle_seed = int(os.environ.get("ROBOTWIN_HINT_SHUFFLE_SEED", "2026"))
        print(
            f"[rt-hint] encoder={encoder} din={self.din} code_dim={self.cd} "
            f"intervention={self.intervention} loaded",
            flush=True,
        )

    @torch.no_grad()
    def _encode_grid(self, imgs):
        """imgs: list[HxWx3 uint8 RGB] → grid [N, 256, 768]."""
        if self.encoder_name == "so400m":
            requested = len(imgs)
            # H20 bf16 selects a numerically different GEMM kernel for batches
            # 1-2. The offline cache was encoded with batches >=4; padding to
            # four reproduces its features exactly and is required at eval.
            padded = list(imgs)
            if requested < 4:
                padded.extend([padded[-1]] * (4 - requested))
            chunk = [self._Image.fromarray(image) for image in padded]
            px = self.proc(images=chunk, return_tensors="pt")["pixel_values"].to(
                self.device, torch.bfloat16
            )
            h = self.mdl.vision_model(pixel_values=px).last_hidden_state
            return h[:requested].float().cpu().numpy()
        g = self.enc.encode_grid(imgs)  # [N,D,P,P]
        g = g.detach().cpu().float().numpy() if hasattr(g, "detach") else np.asarray(g)
        N, D, P, _ = g.shape
        return g.transpose(0, 2, 3, 1).reshape(N, P * P, D)

    def _prep(self, image_uint8):
        """匹配训练 frame cache: RGB, resize 256×256, 无 flip。"""
        img = np.ascontiguousarray(np.asarray(image_uint8))
        if img.shape[:2] != (_FRAME_CACHE_SIZE, _FRAME_CACHE_SIZE):
            img = self._cv2.resize(img, (_FRAME_CACHE_SIZE, _FRAME_CACHE_SIZE), interpolation=self._cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    @torch.no_grad()
    def compute(self, image_uint8):
        """单帧 head_camera → hint[768] fp32 (绝对 ĝ_next)。"""
        grid = self._encode_grid([self._prep(image_uint8)])  # [1,256,768]
        G = torch.from_numpy(
            grid.reshape(1, PGRID, PGRID, self.din).transpose(0, 3, 1, 2).astype(np.float32)
        ).to(self.device)
        logit, mu, ls = self.prd(G)
        code = mu[torch.arange(len(G)), logit.argmax(1)]
        gnext = self.gen(G, code).mean((2, 3))[0].float().cpu().numpy()  # [D]
        current = grid[0].mean(0).astype(np.float32)
        residual = os.environ.get("EVAL_HINT_RESIDUAL", "0") == "1"
        hint = (gnext - current).astype(np.float32) if residual else gnext.astype(np.float32)

        if self.intervention == "zero":
            return np.zeros_like(hint)
        if self.intervention == "current":
            # In residual space, zero is the exact "no predicted change" control.
            return np.zeros_like(hint) if residual else current
        if self.intervention == "shuffle":
            permutation = np.random.default_rng(self.shuffle_seed).permutation(hint.shape[-1])
            return hint[permutation].copy()
        return hint


def _check():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hc = RobotwinHintComputer("dinov3-base", dev)
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    h = hc.compute(img)
    print(f"[rt-hint][check] hint shape={h.shape} dtype={h.dtype} norm={np.linalg.norm(h):.3f} "
          f"finite={np.all(np.isfinite(h))}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    _check()
