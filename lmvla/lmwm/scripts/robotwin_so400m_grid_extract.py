#!/usr/bin/env python
"""Extract RoboTwin cam_high SigLIP-So400m patch-token grids.

Input frames come from the existing RoboTwin JPEG frame cache. Output layout
matches export_pi05_hint.py: ep{E}.npz with key "grid" [N,256,1152].
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import numpy as np


REPO = os.environ.get("RT_REPO", "/vePFS/tim/workspace/deepdive_kai0")
DS = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
CAM = "observation.images.cam_high"
OUT = f"{REPO}/lmvla/lmwm/data/robotwin_so400m_grid"
SO400M_CANDIDATES = [
    f"{REPO}/lmvla/lmwm/data/hf_so400m",
    "/vePFS/tim/workspace/deepdive_kai0/lmvla/lmwm/data/hf_so400m",
    "/vePFS-North-E/vis_robot/workspace/deepdive_kai0/lmvla/lmwm/data/hf_so400m",
]


def build_ep_map() -> dict[int, str]:
    out = {}
    for path in glob.glob(f"{DS}/frame_cache_jpeg256/chunk-*/{CAM}/episode_*.npz"):
        ep = int(os.path.basename(path).split("_")[1].split(".")[0])
        out[ep] = path
    return out


def decode_ep(path: str) -> list[np.ndarray]:
    data = np.load(path)
    frames = []
    for i in range(len(data.files)):
        img = cv2.imdecode(data[str(i)], cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps-file", default=None)
    parser.add_argument("--eps", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    epmap = build_ep_map()
    if args.eps_file:
        want = [int(x) for x in open(args.eps_file).read().strip().split(",") if x]
        eps = [ep for ep in want if ep in epmap]
    elif args.eps:
        eps = [int(x) for x in args.eps.split(",")]
    else:
        eps = sorted(epmap)
    eps = eps[args.shard::args.nshard]

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    so400m = next((p for p in SO400M_CANDIDATES if os.path.isdir(p)), None)
    if not so400m:
        raise FileNotFoundError(f"So400m HF directory not found: {SO400M_CANDIDATES}")
    proc = AutoProcessor.from_pretrained(so400m)
    model = AutoModel.from_pretrained(so400m, torch_dtype=torch.bfloat16).cuda().eval()
    os.makedirs(args.out, exist_ok=True)
    print(f"[enc] so400m={so400m} shard={args.shard}/{args.nshard} eps={len(eps)} out={args.out}", flush=True)

    t0 = time.time()
    done = skipped = nframes = 0
    for ep in eps:
        outp = os.path.join(args.out, f"ep{ep}.npz")
        if os.path.exists(outp) and not args.smoke:
            skipped += 1
            done += 1
            continue
        frames = decode_ep(epmap[ep])
        grids = []
        for i in range(0, len(frames), args.bs):
            chunk = [Image.fromarray(x) for x in frames[i : i + args.bs]]
            px = proc(images=chunk, return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
            with torch.no_grad():
                h = model.vision_model(pixel_values=px).last_hidden_state
            grids.append(h.float().cpu().numpy())
        grid = np.concatenate(grids, 0).astype(np.float16)
        if args.smoke:
            gf = grid.astype(np.float32)
            print(f"ep{ep}: frames={len(frames)} grid={grid.shape} mean={gf.mean():.3f} std={gf.std():.3f}", flush=True)
            break
        np.savez_compressed(outp, grid=grid, frame_index=np.arange(len(grid), dtype=np.int64))
        done += 1
        nframes += len(frames)
        if done % 50 == 0:
            elapsed = max(time.time() - t0, 1)
            print(f"[shard {args.shard}] {done}/{len(eps)} eps, {nframes} frames, {nframes/elapsed:.1f} fr/s, skipped={skipped}", flush=True)
    print(f"SHARD_{args.shard}_SO400M_DONE done={done} skipped={skipped} frames={nframes}", flush=True)


if __name__ == "__main__":
    main()
