#!/usr/bin/env python
"""[B step0] 建 robotwin2.0 cam_high 帧缓存 frame_cache_jpeg256(补 stack 等未覆盖 ep)。
背景: 原 0~4999 帧缓存是一次性建的(未提交), stack(ep 11082~26399)从没建过 → 特征/pairs 全缺。
本脚本按 --eps-file 补建, 格式与现有缓存**逐字一致**:
  frame_cache_jpeg256/chunk-{ep//1000:03d}/observation.images.cam_high/episode_{ep:06d}.npz
  键 "0".."N-1" = cv2.imencode('.jpg', 256x256 BGR) 的 uint8 字节。断点续建(已存在跳过)。
解码: pyav(libdav1d 解 AV1, 已验证 stack 视频可解)。resize 640x480 -> 256x256(与现有缓存同 256px)。
用法: python robotwin_frame_cache_build.py --eps-file <txt 逗号分隔> [--shard i --nshard n]
"""
import os, sys, glob, argparse, time
import numpy as np, cv2, av

REPO = os.environ.get("RT_REPO", "/vePFS/tim/workspace/deepdive_kai0")
DS = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
CAM = "observation.images.cam_high"
OUTROOT = f"{DS}/frame_cache_jpeg256"


def find_video(ep):
    hits = glob.glob(f"{DS}/videos/chunk-*/{CAM}/episode_{ep:06d}.mp4")
    return hits[0] if hits else None


def decode_video_256(path):
    """pyav 解 AV1 -> list[256x256x3 BGR uint8]。"""
    out = []
    c = av.open(path)
    for fr in c.decode(video=0):
        img = fr.to_ndarray(format="bgr24")            # HxWx3 BGR
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        out.append(img)
    c.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-file", required=True, help="逗号分隔 ep 列表文件")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--jpeg-q", type=int, default=92)
    a = ap.parse_args()

    want = [int(x) for x in open(a.eps_file).read().strip().split(",") if x.strip()]
    eps = sorted(set(want))[a.shard::a.nshard]
    print(f"[shard {a.shard}/{a.nshard}] {len(eps)} eps 待建帧缓存", flush=True)
    enc_par = [int(cv2.IMWRITE_JPEG_QUALITY), a.jpeg_q]

    done = skip = miss = 0
    t0 = time.time()
    for i, ep in enumerate(eps):
        op = f"{OUTROOT}/chunk-{ep // 1000:03d}/{CAM}/episode_{ep:06d}.npz"
        if os.path.exists(op):
            skip += 1; continue
        v = find_video(ep)
        if v is None:
            print(f"  ! ep{ep} 无视频, 跳过", flush=True); miss += 1; continue
        try:
            frames = decode_video_256(v)
        except Exception as e:
            print(f"  ! ep{ep} 解码失败: {e}", flush=True); miss += 1; continue
        d = {}
        for j, img in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", img, enc_par)
            d[str(j)] = buf.flatten()  # uint8
        os.makedirs(os.path.dirname(op), exist_ok=True)
        np.savez_compressed(op, **d)
        done += 1
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(eps)} done={done} skip={skip} miss={miss} "
                  f"({dt/(i+1):.2f}s/ep)", flush=True)
    print(f"[完成] done={done} skip={skip} miss={miss} 用时 {time.time()-t0:.0f}s", flush=True)
    print("FRAME_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
