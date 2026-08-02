#!/usr/bin/env python
"""决定性 replay 测试: 把训练数据(视频存储帧=模型训练时所见, 已正立)+ state 喂给训好的 pi05,
比对预测 action chunk 与记录的真实 future action。隔离 env, 只测 [模型 + 推理transform管线] 是否正确。
  pred≈recorded → 模型学好了+推理对 → eval 0% 根因在 env 执行(rollout 侧 flip/gripper/controller)。
  pred≠recorded → 推理时 obs 映射/norm 与训练不一致。
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath("kai0/src"))
import av
from openpi.training import config as _config
from openpi.policies import policy_config as _pc

CKPT = os.environ["CKPT"]; DATA = os.environ["DATA"]; TI = int(os.environ.get("TI","5"))
cfg = _config.get_config("pi05_libero_a0_bj")
policy = _pc.create_trained_policy(cfg, CKPT)
print("[replay] policy loaded")

# 找 task_index==TI 的训练 episode
ep=None; pqf=None
for pq in sorted(glob.glob(DATA+"/data/chunk-000/episode_*.parquet")):
    df=pd.read_parquet(pq)
    if int(df["task_index"].iloc[0])==TI:
        ep=int(df["episode_index"].iloc[0]); pqf=pq; break
tasks={}
import json
for l in open(DATA+"/meta/tasks.jsonl"): d=json.loads(l); tasks[d["task_index"]]=d["task"]
prompt=tasks[TI]
print(f"[replay] episode={ep} task_index={TI} prompt={prompt!r} frames={len(df)}")

# 读视频帧(存储的正是训练所见图像)
def load_frames(vid):
    c=av.open(vid); return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
base=load_frames(DATA+f"/videos/chunk-000/observation.images.image/episode_{ep:06d}.mp4")
wrist=load_frames(DATA+f"/videos/chunk-000/observation.images.wrist_image/episode_{ep:06d}.mp4")
state=np.stack(df["observation.state"].values).astype(np.float32)
act=np.stack(df["action"].values).astype(np.float32)
print(f"[replay] base={len(base)} wrist={len(wrist)} state={state.shape} act={act.shape}")

for fi in [0, 20, 50, 100, min(150,len(df)-9)]:
    if fi+8>len(df): continue
    obs={"observation/image":np.ascontiguousarray(base[fi]),
         "observation/wrist_image":np.ascontiguousarray(wrist[fi]),
         "observation/state":state[fi], "prompt":prompt}
    out=policy.infer(obs)
    pred=np.asarray(out["actions"])[:8]     # [8,7]
    rec=act[fi:fi+8]                          # 记录的未来8步
    mae=np.abs(pred-rec).mean(0)
    print(f"\n[frame {fi}] recorded_state_gripper={state[fi,6:].round(3)}")
    print(f"  pred[0]  ={pred[0].round(3)}")
    print(f"  recorded ={rec[0].round(3)}")
    print(f"  per-dim MAE(8步): {mae.round(3)}  chunk_mean_MAE={mae.mean():.3f}")
    print(f"  pred gripper序列 ={pred[:,6].round(2)}")
    print(f"  rec  gripper序列 ={rec[:,6].round(2)}")
print("\n[replay] done")
