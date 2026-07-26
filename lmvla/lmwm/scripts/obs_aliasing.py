#!/usr/bin/env python
"""② 观测级别名 (harm-aligned): 当前观测跨-ep 相似度 per task.
spatial 应最高(同场景只位置变); object 应低(不同物体). 来源: so400m_grid frame0+mid.
输出: 每 task obs_alias, 与结构度合并, 重算 ρ, 看 spatial 是否爆表 & 是否独立于结构.
"""
import numpy as np, glob, re, json, os
from collections import defaultdict
from scipy import stats
import pandas as pd
SP="/vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/e56c875e-3983-4035-972c-e9cb06ca942f/scratchpad"
ep2t={int(k):v for k,v in json.load(open(f"{SP}/ep2tstr.json")).items()}
t2s=json.load(open(f"{SP}/task2suite.json")); t2s["open the top drawer and put the bowl inside"]="goal"

fs=sorted(glob.glob("lmvla/lmwm/data/libero_so400m_grid/ep*.npz"),
          key=lambda f:int(re.search(r'ep(\d+)',f).group(1)))
# 收集 per-ep: start & mid pooled feature
feat_start=defaultdict(list); feat_mid=defaultdict(list); ep_of=defaultdict(list)
by_task_start=defaultdict(list); by_task_mid=defaultdict(list); by_task_ep=defaultdict(list)
n=0
for f in fs:
    e=int(re.search(r'ep(\d+)',f).group(1))
    ts=ep2t.get(e);  su=t2s.get(ts) if ts else None
    if su is None: continue
    g=np.load(f)["grid"]           # (T,256,1152) fp16
    T=g.shape[0]
    s=g[0].astype(np.float32).mean(0)         # start pooled (1152,)
    m=g[T//2].astype(np.float32).mean(0)      # mid pooled
    by_task_start[ts].append(s); by_task_mid[ts].append(m); by_task_ep[ts].append(e)
    n+=1
print("loaded grid eps:",n)

def cross_ep_cos(vecs):
    X=np.stack(vecs).astype(np.float32)
    X/= (np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    S=X@X.T; iu=np.triu_indices(len(X),k=1)
    return float(S[iu].mean()) if len(iu[0]) else np.nan

rows=[]
for ts in by_task_start:
    if len(by_task_start[ts])<3: continue
    rows.append(dict(tstr=ts[:55], suite=t2s[ts], n_ep=len(by_task_start[ts]),
                     obs_alias_start=cross_ep_cos(by_task_start[ts]),
                     obs_alias_mid=cross_ep_cos(by_task_mid[ts])))
df=pd.DataFrame(rows)
df["obs_alias"]=(df["obs_alias_start"]+df["obs_alias_mid"])/2

# 合并结构度
cs=pd.read_csv(f"{SP}/collin_scores.csv")   # 有 struct, tstr(截断60), suite
cs["key"]=cs["tstr"].str[:55]
df["key"]=df["tstr"]
mg=df.merge(cs[["key","struct","alias"]].rename(columns={"alias":"tgt_alias"}),on="key",how="left")

print("\n=== per-task obs-aliasing (按 suite) ===")
print(mg[["suite","struct","obs_alias","tgt_alias","n_ep","key"]].sort_values(["suite","obs_alias"]).to_string(index=False))
print("\n=== suite 聚合 ===")
agg=mg.groupby("suite").agg(n=("key","size"),struct=("struct","mean"),
      obs_alias=("obs_alias","mean"),obs_start=("obs_alias_start","mean"),
      obs_mid=("obs_alias_mid","mean"),tgt_alias=("tgt_alias","mean"))
print(agg.sort_values("obs_alias",ascending=False).to_string())

sub=mg.dropna(subset=["struct","obs_alias"])
rp,pp=stats.pearsonr(sub["struct"],sub["obs_alias"])
rs,ps=stats.spearmanr(sub["struct"],sub["obs_alias"])
print(f"\n=== 共线诊断: struct vs OBS-alias (n={len(sub)}) ===")
print(f"Pearson  = {rp:+.3f} (p={pp:.2g})")
print(f"Spearman = {rs:+.3f} (p={ps:.2g})")
print("判定:", "共线→一维" if abs(rp)>0.8 else ("中等" if abs(rp)>0.5 else "近正交→二维成立"))
# spatial 是否爆表?
ranked=agg.sort_values("obs_alias",ascending=False).index.tolist()
print(f"obs-alias 排名(高→低): {ranked}  → spatial 第 {ranked.index('spatial')+1} 位")
mg.to_csv(f"{SP}/obs_alias_scores.csv",index=False)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
cmap={"spatial":"#d62728","object":"#2ca02c","goal":"#1f77b4","long":"#ff7f0e"}
for a,(yc,ttl) in zip(ax,[("obs_alias","OBS-aliasing (start+mid cross-ep cos)"),("tgt_alias","target-aliasing (naive)")]):
    for su in cmap:
        d=mg[mg.suite==su]
        a.scatter(d["struct"],d[yc],c=cmap[su],label=su,s=70,alpha=.8,edgecolors="k",lw=.4)
    a.set_xlabel("structure degree (segments/ep)"); a.set_ylabel(yc); a.set_title(ttl); a.grid(alpha=.3)
ax[0].legend()
ax[0].set_title(f"OBS-alias vs struct: ρ={rp:+.2f}")
plt.tight_layout(); plt.savefig(f"{SP}/obs_alias_scatter.png",dpi=130)
print("saved obs_alias_scores.csv, obs_alias_scatter.png")
