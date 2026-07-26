#!/usr/bin/env python
"""① 共线自检 gate: 结构度 vs 别名度 per-task, 决定主张是二维还是一维.
结构度  = 每 episode 的 milestone 段数 (distinct cur_ms), 按 task 平均  [来源 pairs.npz]
别名度  = milestone 目标特征的跨-ep 余弦相似 (高=跨ep看起来一样=不可辨=别名) [来源 target_compact.npz]
输出: 40 task 两分数, ρ(结构,别名), suite 聚合, 散点 PNG.
预注册前的前置 gate: 若 |ρ|>0.8 → 主张收成一维'子目标可利用度'.
"""
import numpy as np, json, glob, pandas as pd
from scipy import stats
SP="/vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/e56c875e-3983-4035-972c-e9cb06ca942f/scratchpad"
ROOT="/vePFS/tim/workspace/deepdive_kai0"

# ---- 映射: ep -> tstr -> suite (含 goal 修正) ----
ep2t={int(k):v for k,v in json.load(open(f"{SP}/ep2tstr.json")).items()}
t2s=json.load(open(f"{SP}/task2suite.json"))
t2s["open the top drawer and put the bowl inside"]="goal"   # 修正误分
from collections import Counter
print("suite counts:", dict(Counter(t2s.values())))
def suite_of_ep(ep): return t2s[ep2t[ep]]

# ---- 结构度: pairs.npz ----
pr=np.load(f"{ROOT}/lmvla/lmwm/data/libero_milestone_finalarch/pairs.npz")
cur_ep, cur_ms, ptask = pr["cur_ep"], pr["cur_ms"], pr["pair_task"]
# pair_task(0-39) -> tstr (取该 task 任一 ep 的串)
pt2tstr={}
for e_,t_ in zip(cur_ep, ptask):
    if t_ not in pt2tstr: pt2tstr[int(t_)]=ep2t[int(e_)]
# 每 episode 的 distinct cur_ms 段数
seg_by_ep={}
import collections
tmp=collections.defaultdict(set)
for e_,m_ in zip(cur_ep, cur_ms): tmp[int(e_)].add(int(m_))
for e_,ss in tmp.items(): seg_by_ep[e_]=len(ss)
# 按 pair_task 聚合结构度
struct=collections.defaultdict(list)
ep_task={}
for e_,t_ in zip(cur_ep, ptask): ep_task[int(e_)]=int(t_)
for e_,n in seg_by_ep.items(): struct[ep_task[e_]].append(n)
struct_deg={t: float(np.mean(v)) for t,v in struct.items()}

# ---- 别名度: target_compact.npz (mean-pool 256 tokens) ----
tc=np.load(f"{ROOT}/lmvla/lmwm/data/libero_milestone/target_compact.npz")
tc_ep=tc["ep"].astype(int)
feat=tc["feat"]   # (13241,256,768) fp16
# mean-pool -> (N,768) fp32, 分块省内存
N=feat.shape[0]; D=feat.shape[2]
pooled=np.empty((N,D),dtype=np.float32)
CH=1000
for i in range(0,N,CH):
    pooled[i:i+CH]=feat[i:i+CH].astype(np.float32).mean(axis=1)
# L2 normalize
pooled/= (np.linalg.norm(pooled,axis=1,keepdims=True)+1e-8)
# tc_ep -> tstr -> 用 tstr 作 task key 聚合 (与 pair_task 对齐: 同 tstr)
tstr2pt={v:k for k,v in pt2tstr.items()}
# 每 task: 收集 (ep, pooled_idx)
by_task=collections.defaultdict(list)
for idx,e_ in enumerate(tc_ep):
    ts=ep2t.get(e_)
    if ts is None: continue
    pt=tstr2pt.get(ts)
    if pt is None: continue
    by_task[pt].append((e_,idx))

def cross_ep_sim(items):
    """跨-ep 平均余弦相似 (只算不同 episode 的 pair)."""
    eps=np.array([e for e,_ in items]); ids=np.array([i for _,i in items])
    X=pooled[ids]                      # (n,768) 已归一
    S=X@X.T                            # 余弦相似矩阵
    same=(eps[:,None]==eps[None,:])
    iu=np.triu_indices(len(items),k=1)
    mask_cross=~same[iu]
    mask_same= same[iu]
    sv=S[iu]
    cross=float(sv[mask_cross].mean()) if mask_cross.any() else np.nan
    within=float(sv[mask_same].mean()) if mask_same.any() else np.nan
    return cross, within

alias_deg={}; within_deg={}
for t,items in by_task.items():
    if len(items)<4: continue
    c,w=cross_ep_sim(items)
    alias_deg[t]=c; within_deg[t]=w

# ---- 汇总表 ----
rows=[]
for t in sorted(struct_deg):
    if t not in alias_deg: continue
    ts=pt2tstr[t]
    rows.append(dict(task=t, suite=t2s[ts],
                     struct=struct_deg[t], alias=alias_deg[t],
                     within=within_deg.get(t,np.nan),
                     alias_ratio=alias_deg[t]/within_deg[t] if within_deg.get(t) else np.nan,
                     tstr=ts[:60]))
df=pd.DataFrame(rows)
# 标准化两轴便于比较
for c in ["struct","alias"]:
    df[c+"_z"]=(df[c]-df[c].mean())/df[c].std()

print("\n=== per-task (40) ===")
print(df[["task","suite","struct","alias","alias_ratio","tstr"]].sort_values(["suite","struct"]).to_string(index=False))

print("\n=== suite 聚合 ===")
agg=df.groupby("suite").agg(n=("task","size"),struct=("struct","mean"),
                            alias=("alias","mean"),alias_ratio=("alias_ratio","mean"))
print(agg.to_string())

r_p,p_p=stats.pearsonr(df["struct"],df["alias"])
r_s,p_s=stats.spearmanr(df["struct"],df["alias"])
print(f"\n=== 共线诊断 (40 task) ===")
print(f"Pearson  ρ(struct,alias) = {r_p:+.3f}  (p={p_p:.2g})")
print(f"Spearman ρ(struct,alias) = {r_s:+.3f}  (p={p_s:.2g})")
verdict = "共线(|ρ|>0.8)→收成一维" if abs(r_p)>0.8 else ("中等相关" if abs(r_p)>0.5 else "近正交→二维成立")
print(f"判定: {verdict}")

df.to_csv(f"{SP}/collin_scores.csv",index=False)
json.dump(dict(pearson=r_p,spearman=r_s,verdict=verdict,
               suite_agg=agg.reset_index().to_dict("records")),
          open(f"{SP}/collin_result.json","w"),indent=2)

# ---- 散点 ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,6))
cmap={"spatial":"#d62728","object":"#2ca02c","goal":"#1f77b4","long":"#ff7f0e"}
for su in cmap:
    d=df[df.suite==su]
    ax.scatter(d["struct"],d["alias"],c=cmap[su],label=f"{su} (n={len(d)})",s=70,alpha=.8,edgecolors="k",linewidths=.4)
ax.set_xlabel("subgoal-structure degree  (milestone segments/episode)")
ax.set_ylabel("aliasing degree  (cross-ep milestone-target cosine sim)")
ax.set_title(f"Collinearity gate: Pearson ρ={r_p:+.2f}, Spearman={r_s:+.2f}\n{verdict}")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{SP}/collin_scatter.png",dpi=130)
print(f"\nsaved: collin_scores.csv, collin_result.json, collin_scatter.png")
