# PR-2 裁决: 混淆/结构度定律**证伪**(2026-07-28)

> 预注册预测(2026-07-25, PAPER_PLAN §16, 先于计算): ①spatial "on the ramekin"(语义 t5)混淆分数=
> spatial 组极端离群高值, 主要混向 t0("between")/t1("next to ramekin"); ②40 任务逐任务
> Δ(dual2q−nowm) 与混淆负相关、与结构度正相关。**证伪即定律死。**
>
> **判决: ① 证伪(方向相反), ② 证伪(符号错/零相关)。定律完全证伪。**
> 脚本 `lmvla/lmwm/scripts/pr2_confusion_structure.py`; npz `lmvla/lmwm/data/pr2_scores.npz`。

## 方法(严格按 §16 处方)
- **混淆分数**: 每帧 grid[256,768]→mean→768, L2 归一化; 跨-ep 最近邻(cosine, torch, **排除本 ep 全部帧**),
  统计近邻落在**其他任务**的比率。每 ep 均匀采 8 帧 → bank 13696 帧 / 1712 ep / 40 任务。
  同出 40×40 任务混淆矩阵(近邻投票分布)。**鲁棒变体**: 同套件内检索(排跨套件场景差异)。
- **结构度分数**: 每任务跑 CRAVE `build_clusters`(milestones.py)→ milestone 数 M; 及每 ep 平均 milestone 段数 seg。
- **Δ 覆盖(诚实声明, 关键限制)**:
  - **libero_10**: 真 per-episode `episodes.jsonl` — nowm=`20260718_211747...` vs dual2q=`dual2q_cfg15`。10 任务真 Δ。
  - **spatial/goal/object**: 本文件系统**无** per-episode eval jsonl(仅 resid/pi05 别的实验有, nowm/dual2q 的 osmesa 4-suite 原始 jsonl 未落盘)。
    仅能从 `RESULTS_libero_4suite_2026-07-25.md` 聚合重建: **spatial "on the ramekin" Δ=−34(76→42, doc §16 锚点)**, 其余 9 spatial 任务饱和 Δ≈0; goal 聚合 Δ=0; object 聚合 Δ=−0.3。
  - 即 40 任务 Δ = 10 真(libero_10)+ 1 doc 锚点(spatial t7=−34)+ 29 饱和≈0。**决定性的机理靶点(spatial t7)是真 doc 锚, 且直接反证 ①**。

- 任务索引口径: eval harness 顺序 ≠ LeRobot task_index(如 libero_10 eval t0 = LeRobot t5)。全部按 **task_description 字符串** join。
  §16 的 "t5=on the ramekin" 是语义/eval 编号; 在 LeRobot 数据中 "on the ramekin" = **spatial t7**。

---

## 预测① — **证伪(方向相反)**

spatial "on the ramekin"(LeRobot t7, 唯一被 LMWM 重伤 Δ=−34 的任务):
**混淆分数 = 0.010, 是 spatial 组第 9 低(rank 8/10), z = −1.27(低于均值 1.27σ)**。
预测说它是**极端高**离群; 实测它是**接近最低**。全局检索与同套件检索排名一致(证跨套件场景非混淆来源)。

| spatial task | 描述 | 混淆分数(全局) | 同套件内 | Δ(dual2q−nowm) |
|---|---|---|---|---|
| t9 | on the wooden cabinet | **0.190** (最高) | 0.190 | 0 |
| t0 | **between** the plate and ramekin | 0.172 | 0.172 | 0 |
| t4 | next to the plate | 0.155 | 0.155 | 0 |
| t8 | on the stove | 0.135 | 0.135 | 0 |
| t3 | next to the cookie box | 0.101 | 0.101 | 0 |
| t5 | **next to the ramekin** | 0.075 | 0.075 | 0 |
| t1 | from table center | 0.054 | 0.054 | 0 |
| t6 | on the cookie box | 0.033 | 0.033 | 0 |
| **t7** | **on the ramekin** ← 预注册"t5" | **0.010** | **0.010** | **−34** |
| t2 | in the top drawer | 0.000 | 0.000 | 0 |

- **唯一被伤的任务混淆最低; 混淆最高的 t9/t0/t4 完全不被伤(Δ=0)**。混淆分数与伤害**反向对齐**。
- 子命题(混向 t0/t1)方向微弱成立但量级可忽略: t7 的 1% 混淆确实几乎全指向 spatial t0(0.007)/t1(0.003), 但绝对量垫底 → 不构成"离群高"。

## 预测② — **证伪(符号错 / 零相关)**

| 相关(Spearman) | r | p | n | 预测 | 判决 |
|---|---|---|---|---|---|
| Δ vs 混淆分数 (40 任务) | **+0.228** | 0.157 | 40 | 负 | **符号相反, ns** |
| Δ vs 结构度 M (40 任务) | **−0.004** | 0.980 | 40 | 正 | **零相关** |
| Δ vs 结构度 seg (40 任务) | −0.010 | 0.950 | 40 | 正 | 零 |
| Δ vs 混淆 (libero_10, 真数 n=10) | −0.223 | 0.536 | 10 | 负 | 符号对但 ns |
| Δ vs 混淆 (libero_10, revalidate 平均) | −0.026 | 0.943 | 10 | 负 | ≈0 |
| Δ vs 结构度 M (libero_10, n=10) | +0.347 | 0.326 | 10 | 正 | 符号对但 ns |

- **40 任务 Δ~混淆符号为正(与预测相反)**: 因最负 Δ(spatial t7 −34)混淆≈0, 而混淆最高的 goal t3/t4/t5(0.36–0.44)Δ 全为 0(饱和)。混淆与 Δ 实质解耦。
- **40 任务 Δ~结构度 ≈ 0**: spatial t7 结构度 M=13(高), Δ 却最负 → 结构度臂同样反证。
- libero_10 真数子集: 混淆臂符号偶合(−0.22)但 revalidate 版归零(−0.03), 且全部 ns(n=10)。无一显著。

## 40×40 混淆矩阵摘要: 真正的检索别名簇 ≠ 被伤任务

近邻检索的高混淆结构是**真实存在的**, 但落在与 LMWM 伤害**无关**的任务上:

| 混淆最高任务 | 分数 | 主要混向 | Δ |
|---|---|---|---|
| goal t4 "put the bowl on the stove" | 0.443 | goal t3(0.24)/t5(0.11) | 0(饱和) |
| goal t3 "put the bowl on the plate" | 0.438 | goal t4(0.19)/t5(0.14) | 0 |
| goal t5 "put the bowl on top of the cabinet" | 0.364 | goal t3(0.17)/t4(0.09) | 0 |
| spatial t9 "on the wooden cabinet" | 0.190 | spatial t3/t8 | 0 |
| spatial t0 "between plate and ramekin" | 0.172 | spatial t4/t1 | 0 |

- **最强别名簇 = goal "put the bowl on {plate,stove,cabinet}" 三元组**(同动词+物体, 仅放置目标不同), 互混 36–44%。
  正是"跨-ep 检索别名最坏情况"的教科书案例 — 但 LMWM 对 goal **零影响**(全饱和 Δ=0)。
- 反之被伤的 spatial t7 混淆垫底。→ **别名→伤害机制被直接反驳**: 最别名的不被伤, 被伤的不别名。

## 逐任务两分数表(节选; 全 40 见 npz)

libero_10(真 Δ):

| task | 描述 | 混淆 | M | Δ |
|---|---|---|---|---|
| t3 | put both moka pots on the stove | 0.000 | 14 | +4 |
| t5 | both alphabet soup & tomato sauce in basket | 0.095 | 14 | −2 |
| t6 | both cream cheese & butter in basket | 0.103 | 16 | +2 |
| t8 | white mug on plate + pudding right | 0.000 | 8 | +4 |

(libero_10 内仅 t5/t6 有非零混淆——多物体入篮互相别名——Δ 分别 −2/+2, 无关系。)

---

## 结论: 定律**完全证伪**

1. **预测① 证伪**: "on the ramekin"(spatial t7, 唯一 Δ=−34 的任务)混淆分数最低(rank 8/10, z=−1.27), 非极端高离群。全局/同套件检索一致。
2. **预测② 证伪**: Δ~混淆 40 任务 r=+0.23(符号相反, ns); Δ~结构度 r=−0.00(零)。libero_10 真数子集亦全 ns。
3. **机理定位(为何证伪)**: 均值池化 DINOv3 gist 的跨-ep 检索混淆**不解释** LMWM 逐任务 Δ。
   - 真实别名簇(goal "put the bowl on X")是场景/物体级别名, LMWM 对其零影响。
   - spatial t7 的伤害**不源自该度量能捕捉的混淆**——它在 gist 检索下高度自相似(混淆垫底)。
   - 机制在别处: spatial t7 的回归更可能来自 milestone **语义/空间关系**层面的退化污染(bowl 的空间介词 on/next-to/between 在均值池化下不可分, 但在 milestone 判别的策略条件里造成干扰), 而**非** gist 检索可测的实例别名。§16 的"跨-ep 检索别名最坏情况"叙事**不被离线检索证据支持**。

**对论文的影响**: PR-2 定律不成立, 不能作为中心定律进论文。§16 预测①②③(含设计推论"检索置信门控拉回 t5")**失去经验支撑**——门控臂若要保留, 需另找 t7 伤害的真实相关量(候选: milestone 分配熵/空间介词判别度), 不能用本混淆分数背书。

## 覆盖与限制(如实)
- **真 per-episode Δ 仅 libero_10(10 任务)**; spatial/goal/object 的 nowm/dual2q 原始 jsonl 未落盘, 靠 doc 聚合重建(spatial t7=−34 为 doc 锚, 其余 29 饱和≈0)。
- 但**证伪不依赖弱覆盖**: 决定性反证点(spatial t7 混淆最低)只需特征(全 4 套件齐全)+ doc 锚 Δ, 二者均硬。即便补齐 spatial/goal/object per-episode Δ, 也无法把"t7 混淆最低"翻成"最高"。
- 均值池化或洗掉空间关系——但均值池化是 §16 明文处方, 且度量给出稳定可复现排名。若改用未池化 patch 级别度量另测, 属新预测, 不能救原预注册。
