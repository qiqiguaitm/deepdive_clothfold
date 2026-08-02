# 调研:latent subgoal / future 信号注入 VLA(2025-2026 最新)— 2026-07-28

> 深度调研(105 agent,fan-out 检索 + 逐 claim 3 票对抗校验)。目的:ICLR 投稿 related work / baseline / 设计借鉴 / novelty 校验。
> 所有引用待 /paper-write 阶段按 bib 纪律二次核实。

## 1. 注入方式四大家族(谱系)

| 家族 | 代表 | 机制 | 推理开销 |
|---|---|---|---|
| **A. 追加 query + 结构化注意力** | DreamVLA(NeurIPS'25, 2507.04447) | `<dream>` query 拆 dynamic/depth/semantics 三 sub-query,block-wise mask 禁止 sub-query 互注意(防串扰);`<action>` query 独立取 latent 条件化 diffusion head;decoder 推理时全跳过 | near-zero |
| **B. 直接 token 替换** | AHEAD(2606.02486) | 4.9M 小 WM 在 **frozen VLA 自身特征空间**自回归预测未来 patch token,**替换**当前观测 token 进 frozen action decoder;光流速度/加速度逐 token 条件化;**不确定度阈值自适应截断 horizon**(唯一自适应先例) | 小 WM 前向 |
| **C. 纯辅助 loss / 表征对齐** | WorldVLA(2506.21539, α=0.04 平衡)、Spatial Forcing(2510.12276, cosine 对齐 VGGT 3D 表征) | 训练期辅助监督,推理零结构 | zero |
| **D. 门控 cross-attention** | FutureVLA(2603.10712, JVPM) | 视觉-运动解耦 + 可学习门 σ(r)⊙M:裸注入掉点→解耦+门控才转正 | 门控层 |

检索式(UR-VC 2607.12892)独立成支:时间标签均值+τ-band,已复现净负(见 roadmap §6.4)。

## 2. ⭐ 三条独立"失效模式"公开证据(=我们诊断的文献支撑)

1. **DreamVLA 消融**:depth-only / semantics-only 监督**低于无预测基线**,只有 dynamic-region 有效(3.64→4.32);原文归因 *"competing gradients dilute the task-relevant features"*。
2. **FutureVLA 消融(Table 5)**:裸注入 future-motor token **掉点 62.5→58.4**(*"absorb noisy or task-irrelevant visual dynamics"*);解耦→65.6,+门控→71.9。
3. **WorldVLA**:自回归动作 chunk 注入自身历史动作反伤(抓取 −10~50%),mask 掉先前动作 token 才修复(54.0→76.6)。
4. (第三方)WAM vs VLA 对照(2603.22078,华为+UofT):RoboTwin2.0-Plus 上 WM 更鲁棒(LingBot-VA 74.2 vs π0.5 58.6 扰动下),**LIBERO-Plus 反转**(纯 π0.5 85.7 全场最高)→ 未来信号收益是**条件性的**。

**共性**:大家都撞上"冗余/干扰/污染",各自打了 ad-hoc 补丁(mask/门控/分解),**无人给出系统性诊断框架**。

## 3. ⭐ Novelty 空位确认(对投稿最关键)

- **残差/delta 目标:零先例**。全部已验证方案(DreamVLA/MoLA/AHEAD/FutureVLA/WorldVLA/SF)均为绝对目标。✅ 我们的核心设计无人占位。
- **任务级分析:无人低于 per-suite 粒度**(最接近的是 FutureVLA 报 Long 套件增益最大)。✅ 我们的任务级定律 + t5 解剖差异化成立。
- **自适应 horizon 唯一先例 = AHEAD**(不确定度截断)——必引,并把我们的门控设计与之对照。
- 系统性失效诊断(张力+冗余+别名)+ 由诊断推导设计:空位。

## 4. 提点证据基准表(写 related work / baseline 用)

| 方法 | LIBERO 平均 | CALVIN ABC-D | 其他 |
|---|---|---|---|
| DreamVLA | 92.6(S97.5/O94.0/G89.5/L89.5) | 4.44 | 真机 Franka 76.7% |
| MoLA(2605.12167) | **97.0**;LIBERO-Plus 92.7 | **4.55** | 对 DreamVLA 最强超越 |
| FutureVLA | **98.3**(自报;Long 96.0 vs π0 85.2 = **+10.8**) | — | SimplerEnv +9.4 |
| WorldVLA | 81.8(L 60.0) | — | 同 backbone 消融 +4.4 |
| Spatial Forcing | 97.1→98.5(**+1.4, 无 seed/误差棒**) | — | 3.8× 收敛加速 |
| AHEAD | —(主打动态任务) | — | 动态场景 79-97 vs 基线 31-58;真机拦截/接取 |

方法论攻击面:SF 的 +1.4 无误差棒(我们的多 seed/1.5pt 纪律可作方法论批评);FutureVLA 自报+近饱和。

## 5. 可借鉴设计(按优先级,映射到我们的线)

1. **可学习/信号驱动门控**(FutureVLA σ(r) + AHEAD 不确定度截断)→ **t5 别名的对症药**:检索置信/残差范数门控 hint(PR-2 预测③;也与主会话"幅度门控"主线合流)。裸注入→门控 = 文献公认的救法,我们的版本可从信号自身导出(免学习)。
2. **解耦注意力 mask**(DreamVLA block-wise)→ 架构级防冗余:禁止 ms 通道 attend 当前观测 token,与残差目标互为替代/互补消融。
3. **"只预测变化"的旁证**(DreamVLA 只有 dynamic-region 有效)→ 空间维度的选择性 ≈ 我们时间维度的残差;论文可写"其空间发现与我们的时间残差发现同构"。
4. **AHEAD 的动态任务定位** → 解释"LIBERO 静态桌面上未来信号提点天然有限",支撑 RoboTwin/真机权重。
5. **WorldVLA 的 α 平衡** → 我们残差 loss 尺度问题(scale=4 臂)有文献同类(量级平衡是公认工程点)。

## 6. 对论文的落地动作
- Related work 重排:四家族 + 检索式 + 我们的定位("绝对目标+套件级评测+经验性补丁" vs "诊断→残差设计→任务级定律→预注册验证")。
- 引用计划新增(全部 [VERIFY]):DreamVLA 2507.04447 / MoLA 2605.12167 / AHEAD 2606.02486 / FutureVLA 2603.10712 / WorldVLA 2506.21539 / Spatial Forcing 2510.12276 / WAM 对照 2603.22078。
- 三条失效证据写进 §1/§4 作独立佐证;WAM 对照支撑"收益条件性"。
- 设计实验新增候选:门控臂(检索置信门控,对症 t5)优先于更多注入变体。
