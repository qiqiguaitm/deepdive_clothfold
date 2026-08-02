# LMWM "效果不理想" 诊断: 是结构缺陷还是概念极限?(2026-07-26)

> 动机: ICLR 主张从"选择性专精"(观察, 弱)升级为"可预测增益"(定律)前, 必须先判定
> spatial −3.3 回归是**任务内在性质(概念极限)**还是**实现/训练伪影(结构缺陷, 可修)**。
> 本文档 = P1 诊断链: 共线 gate → 别名假设证否 → 读码定位架构 → 梯度隔离消融(判决中)。

---

## 1. 共线自检 gate(纯 CPU, 离线)

目的: 拟合"逐任务 Δ ≈ f(结构度, 别名度)"二维律前, 先查两轴是否可识别。

**数据源**: `libero_milestone_finalarch/pairs.npz`(cur_ep/cur_fi/tgt_fi/cur_ms/pair_task, 137154 对) +
`libero_milestone/target_compact.npz`(ep/tgt_fi/feat[13241,256,768]) + merged meta(40 task→suite, 串签名分类, spatial/object/goal/long 各 10)。

- **结构度** = 每 ep 的 distinct cur_ms 段数, 按 task 平均。
- **别名度 v1** = milestone 目标特征(mean-pool)跨-ep 余弦相似(高=不可辨)。

**结果(suite 聚合)**:

| suite | 结构度 | 平均长度 | 目标别名 | 观测别名(start+mid) | LMWM 伤害 |
|---|---|---|---|---|---|
| long | **5.94** | 272 | 0.965 | 0.987 | **+0.9** |
| object | 2.55 | 147 | **0.986(最高)** | **0.992(最高)** | −0.3 |
| goal | 2.70 | 126 | 0.975 | 0.987 | ~0 |
| spatial | 2.89 | 123 | 0.971 | **0.986(最低)** | **−3.0(最狠)** |

**三个硬结论**:
1. **结构度 ≈ horizon**: 结构度 vs episode 长度 Pearson **+0.784**。所谓"结构度预测增益"很大程度只是"帮长程", 近同义反复。且本质=**milestone 靶子退化程度**(spatial ~2 段=61 帧只~2 个恒定终态靶子), 非任务内在性质。
2. **别名假设(spatial 该爆表)被两个独立度量一致证否**: 目标别名 + 观测别名(start/mid 帧跨-ep 余弦, 加载 400 个 so400m_grid ep)**都把 spatial 排最后、object 最高**(小物体+恒定篮子背景→场景最像)。别名 vs 伤害**不对齐**。
3. **无任何 offline 任务特征能追踪 spatial 伤害**: 结构/horizon/两种别名全解释不了 spatial(−3.0)为何比 object/goal(−0.3/0)狠。

→ **二维律不成立**: 轴1≈horizon(平凡+混淆 r=0.78), 轴2(别名)无有效估计量。**offline-可预测路线扛不起主张。** 但"任何数据特征都追踪不到"这件事本身 = **伤害更可能是训练伪影**(非任务性质)的间接证据。

产物: 可复现脚本(tracked)`lmvla/lmwm/scripts/collinearity_gate.py` · `obs_aliasing.py`;
数据/图(磁盘持久, outputs 被 gitignore)`lmvla/lmwm/outputs/collinearity_gate_2026-07-26/`
= `collin_scores.csv` · `obs_alias_scores.csv` · `*_scatter.png` · `task2suite.json`。

---

## 2. 读码定位: LMWM 架构真相

(读 `lmvla/lmwam/adapter/lmwm_milestone_target.py` + `lawam.py`)

- **LMWM 本质 = 换世界模型的辅助监督靶子**, 非"注入 hint 当向导": Path A 把 starVLA 的 WM 目标从
  "t+7 近未来帧"(逐帧动态)换/并成"milestone+1 帧特征"(lawam.py:744-762)。与 t+7 同类 = 表征塑形辅助 loss 的靶子, `perceptual_weight=0.1`。
- **非 dual(替换)**: h_t1_gt 被 milestone 覆盖(丢 t+7)。
- **dual2q(V8, LMWM_DUAL_2Q=1)**: `h_t1_gt=h_t7_gt`(局部**保留** t+7)+ `h_ms_gt=milestone`(独立全局通道+独立 query, lawam.py:484-489,759-762)。
- **监督覆盖率 per suite 全 ≈1.0**(不是稀疏); 但**目标多样性坍缩**: 低结构任务(spatial)milestone 靶子≈恒定终态, 而 t+7 每帧动态。
- **梯度耦合**: ms 全局通道预测 `pred_latent_ms = vlm_to_lam_ms(h_act_q[:,Q:,:])` 来自共享 VLM hidden; config **vision/llm backbone 都没冻** → **ms 目标的梯度回流共享 VLM**。

**⭐核心假设**: dual2q 虽并联保留 t+7、却仍伤 spatial −3.3 → 排除"丢 t+7 密度"。矛头指向:
**低结构任务的 milestone 靶子退化 → ms 全局通道把共享 VLM 往退化 code 拉 → 污染 spatial 判别特征**(即便 t+7 局部通道原封不动)。= 训练伪影(靶子退化×梯度耦合), 非"milestone 误导 spatial"的概念极限。

**一镜重解释全部旧结果**:
- long +0.9(靶子多样→良性塑形)/ spatial −3.0(靶子退化→污染)。
- **残差救 spatial**(需配 tsched) = milestone−当前态把恒定靶子**重新逐帧动态化**(恢复 t+7 式塑形, 在最退化处救最多)——比"减身份/别名"更有码+数据支撑的解释。

---

## 3. 判决消融: 梯度隔离(进行中)

**代码**(lawam.py, env 门控 `LMWM_MS_DETACH_BACKBONE=1`, 4 处 detach):
`h_act_q` ms 切片(489/495)+ `h_t` 进 `_decode_ms_future`(813)。切断 ms→共享 VLM 梯度, 只让
`vlm_to_lam_ms`+`lmwm_dec` 自训, 骨干仅由 t+7/action 塑形。**纯训练期梯度手术, forward/推理/ckpt 结构不变 → A/B 可比**。已 scp 同步 North-E(diff 仅 4 处增量, 备份 `.bak_predetach`)。

**作业**: `t-20260726084743-86qxh`(`libero_dual2q_detachbb_25k_8h20.yaml`, 8×H20, 单变量=只加 detach vs `libero_dual2q_25k`)。yaml 含防呆前检 grep `_ms_detach_bb`。

**预注册(先于数据, 2026-07-26)**:
- 若 detach 后 **spatial ≥ 95.0**(修复 93.2→96.5 缺口≥半)→ **泄漏坐实(结构缺陷可修)**。
- 若 **spatial ≤ 93.5** → 泄漏证否, 伤害是概念级/靶子本身。
- 附加: 若 long/libero_10 增益**同步蒸发** → 增益与伤害**同源耦合**(共享骨干塑形是双刃, 拿增益必带伤害)。

对照: nowm spatial 96.5 / 普通 dual2q 93.2(见 `RESULTS_libero_4suite_2026-07-25.md`)。

## 3.1 判决结果(2026-07-26): 泄漏假设**证否** —— 伤害目标级为主 + 少量结构性

detach 消融 steps_12500, n=4 seed(osmesa, spatial+libero_10):

| 指标 | detach (n=4) | 对照 |
|---|---|---|
| **libero_spatial** | 92.8/93.2/94.4/92.6 → **均值 93.25 (std 0.70)** | nowm 96.5 · 普通dual2q@12500 90.8 |
| libero_10 | 95.0/92.8/95.8/94.6 → 均值 94.55 | nowm 94.3 · dual2q ~95.2 |

**判据**: ≥95=泄漏(可修)/ ≤93.5=概念极限。**93.25 ≤ 93.5 → 泄漏假设不成立**。
- 但**部分恢复** 90.8→93.25(+2.4, 约补回 90.8→96.5 缺口的 43%)→ **有少量结构性成分**(ms→骨干梯度确实污染了一点), 但**不是主因**。
- **主因 = milestone 靶子本身对 spatial 不合适**(目标级/概念性), 不是可用架构手术修掉的实现 bug。
- libero_10 没蒸发(94.55≈nowm)→ detach 去少量 spatial 伤害而未杀 libero_10, 与"部分结构性"一致。

**25k 确认(n=2, cont5k steps_5000)**: spatial [94.0, 94.4] 均值 94.20(恢复 ~60%), libero_10 95.50。→ 与 12500 一致: **detach 部分恢复但始终够不到 nowm 96.5**。

**⭐细化判决: 伤害 = 结构性 + 目标级 各占一半(双重机制, 非单一)**:
- detach 部分恢复 spatial(12500→43% / 25k→60%), **但两点都差 nowm −2.3~−3.2** → 结构性成分(ms→共享骨干梯度污染)真实存在且随训练变可修; 但**不可约的 ~2-3pt = 目标级成分**(milestone 靶子对 spatial 内在不适配)。
- 修正"目标级为主": 更准是**两机制各半**——一半可梯度隔离修掉, 一半是靶子不适配。
- libero_10 保持(94.55/95.50 ≈ nowm)= 结构性修复未伤长程增益。

**这加固"选择性专精"且更有料**: LMWM 伤 spatial 有**双重机制**, 即便最彻底架构手术(全梯度隔离)仍留 ~2-3pt 回归 = 不可约证据。残差救 spatial = 目标级修法(退化靶子重新动态化), 对应目标级那一半。

## 3.1b ⭐三方对比 refine 机制(2026-07-27, t+7 baseline 补齐)

LaWM(t+7)spatial eval(n=4, 同 12500 recipe): **93.35±0.86**; libero_10 95.30。全同口径三方:

| 配置 | spatial | vs nowm | libero_10 |
|---|---|---|---|
| nowm(无WM) | 96.5 | — | 94.3 |
| **t+7(LaWM)** | **93.35** | −3.15 | 95.30 |
| milestone(dual2q) | 90.8 | −5.7 | ~95.2 |
| milestone+detach | 93.25 | −3.25 | 94.55 |

**refine(比"双重机制各半"更准)**:
1. **通用 WM 伤害(~−3)**: **t+7 也伤 spatial**(−3.15)! 任何 WM 辅助塑形共享编码器都与"细粒度空间判别"竞争。非 milestone 独有。
2. **milestone 额外伤害(~−2.5)**: milestone(−5.7)比 t+7 多伤 ~2.5, = 退化靶子的梯度污染。**milestone+detach(93.25)≈ t+7(93.35)** → **detach 精确移除这额外部分, 把 milestone 拉到 t+7 水平**。
3. **libero_10(子目标): WM 帮**(t+7 95.3 / milestone ~95.2 均 > nowm 94.3)。

→ **通用WM harm + milestone额外污染(可detach除)**。⚠️**关键含义**: LIBERO 上 milestone **从不优于 t+7**——spatial 更差(或 detach 后持平), libero_10 持平(95.2≈95.3)。**LMWM 在 LIBERO 无相对 t+7 的优势。** LMWM 价值全押 RoboTwin stack(变长子目标是否胜 t+7 固定近未来)。

⚠️ 复现 gap: LaWAM 论文 t+7 spatial=99.4(帮), 我们 93.35(伤), 差 −6 = 12500步 vs 全量 + nowm 基线口径。**内部 A/B 有效, 绝对值勿对 99.4。** 编码器塑形 A/B(5jvb5/dswq8)将验 B(都不塑形)是否回 96.5。

**P1 完全收口: LIBERO 诚实负; 机制=通用WM harm+milestone额外污染; LMWM 在 LIBERO 不优于 t+7。**

## 3.2 后续(依赖 detach 结果)
冻结 VLM 视觉编码器 = 位点探针(测污染在视觉 vs 下游), 但比 detach 粗
(停所有目标的视觉适应=混淆; 且 LLM 层仍受 ms 污染=不完全隔离)。detach 若修好 spatial 再补冻视觉钉位点; 否则意义不大。

---

## 4. 结论与状态

- **P1 主张收窄**: offline 二维律不成立; 诚实主张 = "milestone 帮时序/子目标结构任务, 伤空间任务",
  用 detach 消融的**因果**证据(泄漏 vs 概念极限)支撑, 不靠脆弱的 offline 拟合。
- **判决未落**: detach 消融训练中(ck 推进健康, preflight 确认 flag 生效)。到 25k → spatial+libero_10 eval 收判。

关联记忆: `project_lmwm_architecture_flaw_hypothesis`, `project_libero_4suite_lmwm_negative`, `project_residual_precision_knob`。
