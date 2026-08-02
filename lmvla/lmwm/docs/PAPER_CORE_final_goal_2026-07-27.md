# 论文核心 & 最终目标(定版思考,2026-07-27)

## 一句话核心
**LMWM(里程碑/子目标世界模型)作为 VLA 的辅助,能带来真实但"有条件"的帮助;本文刻画"何时帮、何时伤、为什么、以及如何注入才只取其利"。**

不是"LMWM 普适让 VLA 更强"(证据证否),而是 **"milestone-WM 是子目标结构任务的专精辅助,其收益可从任务结构预测、其代价有明确机制且可控制去除"**。

## 为什么核心从"SOTA"改成"条件性帮助+机制"(诚实演进)
- 原目标 LMWAM 在 LIBERO+RoboTwin 达 SOTA → **推翻**:LIBERO 已饱和(π₀/OpenVLA-OFT 94-97),且 **LMWM 在 LIBERO 净负**(伤 spatial)。
- 硬堆 SOTA 讲不通;但**"何时/为何帮"这个机制问题是真贡献**,且证据齐全、防弹。

## 四条支撑证据(本会话建立)
1. **现象:选择性帮助,可从结构预测。** milestone 段数(子目标结构度)呈梯度 `hammer 1 < handover 2 < stack 3 < ranking 4`;LMWM 收益随此梯度——帮长程/子目标任务(LIBERO-Long +0.9、RoboTwin stack 待验),伤无子目标任务(LIBERO-spatial −3)。
2. **机制:伤害是双重的(detach 消融 3 方判决)。**
   - 结构性(~半):milestone 目标的梯度污染**共享 VLA 骨干** → 梯度隔离(detach)可修回 ~40-60%。
   - 目标级(~半,不可约):milestone 靶子对"判别线索≠目标"的任务(spatial)内在不适配 → 即便全梯度隔离仍差无-WM 地板 2-3pt。
   - 3 方对标:无WM(nowm)96.5 / t+7(LaWM)待测 / milestone(LMWM)90.8 / +detach 93-94。
3. **控制/设计:可控注入取利去害。** 残差 milestone(目标级修法,退化靶子重新动态化)+ 梯度隔离(结构级修法)→ 在无子目标任务恢复,同时保子目标任务增益。
4. **验证:双 benchmark + 正负对照。** RoboTwin 均衡集(**stack=正例 / hammer=负例**,curation bug 已修)+ LIBERO 全 4 套件(诚实聚合)。

## 核心命题(可证伪)
> milestone-WM 对某任务的 per-task 效应,由该任务的**子目标结构度**决定:结构度高 → 帮;结构度低(判别线索与目标正交)→ 伤;伤害经"共享骨干污染(可修)+ 靶子不适配(不可约)"两条通路。

**证伪条件**:若 RoboTwin stack(高结构、milestone 段数 3)上 LMWM **不**赢 baseline → 核心命题在 RoboTwin 侧崩,需重估。

## 口径(对齐主流,agent 调研确认)
- **LIBERO**:4 套件逐套件+Average,SR%±std,500 rollouts/套件(10×50),n≥4 seed。对标 π₀ 96.8/94.2、OpenVLA-OFT 97.6/97.1。禁单套件。
- **RoboTwin 2.0**:逐任务+平均,**Easy(demo_clean)+Hard(demo_randomized)双列**,100 rollouts/任务;内部 A/B 用同数据同步数(唯一变量=LMWM),对外榜需 on-protocol 单任务 50 demo。

## 当前实验状态(推进中)
- P1 LIBERO:✅ 收口(净负 + 双重机制);t+7 baseline spatial eval 跑中(补 3 方最后一格)。
- P2 RoboTwin:均衡重训(北京 vdptr/snf2v)→ 20k 自动结构梯度 eval(验 stack 正例);**待加 Hard 口径**。

## 下一步(推进清单)
1. t+7 spatial eval 收 → 3 方定表(无WM/t+7/milestone)。
2. RoboTwin 均衡 eval(Easy)→ 若 stack LMWM 赢 = 核心命题正面证据。
3. **加 RoboTwin Hard(demo_randomized)口径**(主流强制)。
4. (可选)残差+梯度隔离在 RoboTwin 训一版,证"可控注入"章节。

关联:`INVESTIGATION_lmwm_architecture_flaw_2026-07-26.md`、`RESULTS_robotwin_P2_2026-07-25.md`、`PIPELINE_robotwin_rebalance_2026-07-26.md`、`PAPER_EXPERIMENT_PLAN_2026-07-26.md`。

---

## 附:LaWAM(arXiv:2606.15768)对标数据 + 口径(2026-07-27 深挖,正式引用前再核原文表)

**LaWAM = 我们的 LaWM/t+7 baseline 本尊**(latent world action model,冻结视觉潜空间预测 t+7 未来特征作潜视觉子目标)。LMWM = 把它的 t+7 靶子换成 milestone。

### LIBERO(Table 1,50 trials/task,无误差棒)
| Method | 参数 | 延迟 | Long | Goal | Object | Spatial | Avg |
|---|---|---|---|---|---|---|---|
| π₀ | 3.5B | 220 | 88.4 | 94.4 | 96.8 | 98.0 | 94.4 |
| π₀.₅ | 3.5B | 220 | 92.4 | 98.0 | 98.2 | 98.8 | 96.9 |
| OpenVLA-OFT | 7B | — | 94.5 | 97.9 | 98.4 | 97.6 | 97.1 |
| Cosmos-Policy | 2.1B | 1413 | 97.6 | 98.2 | 100 | 98.1 | 98.5 |
| **LaWAM** | **2.3B** | **187** | 97.0 | 98.4 | 99.6 | **99.4** | **98.6** |

LIBERO 饱和(全场 97-98.6);LaWAM 精度并列最好,真赢点=最小模型+最快延迟。**注意 LaWAM(t+7)spatial=99.4 > 无WM,即 t+7 帮 spatial;而 LMWM(milestone)伤 spatial —— 这是核心对照(稠密 t+7 全面帮 vs 稀疏 milestone 选择性帮)。**

### RoboTwin 2.0(Table 2,均值)
**⭐口径(关键)**:**训练 = 2,500 clean + 25,000 randomized = 27,500 demo(多任务、50 任务、重数据)**;每任务 **100 trials**,**Clean + Randomized 双报**。**不是 leaderboard 的单任务 50 demo**(那个 π₀ 才 46)——差 550× 数据量。
| Method | Clean | Randomized |
|---|---|---|
| π₀.₅ | 82.74 | 76.76 |
| Motus | 88.66 | 87.02 |
| Fast-WAM | 91.98 | 90.52 |
| LingBot-VA | 91.50 | 90.92 |
| **LaWAM** | **92.64** | 89.80 |

### 对我们的含义
- **对标 LaWAM 的协议 = 全量多任务(27,500 demo, 50 任务),不是 leaderboard**。要正面比 LaWAM 92.64,须在全量 RoboTwin 上训 LMWM+LaWM。
- 我们均衡集(1000ep/6任务)只够做**机制 A/B**(LMWM vs LaWM 同数据),绝对值追不上 LaWAM。
- 真机 Table 3:LaWAM 90.0 avg(pick-place 93.3/drawer 86.7/fold 90.0)。

## 决策(2026-07-27):上全量数据训 LMWAM 对标 LaWAM
目标 = 在 LaWAM 全量协议(27,500 demo / 50 任务 / Clean+Randomized 100 trials)上训 **LaWM baseline + LMWM**,正面对标 92.64。
**前置**:LMWM 需全量 milestone pairs(27,500 ep 的 frame cache+DINOv3 特征+r-field pairs ≈ ~30h 预处理,当前只有 1000 balanced)。→ baseline 可即提;LMWM 走全量 milestone 预处理管线。
