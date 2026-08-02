# PAPER_PLAN — MINT-VLA → ICLR 2027(2026-07-24)

> Naming update (2026-08-01): the paper and integrated system are now **MINT-VLA** (Milestone Integration with Native-Space Targets for Vision-Language-Action Models).
> Earlier experiment/config identifiers such as `LMWAM-DS` remain unchanged so
> that historical checkpoints and logs stay traceable.

**工作标题(候选,按 Story 定稿)**
- A(主推): *Milestones Are Not Enough: Diagnosing and Resolving the Precision–Guidance Tension in World-Model-Conditioned VLAs*
- B(备选): *Recurrence as a Universal Signal: From Cross-Episode Density Fields to Dual-Scale VLA Conditioning*
- C(诊断向兜底): *When Do World-Model Subgoals Help VLAs? A Cross-Architecture Mechanistic Account*

**One-sentence contribution(Story A)**
> 我们发现"世界模型子目标信号注入 VLA"存在一个跨架构可复现的 **precision↔guidance 零和张力**(指引类任务受益、精度类任务受损、聚合不动),定位其两大机理根因(**目标与当前态冗余** + **表征空间错配**),并用 **双尺度并联 + 残差目标 + 相位自适应 dropout** 将其化解,在 LIBERO 难任务与 RoboTwin(有余量基准)上转化为净增益。

**Venue**: ICLR 2027(截稿≈2026-09 下旬,**需确认精确日期**)
**Type**: Method + Analysis 混合(6+1 节)
**Page budget**: 9 页正文(至 Conclusion,不含 ref/appendix)

---

## 0. 战略:三个 Story 与两个决策门

论文能讲什么取决于两个未落地实验。**先定门,再锁 story,写作期不返工。**

| 门 | 实验 | 时间 | 判据 | 影响 |
|---|---|---|---|---|
| **G1 残差门** | v8xpb(V8+LMWM_MS_RESIDUAL)eval | ~本周 | t8 ≥94 且 t6 不塌(≥85)且聚合>95.2 | 过→C4"张力被残差化解"成立,Story A 完整;不过→残差降级为 pi05 侧机理证据,主打诊断+双尺度 |
| **G2 RoboTwin 门** | RoboTwin MINT-VLA vs baseline 聚合 | ~8月中 | 聚合 SR 显著↑(>2×SEM) | 过→Story A/B 的"headroom 净增益"主结果;不过→转 Story C 诊断论文 |
| G3 基线卫生 | armB baseline 多 seed(现 n=1: 96.4) | 本周 | — | 所有聚合对比的合法性前提(§4.16 纪律: <1.5pt 不可声称) |

- **G1 ∧ G2 过** → **Story A**(method+analysis,最强)。
- **G1 过、G2 不过** → Story A 弱化版:诊断为主、修复在 LIBERO 难任务上验证,RoboTwin 作 negative/analysis。
- **均不过** → **Story C** 纯诊断论文:卖点=跨两套独立 VLA 栈(LaWAM/starVLA + pi05/openpi)复现同一张力与机理,ICLR 接受高质量 analysis paper,但需把机理证据链做满(见 §E 实验缺口)。
- Story B(r(o) 普适信号为主线)仅当 UR-VC 差异化实验(D6)做得非常干净时上位;默认作为 Story A 的 §3 组成部分,不独立成篇。

---

## 1. Claims–Evidence Matrix(诚实版)

| # | Claim | Evidence(已有) | Status | 缺口 | 拟放节 |
|---|---|---|---|---|---|
| C1 | 跨-ep recurrence 密度场 r(o) 是免聚类、免调参、跨本体的普适信号;**r-脊**是比 milestone 边界/固定 horizon 更好的 WM 目标 | 内在前向 gain 2.1×(§4.6/4.7);一套超参跨 kai0/LIBERO/robotwin(§4.4);量化解释 milestone+1 的 −4.2pt | 内在指标已支持;下游部分支持 | 下游 SR 层面的 r-脊 vs 边界对照(可复用已有 ckpt) | §3 |
| C2 | **Precision↔guidance 零和张力**:朴素子目标注入=逐任务重分配、聚合封顶 | LaWAM 侧:E1/2Q/2Q+CFG 三种混合全部 94.8±~0.2 封顶,t6/t9↑ 恰被 t7/t8↓ 抵消;推理层(CFG)两次证否 | **已支持(多 seed)** | armB 多 seed 基线(G3) | §4 |
| C3a | 根因一:**目标冗余** — 绝对子目标 86% 是当前态,真信号在残差(判别力 CV 1.1 vs 0.12,10×);flow 条件里 h_t 已在场,绝对目标=重复编码 | pi05 hint 分析 + V8 条件结构论证 | pi05 侧已支持 | **MINT-VLA 侧闭环 = G1(v8xpb)** | §4 |
| C3b | 根因二:**表征空间错配** — 外挂空间(DINOv3)伤害随难度增大(corr +0.66,t8 −10),VLA 自身空间(So400m)精度无损(−0.38,难度无关) | pi05 A1 vs A2 三方 4seed×50trial 定稿;§4.19/4.20 So400m≥DINOv3 同协议平反 | **已支持** | — | §4 |
| C3c | 诊断工具:MDN 模式坍缩检测(π 分布/切换段数)区分"信号退化"与"注入失败" | dino-LMWM π=[0,0,.99,0] 0 段 vs so400m 14 段 | 已支持 | — | §4/附录 |
| C4 | **化解配方**:双尺度并联(不 swap)+ 残差目标 + 相位自适应 drop(+2Q 容量) → 保指引(t6 +8.5, p<0.001)且救精度(t7 100 回满、t8 90→94) | E1/2Q/tsched 多 seed 系列;t6 85.0 vs no-WM 76.5(t≈6) | 部分支持(逐项);**组合净增益待 G1** | G1 + dual(t+7,t+7) 同尺度对照(证互补来自尺度差) | §5-6 |
| C5 | **Headroom 净增益**:LIBERO-10 饱和(94-96)掩盖增益;在 RoboTwin 上 MINT-VLA 聚合 SR 显著超基线 | robotwin pi05 基线在训(9c8dr);LMWM robotwin 特征已抽 | **needs experiment(G2)** | RoboTwin milestone pairs + LMWM ckpt + MINT-VLA 训练 + eval | §6 |
| C6(次) | 张力与机理**跨架构可复现**(LaWAM/starVLA ↔ pi05/openpi 两套独立栈) | pi05 A0/A1/A2/A2-suffix 全套 + LaWAM 全系列同构结论 | 已支持 | 表述层面对齐两侧任务映射 | §4/6 |
| C7(次,附录) | 控制侧正交增益:RTC 前缀连续性在最难精度任务 +10pt,与信号侧修复可复合 | t8: RTC h=5 80% vs naive 70% | 已支持(n=20,单任务) | 扩 seed(可选) | 附录 |

**声称纪律**(继承 §4.16):聚合差异 <1.5pt 一律不声称;per-task 判据只用低方差任务(t6 std≈2,不用 t8 单独作判据);所有主表多 seed ± std。

---

## 2. 结构(9 页,7 节)

### §0 Abstract(~200 词)
- What:发现并化解 WM 子目标注入 VLA 的 precision↔guidance 零和张力。
- Why hard:张力使一切"单一全局平衡"(共享 query/双 query/CFG)只做重分配,聚合封顶——解释了为何该类方法在饱和基准上"看不见收益"。
- How:双尺度并联 + 残差目标 + 相位自适应 dropout;信号侧由免聚类 recurrence 场提供。
- Evidence:两套独立 VLA 栈复现;LIBERO 难任务 +8.5(p<0.001);RoboTwin 聚合 +X(G2 待填)。
- Most remarkable:**(G2 后定)** RoboTwin 聚合首次净增益 / 或跨架构张力定律(corr +0.66 难度-伤害)。

### §1 Introduction(1.5 页)
- Hook:VLA 已能整段模仿,但长程任务需要"下一步该到哪"的子目标指引;世界模型天然产生这种信号——**为什么把它接进 VLA 反而常常不涨点?**
- Gap:现有 WM-VLA(DreamVLA/WorldVLA/AHEAD 类)报聚合;聚合掩盖 per-task 零和(OpenVLA Long 53.7 vs avg 76.5;π0.5 92.4 vs 96.9 同现象)。无人诊断"注入为何零和"。
- One-sentence contribution(上文)。
- Contributions(4 条,对应 C2/C3/C4/C5):①跨架构张力现象与定律;②两大机理根因(冗余/空间错配)+ 坍缩诊断;③双尺度残差化解配方;④headroom 基准净增益 + 免聚类 recurrence 信号源。
- Results preview:t6 76.5→85.0;RoboTwin +X;张力图。
- **Hero figure(Fig 1)**:见 Figure Plan。
- Front-loading check:标题+摘要+Fig1 即可让 skim 读者复述"张力—根因—化解—净增益"。

### §2 Related Work(1 页,按方法族组织)
1. **WM/未来预测条件化 VLA**:DreamVLA、WorldVLA、AHEAD、LaWAM——都预测固定 horizon/边界目标,报聚合;我们:诊断注入机制 + 残差/尺度设计。
2. **子目标/分层策略**(HIRO 系、subgoal diffusion):子目标来自 RL/规划;我们:免训练跨-ep 检索统计量,且聚焦"注入端"failure mode。
3. **跨-ep 检索信号**:**UR-VC(arXiv 2607.12892)最近邻**——检索骨架撞车,差异化三点:时间标签均值(标量 progress,时间代理天花板)vs 密度场三读法(时间无关);τ=0.3 时间带 hack vs 全时间无关;止步标量 vs 分割/WM 目标/蒸馏参数化。列为 baseline(D6)。
4. **动作分块与执行**:ACT/Diffusion Policy/RTC——本文 C7 与 §4 ease-in 分析衔接(附录)。

### §3 Recurrence Signal & LMWM(1.25 页)
- r(o) 定义(kNN 密度、median 带宽、来源多样性加权);三读法(幅值/谷/脊);零超参跨本体。
- r-脊 → milestone 目标;蒸馏为参数化 LMWM(gen/prd,MDN);部署零检索。
- 与 C1 证据:内在 2.1× + 跨本体表;r-脊 vs 边界的下游对照(补)。

### §4 Diagnosing Naive Injection(1.5 页)——**论文的心脏**
- 现象:LaWAM 侧零和表(E1/2Q/CFG 全 94.8 封顶,per-task 重分配图);pi05 侧独立复现。
- 定律:伤害-难度相关(外挂空间 corr +0.66 vs 自身空间 +0.06)。
- 根因一(冗余):残差判别力 10×;条件结构论证(h_t 在场)。
- 根因二(空间):A1 vs A2 精度类 Δ(−1.48 vs −0.38)。
- 诊断工具:MDN 坍缩检测(dino 0 段 vs so400m 14 段)。
- 小结:任何"单一全局平衡"的注入必零和 → 引出 §5。

### §5 LMWAM-DS: Dual-Scale Residual Conditioning(1.25 页)
- 架构:局部 t+7 通道(守精度)∥ 全局 milestone 通道(守指引),不 swap;2Q 独立容量;残差目标 `ms − h_t`;相位自适应 drop `p(t)=0.15+0.85·t^γ`(精步退指引)。
- 每个组件对应 §4 的一个根因(表:根因→设计→消融)。
- 训练/推理一致性;完全向后兼容(env-gated)。

### §6 Experiments(2 页)
- Setup:LIBERO-10(多 seed 协议,SEM 声明)、RoboTwin 2.0(48 任务/clean+random)、两套 VLA 栈。
- 主表 1:LIBERO-10 per-task+聚合(no-WM / LaWM-t+7 / naive-LMWM / +dual / +2Q / +tsched / **+residual**),多 seed±std。
- 主表 2:RoboTwin 聚合+分任务(baseline vs MINT-VLA)——**G2 主结果**。
- 跨架构表 3:pi05 A0/A1/A2/A2-suffix(空间与注入点)。
- 消融表 4:dual(t+7,t+7) 同尺度对照(证互补来自尺度差)、残差 on/off、1Q/2Q、tsched γ、drop 率;UR-VC baseline。
- 分析:t6/t8 逐任务演化图(零和被打破的可视化)。

### §7 Conclusion + Limitations(0.5 页)
- Limitations(诚实):LIBERO 聚合饱和,增益主要显现于难任务与 headroom 基准;残差目标依赖 WM 质量(坍缩需先诊断);真机验证留待后续。
- Future:P3 同编码器空间终局(So400m 证据已埋线);r-场部署期 OOD 监控。

---

## 3. Figure Plan

| ID | 类型 | 内容 | 数据源 | 优先级 |
|---|---|---|---|---|
| **Fig 1** | Hero 三联 | (a) 零和张力:per-task Δ 蝴蝶图(t6/t9↑ 绿 vs t7/t8↓ 红,三种混合聚合全 94.8 平线);(b) 两根因示意:绝对目标=当前态+残差(86% 冗余),外挂 vs 自身空间伤害-难度散点(corr 0.66/0.06);(c) LMWAM-DS 架构(双通道∥+残差+相位 drop)+ 头条 bar:t6 76.5→85.0、RoboTwin +X | 已有数据+手绘 | HIGH |
| Fig 2 | 概念+实例 | r(o) 场在真实 episode 上的幅值/谷/脊三读法;跨本体同一超参 | crave/lmwm 现成 | HIGH |
| Fig 3 | 折线/热图 | 零和被打破:变体系列(naive→E1→2Q→tsched→resid)的 t6 与 t8 轨迹,聚合曲线首次脱离 94.8 平台(G1 后) | eval json | HIGH |
| Fig 4 | 双面板 | 残差 vs 绝对判别力(CV 轨迹);MDN 坍缩 π 分布对比 | pi05 诊断脚本 | MED |
| Table 1-4 | 表 | 见 §6 | 多 seed eval | HIGH |
| Fig A1 | 附录 | RTC 前缀连续性 & ease-in 剖面 | t8 诊断 | LOW |

Hero 图 caption 草稿:*"Naive world-model subgoal injection is zero-sum across tasks (a); we trace it to target redundancy and representation-space misalignment (b), and resolve it with dual-scale residual conditioning (c), yielding net gains where headroom exists."*

---

## 4. Citation Plan(全部 [VERIFY],不凭记忆写 BibTeX)
- §1:π0 [VERIFY]、π0.5 [VERIFY]、OpenVLA [VERIFY]、LIBERO [VERIFY]、RoboTwin 2.0 [VERIFY]
- §2-WM:DreamVLA / WorldVLA / AHEAD / LaWAM 原文 [VERIFY]
- §2-检索:**UR-VC arXiv 2607.12892 [VERIFY]**(必引+必比)
- §2-分层:HIRO 等 subgoal 系 [VERIFY]
- §3:DINOv3 / SigLIP-So400m / MDN(Bishop 1994)[VERIFY]
- §5:CFG(Ho & Salimans)[VERIFY];RTC(real-time chunking)[VERIFY]
- 协议:kNN 密度/median heuristic 经典引 [VERIFY]

---

## 5. 实验缺口清单(映射到在跑任务)

| ID | 实验 | 状态 | 支撑 |
|---|---|---|---|
| E1 | v8xpb 残差 V8 eval(同口径多 seed) | 训练中(北京) | **G1/C4** |
| E2 | armB baseline 多 seed(≥4) | 未起(1 次 eval ~2h,本机可跑) | G3/所有聚合表 |
| E3 | RoboTwin pi05/LaWAM baseline | 9c8dr 训练中 | G2/C5 |
| E4 | RoboTwin milestone pairs + LMWM ckpt + LMWAM-DS 训练 + eval | 管线部分就绪(特征已抽) | **G2/C5(最长线,立即排期)** |
| E5 | dual(t+7,t+7) 同尺度对照 | 未起(12500 步一次训练) | C4 消融核心 |
| E6 | UR-VC baseline 复现(6 公式) | 未起 | §2 差异化 |
| E7 | r-脊 vs 边界下游对照 | 可复用已有 ckpt | C1 |
| E8 | pi05 侧 a2_res / a1_suffix 收尾 | 训练中 | C6 补全矩阵 |
| E9(可选) | LIBERO-40 泛化、RTC 复合、多 seed 扩 t8 | — | 加分项 |
| **E10a** | Task_A 真机数据**离线**分析:r-场三读法/残差判别力/milestone 收敛性(kai0 crave bank 现成) | 未起,零机器人成本,**现在可做** | C1 跨本体(真机数据侧) |
| **E10b** | Task_A 真机**闭环 A/B**:LaWAM基线 vs LMWAM-DS,双臂叠衣服,20-30 trials/臂,阶段式计分(铺平/折1/折2) | **门控:G1 过后启动(≈8月下旬),9/7 硬止损**;sim01+RTC 部署栈现成 | 对标 UR-VC 真机双臂叠布;kill "sim-only";可能是 LMWM>LaWM 最大 delta 的展示(可变形物 milestone 收敛 vs t+7 噪声目标) |

## 6. 时间线(倒排,截稿≈9 月下旬)
- **7/24–7/31**:E1/E2/E3 落地 → **过 G1**;锁 §4 全部图数据;E5/E6 提交。
- **8/1–8/15**:E4 RoboTwin MINT-VLA 全线(pairs→LMWM→训练);E7。
- **8/16–8/31**:E4 eval → **过 G2、锁 Story**;**E10b 真机叠衣服 A/B 启动(G1 已过为前提,与 sim 不抢资源)**;补洞实验;冻结 sim 数字。
- **9/1–9/7**:E10b 收尾硬止损(赶不上→rebuttal 弹药)。
- **9/1–9/14**:/paper-figure → /paper-write → /paper-compile;外审循环(auto-review-loop-minimax/llm,Codex MCP 当前不可用)。
- **9/15–截稿**:/result-to-claim + /paper-claim-audit 数字核对;reproducibility statement;润色。

## 7. 风险与对策
1. **G2 失败**(RoboTwin 无净增益)→ Story C 诊断论文:补第三注入点/更多任务的机理面(E8/E9),卖跨架构定律。
2. **G1 失败**(残差在 MINT-VLA 不涨)→ 残差保留为 pi05 侧机理证据;C4 依赖 dual+2Q+tsched,标题去 "Resolving" 改 "Diagnosing…and Mitigating"。
3. **基线多 seed 后聚合差距缩小** → 有利(平台效应更实),但须如实改写 §0/§1 数字。
4. **UR-VC 先发压力** → §2 差异化表 + E6 实测对比;引用姿态"他们证明检索能修时间代理,我们证明检索统计量本身是更普适的信号"。
5. 页数超 → §3 的 r-场推导、C7 RTC、坍缩诊断细节全部入附录。

## 8. Next Steps
- [ ] 过 G1:v8xpb eval + 深析(自动触发)
- [ ] E2 armB 多 seed(本机今起)
- [ ] E4 RoboTwin MINT-VLA 管线排期(最长线,先行)
- [ ] E5/E6 提交
- [ ] G2 后锁 Story → /paper-figure → /paper-write

## 9. 缺口审计(2026-07-24 全局自查)
**架构** A1 残差尺度失衡(MSE缩~16×,ms通道或饿死;v8xpb裸奔中,逐通道loss仅在wandb二进制)→训完先解析wandb再eval+提交归一化/λ_ms对冲臂|A2 缺2q+resid(无tsched)归因臂|A3 phase-drop仍是全局调度,与自家"单一平衡必零和"论证相抵→limitations如实写|A4 故事-工件一致性:rvalley命名暗示谷导出milestone vs §3卖ridge;V8绕过prd而pi05用prd(dino版坍缩)→审计pairs生成链+分路径描述|A5 pi05侧修复组合(a2_residual_suffix)未训
**框架** B1 ⭐+8.5归因混淆:是vs no-WM;armB(n=1)t6=84→LMWM相对LaWM边际或仅+1~5;E2多seed基线+E5同尺度对照是切割关键,均未起跑|B2 corr+0.66仅10点无CI单栈→bootstrap+LaWAM侧同算;指引/精度分类需操作化定义防循环|B3 ridge>boundary无下游证据(E7未排)|B4 UR-VC对比harness未定义(加权vs条件化)|B5 G2隐藏前置:需LaWAM栈robotwin基线(≠在训的pi05版),SR落点未知,若<20%即自家RoboDojo区|B6 主表seed不均(E1单次/armB n=1)→seed补齐扫
**执行** C1 已宣布未起跑:E2/E10a/图前置(张力图CV图坍缩图数据已冻结)|C2 G1单点押注无对冲|C3 监控跨会话即死×3、逐通道loss无文本可见、开源匿名化未排期|C4 E10b人工复位/评分档期未约
**行动队列(按序)** ①E2 armB多seed(本机即起) ②对冲臂2q+resid±归一化(北京) ③v8xpb wandb通道体检→eval ④E10a离线分析 ⑤图前置 ⑥LaWAM robotwin存量基线快评(定G2可行性) ⑦E5/E6/E7排program ⑧开源清理排期

## 10. ⭐ G2 初步通过(2026-07-24 存量数据挖掘)
North-E 已存在 7/19-20 跑完的 robotwin blocks-6 对比(**两臂均 steps_20000, demo_clean, 50 trials×4 seeds**):
| task | LaWM baseline | LMWM dual2q | Δ |
|---|---|---|---|
| beat_block_hammer | 0.5±0.9 | **20.0±2.4** | **+19.5**(p«0.001)⭐headline |
| blocks_ranking_rgb | 98.0±1.4 | 98.0±1.4 | 0(饱和) |
| blocks_ranking_size | 88.5±4.6 | 88.0±6.0 | −0.5 |
| handover_block | 91.5±0.9 | 90.0±4.5 | −1.5(噪声内) |
| stack_blocks_two/three | 0 | 0 | 0(无competence→无杠杆,自证RoboDojo论点) |
| **聚合** | 46.42±0.76 | **49.33±1.45** | **+2.92**(>1.5声称线, t≈3.6) |
**定律教科书式复现**:饱和任务不动、零能力任务不动、有headroom的多阶段任务大涨且无精度伤害。
**E4 重估**:robotwin milestone管线/LMWM/双臂训练**已存在**(比计划假设领先两周);剩余=①残差臂上robotwin(复用管线,一次训练)②可选扩任务/写作级复核。⚠️范围如实声明:训练/评测均为blocks-family子集(策略只训了该子集)。

## 11. ⭐ E2 结果:96.4 基线是幻影(2026-07-24)
armB 多 seed(n=4, 同口径): **聚合 94.20±0.97**(单评96.4=上尾, 弃用);t6=82.5±3.8;t7=100±0;**t8=89.5±12.4(seed1=68)**;t9=86.0±5.5。
**叙事修正**:
1. "MINT-VLA 追不上 96.4"作废——LaWAM 侧全变体(基线94.2/E1 94.8/2q 94.8/tsched 95.2)**统计平局 @ 94-95 平台**;P1 旧判据(聚合≥96)对着噪声上尾打。
2. **t6 归因分解**:no-WM 76.5 → LaWM +6.0 → LMWM 再 +2.5(弱显著)。"+8.5 全归 LMWM"是错写法。
3. "LMWM 伤 t8"在 LaWAM 侧不成立(基线自己 ±12.4);精度伤害证据收窄为 **t7(100→96)+ pi05 外挂空间定律(corr+0.66)**。
4. **Story 权重进一步压向 robotwin**(hammer +19.5/agg +2.92 是目前最硬的净增益)+ pi05 跨架构定律。LIBERO 角色=平台+per-task结构分析, 不再当聚合战场。
5. 判据重写:G1/G3 比较基准=94.20±0.97;主会话已判 v8xpb 残差=精度旋钮(+11.8精度/−9.3指引, agg+0.21);rl4jj eval(26dlj)裁"去tsched后残差翻转是否保持"。

## 12. ⭐ rl4jj 归因臂结果:残差(无tsched)=全家族最高(2026-07-25)
2q+resid(scale=1, 无tsched), 8seed中6路成功(seed1/4启动失败零输出=无选择偏差, 42ck5回填中):
**聚合 95.60±0.82**;t6=87.0±3.8(+4.5 vs armB);t9=91.7±2.1(+5.7);t7=100±0;t8=86.0±7.1(armB自身±12.4, 方差内)。
**解读**:①指引↑+精度不伤 = 正是 C3a"去冗余指引信号"的预测形状;②与 v8xpb(带tsched: 精度+11.8/指引−9.3)形状相反 → **残差×tsched 交互**:tsched的相位drop把残差通道扭成精度旋钮, 残差单独用才是指引增强;③更简配置更优 = C6优雅性加分, tsched可弃;④vs 修正基线 +1.40(t≈2.4)——差0.1过1.5声称线, n=8后再判;3/6 seed 摸到旧"96.4"。
**下一步**:42ck5 补 n=8 → 若 +Δ≥1.5 → LIBERO 聚合也可声称, Story A 完整度大增;52rnr(scale=4)已排上卡, 出结果补尺度消融。

## 13. ⭐⭐ G1 定稿:残差破线(2026-07-25)
seed1/4 槽位 4 连败(EGL abort, 种子值已排除, 疑同一毒节点)→ 本地 gf0 补 seed8/9(需 ckpt 伴随 config.yaml+dataset_statistics.json + SUITES=libero_10, 两坑已记)。
**n=8 定稿: 2q+residual(无tsched) 聚合 95.75±0.76**, Δ+1.55 vs armB(94.20±0.97, n=4), Welch t≈2.80 → **过 1.5 声称线** ✅。
per-task: t6=87.5±3.8(+5.0) t9=90.8±2.7(+4.8) t7=100±0 t8=87.0±6.3(方差内)。
**判定**: G1 过(修正参照系)。LIBERO 主表主行确定 = 本配置;tsched 弃(残差×tsched 交互写附录);52rnr(scale=4)出后补尺度消融行。
**Story A 现状**: LIBERO 聚合可声称 + RoboTwin(+2.92/hammer+19.5)双腿齐 → 大纲 §6 两主表数据源就位, 图前置可全面开工。
**协议注**: n=8 = 6×North-E H20 + 2×gf0 A100(同代码/数据/egl/50trials), 混机如实脚注;seed 集 {0,2,3,5,6,7,8,9}(1/4 因基建故障弃, 与结果无关)。

## 14. ⭐ 4-suite 冲击与决定性实验(2026-07-25)
主会话判决(见 project_libero_4suite_lmwm_negative / RESULTS_libero_4suite_2026-07-25.md):
**dual2q(绝对milestone)完整LIBERO净负−0.62**——spatial −3.0~−5.8 铁证(n=8 t=−9.76 + 独立ckpt t=−8.91), goal/object饱和持平。只报libero_10=掩盖, 审稿一击即穿。RoboTwin +2.9 但集中hammer且baseline欠训(绝对数需训到位)。
**冲击面**: §13 的 G1"过线"须限定为 libero_10-scope;dual2q 行在4-suite主表为净负;LaWAM Table1 对齐口径=4-suite。
**⭐残差×spatial = 当前最高价值实验(已发车, 本地osmesa 4seed)**: spatial伤害机理=跨-ep别名的场景身份成分, 恰是残差减掉的东西。
判读: resid-spatial ≥96 → "残差修复净负"(Story A完全体: 全基准无害+长程/机理增益+RoboTwin);≤93 → 退守"选择性增益"命题(主会话定调), 诊断补"有害成分幸存于delta"。
协议: osmesa对齐主会话干净口径; 对照 nowm 96.5±0.8(n=8)/dual2q 93.2±0.3。rl4jj训练数据=同libero_merged, 评spatial合法。

## 15. 预注册记录(v2 纪律: 主张→预测→实验)
**PR-1(2026-07-25, 注册于 mm9fj 出分之前, 见会话记录)** 残差×spatial(n=8, osmesa, 对照 nowm 96.5±0.8 / dual2q 93.2±0.3):
- 预测: 聚合 **≥95.0**(修复 dual2q −3.3 缺口的 ≥2/3)。机理: spatial 伤害载体=绝对目标中的场景身份成分(跨-ep 别名), 残差已减除该成分。
- 证伪线: **≤93.5** → 冗余机理对 spatial 伤害的解释被证伪, 伤害另有来源(如 delta 本身别名)。
- 附带预测: 衰减对照(绝对×0.25)只能部分修复(<残差), 因衰减保留身份成分。
**PR-2(待注册)** 结构度/别名度定律盲测: 在 LaWAM-LIBERO 拟合 Δ≈f(结构度,别名度) 后, 先发布 robotwin 逐任务 Δ 与各套件预测, 再对答案。

## 16. PR-1 判定 + spatial t5 解剖(2026-07-25)
**PR-1 判定: 部分修复, 两线之间**(94.40±0.84; 预测≥95未达, 证伪线≤93.5未触)。
**per-task 解剖(决定性)**: spatial 套件级回归 = **单任务 t5** ("black bowl ON the ramekin"):
nowm 76 → dual2q 42 / dual2q#2 18(跨独立训练不稳定) → **resid 52**;其余9任务所有变体 97-100 无伤。
**机理闭合**: t5 与 t1("next to ramekin")/t0("between")同物体集只差空间关系 = 跨-ep 检索别名最坏情况。
绝对目标 = 身份成分+目标成分双错 → 灾难;残差除身份成分, 但 delta 仍源自别名 milestone → 部分修复。
**对定律的意义**: 单位分析必须是任务级; suite/benchmark 聚合全部在掩盖单任务效应(hammer +19.5 / t5 −34 / t6 +5)。
**PR-2(2026-07-25 注册, 先于计算)**: 用特征离线计算跨-ep 检索混淆分数(episode 的近邻落在他任务episode 的比率, 需先建 40 任务混淆矩阵):
预测① t5 的混淆分数是 spatial 组极端离群值(主要混向 t0/t1);②40 任务的逐任务 Δ(dual2q−nowm) 与混淆分数负相关、与结构度正相关;
③(设计推论)检索置信门控/任务条件化检索应把 t5 拉回 ≥65。证伪即定律死。

## 17. 文献深调研落地(2026-07-28, 详见 RESEARCH_latent_subgoal_VLA_2026-07-28.md)
- **Novelty 空位确认**: 残差/delta 目标零先例; 任务级分析无人低于 per-suite; 系统性失效诊断空位。AHEAD=唯一自适应 horizon 先例(必引对照)。
- **三条独立失效证据入 §4**: DreamVLA 竞争梯度(depth/sem-only 低于无预测) / FutureVLA 裸注入掉点(62.5→58.4, 解耦+门控才转正) / WorldVLA 历史动作污染(需mask); + WAM对照(2603.22078)证收益条件性(RoboTwin robust vs LIBERO-Plus 反转)。
- **Related work 四家族**: 追加query+结构化mask(DreamVLA) / token替换(AHEAD) / 纯辅助loss(WorldVLA, SF) / 门控cross-attn(FutureVLA); + 检索式(UR-VC)。
- **基线表参照**: MoLA 97.0/CALVIN 4.55(最强) · FutureVLA 98.3(自报) · SF +1.4无误差棒(方法论批评点)。
- **设计导入**: 门控臂(检索置信/残差范数门控, 对症 t5, 与主会话幅度门控合流)优先级提到 P0; DreamVLA 解耦mask 作残差的替代/互补消融; dynamic-only发现=我们时间残差的空间同构旁证。
- 引用计划新增: 2507.04447/2605.12167/2606.02486/2603.10712/2506.21539/2510.12276/2603.22078 [VERIFY]。

## 18. Workshop 支线(2026-07-28 立项, 任务#4)
IROS'26 Physical World Models Workshop(10/1 匹兹堡): **8/10 截稿, 4-8页非存档** → 诊断核心时间戳防抢发 + ICLR 前同行反馈 + 提前写掉 §4。
骨架已建: `lmvla/paper_iros_pwm_workshop/`(IEEEtran, 共享 numbers.tex, 编译通过)。
标题: "When Do World-Model Subgoals Help VLAs? Redundancy, Aliasing, and a Residual-Target Fix"。
排期: 写作 8/3-8/8, 提交 8/9。待核: workshop 提交口/模板细节、ICLR 非存档双投条款、WorldArena 2.0 原始出处(现为二手源)。
新增第4条失效证据: WorldArena 2.0 "最优画质预测器 100%↔0% 任务翻转, 无预测 ACT 基线 80%"(质量≠效用)。

## 19. 治理定调(2026-07-28, 2026-08-01 更名): ICLR MINT-VLA = 唯一主线
MINT-VLA 展开为 *Milestone Integration with Native-Space Targets for Vision--Language--Action Models*；$\pi_{0.5}$ 是主实验实现而非方法名称的一部分。跨 VLA 架构适配属于 T6 实证任务，在完成前只称为 architecture-compatible design。
1. **IROS workshop = 主线的只读快照**: 只消费 8/8 前已冻结的数据, **不为 workshop 跑任何专属实验**; 写作时间盒 ≤1.5 天; 非存档保护 ICLR 双投。
2. **内容分层防自我抢发**: workshop 只发"诊断核心 + 残差 LIBERO 结果"(已冻结部分); **ICLR 独占**: 任务级定律(PR-2)、门控设计(P0-C)、RoboTwin 训到位+残差臂(P1-E)、跨栈迁移(P1-G)、真机(E10)、r(o) 普适性与 V8 完整故事。workshop 里这些只以 open problems 口吻带过。
3. **资源仲裁**: GPU/执行 agent 带宽冲突时一律让主线; 所有 P0/P1/N 实验的立项理由必须是"ICLR 需要", workshop 顺带受益。
执行现状同步(执行 agent 已推进): P0-B 已提交(tn9xc/rlq6x), P0-A 跑中, P0-D 前向探针完成(注意其更正: 推理期 provider 不参与, 已建 LMWM_SWAP_HINT 注入点; 前向≠闭环, 闭环 SR 探针备好未跑), RoboTwin 逻辑张力的定律自洽解释已入草稿。

## 20. 近端未来调研落地(2026-07-29, 详见 RESEARCH_nearterm_future_2026-07-29.md)
判决: t+7 注入正确(horizon 倒U甜区3-7步, 文献背书); AHEAD 静态patch保留=残差的空间同构(独立收敛); 机理=表征塑形+粗条件化非精确渲染; causal confusion 风险→离线指标不可认证、只认闭环SR。
新实验: **A hint捷径探针**(零训练, 防copycat) / **B ms-only臂**(1训练, 补2×2通道消融=文献空白)。
门控修订(已议): 门控信号改推理可得(残差范数/不确定度), 加检索任务条件化对照臂, eval扫CFG_GUIDANCE。

## 21. ⭐ 骨架 v3(2026-07-29, 判决重构: 两根因+两修复+诚实证伪)
**Claims v3**:
| C | 主张 | 证据 | 状态 |
|---|---|---|---|
| C1 现象 | 张力/平台+任务级重分配(单任务驱动套件回归: t5) | 多seed平台+t5解剖 | 硬 |
| C2 根因① | 目标冗余(全库95%, 1693ep 100%一致, 15×) | N4 | 硬 |
| C3 根因② | **WM梯度污染共享编码器**(detach→spatial 93→95.85, libero_10不伤) | shapeB A/B | 硬(干预级) |
| C4 修复① | 残差去冗余(+1.55) | rl4jj n=8; P0-B/N2 认证中 | 经验硬/机制待认证 |
| C5 修复② | 梯度隔离 | shapeB | 硬 |
| C6 收益条件 | RoboTwin balanced stack_two +6.5(t3.4)/+9.0(t3.6), Hard净+2.0; LIBERO饱和null(pi05双编码器4套件) | balanced eval | 硬(Qwen骨干); pi05侧待#39 |
| C7 方法论 | 预注册证伪自己两个假说(别名律P0-A、可预测/难度corr N5)——负结果入文 | PR记录 | 硬 |
**死亡登记**(不得再卖): 别名机制归因、混淆分数可预测律、难度-伤害corr(降为t8单点观察)。
**t5 重新归因**: 现象保留(单任务解剖), 机制改为梯度污染(shapeB修复spatial为干预证据); ⚠️待查: shapeB per-task 是否显示 t5 本身回升(归因闭环的最后一环)。
**结构**: Intro → 现象(平台+任务级) → 根因①+修复①(残差) → 根因②+修复②(隔离; DreamVLA"competing gradients"轶事的受控版=文献回声) → 收益条件(RoboTwin/饱和) → 诚实证伪节(卖预注册纪律) → Related → Discussion。
