# LMWM 论文实验程序 & 数据/计算估算(2026-07-26)

> 论点(定版): **milestone 世界模型选择性帮助"子目标结构强"的任务, 中性到有害于无子目标结构的任务**;
> 用 LIBERO(诚实负 + 机制)+ RoboTwin(修复后的选择性正)两条腿做扎实, 是一篇"milestone WM 何时有用"的机制论文。

## 核心结论
- **下载量 ≈ 0**: RoboTwin 源(323G, 27500ep, stack/hammer/全任务齐)+ LIBERO 均已本地。至多下 DINOv3/so400m 权重(几 GB, 大概率已有)。
- **真成本 = 计算(重处理 + 重训)**, 不是下载。
- **RoboTwin 重训非可选**: 当前结论全压在 5 个漏网 hammer(pairs curation bug, 见 `RESULTS_robotwin_P2` 附2.2)+ 0 数据 stack 上, 太薄。**不重训就没有能写的 RoboTwin 结果。**

## 为什么重训而非补下载
源数据什么都有, 是 milestone pairs 管线 bug(只覆盖 ep 号 672~4949 + hammer 无子目标被过滤)丢了 stack(1165ep 全在号段外)和 hammer。而 **stack 恰是验证论点的理想正例**(pick-place-pick-place 清晰子目标), hammer 是**天然负例**(单快动作无子目标)。修复=本地重跑 pairs + 重训。

## 实验程序(按论文必需度分档)

### 🔴 CORE(没有就没论文)
| # | 实验 | 数据 | 计算 | 证明 |
|---|---|---|---|---|
| 1 | LIBERO 4-suite: LaWM vs LMWM | 本地(已有) | 已基本完成 | LIBERO 诚实负/选择性 |
| 2 | LIBERO detach 消融(进行中 t-...-86qxh) | 本地 | 1×8卡 ~半天 | 伤害=结构缺陷 vs 概念极限 |
| 3 | 重建 RoboTwin pairs(均衡含 stack) | 本地重处理 | 特征重抽 ~3000ep, 数 GPU-时 + grid ~几百GB本地磁盘 | 修 curation bug |
| 4 | RoboTwin 均衡重训 LaWM vs LMWM | 本地 | 2×8卡到收敛 ~各 1-1.5 天 | 核心正论点: 帮 stack 不帮 hammer |
| 5 | RoboTwin Easy+Hard eval | 本地(Hard 靠 env 随机免数据) | eval only | 鲁棒性口径 |

### 🟡 RIGOR(审稿防弹)
| # | 实验 | 计算 |
|---|---|---|
| 6 | LIBERO 残差/衰减对照(52rnr + 绝对×0.25 + 随机同范数) | 2-3×8卡 |
| 7 | RoboTwin on-protocol 单任务 50-demo(至少 stack)→ 可对 leaderboard | 2×8卡 |
| 8 | 换-hint 因果探针(P1 机制) | eval only |
| 9 | 多 seed 复现关键 eval | eval only |

## 总量估算
| 项 | 量 |
|---|---|
| 下载 | ~0(源全本地; 至多 DINOv3 权重几 GB) |
| 生成数据(本地磁盘) | RoboTwin 均衡 grid 特征 ~几百 GB(评完可删) |
| 训练运行 | CORE ~5 + RIGOR ~5 = **~8-12 个 8卡训练** |
| GPU-时 | 粗估 **~15-25 个 8卡·天**(北京 H20 队列) |
| eval | ~6-10 个多卡作业 |

## 定调建议
1. **最高价值单动作 = 重建含 stack 的均衡 RoboTwin pairs + 重训(#3+#4)**。stack=天然正例、hammer=天然负例, 一次均衡重训把论点从"LIBERO 负 + 薄 RoboTwin"升级成干净的正面机制证据。
2. **先别扩数据集**(不引第三 benchmark); 两条腿做扎实。
3. ~~连带风险: LIBERO pairs 是否有同款 bug~~ **已核(2026-07-26): LIBERO 干净**。`libero_rvalley`(训练用)+ finalarch 对全 1693ep、4 suite **100% 覆盖**(spatial 432/432, object/goal/long 全覆盖, 号段 0~1692 完整); target_compact 98.7-100%(仅缺 10ep)。→ **spatial −3.0 负结论不受 coverage 污染, 是实打实的**。为什么 LIBERO 免疫: merged 小(1693ep 连续)被 milestone 管线全量处理; RoboTwin 巨大(27500ep)只跑前段 → 是规模/scoping 问题, 小数据集天然免疫。

## 状态
- P1 detach 消融进行中; P2 40k 判决完成(但因 curation bug 需重训才可写)。
- 关联: `RESULTS_robotwin_P2_2026-07-25.md`(附2.2 根因), `INVESTIGATION_lmwm_architecture_flaw_2026-07-26.md`。
