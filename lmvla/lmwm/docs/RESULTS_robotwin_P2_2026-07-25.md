# RoboTwin P2 判决: LMWM dual2q vs LaWM baseline(2026-07-25)

> 北极星: LMWAM(=LMWM×LaWAM)在 LIBERO + RoboTwin 达 SOTA。
> P1(LIBERO)已收口: zero-sum 结构性, 门控/残差/base 全卡 95.0-95.4(见 PROGRESS §8/9/10)。
> 本文档 = P2(RoboTwin)首个判决 eval。

## 判决 eval 设置
- **任务**: 6 积木族(beat_block_hammer / blocks_ranking_{rgb,size} / handover_block / stack_blocks_{two,three})。
- **口径**: 2ckpt(A=LaWM baseline `123516+robotwin_baseline_lawm_northe/steps_20000`, B=LMWM dual2q `rt_lmwm_dual2q_20k/steps_20000`)× 4 变 seed, 同 8卡节点同环境, 每任务每路 50 局。
- **作业**: `t-20260725082637-hp2d8`(北京 8×H20, Preemptible: false)。
- **踩坑**: RoboTwin sim wrapper `robotwin_python_wrapper_northe.sh` 随 lawam submodule 重置丢失, 已按 setup 文档配方重建(json_numpy 在 tim/robotwin_client_deps, 见 memory)。

## 结果(n=4 seed 各臂)

| task | A baseline | B dual2q | Δ(B−A) | Welch-t |
|---|---|---|---|---|
| **beat_block_hammer** | 0.5%±0.9 | **20.0%±2.4** | **+19.5** | **+13.0 \*\*\*** |
| blocks_ranking_rgb | 98.0%±1.4 | 98.0%±1.4 | 0.0 | ns |
| blocks_ranking_size | 88.5%±4.6 | 88.0%±6.0 | −0.5 | ns |
| handover_block | 91.5%±0.9 | 90.0%±4.5 | −1.5 | ns |
| stack_blocks_two | 0.0% | 0.0% | 0.0 | ns(双方地板) |
| stack_blocks_three | 0.0% | 0.0% | 0.0 | ns(双方地板) |
| **聚合6任务** | **46.4%** | **49.3%** | **+2.9** | **+3.08** |

## ⭐ 判定: P2 方向成立 —— LMWM 在 RoboTwin 有真净增益(与 LIBERO 零和相反)

- LIBERO: LMWM 零和(精度↔指引互抵, 聚合平 ~95)。
- **RoboTwin: LMWM 净赢 +2.9pt(t=3.08 显著)** —— 未饱和基准上的真增益, LMWAM 方向值得继续。

## 三个结构性观察 → P2 下一步
1. **增益集中 beat_block_hammer(+19.5, t=13, 40×)**: LMWM milestone 的时序/相位信号帮了"击打时机"。**要理解机制并推广**(其他时序敏感任务是否也吃这红利)。
2. **两 stack 任务双方地板 0%**: 堆叠是**基座能力缺口**(非 LMWM 问题)。提聚合的最大杠杆 = 解决堆叠(更长训练/更强基座/长程 milestone)。
3. **rank/handover 已 88-98% 近饱和**, headroom 小。

## ⚠️ 口径警告
baseline beat_block_hammer=0.5% vs `lawam_robotwin_sft_release` 单任务 90% → **这批 20k ckpt 相对完整 SFT 欠训**。A/B 对比(同recipe只差LMWM)干净有效, 但**绝对 SOTA 对比需先训到位**。

## P2 下一步(优先级)
1. **beat_block_hammer 机制深挖**: 为何 LMWM 让 0.5→20? milestone 在此任务学到什么相位? → 推广到其他任务。
2. **解决 stack 地板**: 训到位(20k→更多步)看 stack 是否解冻; 或 LMWM 长程 milestone 针对堆叠。
3. **接 V8 变体到 RoboTwin**: 残差/tsched(LIBERO 已验能移工作点)在 RoboTwin 训练 yaml 接上, 看能否把 tied/floored 任务撬动。
4. **训到位再对齐公开 SOTA**(pi0/RDT/X-VLA 积木族数字)。

---

# 附: 40k 续训重跑判决(2026-07-26, 中间读数)

两条续训(warm-start 20k → +20k = 40k)均已到位:
- baseline `20260725_070522+..._c20k/steps_20000`(满40k)
- dual2q `20260725_133058+..._c20k/steps_20000`(满40k, 四坑修复那次)

重跑判决 eval `t-20260726120516-rn4k4`(同口径 6任务×4seed×2ckpt, 输出 `rt_*_40k`)。
**最终(6任务, n=4 seed, 从 summary.json 聚合 + Welch-t)**:

| task | 40k A/base | 40k B/dual2q | Δ | t |
|---|---|---|---|---|
| **beat_block_hammer** | 0.0±0.0 | **21.5±4.3** | **+21.5** | **+8.6 \*\*\*** |
| handover_block | 85.0±3.6 | 89.5±3.0 | +4.5 | +1.7 ns |
| blocks_ranking_size | 80.5±3.6 | 82.5±7.1 | +2.0 | +0.4 ns |
| blocks_ranking_rgb | 98.0±1.4 | 97.0±1.0 | −1.0 | −1.0 ns |
| stack_blocks_two | 0.0 | 0.0 | 0 | —(双方地板) |
| stack_blocks_three | 0.0 | 0.0 | 0 | —(双方地板) |
| **聚合6任务** | **43.92±0.86** | **48.42±2.10** | **+4.50** | **+3.43** |

(注: stack 两任务双方恒 0、方差 0 → parser 报的 "+inf\*\*\*" 是退化统计, 实为"无差异", 勿引用。)

## ⭐ 预注册判决: "收敛伪影"假设 —— 证否(但聚合读法要诚实)

早先标的最大审稿风险 = "20k 的 +2.9 只是 LMWM 收敛更快, 基线训到位就追平"。**40k 证否**:
- **baseline hammer 在 40k 仍卡死 0.0±0.0**(20k=0.5, 40k=0.0 → **平台, 非欠训**); dual2q 同训练量 21.5(t=8.6\*\*\*)。
  → **LMWM 让基线"再训一倍也够不到"的能力成立 = 加能力, 非加速度。这是 RoboTwin 腿最硬的点。**
- 聚合 Δ 从 +2.9(t=3.08)→ **+4.50(t=3.43)**, 仍显著且略放大。

**⚠️ 聚合读法的诚实修正**: baseline 聚合 **46.4(20k)→43.92(40k) 反而降了 2.5**(ranking_size 88.5→80.5、handover 91.5→85 退化), dual2q 49.3→48.42 只降 0.9。**即 +4.50 里有一部分是 baseline 续训在饱和任务上退化, 不全是 dual2q 变好**。→ 干净可主张的 = **hammer 的能力差(baseline 任何预算都 0 / dual2q 21.5)**; 聚合增益方向正确但含 baseline 回退成分, 论文按此定调, 别把 +4.5 全算作 LMWM 净增益。
- stack 两任务 40k 仍 0/0 —— **真因 = 训练子集里 stack 数据 0 ep**(见附2 分布, 非"基座能力缺口/训练量补不了", 早先解读已更正); 模型从没见过堆叠 demo。绝对值仍低于 `sft_release` 单任务(hammer 21.5 vs 90)。
- **⭐hammer +21.5 是从仅 5 个训练 episode 里来的**(见附2): baseline 5demo→0 / dual2q 5demo→21.5。头条应表述为**数据效率**(5 条 demo 撬出能力)而非泛泛"能力差", 且诚实标注 n=5 训练样本、结果脆弱。

→ **RoboTwin 腿加固且诚实**: 头条 = "LMWM 解锁 baseline 够不到的 hammer 能力(+21.5, t=8.6, 跨 20k/40k 稳健)"; 聚合 +4.5 作辅证但注明含 baseline 回退。

---

# 附2: 与公开 SOTA 排行的可行性 + 成功判据核验(2026-07-26)

(agent 调研 official leaderboard + 论文, 结论: **只能内部 A/B, 不可直接对外排行**。)

## 官方口径(已确认, 来源 https://robotwin-platform.github.io/leaderboard + arXiv:2506.18088)
- 训练: **单任务** 50 条 demo_clean; 评测: 每任务 **100 rollout**; Easy=demo_clean / Hard=demo_randomized(五轴域随机)。
- Easy 逐任务榜(%): hammer RDT77/Pi0 43/DP3 72; handover 45/45/70; ranking_size 0/7/2; stack_two 21/42/24; ranking_rgb 3/19/3; stack_three 2/17/1。全50任务均值 DP3 55.2/Pi0 46.4/RDT 34.5。
- 榜单目前只收 5 法(RDT/Pi0/ACT/DP/DP3); X-VLA(报 72.9% SOTA级)、H-RDT(13任务均 68.7)不在表内、逐任务积木数**未证实**; CVPR 挑战赛冠军(90%+)因"无限数据"完全不可比。

## ⭐ 为何不可直接排行(口径三差)
| | 官方 | 我们 |
|---|---|---|
| 训练 | **单任务 50 demo** | **多任务混训 1315ep 积木子集**(≈219ep/任务, ~4×) |
| 评测 | 100 rollout, **Easy+Hard** | 200 rollout(50×4seed), **仅 Easy** |
| 收敛 | 训到位(hammer 90%级) | baseline 欠训(hammer=0) |

**铁证**: 我们 ranking_rgb 97-98 / ranking_size 80-82 比榜单最高(pi0 19/7)**高 5-12×**, 而 stack_two=0 反低于 pi0 42。"简单碾压/难任务垫底"的分裂 = 口径根本不同(多任务+数据量), **非方法更强**。并排进榜双向误导。

## 成功判据核验(排除"度量坏"红旗)
- eval `TASK_CONFIG=demo_clean`(demo_clean.yml 随机化全关)= 官方 Easy 同分布。
- success = RoboTwin env 自带 `check_success`(与官方同一套 env): ranking_rgb 要 3 块对齐(eps[0.13,0.03])+ 按 x 升序 + 双爪张开; stack_two 要 block2 在 block1 正上方 +0.05m。**与官方判据完全一致。**
- → **98% vs 19% 是训练侧差异(多任务+~4×数据), 不是度量口径不同**; stack 我们 0 而 pi0 42 = 1315ep 子集任务分布不均(stack 欠代表)嫌疑, 亦训练侧。**红旗解除: 度量没坏。**

## 要能对外排行需补(路径 b)
1. **单任务 50 demo_clean** 逐任务训练到收敛(非 1315ep 混训), 配方对齐 pi0/RDT。
2. 每任务 100 rollout, **同报 Easy + Hard**。
3. (可选)核 1315ep 子集逐任务分布, 解释 stack 欠代表。

## 对论文的定调
RoboTwin 腿**不得声称"榜单上超过 pi0/DP3/X-VLA"**; 可主张 = **控制变量干净的内部 A/B(同数据同 step, 唯一变量=LMWM), 头条 hammer 能力差**。RoboTwin 1.0 与 2.0 数字不可混用(2.0 加域随机+任务扩到50)。

信息源: leaderboard / arXiv:2506.18088(RoboTwin2.0) / 2507.23523(H-RDT) / 2506.23351(CVPR挑战赛)。

## 附2.1: 1315ep 子集逐任务分布(2026-07-26 核, 数据集 `robotwin2_lmwm_v30`)

按规范任务聚合 1315 ep(源 meta 是 117 个语言指令变体, 归类):

| 规范任务 | ep 数 | 占比 | 40k eval SR(A/B) |
|---|---|---|---|
| handover_block | ~550 | ~42% | 85.0 / 89.5 |
| blocks_ranking_size | 437 | 33.2% | 80.5 / 82.5 |
| blocks_ranking_rgb | 318 | 24.2% | 98.0 / 97.0 |
| **beat_block_hammer** | **5** | **0.4%** | 0.0 / **21.5** |
| **stack_blocks(two+three)** | **0** | **0%** | 0 / 0 |
| 其它(bell/click 混入) | 5 | 0.4% | — |

**⭐ 子集极度倾斜, 改写两处解读**:
1. **stack 0% = 训练集 0 ep 堆叠数据**(非基座缺口)。模型从没见过 stack demo, eval 0% 是必然。
2. **hammer +21.5 来自仅 5 个训练 ep** —— 是**数据效率**故事(dual2q 5demo→21.5 vs baseline 5demo→0), 惊人但**脆弱**(n=5), 论文必须标注样本量。
3. handover/ranking 占 ~99% 数据 → 我们在这些任务 80-98% 是**数据充足**结果(vs 官方单任务 50 demo), 印证跨榜不可比。

**对内部 A/B 的影响**: LMWM vs LaWM 的对比仍有效(同倾斜数据、同 step、唯一变量=方法), 但主张要落在**数据充足任务的稳健增益 + hammer 的 5-demo 数据效率**, 不碰 stack(无数据)。若要完整 6 任务对比 SOTA, 须先补 stack 训练数据 + 各任务均衡采样。

## 附2.2: 根因 —— 分布倾斜是 milestone pairs 管线的 curation bug(2026-07-26)

子集 = `build_robotwin_v30_subset.py` 取 `robotwin_milestone/pairs.npz` 的 `cur_ep`。**源数据 robotwin2.0(27500ep 全集)什么都有**(stack 1165 / hammer 550 / ranking 各 550 / handover 6730), 但 pairs **只覆盖 ep 号 672~4949 一段**, 双重丢弃:

| 任务 | 源 ep 号段 | 源可用 | pairs 覆盖 | 覆盖率 | 丢弃机制 |
|---|---|---|---|---|---|
| ranking_size | 1650~2199 | 550 | 437 | 79.5% | 在范围内 ✅ |
| ranking_rgb | 1100~1649 | 550 | 318 | 57.8% | 在范围内 ✅ |
| handover | 4~27498 | 6730 | 550 | 8.2% | 部分在范围 |
| **hammer** | **550~1099** | 550 | **5** | **0.9%** | ⚠️ 在范围内却几乎全丢 |
| **stack** | **11082~26399** | 1165 | **0** | **0%** | ⚠️ 整段在 pairs 范围外 |

**两 bug 叠加**:
1. **stack = 纯范围截断**: 所有 stack ep 号 11082~26399, 完全在 pairs 段(672~4949)外 → 管线根本没跑到该号段 → 0%。
2. **hammer = 任务性质丢弃**: 号段 550~1099 有 ~428 个在 pairs 范围内, 只覆盖 5(范围内 ~1%)→ **hammer="抡锤砸一下"单快动作, 无可切分中间子目标, CRAVE/milestone 提取产不出 pairs**。

**定性**: 名不副实的"积木族子集", 实为"一次早期 milestone 提取跑过的 ep(672~4949)", ranking+handover 主导, 含 5 个 bell/click 污染。

**对结论冲击**:
- stack 0/0 **无意义**(0 数据 + pairs bug), 不得作任何解读。
- hammer +21.5 建在 5 个"漏网"样本上, **更脆**; 且 LMWM 在"milestone 管线自己提不出子目标"的任务上反而 +21.5, **需复核**(噪声? 还是靠非-milestone 信号?)。
- **可信只剩 ranking/handover**(数据充足、双方近饱和、差异 ns)→ **RoboTwin 正信号实际非常薄**。

**修复路径**: 重跑 milestone pairs 覆盖**全 ep 号段**(尤其 stack 11082~26399)+ 各任务均衡采样; hammer 若确无子目标, 需 milestone 提取降级策略或排除出 milestone 主张。**在此之前 RoboTwin 腿不能作强主张。**收敛伪影这一刀防住了。
