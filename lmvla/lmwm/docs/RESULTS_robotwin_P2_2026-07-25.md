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
**中间读数(5/6 任务, n=4, stack_blocks_three 未出, 非终值)**:

| task | 20k Δ(B−A) | **40k Δ** | 读法 |
|---|---|---|---|
| beat_block_hammer | 0.5→20.0 (+19.5) | **0.0→21.5 (+21.5)** | ⭐baseline 40k **仍 0**, dual2q 21.5 |
| handover_block | 91.5→90.0 (−1.5) | 85.0→89.5 (**+4.5**) | 40k 转正 |
| blocks_ranking_size | 88.5→88.0 (0) | 80.5→82.5 (+2.0) | baseline 该任务 40k 掉(88.5→80.5, 留意) |
| blocks_ranking_rgb | 98→98 | 98→97 (−1.0) | 近饱和 |
| stack_blocks_two | 0→0 | **0→0** | 仍地板(基座缺口, 训练量补不了) |
| stack_blocks_three | 0→0 | (未出) | |
| **聚合** | +2.9 | **+5.4(5任务)** | 增益**放大** |

## ⭐ 预注册判决: "收敛伪影"假设 —— 证否

早先标的最大审稿风险 = "20k 的 +2.9 只是 LMWM 收敛更快, 基线训到位就追平"。**40k 数据证否**:
- **baseline hammer 在 40k 仍卡死 0.0**(不是欠训, 是**平台**); dual2q 同训练量拿 21.5。
  → **LMWM 让基线"再训一倍也够不到"的能力成立 = 加能力, 非加速度。**
- 聚合增益 **+2.9→+5.4 放大**, handover 从 −1.5 转 +4.5。→ 增益随训练**增强**。

**诚实 caveat**: ①preliminary, stack_three 未出(聚合非终值, 待最终 Welch-t); ②baseline
blocks_ranking_size 掉 88.5→80.5(续训该任务退化/方差, 留意); ③绝对值仍低于 `sft_release`
单任务(hammer 21.5 vs 90)——但"同训练量 baseline=0 / dual2q=21.5"的 A/B 点干净。

→ **RoboTwin 腿加固**: P2 是真净增益, 且防住收敛伪影这一刀。最终表待 `rteval2x4_40k` 收尾(监控 `p2_40k_mon.sh` 自动抓)。
