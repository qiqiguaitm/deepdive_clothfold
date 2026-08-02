# LIBERO 4-suite 补测: LMWM 在完整基准上净负(2026-07-25)

> 动机: 审稿风险——LaWAM 原文 Table 1 是 **4-suite 聚合**, 我们此前只报 libero_10。
> 拿 libero_10 单套件数比 Table 1 = 口径错位; "为什么只报一组" = cherry-picking 红旗。
> 前提确认: 训练用 `data_mix: libero` = `libero_merged_no_noops_20hz`(4套件合并, mixtures.py:36),
> 故评全 4 套件合法。

## 结果(nowm baseline vs dual2q LMWM, n=4 seed, 3额外套件 osmesa)

| 套件 | nowm(LaWAM复现) | dual2q(LMWM) | Δ | 饱和? |
|---|---|---|---|---|
| libero_10 | 94.3† | 95.2† | +0.9 | 否(区分) |
| libero_goal | 98.3 | 98.3 | 0 | 是 |
| libero_object | 99.8 | 99.5 | −0.3 | 是 |
| **libero_spatial** | **96.3±0.7** | **93.3±0.4** | **−3.0 (t=−6.62 \*\*\*)** | **否(区分)** |
| **4-suite 聚合** | **97.20** | **96.58** | **−0.62** | |

† libero_10 用此前 egl 数(ckpt 可能非 211747/231838 同源, 待补 osmesa 精修); spatial/goal/object 是同 ckpt(nowm=211747, dual2q=231838)osmesa 干净数。

## ⭐ 核心发现: LMWM 在完整 LIBERO 上**净负**(−0.62), 不是 null

- **spatial 回归铁证 + 防偶然加测(2026-07-25)**:
  | 配置 | n | spatial SR | Δ vs nowm | t |
  |---|---|---|---|---|
  | nowm(LaWAM) | 8 | 96.5±0.8 | — | — |
  | dual2q#1(231838) | 8 | 93.2±0.3 | −3.3 | −9.76 |
  | dual2q#2(150545, **独立训练**) | 4 | 90.8±1.0 | −5.8 | −8.91 |
  两维度堵死: ①加 seed n=4→8, t −6.62→−9.76(非 seed 偶然)②第二个独立 dual2q ckpt 也显著伤 spatial 且更狠(非 ckpt 特异)。**spatial 回归 = LMWM 鲁棒性质**, 跨 seed 跨独立训练一律显著负(t<−8)。spatial **未饱和**(96/93/91)→ 真实有害。
- **只报 libero_10 会掩盖基准级负结果**: libero_10 的 +0.9 被 spatial −3.0 抵消, 4-suite 净 −0.62。审稿人补 spatial 一看即穿。**坚持补测救了这个坑。**

## ⭐ 重塑的论文命题(跨 LIBERO + RoboTwin 一致)

LMWM(milestone/子目标世界模型)增益**高度集中在时序/子目标结构强的任务**:
- libero_10(长程)+0.9 · RoboTwin beat_block_hammer(击打时机)+19.5

**在缺子目标结构的任务上中性到有害**:
- libero_spatial(空间推理)−3.0 · RoboTwin 多数任务 ns

→ 诚实命题 = **"milestone WM 选择性帮助子目标结构任务, 非普遍提升 manipulation"**。两 benchmark 数据一致支持。这比"WM 全面更好"弱, 但站得住且防弹。

## 待办
1. **补 libero_10 for 211747+231838**(osmesa n=4)→ 一套 osmesa 协议出完全自洽 4-suite, 消除 † 口径混用。
2. 论文口径表: 报 4-suite 聚合(= Table 1 口径)+ 逐套件 breakdown + spatial 回归诚实呈现。
3. 机制: 为何 LMWM 伤 spatial? milestone 在纯空间任务是干扰信号(无时序子目标可利用)。
