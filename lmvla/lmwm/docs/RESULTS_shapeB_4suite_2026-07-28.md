# shapeB(梯度隔离)补齐 goal/object → 三臂 4-suite 对照(2026-07-28)

> shapeB = dual2q + **两通道 target 全 detach**(切断 WM 梯度对共享编码器的塑形)。
> 此前只有 libero_10 / spatial 两套件; 本次补 goal/object, 与残差臂(P1-F)同口径合表。

## 本次产出(North-H20 单卡×4seed, osmesa; jobs q462n/qprcm/xfjq5/wqmr5)

ckpt: `20260727_100245+dual2q_shapeB_neither/checkpoints/steps_12500_pytorch_model.pt`

| 套件 | shapeB SR | 各 seed |
|---|---|---|
| libero_goal | **98.00 ± 0.85** (n=4) | 99.0 / 97.2 / 97.4 / 98.4 |
| libero_object | **99.80 ± 0.23** (n=4) | 100.0 / 99.6 / 99.6 / 100.0 |

既有(同 ckpt, 同 osmesa 协议, n=4): libero_10 **95.30 ± 0.77**、libero_spatial **95.85 ± 1.18**。

## 三臂 + baseline 4-suite 对照(定稿:nowm libero_10 已 osmesa 补齐)

| 套件 | nowm(osmesa) | dual2q绝对 | 残差 | **shapeB(梯度隔离)** |
|---|---|---|---|---|
| libero_10 | **94.55±0.50** | 95.2† | 95.75† | 95.30±0.77 |
| libero_goal | **98.3** | 98.3 | 97.80±0.28 | 98.00±0.85 |
| libero_object | **99.8** | 99.5 | 99.65±0.19 | 99.80±0.23 |
| libero_spatial | **96.3** | 93.3 | 94.4 (n=8) | 95.85±1.18 |
| **聚合** | **97.24** | 96.58 | 96.90 | **97.24** |
| Δ vs nowm | — | −0.62 | −0.30 | **0.00** |

> nowm libero_10 补测完成(2026-07-28, group `nowm_l10_osmesa`, n=4, osmesa, 94.55±0.50)。
> 此前 97.20 是混协议数(egl 94.3), 已废弃。绝对/残差的 libero_10 仍为 egl(†标记),
> 补齐 osmesa 后精密修正 ±0.25 以内, 不影响结论, 优先级低。

## 判读

1. **梯度隔离是唯一把 LMWM 净负清零的臂**: −0.62(绝对)→ −0.30(残差)→ **0.00(shapeB)**。
   spatial 赤字从 −3.0 → −1.9 → **−0.45**, 单调收敛。**机制自洽**:
   伤害来自 WM 梯度塑形共享编码器, 只有 detach 直接切断该通路。
2. **"打平"不是"超越"**: 聚合差 0.00 远小于 seed 噪声(std 0.8~1.2), 只能说
   **"梯度隔离消除了 LMWM 在 LIBERO 上的基准级伤害"**, 不能说 LMWM 带来净增益。
   LIBERO 上 LMWM 的正收益仍只在 libero_10(+0.75), 被 goal/spatial 的同量负抵消。
3. 绝对/残差的 libero_10 是 egl(†), 与其余三套件的 osmesa 混协议。补齐 osmesa 后
   ±0.25 精修, 不影响"打平"结论, 低优先级。
4. **对论文的意义**: 主张应写成 **"诊断 → 两根因 → 梯度隔离修复到与无 WM 基线打平"**,
   LIBERO 定位为**饱和/零和的诊断场**; **净增益的正信号靠 RoboTwin**
   (LaWAM 侧 +2.9pt 已有, pi05 侧 #39 待判)。

## 复现
```
shapeB goal/object: NE/lmvla/lawam/results/eval_runs/libero/shapeB_go/Bgo_s*/suites/*/summary.json
shapeB l10/spatial: gf0/lmvla/lawam/results/eval_runs/libero/shapeB_neither/*/suites/*/summary.json
键: total_episodes / total_successes(无 aggregate_SR)
```
